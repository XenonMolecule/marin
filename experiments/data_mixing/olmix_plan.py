# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The coordinator/child contract for an OLMIX swarm: the manifest and the run identity.

The coordinator samples all K mixtures once, writes them to a **swarm manifest** in GCS,
and submits one child per row. Children read their own row by index.

Why a manifest rather than CLI args or re-sampling in the child:

* 118 float weights per run will not fit sanely in an argv list.
* Re-deriving the weights in each child would work (sampling is deterministic given
  ``(seed, K, domain order)``) but costs minutes of CPU per child -- RNG-parity sampling
  at m=118 is ~6.5M numpy calls plus an O(n^2) dedup over 3,630 candidates -- so 363
  children would repeat hours of identical work.
* The manifest is a durable, auditable record: exactly which mixtures were trained, in
  which domain order, under which sampler settings. It is the `ratios.csv` of the
  experiment and the fit reads it back.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import fsspec
import numpy as np

logger = logging.getLogger(__name__)

# Proxy architecture: the d512 cell of the fixed-model natural sweep (`expFM_natural`),
# NOT olmix's bespoke d256/L4/H8. d256 does not exist in
# `fixed_model_plan.TARGET_HIDDEN_SIZES` or `launch_10k_natural.WIDTHS`, so it has no
# operational history here; d512 is the most-exercised small config in the repo.
PROXY_HIDDEN_DIM = 512
PROXY_BUDGET_FLOPS = 3e18

# `int(w * block_size)` truncates, so a weight below 1/block_size trains on nothing, and
# the leftover from every domain's truncation is dumped on the largest domain. At the
# 2048 default that overweights the biggest cell by ~26% over a 118-domain mixture.
MIXTURE_BLOCK_SIZE = 32768

# Measured snap-and-renormalize wobble: a surviving weight lands at 0.909-1.00 x the clip.
# So `clip * CLIP_SAFETY >= 1/block_size` is the condition for every non-zero weight to
# yield at least one sequence per block.
CLIP_SAFETY = 0.909

SWARM_MANIFEST_REL = "metadata/olmix/{corpus}/swarm_s{seed}_K{k}.json"


@dataclass(frozen=True)
class SwarmManifest:
    """All K sampled mixtures for one corpus, plus the settings that produced them."""

    corpus: str
    region: str
    seed: int
    domains: tuple[str, ...]
    """Canonical domain order. Every weight row, the prior, the caps and the regression
    design matrix are indexed positionally off this -- never re-derive it."""
    weights: tuple[tuple[float, ...], ...]
    tokens: dict[str, int]
    cache_dirs: dict[str, str]
    tokenizer: str = ""
    """The store artifact's tokenizer. Recorded so the child does not re-read the artifact,
    and so a manifest carries proof of which tokenizer its cells were built with."""
    sampler: dict[str, Any] = field(default_factory=dict)
    """The sampler settings, recorded so a manifest is reproducible and auditable."""

    def __post_init__(self) -> None:
        m = len(self.domains)
        if not self.weights:
            raise ValueError("manifest has no mixtures")
        for i, row in enumerate(self.weights):
            if len(row) != m:
                raise ValueError(f"row {i} has {len(row)} weights, expected {m}")
            total = sum(row)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"row {i} sums to {total!r}, expected 1.0")
        if set(self.tokens) != set(self.domains):
            raise ValueError("tokens keys do not match domains")
        if set(self.cache_dirs) != set(self.domains):
            raise ValueError("cache_dirs keys do not match domains")

    @property
    def k(self) -> int:
        return len(self.weights)

    def row(self, index: int) -> dict[str, float]:
        """The mixture for run ``index``, as ``{domain: weight}`` with zeros dropped.

        Zero-weight domains are omitted so the training child never opens a cache it
        will not read -- with 118 domains and ~10 non-zero per mix that is ~108 caches
        left unopened per run.
        """
        if not 0 <= index < self.k:
            raise IndexError(f"index {index} outside [0, {self.k})")
        return {d: w for d, w in zip(self.domains, self.weights[index]) if w > 0.0}

    def assert_trainable(self, block_size: int = MIXTURE_BLOCK_SIZE) -> None:
        """Every non-zero weight must yield >= 1 sequence per block.

        Levanter only ``warnings.warn``s when a non-zero-weight source truncates to zero
        samples, so an unsatisfiable combination would otherwise surface as a warning
        buried in K training logs rather than as a launch-time failure.
        """
        floor = 1.0 / block_size
        for i, row in enumerate(self.weights):
            nz = [w for w in row if w > 0.0]
            if nz and min(nz) < floor:
                raise ValueError(
                    f"mixture {i} has a non-zero weight {min(nz):.3e} below the "
                    f"block-size floor {floor:.3e} (block_size={block_size}); it would "
                    f"train on zero sequences. Raise block_size or the sampler's clip."
                )

    def to_json(self) -> str:
        return json.dumps(
            {
                "corpus": self.corpus,
                "region": self.region,
                "seed": self.seed,
                "domains": list(self.domains),
                "weights": [list(r) for r in self.weights],
                "tokens": self.tokens,
                "cache_dirs": self.cache_dirs,
                "tokenizer": self.tokenizer,
                "sampler": self.sampler,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, payload: str) -> SwarmManifest:
        d = json.loads(payload)
        return cls(
            corpus=d["corpus"],
            region=d["region"],
            seed=d["seed"],
            domains=tuple(d["domains"]),
            weights=tuple(tuple(float(x) for x in row) for row in d["weights"]),
            tokens={k: int(v) for k, v in d["tokens"].items()},
            cache_dirs=dict(d["cache_dirs"]),
            tokenizer=d.get("tokenizer", ""),
            sampler=d.get("sampler", {}),
        )


def manifest_path(bucket: str, corpus: str, seed: int, k: int) -> str:
    return f"{bucket.rstrip('/')}/{SWARM_MANIFEST_REL.format(corpus=corpus, seed=seed, k=k)}"


def write_manifest(manifest: SwarmManifest, path: str) -> None:
    with fsspec.open(path, "w") as fh:
        fh.write(manifest.to_json())
    logger.info("wrote swarm manifest: %s (K=%d, m=%d)", path, manifest.k, len(manifest.domains))


def read_manifest(path: str) -> SwarmManifest:
    with fsspec.open(path) as fh:
        return SwarmManifest.from_json(fh.read())


def weight_hash(weights: dict[str, float] | tuple[float, ...]) -> str:
    """Short stable digest of a mixture, for run identity.

    Run names in the existing sweeps encode (method, budget, arch, batch) but NOT the
    data mixture, so two swarm runs differing only in weights would collide on the
    checkpoint path, the DONE marker, the region-tracker key and the results JSON. The
    hash makes the mixture part of the identity. Rounded to 9 decimals first so a
    float-repr difference cannot produce two names for the same mixture.
    """
    if isinstance(weights, dict):
        items = [(k, round(float(v), 9)) for k, v in sorted(weights.items())]
    else:
        items = [(str(i), round(float(v), 9)) for i, v in enumerate(weights)]
    blob = json.dumps(items, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def run_name(corpus: str, seed: int, k: int, index: int, weights: dict[str, float]) -> str:
    """Stable, collision-resistant run identity. Kept under Iris's 200-char job-name cap."""
    return f"olmix-{corpus}-s{seed}-K{k}-i{index:04d}-w{weight_hash(weights)}"


def effective_domains(tokens: dict[str, int], weights: np.ndarray) -> tuple[list[str], list[str]]:
    """Split domains into (sampled, never-sampled) across a whole swarm.

    A never-sampled domain has an all-zero design column, so the log-linear loss is
    exactly flat in its coefficient and LBFGS leaves it at its restart initialisation --
    measured: a dead column keeps its raw ``U(0,1)*0.1`` init while live domains recover
    their true coefficients to 3 decimals. The solver then acts on that garbage, driving
    the domain to ~0 (measured 0.00028 against a natural share of 0.167). So these must
    be dropped from the fit and the solve and re-inserted at their natural weight.
    """
    names = list(tokens)
    if weights.shape[1] != len(names):
        raise ValueError(f"weights has {weights.shape[1]} columns, tokens has {len(names)} domains")
    appeared = (weights != 0).sum(axis=0)
    sampled = [n for n, a in zip(names, appeared) if a > 0]
    dead = [n for n, a in zip(names, appeared) if a == 0]
    return sampled, dead
