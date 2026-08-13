# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The domain set: one OLMIX domain per (topic, quality) grid cell.

Reads a corpus's materialized store artifact and produces everything the rest of the
pipeline indexes positionally off: the ordered domain list, each domain's Levanter
`cache_dir`, its real token count, and the natural prior.

**Domain order is load-bearing.** olmix keys the prior, the availability caps and the
regression design matrix positionally off one ordering, so a mismatch anywhere silently
pairs a domain's weight with another domain's coefficient. :func:`load_grid_domains`
returns a single canonical ordering (lexicographic on `c{cluster:02d}_q{quality}`) and
every consumer must take it from there rather than re-deriving it.

Token counts come from the store artifact's per-cell `total_tokens`, i.e. real
marin-tokenizer counts. Do NOT substitute the gte-base counts in
`metadata/grid_v1/*/distribution.json` — those come from the topic classifier's
tokenizer and differ by several percent (dclm 6.88B gte vs 7.33B real, high_quality
22.45B gte vs 21.30B real), in opposite directions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from experiments.datakit.store.datakit_store import ClusteredStoreData
from experiments.datakit.store.store_compat import read_artifact
from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logger = logging.getLogger(__name__)

# Grid geometry. Both are asserted against the store artifact rather than trusted.
NUM_TOPICS = 24
NUM_QUALITY_BUCKETS = 5

# `<bucket>/datakit/store/<corpus>_gridv1/`
STORE_REL_PATH = "datakit/store/{corpus}_gridv1"

# Levanter appends `/<split>` itself, so a cell's `cache_dir` is its `path` minus that
# level. A cell whose path lacks the split level cannot be loaded by a mixture at all.
EXPECTED_SPLIT = "train"


@dataclass(frozen=True)
class GridDomain:
    """One (topic, quality) cell as an OLMIX mixing domain."""

    name: str
    """Canonical id, `c{cluster:02d}_q{quality}`. Defines the positional ordering."""
    cluster_id: int
    quality_bucket: int
    cache_dir: str
    """What a Levanter ``DatasetComponent.cache_dir`` takes (parent of the split level)."""
    tokens: int
    docs: int


def domain_name(cluster_id: int, quality_bucket: int) -> str:
    return f"c{cluster_id:02d}_q{quality_bucket}"


def store_path(corpus: str, region: str) -> str:
    """Region-local store root for a corpus. Region-agnostic by construction: the
    caller supplies the region and `REGION_TO_BUCKET` resolves the bucket (some regions
    have non-obvious names -- `europe-west4` is `gs://marin-eu-west4`)."""
    if region not in REGION_TO_BUCKET:
        raise ValueError(f"unknown region {region!r}; known: {sorted(REGION_TO_BUCKET)}")
    return f"{REGION_TO_BUCKET[region]}/{STORE_REL_PATH.format(corpus=corpus)}"


def _rebase(path: str, artifact_root: str, local_root: str) -> str:
    """Re-root a cell path recorded against ``artifact_root`` onto ``local_root``.

    Only the suffix below the store root is reused, so the region prefix cannot leak
    through from a cross-region copy. Fails loudly if the path is not actually under the
    root the artifact claims -- that would mean the artifact is describing some other
    tree and rebasing would fabricate a plausible-looking wrong path.
    """
    artifact_root = artifact_root.rstrip("/")
    path = path.rstrip("/")
    if path == artifact_root:
        return local_root.rstrip("/")
    prefix = artifact_root + "/"
    if not path.startswith(prefix):
        raise ValueError(
            f"cell path {path!r} is not under the artifact's own cache_path {artifact_root!r}; "
            f"cannot rebase it onto {local_root!r}"
        )
    return f"{local_root.rstrip('/')}/{path[len(prefix):]}"


def load_grid_domains(corpus: str, region: str) -> tuple[GridDomain, ...]:
    """Load a corpus's grid cells as domains, in canonical order.

    Validates, rather than assumes, everything the mixture depends on: the split level
    is present on every cell path, every cache is inside the region-local bucket, cell
    coordinates are inside the 24x5 grid, and no coordinate appears twice.
    """
    root = store_path(corpus, region)
    artifact = read_artifact(root, ClusteredStoreData)

    if artifact.split != EXPECTED_SPLIT:
        raise ValueError(f"{root}: artifact split is {artifact.split!r}, expected {EXPECTED_SPLIT!r}")
    if artifact.cluster_view != NUM_TOPICS:
        raise ValueError(f"{root}: cluster_view is {artifact.cluster_view}, expected {NUM_TOPICS}")
    if len(artifact.bucket_edges) + 1 != NUM_QUALITY_BUCKETS:
        raise ValueError(
            f"{root}: {len(artifact.bucket_edges)} bucket edges implies "
            f"{len(artifact.bucket_edges) + 1} quality buckets, expected {NUM_QUALITY_BUCKETS}"
        )

    expected_prefix = REGION_TO_BUCKET[region] + "/"
    domains: list[GridDomain] = []
    for cell in artifact.buckets:
        if not 0 <= cell.cluster_id < NUM_TOPICS:
            raise ValueError(f"{root}: cluster_id {cell.cluster_id} outside [0, {NUM_TOPICS})")
        if not 0 <= cell.quality_bucket < NUM_QUALITY_BUCKETS:
            raise ValueError(f"{root}: quality_bucket {cell.quality_bucket} outside [0, {NUM_QUALITY_BUCKETS})")
        # The store writes `.../sub=<S>/<split>`; `component_cache_dir` strips the split.
        # Verify rather than trust: a cell written without the split level would produce
        # a cache_dir that silently points one level too high.
        if os.path.basename(cell.path.rstrip("/")) != EXPECTED_SPLIT:
            raise ValueError(
                f"{root}: cell path {cell.path!r} does not end in {EXPECTED_SPLIT!r}. "
                f"Levanter resolves a component as <cache_dir>/<split>, so this cell "
                f"cannot be loaded by a mixture. The store must write the split level."
            )
        # REBASE onto the requested region rather than trusting the artifact's absolute
        # paths. A store copied between regions keeps the ORIGINAL paths in its
        # `.artifact.json` (observed: the us-east5 dclm artifact's cells all point at
        # gs://marin-us-central1/...), so a consumer that trusted them would read every
        # token cross-region while believing it was local. The artifact defines the grid
        # structure and token counts; the region argument defines where we read.
        cache_dir = _rebase(cell.component_cache_dir, artifact.cache_path, root)
        if not cache_dir.startswith(expected_prefix):
            raise ValueError(
                f"{root}: rebased cache_dir {cache_dir!r} is not under {expected_prefix!r}. "
                f"Cross-region reads are forbidden -- use the store in the training region."
            )
        if cell.total_tokens <= 0:
            raise ValueError(f"{root}: cell {cell.cluster_id}/{cell.quality_bucket} reports {cell.total_tokens} tokens")
        domains.append(
            GridDomain(
                name=domain_name(cell.cluster_id, cell.quality_bucket),
                cluster_id=cell.cluster_id,
                quality_bucket=cell.quality_bucket,
                cache_dir=cache_dir,
                tokens=cell.total_tokens,
                docs=cell.total_elements,
            )
        )

    names = [d.name for d in domains]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"{root}: duplicate cell coordinates {dupes}")

    domains.sort(key=lambda d: d.name)
    logger.info(
        "%s @ %s: %d cells, %d tokens, %d docs (tokenizer=%s)",
        corpus,
        region,
        len(domains),
        sum(d.tokens for d in domains),
        sum(d.docs for d in domains),
        artifact.tokenizer,
    )
    return tuple(domains)


def domain_tokens(domains: tuple[GridDomain, ...]) -> dict[str, int]:
    """`{name: tokens}` in canonical order (dicts preserve insertion order)."""
    return {d.name: d.tokens for d in domains}


def domain_cache_dirs(domains: tuple[GridDomain, ...]) -> dict[str, str]:
    return {d.name: d.cache_dir for d in domains}


def quality_bucket_of(domains: tuple[GridDomain, ...]) -> dict[str, int]:
    """`{name: quality_bucket}` -- for per-bucket reporting, never for weighting."""
    return {d.name: d.quality_bucket for d in domains}


def grid_tokenizer(corpus: str, region: str) -> str:
    """The tokenizer the corpus's cells were built with, from the store artifact.

    Recorded into the swarm manifest so the training child does not re-read the artifact,
    and so a manifest carries proof of which tokenizer its cells assume -- a mixture whose
    components disagree on tokenizer is silently meaningless.
    """
    artifact = read_artifact(store_path(corpus, region), ClusteredStoreData)
    if not artifact.tokenizer:
        raise ValueError(f"{corpus} @ {region}: store artifact records no tokenizer")
    return artifact.tokenizer
