# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared source-of-truth constants for the spec-explorer dashboard.

Every service (eval aggregation, coverage, search, spec display) reads its
method identities, colors, model-size labels, run-name parsing, and the
dataset -> region map for the 300-WARC index artifacts from here, so the
frontend renders one consistent vocabulary of methods across all tabs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Regions where eval results and index artifacts can land. Union of the region
# lists scanned by the existing random-ladder plotters (core_v2 scans 6, olmo
# scans 4); results are pinned/floated across regions so we scan all of them.
EVAL_REGIONS: tuple[str, ...] = (
    "us-east5",
    "us-central1",
    "us-east1",
    "eu-west4",
    "us-central2",
    "us-west4",
)

# Locked color/label convention (mirrors plot_core_v2_random_ladder.METHOD_STYLE,
# 2026-07-22): dclm=blue, high_quality=green, nemotron=orange, resiliparse=purple,
# llm_pipeline_v1=red, llm_simple_v1=cyan, fastpipe bands=brown ramp.
METHOD_STYLE: dict[str, tuple[str, str]] = {
    "llm_pipeline_v1": ("#d62728", "llm_pipeline_v1"),
    "llm_pipeline_v1_1": ("#fb6a4a", "llm_pipeline_v1_1"),  # llm_pipeline family — lighter red
    "llm_simple_v1": ("#17becf", "llm_simple_v1"),
    "dclm": ("#1f77b4", "dclm"),
    "high_quality": ("#2ca02c", "high_quality"),
    "nemotron_full": ("#ff7f0e", "nemotron"),
    "nemotron": ("#ffbb78", "nemotron (10k)"),
    "nemotron_qhigh": ("#fdae6b", "nemotron_qhigh"),
    "nemotron_decon": ("#bd7e2a", "nemotron_decon"),
    "dclm_decon": ("#6baed6", "dclm_decon"),
    # resiliparse family — purple shades
    "resiliparse": ("#9467bd", "resiliparse"),
    "resiliparse_dedup": ("#7b4fa0", "resiliparse_dedup"),
    "resiliparse_random_dedup": ("#b39ddb", "resiliparse_random_dedup"),
    # high_quality family — greens (v2 lighter than v1)
    "high_quality_v2": ("#98df8a", "high_quality_v2"),
    # quality-tier ablations — red-brown ramp (distinct from llm_pipeline red)
    "low_quality": ("#843c39", "low_quality"),
    "med_low_quality": ("#ad494a", "med_low_quality"),
    "med_quality": ("#e7969c", "med_quality"),
    # llm_curated variants — pale, low-emphasis (rarely the focus)
    "llm_curated": ("#fdd0a2", "llm_curated"),
    "llm_curated_dedup": ("#c7e9c0", "llm_curated_dedup"),
    "llm_curated_dclm_filtered": ("#c6dbef", "llm_curated_dclm_filtered"),
    "fineweb_cc": ("#e377c2", "fineweb_cc"),
    "fineweb_edu": ("#bcbd22", "fineweb_edu"),
    "fastpipe_v3_100": ("#5c3d2e", "fastpipe 100%"),
    "fastpipe_v3_80": ("#7d5540", "fastpipe 80%"),
    "fastpipe_v3_60": ("#a06e52", "fastpipe 60%"),
    "fastpipe_v3_40": ("#c08a6a", "fastpipe 40%"),
    "fastpipe_v3_20": ("#dbb28f", "fastpipe 20%"),
    # fastpipe over llm_pipeline_v1_1 — gold, distinct from the v3 brown ramp
    "lpv11_fastpipe_v1": ("#bf9b30", "lpv11 fastpipe v1"),
}


def method_color(method: str) -> str:
    return METHOD_STYLE.get(method, ("#888888", method))[0]


def method_label(method: str) -> str:
    return METHOD_STYLE.get(method, ("#888888", method))[1]


# Hidden-dim -> (label, numeric params). Params feed the C ~= 6*N*D epoch-FLOPs
# conversion; labels are display only. Mirrors the plotters' PARAMS/_PARAMS_NUM.
# dim -> (label, params). Known configs (d256..d3584) match the plotters' PARAMS;
# d1280 (L13) and d2048 (L21) are new intermediate sizes from the latest evals —
# their params are interpolated on the L*d^2 scaling the known sizes follow
# (locally linear to ±2%), so they carry a "~" like the other estimates.
MODEL_PARAMS: dict[int, tuple[str, float]] = {
    256: ("69M", 69e6),
    512: ("157M", 157e6),
    768: ("273M", 273e6),
    1024: ("~500M", 500e6),
    1280: ("~700M", 700e6),
    1536: ("998M", 998e6),
    2048: ("~1.9B", 1.9e9),
    2432: ("2.9B", 2.9e9),
    3328: ("~6B", 6e9),
    3584: ("8.1B", 8.1e9),
}


def model_label(dim: int) -> str:
    return MODEL_PARAMS.get(dim, (f"d{dim}", 0.0))[0]


def method_tokens(method: str, n_warcs: int) -> int | None:
    """Observed extracted-token count for a method's N-WARC data, from the curation registry.

    This is what a fixed WARC count actually yields per method (each extractor keeps a
    different amount), so it only depends on ``(method, n_warcs)`` — not model size.
    """
    from experiments.scaling_law_sweeps.curation_plan import METHODS

    m = METHODS.get(f"{method}_random_{n_warcs}")
    return int(m.d_obs_tokens) if m is not None and m.d_obs_tokens else None


def epoch_flops(method: str, n_warcs: int, dim: int) -> tuple[float, float] | None:
    """(1-epoch, 2-epoch) training FLOPs for a method's 300/N-WARC data at ``dim``.

    Uses ``C ~= 6 * params * tokens`` with the method's observed token count from
    the curation ``METHODS`` registry. Returns None when the method/size isn't
    registered (so callers just omit the epoch marks). Mirrors
    ``plot_olmo_bpb_random_ladder._epoch_flops``.
    """
    from experiments.scaling_law_sweeps.curation_plan import METHODS

    m = METHODS.get(f"{method}_random_{n_warcs}")
    params = MODEL_PARAMS.get(dim)
    if m is None or params is None or not params[1]:
        return None
    x1 = 6.0 * params[1] * m.d_obs_tokens
    return x1, 2.0 * x1


# Region that holds each dataset's 300-WARC (SMALL) index/text artifacts
# (bm25_indices/small + url_index/small). Measured from the built keys.parquet
# set on 2026-07-23. Coverage/search route reads to the owning region.
# Consolidated 2026-07-24: dclm + nemotron_full copied from us-central2 into
# us-central1, so one worker there serves all non-fastpipe datasets. fastpipe
# bands stay in us-east5 (deliberately not moved).
INDEX_DATASET_REGION: dict[str, str] = {
    "dclm": "us-central1",
    "nemotron_full": "us-central1",
    "high_quality": "us-central1",
    "high_quality_v2": "us-central1",  # 300-WARC keys+text built 2026-07-24
    "med_quality": "us-central1",  # 300-WARC keys+text built 2026-07-24
    "llm_pipeline_v1": "us-central1",
    "llm_pipeline_v1_1": "us-central1",  # 300-WARC keys+text built 2026-07-28
    "llm_simple_v1": "us-central1",
    # Coverage/cross-ref only: 300-WARC keys live in us-central2 and there is no
    # BM25 index, so these appear in the Jaccard matrix + search kept/dropped
    # (local keys) but are not retrieval sources or text-viewable.
    "fineweb_edu": "us-central2",
    # resiliparse: the "extract every page" baseline -> effectively the universe
    # at 300 WARCs (~12.2M url_h). Keys-only coverage tier (no BM25/text), keys in
    # us-central2 at url_index/small/resiliparse/keys.parquet.
    "resiliparse": "us-central2",
    # fineweb_cc: 10k survivors extracted 2026-07-28 (54.6 GB, 41.9M docs), then
    # subset to the canonical random-300 WARCs for keys-only coverage.
    "fineweb_cc": "us-central2",
    "fastpipe_v3": "us-east5",
    "fastpipe_v3_20": "us-east5",
    "fastpipe_v3_40": "us-east5",
    "fastpipe_v3_60": "us-east5",
    "fastpipe_v3_80": "us-east5",
    # lpv11_fastpipe_v1: fastpipe (fasttext+modernbert) survivors over the
    # llm_pipeline_v1_1 300-WARC extraction. Keys-only coverage tier (rid_h null;
    # fast_curation carries no warc_record_id) at url_index/small/lpv11_fastpipe_v1.
    "lpv11_fastpipe_v1": "us-east5",
}


def index_datasets() -> list[str]:
    """Datasets with a 300-WARC coverage/search index, in a stable order."""
    return list(INDEX_DATASET_REGION)


def regions_for(datasets: list[str]) -> dict[str, list[str]]:
    """Group requested datasets by their owning region."""
    out: dict[str, list[str]] = {}
    for ds in datasets:
        region = INDEX_DATASET_REGION.get(ds)
        if region is None:
            raise ValueError(f"No 300-WARC index region known for dataset {ds!r}; known: {index_datasets()}")
        out.setdefault(region, []).append(ds)
    return out


# run_name_core = curation-<method_full>-<tag>-<budget>-d<H>-L<L>-B<B>
# method_full is non-greedy so it stops at the first "-exp..." tag boundary.
RUN_NAME_RE = re.compile(
    r"^curation-(?P<method_full>.+?)-(?P<tag>exp[^-]+)"
    r"-(?P<budget>[0-9eE+.\-]+)-d(?P<dim>\d+)-L(?P<layers>\d+)-B(?P<batch>\d+)$"
)

# Method_full values whose N is an implicit 3000 (fixed-model sweep), i.e. they
# do NOT carry a trailing _<int> WARC count.
_FIXED_MODEL_METHODS: frozenset[str] = frozenset({"llm_curated_dedup", "llm_curated_dclm_filtered", "high_quality_3000"})


@dataclass(frozen=True)
class RunIdentity:
    """Parsed identity of a training run from its run_name_core."""

    run_name_core: str
    method: str  # base method (WARC count stripped)
    n_warcs: int
    budget: float
    dim: int
    layers: int
    batch: int
    tag: str


# Variant suffixes that trail the scale token in a method_full (e.g.
# ``dclm_10k_decon``). They stay on the method base so a deconned/deduped run is
# kept distinct from its plain counterpart in the comparison.
_METHOD_VARIANT_SUFFIXES: frozenset[str] = frozenset({"decon", "dedup"})


def _parse_scale(token: str) -> int | None:
    """A scale token -> WARC/document count: ``300`` -> 300, ``10k`` -> 10000."""
    if token.isdigit():
        return int(token)
    m = re.fullmatch(r"(\d+)([km])", token)
    if m:
        return int(m.group(1)) * (1000 if m.group(2) == "k" else 1_000_000)
    return None


def _decode_method_n(method_full: str) -> tuple[str, int] | None:
    """(method_base, N) from a method_full string, or None if undecodable.

    WARC sweep: ``high_quality_random_300`` -> ("high_quality_random", 300) (the
    ``_random`` is stripped later). Abbreviated scale: ``dclm_10k`` ->
    ("dclm", 10000). Trailing variant: ``dclm_10k_decon`` -> ("dclm_decon",
    10000), kept distinct from plain dclm. Fixed-model methods with no count
    decode to N=3000.
    """
    if not method_full:
        return None
    if "_" in method_full:
        rest, last = method_full.rsplit("_", 1)
        n = _parse_scale(last)
        if n is not None:
            return rest, n
        if last in _METHOD_VARIANT_SUFFIXES:
            inner = _decode_method_n(rest)
            if inner is not None:
                base, n = inner
                return f"{base}_{last}", n
    if method_full in _FIXED_MODEL_METHODS:
        return method_full, 3000
    return None


def parse_run_name(run_name_core: str) -> RunIdentity | None:
    """Parse a run_name_core into a RunIdentity, or None if it doesn't match.

    Also normalizes the WARC-sweep method base: ``high_quality_random`` (left
    over after stripping the ``_300``) collapses to ``high_quality`` so it joins
    the METHOD_STYLE vocabulary.
    """
    m = RUN_NAME_RE.match(run_name_core)
    if not m:
        return None
    decoded = _decode_method_n(m.group("method_full"))
    if decoded is None:
        return None
    method, n = decoded
    if method.endswith("_random"):
        method = method[: -len("_random")]
    try:
        return RunIdentity(
            run_name_core=run_name_core,
            method=method,
            n_warcs=n,
            budget=float(m.group("budget")),
            dim=int(m.group("dim")),
            layers=int(m.group("layers")),
            batch=int(m.group("batch")),
            tag=m.group("tag"),
        )
    except ValueError:
        return None
