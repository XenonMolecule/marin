# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan enumeration for the WARC-scaling data-curation sweep.

Scientific question: how does the per-budget loss-vs-FLOPs curve (and the
crossover between curation methods) shift as the number of source WARCs is
subsampled? We hold the params/WARC ratio approximately fixed, snap one slot
per N to the canonical 157M (h=512) architecture so the cross-N comparison is
clean for at least one architecture, and let budgets follow the data-cliff
positions per N.

Design vs `fixed_model_plan.py`:
  - Per-N model lineup (varying hidden_size) instead of one fixed trio.
  - Per-N budget grid (centered on the data-cliff at that N) instead of one
    canonical 7-budget grid.
  - Methods: subsample CurationMethods registered with sampled_warcs=N
    (e.g. `dclm_100`, `llm_curated_500`, ...). Forces correct slicing math
    via `D_obs(N)` and `s = TOTAL_WARCS_CC / N`.
  - Distinct experiment_tag ("expWARC_natural") so outputs don't collide with
    fixed-model runs.

Same downstream contract: emits `PlannedRun`s consumed by
`launch_curation_sweep.submit_one` and `run_curation_train_standalone.py`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from experiments.scaling_law_sweeps.completed_adamh import SEQ_LEN
from experiments.scaling_law_sweeps.curation_plan import (
    METHODS,
    PlannedRun,
    _planned_run_from_candidate,
)
from experiments.scaling_law_sweeps.data_curation_math import (
    CurationMethod,
    implicit_target_exp_a,
)
from experiments.scaling_law_sweeps.fixed_model_plan import _candidate_for_fixed_model

# Override the PlannedRun.memory_gb default (256, set for fixed-model d4096+L40
# multi-host plans). Host RAM has a ~constant floor INDEPENDENT of model size:
# the in-training eval harness datasets (paloma / uncheatable), the data-loader
# prefetch, the tokenizer, and JAX/wandb overhead together need tens of GB no
# matter how small the model is. The old size-scaled tiers (24 GB for h<=256,
# 32 GB for h<=768) sized only on params+activations and so UNDER-provisioned
# tiny models: d256 cells OOM-killed (exit 137) at 24 GB even at batch 16
# (resiliparse_dedup_100 1e18/B16, 3e18/B64, 2026-05-30). The param/activation
# term only dominates for big models, so we floor EVERY cell at 48 GB (which
# cleared the OOMs in practice); v5p/v4 hosts have ample RAM so this does not
# hurt scheduling. Large-batch corners are bumped further in _memory_gb_for_plan.
_MEMORY_FLOOR_GB: int = 48


def _memory_gb_for_hidden(hidden_dim: int) -> int:
    return _MEMORY_FLOOR_GB


# Big batch sizes balloon host RAM (gradient bookkeeping, JAX layout caches,
# loader prefetch) even when TPU HBM is comfortable. The per-hidden tier is
# correct at canonical batches but undershoots when AdamH HP produces large B
# at small d (large-budget x small-model corner of the IsoFLOP grid).
# Calibrated against observed exit-137 OOMs:
#   - h=256, B=256 (1e19 budget) OOM'd at 24 GB -> 64 GB needed
def _memory_gb_for_plan(hidden_dim: int, batch_size: int) -> int:
    base = _memory_gb_for_hidden(hidden_dim)
    if hidden_dim <= 256 and batch_size >= 128:
        return max(base, 64)
    return base


# All WARC subsample sizes this sweep covers.
WARC_COUNTS: tuple[int, ...] = (100, 500, 1000, 2000)

# Canonical anchor architecture present at every N for cross-N comparison.
ANCHOR_HIDDEN_SIZE: int = 512  # 156.5M params under AdamH heuristic

# Architecture floor for the small slot when params/WARC scaling would request
# something below it. h=128 / h=192 are below where the AdamH HP recipe was
# fit; collapse to h=256.
HIDDEN_SIZE_FLOOR: int = 256

# 5 methods (fineweb_edu dropped per user direction 2026-04-29).
WARC_METHOD_BASE_NAMES: tuple[str, ...] = (
    "dclm",
    "nemotron_full",
    # Nemotron-CC-HQ (quality=high only -- Nvidia's "HQ" subset definition,
    # ~37% of full Nemotron-CC tokens). Subsamples built via subset_baselines.py
    # from the existing baseline_nemotron_qhigh-v1 3k filtered output.
    "nemotron_qhigh",
    "llm_curated",
    "resiliparse",
    # resiliparse with per-N fuzzy dedup. N=100 only so far; the larger
    # N's are pending a viable compute strategy (us-central2 v4 was
    # exhausted; eu-west4 was used as one-off smoke).
    "resiliparse_dedup",
    "llm_curated_dclm_filtered",
    # LLM-extracted quality-band sweeps. Registered via dedup_extracted.py +
    # tokenize_deduped_extracted.py at N=100 (others to follow as extraction
    # for those N's lands). Method names match the curation_plan.METHODS keys.
    "low_quality",
    "med_low_quality",
    "med_quality",
    "high_quality",
)

# Per-N model size lineup. Each entry is a list of hidden_sizes under the
# AdamH heuristic. Derivation:
#   - Targets: 0.052M / 0.333M / 1.0M params per WARC (matching the 3000-WARC
#     anchor's 157M / 1B / 3B trio).
#   - Map each target to the closest hidden_size with mlp_ratio + layers
#     given by `_compute_num_layers`. Not every hidden_size is canonical in
#     delphi's enumerator (which only steps from h=512 upward) but the
#     formula is continuous, so off-grid h is well-defined; HP recipe is
#     extrapolating for very small h (accepted trade-off).
#   - 2x snap-to-anchor band: any slot within [78.5M, 314M] params (i.e. 0.5x-2x
#     of 157M) snaps to h=512. Otherwise, if no slot is in band, h=512 is
#     added as an EXTRA slot for the cross-N anchor.
#   - At N=100, scaled small (5.2M) and mid (33M) targets both round up to
#     h=256 (69M) given the floor; the small slot is dropped (redundant).
_HIDDEN_SIZES_PER_N: dict[int, tuple[int, ...]] = {
    # N=100: small/mid collapse to h=256 (drop small); large 100M snaps to anchor.
    # 768 and 1536 added 2026-05-24 for the dense 100-WARC quality-tier grid
    # (low/med/high_quality). Existing N=100 methods unaffected unless launched
    # with the new hidden sizes via --only-hidden-sizes.
    100: (256, 512, 768, 1536),
    # N=500: mid 167M snaps to anchor; small 26M floors to h=256; large 500M -> h=1024.
    500: (256, 512, 1024),
    # N=1000: scaled trio + anchor extra (none of the trio in the snap band).
    1000: (256, 768, 1536, 512),
    # N=2000: small 105M snaps to anchor; mid 666M -> h=1280; large 2B -> h=2048.
    2000: (512, 1280, 2048),
}

# Per-N compute (FLOPs) budget grid, log-spaced to bracket the cliff range
# `[6 * smallest_params * D_obs(dclm,N), 6 * largest_params * D_obs(dclm,N)]`
# with ~1 decade of padding either side. DCLM is the data-constrained anchor
# whose cliff drives the crossover position.
_BUDGETS_PER_N: dict[int, tuple[float, ...]] = {
    # 1e19 and 1e20 added 2026-05-24 for the dense 100-WARC quality-tier grid.
    100: (3e15, 1e16, 3e16, 1e17, 3e17, 1e18, 3e18, 1e19, 1e20),
    500: (1e16, 3e16, 1e17, 3e17, 1e18, 3e18, 1e19),
    1000: (3e16, 1e17, 3e17, 1e18, 3e18, 1e19, 3e19, 1e20),
    2000: (1e17, 3e17, 1e18, 3e18, 1e19, 3e19, 1e20, 3e20),
}

# Two-point high-end extension per N -- used for resiliparse + llm_curated only,
# to give the data-rich methods a fair shot in the high-compute regime where
# DCLM/Nemotron have already hit their data cliff. Follows the canonical
# 3-9-1.8 progression from each grid's top.
#
# N=2000's would-be top (1.8e21) was dropped because the resulting plan needs
# v4-512 / v5p-512 gang-scheduled across 64 VMs, which doesn't realistically
# land at interactive priority within paper-deadline timescales.
_BUDGET_EXTENSIONS_PER_N: dict[int, tuple[float, ...]] = {
    # 1.8e20 added 2026-05-19 to push one tier above canonical for
    # low_quality/med_quality at N=100 d=512 and low_quality/high_quality at
    # N=500 d=1024, per user request to extend the curve.
    100: (9e18, 1.8e19, 3e19, 9e19, 1.8e20),
    500: (3e19, 9e19, 1.8e20),
    1000: (3e20, 9e20),
    2000: (9e20,),
}

# Tag the same way as fixed_model: "expWARC_<region>" -> since we run natural
# epoching only (no slicing), use the natural suffix. Distinct prefix
# ("expWARC") so:
#   - run_name_core differs from fixed-model runs (no GCS path collision)
#   - `_build_summary` routes through the natural-epoch branch (slice = D_obs)
#   - WandB filter "experiment_tag=expWARC_natural" cleanly isolates this sweep
EXPERIMENT_TAG: str = "expWARC_natural"


def hidden_sizes_for(n_warcs: int) -> tuple[int, ...]:
    """Hidden sizes to train at this WARC count."""
    if n_warcs not in _HIDDEN_SIZES_PER_N:
        raise ValueError(f"Unknown n_warcs={n_warcs}; expected one of {WARC_COUNTS}.")
    return _HIDDEN_SIZES_PER_N[n_warcs]


def budgets_for(n_warcs: int, include_extensions: bool = False) -> tuple[float, ...]:
    """Compute budgets to sweep at this WARC count.

    `include_extensions=True` appends the 2-point high-end extension to give
    data-rich methods (resiliparse, llm_curated) a fair shot beyond the cliff.
    Should NOT be used for dclm/nemotron_full (they're past their data cliff
    at the existing top budgets -- extra runs would be uninformative).
    """
    if n_warcs not in _BUDGETS_PER_N:
        raise ValueError(f"Unknown n_warcs={n_warcs}; expected one of {WARC_COUNTS}.")
    base = _BUDGETS_PER_N[n_warcs]
    if include_extensions:
        return base + _BUDGET_EXTENSIONS_PER_N.get(n_warcs, ())
    return base


def method_for(base_name: str, n_warcs: int) -> CurationMethod:
    """Resolve `(base_name, n_warcs)` to the registered CurationMethod.

    Raises if the method isn't registered (likely a typo in `base_name` or an
    unsupported `n_warcs`).
    """
    method_key = f"{base_name}_{n_warcs}"
    if method_key not in METHODS:
        raise KeyError(
            f"Method {method_key!r} not registered in curation_plan.METHODS. "
            f"Add it (and the corresponding _D_OBS_DEFAULTS entry) before enumerating."
        )
    return METHODS[method_key]


def resolve_methods(base_names: Iterable[str]) -> list[str]:
    """Resolve --methods CLI values. 'all' expands to WARC_METHOD_BASE_NAMES."""
    names = list(base_names)
    if "all" in names:
        return list(WARC_METHOD_BASE_NAMES)
    unknown = [n for n in names if n not in WARC_METHOD_BASE_NAMES]
    if unknown:
        raise ValueError(f"Unknown method base name(s) {unknown}. Available: {WARC_METHOD_BASE_NAMES}")
    return names


def enumerate_warc_scaling_plans(
    method_base_names: Iterable[str],
    *,
    n_warcs_list: Iterable[int] = WARC_COUNTS,
    seq_len: int = SEQ_LEN,
    extension_methods: tuple[str, ...] = (),
    extension_only: bool = False,
    only_budgets: tuple[float, ...] | None = None,
    only_hidden_sizes: tuple[int, ...] | None = None,
    batch_divisor: int = 1,
) -> list[PlannedRun]:
    """Cartesian product over (method_base_name x n_warcs x hidden_size x budget).

    For each cell, build a CandidateConfig via `_candidate_for_fixed_model`
    (forces corners, floors batch at min_batch_size). Skip cells whose
    heuristic returns None (would still trip clamps even at the batch floor).

    `t_target` is set to the implicit ExpA target (`t_exp * s`) so downstream
    summary/plot code interprets it identically to the fixed-model sweep's
    natural-epoching runs.

    `extension_methods`: subset of base_names that should ALSO get the
    `_BUDGET_EXTENSIONS_PER_N` high-end budgets. Used to give data-rich
    methods (resiliparse, llm_curated) more room past the DCLM/Nemotron cliff.

    `extension_only`: if True, ONLY emit plans for the extension budgets (not
    the base grid). Use for an interactive-priority coord that races just the
    high-end plans against capacity contention.
    """
    base_names = resolve_methods(method_base_names)
    plans: list[PlannedRun] = []
    extension_set = set(extension_methods)
    only_budgets_set = {float(b) for b in only_budgets} if only_budgets else None
    only_hidden_set = set(only_hidden_sizes) if only_hidden_sizes else None
    for n_warcs in n_warcs_list:
        hidden_sizes = hidden_sizes_for(n_warcs)
        if only_hidden_set is not None:
            hidden_sizes = tuple(h for h in hidden_sizes if h in only_hidden_set)
        for base_name in base_names:
            if extension_only:
                # Restrict to the extension budgets only -- used by the
                # priority-bump coord. Skips base budgets entirely.
                if base_name not in extension_set:
                    continue
                budgets = _BUDGET_EXTENSIONS_PER_N.get(n_warcs, ())
                if not budgets:
                    continue
            else:
                budgets = budgets_for(n_warcs, include_extensions=base_name in extension_set)
            method = method_for(base_name, n_warcs)
            for hidden_size in hidden_sizes:
                for budget in budgets:
                    if only_budgets_set is not None and not any(abs(budget - b) / b < 0.01 for b in only_budgets_set):
                        continue
                    candidate = _candidate_for_fixed_model(hidden_size, budget, seq_len=seq_len)
                    if candidate is None:
                        continue
                    # Apply batch_divisor: shrink batch + re-derive HP for the
                    # smaller batch (using the SAME AdamH formulas that built
                    # the original candidate, just at a smaller B).
                    if batch_divisor > 1:
                        candidate = _shrink_candidate_batch(candidate, batch_divisor, seq_len=seq_len)
                        if candidate is None:
                            continue
                    target_budget = int(implicit_target_exp_a(method, candidate.tokens))
                    plan = _planned_run_from_candidate(
                        method,
                        candidate,
                        budget,
                        target_budget,
                        EXPERIMENT_TAG,
                        seq_len,
                    )
                    # Override memory_gb so children schedule alongside other
                    # tenants on shared workers (which report ~16 GB free).
                    plan = dataclasses.replace(plan, memory_gb=_memory_gb_for_plan(plan.hidden_dim, plan.batch_size))
                    # If batch was shrunk, override the TPU shape selection.
                    # The default `_planned_run_from_candidate` bumps v5p UP to
                    # match v4's vm_count, which defeats the purpose of the
                    # batch shrink (v5p-256 instead of v5p-32). Use raw
                    # pick_v5p_type and align v4 to the SAME vm_count as v5p.
                    if batch_divisor > 1:
                        from iris.cluster.types import get_tpu_topology
                        from marin.scaling_laws import pick_v5p_type

                        v5p_raw = pick_v5p_type(plan.estimated_memory_bytes)
                        vm_count_v5p = get_tpu_topology(v5p_raw).vm_count
                        # Pair with a v4 of the same vm_count (cores = vm_count*8).
                        # The v4 alt may not actually fit the model, but iris
                        # won't dispatch it in our regions anyway -- v5p in us-east5
                        # is the real target. The match is only needed so the
                        # device_variant_constraint filter (line 148 of
                        # launch_curation_sweep) doesn't drop v5p as a "wrong
                        # vm_count" alternative.
                        v4_match = f"v4-{vm_count_v5p * 8}"
                        plan = dataclasses.replace(plan, v5p_tpu=v5p_raw, v4_tpu=v4_match)
                    plans.append(plan)
    return plans


def _shrink_candidate_batch(candidate, divisor: int, *, seq_len: int = SEQ_LEN):
    """Return a CandidateConfig with batch_size÷divisor and stepsxdivisor.

    Re-runs the AdamH HP formulas (`_compute_learning_rate`, `_compute_adam_lr`,
    `_compute_beta2`) for the smaller batch, then rebuilds the optimizer config
    via `build_optimizer_config(new_batch, tokens)` so all hyperparameters
    match what the heuristic would have produced if it had naturally chosen
    the smaller batch.

    Used to shrink the TPU footprint of high-budget plans (e.g. v5p-256 ->
    v5p-32) without changing the FLOP budget. Returns None if shrunk batch
    falls below the heuristic's `min_batch_size`.
    """
    from marin.scaling_laws import CandidateConfig

    from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

    h = completed_adamh_heuristic
    new_batch = candidate.batch_size // divisor
    if new_batch < h.min_batch_size:
        return None
    new_steps = candidate.train_steps * divisor
    actual_tokens = new_batch * new_steps * seq_len
    new_opt = h.build_optimizer_config(new_batch, actual_tokens)
    return CandidateConfig(
        model_config=candidate.model_config,
        optimizer_config=new_opt,
        batch_size=new_batch,
        train_steps=new_steps,
        tokens=actual_tokens,
        flops_budget=candidate.flops_budget,
    )
