# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan enumeration for the fixed-model data-curation sweep.

Scientific question: for each of three fixed model sizes (157M / 998M / 2.9B),
how does final loss scale with compute budget across curation methods?

Differences from the older `curation_plan.enumerate_plans`:
  - Fixed model grid: hidden_dim in {512, 1536, 2432} (no per-budget
    Chinchilla-optimal model search).
  - Bypasses `CompletedAdamHHeuristic._max_params_for_budget` so forced corners
    like 8.11B@3e18 (undertrained) and 157M@3e20 (overtrained) are included.
  - Single experiment regime: natural epoching (no ExpA/B toggle). Distinct
    `experiment_tag` ("expFM_natural") so outputs don't collide with the older
    sweep's runs in GCS/WandB.
  - Per-run hparams still come from the AdamH heuristic
    (`_build_candidate_config`), so each (hidden_size, budget) point is native
    compute-optimal for that (B, T) pair.

`PlannedRun` is unchanged -- everything downstream (`launch_curation_sweep.submit_one`,
`run_curation_train_standalone.py`, `plot_curation_isoflop.py`) consumes it as-is.
"""

from __future__ import annotations

import dataclasses

from marin.scaling_laws import CandidateConfig

from experiments.scaling_law_sweeps.completed_adamh import (
    SEQ_LEN,
    completed_adamh_heuristic,
)
from experiments.scaling_law_sweeps.curation_plan import (
    BUDGETS,
    METHODS,
    PlannedRun,
    _planned_run_from_candidate,
)
from experiments.scaling_law_sweeps.data_curation_math import (
    CurationMethod,
    implicit_target_exp_a,
)

# Hidden dims for the fixed model sizes: 157M / 998M / 2.9B Qwen3.
# Derived via `completed_adamh_heuristic._build_model_config(hidden_size)`
# which sets L, heads, and d_ff from hidden_size alone.
# 2026-05-31: dropped 3328 (6.7B) and 3584 (8.11B). The canon-3k anchor never
# trained those widths, so fixed-model / random-3k runs there have no point of
# comparison -- pure wasted compute. To restore, re-add them to the tuple:
#     TARGET_HIDDEN_SIZES = (512, 1536, 2432, 3328, 3584)
# (The 10k natural-epoch sweep still trains 3584 via launch_10k_natural.WIDTHS,
# which passes hidden_sizes explicitly and does not use this default -- unaffected.)
TARGET_HIDDEN_SIZES: tuple[int, ...] = (512, 1536, 2432)

# Tag distinguishes this sweep's runs from the older ExpA/ExpB sweep in GCS
# paths, WandB run names, and region-tracker keys. Still starts with "exp"
# and doesn't start with "expB", so `_build_summary` in
# `run_curation_train_standalone.py` routes it through the natural-epoch
# branch (slice_tokens = d_obs_tokens).
EXPERIMENT_TAG: str = "expFM_natural"


def _candidate_for_fixed_model(
    hidden_size: int,
    budget: float,
    seq_len: int = SEQ_LEN,
) -> CandidateConfig | None:
    """Build a CandidateConfig for a fixed hidden_size at a given budget.

    Bypasses `candidates_for_budget`'s `_max_params_for_budget` + tokens/param
    filters -- we explicitly want forced corners (157M @ 3e20 overtrained, 8.11B
    @ 3e18 undertrained). When the AdamH heuristic's natural `batch_exact =
    tokens / (STEPS_PER_RUN * seq_len)` drops below `min_batch_size=8` (happens
    for large-model x small-budget corners like 8.11B @ 3e18 where tokens
    ~= 5.8e7), we floor batch at `min_batch_size` and let steps shrink instead of
    returning None. The resulting runs are very thin (1-2k steps) but that's
    scientifically correct -- the data point answers "what does 3e18 FLOPs do
    to an 8.11B model" with "essentially nothing", which IS the right answer.
    Returns None only if hparams trip their clamps even at floored batch.
    """
    h = completed_adamh_heuristic
    model_config = h._build_model_config(hidden_size, seq_len=seq_len)
    flops_per_token = model_config.flops_per_token(h.vocab_size, seq_len)
    tokens = budget / (3 * flops_per_token)

    candidate = h._build_candidate_config(model_config, tokens, budget, seq_len=seq_len)
    if candidate is not None:
        return candidate

    # Floor batch at min_batch_size for corners where natural batch is tiny.
    batch_size = h.min_batch_size
    lr = h._compute_learning_rate(batch_size, tokens)
    adam_lr = h._compute_adam_lr(batch_size, tokens)
    beta2 = h._compute_beta2(batch_size)
    if lr >= h.max_learning_rate or adam_lr >= h.max_learning_rate or beta2 <= h.min_beta2:
        # Would still trip clamps even at floor -- genuinely unschedulable.
        return None
    train_steps = max(1, round(tokens / (batch_size * seq_len)))
    actual_tokens = batch_size * train_steps * seq_len
    return CandidateConfig(
        model_config=model_config,
        optimizer_config=h.build_optimizer_config(batch_size, tokens),
        batch_size=batch_size,
        train_steps=train_steps,
        tokens=actual_tokens,
        flops_budget=budget,
    )


def enumerate_fixed_model_plans(
    methods: list[CurationMethod],
    *,
    hidden_sizes: tuple[int, ...] = TARGET_HIDDEN_SIZES,
    budgets: tuple[float, ...] = BUDGETS,
    seq_len: int = SEQ_LEN,
    batch_divisor: int = 1,
) -> list[PlannedRun]:
    """Return PlannedRuns for the cartesian product methods x hidden_sizes x budgets.

    Natural epoching only (no ExpA/B distinction). t_target is set to the
    implicit Experiment-A target (`t_exp * s`) so downstream summary/plot code
    can interpret it identically to ExpA runs from the older sweep.

    `batch_divisor`: mirror of `warc_scaling_plan.enumerate_warc_scaling_plans`
    -- shrink each plan's batch by this factor (and re-derive HP) to drop the
    TPU shape (e.g. v5p-256 -> v5p-32) when preemption gang-bounces on
    multi-host slices make a clean run impractical. run_name_core differs from
    the un-shrunk variant (it embeds the new B), so the shrunk cell starts
    fresh -- no topology-mismatch resume from the old multi-host checkpoint.
    """
    # Lazy import to avoid a cycle (warc_scaling_plan imports from fixed_model_plan).
    from experiments.scaling_law_sweeps.warc_scaling_plan import _shrink_candidate_batch

    plans: list[PlannedRun] = []
    for method in methods:
        for hidden_size in hidden_sizes:
            for budget in budgets:
                candidate = _candidate_for_fixed_model(hidden_size, budget, seq_len=seq_len)
                if candidate is None:
                    # Heuristic rejected the (B, T) combo -- e.g. batch size too
                    # small after the hparam-clamp descent. Shouldn't happen for
                    # our intended grid, but log-skip rather than crash.
                    continue
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
                # When batch is shrunk, override the default TPU shape
                # selection -- same fix as warc_scaling_plan does -- so the
                # shrink actually drops the TPU footprint (without this,
                # _planned_run_from_candidate bumps v5p UP to match v4's
                # vm_count, undoing the point of the divisor).
                if batch_divisor > 1:
                    from iris.cluster.types import get_tpu_topology
                    from marin.scaling_laws import pick_v5p_type

                    from experiments.scaling_law_sweeps.curation_plan import pick_v6e_type_single_vm

                    v5p_raw = pick_v5p_type(plan.estimated_memory_bytes)
                    vm_count_v5p = get_tpu_topology(v5p_raw).vm_count
                    v4_match = f"v4-{vm_count_v5p * 8}"
                    # Also recompute v6e for the shrunk batch — without this the v6e slice stays at the
                    # pre-shrink value from _planned_run_from_candidate, so batch-divisor runs never offer
                    # v6e as a scheduling variant.
                    plan = dataclasses.replace(
                        plan,
                        v5p_tpu=v5p_raw,
                        v4_tpu=v4_match,
                        v6e_tpu=pick_v6e_type_single_vm(plan.estimated_memory_bytes) or "",
                    )
                plans.append(plan)
    return plans


def resolve_methods(method_names: list[str]) -> list[CurationMethod]:
    """Resolve --methods CLI values. "all" expands to every registered non-deferred method.

    Excludes:
      - resiliparse (deferred per plan: 571 GB cache; mirrored to both
        us-central1 and us-central2 — pin_region="us-central1" forces ExpC
        runs to the same region as fixed-model summaries for cross-experiment
        dedup).
      - ExpC 10k placeholders (dclm_10k, nemotron_10k) — these are registered
        with d_obs=0 until tokenization completes; passing them through the
        fixed-model pipeline would silently produce broken plans. Callers who
        explicitly want them must list them by name AFTER filling in real
        d_obs values in `curation_plan._D_OBS_DEFAULTS`.
    """
    expc_placeholders = {"dclm_10k", "nemotron_10k"}
    if "all" in method_names:
        return [m for name, m in METHODS.items() if name != "resiliparse" and name not in expc_placeholders]
    missing = [n for n in method_names if n not in METHODS]
    if missing:
        raise ValueError(f"Unknown method(s): {missing}. Available: {list(METHODS)}")
    return [METHODS[n] for n in method_names]


def resolve_hidden_sizes(values: list[int] | None) -> tuple[int, ...]:
    """Resolve --hidden-sizes CLI values. None expands to all three fixed sizes."""
    if not values:
        return TARGET_HIDDEN_SIZES
    unknown = [v for v in values if v not in TARGET_HIDDEN_SIZES]
    if unknown:
        raise ValueError(f"Unsupported hidden_size(s) {unknown}. Expected subset of {TARGET_HIDDEN_SIZES}.")
    return tuple(values)
