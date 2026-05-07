# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data curation IsoFLOP sweep with configurable simulated epoching.

Benchmarks web data curation methods (DCLM, Nemotron-CC, FineWeb-Edu,
Resiliparse, and an LLM-based method) across a compute sweep, at two
regimes:

- **Experiment A** (natural epoching): each method uses all its observed
  data; the simulated target regime scales with T_exp (T_target = T_exp * s).
  Covers the full delphi budget range.

- **Experiment B** (pinned target): T_target = 20T fixed. Applies simulated
  epoching. Candidates with T_exp > T_target / s are rejected (they live
  only in A).

See `/Users/michaelryan/.claude/plans/golden-questing-fountain.md` for the
full design.

Usage:

    # Dry-run to see expected run counts
    uv run experiments/scaling_law_sweeps/data_curation_isoflop.py --dry-run

    # Just DCLM, both experiments
    uv run experiments/scaling_law_sweeps/data_curation_isoflop.py --methods dclm

    # All methods, Experiment A only
    uv run experiments/scaling_law_sweeps/data_curation_isoflop.py --experiments A

    # Target us-east5-a (v5p TPUs) instead of us-central1 (v4)
    uv run experiments/scaling_law_sweeps/data_curation_isoflop.py --tpu-generation v5p
"""

import argparse
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace

from fray.cluster import ResourceConfig

from experiments.defaults import simulated_epoching_train
from experiments.scaling_law_sweeps.completed_adamh import (
    SEQ_LEN,
    _compute_tensor_parallel_size,
    _format_run_name,
    completed_adamh_heuristic,
)
from experiments.scaling_law_sweeps.data_curation_math import (
    CurationMethod,
    implicit_target_exp_a,
    load_d_obs_from_stats,
    slice_tokens_for,
    t_exp_ceiling,
)
from experiments.scaling_law_sweeps import region_tracker

# Hardcoded D_obs values (from .stats.json at the time of registry definition).
# We prefer to read these from GCS at plan-build time via `load_d_obs_from_stats`,
# but fall back to these values if GCS is unreachable (local dev, sandbox, or
# offline). Update these whenever the tokenize steps are re-materialized.
_D_OBS_DEFAULTS: dict[str, int] = {
    "baseline_dclm-23e9be": 2_663_454_015,
    "baseline_nemotron-c67de9": 1_919_401_016,
    "baseline_fineweb_edu-7a3bc5": 817_221_529,
    "baseline_resiliparse-7278c1": 142_652_598_588,
}

# Source bucket where baseline caches were originally materialized. Only used
# to read `.stats.json` at plan-build time; training itself uses `mirror://`.
_SOURCE_BUCKET: str = "gs://marin-us-central2"
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.scaling_laws import CandidateConfig, pick_v4_type, pick_v5p_type

# Imports for the unique-job-name wrapper (works around the hardcoded "train_lm"
# in marin.training.training._submit_training_job which collides when 127
# concurrent ExecutorSteps all dispatch through the same name within one
# coordinator scope).
import importlib as _importlib
import dataclasses as _dataclasses
from marin.training.training import (
    _submit_training_job,
    _prepare_training_run,
    TrainLmOnPodConfig,
)

logger = logging.getLogger(__name__)


# --- Compute budgets ----------------------------------------------------------
# Matches delphi's `completed_adamh` defaults (7 log-spaced FLOP points).
BUDGETS: tuple[float, ...] = (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)


# --- Experiment B target tokens ----------------------------------------------
# Grounded per internal review (20T ≈ 1T-param Chinchilla).
# Extension lever #1: append additional targets (e.g. 200e12 for Grok-scale)
# and the sweep automatically fans out — no other code changes needed.
DEFAULT_T_TARGETS: tuple[float, ...] = (20e12,)


# --- Curation methods registry ------------------------------------------------
# Extension lever #2: per-method `sampled_warcs`. Change a method's
# `sampled_warcs=3000` to e.g. 10000 after re-running extraction on a larger
# WARC sample. `s` and `d_proj` recompute automatically via properties.
#
# NOTE: `d_obs_tokens` is read at import time from `{path}/train/.stats.json`.
# Keeping the loader explicit (rather than hardcoding numbers) means the code
# stays correct if a method is re-tokenized.
def _method(
    name: str,
    cache_hash: str,
    sampled_warcs: int = 3_000,
    reproduce_per_region: bool = False,
    skip_stats_read: bool = False,
) -> CurationMethod:
    """Build a `CurationMethod` from a cache hash (the directory name under `tokenized/`).

    - `tokenized_path` is set to `mirror://tokenized/{cache_hash}/` so training
      reads from the local region's bucket (mirrored once on first read).
    - `d_obs_tokens` defaults to `_D_OBS_DEFAULTS[cache_hash]`. Unless
      `skip_stats_read=True`, we also verify against `.stats.json` at plan-build
      time — this catches stale hardcoded values after re-tokenization.
    """
    mirror_path = f"mirror://tokenized/{cache_hash}/"
    if cache_hash not in _D_OBS_DEFAULTS:
        raise KeyError(
            f"No hardcoded D_obs for {cache_hash!r}. Add it to _D_OBS_DEFAULTS "
            f"(read from {_SOURCE_BUCKET}/tokenized/{cache_hash}/train/.stats.json)."
        )
    d_obs = _D_OBS_DEFAULTS[cache_hash]

    if not skip_stats_read:
        source_stats_path = f"{_SOURCE_BUCKET}/tokenized/{cache_hash}/"
        try:
            live_d_obs = load_d_obs_from_stats(source_stats_path)
            if live_d_obs != d_obs:
                logger.warning(
                    "Stale d_obs for %s: hardcoded=%d but live=%d. " "Update _D_OBS_DEFAULTS[%r] = %d",
                    cache_hash,
                    d_obs,
                    live_d_obs,
                    cache_hash,
                    live_d_obs,
                )
                d_obs = live_d_obs
        except Exception as e:
            logger.warning(
                "Could not verify d_obs from %s (%s); using hardcoded=%d",
                source_stats_path,
                e,
                d_obs,
            )

    return CurationMethod(
        name=name,
        tokenized_path=mirror_path,
        d_obs_tokens=d_obs,
        sampled_warcs=sampled_warcs,
        reproduce_per_region=reproduce_per_region,
    )


def _build_methods(skip_stats_read: bool = False) -> dict[str, CurationMethod]:
    """Build the methods registry. `skip_stats_read=True` bypasses GCS verification (fast path for --dry-run)."""
    mk = lambda name, h, **kw: _method(name, h, skip_stats_read=skip_stats_read, **kw)  # noqa: E731
    return {
        "dclm": mk("dclm", "baseline_dclm-23e9be"),
        "nemotron_org": mk("nemotron_org", "baseline_nemotron-c67de9"),
        "fineweb_edu": mk("fineweb_edu", "baseline_fineweb_edu-7a3bc5"),
        # Resiliparse: 571 GB cache — `reproduce_per_region=True` signals that
        # the cache should be re-tokenized locally per region rather than
        # mirrored (for cost reasons). The flag is currently informational;
        # mirror:// still works but is expensive cross-continentally.
        "resiliparse": mk("resiliparse", "baseline_resiliparse-7278c1", reproduce_per_region=True),
    }


# --- Sweep builder ------------------------------------------------------------
MIN_SLICE_TOKENS_DEFAULT: float = 25e6
"""Safety belt — non-binding with current BUDGETS + T_TARGETS but asserted by tests.

If a future budget/target pair pushes a method's slice below this, those runs
are silently skipped. Tests are expected to flag any unexpected binding.
"""


def _pick_tpu_type(memory_bytes: int, generation: str) -> str:
    if generation == "v4":
        return pick_v4_type(memory_bytes)
    if generation == "v5p":
        return pick_v5p_type(memory_bytes)
    raise ValueError(f"Unknown tpu generation: {generation!r}. Expected 'v4' or 'v5p'.")


def _run_key_for_step(method_name: str, experiment_tag: str, run_name_core: str) -> str:
    """Stable identifier used by the region tracker."""
    return region_tracker.run_key_for(method_name, experiment_tag, run_name_core)


def preflight_region_lock(
    plans: list[tuple[CurationMethod, str, str]],
    *,
    local_region: str | None = None,
    tracker_prefix: str = region_tracker.DEFAULT_TRACKER_PREFIX,
    dry_run: bool = False,
) -> dict[str, str]:
    """Claim-or-verify region lock for every planned run.

    Call this BEFORE submitting training jobs. For each (method, experiment_tag,
    run_name) triple, claims `local_region` if unclaimed or verifies the claim
    matches local. Aborts on any mismatch by re-raising `RegionMismatch`.

    Guarantees: if this function returns successfully, every planned run is
    either (a) newly claimed for `local_region`, or (b) already pinned to
    `local_region` from a prior launch. Either way, checkpoints for these runs
    will never cross-region resume.

    `dry_run=True` skips the tracker entirely — useful for local enumeration.
    """
    if dry_run:
        return {}
    if local_region is None:
        local_region = region_tracker.detect_current_region()
    pinned: dict[str, str] = {}
    for method, experiment_tag, run_name_core in plans:
        key = _run_key_for_step(method.name, experiment_tag, run_name_core)
        claim = region_tracker.claim_or_read_region(
            key,
            local_region,
            tracker_prefix=tracker_prefix,
        )
        if claim.region != local_region:
            raise region_tracker.RegionMismatch(
                f"Run {key!r} is region-locked to {claim.region!r} but this "
                f"launch is in {local_region!r}. Re-schedule this launch to "
                f"{claim.region!r} or drop that method/experiment from the sweep."
            )
        pinned[key] = claim.region
        if claim.was_first_claim:
            logger.info("Region-locked %s → %s", key, local_region)
    return pinned


def _iter_valid_candidates(
    method: CurationMethod,
    *,
    t_target: float | None,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
) -> Iterator[tuple[float, CandidateConfig, int]]:
    """Yield (budget, candidate, target_budget) for every candidate that survives filtering.

    - Experiment A (t_target=None): no ceiling, implicit target = T_exp * s.
    - Experiment B (t_target set): reject T_exp > ceiling; reject slice < floor.

    Pure function — no ExecutorStep construction. Usable by tests and the
    dry-run enumerator.
    """
    for budget in budgets:
        for cand in completed_adamh_heuristic.candidates_for_budget(budget, seq_len=seq_len):
            t_exp = cand.tokens
            if t_target is not None:
                if t_exp > t_exp_ceiling(method, t_target):
                    continue
                if slice_tokens_for(method, t_exp, t_target) < min_slice_tokens:
                    continue
                target_budget = int(t_target)
            else:
                target_budget = int(implicit_target_exp_a(method, t_exp))
            yield budget, cand, target_budget


def _experiment_tag(t_target: float | None) -> str:
    if t_target is None:
        return "expA_natural"
    # For integer-T targets like 20T, 200T; use 1 decimal otherwise.
    trillions = t_target / 1e12
    if trillions == int(trillions):
        return f"expB_T{int(trillions)}T"
    return f"expB_T{trillions:.1f}T"


def build_curation_sweep(
    method: CurationMethod,
    *,
    t_target: float | None,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
    tpu_generation: str = "v4",
) -> list[ExecutorStep]:
    """Build all ExecutorSteps for one method at one experiment target.

    Args:
        method: curation method metadata (path, D_obs, sampled_warcs).
        t_target: None for Experiment A (natural epoching); a token count
            for Experiment B (pinned target).
        budgets: compute FLOP budgets to sweep.
        min_slice_tokens: safety floor on the unique-token pool in Exp B.
        seq_len: training sequence length (must match heuristic's expectations).
        tpu_generation: "v4" (us-central1) or "v5p" (us-east5-a). Selects the
            TPU picker.

    The training runs use `simulated_epoching_train`, which sets both
    `target_budget` and `experiment_budget` on the `LMMixtureDatasetConfig`.
    In Experiment A we still route through `simulated_epoching_train` with
    `target_budget = T_exp * s` so the config self-documents the implicit
    target regime.
    """
    tokenized = method.as_lm_mixture_config()
    tag = _experiment_tag(t_target)

    steps: list[ExecutorStep] = []
    for budget, candidate, target_budget in _iter_valid_candidates(
        method,
        t_target=t_target,
        budgets=budgets,
        min_slice_tokens=min_slice_tokens,
        seq_len=seq_len,
    ):
        steps.append(
            _make_train_step(
                method=method,
                candidate=candidate,
                budget=budget,
                target_budget=target_budget,
                experiment_tag=tag,
                tokenized=tokenized,
                tpu_generation=tpu_generation,
            )
        )
    return steps


def _unique_job_name_from_config(config: TrainLmOnPodConfig) -> str:
    """Derive a stable unique iris job name from the step's output_path.

    Marin's `run_levanter_train_lm` hardcodes `job_name="train_lm"` which
    collides when many concurrent ExecutorSteps dispatch under the same
    coordinator (each adopt_existing returns the same job handle, so only
    ONE training actually runs and 126 silently 'succeed' empty).

    By deriving the job name from `output_path` we make each dispatch unique
    while keeping `adopt_existing=True` semantics for resumes (same step →
    same name → resume the existing job).
    """
    if not config.output_path:
        return "train_lm"
    # Use the last path component, sanitized to fit iris job name constraints.
    leaf = config.output_path.rstrip("/").rsplit("/", 1)[-1]
    # Iris job names allow alphanumerics, dashes, underscores. Replace +/. with -.
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in leaf)
    # Prefix to keep the iris listings recognizable as marin train_lm jobs.
    return f"train_lm-{safe}"[:200]  # iris caps name length; be conservative


def _run_levanter_train_lm_unique(config: TrainLmOnPodConfig) -> None:
    """Same as marin's `run_levanter_train_lm` but uses unique job_name per step.

    This is the function we set as `fn=` on each ExecutorStep so concurrent
    parallel steps each create their OWN iris sub-job rather than colliding
    on a single hardcoded `train_lm` slot.
    """
    config, train_config, env, extras = _prepare_training_run(config)
    job_name = _unique_job_name_from_config(config)
    logger.info("Dispatching unique training job: %s", job_name)
    _submit_training_job(
        job_name=job_name,
        main_fn=_importlib.import_module("levanter.main.train_lm").main,
        train_config=train_config,
        resources=config.resources,
        env=env,
        extras=extras,
    )


def _make_train_step(
    *,
    method: CurationMethod,
    candidate: CandidateConfig,
    budget: float,
    target_budget: int,
    experiment_tag: str,
    tokenized,
    tpu_generation: str,
) -> ExecutorStep:
    """Construct one ExecutorStep for a single candidate. Mirrors `create_isoflop_sweep_steps`."""
    model_config = candidate.model_config
    estimated_memory = completed_adamh_heuristic.estimate_memory_bytes(candidate)
    tpu_type = _pick_tpu_type(estimated_memory, tpu_generation)
    tp = _compute_tensor_parallel_size(tpu_type, candidate.batch_size, model_config.hidden_dim)

    run_name_core = _format_run_name(
        budget,
        model_config.hidden_dim,
        model_config.num_layers,
        candidate.batch_size,
        experiment_name=f"curation-{method.name}-{experiment_tag}",
    )
    output_path = f"checkpoints/isoflop-curation/{run_name_core}"

    params = model_config.total_trainable_params(completed_adamh_heuristic.vocab_size)
    tags = (
        f"method={method.name}",
        f"exp={experiment_tag}",
        f"FLOPs={budget:.1e}",
        f"N={params:.1e}",
        f"B={candidate.batch_size}",
        f"steps={candidate.train_steps}",
        f"T_exp={candidate.tokens:.2e}",
        f"T_target={target_budget:.2e}",
        f"d_obs={method.d_obs_tokens:.2e}",
        f"d_proj={method.d_proj:.2e}",
        f"sampled_warcs={method.sampled_warcs}",
        f"s={method.s:.1f}",
        "optimizer=completed-adamh",
    )

    base_train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type),
        train_batch_size=candidate.batch_size,
        num_train_steps=candidate.train_steps,
        learning_rate=candidate.optimizer_config.learning_rate,
        z_loss_weight=completed_adamh_heuristic.z_loss_weight,
        env_vars={"LIBTPU_INIT_ARGS": "--xla_tpu_scoped_vmem_limit_kib=16000"},
    )
    train_cfg = replace(
        base_train_config,
        optimizer_config=candidate.optimizer_config,
        tensor_parallel_size=tp,
    )

    step = simulated_epoching_train(
        name=run_name_core,
        tokenized=tokenized,
        model_config=model_config,
        train_config=train_cfg,
        target_budget=target_budget,
        tags=tags,
        eval_harness_tasks=[],
        # Disable default validation sets (Paloma / Uncheatable-Eval). They're
        # tokenized in eu-west4 and conflict with our compute-region constraint.
        # TODO: re-enable uncheatable_eval via mirror:// once validation caches
        # are wrapped to be region-flexible.
        use_default_validation=False,
    )
    # Replace the default `run_levanter_train_lm` (hardcodes `job_name="train_lm"`
    # → 127 concurrent steps adopt one slot, only ONE actually trains) with our
    # unique-name wrapper so each ExecutorStep dispatches its own iris job.
    step = _dataclasses.replace(step, fn=_run_levanter_train_lm_unique)
    return step.with_output_path(output_path)


# --- Dry-run enumeration (no executor calls) ---------------------------------


def _count_valid(
    method: CurationMethod,
    *,
    t_target: float | None,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
) -> int:
    """Count valid candidates for a (method, t_target) pair. Used by tests + --dry-run."""
    return sum(
        1
        for _ in _iter_valid_candidates(
            method,
            t_target=t_target,
            budgets=budgets,
            min_slice_tokens=min_slice_tokens,
            seq_len=seq_len,
        )
    )


def _per_budget_counts(
    method: CurationMethod,
    *,
    t_target: float | None,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
) -> dict[float, int]:
    counts: dict[float, int] = {b: 0 for b in budgets}
    for budget, _, _ in _iter_valid_candidates(
        method,
        t_target=t_target,
        budgets=budgets,
        min_slice_tokens=min_slice_tokens,
        seq_len=seq_len,
    ):
        counts[budget] += 1
    return counts


def _tpu_distribution(
    method: CurationMethod,
    *,
    t_target: float | None,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
    tpu_generation: str = "v4",
) -> Counter:
    tpu_counts: Counter = Counter()
    for _, cand, _ in _iter_valid_candidates(
        method,
        t_target=t_target,
        budgets=budgets,
        min_slice_tokens=min_slice_tokens,
        seq_len=seq_len,
    ):
        mem = completed_adamh_heuristic.estimate_memory_bytes(cand)
        tpu_counts[_pick_tpu_type(mem, tpu_generation)] += 1
    return tpu_counts


def _print_dry_run(
    methods: list[CurationMethod],
    experiments: list[float | None],
    *,
    tpu_generation: str,
) -> None:
    total = 0
    for method in methods:
        for t_target in experiments:
            tag = _experiment_tag(t_target)
            counts = _per_budget_counts(method, t_target=t_target)
            n = sum(counts.values())
            total += n
            ceiling_str = f"ceiling={t_exp_ceiling(method, t_target)/1e9:.2f}B" if t_target is not None else "no ceiling"
            per_budget = [counts[b] for b in BUDGETS]
            tpus = _tpu_distribution(method, t_target=t_target, tpu_generation=tpu_generation)
            tpu_breakdown = ", ".join(
                f"{k}:{v}" for k, v in sorted(tpus.items(), key=lambda kv: int(kv[0].split("-")[1]))
            )
            print(
                f"{method.name:>15} | {tag:>14} | runs={n:>3} | "
                f"{ceiling_str:>18} | per-budget={per_budget} | tpus=[{tpu_breakdown}]"
            )
    print(f"\n  TOTAL runs across all method × experiment combinations: {total}")


# --- CLI ----------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Subset of curation methods to launch (or 'all'). See METHODS for names.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=["A", "B", "all"],
        help="Which experiments to run. A=natural epoching, B=pinned target (default both).",
    )
    parser.add_argument(
        "--t-targets",
        nargs="+",
        type=float,
        default=list(DEFAULT_T_TARGETS),
        help="T_target values (tokens) for Experiment B. Extension lever #1 — append more here.",
    )
    parser.add_argument(
        "--tpu-generation",
        choices=["v4", "v5p"],
        default="v4",
        help="TPU generation for resource selection. v4 for us-central1, v5p for us-east5-a.",
    )
    parser.add_argument(
        "--min-slice-tokens",
        type=float,
        default=MIN_SLICE_TOKENS_DEFAULT,
        help="Safety floor on Experiment B slice size (unique tokens). Default 25M.",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=list(BUDGETS),
        help="Compute (FLOPs) budgets to sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate and print expected run counts; do not submit jobs.",
    )
    parser.add_argument(
        "--skip-region-lock",
        action="store_true",
        help="DANGER: disables the region-lock preflight check. Only use if you "
        "understand the cross-region egress risk and have another mechanism "
        "to prevent resume-from-wrong-region.",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def _resolve_methods(method_names: list[str], registry: dict[str, CurationMethod]) -> list[CurationMethod]:
    """Resolve method names against the registry. `--methods all` includes every registered method."""
    if "all" in method_names:
        return list(registry.values())
    missing = [m for m in method_names if m not in registry]
    if missing:
        raise ValueError(f"Unknown method(s): {missing}. Available: {list(registry)}")
    return [registry[m] for m in method_names]


def _resolve_experiments(experiments: list[str], t_targets: list[float]) -> list[float | None]:
    resolved: list[float | None] = []
    want_a = "A" in experiments or "all" in experiments
    want_b = "B" in experiments or "all" in experiments
    if want_a:
        resolved.append(None)
    if want_b:
        resolved.extend(t_targets)
    return resolved


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    # For --dry-run, skip the GCS verification (it's the slow path; uses hardcoded d_obs).
    registry = _build_methods(skip_stats_read=args.dry_run)
    methods = _resolve_methods(args.methods, registry)
    experiments = _resolve_experiments(args.experiments, args.t_targets)
    budgets = tuple(args.budgets)

    if args.dry_run:
        _print_dry_run(methods, experiments, tpu_generation=args.tpu_generation)
        return

    # Build plans (method, experiment_tag, run_name) for preflight region-lock.
    steps: list[ExecutorStep] = []
    plans: list[tuple[CurationMethod, str, str]] = []
    for method in methods:
        for t_target in experiments:
            tag = _experiment_tag(t_target)
            logger.info("Building sweep: method=%s experiment=%s", method.name, tag)
            method_steps = build_curation_sweep(
                method,
                t_target=t_target,
                budgets=budgets,
                min_slice_tokens=args.min_slice_tokens,
                tpu_generation=args.tpu_generation,
            )
            steps.extend(method_steps)
            for step in method_steps:
                # Output path: "checkpoints/isoflop-curation/{run_name_core}"
                run_name_core = step.override_output_path.split("/", maxsplit=2)[-1]
                plans.append((method, tag, run_name_core))

    # Region-lock preflight: claim/verify EVERY run against the tracker.
    # Aborts before submission if any run is pinned to a different region.
    # Skip when explicitly disabled (rare — only for dev/experimentation).
    if not args.skip_region_lock:
        logger.info("Running preflight region-lock check for %d runs", len(plans))
        preflight_region_lock(plans)
        logger.info("Preflight passed: all runs pinned to the launch region")
    else:
        logger.warning("Region-lock preflight SKIPPED (--skip-region-lock). Cross-region egress risk!")

    logger.info("Total steps to submit: %d", len(steps))
    # executor_main is wrapped with @draccus.wrap() which re-parses sys.argv.
    # We already consumed our CLI args, so clear sys.argv to avoid a re-parse
    # collision with our --methods/--experiments/etc flags.
    import sys

    sys.argv = [sys.argv[0]]
    executor_main(steps=steps, description="Data curation IsoFLOP sweep (DCLM + baselines)")


if __name__ == "__main__":
    main()
