# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator for the fixed-model data-curation sweep.

Iris CPU-only parent job that enumerates the fixed-model grid (see
`fixed_model_plan.enumerate_fixed_model_plans`) and submits each plan as an
independent Iris TPU child running `run_curation_train_standalone.py`.

Mirrors `launch_curation_sweep.py` but:
  - swaps `curation_plan.enumerate_plans` for the fixed-model enumerator,
  - adds `--hidden-sizes` so the user can stage tiers (998M first, then 157M,
    then 8.11B) by launching successive coordinator jobs with different
    hidden-size subsets,
  - uses a distinct `--wandb-group` and `--tracker-prefix` so outputs don't
    collide with the older ExpA/B sweep.

Typical usage (from an Iris parent):

    # Tier 1: 998M across all 4 methods x 7 budgets = 28 runs.
    iris --cluster marin job run --priority production --no-wait \\
        --memory 4GB --cpu 4 --job-name curation-fm-998m \\
        -- python experiments/scaling_law_sweeps/launch_fixed_model_sweep.py \\
        --methods all --hidden-sizes 1536 --child-priority batch

    # Tier 2: 157M.
    iris ... -- python ...launch_fixed_model_sweep.py \\
        --methods all --hidden-sizes 512 --child-priority batch

    # Tier 3 (optional): 8.11B.
    iris ... -- python ...launch_fixed_model_sweep.py \\
        --methods all --hidden-sizes 3584 --child-priority batch
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from experiments.scaling_law_sweeps import curation_plan, fixed_model_plan
from experiments.scaling_law_sweeps.launch_curation_sweep import (
    PRIORITY_BAND_MAP,
    submit_all,
)
from iris.client.client import IrisClient

logger = logging.getLogger(__name__)

# Distinct WandB group, tracker prefix, and results prefix so fixed-model runs
# land in their own namespaces and don't mingle with the older ExpA/B sweep's
# outputs. All three live under gs://marin-us-central1 for zero cross-region
# reads during analysis / region-lock lookup.
DEFAULT_WANDB_GROUP = "data-curation-fixed-model"
DEFAULT_TRACKER_PREFIX = "gs://marin-us-central1/metadata/region_locks/data_curation_fixed_model/"
DEFAULT_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Curation methods. 'all' = every registered method except deferred ones (resiliparse).",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Fixed-model hidden dims to include. Default: all three "
            f"{fixed_model_plan.TARGET_HIDDEN_SIZES}. Use a single value for tiered "
            "rollout (e.g. --hidden-sizes 1536 for 998M only)."
        ),
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=list(curation_plan.BUDGETS),
        help="Compute (FLOPs) budgets to sweep. Default: 7 log-spaced delphi budgets.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="If set, cap the total number of children submitted (smoke testing).",
    )
    parser.add_argument(
        "--filter-name-contains",
        type=str,
        default=None,
        help="If set, only submit plans whose run_name_core contains this substring.",
    )
    parser.add_argument(
        "--filter-name-contains-any",
        nargs="+",
        default=None,
        help="If set, only submit plans whose run_name_core contains AT LEAST ONE of these "
        "substrings. Combined with --filter-name-contains via AND. Use to target a precise "
        "subset of plans for surgical resubmits.",
    )
    parser.add_argument(
        "--force-primary-tpu",
        type=str,
        default=None,
        help="Pin every plan's primary TPU to this shape (e.g. 'v5p-16'). Smoke/debug.",
    )
    parser.add_argument(
        "--force-memory-gb",
        type=int,
        default=None,
        help="Override every plan's per-host CPU memory (e.g. 64). Use when the "
        "default memory tier OOMs at large batch sizes on tiny models.",
    )
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="batch",
    )
    parser.add_argument("--wandb-project", default="marin")
    parser.add_argument("--wandb-entity", default="marin-community")
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument("--tracker-prefix", default=DEFAULT_TRACKER_PREFIX)
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--dry-run", action="store_true", help="Print plan, do not submit.")
    parser.add_argument(
        "--no-keep-alive",
        action="store_true",
        help="Skip the post-submit keep-alive sleep. Useful for testing outside Iris.",
    )
    parser.add_argument(
        "--no-skip-if-done",
        action="store_true",
        help="Disable skip-if-already-done (re-submit completed runs).",
    )
    parser.add_argument(
        "--allowed-regions",
        nargs="+",
        default=None,
        help="HARD-restrict children to a subset of regions. Unset = SOFT preference for all.",
    )
    parser.add_argument("--run-suffix", default="")
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "offline_no_sync"],
        default="auto",
    )
    parser.add_argument(
        "--allow-custom-budgets",
        action="store_true",
        help=(
            "Escape hatch: accept --budgets values that are not in "
            "curation_plan.BUDGETS. Without this flag, non-canonical budgets "
            "cause the launcher to refuse. See _validate_budgets() for why."
        ),
    )
    return parser.parse_args(argv)


def _validate_budgets(budgets: list[float], allow_custom: bool) -> None:
    """Refuse to launch non-canonical budgets unless the user opts in.

    Why: run_name_core uses f"{budget:.0e}" which rounds 1.8e20 and 2.0e20 to
    the same string "2e+20". If a caller accidentally passes --budgets 2e20
    (thinking of the canonical slot), the launcher would previously happily
    produce a run with budget_flops=2.0e20 that SHARES a run_name with the
    canonical 1.8e20 run -- silently divergent, discoverable only by reading
    budget_flops in each summary. We lost ~4 runs of compute to this in the
    original fixed-model sweep; this guard prevents a repeat.

    Validation is exact-float equality against curation_plan.BUDGETS, which is
    acceptable here because the canonical values are Python float literals.
    """
    canonical = set(curation_plan.BUDGETS)
    nonstandard = [b for b in budgets if b not in canonical]
    if not nonstandard:
        return
    # Build a helpful error that lists BOTH the non-standard budgets AND any
    # canonical budgets they'd display-collide with under the current
    # run_name format.
    collisions: list[str] = []
    for b in nonstandard:
        label = f"{b:.0e}"
        clashes = [c for c in canonical if f"{c:.0e}" == label]
        clash_str = f" (display-collides with canonical {clashes})" if clashes else ""
        collisions.append(f"  {b:.3e} -> run_name label {label!r}{clash_str}")
    msg = (
        "Refusing to launch: --budgets contains values NOT in curation_plan.BUDGETS "
        f"{sorted(canonical)}:\n"
        + "\n".join(collisions)
        + "\n\nIf this is intentional (e.g. a one-off experiment), pass "
        "--allow-custom-budgets. Strongly recommend pairing with --run-suffix "
        "to avoid output-path collisions with any canonical run."
    )
    if not allow_custom:
        raise SystemExit(msg)
    logger.warning("NON-CANONICAL BUDGETS (--allow-custom-budgets passed):\n%s", msg)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    methods = fixed_model_plan.resolve_methods(args.methods)
    hidden_sizes = fixed_model_plan.resolve_hidden_sizes(args.hidden_sizes)
    _validate_budgets(args.budgets, allow_custom=args.allow_custom_budgets)
    plans = fixed_model_plan.enumerate_fixed_model_plans(
        methods,
        hidden_sizes=hidden_sizes,
        budgets=tuple(args.budgets),
    )
    if args.filter_name_contains is not None:
        plans = [p for p in plans if args.filter_name_contains in p.run_name_core]
    if args.filter_name_contains_any:
        plans = [p for p in plans if any(s in p.run_name_core for s in args.filter_name_contains_any)]
    if args.force_memory_gb is not None:
        import dataclasses
        plans = [dataclasses.replace(p, memory_gb=args.force_memory_gb) for p in plans]
    if args.max_count is not None:
        plans = plans[: args.max_count]

    if args.dry_run:
        curation_plan.print_dry_run(plans)
        return

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set -- this coordinator must run inside an Iris job.")
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")  # optional

    child_priority_band = PRIORITY_BAND_MAP[args.child_priority]

    logger.info(
        "Submitting %d children (methods=%s, hidden_sizes=%s, priority=%s)...",
        len(plans),
        [m.name for m in methods],
        hidden_sizes,
        args.child_priority,
    )
    submitted, skipped = submit_all(
        client,
        plans,
        tracker_prefix=args.tracker_prefix,
        skip_if_done=not args.no_skip_if_done,
        child_priority_band=child_priority_band,
        wandb_api_key=wandb_api_key,
        hf_token=hf_token,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        allowed_regions=args.allowed_regions,
        run_suffix=args.run_suffix,
        wandb_mode=args.wandb_mode,
        force_primary_tpu=args.force_primary_tpu,
        results_prefix=args.results_prefix,
    )
    logger.info(
        "Result: %d submitted, %d skipped (already done), %d failed",
        len(submitted),
        len(skipped),
        len(plans) - len(submitted) - len(skipped),
    )

    if args.no_keep_alive:
        return

    logger.info("Coordinator entering keep-alive (sleep 3600 forever)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
