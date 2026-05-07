# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator for the WARC-scaling data-curation sweep.

Iris CPU-only parent that enumerates the WARC-scaling grid (see
`warc_scaling_plan.enumerate_warc_scaling_plans`) and submits each plan as an
independent Iris TPU child running `run_curation_train_standalone.py`.

Mirrors `launch_fixed_model_sweep.py` in shape but:
  - swaps the enumerator for `enumerate_warc_scaling_plans` (per-N model and
    budget grids)
  - uses --n-warcs subsets for staged rollout (e.g. priority N=100/2000 first)
  - distinct `--wandb-group`, `--tracker-prefix`, `--results-prefix` so outputs
    don't collide with the fixed-model sweep

Typical staged rollout from an Iris parent (us-central1, CPU-only):

    # Priority: N=100 + N=2000 first.
    iris --cluster marin job run --priority production --no-wait \\
        --memory 4GB --cpu 4 --job-name warc-scaling-priority \\
        -- python experiments/scaling_law_sweeps/launch_warc_scaling_sweep.py \\
        --methods all --n-warcs 100 2000 --child-priority batch

    # Mid-N: N=500 + N=1000.
    iris ... -- python ...launch_warc_scaling_sweep.py \\
        --methods all --n-warcs 500 1000 --child-priority batch
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from experiments.scaling_law_sweeps import curation_plan, warc_scaling_plan
from experiments.scaling_law_sweeps.launch_curation_sweep import (
    PRIORITY_BAND_MAP,
    submit_all,
)
from iris.client.client import IrisClient

logger = logging.getLogger(__name__)

# Distinct WandB group, tracker prefix, and results prefix so WARC-scaling runs
# land in their own namespaces and don't mingle with the fixed-model sweep's
# outputs. All three live under gs://marin-us-central1 for zero cross-region
# reads during analysis / region-lock lookup.
DEFAULT_WANDB_GROUP = "data-curation-warc-scaling"
DEFAULT_TRACKER_PREFIX = "gs://marin-us-central1/metadata/region_locks/data_curation_warc_scaling/"
DEFAULT_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_warc_scaling_results/"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help=f"Curation method base names. 'all' = {warc_scaling_plan.WARC_METHOD_BASE_NAMES}.",
    )
    parser.add_argument(
        "--n-warcs",
        nargs="+",
        type=int,
        default=list(warc_scaling_plan.WARC_COUNTS),
        help=f"WARC counts to sweep. Default: all of {warc_scaling_plan.WARC_COUNTS}. "
        "Use a subset for staged rollout (e.g. --n-warcs 100 2000 for priority).",
    )
    parser.add_argument(
        "--extension-methods",
        nargs="*",
        default=[],
        help="Methods that should get the high-end 2-point budget extension. "
        "Use for resiliparse / llm_curated to give data-rich methods a fair shot "
        "past the DCLM/Nemotron cliff. Other methods stick to the base budget grid.",
    )
    parser.add_argument(
        "--extension-only",
        action="store_true",
        help="Only submit the extension budgets (not the base grid). Pair with "
        "--extension-methods + --child-priority interactive to race just the "
        "high-end plans against capacity contention without doubling base submissions.",
    )
    parser.add_argument(
        "--only-budgets",
        nargs="*",
        type=float,
        default=None,
        help="Restrict to plans matching these specific budgets (e.g. 3e20 9e20). "
        "Combined with --only-hidden-sizes, lets you target a precise subset.",
    )
    parser.add_argument(
        "--only-hidden-sizes",
        nargs="*",
        type=int,
        default=None,
        help="Restrict to plans matching these hidden_sizes (e.g. 1280 2048).",
    )
    parser.add_argument(
        "--batch-divisor",
        type=int,
        default=1,
        help="Divide each plan's batch_size by this factor and multiply train_steps "
        "by the same factor. Total FLOP budget is preserved. Use to shrink the TPU "
        "footprint of high-budget plans (e.g. v5p-256 → v5p-32 with divisor=2). "
        "HP recipe is re-derived for the smaller batch.",
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
        help="Pin every plan's primary TPU to this shape (e.g. 'v5p-8'). Smoke/debug.",
    )
    parser.add_argument(
        "--force-memory-gb",
        type=int,
        default=None,
        help="Override every plan's per-host CPU memory (e.g. 48). Use when the "
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    plans = warc_scaling_plan.enumerate_warc_scaling_plans(
        method_base_names=args.methods,
        n_warcs_list=tuple(args.n_warcs),
        extension_methods=tuple(args.extension_methods),
        extension_only=args.extension_only,
        only_budgets=tuple(args.only_budgets) if args.only_budgets else None,
        only_hidden_sizes=tuple(args.only_hidden_sizes) if args.only_hidden_sizes else None,
        batch_divisor=args.batch_divisor,
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
        "Submitting %d children (methods=%s, n_warcs=%s, priority=%s)...",
        len(plans),
        args.methods,
        args.n_warcs,
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
