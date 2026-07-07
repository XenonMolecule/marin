# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off coordinator: re-run a single curation cell at a different RNG seed.

The whole 10k-natural sweep trains every cell at Levanter `TrainerConfig.seed=0`
(model init, loader, AND the feistel data-shuffle permutation all split from
`PRNGKey(0)`), and there is NO per-epoch reshuffle -- a K-epoch run replays the
exact same permuted token order K times. So a single cell's loss is one draw from
a fixed (seed, data-order) pair.

This coordinator re-runs the SAME cell at one or more alternate seeds to measure
seed/data-order noise. Each alternate seed gives a fully independent draw (different
init AND different data permutation). If an off-trend "spike" cell reverts to the
local trend under a new seed, it was unlucky data-order/init; if it persists, the
anomaly is structural to that budget rung.

DEFAULTS target the fineweb_edu d512 / 1.8e19-FLOP / B64 cell -- the batch-independent
Paloma/uncheatable/LIMA spike (~11 epochs of the 2.3B-tok fineweb_edu corpus). Override
--methods / --hidden / --budget / --batch-divisor to probe a different cell.

Each seed run is named with a `seed<N>` run-suffix, so it lands in a fresh run
name / output dir / WandB run / summary.json and never collides with or skips the
original seed-0 cell. Submits through the identical `submit_all` path as
`launch_10k_natural.py` (same tracker/results prefixes, wandb group, TPU selection).

Run inside an Iris job (needs IRIS_CONTROLLER_ADDRESS + WANDB_API_KEY [+ HF_TOKEN]).
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from iris.client.client import IrisClient

from experiments.scaling_law_sweeps import curation_plan, fixed_model_plan
from experiments.scaling_law_sweeps.launch_10k_natural import (
    DEFAULT_RESULTS_PREFIX,
    DEFAULT_TRACKER_PREFIX,
    DEFAULT_WANDB_GROUP,
    METHOD_NAMES,
)
from experiments.scaling_law_sweeps.launch_curation_sweep import PRIORITY_BAND_MAP, submit_all

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", nargs="+", default=["fineweb_edu_10k"], choices=list(METHOD_NAMES))
    p.add_argument("--hidden", type=int, default=512, help="Hidden size of the target cell.")
    p.add_argument(
        "--budget",
        type=float,
        default=1.8e19,
        help="FLOP budget of the target cell (the run_name prints it as e.g. '2e+19' via %%.0e).",
    )
    p.add_argument(
        "--batch-divisor",
        type=int,
        default=2,
        help="Batch divisor applied to the cell's natural batch (2 = B128->B64, the kept fineweb_edu point).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7],
        help="Alternate Levanter seeds to run (must be non-zero; 0 is the original sweep run).",
    )
    p.add_argument("--child-priority", choices=["production", "interactive", "batch", "unspecified"], default="batch")
    p.add_argument(
        "--allowed-regions",
        nargs="+",
        default=None,
        help="HARD-restrict children to these regions. Pass regions where the cell's tokenized cache "
        "exists to avoid cross-region reads (fineweb_edu: us-east5 us-central1 us-central2).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-skip-if-done", action="store_true")
    p.add_argument("--wandb-project", default="marin")
    p.add_argument("--wandb-entity", default="marin-community")
    p.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    p.add_argument("--tracker-prefix", default=DEFAULT_TRACKER_PREFIX)
    p.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    p.add_argument("--wandb-mode", choices=["auto", "online", "offline", "offline_no_sync"], default="auto")
    p.add_argument("--no-keep-alive", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if any(s == 0 for s in args.seeds):
        raise ValueError("--seeds must be non-zero; seed 0 is the original sweep run, nothing to re-run.")

    methods = [curation_plan.METHODS[n] for n in args.methods]
    plans = fixed_model_plan.enumerate_fixed_model_plans(
        methods, hidden_sizes=(args.hidden,), budgets=(args.budget,), batch_divisor=args.batch_divisor
    )

    if args.dry_run:
        curation_plan.print_dry_run(plans)
        logger.info(
            "seed-noise probe: %d cell(s) x %d seed(s)=%s (methods=%s)",
            len(plans),
            len(args.seeds),
            args.seeds,
            args.methods,
        )
        return

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set -- this coordinator must run inside an Iris job.")
    client = IrisClient.remote(controller_address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")

    for seed in args.seeds:
        logger.info("Submitting %d cell(s) at seed=%d (methods=%s)...", len(plans), seed, args.methods)
        submitted, skipped = submit_all(
            client,
            plans,
            tracker_prefix=args.tracker_prefix,
            skip_if_done=not args.no_skip_if_done,
            child_priority_band=PRIORITY_BAND_MAP[args.child_priority],
            wandb_api_key=wandb_api_key,
            hf_token=hf_token,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            allowed_regions=args.allowed_regions,
            run_suffix=f"seed{seed}",
            wandb_mode=args.wandb_mode,
            force_primary_tpu=None,
            results_prefix=args.results_prefix,
            seed=seed,
        )
        logger.info(
            "seed=%d: %d submitted, %d skipped, %d failed",
            seed,
            len(submitted),
            len(skipped),
            len(plans) - len(submitted) - len(skipped),
        )

    if args.no_keep_alive:
        return
    logger.info("Coordinator entering keep-alive...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
