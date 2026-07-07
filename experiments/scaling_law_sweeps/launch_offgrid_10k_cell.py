# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off coordinator: launch off-grid (width, budget, batch_divisor) cells for the 10k methods.

The frozen 10k-natural grid (`launch_10k_natural.py`) fixes one batch per cell and
caps each width's budget ladder. This launcher reruns/extends ARBITRARY cells for
the 10k methods -- a single (hidden_size, budget) at a chosen batch_divisor --
through the IDENTICAL `submit_all` path (same tracker/results prefix, wandb group,
TPU selection), so the runs land in the same sweep namespace and the
viewer/audit/repair/plot tooling sees them automatically.

Uses:
  - Batch sensitivity rerun (e.g. d512/2e19 at B64 to test an off-trend point):
        --hidden-size 512 --budget 2e19 --batch-divisor 2
  - Off-grid high-end extension (e.g. d1536/1.8e21 to match the N=3000 push,
    which ran B2048/div=1 in us-east5):
        --hidden-size 1536 --budget 1.8e21 --batch-divisor 1 \\
        --child-priority batch --allowed-regions us-east5

`batch_divisor` re-derives lr/adam_lr/beta2/warmup/steps via the same AdamH
formulas and embeds the resulting batch in the run_name, so cells never collide
across batch choices. Run inside an Iris job (IRIS_CONTROLLER_ADDRESS +
WANDB_API_KEY [+ HF_TOKEN]).
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
    p.add_argument("--hidden-size", type=int, required=True, help="Model width (e.g. 512, 1536).")
    p.add_argument("--budget", type=float, required=True, help="FLOPs budget (e.g. 2e19, 1.8e21).")
    p.add_argument("--batch-divisor", type=int, default=1, help="Shrink natural batch by this factor (>=1).")
    p.add_argument("--methods", nargs="+", default=list(METHOD_NAMES), choices=list(METHOD_NAMES))
    p.add_argument("--child-priority", choices=["production", "interactive", "batch", "unspecified"], default="batch")
    p.add_argument(
        "--allowed-regions",
        nargs="+",
        default=None,
        help="HARD-restrict children to these regions. high_quality_10k lacks us-central2.",
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

    methods = [curation_plan.METHODS[n] for n in args.methods]
    plans = fixed_model_plan.enumerate_fixed_model_plans(
        methods, hidden_sizes=(args.hidden_size,), budgets=(args.budget,), batch_divisor=args.batch_divisor
    )

    if args.dry_run:
        curation_plan.print_dry_run(plans)
        logger.info(
            "off-grid d%d/%.2g (div=%d): %d cells across methods=%s",
            args.hidden_size,
            args.budget,
            args.batch_divisor,
            len(plans),
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

    logger.info("Submitting %d off-grid children (methods=%s)...", len(plans), args.methods)
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
        run_suffix="",
        wandb_mode=args.wandb_mode,
        force_primary_tpu=None,
        results_prefix=args.results_prefix,
    )
    logger.info(
        "Result: %d submitted, %d skipped, %d failed",
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
