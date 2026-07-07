# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off coordinator: rerun the 157M (d512) / 1.8e19-FLOP cell at batch 64.

The frozen 10k-natural grid runs the d512/1.8e19 cell only at its natural batch
(B128). That point came out off-trend (the "1.8e19 is weird" cell). This
coordinator reruns the SAME cell for every 10k method at `batch_divisor=2`
(B64), which re-derives lr/adam_lr/beta2/warmup/steps via the same AdamH formulas.
The budget MUST be the true ladder rung 1.8e19 (run_name prints it as "2e+19" via
%.0e) -- using 2e19 puts the rerun ~11% off-ladder vs every other cell. The -B64
run_name differs only in batch, so these supersede the B128 points at the same
budget. Comparing B64/B128 isolates whether the off-trend point is a
batch/optimization artifact vs. real.

Submits through the identical `submit_all` path as `launch_10k_natural.py`
(same tracker/results prefixes, wandb group, TPU selection), so the runs land in
the same sweep namespace and the viewer/audit/plot tooling sees them.

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

HIDDEN_SIZE = 512  # 156.5M params
BUDGET = 1.8e19  # the true ladder rung for this cell (run_name prints it as "2e+19" via %.0e)
BATCH_DIVISOR = 2  # B128 -> B64


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", nargs="+", default=list(METHOD_NAMES), choices=list(METHOD_NAMES))
    p.add_argument("--child-priority", choices=["production", "interactive", "batch", "unspecified"], default="batch")
    p.add_argument(
        "--allowed-regions",
        nargs="+",
        default=None,
        help="HARD-restrict children to these regions. high_quality_10k lacks us-central2, so when it is "
        "included pass `us-central1 us-east1 us-east5 us-west4 europe-west4`.",
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
        methods, hidden_sizes=(HIDDEN_SIZE,), budgets=(BUDGET,), batch_divisor=BATCH_DIVISOR
    )

    if args.dry_run:
        curation_plan.print_dry_run(plans)
        logger.info("d512/2e19 B64: %d cells across methods=%s", len(plans), args.methods)
        return

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set -- this coordinator must run inside an Iris job.")
    client = IrisClient.remote(controller_address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")

    logger.info("Submitting %d d512/2e19-B64 children (methods=%s)...", len(plans), args.methods)
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
