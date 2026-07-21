# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the two d1536 high-budget cells the frozen 10k grid excludes, for the
CORE-v2-decontaminated dclm/nemotron methods — so the decon sweep matches the
non-decon baselines' ACTUAL finished grid (44 cells/method, not the frozen 38).

The frozen ``launch_10k_natural.py`` restricts the 9e20 / 1.8e21 budget extension
to the big widths (2432, 3584). But the non-decon dclm_10k / nemotron_10k baselines
ALSO have finished runs at d1536 for both budgets (legacy points outside the frozen
grid). These reproduce those exact cells for the decon methods:

  - 9e20  d1536  batch_divisor=1  -> B1024 (v5p-256)
  - 1.8e21 d1536 batch_divisor=2  -> B1024 (v5p-64)

Everything else (regions, HP recipe, tracker/results prefixes, wandb group) is
identical to launch_10k_natural, so the extra cells slot into the same analysis.
pin_region="us-east5" on the decon methods forces training to the mirrored cache.

USAGE (CPU coordinator on Iris; parent interactive, children batch)::

    iris --cluster marin job run --priority interactive --no-wait \\
        --region us-east5 --memory 8GB --cpu 2 --extra cpu --enable-extra-resources \\
        --job-name 10k-decon-extras-coord \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/launch_10k_decon_extras.py

    # dry-run locally to inspect the cells:
    python experiments/scaling_law_sweeps/launch_10k_decon_extras.py --dry-run
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
)
from experiments.scaling_law_sweeps.launch_curation_sweep import PRIORITY_BAND_MAP, submit_all

logger = logging.getLogger(__name__)

METHOD_NAMES: tuple[str, ...] = ("dclm_10k_decon", "nemotron_10k_decon")

# (budget, batch_divisor) cells the frozen grid omits at d1536 but the non-decon
# baselines actually ran. Divisors chosen to reproduce the non-decon B1024 runs.
D1536_EXTRA_CELLS: tuple[tuple[float, int], ...] = (
    (9e20, 1),
    (1.8e21, 2),
)
HIDDEN_DIM = 1536


def enumerate_extra_plans(method_names: tuple[str, ...] = METHOD_NAMES) -> list:
    methods = [curation_plan.METHODS[n] for n in method_names]
    plans: list = []
    for budget, divisor in D1536_EXTRA_CELLS:
        plans += fixed_model_plan.enumerate_fixed_model_plans(
            methods, hidden_sizes=(HIDDEN_DIM,), budgets=(budget,), batch_divisor=divisor
        )
    return plans


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", nargs="+", default=list(METHOD_NAMES), choices=list(METHOD_NAMES))
    p.add_argument("--child-priority", choices=["production", "interactive", "batch", "unspecified"], default="batch")
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

    plans = enumerate_extra_plans(tuple(args.methods))

    if args.dry_run:
        curation_plan.print_dry_run(plans)
        logger.info("10k-decon-extras: %d cells across methods=%s", len(plans), args.methods)
        return

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set -- this coordinator must run inside an Iris job.")
    client = IrisClient.remote(controller_address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")

    logger.info("Submitting %d 10k-decon-extra children (methods=%s)...", len(plans), args.methods)
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
        allowed_regions=None,  # pin_region="us-east5" on the methods constrains children
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
