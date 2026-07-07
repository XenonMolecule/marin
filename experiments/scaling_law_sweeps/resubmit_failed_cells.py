# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Targeted resubmit of specific 10k-natural sweep cells that hit a terminal FAILED.

The frozen launch_10k_natural coordinators are submit-once + keep-alive (no
resubmit loop), so a child that exhausts its iris restart budget (e.g. a transient
HuggingFace-fetch failure during a gang-schedule burst) becomes a dead gap that
never self-revives. Relaunching the whole method coordinator (-r8) while -r7 is
alive would DUPLICATE the still-running cells (concurrent checkpoint writers), so
instead we resubmit ONLY the named cells here. Each resumes from its persisted
rolling checkpoint (delete_old_temp_checkpoints=False), so no progress is lost.

Targets are (method, run_name_core substring, priority) — float regions (normal
sweep hardware, NOT v6e). Edit TARGETS and relaunch to resubmit a different set.
"""

from __future__ import annotations

import logging
import os
import time

from iris.client.client import IrisClient

from experiments.scaling_law_sweeps.launch_10k_natural import (
    DEFAULT_RESULTS_PREFIX,
    DEFAULT_TRACKER_PREFIX,
    DEFAULT_WANDB_GROUP,
    enumerate_10k_natural_plans,
)
from experiments.scaling_law_sweeps.launch_curation_sweep import PRIORITY_BAND_MAP, submit_one

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# (method, run_name_core substring uniquely identifying the cell, priority band)
TARGETS = [
    ("nemotron_10k", "3e+20-d1536-L16-B256", "batch"),
    ("fineweb_edu_10k", "2e+20-d1024-L11-B512", "batch"),
    ("fineweb_cc_10k", "3e+20-d1024-L11-B512", "interactive"),
]


def main() -> None:
    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set -- run inside an iris job.")
    client = IrisClient.remote(controller_address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")

    for method, substr, prio in TARGETS:
        plans = [p for p in enumerate_10k_natural_plans((method,)) if substr in p.run_name_core]
        if len(plans) != 1:
            raise SystemExit(f"Expected exactly 1 plan for {method} {substr!r}, got {len(plans)}: {[p.run_name_core for p in plans]}")
        plan = plans[0]
        jid = submit_one(
            client,
            plan,
            child_priority_band=PRIORITY_BAND_MAP[prio],
            wandb_api_key=wandb_api_key,
            hf_token=hf_token,
            wandb_project="marin",
            wandb_entity="marin-community",
            wandb_group=DEFAULT_WANDB_GROUP,
            tracker_prefix=DEFAULT_TRACKER_PREFIX,
            allowed_regions=None,  # float (normal sweep hardware)
            results_prefix=DEFAULT_RESULTS_PREFIX,
        )
        logger.info("resubmitted %s (%s) -> %s", plan.run_name_core, prio, jid)

    logger.info("Coordinator entering keep-alive (kill to stop the resubmitted children)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
