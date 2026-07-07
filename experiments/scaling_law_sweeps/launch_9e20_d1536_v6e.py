# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a NEW 9e20 @ 998M (d1536) isoFLOP point for the 10k-natural methods on v6e-128.

This extends the 10k-natural curve with a high-budget point at the 998M width that
is NOT in the frozen base ladder (the 9e20 extension was big-widths-only, d2432/
d3584). These are genuinely new cells, launched OUTSIDE `launch_10k_natural.py`'s
frozen grid so the six running coordinators are untouched -- the run names
(`curation-<method>_10k-expFM_natural-9e+20-d1536-L16-B1024`) don't exist in the
frozen 228, so there is no collision and skip-if-done is a no-op on first run.

Hardware: forced to **v6e-128** via `force_primary_tpu`. The 9e20-d1536 cell needs
~2.8 TB at its natural batch (1024), which fits v6e-128 (128 chips x 32 GiB = 4 TB)
but not v5e-128 (2 TB). The planner's default v5p-256 is currently unobtainable
(17-224 consecutive provisioning failures); v6e-128 is the attainable slice.

Region: pinned to **us-east5**, the one region that has both a v6e-128 pool AND all
six methods' tokenized caches co-located (incl. resiliparse, whose cache only lives
in us-east5 / us-central2) -- so no cross-region mirror egress. us-east1-d has
better v6e-128 availability (0 recent fails vs us-east5's ~14) but not every cache,
so we trade a bit of capacity contention for zero egress.

Run as a CPU iris coordinator (interactive parent, batch children), same env as
launch_10k_natural. Set LAUNCH_METHODS (comma-separated) to launch a subset; the
default is all six. Children are nested under this coordinator, so it must stay
alive (keep-alive loop) or iris orphan-kills them.
"""

from __future__ import annotations

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

ALL_METHODS = (
    "high_quality_10k",
    "dclm_10k",
    "nemotron_10k",
    "fineweb_cc_10k",
    "fineweb_edu_10k",
    "resiliparse_10k",
)
BUDGET = 9e20
WIDTH = 1536
FORCE_TPU = "v6e-128"
REGIONS = ["us-east5"]


def main() -> None:
    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set -- run inside an iris job.")
    client = IrisClient.remote(controller_address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")

    selected = os.environ.get("LAUNCH_METHODS", "").strip()
    method_names = [m.strip() for m in selected.split(",") if m.strip()] if selected else list(ALL_METHODS)
    methods = [curation_plan.METHODS[n] for n in method_names]

    plans = fixed_model_plan.enumerate_fixed_model_plans(
        methods, hidden_sizes=(WIDTH,), budgets=(BUDGET,), batch_divisor=1
    )
    logger.info(
        "Launching %d cell(s) @ %.0e d%d on %s (region=%s): %s",
        len(plans),
        BUDGET,
        WIDTH,
        FORCE_TPU,
        REGIONS,
        [p.run_name_core for p in plans],
    )

    submitted, skipped = submit_all(
        client,
        plans,
        tracker_prefix=DEFAULT_TRACKER_PREFIX,
        skip_if_done=True,
        child_priority_band=PRIORITY_BAND_MAP["batch"],
        wandb_api_key=wandb_api_key,
        hf_token=hf_token,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group=DEFAULT_WANDB_GROUP,
        allowed_regions=REGIONS,
        wandb_mode="auto",
        force_primary_tpu=FORCE_TPU,
        results_prefix=DEFAULT_RESULTS_PREFIX,
    )
    logger.info("submitted=%s skipped=%s", submitted, skipped)

    logger.info("Coordinator entering keep-alive (kill this job to stop the children)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
