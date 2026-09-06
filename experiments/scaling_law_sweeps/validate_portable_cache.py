# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off validation: run a SINGLE multi-host cell with LEVANTER_PORTABLE_TPU_CACHE=1.

Purpose: the slice-portable TPU compile cache (strips the device assignment from
the JAX persistent-cache key) is documented correct only for single-host
data-parallel. We need to confirm it's also correct for MULTI-host data-parallel
before enabling it fleet-wide -- a wrong-topology cache hit would silently corrupt
training, which is worse than the recompile churn it cures.

Test design: submit ONE multi-host cell (4-host v5p-16: high_quality d1536-9e19)
with the flag and an isolated run-suffix so it never touches the real sweep cell.
On preemptible TPU it will take a GCP preemption within a few hours, reschedule
onto a different physical slice, and (with the flag) HIT the portable cache.
VERDICT = the train/eval loss is CONTINUOUS across that resume (no jump / NaN /
divergence) AND the resume is fast (~minutes, cache hit, not ~76min recompile).

Run as a CPU iris coordinator job, same env as launch_10k_natural.
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
from experiments.scaling_law_sweeps.launch_curation_sweep import PRIORITY_BAND_MAP, submit_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# 4-host v5p-16 multi-host cell -- smallest multi-host shape => fastest verdict,
# still exercises the cross-host collective compilation the flag could break.
TARGET_SUBSTR = "9e+19-d1536"
METHOD = "high_quality_10k"
RUN_SUFFIX = "pcache-v1"  # isolates checkpoints/results from the real sweep cell


def main() -> None:
    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set -- run inside an iris job.")
    client = IrisClient.remote(controller_address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")

    plans = enumerate_10k_natural_plans((METHOD,))
    matches = [p for p in plans if TARGET_SUBSTR in p.run_name_core]
    if not matches:
        raise SystemExit(f"No plan matching {TARGET_SUBSTR!r} in {METHOD}")
    plan = matches[0]
    logger.info(
        "PORTABLE-CACHE VALIDATION cell=%s suffix=%s flag=LEVANTER_PORTABLE_TPU_CACHE=1", plan.run_name_core, RUN_SUFFIX
    )

    submitted, skipped = submit_all(
        client,
        [plan],
        tracker_prefix=DEFAULT_TRACKER_PREFIX,
        skip_if_done=False,  # force it to run even though the base cell is done
        child_priority_band=PRIORITY_BAND_MAP["interactive"],
        wandb_api_key=wandb_api_key,
        hf_token=hf_token,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group=DEFAULT_WANDB_GROUP,
        allowed_regions=["us-central1", "us-east1", "us-east5", "us-west4", "europe-west4"],
        run_suffix=RUN_SUFFIX,
        wandb_mode="auto",
        force_primary_tpu=None,
        results_prefix=DEFAULT_RESULTS_PREFIX,
        extra_env={"LEVANTER_PORTABLE_TPU_CACHE": "1"},
    )
    logger.info("submitted=%s skipped=%s", submitted, skipped)

    # Keep-alive: the child is nested under this coordinator, so if we exit iris
    # orphan-kills it. Stay alive through the preemption-resume the test needs.
    logger.info("Coordinator entering keep-alive (kill this job to stop the test)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
