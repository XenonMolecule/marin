# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris coordinator: submit one inventory job per region.

Each child is a region-pinned CPU job running ``inventory_region.py`` on its
local bucket. All children write their per-region manifest directly to
us-central1 so the downstream resolver reads from a single location.

us-central2 is skipped (no extraction output there). us-central1 is included
so that the consolidated inventory covers every region uniformly.

Typical invocation (from laptop or from inside an Iris parent)::

    iris --cluster marin job run --priority production --no-wait \\
        --memory 2GB --cpu 2 --job-name extract-inventory-coordinator \\
        -- python experiments/baseline_collection/consolidate/launch_inventory.py

Add ``--dry-run`` to see what it would submit without hitting the controller.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from iris.client.client import IrisClient
from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    preemptible_constraint,
)
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

SCRIPT = "experiments/baseline_collection/consolidate/inventory_region.py"

# Region labels Iris understands. "europe-west4" is the full label; the bucket
# is "marin-eu-west4" — the child script handles both. us-central2 is skipped
# (no extraction output there).
REGIONS: tuple[str, ...] = ("us-central1", "us-east1", "us-east5", "us-west4", "europe-west4")

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}


def submit_one(
    client: IrisClient,
    region: str,
    *,
    priority_band: int,
    max_workers: int,
    cpu: float,
    memory: str,
    disk: str,
) -> str:
    cmd = [
        "python",
        SCRIPT,
        "--region",
        region,
        "--max-workers",
        str(max_workers),
    ]
    constraints = [
        preemptible_constraint(True),
        Constraint(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=(region,)),
    ]
    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd),
        name=f"extract-inventory-{region}",
        resources=ResourceSpec(cpu=cpu, memory=memory, disk=disk),
        environment=EnvironmentSpec(extras=["cpu"], env_vars={"PYTHONUNBUFFERED": "1"}),
        constraints=constraints,
        max_retries_preemption=10,
        max_retries_failure=3,
        priority_band=priority_band,
    )
    return str(job.job_id)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
    p.add_argument("--priority", choices=sorted(PRIORITY_BAND_MAP), default="batch")
    p.add_argument("--max-workers", type=int, default=64, help="Per-job decompression threads.")
    p.add_argument("--cpu", type=float, default=4.0)
    p.add_argument("--memory", default="16GB")
    p.add_argument("--disk", default="20GB")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()

    if args.dry_run:
        for r in args.regions:
            logger.info("DRY: would submit inventory for region=%s", r)
        return

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — launch via `iris ... job run -- python <this>`")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    band = PRIORITY_BAND_MAP[args.priority]
    submitted: list[tuple[str, str]] = []
    for region in args.regions:
        try:
            jid = submit_one(
                client,
                region=region,
                priority_band=band,
                max_workers=args.max_workers,
                cpu=args.cpu,
                memory=args.memory,
                disk=args.disk,
            )
            submitted.append((region, jid))
            logger.info("submitted %s -> %s", region, jid)
        except Exception:
            logger.exception("failed to submit %s", region)

    logger.info("Total submitted: %d/%d", len(submitted), len(args.regions))
    # Keep the coordinator alive so Iris keeps the hierarchy around while children run.
    logger.info("Coordinator entering keep-alive (sleep 3600 forever)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
