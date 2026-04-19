# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris launcher: run resolve_duplicates.py inside us-central1.

The resolver reads 5 small inventory manifests + the central _completed/
registry (all in us-central1) and writes resolved.jsonl.gz + duplicates +
integrity_report.json. All intra-region — no egress.

Wraps it as an Iris CPU job to keep the "run on cluster not laptop" habit.

Usage::

    iris --cluster marin job run --priority production --no-wait \\
        --memory 4GB --cpu 2 --job-name extract-resolve-coord \\
        -- python experiments/baseline_collection/consolidate/launch_resolve.py
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

SCRIPT = "experiments/baseline_collection/consolidate/resolve_duplicates.py"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="us-central1", help="Hard region constraint (resolver data lives here).")
    p.add_argument("--priority", choices=["production", "interactive", "batch"], default="batch")
    p.add_argument("--cpu", type=float, default=2.0)
    p.add_argument("--memory", default="8GB")
    p.add_argument("--disk", default="10GB")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — launch via iris CLI.")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    band = {
        "production": job_pb2.PRIORITY_BAND_PRODUCTION,
        "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
        "batch": job_pb2.PRIORITY_BAND_BATCH,
    }[args.priority]
    constraints = [
        preemptible_constraint(True),
        Constraint(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=(args.region,)),
    ]
    job = client.submit(
        entrypoint=Entrypoint.from_command("python", SCRIPT),
        name=f"extract-resolve-{args.region}",
        resources=ResourceSpec(cpu=args.cpu, memory=args.memory, disk=args.disk),
        environment=EnvironmentSpec(extras=["cpu"], env_vars={"PYTHONUNBUFFERED": "1"}),
        constraints=constraints,
        max_retries_preemption=5,
        max_retries_failure=3,
        priority_band=band,
    )
    logger.info("submitted resolve -> %s", job.job_id)
    logger.info("Coordinator entering keep-alive (sleep 3600 forever)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
