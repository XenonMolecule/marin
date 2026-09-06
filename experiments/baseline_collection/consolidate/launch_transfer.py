# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris coordinator: submit one transfer job per region.

Each child is a region-pinned CPU job that runs ``gcloud storage rsync`` from
its local bucket into ``gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/{region}/``.

us-central1 is included so that its own extraction output is mirrored into the
consolidated layout uniformly (intra-region GCS copy is free). us-central2 is
skipped (no extraction output there).

This is destructive-safe by construction: no ``--delete-*`` flags, no
overwrites of existing targets (rsync short-circuits on same name + size), and
the per-region source trees are NEVER touched.

Typical invocation::

    iris --cluster marin job run --priority production --no-wait \\
        --memory 2GB --cpu 2 --job-name extract-transfer-coordinator \\
        -- python experiments/baseline_collection/consolidate/launch_transfer.py

Add ``--dry-run`` to submit children that only call ``gcloud rsync --dry-run``.
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

SCRIPT = "experiments/baseline_collection/consolidate/transfer_region.py"
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
    cpu: float,
    memory: str,
    disk: str,
    max_workers: int,
    dry_run_child: bool,
    spec: str,
    child_preemptible: bool,
) -> str:
    cmd = ["python", SCRIPT, "--region", region, "--spec", spec, "--max-workers", str(max_workers)]
    if dry_run_child:
        cmd.append("--dry-run")
    # Constraint.create wraps raw strings; bare Constraint(values=(...)) regressed.
    # A big region's transfer needs hours of UNINTERRUPTED work: a preempted child
    # restarts from scratch and must re-list/re-skip everything already copied
    # (~1h for europe-west4's 3.9M objects), so it nets only ~1h of real progress
    # per preemption cycle. Pass child_preemptible=False for those.
    constraints = [
        preemptible_constraint(child_preemptible),
        Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=(region,)),
    ]
    suffix = "" if spec == "low_quality" else f"-{spec}"
    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd),
        name=f"extract-transfer-{region}{suffix}",
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
    p.add_argument("--spec", default="low_quality", help="Extraction spec to transfer.")
    p.add_argument("--priority", choices=sorted(PRIORITY_BAND_MAP), default="batch")
    p.add_argument("--cpu", type=float, default=2.0, help="Per-child CPU (GCS IO is the bottleneck).")
    p.add_argument("--memory", default="8GB")
    p.add_argument("--disk", default="20GB")
    p.add_argument(
        "--child-preemptible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run children on preemptible workers (default). Use --no-child-preemptible for "
        "large regions where a restart costs an hour of re-listing.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=128,
        help="Per-child thread pool size for concurrent GCS server-side copies.",
    )
    p.add_argument("--dry-run", action="store_true", help="Coordinator prints plan only; does not submit.")
    p.add_argument(
        "--dry-run-child",
        action="store_true",
        help="Submit children that run `gcloud rsync --dry-run` (safe to preview).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()

    if args.dry_run:
        for r in args.regions:
            logger.info(
                "DRY: would submit transfer for region=%s spec=%s (dry_run_child=%s)",
                r,
                args.spec,
                args.dry_run_child,
            )
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
                cpu=args.cpu,
                memory=args.memory,
                disk=args.disk,
                max_workers=args.max_workers,
                dry_run_child=args.dry_run_child,
                spec=args.spec,
                child_preemptible=args.child_preemptible,
            )
            submitted.append((region, jid))
            logger.info("submitted %s -> %s", region, jid)
        except Exception:
            logger.exception("failed to submit %s", region)

    logger.info("Total submitted: %d/%d", len(submitted), len(args.regions))
    logger.info("Coordinator entering keep-alive (sleep 3600 forever)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
