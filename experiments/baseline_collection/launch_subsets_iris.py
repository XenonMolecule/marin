# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris coordinator: submit one subset+tokenize job per (method, n_warcs) pair.

Each child is a region-pinned CPU Iris job that filters the existing baseline
output to the first N WARCs of ``baseline_warcs_3000.txt`` (deterministic
prefix subset) and re-tokenizes with Llama-3.1-8B.

Subset sizes: 100, 500, 1000, 2000.
Methods (region):
    us-central2: dclm, fineweb_edu
    us-central1: nemotron_full, llm_curated

Total children: 4 methods × 4 sizes = 16.

Typical invocation::

    uv run iris --config lib/iris/examples/marin.yaml job run \\
        --priority interactive --memory 4GB --cpu 2 --no-wait \\
        --job-name subset-coord \\
        -- python experiments/baseline_collection/launch_subsets_iris.py
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

SCRIPT = "experiments/baseline_collection/subset_one.py"

SUBSET_SIZES_DEFAULT: tuple[int, ...] = (100, 500, 1000, 2000)

# (method_name, region) — region is the Iris constraint and matches the source bucket region.
METHOD_REGIONS: dict[str, str] = {
    "dclm": "us-central2",
    "fineweb_edu": "us-central2",
    "resiliparse": "us-central2",
    "nemotron_full": "us-central1",
    "nemotron_qhigh": "us-central1",
    "llm_curated": "us-central1",
    "llm_curated_dclm_filtered": "us-central1",
}

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}


def _submit_one(
    client: IrisClient,
    method: str,
    n: int,
    region: str,
    *,
    priority_band: int,
    cpu: float,
    memory: str,
    disk: str,
    skip_filter: bool = False,
) -> str:
    cmd = ["python", SCRIPT, "--method", method, "--n", str(n)]
    if skip_filter:
        cmd.append("--skip-filter")
    constraints = [
        preemptible_constraint(True),
        Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value=region),
    ]
    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd),
        name=f"subset-{method}-{n}warcs",
        resources=ResourceSpec(cpu=cpu, memory=memory, disk=disk),
        environment=EnvironmentSpec(
            extras=["cpu"],
            env_vars={"PYTHONUNBUFFERED": "1"},
        ),
        constraints=constraints,
        max_retries_preemption=10,
        max_retries_failure=2,
        priority_band=priority_band,
    )
    return str(job.job_id)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_REGIONS.keys()),
        choices=list(METHOD_REGIONS.keys()),
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(SUBSET_SIZES_DEFAULT),
    )
    p.add_argument(
        "--priority",
        choices=sorted(PRIORITY_BAND_MAP),
        default="batch",
        help="Priority band for child jobs. Coordinator should run at higher priority via the launching iris flag.",
    )
    p.add_argument("--cpu", type=float, default=8.0, help="CPU per child job (driver).")
    p.add_argument("--memory", default="64GB", help="Memory per child job (driver).")
    p.add_argument("--disk", default="50GB", help="Disk per child job (driver).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-filter",
        action="store_true",
        help="Pass --skip-filter to each child (tokenize-only, assumes filter output exists).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()

    pairs: list[tuple[str, int, str]] = []
    for method in args.methods:
        region = METHOD_REGIONS[method]
        for n in args.sizes:
            pairs.append((method, n, region))

    if args.dry_run:
        for m, n, r in pairs:
            logger.info("DRY: would submit method=%s n=%d region=%s", m, n, r)
        logger.info("DRY: total %d children", len(pairs))
        return

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — launch via `iris ... job run -- python <this>`")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    band = PRIORITY_BAND_MAP[args.priority]
    submitted: list[tuple[str, int, str, str]] = []
    failed: list[tuple[str, int, str, str]] = []

    for method, n, region in pairs:
        try:
            jid = _submit_one(
                client,
                method=method,
                n=n,
                region=region,
                priority_band=band,
                cpu=args.cpu,
                memory=args.memory,
                disk=args.disk,
                skip_filter=args.skip_filter,
            )
            submitted.append((method, n, region, jid))
            logger.info("submitted method=%s n=%d region=%s -> %s", method, n, region, jid)
        except Exception:
            logger.exception("failed to submit method=%s n=%d", method, n)
            failed.append((method, n, region, ""))

    logger.info("Submission summary: %d ok / %d failed (of %d total)", len(submitted), len(failed), len(pairs))
    for method, n, region, jid in submitted:
        logger.info("  OK   %-14s %4d  region=%-12s jid=%s", method, n, region, jid)
    for method, n, region, _ in failed:
        logger.info("  FAIL %-14s %4d  region=%s", method, n, region)

    # Coordinator stays alive so the children inherit it as parent. Children
    # are resilient on their own (they retry preemption), so this is mostly
    # cosmetic for the dashboard hierarchy.
    logger.info("Coordinator entering keep-alive...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
