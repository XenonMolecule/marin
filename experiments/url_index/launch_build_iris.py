# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris coordinator: one CPU job per URL-index build target, pinned to its region.

Each child runs :mod:`experiments.url_index.build` for one (dataset, collection),
region-constrained so every read/write stays in-region. By default it fans out
over the whole registry at BOTH scales (SMALL=random-300 and FULL=10,364) --
resolution simply fails a target whose tier hasn't landed yet, so ``--only-landed``
ships everything currently available.

Launch as an Iris job so ``IRIS_CONTROLLER_ADDRESS`` is set::

    iris job run --cluster marin --region us-central1 -- \
        python -m experiments.url_index.launch_build_iris --only-landed
"""

import argparse
import logging
import os
import time

from iris.client.client import IrisClient
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2

from experiments.infinigram.targets import Collection, IndexTarget, all_targets, get_target

logger = logging.getLogger(__name__)

JOB_WORK_ROOT = "/tmp/url_index"

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}

# FULL (10k) text stores can reach tens of GiB before compression; SMALL (300) is ~1/34.
RESOURCES: dict[Collection, dict[str, object]] = {
    Collection.FULL: {"cpu": 16.0, "memory": "64g", "disk": "256g"},
    Collection.SMALL: {"cpu": 8.0, "memory": "32g", "disk": "64g"},
}


def _submit_one(client: IrisClient, target: IndexTarget, *, priority_band: int, overwrite: bool) -> str:
    cmd = [
        "python",
        "-m",
        "experiments.url_index.build",
        "--dataset",
        target.dataset,
        "--collection",
        target.collection.value,
        "--local-root",
        JOB_WORK_ROOT,
    ]
    if overwrite:
        cmd.append("--overwrite")

    res = RESOURCES[target.collection]
    constraints = [
        preemptible_constraint(True),
        Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value=target.region),
    ]
    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd),
        name=f"urlindex-{target.name}",
        resources=ResourceSpec(cpu=res["cpu"], memory=res["memory"], disk=res["disk"]),
        environment=EnvironmentSpec(extras=["cpu"], env_vars={"PYTHONUNBUFFERED": "1"}),
        constraints=constraints,
        max_retries_preemption=10,
        max_retries_failure=1,
        priority_band=priority_band,
    )
    return str(job.job_id)


def _select_targets(args: argparse.Namespace) -> list[IndexTarget]:
    collections = [Collection(c) for c in args.collections]
    if args.datasets is None:
        return [t for t in all_targets(only_landed=args.only_landed) if t.collection in collections]
    out: list[IndexTarget] = []
    for ds in args.datasets:
        for col in collections:
            target = get_target(ds, col)
            if args.only_landed and not target.source.landed:
                continue
            out.append(target)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit URL-index build jobs, one per (dataset, collection).")
    p.add_argument("--datasets", nargs="+", default=None, help="Datasets to build (default: whole registry).")
    p.add_argument(
        "--collections",
        nargs="+",
        choices=[c.value for c in Collection],
        default=[c.value for c in Collection],
    )
    p.add_argument("--only-landed", action="store_true", help="Skip targets whose source isn't landed yet.")
    p.add_argument("--priority", choices=sorted(PRIORITY_BAND_MAP), default="batch")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    targets = _select_targets(args)

    if args.dry_run:
        for t in targets:
            logger.info("DRY: would submit %s (region=%s, landed=%s)", t.name, t.region, t.source.landed)
        logger.info("DRY: %d targets", len(targets))
        return

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — launch via `iris job run -- python <this>`")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))
    band = PRIORITY_BAND_MAP[args.priority]

    submitted: list[tuple[str, str]] = []
    for t in targets:
        try:
            jid = _submit_one(client, t, priority_band=band, overwrite=args.overwrite)
            submitted.append((t.name, jid))
            logger.info("submitted %s -> %s", t.name, jid)
        except Exception:
            logger.exception("failed to submit %s", t.name)

    logger.info("Submitted %d/%d targets", len(submitted), len(targets))
    logger.info("Coordinator entering keep-alive...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
