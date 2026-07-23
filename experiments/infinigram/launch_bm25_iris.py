# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris coordinator: submit one CPU job per BM25 backup index target.

The BM25 mirror of :mod:`experiments.infinigram.launch_infinigram_iris`. Because
``bm25s`` is a normal pip package (numpy/scipy only), the child runs the pipeline
directly -- no ``run_in_toolchain.sh`` toolchain bootstrap. Each child is pinned
to its dataset's region so all reads/writes are in-region.

Launch as an Iris job so ``IRIS_CONTROLLER_ADDRESS`` is set:

    iris job run --cluster marin --region us-central2 -- \
        python -m experiments.infinigram.launch_bm25_iris --datasets dclm --collections full
"""

import argparse
import logging
import os
import time

from iris.client.client import IrisClient
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from marin.utils import fsspec_exists, fsspec_glob

from experiments.infinigram.bm25_build import bm25_index_dir
from experiments.infinigram.bm25_query import MANIFEST_NAME
from experiments.infinigram.resolve import resolve_prefix
from experiments.infinigram.targets import Collection, IndexTarget, all_targets, get_target

logger = logging.getLogger(__name__)

# Writable staging root on the job's ephemeral disk (staged corpus + local index).
JOB_WORK_ROOT = "/tmp/bm25"

# bm25s is not in the base image; install it per-job. numpy/scipy come with it.
BM25_PIP_PACKAGES = ["bm25s"]

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}

# Per-collection resources. Every worker node caps at 100 GiB disk (marin.yaml),
# so disk MUST stay under it -- the streaming build bounds disk to one sub-index,
# so even FULL needs little. RAM is set generously (nodes have 192-720 GiB) to
# hold a text_bytes_budget's worth of text + tokens; a 2 GiB budget peaks ~12 GiB.
RESOURCES: dict[Collection, dict[str, object]] = {
    Collection.FULL: {"cpu": 16.0, "memory": "96g", "disk": "80g", "text_bytes_budget": 2 * 1024**3},
    Collection.SMALL: {"cpu": 8.0, "memory": "48g", "disk": "48g", "text_bytes_budget": 1024**3},
}


def _submit_one(client: IrisClient, target: IndexTarget, *, priority_band: int, overwrite: bool) -> str:
    res = RESOURCES[target.collection]
    cmd = [
        "python",
        "-m",
        "experiments.infinigram.bm25_pipeline",
        "--dataset",
        target.dataset,
        "--collection",
        target.collection.value,
        "--local-root",
        f"{JOB_WORK_ROOT}/run",
        "--text-bytes-budget",
        str(res["text_bytes_budget"]),
    ]
    if overwrite:
        cmd.append("--overwrite")

    constraints = [
        preemptible_constraint(True),
        Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value=target.region),
    ]
    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd),
        name=f"bm25-{target.name}",
        resources=ResourceSpec(cpu=res["cpu"], memory=res["memory"], disk=res["disk"]),
        environment=EnvironmentSpec(
            extras=["cpu"],
            pip_packages=BM25_PIP_PACKAGES,
            env_vars={
                "PYTHONUNBUFFERED": "1",
                "BM25_LOCAL_ROOT": f"{JOB_WORK_ROOT}/run",
            },
        ),
        constraints=constraints,
        max_retries_preemption=10,
        max_retries_failure=1,
        priority_band=priority_band,
    )
    return str(job.job_id)


def _is_built(target: IndexTarget) -> bool:
    """True once the target's BM25 index manifest exists in GCS (build finished)."""
    return fsspec_exists(f"{bm25_index_dir(target).rstrip('/')}/{MANIFEST_NAME}")


def _is_landed(target: IndexTarget) -> bool:
    """True if the target's source glob matches >=1 shard now.

    Existence-only (a single list op) -- deliberately does NOT sum shard sizes
    like ``resolve_target`` does, so polling a 10k-shard corpus costs one glob,
    not thousands of stat calls. The child re-resolves fully (with sizes) when it
    actually builds.
    """
    src = target.source
    try:
        if src.prefix is not None:
            return bool(resolve_prefix(src.prefix))
        return any(fsspec_glob(pattern) for pattern in src.globs)
    except Exception as e:
        logger.info("%s not landed yet: %s", target.name, str(e)[:100])
        return False


def _coordinator_loop(
    client: IrisClient,
    targets: list[IndexTarget],
    *,
    band: int,
    overwrite: bool,
    poll_interval: float,
) -> None:
    """Submit each target as soon as its source lands; return when all are built.

    Each target is submitted at most once (when it first resolves). The loop
    exits cleanly once every target has an uploaded manifest, so the coordinator
    only terminates -- and stops parenting children -- when no build is in flight.
    Not-yet-landed targets are re-checked every ``poll_interval`` seconds.
    """
    submitted: dict[str, str] = {}
    built: set[str] = set()
    while True:
        for t in targets:
            if t.name in built:
                continue
            if _is_built(t):
                built.add(t.name)
                logger.info("BUILT %s (%d/%d)", t.name, len(built), len(targets))
                continue
            if t.name in submitted:
                continue  # build in flight; wait for its manifest
            if overwrite or _is_landed(t):
                try:
                    jid = _submit_one(client, t, priority_band=band, overwrite=overwrite)
                    submitted[t.name] = jid
                    logger.info("submitted bm25-%s -> %s", t.name, jid)
                except Exception:
                    logger.exception("failed to submit %s", t.name)

        remaining = [t.name for t in targets if t.name not in built]
        if not remaining:
            logger.info("All %d targets built. Coordinator exiting.", len(targets))
            return
        logger.info(
            "Progress: %d/%d built, %d submitted/in-flight, %d awaiting-landing. Sleeping %.0fs.",
            len(built),
            len(targets),
            len([t for t in targets if t.name in submitted and t.name not in built]),
            len([t for t in targets if t.name not in submitted and t.name not in built]),
            poll_interval,
        )
        time.sleep(poll_interval)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit BM25 backup index build jobs.")
    p.add_argument("--datasets", nargs="+", default=None, help="Datasets to build (default: all in registry).")
    p.add_argument(
        "--collections",
        nargs="+",
        choices=[c.value for c in Collection],
        default=[c.value for c in Collection],
    )
    p.add_argument("--only-landed", action="store_true", help="Skip targets whose source isn't landed yet.")
    p.add_argument(
        "--poll-until-complete",
        action="store_true",
        help="Keep polling and submit each target as its source lands; exit when all are built.",
    )
    p.add_argument("--poll-interval", type=float, default=600.0, help="Seconds between landing re-checks in poll mode.")
    p.add_argument("--priority", choices=sorted(PRIORITY_BAND_MAP), default="batch")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _select_targets(args: argparse.Namespace) -> list[IndexTarget]:
    collections = [Collection(c) for c in args.collections]
    if args.datasets is None:
        return [t for t in all_targets(only_landed=args.only_landed) if t.collection in collections]
    out: list[IndexTarget] = []
    for ds in args.datasets:
        for col in collections:
            target = get_target(ds, col)  # raises on unknown dataset/collection
            if args.only_landed and not target.source.landed:
                continue
            out.append(target)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    targets = _select_targets(args)

    if args.dry_run:
        for t in targets:
            logger.info("DRY: would submit bm25-%s (region=%s, landed=%s)", t.name, t.region, t.source.landed)
        logger.info("DRY: %d targets", len(targets))
        return

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — launch via `iris ... job run -- python <this>`")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))
    band = PRIORITY_BAND_MAP[args.priority]

    if args.poll_until_complete:
        _coordinator_loop(client, targets, band=band, overwrite=args.overwrite, poll_interval=args.poll_interval)
        return

    submitted: list[tuple[str, str]] = []
    for t in targets:
        try:
            jid = _submit_one(client, t, priority_band=band, overwrite=args.overwrite)
            submitted.append((t.name, jid))
            logger.info("submitted bm25-%s -> %s", t.name, jid)
        except Exception:
            logger.exception("failed to submit %s", t.name)

    logger.info("Submitted %d/%d targets", len(submitted), len(targets))

    # Coordinator stays alive so children keep it as parent (submit-and-exit
    # orphan-kills children on this cluster).
    logger.info("Coordinator entering keep-alive...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
