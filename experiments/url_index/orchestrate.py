# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end in-cluster orchestrator: build every dataset, then analyze.

Runs as a single Iris job (launched via ``iris job run``). It:

1. submits one region-pinned :mod:`~experiments.url_index.build` child per
   (dataset, collection) at BOTH scales, tolerating the ones whose tier hasn't
   landed (their child just fails and is skipped);
2. waits for the children;
3. for each collection, consolidates the succeeded datasets' ``meta.parquet``
   into a lookup DuckDB and computes pairwise coverage (``rid_h`` and ``text_h``);
4. uploads the lookup DBs + coverage CSVs to a results prefix on GCS, and **logs
   the full matrices to stdout** so they're visible in the job logs.

Because it runs in-cluster, all reads/writes hit GCS directly. It reads only the
compact ``keys``/``meta`` artifacts across regions (megabytes), never text.

    iris job run --cluster marin --region us-central1 --extra cpu -- \
        python -m experiments.url_index.orchestrate
"""

import argparse
import logging
import os
import shutil
import tempfile

from iris.client.client import IrisClient, Job
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2

from experiments.infinigram.targets import Collection, IndexTarget, all_targets, get_target
from experiments.url_index.analyze import analyze
from experiments.url_index.launch_build_iris import JOB_WORK_ROOT, PRIORITY_BAND_MAP, RESOURCES

logger = logging.getLogger(__name__)

# Per-child wait ceiling. Children run in parallel, so wall-clock ~ slowest child.
_CHILD_TIMEOUT = 6 * 3600.0
_ANALYSIS_BUCKET = "gs://marin-us-central1"
RESULTS_ROOT = f"{_ANALYSIS_BUCKET}/url_index/analysis"


def _select_targets(datasets: list[str] | None, collections: list[Collection], only_landed: bool) -> list[IndexTarget]:
    if datasets is None:
        return [t for t in all_targets(only_landed=only_landed) if t.collection in collections]
    out: list[IndexTarget] = []
    for ds in datasets:
        for col in collections:
            t = get_target(ds, col)
            if only_landed and not t.source.landed:
                continue
            out.append(t)
    return out


def _submit_build(client: IrisClient, target: IndexTarget, band: int, overwrite: bool) -> Job:
    """Submit one region-pinned build child and return its Job handle."""
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
    return client.submit(
        entrypoint=Entrypoint.from_command(*cmd),
        name=f"urlindex-{target.name}",
        resources=ResourceSpec(cpu=res["cpu"], memory=res["memory"], disk=res["disk"]),
        environment=EnvironmentSpec(extras=["cpu"], env_vars={"PYTHONUNBUFFERED": "1"}),
        constraints=constraints,
        max_retries_preemption=10,
        max_retries_failure=1,
        priority_band=band,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build + analyze the whole URL index in one in-cluster job.")
    p.add_argument("--datasets", nargs="+", default=None, help="Datasets (default: whole registry).")
    p.add_argument(
        "--collections", nargs="+", choices=[c.value for c in Collection], default=[c.value for c in Collection]
    )
    p.add_argument("--only-landed", action="store_true", help="Only attempt tiers flagged landed.")
    p.add_argument("--priority", choices=sorted(PRIORITY_BAND_MAP), default="batch")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    collections = [Collection(c) for c in args.collections]
    targets = _select_targets(args.datasets, collections, args.only_landed)
    logger.info("Orchestrating %d build targets across both scales", len(targets))

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — launch via `iris job run -- python <this>`")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))
    band = PRIORITY_BAND_MAP[args.priority]

    # 1. Submit every build child.
    handles: list[tuple[IndexTarget, Job]] = []
    for t in targets:
        try:
            job = _submit_build(client, t, band, args.overwrite)
            handles.append((t, job))
            logger.info("submitted %s -> %s", t.name, job.job_id)
        except Exception:
            logger.exception("submit failed for %s", t.name)

    # 2. Wait for each; collect the ones that succeeded.
    succeeded: dict[Collection, list[str]] = {c: [] for c in collections}
    for t, job in handles:
        try:
            status = job.wait(timeout=_CHILD_TIMEOUT, poll_interval=60.0, raise_on_failure=False)
            if status.state == job_pb2.JOB_STATE_SUCCEEDED:
                succeeded[t.collection].append(t.dataset)
                logger.info("built %s", t.name)
            else:
                logger.warning("%s did not succeed (state=%s) — skipping from analysis", t.name, status.state)
        except TimeoutError:
            logger.warning("%s timed out after %.0fs — skipping from analysis", t.name, _CHILD_TIMEOUT)
        except Exception:
            logger.exception("wait failed for %s", t.name)

    # 3. Analyze each collection over its succeeded datasets.
    local_dir = tempfile.mkdtemp(prefix="url_index_analysis_")
    try:
        for collection in collections:
            datasets = sorted(succeeded[collection])
            logger.info("collection %s built datasets: %s", collection.value, datasets)
            if datasets:
                analyze(collection, datasets, f"{RESULTS_ROOT}/{collection.value}", local_dir)
    finally:
        shutil.rmtree(local_dir, ignore_errors=True)

    logger.info("Orchestration complete. Results under %s", RESULTS_ROOT)


if __name__ == "__main__":
    main()
