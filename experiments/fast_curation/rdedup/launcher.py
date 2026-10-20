# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create a dedup run, keep workers standing until it finishes, and report its progress.

``init`` freezes the plan (inputs, units, partitions, output root) under a scratch run root.
``run`` is a tiny control-plane loop meant for an Iris job outside the data region: it submits
preemptible worker jobs pinned to the data region, adds a chunk only when every submitted
worker task is running (demand follows what the pool delivers), replaces worker jobs that end
while stages remain, and exits when the final stage is sealed. Workers hold no state, so the
launcher dying or restarting loses nothing but the in-flight tasks of the workers it owned.
``status`` prints per-stage progress from claims and done marks.

    uv run python -m experiments.fast_curation.rdedup.launcher init \\
        --run-root gs://marin-us-east1/tmp/ttl=14d/rdedup/2m-v1 \\
        --output-root gs://marin-us-east1/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/2000987warcs \\
        --spec lpv11_fastpipe_v2_1_fused --bucket gs://marin-us-east1 --rules cc_canonical c075

    uv run iris --cluster marin job run --no-wait --region us-central2 --cpu 1 --memory 2GB \\
        --no-preemptible --priority interactive --job-name rdedup-launch-2m-v1 -- \\
        python -m experiments.fast_curation.rdedup.launcher run \\
        --run-root gs://marin-us-east1/tmp/ttl=14d/rdedup/2m-v1 --region us-east1 --max-workers 150
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import time

import pyarrow.parquet as pq
from iris.client.client import IrisClient
from iris.cluster.constraints import preemptible_constraint, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec, tpu_device
from iris.rpc import job_pb2

from experiments.fast_curation.rdedup.plan import build_plan, load_plan, write_plan
from experiments.fast_curation.rdedup.stages import Pipeline
from experiments.fast_curation.rdedup.store import ObjectInfo, ObjectStore, dumps, read_json, store_for
from experiments.fast_curation.rdedup.tasks import ClaimConfig, TaskBoard
from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)

RUN_MARKER = ".rdedup_run.json"


def _warc_hash(path: str) -> str:
    return os.path.basename(path).removeprefix("data-").removesuffix(".parquet")


def _read_warc_hashes(store: ObjectStore, path: str) -> set[str]:
    """WARC hashes from a manifest: a parquet with a ``warc_hash`` column, or one hash per line."""
    data = store.read(path)
    if path.endswith(".parquet"):
        return set(pq.read_table(io.BytesIO(data), columns=["warc_hash"]).column("warc_hash").to_pylist())
    return {line.strip() for line in data.decode().splitlines() if line.strip() and not line.startswith("#")}


def cmd_init(args: argparse.Namespace) -> None:
    store = store_for(args.run_root)
    kept = args.kept or f"{get_spec(args.spec).namespace(args.bucket)}/kept/"
    if args.listing:
        table = pq.read_table(args.listing, columns=["path", "size"])
        listing = [
            ObjectInfo(path, 0, 0.0, size)
            for path, size in zip(table.column("path").to_pylist(), table.column("size").to_pylist(), strict=True)
            if path.startswith(kept)
        ]
    else:
        listing = [info for info in store.list(kept) if info.path.endswith(".parquet")]
    if not listing:
        raise FileNotFoundError(f"no kept parquet under {kept}")
    listing.sort(key=lambda info: info.path)
    if args.warc_hashes:
        wanted = _read_warc_hashes(store_for(args.warc_hashes), args.warc_hashes)
        listing = [info for info in listing if _warc_hash(info.path) in wanted]
        missing = wanted - {_warc_hash(info.path) for info in listing}
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest WARCs have no kept shard, e.g. {sorted(missing)[:5]}")
    if args.limit_files:
        listing = listing[: args.limit_files]
    if args.expect_files is not None and len(listing) != args.expect_files:
        raise ValueError(f"found {len(listing)} kept files, expected {args.expect_files}")

    output_store = store_for(args.output_root)
    marker = f"{args.output_root.rstrip('/')}/{RUN_MARKER}"
    if output_store.stat(marker) is not None:
        owner = read_json(output_store, marker)["run_root"]
        if owner != args.run_root.rstrip("/"):
            raise ValueError(f"{args.output_root} belongs to run {owner}; pick a new output root")
    else:
        for folder in ("deduped/", "deduped_c075/", "stats/"):
            if output_store.list(f"{args.output_root.rstrip('/')}/{folder}"):
                raise ValueError(f"{args.output_root}/{folder} already holds objects from another run")

    paths = [info.path for info in listing]
    sizes = [info.size for info in listing]
    plan = build_plan(
        run_root=args.run_root,
        output_root=args.output_root,
        spec_id=args.spec,
        sizes=sizes,
        unit_bytes=int(args.unit_gb * 1e9),
        rules=args.rules,
        node_partitions=args.node_partitions,
        files_per_output=args.files_per_output,
    )
    write_plan(store, plan, paths, sizes)
    if output_store.stat(marker) is None:
        output_store.write(
            marker,
            dumps({"run_root": plan.run_root, "inputs_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest()}),
        )
    summary = {
        "files": plan.n_files,
        "bytes": sum(sizes),
        "units": plan.n_units,
        "exact_partitions": plan.exact_partitions,
        "lsh_partitions": plan.lsh_partitions,
        "node_partitions": plan.node_partitions,
        "c075_partitions": plan.c075_partitions,
        "rules": plan.rules,
    }
    logger.info("plan written to %s: %s", plan.run_root, json.dumps(summary))


def stage_progress(store: ObjectStore, run_root: str) -> list[dict]:
    """Progress of every stage up to the frontier: done / claimed / total and seal state."""
    plan = load_plan(store, run_root)
    pipeline = Pipeline(store, plan, procs=1)
    rows = []
    for spec in pipeline.stages_so_far():
        board = TaskBoard(store, pipeline.ctx.stage_dir(spec.name), spec.n_tasks, ClaimConfig())
        sealed = board.is_sealed()
        row = {"stage": spec.name, "tasks": spec.n_tasks, "sealed": sealed}
        if not sealed:
            state = board.state()
            in_flight = sum(
                1
                for task, (_generation, updated) in state.claims.items()
                if task not in state.done and state.now - updated <= board.config.stale_seconds
            )
            row.update(
                done=len(state.done), in_flight=in_flight, failures=len(store.list(f"{board.stage_dir}/failures/"))
            )
        rows.append(row)
    return rows


def cmd_status(args: argparse.Namespace) -> None:
    store = store_for(args.run_root)
    for row in stage_progress(store, args.run_root):
        print(json.dumps(row))


# Interactive outranks batch: it lets a run's tail claim CPU that batch work (including our own
# extraction fleet) holds, e.g. us-east1 where TPU-host spare cores are the only CPU pool.
WORKER_PRIORITY_BANDS = {
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
}


# Distinguishes worker job names across restarts of the launcher task: a restarted launcher counts
# its submissions from zero again, and reusing a name that is still shutting down raises
# JobAlreadyExists (2026-10-17: that crash failed the launcher and killed every worker under it).
_LAUNCH_ID = f"{int(time.time()) % 1_000_000:06d}"


def _submit_workers(client: IrisClient, args: argparse.Namespace, index: int, replicas: int) -> str:
    name = f"rdedup-w-{args.tag}-{_LAUNCH_ID}-{index:03d}"
    command = [
        "python",
        "-m",
        "experiments.fast_curation.rdedup.worker",
        "--run-root",
        args.run_root,
        "--cpus",
        str(args.worker_cpus),
        "--memory-gb",
        str(args.worker_memory_gb - 2),
    ]
    job = client.submit(
        entrypoint=Entrypoint.from_command(*command),
        name=name,
        resources=ResourceSpec(
            cpu=args.worker_cpus,
            memory=f"{args.worker_memory_gb}GB",
            disk=args.worker_disk,
            # A TPU device makes each worker a whole TPU host (chips idle). Iris only lets a task
            # preempt victims of its own device variant, so this is how an interactive dedup
            # reclaims host CPU that batch TPU jobs hold.
            device=tpu_device(args.worker_tpu) if args.worker_tpu else None,
        ),
        environment=EnvironmentSpec(),
        constraints=[preemptible_constraint(True), region_constraint([args.region])],
        replicas=replicas,
        max_retries_preemption=1000,
        max_retries_failure=50,
        max_task_failures=100000,
        priority_band=WORKER_PRIORITY_BANDS[args.worker_priority],
    )
    logger.info("submitted %s (%d workers) -> %s", name, replicas, job.job_id)
    return str(job.job_id)


def _task_counts(client: IrisClient, job_ids: list[str]) -> tuple[int, int, list[str]]:
    """(running tasks, pending tasks, job ids still live)."""
    running = pending = 0
    live = []
    for job_id in job_ids:
        try:
            status = client.status(JobName.from_wire(job_id))
        except Exception:
            logger.exception("status of %s failed; counting it as pending", job_id)
            live.append(job_id)
            continue
        if status.state in (job_pb2.JOB_STATE_SUCCEEDED, job_pb2.JOB_STATE_FAILED, job_pb2.JOB_STATE_KILLED):
            logger.info("worker job %s ended in state %s", job_id, job_pb2.JobState.Name(status.state))
            continue
        live.append(job_id)
        counts = dict(status.task_state_counts)
        running += counts.get("running", 0)
        pending += sum(v for k, v in counts.items() if k in ("pending", "assigned", "building"))
    return running, pending, live


def cmd_run(args: argparse.Namespace) -> None:
    store = store_for(args.run_root)
    plan = load_plan(store, args.run_root)
    pipeline = Pipeline(store, plan, procs=args.worker_cpus)
    address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set: run the launcher inside an Iris job")
    client = IrisClient.remote(address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))
    jobs: list[str] = []
    submitted_jobs = 0
    job_cap = 3 * max(1, args.max_workers // args.chunk) + 20
    while True:
        frontier = pipeline.frontier()
        if frontier is None:
            logger.info("every stage is sealed; launcher exiting (workers exit on their own)")
            return
        running, pending, jobs = _task_counts(client, jobs)
        live = running + pending
        logger.info("frontier %s; workers running=%d pending=%d jobs=%d", frontier.name, running, pending, len(jobs))
        if pending == 0 and live < args.max_workers:
            if submitted_jobs >= job_cap:
                logger.error("submitted %d worker jobs already; not submitting more (check worker logs)", submitted_jobs)
            else:
                jobs.append(_submit_workers(client, args, submitted_jobs, min(args.chunk, args.max_workers - live)))
                submitted_jobs += 1
        time.sleep(args.check_interval)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--run-root", required=True)
    init.add_argument("--output-root", required=True)
    init.add_argument("--spec", required=True)
    init.add_argument("--bucket", default=None, help="Bucket holding the consolidated kept/, e.g. gs://marin-us-east1")
    init.add_argument("--kept", default=None, help="Explicit kept/ prefix (overrides --spec/--bucket lookup).")
    init.add_argument(
        "--warc-hashes", default=None, help="Subset manifest: parquet with warc_hash, or one hash per line."
    )
    init.add_argument("--listing", default=None, help="Local parquet (path, size) of kept/, instead of listing it.")
    init.add_argument("--limit-files", type=int, default=None, help="Keep only the first N files (smoke runs).")
    init.add_argument("--expect-files", type=int, default=None)
    init.add_argument(
        "--rules",
        nargs="+",
        default=["c075"],
        choices=["cc_canonical", "c075"],
        help="cc_canonical over-removes through chained components; add it only for comparability with old trees.",
    )
    init.add_argument("--unit-gb", type=float, default=8.0, help="Compressed kept bytes per unit (GB).")
    init.add_argument("--node-partitions", type=int, default=None)
    init.add_argument("--files-per-output", type=int, default=32)
    init.set_defaults(fn=cmd_init)

    run = sub.add_parser("run")
    run.add_argument("--run-root", required=True)
    run.add_argument("--region", required=True)
    run.add_argument("--tag", required=True, help="Short name used in worker job names.")
    run.add_argument("--max-workers", type=int, required=True)
    run.add_argument("--chunk", type=int, default=16)
    run.add_argument("--worker-cpus", type=int, default=8)
    run.add_argument("--worker-memory-gb", type=int, default=32)
    run.add_argument("--worker-disk", default="10GB")
    run.add_argument("--check-interval", type=float, default=120.0)
    run.add_argument("--worker-priority", choices=sorted(WORKER_PRIORITY_BANDS), default="batch")
    run.add_argument(
        "--worker-tpu",
        default=None,
        help="Run each worker on a whole TPU host of this type (e.g. v6e-8); "
        "size --worker-cpus/--worker-memory-gb to the host.",
    )
    run.set_defaults(fn=cmd_run)

    status = sub.add_parser("status")
    status.add_argument("--run-root", required=True)
    status.set_defaults(fn=cmd_status)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
