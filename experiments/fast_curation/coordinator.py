# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adaptive fastcur coordinator: one per (TPU type x region), keeps the pool saturated.

Ports the proven control loop of ``experiments/baseline_collection/launch_adaptive.py`` (the
extraction coordinators) to the fast_curation workers, unchanged in spirit:

* submit a small initial batch, scale up in chunks ONLY when every submitted child is
  running — demand follows what the pool actually delivers, never a guess;
* ``--max-count`` caps LIVE children; dead children (preemption, a fleet-wide hang like the
  2026-08-31 freezes) free their slot and are backfilled within one check interval;
* stalls back off exponentially and probe with tiny batches — the coordinator never gives
  up on a pool, it just breathes with the weather;
* the coordinator itself is a tiny non-preemptible CPU job ON the cluster, immune to
  laptop-side auth/network/submit failures.

Unlike extraction, fastcur work is finite and region-bound: children are HARD-pinned to the
coordinator's region (their reads/writes must stay region-local), and the coordinator exits
once every one of its region's shards has a ``_b_done`` sentinel (children then idle-exit on
their own).

Launch one coordinator per pool (see the fused benchmark / 8M plan)::

    uv run iris --cluster marin job run --cpu 2 --memory 4GB --no-preemptible \\
        --priority batch --no-wait --job-name fastcur-coord-fused-v6e-8-us-east1 \\
        -e HF_TOKEN $HF_TOKEN -- \\
        python -m experiments.fast_curation.coordinator --mode fused \\
        --spec lpv11_fastpipe_v2_1_fused --tpu-type v6e-8 --region us-east1 \\
        --max-count 16 --max-shard 500
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import fsspec
from iris.client.client import IrisClient
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec, tpu_device
from iris.rpc import job_pb2

from experiments.fast_curation import shard_worklist as sw
from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)

SINGLE_HOST_TYPES = {"v6e-4", "v6e-8", "v5litepod-4", "v5litepod-8", "v5p-8", "v4-8"}
# Multihost slices run ONE INDEPENDENT fused worker per host (the extraction fleet's
# MULTIHOST_ENV trick): TPU_PROCESS_BOUNDS=1,1,1 confines each host's process to its local
# chips, and workers coordinate through GCS claims exactly like separate jobs — a
# v5litepod-16 is simply 4 fused workers. This is where the deep pools live (the April
# extraction fleet held 110 v5litepod-16 slices at once).
MULTIHOST_TYPES = {"v5litepod-16", "v5litepod-32", "v5p-16", "v5p-32", "v5p-64", "v6e-16", "v6e-32"}
MULTIHOST_ENV = {"TPU_PROCESS_BOUNDS": "1,1,1", "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1"}
BACKOFF_CAP_SECONDS = 1800


def _child_command(args: argparse.Namespace, seed: int) -> list[str]:
    bucket = sw.REGION_TO_BUCKET[args.region]
    if args.mode == "fused":
        cmd = [
            "-m",
            "experiments.fast_curation.fused_phase",
            "--spec",
            args.spec,
            "--bucket",
            bucket,
            "--a-procs",
            str(args.a_procs),
            "--extract-procs-per-worker",
            str(args.extract_procs_per_worker),
            "--queue-depth",
            str(args.queue_depth),
        ]
    else:
        cmd = [
            "-m",
            "experiments.fast_curation.tpu_phase",
            "--spec",
            args.spec,
            "--manifest",
            args.manifest,
            "--bucket",
            bucket,
            "--mode",
            "textb",
        ]
    cmd += [
        "--batch-size",
        str(args.batch_size),
        "--shuffle-seed",
        str(seed),
        "--claim-stale-hours",
        str(args.claim_stale_hours),
        "--max-idle-passes",
        str(args.child_max_idle_passes),
    ]
    if args.max_shard is not None:
        cmd += ["--max-shard", str(args.max_shard)]
    if args.mode == "fused":
        cmd.append("--any-region")  # no data gravity: drain the global queue, never starve
    return cmd


def submit_chunk(client: IrisClient, args: argparse.Namespace, start_seed: int, count: int) -> list[str]:
    extras = ["tpu", "gigatoken"] + (["cpu", "dclm"] if args.mode == "fused" else [])
    env_vars = {"HF_TOKEN": os.environ["HF_TOKEN"]}
    if args.tpu_type in MULTIHOST_TYPES:
        env_vars.update(MULTIHOST_ENV)
    submitted = []
    for i in range(count):
        seed = start_seed + i
        name = f"fastcur-{args.mode}-{args.spec}-{args.tpu_type}-{args.region}-{seed}"
        try:
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", *_child_command(args, seed)),
                name=name,
                resources=ResourceSpec(
                    cpu=args.child_cpu, memory=args.child_memory, disk="24GB", device=tpu_device(args.tpu_type)
                ),
                environment=EnvironmentSpec(extras=extras, env_vars=env_vars),
                constraints=[
                    preemptible_constraint(True),
                    # HARD region pin: fastcur children read the shard index / write kept
                    # region-locally; a child landing elsewhere would idle or read x-region.
                    Constraint.create(
                        key=WellKnownAttribute.REGION,
                        op=ConstraintOp.IN,
                        values=[args.region],
                        mode=job_pb2.CONSTRAINT_MODE_REQUIRED,
                    ),
                ],
                max_retries_preemption=100,
                max_retries_failure=3,
                priority_band=job_pb2.PRIORITY_BAND_BATCH,
            )
            logger.info("submitted %s -> %s", name, job.job_id)
            submitted.append(str(job.job_id))
        except Exception as e:
            logger.error("submit FAILED for %s: %s", name, e)
    return submitted


def _count_child_states(client: IrisClient, job_ids: list[str]) -> tuple[int, int, int]:
    """(running, pending, failed) — succeeded children gate nothing (they hold no capacity)."""
    running = pending = failed = 0
    for jid in job_ids:
        try:
            status = client.status(JobName.from_wire(jid))
            if status.state == job_pb2.JOB_STATE_RUNNING:
                running += 1
            elif status.state == job_pb2.JOB_STATE_SUCCEEDED:
                continue
            elif status.state in (job_pb2.JOB_STATE_FAILED, job_pb2.JOB_STATE_KILLED):
                failed += 1
            else:
                pending += 1  # pending/building/unqueryable all count as pending
        except Exception:
            pending += 1
    return running, pending, failed


def region_work_done(spec, region: str, max_shard: int | None) -> bool:
    """True when every one of this region's ladder shards has a ``_b_done`` sentinel."""
    fs = fsspec.filesystem("gcs")
    mine = {
        e["shard"]
        for e in sw.load_index(spec)
        if e["region"] == region and (max_shard is None or e["shard"] < max_shard)
    }
    if not mine:
        return True
    try:
        done = {
            int(p.split("/")[-1].replace("data-", ""))
            for p in fs.ls(sw.sentinel_prefix(spec, "b").removeprefix("gs://"), refresh=True)
        }
    except FileNotFoundError:
        return False
    return mine <= done


def run(client: IrisClient, args: argparse.Namespace) -> None:
    spec = get_spec(args.spec)
    logger.info(
        "fastcur coordinator: mode=%s %s x %s, max_live=%d, initial=%d, chunk=%d",
        args.mode,
        args.tpu_type,
        args.region,
        args.max_count,
        args.initial_batch,
        args.chunk_size,
    )
    total_submitted = 0
    seed_counter = args.seed_start
    stall_count = 0
    backoff_multiplier = 1
    all_job_ids: list[str] = []
    submit_cap = args.max_count * 3  # runaway backstop, as in launch_adaptive

    def _submit(count: int) -> None:
        nonlocal total_submitted, seed_counter
        names = submit_chunk(client, args, seed_counter, count)
        total_submitted += len(names)
        seed_counter += count
        all_job_ids.extend(names)

    _submit(min(args.initial_batch, args.max_count))
    while True:
        time.sleep(args.check_interval)
        if region_work_done(spec, args.region, args.max_shard):
            logger.info("all region shards B-complete; coordinator exiting (children idle-exit).")
            return
        if total_submitted >= submit_cap:
            logger.warning("lifetime submit cap %d reached — pool looks structurally broken; holding.", submit_cap)
            continue
        running, pending, failed = _count_child_states(client, all_job_ids)
        live = running + pending
        logger.info(
            "children: %d running, %d pending, %d failed (of %d submitted)", running, pending, failed, total_submitted
        )

        if pending > 0 or running == 0:
            stall_count += 1
            if stall_count >= args.patience:
                cooldown = min(args.check_interval * backoff_multiplier, BACKOFF_CAP_SECONDS)
                backoff_multiplier = min(backoff_multiplier * 2, 6)
                logger.info("stalled %d checks; backing off %ds (never giving up).", stall_count, cooldown)
                time.sleep(cooldown)
                stall_count = 0
                # Probe ONLY when everything is dead: pending children are already queued
                # demand, and piling probes on top of them just inflates the queue.
                if running == 0 and pending == 0 and live < args.max_count:
                    _submit(min(2, args.max_count - live))
            continue

        stall_count = 0
        backoff_multiplier = 1
        remaining = min(args.max_count - live, submit_cap - total_submitted)
        if remaining > 0:
            logger.info("all %d live children running — scaling up by %d.", live, min(args.chunk_size, remaining))
            _submit(min(args.chunk_size, remaining))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["fused", "textb"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--tpu-type", required=True, choices=sorted(SINGLE_HOST_TYPES | MULTIHOST_TYPES))
    ap.add_argument("--region", required=True, choices=sorted(sw.REGION_TO_BUCKET))
    ap.add_argument("--max-count", type=int, required=True, help="Ceiling on LIVE children.")
    ap.add_argument("--initial-batch", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=3)
    ap.add_argument("--check-interval", type=int, default=300)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-shard", type=int, default=None)
    # child worker knobs
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt", help="textb children only.")
    ap.add_argument("--a-procs", type=int, default=14)
    ap.add_argument("--extract-procs-per-worker", type=int, default=6)
    ap.add_argument("--queue-depth", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--claim-stale-hours", type=float, default=0.2)
    ap.add_argument("--child-max-idle-passes", type=int, default=40)
    ap.add_argument("--child-cpu", type=float, default=None)
    ap.add_argument("--child-memory", default=None)
    args = ap.parse_args()
    if args.child_cpu is None:
        # fused children own most of the host's CPUs; textb children need almost none.
        args.child_cpu = float(args.a_procs * (1 + args.extract_procs_per_worker)) if args.mode == "fused" else 4.0
    if args.child_memory is None:
        args.child_memory = "350GB" if args.mode == "fused" else "48GB"
    if os.environ.get("HF_TOKEN") is None:
        raise RuntimeError("HF_TOKEN must be in the coordinator's environment (children inherit it)")

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — run the coordinator inside an Iris job")
    client = IrisClient.remote(controller_address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))
    run(client, args)


if __name__ == "__main__":
    main()
