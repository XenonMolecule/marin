# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adaptive extraction launcher — scales up based on actual allocation.

Submits jobs in small chunks, waits to see how many get scheduled,
and only submits more when the cluster is actually delivering.

Usage::

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --memory 2GB --cpu 2 --no-wait --job-name extract-adaptive-v5litepod16 \
        -- python experiments/baseline_collection/launch_adaptive.py \
        --tpu-type v5litepod-16 --max-count 64 --initial-batch 10 --chunk-size 5
"""

import argparse
import logging
import os
import time

from iris.client.client import IrisClient
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import job_pb2
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT
from rigging.filesystem import data_config

# Priority band name → proto enum. Used by --child-priority flag.
PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_INHERIT,
}

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = "experiments/distill/baseline_warcs_3000.txt"
SCRIPT = "experiments/baseline_collection/run_extract_standalone.py"
ALL_REGIONS = sorted(data_config().region_buckets.keys())

MULTIHOST_TYPES = {
    "v5p-16",
    "v5p-32",
    "v5p-64",
    "v5p-128",
    "v5p-256",
    "v5litepod-16",
    "v5litepod-32",
    "v6e-16",
    "v6e-32",
}
MULTIHOST_ENV = {
    "TPU_PROCESS_BOUNDS": "1,1,1",
    "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
}


def check_job_states(client: IrisClient, parent_prefix: str, submitted_names: list[str]) -> dict[str, int]:
    """Count running/pending/failed children. Returns dict of counts."""
    # We can't easily query Iris from inside a job without hitting the slow API.
    # Instead, we track by checking if the job's tasks are running.
    # For now, use a simple time-based heuristic:
    # - Jobs submitted > 5 min ago that haven't failed are assumed running or pending
    # This is imperfect but avoids hammering the Iris API.
    return {"submitted": len(submitted_names)}


def submit_chunk(
    client: IrisClient,
    tpu_type: str,
    output_subdir: str,
    start: int,
    end: int | None,
    chunk_start_seed: int,
    chunk_size: int,
    manifest: str,
    spec: str | None = None,
    pipeline: str | None = None,
    group_size: int | None = None,
    model: str | None = None,
    region: str | None = None,
    child_priority_band: int = job_pb2.PRIORITY_BAND_INHERIT,
    preemptible: bool = True,
) -> list[str]:
    """Submit a chunk of jobs. Returns list of submitted job names.

    If ``pipeline`` is set, child jobs run with ``--pipeline {pipeline}`` (the
    multi-call two-stage/one-call systems) and write to that pipeline's canonical
    namespace; ``group_size`` forwards ``--group-size``. Otherwise, if ``spec`` is
    set, child jobs run with ``--spec {spec}`` and the prompt
    comes from the spec registry. ``output_subdir`` is passed through only when
    it is non-None: with a spec that redirects output + the skip-registry to a
    benchmark namespace (keeping the spec's prompt); without a spec it is the
    legacy namespace. ``model``, when set, overrides the child's default
    checkpoint (e.g. to benchmark a smaller model on the same spec).
    """
    is_multihost = tpu_type in MULTIHOST_TYPES
    env_vars = dict(MULTIHOST_ENV) if is_multihost else {}
    # The TPU vLLM fork lives outside the workspace lock (no `vllm` extra anymore);
    # it is provisioned into the worker venv via pip_packages below. These steer its
    # source build: TPU target instead of CUDA, and CPU torch wheels for its torch dep
    # (jax/libtpu do the TPU compute), mirroring marin.inference's IsolatedTpuVllm.
    env_vars["VLLM_TARGET_DEVICE"] = "tpu"
    env_vars["UV_TORCH_BACKEND"] = "cpu"
    tp_args = ["--tp", "4"] if is_multihost else []

    name_prefix = f"extract-{pipeline}-" if pipeline else (f"extract-{spec}-" if spec else "extract-")

    submitted = []
    for i in range(chunk_size):
        seed = chunk_start_seed + i
        name = f"{name_prefix}{tpu_type}-{seed}"

        cmd_args = [
            "python",
            SCRIPT,
            "--manifest",
            manifest,
            "--shuffle-seed",
            str(seed),
            *tp_args,
        ]
        if pipeline:
            cmd_args.extend(["--pipeline", pipeline])
            if group_size is not None:
                cmd_args.extend(["--group-size", str(group_size)])
            # Like --spec: --output-subdir is only an explicit benchmark override;
            # without it the child uses the pipeline's canonical namespace.
            if output_subdir is not None:
                cmd_args.extend(["--output-subdir", output_subdir])
        elif spec:
            cmd_args.extend(["--spec", spec])
            # Pass --output-subdir alongside --spec only as an explicit benchmark
            # override; without it the child uses the spec's canonical namespace.
            if output_subdir is not None:
                cmd_args.extend(["--output-subdir", output_subdir])
        else:
            cmd_args.extend(["--output-subdir", output_subdir])
        if model:
            cmd_args.extend(["--model", model])
        if start > 0:
            cmd_args.extend(["--start", str(start)])
        if end is not None:
            cmd_args.extend(["--end", str(end)])

        try:
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", *cmd_args[1:]),
                name=name,
                resources=ResourceSpec(
                    cpu=0.5,
                    memory="64GB",
                    disk="5GB",
                    device=tpu_device(tpu_type),
                ),
                environment=EnvironmentSpec(
                    extras=["tpu"],
                    pip_packages=[VLLM_FORK_REQUIREMENT, TPU_INFERENCE_FORK_REQUIREMENT],
                    env_vars=env_vars,
                ),
                # Region constraint is SOFT (preferred, not required) so it prevents
                # parent region inheritance (line 645 of client.py) without restricting
                # the autoscaler to a single region. Soft routing constraints influence
                # group ordering but never exclude groups.
                #
                # Use ``Constraint.create`` — it auto-wraps raw strings into
                # ``AttributeValue``. Direct ``Constraint(values=...)`` requires
                # already-wrapped values and silently fails with ``'str' object has
                # no attribute 'to_proto'`` if you pass raw strings.
                constraints=[
                    # preemptible=True is SOFT (prefer spot, fall back to reserved);
                    # preemptible=False is HARD (require reserved/on-demand), used to
                    # target the low-churn reserved v4 pool via --capacity-type reserved.
                    preemptible_constraint(preemptible),
                    # Default: SOFT preference over all regions (prevents parent
                    # region inheritance without restricting the autoscaler). When
                    # ``region`` is set (e.g. the model checkpoint lives in only one
                    # region), HARD-pin children there so the model read stays local.
                    Constraint.create(
                        key=WellKnownAttribute.REGION,
                        op=ConstraintOp.IN,
                        values=[region] if region else ALL_REGIONS,
                        mode=job_pb2.CONSTRAINT_MODE_REQUIRED if region else job_pb2.CONSTRAINT_MODE_PREFERRED,
                    ),
                ],
                max_retries_preemption=100,
                max_retries_failure=3,
                priority_band=child_priority_band,
            )
            logger.info("  Submitted %s -> %s", name, job.job_id)
            submitted.append(str(job.job_id))
        except Exception as e:
            logger.error("  Failed to submit %s: %s", name, e)

    return submitted


def _count_child_states(client: IrisClient, submitted_job_ids: list[str]) -> tuple[int, int, int]:
    """Query actual child job states. Returns (running, pending, failed).

    Succeeded children count toward none of the three: they hold no capacity, so they must
    not gate scale-up. (Counting them as pending froze the OLMIX resiliparse coordinator at
    its initial batch the moment its first child finished.)
    """
    running = 0
    pending = 0
    failed = 0
    for job_id_str in submitted_job_ids:
        try:
            from iris.cluster.types import JobName
            from iris.rpc import job_pb2

            job_id = JobName.from_wire(job_id_str)
            status = client.status(job_id)
            if status.state == job_pb2.JOB_STATE_RUNNING:
                running += 1
            elif status.state == job_pb2.JOB_STATE_SUCCEEDED:
                continue
            elif status.state == job_pb2.JOB_STATE_PENDING:
                pending += 1
            elif status.state in (job_pb2.JOB_STATE_FAILED, job_pb2.JOB_STATE_KILLED):
                failed += 1
            else:
                pending += 1  # building, etc. count as pending
        except Exception:
            pending += 1  # if we can't query, assume pending
    return running, pending, failed


def run_adaptive(
    client: IrisClient,
    tpu_type: str,
    output_subdir: str,
    max_count: int,
    initial_batch: int,
    chunk_size: int,
    check_interval: int,
    patience: int,
    start: int,
    end: int | None,
    manifest: str,
    spec: str | None = None,
    pipeline: str | None = None,
    group_size: int | None = None,
    model: str | None = None,
    region: str | None = None,
    child_priority_band: int = job_pb2.PRIORITY_BAND_INHERIT,
    preemptible: bool = True,
):
    """Adaptive scaling loop. Only submits more when ALL previous jobs are running."""
    logger.info(
        "Adaptive launcher: %s, manifest=%s, mode=%s, max=%d, initial=%d, chunk=%d, child_priority=%s",
        tpu_type,
        manifest,
        f"pipeline={pipeline}" if pipeline else (f"spec={spec}" if spec else "(legacy / hardcoded prompt)"),
        max_count,
        initial_batch,
        chunk_size,
        job_pb2.PriorityBand.Name(child_priority_band).replace("PRIORITY_BAND_", "").lower(),
    )

    total_submitted = 0
    seed_counter = 0
    stall_count = 0
    all_job_ids: list[str] = []

    def _submit(seed_offset: int, count: int) -> list[str]:
        return submit_chunk(
            client,
            tpu_type,
            output_subdir,
            start,
            end,
            seed_offset,
            count,
            manifest=manifest,
            spec=spec,
            pipeline=pipeline,
            group_size=group_size,
            model=model,
            region=region,
            child_priority_band=child_priority_band,
            preemptible=preemptible,
        )

    # Initial batch
    initial = min(initial_batch, max_count)
    logger.info("=== Initial batch: %d jobs ===", initial)
    names = _submit(seed_counter, initial)
    total_submitted += len(names)
    seed_counter += initial
    all_job_ids.extend(names)

    # Adaptive loop: only scale up when ALL submitted jobs are running.
    # Never permanently gives up — backs off with increasing cooldown, then retries.
    backoff_multiplier = 1
    # Runaway backstop: backfill is unbounded in time, so a pool whose children always
    # die instantly (hardware the inference stack cannot run on) would resubmit forever,
    # burning TPU on model loads that never produce a batch. Cap lifetime submissions at
    # a multiple of the ceiling: hitting it means the pool is structurally broken rather
    # than merely being preempted.
    submit_cap = max_count * 3
    # ``max_count`` is a ceiling on LIVE children, not a lifetime submission quota:
    # children that die (preemption, an upstream outage) free their slot and are
    # backfilled. Counting dead children against the cap used to retire a pool
    # permanently after a transient outage -- the coordinator stayed alive with zero
    # children and no ability to submit more.
    while total_submitted < submit_cap:
        logger.info("Waiting %ds before checking allocation...", check_interval)
        time.sleep(check_interval)

        # Actually check child job states
        running, pending, failed = _count_child_states(client, all_job_ids)
        live = running + pending
        logger.info(
            "Child states: %d running, %d pending, %d failed (of %d submitted)",
            running,
            pending,
            failed,
            total_submitted,
        )

        if pending > 0:
            stall_count += 1
            logger.info(
                "%d jobs still pending. Not scaling up (stall %d/%d).",
                pending,
                stall_count,
                patience,
            )
            if stall_count >= patience:
                # Back off instead of giving up. Double the cooldown each time, cap at 30 min.
                cooldown = min(check_interval * backoff_multiplier, 1800)
                backoff_multiplier = min(backoff_multiplier * 2, 6)
                logger.info(
                    "Stalled %d times. Backing off for %ds (not giving up, %d/%d submitted).",
                    stall_count,
                    cooldown,
                    total_submitted,
                    max_count,
                )
                time.sleep(cooldown)
                stall_count = 0  # Reset stall count after cooldown
            continue

        # If ALL children are dead (0 running, 0 pending), back off then retry
        # with a small probe batch instead of continuing to scale up.
        if running == 0:
            stall_count += 1
            logger.info(
                "All %d children dead (failed/killed). Not scaling up (stall %d/%d).",
                failed,
                stall_count,
                patience,
            )
            if stall_count >= patience:
                cooldown = min(check_interval * backoff_multiplier, 1800)
                backoff_multiplier = min(backoff_multiplier * 2, 6)
                logger.info(
                    "All-dead stall %d times. Backing off for %ds then retrying with probe batch.",
                    stall_count,
                    cooldown,
                )
                time.sleep(cooldown)
                stall_count = 0
                # Submit a small probe batch to test if TPUs are available again
                if live < max_count:
                    probe_size = min(2, max_count - live)
                    logger.info(
                        "=== Probe batch: %d jobs (total will be %d/%d) ===",
                        probe_size,
                        total_submitted + probe_size,
                        max_count,
                    )
                    names = _submit(seed_counter, probe_size)
                    total_submitted += len(names)
                    seed_counter += probe_size
                    all_job_ids.extend(names)
            continue

        # All submitted jobs are running. Scale up!
        stall_count = 0
        backoff_multiplier = 1  # Reset backoff on success
        remaining = min(max_count - live, submit_cap - total_submitted)
        if remaining <= 0:
            # Pool is at its ceiling; keep watching so deaths get backfilled.
            continue
        next_chunk = min(chunk_size, remaining)
        logger.info(
            "=== All %d jobs running! Submitting %d more (total will be %d/%d) ===",
            running,
            next_chunk,
            total_submitted + next_chunk,
            max_count,
        )
        names = _submit(seed_counter, next_chunk)
        total_submitted += len(names)
        seed_counter += next_chunk
        all_job_ids.extend(names)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu-type", required=True, help="TPU type (e.g. v5litepod-16)")
    parser.add_argument("--max-count", type=int, default=64, help="Maximum slices to request")
    parser.add_argument("--initial-batch", type=int, default=5, help="First batch size (canary)")
    parser.add_argument("--chunk-size", type=int, default=5, help="Jobs per subsequent chunk")
    parser.add_argument("--check-interval", type=int, default=300, help="Seconds between chunks (default 5 min)")
    parser.add_argument("--patience", type=int, default=3, help="Stall cycles before stopping")
    parser.add_argument(
        "--output-subdir",
        default=None,
        help=(
            "Output subdir for children. Without --spec it is the legacy namespace "
            "(default: documents/baseline_llm_extraction). With --spec, leave unset to "
            "use the spec's canonical namespace, or set it to redirect output + the "
            "skip-registry to a benchmark namespace while keeping the spec's prompt."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the child extraction model checkpoint (gs:// HF dir vLLM can "
            "load). Default (unset): children resolve the 8B rephraser from the local "
            "region. Set this to benchmark a different model on the same spec."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Path to the WARC manifest. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--spec",
        default=None,
        help=(
            "Extraction spec id (key in extraction_specs.SPECS). If set, child "
            "jobs run with ``--spec`` and use that spec's prompt + namespace "
            "(no fallback to the legacy hardcoded prompt). If unset, children "
            "use the legacy prompt and write to --output-subdir."
        ),
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "Multi-call pipeline id (llm_pipeline_v1 or llm_simple_v1). Mutually "
            "exclusive with --spec. Children run with --pipeline and write to that "
            "pipeline's canonical namespace documents/baseline_llm_extraction/{id}."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Forwarded to children as --group-size (docs per checkpoint group). Only with --pipeline.",
    )
    parser.add_argument(
        "--child-region",
        default=None,
        help=(
            "Hard-pin child jobs to this region (e.g. us-east5). Use when the model "
            "checkpoint exists in only one region so the model read stays local. "
            "Default (unset): soft preference across all regions."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--capacity-type",
        choices=["preemptible", "reserved"],
        default="preemptible",
        help=(
            "Capacity class for child TPUs. 'preemptible' (default) softly prefers "
            "spot and falls back to reserved. 'reserved' HARD-requires non-preemptible "
            "capacity (e.g. the reserved v4 pool in us-central2) — far lower churn, at "
            "the cost of contending with reserved workloads; pair with --child-priority "
            "batch so those workloads still win."
        ),
    )
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="batch",
        help=(
            "Priority band for child TPU jobs. Defaults to 'batch' — children only "
            "get scheduled when higher-priority demand is satisfied. Run the parent "
            "itself at higher priority via `iris job run --priority production` so "
            "it doesn't get preempted and lose track of its children."
        ),
    )
    args = parser.parse_args()
    if args.pipeline and args.spec:
        parser.error("--pipeline and --spec are mutually exclusive")
    if args.group_size is not None and not args.pipeline:
        parser.error("--group-size only applies with --pipeline (it would be silently ignored)")

    # Validate spec/pipeline at parent startup so a typo fails before any children
    # are submitted (would otherwise surface only in child stderr).
    if args.pipeline is not None:
        from experiments.baseline_collection.pipelines.pipeline_specs import get_pipeline

        pipe_obj = get_pipeline(args.pipeline)  # raises ValueError on miss
        logger.info("Using pipeline=%s (frozen cert %s)", args.pipeline, pipe_obj.source_cert)
    elif args.spec is not None:
        from experiments.baseline_collection.extraction_specs import get_spec

        spec_obj = get_spec(args.spec)  # raises ValueError on miss
        logger.info("Using spec=%s — %s", args.spec, spec_obj.description or "(no description)")

    # Without a spec OR pipeline, children need the legacy namespace default. With
    # either, None means "use the canonical namespace" and a value is a benchmark
    # override — both are passed through to children as-is.
    output_subdir = args.output_subdir
    if args.spec is None and args.pipeline is None and output_subdir is None:
        output_subdir = "documents/baseline_llm_extraction"

    child_priority_band = PRIORITY_BAND_MAP[args.child_priority]

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — must run inside an Iris job")
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)

    run_adaptive(
        client,
        tpu_type=args.tpu_type,
        output_subdir=output_subdir,
        max_count=args.max_count,
        initial_batch=args.initial_batch,
        chunk_size=args.chunk_size,
        check_interval=args.check_interval,
        patience=args.patience,
        start=args.start,
        end=args.end,
        manifest=args.manifest,
        spec=args.spec,
        pipeline=args.pipeline,
        group_size=args.group_size,
        model=args.model,
        region=args.child_region,
        child_priority_band=child_priority_band,
        preemptible=(args.capacity_type == "preemptible"),
    )

    # Keep parent alive
    logger.info("Parent staying alive to keep children running...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
