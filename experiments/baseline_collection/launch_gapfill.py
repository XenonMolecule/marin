# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the claim-free gap-fill fleet that finishes an extraction's long tail.

The claim-based fleet (``launch_adaptive.py``) processes one WARC per
claim-holder, so once the remaining WARCs are fewer than the workers the surplus
workers idle and thrash on claims, and a hung claim-holder blocks its WARC for
the full stale window. ``fill_missing_batches.py`` sidesteps both: it takes no
claims and splits a WARC's *missing batch indices* across shards, so N workers
make progress on the same WARC and a hung shard costs only its own slice.

This launcher submits the leaves directly — there is no coordinator, because the
remaining-WARC list is fixed and finite, so nothing needs to backfill.

WARCs are partitioned into groups; every shard of a group is pinned to the same
region so the second shard reads the WARC from that region's read-through cache
(``tmp/ttl=2d/warc-cache/``) instead of re-pulling it from Common Crawl.

Usage::

    uv run python experiments/baseline_collection/launch_gapfill.py \
        --manifest experiments/distill/lpv11_remaining.txt \
        --pipeline llm_pipeline_v1_1 --warcs-per-group 2 --num-shards 3
"""

import argparse
import logging
import pathlib

from iris.cli.connect import open_iris_client
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import job_pb2
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT
from rigging.filesystem import data_config

logger = logging.getLogger(__name__)

MODULE = "experiments.baseline_collection.fill_missing_batches"
GROUP_MANIFEST_DIR = pathlib.Path("experiments/distill/lpv11_gapfill")
ALL_REGIONS = sorted(data_config().region_buckets.keys())

# Single-host TPU targets only: one process owns all chips, so ``fill`` needs no
# multihost bounds and a preemption kills exactly one shard. Each entry is
# (tpu_type, region) and the model checkpoint is mirrored in all of these
# regions, so both the WARC read and the 32.8GB checkpoint read stay in-region.
TARGETS = [
    ("v6e-4", "us-east5"),
    ("v6e-4", "europe-west4"),
    ("v6e-4", "us-east1"),
    ("v5litepod-4", "us-west4"),
]


def _write_group_manifests(warcs: list[str], per_group: int) -> list[pathlib.Path]:
    """Split the remaining-WARC list into per-group manifest files in the repo.

    They ride along in the iris bundle captured at submission, so each worker
    reads its group list from local disk — no GCS round-trip, no cross-region
    read of a manifest that happens to live in another region's bucket.
    """
    GROUP_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for stale in GROUP_MANIFEST_DIR.glob("g*.txt"):
        stale.unlink()
    paths = []
    for idx in range(0, len(warcs), per_group):
        group = warcs[idx : idx + per_group]
        path = GROUP_MANIFEST_DIR / f"g{idx // per_group:03d}.txt"
        path.write_text("\n".join(group) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Remaining-WARC list (one path per line)")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--warcs-per-group", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=3, help="Shards per group; splits missing batch indices")
    parser.add_argument("--batch-size", type=int, default=250, help="MUST match the run's --group-size")
    parser.add_argument("--name-prefix", default="gapfill-lpv11")
    parser.add_argument("--max-jobs", type=int, default=None, help="Submit at most this many jobs (canary runs)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tpu-type",
        default=None,
        help="Override every group's TPU type. Use to retry a batch on different hardware "
        "when a failure looks device-specific.",
    )
    parser.add_argument(
        "--priority",
        choices=["batch", "interactive"],
        default="batch",
        help="Scheduling band. batch is the default for bulk extraction; interactive jumps "
        "the batch queue and is for a small tail that is blocking a run from closing.",
    )
    parser.add_argument(
        "--pin-region",
        default=None,
        help="Override every group's region (with --region-mode required). Use to aim a "
        "launch at a region with idle capacity when the default round-robin is starved.",
    )
    parser.add_argument(
        "--region-mode",
        choices=["required", "preferred"],
        default="required",
        help="required = hard-pin each group to one region (shares the WARC cache); "
        "preferred = soft preference over all regions (schedulability)",
    )
    parser.add_argument(
        "--start-group", type=int, default=0, help="Skip groups below this index (resume a launch that was cut short)"
    )
    parser.add_argument("--config", default="lib/iris/config/marin.yaml")
    args = parser.parse_args()

    warcs = [line.strip() for line in pathlib.Path(args.manifest).read_text().splitlines() if line.strip()]
    groups = _write_group_manifests(warcs, args.warcs_per_group)
    logger.info("%d WARCs -> %d groups x %d shards", len(warcs), len(groups), args.num_shards)

    submitted = 0
    with open_iris_client(config_file=pathlib.Path(args.config), workspace=pathlib.Path.cwd()) as client:
        for group_idx, group_path in enumerate(groups):
            if group_idx < args.start_group:
                continue
            tpu_type, region = TARGETS[group_idx % len(TARGETS)]
            tpu_type = args.tpu_type or tpu_type
            region = args.pin_region or region
            for shard in range(args.num_shards):
                if args.max_jobs is not None and submitted >= args.max_jobs:
                    logger.info("Reached --max-jobs=%d; stopping", args.max_jobs)
                    return
                name = f"{args.name_prefix}-g{group_idx:03d}-s{shard}"
                cmd = [
                    "-m",
                    MODULE,
                    "--pipeline",
                    args.pipeline,
                    "--manifest",
                    str(group_path),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-shards",
                    str(args.num_shards),
                    "--shard",
                    str(shard),
                ]
                if args.dry_run:
                    logger.info("[dry-run] %s on %s/%s -> python %s", name, tpu_type, region, " ".join(cmd))
                    submitted += 1
                    continue
                try:
                    job = client.submit(
                        entrypoint=Entrypoint.from_command("python", *cmd),
                        name=name,
                        resources=ResourceSpec(
                            cpu=0.5,
                            memory="64GB",
                            disk="5GB",
                            device=tpu_device(tpu_type),
                        ),
                        environment=EnvironmentSpec(
                            extras=["tpu"],
                            # The TPU vLLM fork lives outside the workspace lock; these steer
                            # its source build to TPU with CPU torch wheels, matching the
                            # production extraction workers exactly.
                            pip_packages=[VLLM_FORK_REQUIREMENT, TPU_INFERENCE_FORK_REQUIREMENT],
                            env_vars={"VLLM_TARGET_DEVICE": "tpu", "UV_TORCH_BACKEND": "cpu"},
                        ),
                        constraints=[
                            preemptible_constraint(True),
                            # ``required`` hard-pins a group's shards to one region so
                            # they share that region's WARC cache. That starves the
                            # launch when the pinned pools have no capacity: pinned jobs
                            # register ZERO autoscaler demand and simply never schedule.
                            # ``preferred`` trades cache sharing -- each region re-pulls
                            # the WARC from Common Crawl over HTTP, not paid egress --
                            # for schedulability. The checkpoint is mirrored everywhere
                            # either way, so neither mode risks a cross-region read.
                            Constraint.create(
                                key=WellKnownAttribute.REGION,
                                op=ConstraintOp.IN,
                                values=[region] if args.region_mode == "required" else ALL_REGIONS,
                                mode=(
                                    job_pb2.CONSTRAINT_MODE_REQUIRED
                                    if args.region_mode == "required"
                                    else job_pb2.CONSTRAINT_MODE_PREFERRED
                                ),
                            ),
                        ],
                        max_retries_preemption=100,
                        max_retries_failure=3,
                        priority_band=(
                            job_pb2.PRIORITY_BAND_INTERACTIVE
                            if args.priority == "interactive"
                            else job_pb2.PRIORITY_BAND_BATCH
                        ),
                    )
                    logger.info("Submitted %s (%s/%s) -> %s", name, tpu_type, region, job.job_id)
                    submitted += 1
                except Exception as e:
                    logger.error("Failed to submit %s: %s", name, e)

    logger.info("Submitted %d jobs", submitted)


if __name__ == "__main__":
    main()
