# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Production extraction launcher — runs as a single Iris parent job.

Submits child extraction jobs across all TPU types. On the Iris dashboard,
everything appears under one parent: /user/extract-production.

Usage::

    # Launch the parent (lightweight CPU job that submits children)
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --memory 2GB --no-wait --job-name extract-production \
        -- python experiments/baseline_collection/launch_production.py \
        --output-subdir documents/baseline_llm_extraction
"""

import argparse
import logging
import os
import time

from iris.client.client import IrisClient
from iris.cluster.constraints import preemptible_constraint, region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.marin_fs import REGION_TO_DATA_BUCKET

# All known regions — used to override parent region inheritance
ALL_REGIONS = sorted(REGION_TO_DATA_BUCKET.keys())

logger = logging.getLogger(__name__)

MANIFEST = "experiments/distill/baseline_warcs_3000.txt"
SCRIPT = "experiments/baseline_collection/run_extract_standalone.py"

# Multi-host env vars
MULTIHOST_ENV = {
    "TPU_PROCESS_BOUNDS": "1,1,1",
    "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
}


def submit_fleet(
    client: IrisClient,
    output_subdir: str,
    start: int,
    end: int | None,
    wave_size: int = 50,
    wave_pause: int = 300,
    fleet_filter: str | None = None,
    max_count: int | None = None,
):
    """Submit extraction jobs in waves across all TPU types as child jobs.

    Args:
        wave_size: Jobs per wave before pausing.
        wave_pause: Seconds to wait between waves (default 5 min).
        fleet_filter: If set, only submit jobs for this TPU type.
    """

    # Fleet definition: (tpu_type, count, is_multihost, tp_override)
    fleet = [
        # === Aim for the stars ===
        # Over-request freely. Autoscaler serves what it can.
        # No cost for pending jobs. Preemptible pricing for what runs.
        # Multi-host capped at pool max. Single-host pools are deep — push hard.
        #
        # EU multi-host (capped at pool max per region)
        ("v5litepod-16", 128, True, 4),  # 128 x 4 = 512 eff (max 512 = 256/region x 2)
        ("v6e-16", 64, True, 4),  # 64 x 4 = 256 eff (max 256)
        ("v5litepod-32", 32, True, 4),  # 32 x 8 = 256 eff (max 256 = 128/region x 2)
        ("v6e-32", 32, True, 4),  # 32 x 8 = 256 eff (max 256)
        # EU single-host (deep pools — this is where extra capacity lives)
        ("v5litepod-4", 500, False, None),  # 500 eff (max 2048)
        ("v5litepod-8", 300, False, None),  # 300 eff (max 1024)
        ("v6e-4", 400, False, None),  # 400 eff (max 1024)
        ("v6e-8", 200, False, None),  # 200 eff (max 512)
        # US supplemental (light touch)
        ("v5p-8", 5, False, None),  # 5 eff
        ("v5p-16", 3, True, 4),  # 3 x 2 = 6 eff
        ("v5p-32", 1, True, 4),  # 1 x 4 = 4 eff
        # Total: 1665 slices → ~2695 effective jobs (all achievable within pool max)
        # At 25% fill → ~674 actual → ~5 days
        # At 50% fill → ~1348 actual → ~2.5 days
        # At 100% fill → ~2695 actual → ~1.2 days
    ]

    if fleet_filter:
        fleet = [(t, c, m, tp) for t, c, m, tp in fleet if t == fleet_filter]
        if not fleet:
            logger.error("No fleet entries match --fleet-filter=%s", fleet_filter)
            return
        if max_count is not None:
            fleet = [(t, min(c, max_count), m, tp) for t, c, m, tp in fleet]
        logger.info("Filtered fleet to: %s (count=%d)", fleet_filter, fleet[0][1])

    seed = 0
    total_submitted = 0
    total_effective = 0

    for tpu_type, count, is_multihost, tp in fleet:
        logger.info("--- %s (x%d%s) ---", tpu_type, count, " multi-host" if is_multihost else "")

        for i in range(count):
            name = f"{tpu_type}-{i}"

            # Build command
            cmd_args = [
                "python",
                SCRIPT,
                "--manifest",
                MANIFEST,
                "--output-subdir",
                output_subdir,
                "--shuffle-seed",
                str(seed),
            ]
            if start > 0:
                cmd_args.extend(["--start", str(start)])
            if end is not None:
                cmd_args.extend(["--end", str(end)])
            if tp is not None:
                cmd_args.extend(["--tp", str(tp)])

            # Build environment
            env_vars = {}
            if is_multihost:
                env_vars.update(MULTIHOST_ENV)

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
                        extras=["vllm", "tpu"],
                        env_vars=env_vars,
                    ),
                    # CRITICAL: override parent inheritance.
                    # 1. preemptible=True: match preemptible workers (parent is non-preemptible)
                    # 2. region IN all_regions: prevents Iris from injecting the parent's
                    #    single-region pin (line 690 of client.py skips inheritance if child
                    #    already has a region constraint)
                    constraints=[preemptible_constraint(True), region_constraint(ALL_REGIONS)],
                    max_retries_preemption=100,
                    max_retries_failure=3,
                )
                logger.info("  Submitted %s -> %s", name, job.job_id)
                total_submitted += 1

                # Wave pause: first wave is tiny (canary), then ramp up
                if total_submitted == 5:
                    logger.info(
                        "=== CANARY WAVE: 5 jobs submitted. Pausing %ds to verify they get scheduled ===",
                        wave_pause,
                    )
                    time.sleep(wave_pause)
                elif total_submitted > 5 and (total_submitted - 5) % wave_size == 0:
                    wave_num = (total_submitted - 5) // wave_size
                    logger.info(
                        "=== Wave %d complete (%d jobs total). Pausing %ds ===",
                        wave_num,
                        total_submitted,
                        wave_pause,
                    )
                    time.sleep(wave_pause)
            except Exception as e:
                logger.error("  Failed to submit %s: %s", name, e)

            seed += 1

    # Estimate effective jobs (VMs)
    for tpu_type, count, is_multihost, _ in fleet:
        if is_multihost:
            # v5p-16 = 2 VMs, v5p-32 = 4 VMs, *-16 = 4 VMs
            if "32" in tpu_type:
                total_effective += count * 4
            elif "16" in tpu_type:
                vms = 2 if "v5p" in tpu_type else 4
                total_effective += count * vms
        else:
            total_effective += count

    logger.info("")
    logger.info("=== Submitted %d jobs (~%d effective VMs) ===", total_submitted, total_effective)
    logger.info("Monitor: iris job list --prefix /michaelryan/extract-production")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-subdir", default="documents/baseline_llm_extraction")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--wave-size", type=int, default=50, help="Jobs per wave before pausing")
    parser.add_argument("--wave-pause", type=int, default=300, help="Seconds between waves (default 5 min)")
    parser.add_argument(
        "--fleet-filter",
        type=str,
        default=None,
        help="Only submit jobs for this TPU type (e.g. 'v5litepod-16'). Default: all types.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Override the count for the filtered type (e.g. --fleet-filter v6e-4 --max-count 16)",
    )
    args = parser.parse_args()

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — must run inside an Iris job")
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)

    logger.info("Launching production extraction fleet")
    logger.info("Output: %s", args.output_subdir)
    logger.info("WARC range: [%d:%s]", args.start, args.end)

    submit_fleet(
        client,
        args.output_subdir,
        args.start,
        args.end,
        wave_size=args.wave_size,
        wave_pause=args.wave_pause,
        fleet_filter=args.fleet_filter,
        max_count=args.max_count,
    )

    # Keep parent alive so children stay nested in the hierarchy.
    # Parent dying kills children via cascading termination.
    logger.info("Parent job staying alive to keep children running...")
    logger.info("To stop everything: iris job stop /michaelryan/extract-production")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
