# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fan out :mod:`experiments.baseline_collection.grid_projection` across regions.

Unlike the grid_v1 fan-out, which mirrors a corpus into one compute region and
labels it there, this one goes to the data. The extraction run being measured is
un-consolidated and its batches sit in five regional buckets, so each region gets
its own jobs reading its own manifest. Nothing but kilobyte tallies leaves a
region.

TPU shape is per region because the pools differ, and in every case it is the
smallest **single-task** shape that region offers: Iris derives replicas as
chips/8, and anything above one task is gang-scheduled and will not be placed by
preemption (measured on 2026-07-29 — v5p-32 sat PENDING 12 min against 116 free
hosts while v5p-8 was ASSIGNED in under 30 s).

    python -m experiments.baseline_collection.launch_grid_projection \\
        --tasks-per-region 8 --priority interactive
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

CLUSTER = "marin"
QUALITY_MODEL = "gs://marin-us-central1/resources/datakit/quality/pooled_junkgate2"
MANIFEST_DIR = "gs://marin-us-central1/metadata/lpv1_1_projection"
# Iris region label -> the smallest single-task TPU shape that region has.
REGION_TPU: dict[str, str] = {
    "us-central1": "v5p-8",
    "us-east5": "v5p-8",
    "us-east1": "v6e-4",
    "us-west4": "v5litepod-4",
    "europe-west4": "v6e-4",
}
# The manifest is named by bucket short-name, which differs from the Iris region
# label for europe-west4 only.
REGION_MANIFEST_KEY: dict[str, str] = {
    "us-central1": "us-central1",
    "us-east5": "us-east5",
    "us-east1": "us-east1",
    "us-west4": "us-west4",
    "europe-west4": "eu-west4",
}
MEMORY = "64GB"
# Every `iris job run` opens its own SSH tunnel to the single controller, and
# those contend hard: 16 was fine for one region at a time, but three launchers
# at 16 put ~26 tunnels up at once on 2026-07-31 and EVERY submission blew the
# 900 s timeout. Submission is not the slow part of this pipeline, so keep the
# concurrency low enough that it always succeeds.
SUBMIT_PARALLELISM = 4
SUBMIT_TIMEOUT = 900


def submit(name: str, region: str, tpu: str, command: list[str], priority: str) -> tuple[str, bool]:
    """Submit one Iris job. Returns ``(name, ok)``; a rejected submit never raises.

    Partial capacity is the expected case for a preemptible fan-out, so one
    failure must not abort the wave — re-running the launcher retries it.
    """
    argv = [
        "uv", "run", "iris", "--cluster", CLUSTER, "job", "run",
        "--region", region,
        "--priority", priority,
        "--no-wait",
        "--job-name", name,
        "--tpu", tpu,
        "--cpu", "0.1",
        "--memory", MEMORY,
        "--extra", "tpu",
        "--enable-extra-resources",
    ]  # fmt: skip
    # The llama3 tokenizer used for the chars -> training-tokens ratio is a gated
    # HF repo, so the job cannot start without this.
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set; the gated llama3 tokenizer will fail to download")
    argv += ["-e", "HF_TOKEN", token, "--", *command]

    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=SUBMIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        # Each submission opens its own SSH tunnel to the controller, and those
        # serialize under load — a slow one must not take the wave down with it,
        # which is what an uncaught TimeoutExpired does inside ThreadPoolExecutor.map.
        logger.warning("submit TIMED OUT after %ds: %s", SUBMIT_TIMEOUT, name)
        return name, False
    ok = result.returncode == 0
    if not ok:
        logger.warning("submit FAILED %s: %s", name, (result.stderr or result.stdout)[-300:].strip())
    return name, ok


def run(regions: list[str], tasks_per_region: int, priority: str, wave: str, tpu: str | None = None) -> None:
    jobs: list[tuple[str, str, str, list[str]]] = []
    for region in regions:
        key = REGION_MANIFEST_KEY[region]
        shape = tpu or REGION_TPU[region]
        for idx in range(tasks_per_region):
            command = [
                "python", "-m", "experiments.baseline_collection.grid_projection",
                "--manifest", f"{MANIFEST_DIR}/manifest_{key}.jsonl",
                "--num-chunks", str(tasks_per_region),
                "--chunk-idx", str(idx),
                "--quality-model", QUALITY_MODEL,
            ]  # fmt: skip
            suffix = f"-{wave}" if wave else ""
            jobs.append((f"gridproj-{key}{suffix}-{idx:02d}", region, shape, command))

    logger.info("submitting %d jobs across %d regions at priority=%s", len(jobs), len(regions), priority)
    with ThreadPoolExecutor(max_workers=SUBMIT_PARALLELISM) as pool:
        results = list(pool.map(lambda j: submit(j[0], j[1], j[2], j[3], priority), jobs))
    landed = sum(ok for _, ok in results)
    logger.info("submitted %d/%d", landed, len(jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regions", nargs="+", default=list(REGION_TPU))
    parser.add_argument("--tasks-per-region", type=int, default=8)
    parser.add_argument("--priority", choices=["batch", "interactive"], default="interactive")
    parser.add_argument("--wave", default="", help="Tag for a top-up wave; Iris job names must be unique.")
    parser.add_argument(
        "--tpu",
        default=None,
        help="Override the region's default shape. A region's preferred pool can be fully subscribed — "
        "us-east5's v5p sat PENDING for two hours on 2026-07-31 while its v6e was free — and moving to "
        "another single-task shape in the same region is the cheapest way out.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    unknown = set(args.regions) - set(REGION_TPU)
    if unknown:
        raise ValueError(f"unknown region(s) {sorted(unknown)}; expected any of {sorted(REGION_TPU)}")
    run(args.regions, args.tasks_per_region, args.priority, args.wave, args.tpu)


if __name__ == "__main__":
    main()
