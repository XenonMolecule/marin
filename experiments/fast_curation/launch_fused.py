# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch fused single-phase workers: one Iris TPU job per slice, whole cascade on-node.

Each worker runs ``fused_phase.py`` — Phase A on the host's CPUs, scoring on the chip, no
presurvivors on GCS. Memory must cover ``queue_depth`` in-flight WARCs — each holds its FULL
decoded record set (~10-20GB peak for large WARCs) until extraction finishes — plus the
extraction pools; 100GB OOMed on the first pilot (2026-09-01), hence the 350GB default
(v6e-8 hosts carry 1.4TB).

    # pilot: 2 v6e-8 workers over the benchmark shards
    python -m experiments.fast_curation.launch_fused --num-workers 2 --tpu-type v6e-8 \\
        --region us-east1 --bucket gs://marin-us-east1 --max-shard 500
"""

from __future__ import annotations

import argparse
import logging
import os

from experiments.fast_curation._launch_common import submit_workers

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ["HF_TOKEN"]


def build_command(args: argparse.Namespace, seed: int) -> list[str]:
    cmd: list[str] = [
        "uv",
        "run",
        "iris",
        "--cluster",
        args.cluster,
        "job",
        "run",
        "--region",
        args.region,
        "--tpu",
        args.tpu_type,
        # The whole point of fused: the host's CPUs run Phase A. Request them explicitly
        # or the cgroup throttles the extraction pools to the default sliver.
        "--cpu",
        str(args.host_cpus),
        "--enable-extra-resources",
        "--extra",
        "tpu",
        "--extra",
        "gigatoken",
        "--extra",
        "cpu",  # fastText + resiliparse deps for the on-node Phase A
        "--extra",
        "dclm",
        "--memory",
        args.memory,
        "--priority",
        args.priority,
        "--preemptible" if args.preemptible else "--no-preemptible",
        "--max-retries",
        str(args.max_retries),
        "--no-wait",
        "--job-name",
        f"fastcur-fused-{args.spec}-{seed}",
        "-e",
        "HF_TOKEN",
        HF_TOKEN,
        "--",
        "python",
        "-m",
        "experiments.fast_curation.fused_phase",
        "--spec",
        args.spec,
        "--bucket",
        args.bucket,
        "--a-procs",
        str(args.a_procs),
        "--extract-procs-per-worker",
        str(args.extract_procs_per_worker),
        "--queue-depth",
        str(args.queue_depth),
        "--batch-size",
        str(args.batch_size),
        "--shuffle-seed",
        str(seed),
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-idle-passes",
        str(args.max_idle_passes),
        "--claim-stale-hours",
        str(args.claim_stale_hours),
    ]
    if args.max_shard is not None:
        cmd += ["--max-shard", str(args.max_shard)]
    if args.any_region:
        cmd.append("--any-region")
    return cmd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="lpv11_fastpipe_v2_1_fused")
    ap.add_argument("--bucket", default="gs://marin-us-east1")
    ap.add_argument("--region", default="us-east1")
    ap.add_argument("--cluster", default="marin")
    ap.add_argument("--tpu-type", default="v6e-8")
    ap.add_argument(
        "--memory", default="350GB", help="Queue-depth WARCs hold full decoded records (~10-20GB peak each)."
    )
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--host-cpus", type=int, default=100, help="CPU request (A pool + extraction).")
    ap.add_argument("--a-procs", type=int, default=14)
    ap.add_argument("--extract-procs-per-worker", type=int, default=6)
    ap.add_argument("--queue-depth", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--priority", default="batch", choices=["production", "interactive", "batch"])
    ap.add_argument("--preemptible", dest="preemptible", action="store_true", default=True)
    ap.add_argument("--no-preemptible", dest="preemptible", action="store_false")
    ap.add_argument("--max-retries", type=int, default=100)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-idle-passes", type=int, default=40)
    ap.add_argument("--claim-stale-hours", type=float, default=0.2)
    ap.add_argument("--max-shard", type=int, default=None)
    ap.add_argument("--any-region", action="store_true", help="Drain the global shard queue, not just own-region.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    submit_workers(args, build_command)


if __name__ == "__main__":
    main()
