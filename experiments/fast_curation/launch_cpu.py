# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Phase 1 (CPU): submit N independent standalone WARC-worker jobs.

Each worker is a top-level Iris CPU job running ``cpu_phase.py`` (so each gets its own
``--extra`` deps — this is why we DON'T use Zephyr's worker actor-groups, whose child jobs
did not inherit the coordinator's extras). Workers coordinate through GCS (atomic per-WARC
claims + a central registry), so a preempted worker is simply reclaimed and there is no
coordinator to keep alive. A distinct ``--shuffle-seed`` per worker diversifies claim order.

    python -m experiments.fast_curation.launch_cpu --num-workers 2 --priority interactive \\
        --no-preemptible --limit 10            # canary
    python -m experiments.fast_curation.launch_cpu --num-workers 300 --priority batch   # full fleet
"""

from __future__ import annotations

import argparse
import logging
import subprocess

logger = logging.getLogger(__name__)

# cpu: foundational base for CPU workers; dclm: fasttext-wheel; extraction-bakeoff: justext[fasttext];
# transformers -> core via marin-levanter[serve]; warcio/resiliparse/charset_normalizer -> core.
CPU_EXTRAS = ["cpu", "dclm", "extraction-bakeoff"]

WANDB_API_KEY = "***REMOVED-WANDB-KEY***"
HF_TOKEN = "***REMOVED-HF-TOKEN***"


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
        "--cpu",
        str(args.cpu),
        "--memory",
        args.memory,
        "--disk",
        args.disk,
        "--enable-extra-resources",  # >= 4GB RAM / >= 10GB disk
        "--priority",
        args.priority,
        "--preemptible" if args.preemptible else "--no-preemptible",
        "--max-retries",
        str(args.max_retries),
        "--no-wait",
        "--job-name",
        f"fastcur-cpu-{args.spec}-{seed}",
        "-e",
        "WANDB_API_KEY",
        WANDB_API_KEY,
        "-e",
        "HF_TOKEN",
        HF_TOKEN,
    ]
    for extra in CPU_EXTRAS:
        cmd += ["--extra", extra]
    cmd += [
        "--",
        "python",
        "-m",
        "experiments.fast_curation.cpu_phase",
        "--spec",
        args.spec,
        "--manifest",
        args.manifest,
        "--bucket",
        args.bucket,
        "--shuffle-seed",
        str(seed),
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-idle-passes",
        str(args.max_idle_passes),
        "--justext-procs",
        str(args.justext_procs if args.justext_procs is not None else int(args.cpu)),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.start:
        cmd += ["--start", str(args.start)]
    return cmd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v1")
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--bucket", default="gs://marin-us-east5")
    ap.add_argument("--region", default="us-east5")
    ap.add_argument("--cluster", default="marin")
    ap.add_argument("--cpu", type=float, default=8, help="Cores per worker; JustText fans out across them.")
    ap.add_argument("--memory", default="32GB")
    ap.add_argument("--disk", default="24GB", help="Holds the 1.88 GB fastText model + WARC temp.")
    ap.add_argument("--justext-procs", type=int, default=None, help="JustText ProcessPool size (default = --cpu).")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--priority", default="batch", choices=["production", "interactive", "batch"])
    ap.add_argument("--preemptible", dest="preemptible", action="store_true", default=True)
    ap.add_argument("--no-preemptible", dest="preemptible", action="store_false")
    ap.add_argument("--max-retries", type=int, default=100)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-idle-passes", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for i in range(args.num_workers):
        seed = args.seed_start + i
        cmd = build_command(args, seed)
        logger.info("worker %d:\n  %s", seed, " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
