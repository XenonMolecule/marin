# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Phase 2 (TPU): submit N independent ModernBERT-scoring worker jobs.

Each worker is a standalone Iris TPU job running ``tpu_phase.py``; they coordinate purely
through GCS (atomic per-WARC claims + the central completed registry), so there is no
coordinator to keep alive and a preempted worker is simply reclaimed after its claim goes
stale. A distinct ``--shuffle-seed`` per worker diversifies claim order.

The workers can start while Phase 1 is still running — they drain survivors as the CPU phase
produces them, and exit once every WARC is in the registry.

    # canary: one non-preemptible worker over the first 5 WARCs, parity-checking ON
    python -m experiments.fast_curation.launch_tpu --num-workers 1 --priority interactive \\
        --no-preemptible --limit 5

    # full fleet: many preemptible batch workers
    python -m experiments.fast_curation.launch_tpu --num-workers 32 --priority batch
"""

from __future__ import annotations

import argparse
import logging

from experiments.fast_curation._launch_common import submit_workers

logger = logging.getLogger(__name__)

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
        "--tpu",
        args.tpu_type,
        "--enable-extra-resources",
        "--extra",
        "tpu",
        "--memory",
        args.memory,
        "--priority",
        args.priority,
        "--preemptible" if args.preemptible else "--no-preemptible",
        "--max-retries",
        str(args.max_retries),
        "--no-wait",
        "--job-name",
        f"fastcur-tpu-{args.spec}-{seed}",
        "-e",
        "HF_TOKEN",
        HF_TOKEN,
        "--",
        "python",
        "-m",
        "experiments.fast_curation.tpu_phase",
        "--spec",
        args.spec,
        "--manifest",
        args.manifest,
        "--bucket",
        args.bucket,
        "--batch-size",
        str(args.batch_size),
        "--shuffle-seed",
        str(seed),
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-idle-passes",
        str(args.max_idle_passes),
        "--mode",
        args.mode,
    ]
    if not args.bucket_tokens:
        cmd.append("--no-bucket-tokens")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v3")
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--bucket", default="gs://marin-us-east5")
    ap.add_argument("--region", default="us-east5")
    ap.add_argument("--cluster", default="marin")
    ap.add_argument("--tpu-type", default="v6e-4")
    ap.add_argument("--memory", default="64GB")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0, help="First shuffle seed; workers use seed-start..+N.")
    ap.add_argument("--batch-size", type=int, default=32, help="Per-job ModernBERT batch (multiple of chip count).")
    ap.add_argument("--priority", default="batch", choices=["production", "interactive", "batch"])
    ap.add_argument("--preemptible", dest="preemptible", action="store_true", default=True)
    ap.add_argument("--no-preemptible", dest="preemptible", action="store_false")
    ap.add_argument("--max-retries", type=int, default=100, help="Iris failure/preemption retries per worker.")
    ap.add_argument("--no-bucket-tokens", dest="bucket_tokens", action="store_false", default=True)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-idle-passes", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="v2b", choices=["v1", "v2b"], help="v2b scores a_presurvivors -> b_keeplist.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without submitting.")
    args = ap.parse_args()

    submit_workers(args, build_command)


if __name__ == "__main__":
    main()
