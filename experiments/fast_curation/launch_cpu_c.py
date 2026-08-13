# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch v2 Phase C (CPU): N standalone JustText workers over the ModernBERT-kept docs.

Phase C runs JustText (the dominant cost) only on docs that passed ModernBERT (~5x fewer than v1),
fanned out across the worker's cores. Each worker reads per-WARC pre-survivor html + Phase-B
keeplist and writes the final ``kept/`` parquet.

    python -m experiments.fast_curation.launch_cpu_c --num-workers 100 --priority batch
"""

from __future__ import annotations

import argparse
import logging
import os

from experiments.fast_curation._launch_common import submit_workers

logger = logging.getLogger(__name__)

# Phase C needs justext (extraction-bakeoff). No fasttext/tokenizer.
CPU_C_EXTRAS = ["cpu", "extraction-bakeoff"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]


def build_command(args: argparse.Namespace, seed: int) -> list[str]:
    cmd = [
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
        "--enable-extra-resources",
        "--priority",
        args.priority,
        "--preemptible" if args.preemptible else "--no-preemptible",
        "--max-retries",
        str(args.max_retries),
        "--no-wait",
        "--job-name",
        f"fastcur-c-{args.spec}-{seed}",
        "-e",
        "WANDB_API_KEY",
        WANDB_API_KEY,
        "-e",
        "HF_TOKEN",
        HF_TOKEN,
    ]
    for extra in CPU_C_EXTRAS:
        cmd += ["--extra", extra]
    cmd += [
        "--",
        "python",
        "-m",
        "experiments.fast_curation.cpu_phase_c",
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
    if args.resiliparse_artifact is not None:
        cmd += ["--resiliparse-artifact", args.resiliparse_artifact]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.start:
        cmd += ["--start", str(args.start)]
    return cmd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v3")
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--bucket", default="gs://marin-us-east5")
    ap.add_argument("--region", default="us-east5")
    ap.add_argument("--cluster", default="marin")
    ap.add_argument("--cpu", type=float, default=8, help="Cores per worker; JustText fans out across them.")
    ap.add_argument("--memory", default="32GB")
    ap.add_argument("--disk", default="20GB")
    ap.add_argument("--justext-procs", type=int, default=None, help="Extraction ProcessPool size (default = --cpu).")
    ap.add_argument(
        "--resiliparse-artifact",
        default=None,
        help="Prebuilt resiliparse-rs artifact prefix; pass a same-region mirror when Phase C runs "
        "outside us-east5 so workers don't each read the artifact cross-region. Only used by specs "
        "whose extractor is resiliparse_rs; omitted => the spec module's default.",
    )
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

    submit_workers(args, build_command)


if __name__ == "__main__":
    main()
