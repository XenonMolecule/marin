# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch v2 Phase A (CPU): N standalone decode->fastText->tokenize workers (NO JustText).

Phase A is cheap per WARC (~2-8 min: decode + fastText + tokenize), so it does NOT need many
cores per worker or the JustText extra. Launch a wide fleet of small workers.

    python -m experiments.fast_curation.launch_cpu_a --num-workers 50 --priority batch
"""

from __future__ import annotations

import argparse
import logging
import os

from experiments.fast_curation._launch_common import submit_workers

logger = logging.getLogger(__name__)

# Phase A only needs fasttext (dclm) + the tokenizer (transformers via core). NO extraction-bakeoff.
CPU_A_EXTRAS = ["cpu", "dclm"]
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
        f"fastcur-a-{args.spec}-{seed}",
        "-e",
        "WANDB_API_KEY",
        WANDB_API_KEY,
        "-e",
        "HF_TOKEN",
        HF_TOKEN,
    ]
    extras = [*CPU_A_EXTRAS, "gigatoken"] if args.tokenizer_impl == "gigatoken" else CPU_A_EXTRAS
    for extra in extras:
        cmd += ["--extra", extra]
    cmd += [
        "--",
        "python",
        "-m",
        "experiments.fast_curation.cpu_phase_a",
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
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.start:
        cmd += ["--start", str(args.start)]
    if args.rescue:
        cmd += ["--rescue", "--rescue-stale-minutes", str(args.rescue_stale_minutes)]
    # TEXT-line workers extract in Phase A; point them at a same-region artifact mirror and size
    # the extraction pool to the worker's cores (these flags are ignored by html-line specs).
    if args.resiliparse_artifact:
        cmd += ["--resiliparse-artifact", args.resiliparse_artifact]
    cmd += ["--extract-procs", str(args.extract_procs)]
    cmd += ["--tokenizer-impl", args.tokenizer_impl]
    if args.max_shard is not None:
        cmd += ["--max-shard", str(args.max_shard)]
    return cmd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v3")
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--bucket", default="gs://marin-us-east5")
    ap.add_argument("--region", default="us-east5")
    ap.add_argument("--cluster", default="marin")
    ap.add_argument("--cpu", type=float, default=4)
    # v2 Phase A holds the full decoded WARC (every record's raw html) + the survivor rows (raw
    # html again) before writing; 24GB OOM-killed ~all workers on big WARCs. 64GB gives headroom.
    ap.add_argument("--memory", default="64GB")
    ap.add_argument("--disk", default="24GB")
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
    ap.add_argument(
        "--rescue", action="store_true", help="Rescue mode: re-decode B-stuck WARCs into this region from S3."
    )
    ap.add_argument("--rescue-stale-minutes", type=float, default=20.0)
    ap.add_argument(
        "--resiliparse-artifact",
        default=None,
        help="TEXT line: prebuilt resiliparse-rs artifact prefix (same-region mirror outside us-east5). "
        "Default: the worker's canonical us-east5 artifact.",
    )
    ap.add_argument("--extract-procs", type=int, default=4, help="TEXT line: extraction ProcessPool size.")
    ap.add_argument(
        "--tokenizer-impl",
        default="hf",
        choices=["hf", "gigatoken"],
        help="TEXT line tokenizer implementation (ids byte-identical; gigatoken adds its extra and "
        "parity-asserts at worker startup). Launch part of a fleet with each to A/B tokenize_s.",
    )
    ap.add_argument("--max-shard", type=int, default=None, help="V3 ladder cap (shards < N).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    submit_workers(args, build_command)


if __name__ == "__main__":
    main()
