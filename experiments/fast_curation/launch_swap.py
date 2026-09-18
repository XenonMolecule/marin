# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a fleet of ``swap_extractor`` workers (arm B of the extractor ablation).

Workers refetch WARCs from Common Crawl (free ingress) and write to us-east5 only, so the fleet
runs in us-east5. resiliparse is a core marin dependency: the ``cpu`` extra is all a worker needs.

    python -m experiments.fast_curation.launch_swap --num-workers 60 --priority batch
    python -m experiments.fast_curation.launch_swap --num-workers 1 --limit 1   # canary
"""

from __future__ import annotations

import argparse
import logging
import os

from experiments.fast_curation._launch_common import submit_workers
from experiments.fast_curation.swap_extractor import OUTPUT_ROOT

logger = logging.getLogger(__name__)

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
        "--extra",
        "cpu",
        "--priority",
        args.priority,
        "--preemptible" if args.preemptible else "--no-preemptible",
        "--max-retries",
        str(args.max_retries),
        "--no-wait",
        "--job-name",
        f"fastcur-swap-{seed}",
        "-e",
        "HF_TOKEN",
        HF_TOKEN,
        "--",
        "python",
        "-m",
        "experiments.fast_curation.swap_extractor",
        "--manifest",
        args.manifest,
        "--root",
        args.root,
        "--shuffle-seed",
        str(seed),
        "--extract-procs",
        str(args.extract_procs),
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-idle-passes",
        str(args.max_idle_passes),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.start:
        cmd += ["--start", str(args.start)]
    return cmd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--root", default=OUTPUT_ROOT)
    ap.add_argument("--region", default="us-east5")
    ap.add_argument("--cluster", default="marin")
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--memory", default="16GB")
    ap.add_argument("--disk", default="24GB")
    ap.add_argument("--extract-procs", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--priority", default="batch", choices=["interactive", "batch"])
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
