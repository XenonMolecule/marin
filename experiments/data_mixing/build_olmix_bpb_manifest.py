# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn completed OLMIX swarm runs into a checkpoint manifest for the bpb eval launcher.

A swarm child writes ``metadata/olmix_swarm_results/<corpus>/<run_name>.json`` when its
training finishes, carrying the ``output_path`` its checkpoints live under. That JSON set is
the authoritative list of runs whose weights are exportable, so the eval manifest is derived
from it rather than from job state (an Iris "succeeded" does not imply the HF export landed).

Emits the ``run_name,region,output_path`` CSV that
``scaling_law_sweeps.olmo_bpb.launch_olmo_bpb_manifest`` reads.

    python -m experiments.data_mixing.build_olmix_bpb_manifest \\
        --corpus dclm_10k --corpus high_quality_10k --region us-east5 \\
        --out experiments/core_eval_manifests/olmix_swarm_bpb.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging

import fsspec

from experiments.data_mixing.run_olmix_swarm_standalone import DEFAULT_RESULTS_PREFIX, PROXY_TRAIN_STEPS
from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logger = logging.getLogger(__name__)

# Where swarm bpb results go. Deliberately NOT `olmo_bpb_results/` -- that namespace holds the
# curation sweeps' canonical numbers and a 363-run side sweep must not collide with it.
SWARM_BPB_ROOT = "metadata/olmix_swarm_bpb"
SWARM_BPB_SUBPATH = f"{SWARM_BPB_ROOT}/{{run_name}}/"


def completed_runs(bucket: str, corpus: str) -> list[dict]:
    """Every results JSON for one corpus, oldest-completed first."""
    prefix = f"{bucket}/{DEFAULT_RESULTS_PREFIX.format(corpus=corpus)}"
    fs, root = fsspec.core.url_to_fs(prefix)
    if not fs.exists(root):
        return []
    summaries = []
    for path in sorted(fs.ls(root, detail=False)):
        if not path.endswith(".json"):
            continue
        with fs.open(path) as fh:
            summaries.append(json.load(fh))
    return summaries


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", action="append", required=True, help="Repeatable.")
    p.add_argument(
        "--region",
        action="append",
        required=True,
        choices=sorted(REGION_TO_BUCKET),
        help="Repeatable. The swarm is split across regions, and results are bucket-local, so a "
        "single region silently yields ZERO rows for a corpus running elsewhere rather than "
        "failing -- pass every region the corpus runs in.",
    )
    p.add_argument("--out", required=True, help="CSV path to write.")
    p.add_argument(
        "--require-train-steps",
        type=int,
        default=PROXY_TRAIN_STEPS,
        help="Skip runs trained for a different number of steps. A smoke or dev run launched "
        "without a distinct --results-prefix writes a summary under the canonical run name; "
        "its short checkpoint must never enter the mixture objective.",
    )
    args = p.parse_args()

    rows: list[dict] = []
    seen: set[str] = set()
    for corpus in args.corpus:
        found = 0
        for region in args.region:
            summaries = completed_runs(REGION_TO_BUCKET[region], corpus)
            logger.info("%s in %s: %d completed runs", corpus, region, len(summaries))
            for summary in summaries:
                if summary["train_steps"] != args.require_train_steps:
                    logger.warning(
                        "SKIP %s: train_steps=%d, expected %d",
                        summary["run_name"],
                        summary["train_steps"],
                        args.require_train_steps,
                    )
                    continue
                # A run completes in exactly one region, but deduping keeps a repeated
                # --region from double-listing a checkpoint and launching two evals for it.
                if summary["run_name"] in seen:
                    continue
                seen.add(summary["run_name"])
                found += 1
                rows.append(
                    {
                        "run_name": summary["run_name"],
                        "region": summary["region"],
                        "output_path": summary["output_path"],
                    }
                )
        if not found:
            logger.warning(
                "%s: no completed runs in any of %s -- if it is running elsewhere, its "
                "checkpoints are being silently omitted from this manifest",
                corpus,
                args.region,
            )

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_name", "region", "output_path"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %d rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
