# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the manifests for the eval wave that closes out the swarm.

Two facts about this sweep make a naive "just re-run the launcher" wrong, and both cost real
hours when learned the hard way:

* **A wave must exclude what other waves already have in flight.** `--skip-existing` only sees
  runs whose ``results.json`` has landed, so a run that is queued or mid-eval elsewhere looks
  un-evaluated and gets submitted twice. Earlier waves duplicated work exactly this way.
* **us-west4 needs a different TPU variant.** The launcher's default variants
  (``v5p-8 / v4-8 / v6e-4``) do not exist there -- it only has ``v5litepod-*`` -- and a child
  asking for an absent variant is rejected at SUBMIT time, then swallowed into a counter that
  reads like ordinary lag. 132 evals sat unrunnable behind that. So us-west4 gets its own
  manifest and its own wave.

Emits one manifest per region-group and prints the exact launch commands. Splitting by group
rather than emitting one file is deliberate: ``--tpu-variants`` applies to a whole wave.

    python -m experiments.data_mixing.build_final_eval_wave --out-dir experiments/core_eval_manifests
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter

import fsspec

from experiments.data_mixing.build_olmix_bpb_manifest import SWARM_BPB_ROOT
from experiments.data_mixing.run_olmix_swarm_standalone import DEFAULT_RESULTS_PREFIX
from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logger = logging.getLogger(__name__)

CORPORA = ("dclm_10k", "high_quality_10k")
REGIONS = ("us-east5", "us-central1", "europe-west4", "us-west4")

# us-west4 has only v5litepod-*; everything else has the launcher's defaults.
VARIANT_BY_GROUP = {"usw4": "v5litepod-4", "rest": None}


def trained_runs(regions: tuple[str, ...], corpora: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """run_name -> (region, output_path) for every run whose training results JSON exists."""
    found: dict[str, tuple[str, str]] = {}
    for region in regions:
        bucket = REGION_TO_BUCKET[region]
        for corpus in corpora:
            prefix = f"{bucket}/{DEFAULT_RESULTS_PREFIX.format(corpus=corpus)}"
            fs, root = fsspec.core.url_to_fs(prefix)
            if not fs.exists(root):
                continue
            for path in fs.ls(root, detail=False):
                if not path.endswith(".json"):
                    continue
                with fs.open(path) as fh:
                    summary = json.load(fh)
                run_name = summary["run_name"]
                # A run duplicated across regions keeps its first sighting, matching how the
                # collector dedupes, so the eval lands where the fit will look for it.
                found.setdefault(run_name, (region, summary["output_path"].rstrip("/")))
    return found


def evaluated_runs(regions: tuple[str, ...]) -> set[str]:
    """Runs with a completed bpb results.json, checked in every region."""
    done: set[str] = set()
    for region in regions:
        bucket = REGION_TO_BUCKET[region]
        fs, root = fsspec.core.url_to_fs(f"{bucket}/{SWARM_BPB_ROOT}")
        if not fs.exists(root):
            continue
        for run_dir in fs.ls(root, detail=False):
            if fs.exists(f"{run_dir.rstrip('/')}/results.json"):
                done.add(run_dir.rstrip("/").rsplit("/", 1)[-1])
    return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default="experiments/core_eval_manifests")
    p.add_argument("--tag", default="w12", help="Wave name; also the manifest filename stem.")
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="File of run names already in flight in another wave (repeatable). REQUIRED "
        "whenever another wave has pending children, or those runs get evaluated twice.",
    )
    args = p.parse_args()

    inflight: set[str] = set()
    for path in args.exclude:
        with open(path) as fh:
            inflight |= {line.strip() for line in fh if line.strip()}
    logger.info("excluding %d runs already in flight elsewhere", len(inflight))

    trained = trained_runs(REGIONS, CORPORA)
    done = evaluated_runs(REGIONS)
    todo = {name: loc for name, loc in trained.items() if name not in done and name not in inflight}
    logger.info(
        "trained=%d evaluated=%d in_flight=%d -> needs_eval=%d",
        len(trained),
        len(done),
        len(inflight),
        len(todo),
    )
    logger.info("needs_eval by region: %s", dict(Counter(r for r, _ in todo.values())))

    groups = {"usw4": [], "rest": []}
    for name, (region, output_path) in sorted(todo.items()):
        groups["usw4" if region == "us-west4" else "rest"].append((name, region, output_path))

    for group, rows in groups.items():
        if not rows:
            logger.info("group %s: nothing to do", group)
            continue
        out = f"{args.out_dir.rstrip('/')}/olmix_swarm_bpb_{args.tag}_{group}.txt"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run_name", "region", "output_path"])
            w.writerows(rows)
        variant = VARIANT_BY_GROUP[group]
        logger.info("group %s: %d rows -> %s", group, len(rows), out)
        logger.info(
            "  launch with: --manifest %s%s",
            out,
            f" --tpu-variants {variant}" if variant else " (default variants)",
        )


if __name__ == "__main__":
    main()
