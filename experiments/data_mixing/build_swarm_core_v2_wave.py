# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit the next DCLM Core v2 eval wave over the OLMIX swarm: everything not yet done.

Core v2 children fail transiently often enough that the fleet never completes in one pass
-- HF Hub throttling surfaces as ``ValueError: Failed to load task {...}`` and kills a
child mid-run. Recovery is cheap (per-task partials survive, so a resubmitted child resumes
at the failed task), but it has to actually be driven. This is that driver: run it, launch
what it emits, repeat until it reports 0.

Two traps it encodes, both of which have already cost us work:

* **``--skip-existing`` cannot see in-flight children.** It only checks for a finished
  ``_summary.json``, so a run that is queued or mid-eval elsewhere looks un-evaluated and
  gets submitted twice. Pass the currently-pending run names via ``--exclude``.
* **us-west4 has only ``v5litepod-*``.** A child asking for the default variants is rejected
  at submit time, so that region gets its own manifest and its own wave.

Usage::

    python -m experiments.data_mixing.build_swarm_core_v2_wave --tag w2
    # then launch each emitted manifest with launch_10k_manifest (command is printed)
"""

from __future__ import annotations

import argparse
import csv
import logging

import fsspec

from experiments.data_mixing.collect_swarm_core_v2 import SWARM_CORE_V2_ROOT, parse_run_name
from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logger = logging.getLogger(__name__)

REGIONS = ("us-east5", "us-central1", "europe-west4", "us-west4")
CHECKPOINT_PREFIX = "checkpoints/olmix-swarm"
# Region preference when a run was trained in more than one place. Matches the collector's
# dedupe order so the eval lands where the fit will look for it.
REGION_PREFERENCE = REGIONS
VARIANT_BY_GROUP = {"usw4": "v5litepod-4", "rest": None}


def trained_runs(regions: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """``{run_name: (region, output_path)}`` for every swarm checkpoint dir.

    Deliberately does *not* probe for an ``hf/`` export: that is one GCS round trip per run
    (~818 of them, minutes of wall time) to re-derive something the launcher already checks.
    ``_resolve_final_step`` returns None for a missing export and the launcher logs
    ``NO HF STEP found`` and files it under ``incomplete``.
    """
    found: dict[str, tuple[str, str]] = {}
    for region in regions:
        bucket = REGION_TO_BUCKET[region]
        fs, root = fsspec.core.url_to_fs(f"{bucket}/{CHECKPOINT_PREFIX}")
        if not fs.exists(root):
            continue
        for run_dir in fs.ls(root, detail=False):
            run_name = run_dir.rstrip("/").rsplit("/", 1)[-1]
            if parse_run_name(run_name) is None:
                continue
            prior = found.get(run_name)
            output_path = f"{bucket}/{CHECKPOINT_PREFIX}/{run_name}"
            if prior is None or REGION_PREFERENCE.index(region) < REGION_PREFERENCE.index(prior[0]):
                found[run_name] = (region, output_path)
    return found


def evaluated_runs(regions: tuple[str, ...]) -> set[str]:
    """Runs with a Core v2 scores summary in any region."""
    done: set[str] = set()
    for region in regions:
        bucket = REGION_TO_BUCKET[region]
        fs, root = fsspec.core.url_to_fs(f"{bucket}/{SWARM_CORE_V2_ROOT}")
        if not fs.exists(root):
            continue
        for path in fs.ls(root, detail=False):
            leaf = path.rstrip("/").rsplit("/", 1)[-1]
            if leaf.endswith("_summary.json"):
                done.add(leaf[: -len("_summary.json")])
    return done


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default="experiments/core_eval_manifests")
    p.add_argument("--tag", default="w2", help="Wave name; also the manifest filename stem.")
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

    trained = trained_runs(REGIONS)
    done = evaluated_runs(REGIONS)
    todo = {name: loc for name, loc in trained.items() if name not in done and name not in inflight}

    logger.info(
        "trained=%d evaluated=%d in_flight=%d -> needs_eval=%d",
        len(trained),
        len(done),
        len(inflight),
        len(todo),
    )
    by_region: dict[str, int] = {}
    for _, (region, _) in todo.items():
        by_region[region] = by_region.get(region, 0) + 1
    logger.info("needs_eval by region: %s", by_region or "{}")

    if not todo:
        logger.info("Nothing to do -- every trained run has a Core v2 summary.")
        return

    groups: dict[str, list[tuple[str, str, str]]] = {"usw4": [], "rest": []}
    for name, (region, output_path) in sorted(todo.items()):
        groups["usw4" if region == "us-west4" else "rest"].append((name, region, output_path))

    for group, rows in groups.items():
        if not rows:
            continue
        out = f"{args.out_dir.rstrip('/')}/olmix_swarm_core_v2_{args.tag}_{group}.txt"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_name", "region", "output_path"])
            w.writerows(rows)
        variant = VARIANT_BY_GROUP[group]
        logger.info("wrote %s (%d rows)", out, len(rows))
        logger.info(
            "  launch: python -m experiments.scaling_law_sweeps.dclm_core.launch_10k_manifest "
            "--manifest %s "
            "--scores-subpath 'metadata/olmix_swarm_core_v2/{run_name}_summary.json' "
            "--samples-subpath 'tmp/ttl=30d/olmix_swarm_core_v2/{run_name}.json' "
            # --name-suffix=-w2, NOT --name-suffix -w2: a value starting with '-' is parsed
            # as a flag and argparse fails with "expected one argument".
            "%s--launch --child-priority batch --wave-size 8 --wave-delay 120 --name-suffix=-%s",
            out,
            f"--tpu-variants {variant} " if variant else "",
            args.tag,
        )


if __name__ == "__main__":
    main()
