# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics for the olmix bpb objective: per-task sanity, spread, and reproducibility.

The mixture objective is a flat mean of one log-linear regression **per task**, so a
task that fails to load, returns a nonsense bpb, or is not reproducible corrupts the
objective directly. This script reads the ``results.json`` files a bpb sweep wrote and
reports three things:

**Sanity (actionable).** bpb must be finite, positive, and physically plausible. A NaN,
an inf, a zero, or a value outside :data:`PLAUSIBLE_BPB_RANGE` is a **bug in the task**
-- report and fix it. Missing tasks are equally actionable.

**Spread (context only, NOT a pruning signal).** Per-task mean/std/range across
checkpoints, and each task's spread relative to the suite macro average's spread.
Read this with a large caveat: these checkpoints vary by **model size and curation
method**, while the real swarm varies by **topic/quality mixture at fixed size and
token budget**. Those are different axes. A task that looks flat here may separate
mixtures strongly, and a task that separates model sizes here may be measuring scale
rather than mixture. Nothing in this report justifies dropping a task.

**Reproducibility (actionable).** With ``--repeat-dir``, the same checkpoint scored
twice must give identical bpb. Any nonzero difference means the scoring path is not
deterministic, which would show up as regression noise across all 363 proxy models.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os

import numpy as np

from experiments.data_mixing.olmix_tasks import build_target_tasks, task_family

logger = logging.getLogger(__name__)

# Hard bounds on a gold-continuation bpb, chosen so a hit means a real defect.
#
# Upper 8.0: a uniform prior over byte values costs exactly 8 bits/byte, so a model
# scoring worse than that is not modelling the continuation at all.
# Lower 0.1: a continuation this cheap is essentially memorised or empty.
#
# Deliberately wide. bpb normalises by continuation *bytes*, so tasks whose gold answer
# is a few bytes long sit far above prose: measured over 157 existing checkpoints,
# `drop/bpb_5shot` (gold answers ~10 bytes) spans 4.47-6.93 and the `basic_skills_*`
# tasks reach 4.8, while `mt_mbpp_java` goes down to 0.48. Those are the metric's normal
# behaviour, not defects, and a tighter band would just cry wolf on them.
PLAUSIBLE_BPB_RANGE = (0.1, 8.0)


def load_results(results_root: str) -> dict[str, dict[str, float]]:
    """``{run_name: {task_variant: bpb}}`` from ``<root>/<run_name>/results.json``."""
    import fsspec

    fs, root = fsspec.core.url_to_fs(results_root.rstrip("/"))
    out: dict[str, dict[str, float]] = {}
    for run_dir in fs.ls(root, detail=False):
        run_name = os.path.basename(run_dir.rstrip("/"))
        path = f"{run_dir.rstrip('/')}/results.json"
        if not fs.exists(path):
            logger.warning("no results.json for %s", run_name)
            continue
        with fs.open(path, "r") as f:
            payload = json.load(f)
        out[run_name] = {task: res["bpb"] for task, res in payload["tasks"].items()}
    return out


def sanity_report(results: dict[str, dict[str, float]], tasks: tuple[str, ...]) -> list[str]:
    """Hard problems only: missing tasks and physically impossible bpb values."""
    problems: list[str] = []
    for run_name, task_bpb in sorted(results.items()):
        missing = [t for t in tasks if t not in task_bpb]
        if missing:
            problems.append(f"{run_name}: MISSING {len(missing)} task(s): {missing}")
        for task in tasks:
            value = task_bpb.get(task)
            if value is None:
                continue
            if not math.isfinite(value):
                problems.append(f"{run_name}/{task}: non-finite bpb {value!r}")
            elif value <= 0:
                problems.append(f"{run_name}/{task}: non-positive bpb {value!r}")
            elif not PLAUSIBLE_BPB_RANGE[0] <= value <= PLAUSIBLE_BPB_RANGE[1]:
                problems.append(f"{run_name}/{task}: bpb {value:.4f} outside {PLAUSIBLE_BPB_RANGE}")
    return problems


def spread_table(results: dict[str, dict[str, float]], tasks: tuple[str, ...]) -> list[dict]:
    """Per-task spread across checkpoints, plus correlation with the suite macro average.

    ``spread_ratio`` is the task's coefficient of variation divided by the macro
    average's, i.e. how much a task moves relative to how much the whole suite moves.
    """
    runs = sorted(results)
    matrix = np.array([[results[r].get(t, np.nan) for t in tasks] for r in runs], dtype=float)
    macro = np.nanmean(matrix, axis=1)
    macro_cv = float(np.std(macro) / np.mean(macro))

    rows: list[dict] = []
    for i, task in enumerate(tasks):
        column = matrix[:, i]
        finite = column[np.isfinite(column)]
        mean = float(np.mean(finite)) if finite.size else float("nan")
        std = float(np.std(finite)) if finite.size else float("nan")
        cv = std / mean if finite.size and mean else float("nan")
        # Correlation with the macro average: negative means the task disagrees with the
        # suite about which checkpoint is better. Worth knowing, not inherently wrong.
        corr = float("nan")
        if finite.size == column.size and std > 0:
            corr = float(np.corrcoef(column, macro)[0, 1])
        rows.append(
            {
                "task": task,
                "family": task_family(task),
                "n_runs": int(finite.size),
                "mean_bpb": mean,
                "std_bpb": std,
                "min_bpb": float(np.min(finite)) if finite.size else float("nan"),
                "max_bpb": float(np.max(finite)) if finite.size else float("nan"),
                "cv": cv,
                "spread_ratio": cv / macro_cv if macro_cv else float("nan"),
                "corr_with_macro": corr,
            }
        )
    return rows


def reproducibility_report(
    primary: dict[str, dict[str, float]], repeat: dict[str, dict[str, float]], tasks: tuple[str, ...]
) -> list[str]:
    """Per-task |primary - repeat| for every checkpoint scored twice."""
    lines: list[str] = []
    shared = sorted(set(primary) & set(repeat))
    if not shared:
        return ["no checkpoint appears in both result sets"]
    for run_name in shared:
        diffs = {
            t: abs(primary[run_name][t] - repeat[run_name][t])
            for t in tasks
            if t in primary[run_name] and t in repeat[run_name]
        }
        if not diffs:
            lines.append(f"{run_name}: no overlapping tasks")
            continue
        worst_task = max(diffs, key=lambda k: diffs[k])
        lines.append(
            f"{run_name}: {len(diffs)} tasks compared, max |diff| = {diffs[worst_task]:.3e} ({worst_task}), "
            f"exact matches = {sum(1 for d in diffs.values() if d == 0.0)}/{len(diffs)}"
        )
    return lines


def format_table(rows: list[dict]) -> str:
    header = (
        f"{'task':52s} {'fam':5s} {'n':>2s} {'mean':>7s} {'std':>8s} "
        f"{'min':>7s} {'max':>7s} {'cv':>7s} {'sprd':>6s} {'r_macro':>8s}"
    )
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda r: r["cv"] if math.isfinite(r["cv"]) else -1):
        lines.append(
            f"{row['task']:52s} {row['family']:5s} {row['n_runs']:2d} "
            f"{row['mean_bpb']:7.4f} {row['std_bpb']:8.5f} {row['min_bpb']:7.4f} {row['max_bpb']:7.4f} "
            f"{row['cv']:7.4f} {row['spread_ratio']:6.2f} {row['corr_with_macro']:8.3f}"
        )
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="Dir of <run_name>/results.json (local or gs://).")
    ap.add_argument("--repeat-dir", default=None, help="Second scoring of some checkpoints, for reproducibility.")
    ap.add_argument("--json-out", default=None, help="Write the spread table here as JSON.")
    args = ap.parse_args()

    tasks = build_target_tasks()
    results = load_results(args.results_dir)
    logger.info("Loaded %d runs x %d target tasks", len(results), len(tasks))

    print("\n=== SANITY (actionable: a hit here is a task bug, not a pruning signal) ===")
    problems = sanity_report(results, tasks)
    print("\n".join(problems) if problems else "no missing tasks, no non-finite/non-positive/implausible bpb")

    rows = spread_table(results, tasks)
    print("\n=== SPREAD ACROSS CHECKPOINTS (diagnostic only; wrong axis -- see module docstring) ===")
    print(format_table(rows))

    if args.repeat_dir:
        print("\n=== REPRODUCIBILITY (actionable: must be exact) ===")
        print("\n".join(reproducibility_report(results, load_results(args.repeat_dir), tasks)))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"tasks": rows, "problems": problems, "runs": sorted(results)}, f, indent=2)
        logger.info("Wrote %s", args.json_out)


if __name__ == "__main__":
    main()
