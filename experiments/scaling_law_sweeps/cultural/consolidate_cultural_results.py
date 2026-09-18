# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate every cultural-suite results.json (5 buckets) into ONE long-format CSV.

Sibling of `mmlu/consolidate_mmlu_results.py`. Each results.json holds one row per task,
keyed `cultural_sl_verb_<task>_<N>shot`, carrying the seven sl_verb metrics. This emits
three levels per run:

  * `task`      — the 19 task rows as scored;
  * `benchmark` — blend = unweighted mean over its 16 country tasks, the other three
                  benchmarks are their single task;
  * `macro`     — unweighted mean over the four benchmarks (the suite headline).

Because chance differs by benchmark (25% for MC-4, 50% for 2-way), every level also
carries `acc_centered = (acc - 1/k) / (1 - 1/k)`, computed per task before averaging.
`n_tasks` counts the tasks that actually scored; a benchmark/macro row built from fewer
than expected is flagged so a partial run cannot masquerade as a complete one.

Runs as a CPU iris job so the per-run results.json (samples logged for every document)
are read in-cloud:

    iris --cluster marin job run --region us-central1 --cpu 2 --memory 16GB \\
        --enable-extra-resources \\
        -- python -m experiments.scaling_law_sweeps.cultural.consolidate_cultural_results --shots 5
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
from statistics import fmean

from rigging.filesystem import filesystem as marin_filesystem

from experiments.scaling_law_sweeps.core_tasks.consolidate_results import (
    BUCKETS,
    RUN_RE,
    norm_method,
    tokens_for,
)
from experiments.scaling_law_sweeps.cultural.cultural_tasks_set import (
    BENCHMARKS,
    METRIC_NAMES,
    TASK_SPECS,
    TASKS_BY_BENCHMARK,
    result_key,
)

logger = logging.getLogger(__name__)

RESULTS_PREFIX = "metadata/cultural_sl_verb_results"
ACC_CENTERED = "acc_centered"
COLUMNS = (*METRIC_NAMES, ACC_CENTERED)
FIELDS = [
    "run_stem",
    "method",
    "hidden_dim",
    "num_layers",
    "budget",
    "tokens",
    "num_fewshot",
    "level",
    "name",
    *COLUMNS,
    "n_tasks",
    "n_tasks_expected",
]


def centered_accuracy(acc: float, num_choices: int) -> float:
    chance = 1.0 / num_choices
    return (acc - chance) / (1.0 - chance)


def task_metrics(doc: dict, shots: int) -> dict[str, dict[str, float]]:
    """Per-task metric dicts for every task row present in `doc["results"]`."""
    out: dict[str, dict[str, float]] = {}
    results = doc.get("results", {})
    for spec in TASK_SPECS:
        row = results.get(result_key(spec.name, shots))
        if not isinstance(row, dict):
            continue
        metrics = {}
        for m in METRIC_NAMES:
            v = row.get(f"{m},none")
            if isinstance(v, (int, float)):
                metrics[m] = float(v)
        if "acc" in metrics:
            metrics[ACC_CENTERED] = centered_accuracy(metrics["acc"], spec.num_choices)
        out[spec.name] = metrics
    return out


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {c: fmean(r[c] for r in rows) for c in COLUMNS if all(c in r for r in rows)}


def aggregate(per_task: dict[str, dict[str, float]]) -> list[dict]:
    """Task, benchmark and macro rows (each a dict with level/name/metrics/n_tasks)."""
    out: list[dict] = []
    for name, metrics in per_task.items():
        out.append(dict(level="task", name=name, n_tasks=1, n_tasks_expected=1, **metrics))
    benchmark_rows: list[dict[str, float]] = []
    for b in BENCHMARKS:
        present = [per_task[t] for t in TASKS_BY_BENCHMARK[b] if t in per_task]
        if not present:
            continue
        metrics = _mean_rows(present)
        out.append(
            dict(level="benchmark", name=b, n_tasks=len(present), n_tasks_expected=len(TASKS_BY_BENCHMARK[b]), **metrics)
        )
        benchmark_rows.append(metrics)
    if benchmark_rows:
        out.append(
            dict(
                level="macro",
                name="cultural_macro",
                n_tasks=len(benchmark_rows),
                n_tasks_expected=len(BENCHMARKS),
                **_mean_rows(benchmark_rows),
            )
        )
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--out", default=None, help="Consolidated CSV to write. Default: derived from --shots.")
    args = ap.parse_args()
    out = args.out or f"gs://marin-us-central1/metadata/cultural_sl_verb_{args.shots}shot_summary.csv"

    fs = marin_filesystem("gcs")
    rows: list[dict] = []
    n_runs = 0
    for b in BUCKETS:
        for path in fs.glob(f"gs://{b}/{RESULTS_PREFIX}/{args.shots}shot/*/results.json"):
            run = path.split(f"/{args.shots}shot/")[1].split("/")[0]
            m = RUN_RE.match(run)
            if not m:
                logger.warning("unparsed run: %s", run)
                continue
            with fs.open(path if path.startswith("gs://") else f"gs://{path}", "r") as f:
                per_task = task_metrics(json.load(f), args.shots)
            if not per_task:
                logger.warning("no %dshot task rows in %s", args.shots, run)
                continue
            n_runs += 1
            d, L = int(m["d"]), int(m["L"])
            base = dict(
                run_stem=run,
                method=norm_method(m["method"]),
                hidden_dim=d,
                num_layers=L,
                budget=m["budget"],
                tokens=tokens_for(m["budget"], d, L),
                num_fewshot=args.shots,
            )
            for agg in aggregate(per_task):
                rows.append({**base, **agg})
    logger.info("consolidated %d runs into %d rows", n_runs, len(rows))

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, restval="")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["method"], r["hidden_dim"], r["tokens"] or 0, r["level"], r["name"])))
    with fs.open(out, "w") as f:
        f.write(buf.getvalue())
    logger.info("wrote %s", out)

    short = [(r["run_stem"], r["name"]) for r in rows if r["n_tasks"] < r["n_tasks_expected"]]
    if short:
        logger.warning("%d benchmark/macro rows built from missing tasks (first 5): %s", len(short), short[:5])


if __name__ == "__main__":
    main()
