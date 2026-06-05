# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Port of DCLM's `eval/aggregated_metrics.py` centering + CORE aggregation.

Source: https://github.com/mlfoundations/dclm/blob/main/eval/aggregated_metrics.py
This is a faithful pure-Python rewrite (no pandas dependency) of the v2
aggregation. Numbers must match upstream when fed identical raw per-task
results.

DCLM's "CORE" score is the mean of the centered scores over the 22 tasks
listed under `low_variance_datasets` in `additional_aggregation.json`.
Centering per task:

    centered_i = (raw_i - 0.01 * baseline_i) / (1.0 - 0.01 * baseline_i)

where `baseline_i` is the per-task random-baseline accuracy (as a percent,
0-100) read from `eval_meta_data.csv`.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

CURRENT_VERSION = "v2"

# Filenames colocated with this module.
_HERE = Path(__file__).parent
DEFAULT_META_CSV = _HERE / "eval_meta_data.csv"
DEFAULT_AGGREGATION_JSON = _HERE / "additional_aggregation.json"


@dataclass(frozen=True)
class TaskMeta:
    name: str
    category: str
    task_type: str
    shots: int
    n_datapoints: int
    random_baseline: float  # percent, 0-100


def load_meta(csv_path: Path = DEFAULT_META_CSV) -> dict[str, TaskMeta]:
    """Load eval_meta_data.csv as a dict keyed by task name."""
    out: dict[str, TaskMeta] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Eval Task"]
            out[name] = TaskMeta(
                name=name,
                category=row["Task Category"],
                task_type=row["Task Type"],
                shots=int(row["#shots"]),
                n_datapoints=int(row["#datapoints"]),
                random_baseline=float(row["Random baseline"]),
            )
    return out


def load_aggregation(json_path: Path = DEFAULT_AGGREGATION_JSON) -> dict[str, list[str]]:
    """Load additional_aggregation.json — the task-set definitions."""
    with json_path.open() as f:
        return json.load(f)


def center(raw: float, baseline_pct: float) -> float:
    """DCLM centering: maps random-baseline -> 0, perfect -> 1."""
    b = 0.01 * baseline_pct
    return (raw - b) / (1.0 - b)


def compute_core(
    raw_results: dict[str, float],
    *,
    meta: dict[str, TaskMeta] | None = None,
    aggregation: dict[str, list[str]] | None = None,
    version: str = CURRENT_VERSION,
) -> dict:
    """Compute the full DCLM aggregation from a raw per-task results dict.

    `raw_results` is a flat mapping {task_name: metric_value}. Tasks not
    present in `meta` are ignored; tasks present in `meta` but missing
    from `raw_results` are reported under `missing_tasks`.

    Returns a dict shaped like DCLM's `eval_metrics_results.json`:
        {
            "raw_results": {task: value, ...},
            "centered_results": {task: value, ...},
            "aggregated_task_categories_centered": {category: mean, ...},
            "aggregated_results": float,            # mean of raw over ALL meta tasks
            "aggregated_centered_results": float,   # mean of centered over ALL meta tasks
            "Core": float | str,                    # mean centered over low_variance_datasets
            "Core_v2": float | str,
            "Extended": float | str,
            "Extended_v2": float | str,
            "missing_tasks": [...],
            "eval_version": version,
        }
    """
    if meta is None:
        meta = load_meta()
    if aggregation is None:
        aggregation = load_aggregation()

    missing = [t for t in meta if t not in raw_results]

    # Per-task centered, only for tasks we have raw values for.
    centered: dict[str, float] = {}
    for task, raw in raw_results.items():
        if task not in meta:
            continue
        centered[task] = center(float(raw), meta[task].random_baseline)

    out: dict = {
        "raw_results": {t: float(raw_results[t]) for t in raw_results if t in meta},
        "centered_results": centered,
        "missing_tasks": missing,
        "eval_version": version,
    }

    # Per-category centered mean (uses tasks present in centered).
    by_category: dict[str, list[float]] = {}
    for task, value in centered.items():
        cat = meta[task].category
        by_category.setdefault(cat, []).append(value)
    out["aggregated_task_categories_centered"] = {
        cat: statistics.fmean(vals) for cat, vals in by_category.items() if vals
    }

    # All-task averages (matches DCLM's "Extended" set when no missing tasks).
    all_raw = [float(raw_results[t]) for t in meta if t in raw_results]
    out["aggregated_results"] = statistics.fmean(all_raw) if all_raw else float("nan")
    all_centered = list(centered.values())
    out["aggregated_centered_results"] = statistics.fmean(all_centered) if all_centered else float("nan")

    # Named aggregations from additional_aggregation.json.
    for key, tasks in aggregation.items():
        present_raw = [float(raw_results[t]) for t in tasks if t in raw_results]
        present_centered = [centered[t] for t in tasks if t in centered]
        out[key] = statistics.fmean(present_raw) if present_raw else float("nan")
        out[f"{key}_centered"] = statistics.fmean(present_centered) if present_centered else float("nan")

    # CORE = low_variance_datasets_centered, but N/A if any of the 22 are missing.
    core_tasks = aggregation.get("low_variance_datasets", [])
    missing_for_core = [t for t in core_tasks if t not in raw_results]
    core_value: float | str
    if missing_for_core:
        core_value = f"N/A due to missing tasks: {missing_for_core}"
    else:
        core_value = out["low_variance_datasets_centered"]
    out[f"Core_{version}"] = core_value
    out["Core"] = core_value
    out["missing_tasks_for_core"] = missing_for_core

    # Extended = full-set centered mean, but N/A if any meta task is missing.
    if missing:
        ext_value: float | str = f"N/A due to missing tasks: {missing}"
    else:
        ext_value = out["aggregated_centered_results"]
    out[f"Extended_{version}"] = ext_value
    out["Extended"] = ext_value

    return out


__all__ = [
    "CURRENT_VERSION",
    "DEFAULT_AGGREGATION_JSON",
    "DEFAULT_META_CSV",
    "TaskMeta",
    "center",
    "compute_core",
    "load_aggregation",
    "load_meta",
]
