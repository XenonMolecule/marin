# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate SFT training + eval results into a flat table.

Joins by run_name across:
  - training summaries: `gs://marin-us-central1/metadata/medical_sft_base_results/*.json`
  - eval summaries: `gs://marin-us-central1/metadata/medical_sft_base_eval_results/*.json`
  - per-subtask raw eval results (read from the eval `output_path`)

Produces a CSV / Markdown table with one row per (model_size, domain, branch,
config_name) showing the MMLU-medical average + per-subtask scores + train HP.

Usage:

    .venv/bin/python experiments/rephraser/aggregate_sft_results.py
    .venv/bin/python experiments/rephraser/aggregate_sft_results.py --markdown
    .venv/bin/python experiments/rephraser/aggregate_sft_results.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import fsspec

logger = logging.getLogger(__name__)


DEFAULT_TRAINING_PREFIX = "gs://marin-us-central1/metadata/medical_sft_base_results/"
DEFAULT_EVAL_PREFIX = "gs://marin-us-central1/metadata/medical_sft_base_eval_results/"

# 7 MMLU subtasks the medical eval uses. The published table averages these.
MMLU_MEDICAL_SUBTASKS = [
    "mmlu_anatomy_generative_5shot",
    "mmlu_clinical_knowledge_generative_5shot",
    "mmlu_college_biology_generative_5shot",
    "mmlu_college_medicine_generative_5shot",
    "mmlu_high_school_biology_generative_5shot",
    "mmlu_medical_genetics_generative_5shot",
    "mmlu_professional_medicine_generative_5shot",
]


def _list_jsons(prefix: str) -> list[str]:
    fs, urlpath = fsspec.core.url_to_fs(prefix.rstrip("/") + "/")
    if not fs.exists(urlpath):
        return []
    return [p for p in fs.ls(urlpath, detail=False) if p.endswith(".json")]


def _read_json(path: str) -> dict | None:
    try:
        with fsspec.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _read_subtask_score(eval_output_path: str, subtask: str) -> float | None:
    """Find the lm-eval-harness `results_*.json` for one subtask and return exact_match.

    Eval results layout (vLLM + lm-eval-harness):
        {output_path}/{subtask}/{model_dir}/results_{timestamp}.json
    """
    fs, urlpath = fsspec.core.url_to_fs(eval_output_path.rstrip("/") + "/" + subtask + "/")
    try:
        if not fs.exists(urlpath):
            return None
        # Walk one level down (model dir), then list results_*.json files.
        for inner in fs.ls(urlpath, detail=False):
            for f_path in fs.ls(inner, detail=False):
                if "results_" in f_path and f_path.endswith(".json"):
                    with fsspec.open(f_path, "r") as f:
                        d = json.load(f)
                    inner_results = list(d.get("results", {}).values())
                    if not inner_results:
                        continue
                    r = inner_results[0]
                    return r.get("exact_match,default") or r.get("acc,default") or r.get("acc_flex,default")
    except Exception as e:
        logger.warning("Failed to read subtask %s at %s: %s", subtask, eval_output_path, e)
    return None


def _aggregate(training_prefix: str, eval_prefix: str) -> list[dict[str, Any]]:
    train_paths = _list_jsons(training_prefix)
    eval_paths = _list_jsons(eval_prefix)
    logger.info("Found %d training summaries, %d eval summaries", len(train_paths), len(eval_paths))

    # Index by run_name. For training summaries, run_name is the dict's
    # `run_name` field; for eval summaries, the `model_name` field — and the
    # eval coordinator sets eval `model_name = training run_name` for SFT'd
    # models, so the two indexes align.
    train_by = {}
    for p in train_paths:
        d = _read_json(p)
        if d:
            train_by[d.get("run_name")] = d

    eval_by = {}
    for p in eval_paths:
        d = _read_json(p)
        if d:
            eval_by[d.get("model_name")] = d

    rows: list[dict[str, Any]] = []
    # Walk eval summaries first — these are the "real" data points (a training
    # run without an eval is not yet measurable). Baseline evals (no training
    # counterpart) get an empty `train` dict.
    for run_name, ev in eval_by.items():
        train = train_by.get(run_name, {})

        # Read per-subtask scores from the eval's output_path.
        per_subtask: dict[str, float | None] = {}
        if ev.get("output_path"):
            for sub in MMLU_MEDICAL_SUBTASKS:
                per_subtask[sub] = _read_subtask_score(ev["output_path"], sub)

        valid = [v for v in per_subtask.values() if isinstance(v, (int, float))]
        mmlu_avg = sum(valid) / len(valid) if valid else None

        rows.append(
            {
                "run_name": run_name,
                "domain": (
                    train.get("domain") or ev.get("model_name", "").split("-")[1]
                    if "-" in ev.get("model_name", "")
                    else None
                ),
                "branch": train.get("branch"),
                "config_name": train.get("config_name", "baseline"),
                "model_variant": train.get("model_variant") or ev.get("model_path"),
                "model_size": (
                    train.get("model_size")
                    or (
                        "0.6b"
                        if "0.6b" in (ev.get("model_name") or "")
                        else (
                            "8b"
                            if "8b" in (ev.get("model_name") or "")
                            else "14b" if "14b" in (ev.get("model_name") or "") else None
                        )
                    )
                ),
                "lr": train.get("hyperparameters", {}).get("learning_rate"),
                "bs": train.get("hyperparameters", {}).get("batch_size"),
                "wd": train.get("hyperparameters", {}).get("weight_decay"),
                "warmup": train.get("hyperparameters", {}).get("warmup"),
                "mmlu_med_avg": mmlu_avg,
                "n_subtasks": len(valid),
                **per_subtask,
            }
        )

    rows.sort(
        key=lambda r: (
            r.get("model_size") or "",
            r.get("domain") or "",
            r.get("branch") or "",
            -(r.get("mmlu_med_avg") or 0),
        )
    )
    return rows


def _print_markdown_table(rows: list[dict[str, Any]]) -> None:
    headers = ["model_size", "domain", "branch", "config_name", "lr", "bs", "wd", "warmup", "mmlu_med_avg"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        row = [str(r.get(h)) if r.get(h) is not None else "—" for h in headers]
        if isinstance(r.get("mmlu_med_avg"), float):
            row[-1] = f"{r['mmlu_med_avg'] * 100:.2f}%"
        print("| " + " | ".join(row) + " |")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--training-prefix", default=DEFAULT_TRAINING_PREFIX)
    parser.add_argument("--eval-prefix", default=DEFAULT_EVAL_PREFIX)
    parser.add_argument("--markdown", action="store_true", help="Emit a Markdown table instead of JSON.")
    parser.add_argument("--json", default=None, help="Write raw rows to this JSON path.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    rows = _aggregate(args.training_prefix, args.eval_prefix)

    if args.json:
        with fsspec.open(args.json, "w") as f:
            f.write(json.dumps(rows, indent=2))
        logger.info("Wrote %d rows to %s", len(rows), args.json)

    if args.markdown:
        _print_markdown_table(rows)
    else:
        print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
