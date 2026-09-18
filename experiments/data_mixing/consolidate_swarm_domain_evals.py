# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate swarm-proxy domain evals into per-run CSVs for mixture fitting.

Scans `metadata/olmix_swarm_mmlu/<shots>shot/<run>/results.json` and
`metadata/olmix_swarm_cultural/<shots>shot/<run>/results.json` across the regional
buckets (the swarm side-sweep namespaces written by `launch_swarm_mmlu` /
`launch_swarm_cultural`) and emits one row per (run, category) with the seven soft
metrics. MMLU rows aggregate the 57 subjects into the official Hendrycks categories
(stem / humanities / social_sciences / other, unweighted mean over subjects for soft
metrics — the sl_verb convention); cultural rows reuse the sweep consolidator's
benchmark + macro aggregation. These CSVs are the objective inputs for domain-aware
mixture fits, so rows carry `corpus` and `swarm_index` parsed from the run name
(`olmix-<corpus>-s42-K363-i####-w########`).

    python -m experiments.data_mixing.consolidate_swarm_domain_evals \\
        --shots 5 --out-dir scratch/swarm_domain_evals/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import fsspec

from experiments.scaling_law_sweeps.cultural.consolidate_cultural_results import aggregate as cultural_aggregate
from experiments.scaling_law_sweeps.cultural.consolidate_cultural_results import task_metrics as cultural_task_metrics
from experiments.scaling_law_sweeps.mmlu.consolidate_mmlu_categories import FIELDS, category_rows

logger = logging.getLogger(__name__)

BUCKETS = ("marin-us-east5", "marin-us-central1", "marin-us-east1", "marin-us-west4", "marin-eu-west4")
RUN_NAME_RE = re.compile(r"^olmix-(?P<corpus>.+)-s\d+-K\d+-i(?P<index>\d+)-w[0-9a-f]+$")


def _run_rows(suite: str, run_name: str, doc: dict, shots: int) -> list[dict]:
    m = RUN_NAME_RE.match(run_name)
    corpus, index = (m["corpus"], int(m["index"])) if m else (run_name, -1)
    base = dict(suite=suite, corpus=corpus, swarm_index=index, run_name=run_name)
    rows: list[dict] = []
    if suite == "mmlu":
        for category, (metrics, n_subjects) in category_rows(doc, shots).items():
            rows.append(dict(**base, level="category", name=category, n_tasks=n_subjects, **metrics))
    else:
        for agg in cultural_aggregate(cultural_task_metrics(doc, shots)):
            if agg["level"] == "task":
                continue  # per-run task rows are bulk; benchmark + macro suffice for fitting
            rows.append(
                dict(
                    **base,
                    **{k: agg[k] for k in ("level", "name", "n_tasks")},
                    **{f: agg[f] for f in FIELDS if f in agg},
                )
            )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--out-dir", default="scratch/swarm_domain_evals")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("gcs")

    for suite, ns in (("mmlu", "olmix_swarm_mmlu"), ("cultural", "olmix_swarm_cultural")):
        rows: list[dict] = []
        n_runs = 0
        for bucket in BUCKETS:
            paths = fs.glob(f"gs://{bucket}/metadata/{ns}/{args.shots}shot/*/results.json")
            logger.info("%s: %d results in %s", suite, len(paths), bucket)
            for path in paths:
                run_name = path.rsplit("/", 2)[-2]
                with fs.open(path) as fh:
                    doc = json.load(fh)
                rows.extend(_run_rows(suite, run_name, doc, args.shots))
                n_runs += 1
        if not rows:
            logger.warning("%s: no results found; skipping CSV", suite)
            continue
        out = out_dir / f"swarm_{suite}_{args.shots}shot_by_category.csv"
        fieldnames = list(rows[0].keys())
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info("%s: wrote %d rows from %d runs to %s", suite, len(rows), n_runs, out)


if __name__ == "__main__":
    main()
