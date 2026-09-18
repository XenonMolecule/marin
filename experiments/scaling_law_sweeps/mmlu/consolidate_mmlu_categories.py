# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate `mmlu_sl_verb` results into per-CATEGORY rows (stem / humanities /
social_sciences / other) plus the whole-suite row.

`consolidate_mmlu_results.py` reads only the group aggregate. The sl_verb subject YAMLs
carry `tag: mmlu_sl_verb_<category>`, but tags produce NO rows in results.json (verified
on a real 5-shot file: 1 group row + 57 subject rows, nothing per category), so the
category numbers have to be rebuilt from the subject rows here. Aggregation follows the
sl_verb group YAML exactly: `acc` / `acc_norm` weighted by subject example count (the
`OLMIX_MMLU_WEIGHTS` fractions, renormalised over the subjects that scored) and the five
soft metrics as an unweighted mean over subjects.

Category membership is the official MMLU release's own grouping (Hendrycks et al.,
`categories.py` in `hendrycks/test`: 18 / 13 / 12 / 14 = 57), read from the verbatim copy
in `olmo_bpb/mmlu_olmes.py`. `n_subjects` below `n_subjects_expected` flags a run whose
subjects partly failed to score.

Columns: run_stem, method, hidden_dim, num_layers, budget, tokens, num_fewshot, category,
         acc, acc_norm, bpb, logprob, choice_logprob, choice_prob_norm, choice_logprob_norm,
         n_subjects, n_subjects_expected

Usage:
    iris --cluster marin job run --region us-central1 --cpu 2 --memory 16GB \\
        --enable-extra-resources \\
        -- python -m experiments.scaling_law_sweeps.mmlu.consolidate_mmlu_categories --shots 5
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
from experiments.scaling_law_sweeps.mmlu.consolidate_mmlu_results import (
    METRICS,
    RESULTS_PREFIX,
    group_row,
)
from experiments.scaling_law_sweeps.olmo_bpb.mmlu_olmes import (
    MMLU_CATEGORIES,
    OLMIX_MMLU_WEIGHTS,
    olmix_metric_name,
)

logger = logging.getLogger(__name__)

ALL_CATEGORY = "all"
SIZE_WEIGHTED = ("acc", "acc_norm")
FIELDS = [
    "run_stem",
    "method",
    "hidden_dim",
    "num_layers",
    "budget",
    "tokens",
    "num_fewshot",
    "category",
    *METRICS,
    "n_subjects",
    "n_subjects_expected",
]


def subject_key(subject: str, shots: int) -> str:
    return f"mmlu_sl_verb_{subject}_{shots}shot"


def _metrics_of(row: dict) -> dict[str, float]:
    return {m: float(row[f"{m},none"]) for m in METRICS if isinstance(row.get(f"{m},none"), (int, float))}


def category_rows(doc: dict, shots: int) -> dict[str, tuple[dict[str, float], int]]:
    """{category: (metrics, n_subjects_scored)} rebuilt from the subject rows, plus the
    harness's own group aggregate under `all` when present."""
    results = doc.get("results", {})
    out: dict[str, tuple[dict[str, float], int]] = {}
    for category, subjects in MMLU_CATEGORIES.items():
        scored: list[tuple[float, dict[str, float]]] = []
        for s in subjects:
            row = results.get(subject_key(s, shots))
            if isinstance(row, dict):
                scored.append((OLMIX_MMLU_WEIGHTS[category][olmix_metric_name(s)], _metrics_of(row)))
        if not scored:
            continue
        total_w = sum(w for w, _ in scored)
        metrics: dict[str, float] = {}
        for m in METRICS:
            values = [(w, r[m]) for w, r in scored if m in r]
            if len(values) != len(scored):
                continue
            if m in SIZE_WEIGHTED:
                metrics[m] = sum(w * v for w, v in values) / total_w
            else:
                metrics[m] = fmean(v for _, v in values)
        out[category] = (metrics, len(scored))
    got = group_row(doc, shots)
    if got is not None:
        out[ALL_CATEGORY] = got
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--out", default=None, help="CSV to write. Default: derived from --shots.")
    args = ap.parse_args()
    out = args.out or f"gs://marin-us-central1/metadata/mmlu_sl_verb_{args.shots}shot_by_category.csv"
    expected = {c: len(s) for c, s in MMLU_CATEGORIES.items()}
    expected[ALL_CATEGORY] = sum(expected.values())

    fs = marin_filesystem("gcs")
    rows: list[dict] = []
    for b in BUCKETS:
        for path in fs.glob(f"gs://{b}/{RESULTS_PREFIX}/{args.shots}shot/*/results.json"):
            run = path.split(f"/{args.shots}shot/")[1].split("/")[0]
            m = RUN_RE.match(run)
            if not m:
                logger.warning("unparsed run: %s", run)
                continue
            with fs.open(path if path.startswith("gs://") else f"gs://{path}", "r") as f:
                per_category = category_rows(json.load(f), args.shots)
            if not per_category:
                logger.warning("no %dshot subject rows in %s", args.shots, run)
                continue
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
            for category, (metrics, n) in per_category.items():
                rows.append(
                    {**base, "category": category, **metrics, "n_subjects": n, "n_subjects_expected": expected[category]}
                )
    logger.info("consolidated %d rows", len(rows))

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, restval="")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["method"], r["hidden_dim"], r["tokens"] or 0, r["category"])))
    with fs.open(out, "w") as f:
        f.write(buf.getvalue())
    logger.info("wrote %s", out)

    short = [(r["run_stem"], r["category"], r["n_subjects"]) for r in rows if r["n_subjects"] < r["n_subjects_expected"]]
    if short:
        logger.warning("%d category rows scored fewer subjects than expected (first 5): %s", len(short), short[:5])


if __name__ == "__main__":
    main()
