# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate all MMLU (`mmlu_sl_verb`) results.json (5 buckets) into ONE CSV on GCS.

Sibling of `core_tasks/consolidate_results.py` and `olmes_base/consolidate_olmes_results.py`,
pointed at the MMLU result prefix, and reusing their run-name regex / N(d,L) token
calibration rather than keeping a third copy. Runs as a CPU iris job so the per-run
results.json (~8MB each here — see below) are read in-cloud, then emits a single CSV a
human can pull with one `gcloud storage cp` and plot locally.

TWO WAYS THIS DIFFERS FROM ITS SIBLINGS, both load-bearing:

1. It reads the GROUP AGGREGATE, not a flat mean over rows. `mean_core`/`mean_olmes`
   average the primary metric across every row in `results`, which works only because
   those suites are flat lists of tasks. MMLU is a GROUP: `results` holds 58 rows = 1
   group row + 57 subject rows, so flat-averaging would average the aggregate together
   with the very subjects it aggregates, AND would silently replace lm-eval's
   size-weighted acc with an unweighted per-subject mean. We take
   `doc["groups"]["mmlu_sl_verb_<N>shot"]` instead — one authoritative row, already
   size-weighted for acc/acc_norm per the task's `aggregate_metric_list`.

2. It carries the SOFT METRICS, which are the point of sl_verb over stock mmlu. Across
   this sweep (1e17..2e21) `acc` is pinned near the 0.25 random baseline for most cells;
   `bpb` / `choice_logprob` / `choice_prob_norm` stay informative down there. Emitting
   only an accuracy column would throw away the reason we chose this variant.

Note the group row is keyed by the ALIAS (`mmlu_sl_verb_5shot`), not `mmlu_sl_verb`, and
metrics are suffixed `,none`. Every `*_stderr,none` is the STRING "N/A", so it is
skipped rather than parsed.

Columns: run_stem, method, hidden_dim, num_layers, budget, tokens, num_fewshot,
         acc, acc_norm, bpb, logprob, choice_logprob, choice_prob_norm,
         choice_logprob_norm, n_subtasks
  * tokens = budget / (6 * N(d,L)) (isoflop C=6ND), same calibration as the siblings.
  * n_subtasks should be 57; anything less means subjects silently failed to score.

Usage:
    iris --cluster marin job run --region us-central1 --cpu 2 --memory 8GB \\
        --enable-extra-resources \\
        -- python -m experiments.scaling_law_sweeps.mmlu.consolidate_mmlu_results --shots 5
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
from collections import defaultdict

from rigging.filesystem import filesystem as marin_filesystem

from experiments.scaling_law_sweeps.core_tasks.consolidate_results import (
    BUCKETS,
    RUN_RE,
    norm_method,
    tokens_for,
)

logger = logging.getLogger(__name__)

RESULTS_PREFIX = "metadata/mmlu_sl_verb_results"
EXPECTED_SUBTASKS = 57  # MMLU's 57 subjects; sl_verb ships a yaml per subject.

# The group row's metrics, in CSV column order. Sourced from sl_verb's
# aggregate_metric_list: acc/acc_norm are size-weighted, the rest are unweighted means.
METRICS = (
    "acc",
    "acc_norm",
    "bpb",
    "logprob",
    "choice_logprob",
    "choice_prob_norm",
    "choice_logprob_norm",
)


def group_row(doc: dict, shots: int) -> tuple[dict[str, float], int] | None:
    """The `mmlu_sl_verb_<shots>shot` group aggregate and its subtask count.

    Returns (metrics, n_subtasks), or None if the group row is absent (a run that died
    partway, or a shot count that does not match this results.json).
    """
    key = f"mmlu_sl_verb_{shots}shot"
    row = doc.get("groups", {}).get(key) or doc.get("results", {}).get(key)
    if not isinstance(row, dict):
        return None
    metrics = {}
    for m in METRICS:
        v = row.get(f"{m},none")
        if isinstance(v, (int, float)):  # skips the "N/A" string stderrs and any None
            metrics[m] = float(v)
    n_subtasks = len(doc.get("group_subtasks", {}).get(key, []))
    return metrics, n_subtasks


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=5, help="Which shot count's results to consolidate.")
    ap.add_argument("--out", default=None, help="Consolidated CSV to write. Default: derived from --shots.")
    args = ap.parse_args()
    out = args.out or f"gs://marin-us-central1/metadata/mmlu_sl_verb_{args.shots}shot_summary.csv"

    fs = marin_filesystem("gcs")
    rows = []
    for b in BUCKETS:
        for path in fs.glob(f"gs://{b}/{RESULTS_PREFIX}/{args.shots}shot/*/results.json"):
            run = path.split(f"/{args.shots}shot/")[1].split("/")[0]
            m = RUN_RE.match(run)
            if not m:
                logger.warning("unparsed run: %s", run)
                continue
            with fs.open(path if path.startswith("gs://") else f"gs://{path}", "r") as f:
                got = group_row(json.load(f), args.shots)
            if got is None:
                logger.warning("no %dshot group row in %s", args.shots, run)
                continue
            metrics, n_subtasks = got
            d, L = int(m["d"]), int(m["L"])
            rows.append(
                dict(
                    run_stem=run,
                    method=norm_method(m["method"]),
                    hidden_dim=d,
                    num_layers=L,
                    budget=m["budget"],
                    tokens=tokens_for(m["budget"], d, L),
                    num_fewshot=args.shots,
                    **metrics,
                    n_subtasks=n_subtasks,
                )
            )
    logger.info("consolidated %d runs", len(rows))

    fields = [
        "run_stem",
        "method",
        "hidden_dim",
        "num_layers",
        "budget",
        "tokens",
        "num_fewshot",
        *METRICS,
        "n_subtasks",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["method"], r["hidden_dim"], r["tokens"] or 0)))
    with fs.open(out, "w") as f:
        f.write(buf.getvalue())
    logger.info("wrote %s (%d rows)", out, len(rows))

    cnt = defaultdict(int)
    for r in rows:
        cnt[(r["hidden_dim"], r["method"])] += 1
    logger.info("coverage (method x width):")
    for mth in sorted({r["method"] for r in rows}):
        logger.info("  %-16s %s", mth, {d: cnt[(d, mth)] for d in (512, 1024, 1536, 2432, 3584) if cnt[(d, mth)]})

    # A run that scored fewer than all 57 subjects has a silently-short aggregate.
    short = [(r["run_stem"], r["n_subtasks"]) for r in rows if r["n_subtasks"] < EXPECTED_SUBTASKS]
    if short:
        logger.warning("%d runs scored <%d subtasks (first 5): %s", len(short), EXPECTED_SUBTASKS, short[:5])


if __name__ == "__main__":
    main()
