# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Best-threshold F1 of every scalar classifier column vs the 8B gold label, on the 100k sample.

Gold = ``label_8b`` (``useful`` is the ~3% minority positive). For each score column we sweep
thresholds in BOTH orientations and keep the best F1 — so probability columns (higher = useful)
and the LLM marker-logprob columns (higher = NOT useful) are handled uniformly without hand-coding
direction. Answers "how well does each classifier reproduce the 8B keep/drop decision", including
the usefulness-vs-context question for the ``*_ctx{4,8,16}k`` logprob columns.

Reads ONLY the label + float score columns (projection) so it's cheap and in-region. Run::

    iris --cluster marin job run --region us-east5 --cpu 4 --memory 16GB --enable-extra-resources \\
      --extra cpu --priority interactive --no-wait --job-name clf-f1 -- \\
      python -m experiments.baseline_collection.analyze_classifier_f1
"""

from __future__ import annotations

import json
import logging

import fsspec
import numpy as np
import pyarrow.parquet as pq
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_ROOT = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc"
SCORED_DIR = f"{OUT_ROOT}/sample_100k_scored"
RESULT_JSON = f"{OUT_ROOT}/analysis/classifier_f1.json"
LABEL_COL = "label_8b"
SCORE_PREFIXES = ("bert_useful_prob", "fasttext_useful_prob", "llm_logprob_marker")
N_THRESHOLDS = 256


def _best_f1(scores: np.ndarray, gold: np.ndarray) -> dict:
    """Best-F1 over thresholds in both orientations. gold is bool (True=useful=positive)."""
    valid = ~np.isnan(scores)
    s, y = scores[valid], gold[valid]
    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        return {"f1": float("nan"), "note": "degenerate gold"}
    qs = np.quantile(s, np.linspace(0.0, 1.0, N_THRESHOLDS))
    best = {"f1": -1.0}
    for sign in (1.0, -1.0):  # +1: score>=thr -> useful;  -1: score<=thr -> useful
        ss = sign * s
        tt = sign * qs
        for thr in np.unique(tt):
            pred = ss >= thr
            tp = int(np.sum(pred & y))
            fp = int(np.sum(pred & ~y))
            fn = int(np.sum(~pred & y))
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom else 0.0
            if f1 > best["f1"]:
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                best = {
                    "f1": f1,
                    "threshold": float(sign * thr),
                    "direction": "high=useful" if sign > 0 else "low=useful",
                    "precision": prec,
                    "recall": rec,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
    best["n"] = int(len(y))
    best["n_useful"] = pos
    return best


def main() -> None:
    files = sorted(fsspec_glob(f"{SCORED_DIR}/*.parquet"))
    if not files:
        raise RuntimeError(f"no scored parquet under {SCORED_DIR}")
    logger.info("F1 analysis starting: %d scored shards under %s", len(files), SCORED_DIR)
    with fsspec.open(files[0], "rb") as fh:  # context manager — OpenFile isn't a usable file handle directly
        schema = pq.ParquetFile(fh).schema_arrow
    score_cols = [
        n
        for n, t in zip(schema.names, schema.types)
        if n.startswith(SCORE_PREFIXES) and str(t).startswith("float")
    ]
    logger.info("scoring %d classifier columns: %s", len(score_cols), score_cols)

    labels: list[str] = []
    cols: dict[str, list[float]] = {c: [] for c in score_cols}
    for i, path in enumerate(files):
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=[LABEL_COL, *score_cols])
        labels.extend(t.column(LABEL_COL).to_pylist())
        for c in score_cols:
            cols[c].extend(t.column(c).to_pylist())
        if (i + 1) % 50 == 0:
            logger.info("read %d/%d shards (%d docs)", i + 1, len(files), len(labels))
    gold = np.array([lv == "useful" for lv in labels], dtype=bool)
    logger.info("docs=%d  useful(8B)=%d (%.1f%%)", len(gold), int(gold.sum()), 100 * gold.mean())

    results = {}
    for c in score_cols:
        arr = np.array([np.nan if v is None else v for v in cols[c]], dtype=np.float64)
        results[c] = _best_f1(arr, gold)

    # Markdown table block the babysit can lift straight into the results doc §4.
    print("=== F1 TABLE START ===")
    print("| column | best F1 | precision | recall | dir | threshold |")
    print("|--------|--------|-----------|--------|-----|-----------|")
    for c in sorted(results, key=lambda k: -(results[k].get("f1") or -1)):
        r = results[c]
        if r["f1"] != r["f1"]:  # nan
            print(f"| `{c}` | n/a | | | | {r.get('note','')} |")
        else:
            print(
                f"| `{c}` | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} "
                f"| {r['direction']} | {r['threshold']:.4g} |"
            )
    print("=== F1 TABLE END ===")

    with fsspec.open(RESULT_JSON, "w") as fh:
        json.dump({"n": int(len(gold)), "n_useful": int(gold.sum()), "columns": results}, fh, indent=2)
    logger.info("wrote %s", RESULT_JSON)


if __name__ == "__main__":
    main()
