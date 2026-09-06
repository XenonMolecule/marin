# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Does a TEXT classifier tolerate the one-extraction deployment input? Decide from scores, not text diffs.

For each (as-trained column, deployment column) pair — the SAME checkpoint scored on
``clf_text_resiliparse_rs`` (extract(body_strip HTML), as trained) vs ``clf_text_resiliparse_rs_fromraw``
(lower(collapse(extract(raw_html))), the production contract) — report over the lpv11-covered docs:
score correlation, keep-set agreement at 0.5, F1/precision/recall vs lpv11 at 0.5, best-threshold F1,
and precision at recall 0.97 (the cascade operating point). A drop in F1 / P@R.97 = retrain.

Aggregate deltas dilute a real effect ~2.5x (61% of docs are byte-identical), so every metric is ALSO
reported per stratum: identical / char-sim >= 0.9 / [0.5, 0.9) / < 0.5, and the empty-mismatch docs
(one input empty, the other not — the neural corpora's ``__empty__`` token pattern). Per-doc similarity is
recomputed here from the two text artifacts, so run this IN-REGION (rapidfuzz => --extra extraction-bakeoff)::

    uv run iris --cluster marin job run --region us-east5 --cpu 4 --memory 32GB \
      --enable-extra-resources --extra cpu --extra extraction-bakeoff --priority interactive --no-wait \
      --job-name clf-text-compare -- python -m experiments.baseline_collection.compare_text_input_scores
"""

from __future__ import annotations

import json

import fsspec
import numpy as np
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from experiments.baseline_collection.comparison_sample import (
    CLF_TEXT_COLUMN,
    CLF_TEXT_FROM_RAW_COLUMN,
    OUT_ROOT,
    SCORES_DIR,
    clf_text_dir,
    clf_text_from_raw_dir,
)
from experiments.fsspec_paths import fsspec_glob

PAIRS = [
    ("bert_lpv11_text_prob_base_10M", "bert_lpv11_textraw_prob_base_10M"),
    ("bert_lpv11_text_prob_ettin68_10M", "bert_lpv11_textraw_prob_ettin68_10M"),
    ("bert_lpv11_text_prob_base_1M", "bert_lpv11_textraw_prob_base_1M"),
    ("fasttext_lpv11_text_prob_w640", "fasttext_lpv11_textraw_prob_w640"),
]
TARGET_DIR = f"{OUT_ROOT}/target_lpv11"
LEV_CAP = 20_000
STRATA = ["identical", "sim>=0.9", "0.5<=sim<0.9", "sim<0.5", "only-deploy-empty", "only-trained-empty"]


def _read(directory: str, col: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for f in sorted(fsspec_glob(f"{directory}/*.parquet")):
        with fsspec.open(f, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["warc_record_id", col])
        out.update(zip(t.column("warc_record_id").to_pylist(), t.column(col).to_pylist(), strict=True))
    return out


def _prf(keep: np.ndarray, gold: np.ndarray) -> tuple[float, float, float]:
    tp = int((keep & gold).sum())
    fp = int((keep & ~gold).sum())
    fn = int((~keep & gold).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def _best_f1(s: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    best = (0.0, 0.5)
    for t in np.linspace(0.02, 0.98, 49):
        f1, _, _ = _prf(s >= t, gold)
        if f1 > best[0]:
            best = (f1, float(t))
    return best


def _p_at_recall(s: np.ndarray, gold: np.ndarray, recall: float) -> tuple[float, float, float]:
    """(precision, threshold, keep_frac) at the largest threshold whose recall >= `recall`."""
    pos = np.sort(s[gold])[::-1]
    t = pos[min(len(pos) - 1, int(np.ceil(recall * len(pos)) - 1))]
    keep = s >= t
    _, p, _ = _prf(keep, gold)
    return p, float(t), float(keep.mean())


def _strata(ids: list[str]) -> np.ndarray:
    """Per-doc stratum from the two text artifacts (as-trained vs deployment input)."""
    a_map = _read(clf_text_dir("marin-us-east5"), CLF_TEXT_COLUMN)
    b_map = _read(clf_text_from_raw_dir("marin-us-east5"), CLF_TEXT_FROM_RAW_COLUMN)
    out = np.empty(len(ids), dtype=object)
    for k, i in enumerate(ids):
        x, y = a_map[i] or "", b_map[i] or ""
        if bool(x) != bool(y):
            # y (deployment) empty while x (trained) had text = a one-directional recall hazard at serve time.
            out[k] = "only-deploy-empty" if x else "only-trained-empty"
        elif x == y:
            out[k] = "identical"
        else:
            sim = Levenshtein.normalized_similarity(x[:LEV_CAP], y[:LEV_CAP])
            out[k] = "sim>=0.9" if sim >= 0.9 else ("0.5<=sim<0.9" if sim >= 0.5 else "sim<0.5")
    return out


def _block(a: np.ndarray, b: np.ndarray, gold: np.ndarray) -> dict:
    ka, kb = a >= 0.5, b >= 0.5
    fa, pa, ra = _prf(ka, gold)
    fb, pb, rb = _prf(kb, gold)
    return {
        "n": len(a),
        "gold_keep": float(gold.mean()) if len(a) else 0.0,
        "keep_agreement@0.5": float((ka == kb).mean()) if len(a) else 1.0,
        "mean_abs_dP": float(np.abs(a - b).mean()) if len(a) else 0.0,
        "f1@0.5_trained": fa,
        "f1@0.5_deploy": fb,
        "delta_f1@0.5": fb - fa,
        "delta_recall@0.5": rb - ra,
        "delta_precision@0.5": pb - pa,
    }


def main() -> None:
    labels = _read(TARGET_DIR, "label_lpv11")
    ids = [i for i, v in labels.items() if v is not None]  # lpv11-covered docs only
    gold = np.array([labels[i] == "useful" for i in ids])
    print(f"covered docs {len(ids)}  gold keep {gold.mean():.4f}")
    strata = _strata(ids)
    counts = {st: int((strata == st).sum()) for st in STRATA}
    print("strata:", counts)
    report = {"strata_counts": counts}
    for a_col, b_col in PAIRS:
        if not fsspec_glob(f"{SCORES_DIR}/{b_col}/*.parquet"):
            print(f"\n== {a_col}: deployment column {b_col} not scored yet — skipped")
            continue
        a_map = _read(f"{SCORES_DIR}/{a_col}", a_col)
        b_map = _read(f"{SCORES_DIR}/{b_col}", b_col)
        a = np.array([a_map[i] for i in ids], dtype=float)
        b = np.array([b_map[i] for i in ids], dtype=float)
        ka, kb = a >= 0.5, b >= 0.5
        fa, pa, ra = _prf(ka, gold)
        fb, pb, rb = _prf(kb, gold)
        ba, bb = _best_f1(a, gold), _best_f1(b, gold)
        pra, prb = _p_at_recall(a, gold, 0.97), _p_at_recall(b, gold, 0.97)
        rec = {
            "pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "mean_abs_dP": float(np.abs(a - b).mean()),
            "frac_abs_dP_gt_0.2": float((np.abs(a - b) > 0.2).mean()),
            "keep_agreement@0.5": float((ka == kb).mean()),
            "as_trained": {
                "f1@0.5": fa,
                "p@0.5": pa,
                "r@0.5": ra,
                "best_f1": ba[0],
                "best_thr": ba[1],
                "p@r0.97": pra[0],
                "keep_frac@r0.97": pra[2],
            },
            "deployment": {
                "f1@0.5": fb,
                "p@0.5": pb,
                "r@0.5": rb,
                "best_f1": bb[0],
                "best_thr": bb[1],
                "p@r0.97": prb[0],
                "keep_frac@r0.97": prb[2],
            },
            "delta_f1@0.5": fb - fa,
            "delta_best_f1": bb[0] - ba[0],
            "delta_p@r0.97": prb[0] - pra[0],
        }
        rec["by_stratum"] = {st: _block(a[strata == st], b[strata == st], gold[strata == st]) for st in STRATA}
        report[a_col] = rec
        print(f"\n== {a_col}  vs  {b_col}")
        print(
            f"  r={rec['pearson_r']:.4f}  mean|dP|={rec['mean_abs_dP']:.4f}  "
            f"|dP|>0.2: {rec['frac_abs_dP_gt_0.2']:.3%}  keep-agree@0.5={rec['keep_agreement@0.5']:.4f}"
        )
        for name, (f1, pr, rc, best, par) in (
            ("as-trained", (fa, pa, ra, ba, pra)),
            ("deployment", (fb, pb, rb, bb, prb)),
        ):
            print(
                f"  {name:10s}: F1@.5={f1:.4f} (P {pr:.3f} R {rc:.3f})  bestF1={best[0]:.4f}@{best[1]:.2f}  "
                f"P@R.97={par[0]:.4f} keep {par[2]:.3f}"
            )
        print(f"  DELTA      : F1@.5 {fb - fa:+.4f}   bestF1 {bb[0] - ba[0]:+.4f}   P@R.97 {prb[0] - pra[0]:+.4f}")
        for st in STRATA:
            bl = rec["by_stratum"][st]
            print(
                f"    [{st:18s}] n={bl['n']:6d} gold={bl['gold_keep']:.3f} agree@.5={bl['keep_agreement@0.5']:.4f} "
                f"|dP|={bl['mean_abs_dP']:.4f}  F1 {bl['f1@0.5_trained']:.4f}->{bl['f1@0.5_deploy']:.4f} "
                f"({bl['delta_f1@0.5']:+.4f}; dR {bl['delta_recall@0.5']:+.4f} dP {bl['delta_precision@0.5']:+.4f})"
            )
    with fsspec.open(f"{OUT_ROOT}/clf_text_input_score_comparison.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
