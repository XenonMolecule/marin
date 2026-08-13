# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token / shingle F1 for comparing extractor outputs.

This is the metric from the article-extraction-benchmark
(submodules/article-extraction-benchmark/evaluate.py), reproduced here so it can
be reused without a submodule path dependency:

- tokenize on \\w+ (unicode word characters)
- build n-gram "shingles" (default n=4)
- TP/FP/FN over the shingle multiset, normalized per-document so long docs don't
  dominate
- per-doc precision/recall with the benchmark's edge-case conventions
- corpus aggregate = mean precision and mean recall over eligible docs, then
  F1 = harmonic mean of those means

Empty predictions (e.g. a sentinel like [NO_USEFUL_CONTENT] mapped to "") are
handled naturally: they yield no shingles, so recall against a non-empty
reference is 0.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from rapidfuzz.distance import LCSseq

TP_FP_FN = tuple[float, float, float]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE | re.MULTILINE | re.IGNORECASE | re.DOTALL)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def _ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    tokens = tokenize(text)
    result = []
    for i in range(0, max(1, len(tokens) - n + 1)):
        shingle = tuple(tokens[i : i + n])
        if shingle:
            result.append(shingle)
    return result


def _all_shingles(text: str, ngram_n: int) -> dict[tuple[str, ...], int]:
    return dict(Counter(_ngrams(text, ngram_n)))


def string_shingle_matching(true: str, pred: str, ngram_n: int = 4) -> TP_FP_FN:
    """Normalized (tp, fp, fn) over shingles. `true` is the reference text."""
    true_shingles = _all_shingles(true, ngram_n)
    pred_shingles = _all_shingles(pred, ngram_n)
    tp = fp = fn = 0.0
    for key in set(true_shingles) | set(pred_shingles):
        true_count = true_shingles.get(key, 0)
        pred_count = pred_shingles.get(key, 0)
        tp += min(true_count, pred_count)
        fp += max(0, pred_count - true_count)
        fn += max(0, true_count - pred_count)
    tp_fp_fn = [tp, fp, fn]
    s = sum(tp_fp_fn)
    if s > 0:
        tp_fp_fn = [x / s for x in tp_fp_fn]
    return tuple(tp_fp_fn)  # type: ignore[return-value]


def precision_score(tp: float, fp: float, fn: float) -> float:
    if fp == fn == 0:
        return 1.0
    if tp == fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_score(tp: float, fp: float, fn: float) -> float:
    if fp == fn == 0:
        return 1.0
    if tp == fn == 0:
        return 0.0
    return tp / (tp + fn)


def doc_prf(true: str, pred: str, ngram_n: int = 4) -> dict[str, float]:
    """Per-document precision/recall/F1 (and raw normalized tp/fp/fn)."""
    tp, fp, fn = string_shingle_matching(true, pred, ngram_n)
    p = precision_score(tp, fp, fn)
    r = recall_score(tp, fp, fn)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def aggregate(tp_fp_fns: list[TP_FP_FN]) -> dict[str, float]:
    """Corpus aggregate matching the benchmark: mean precision / mean recall over
    eligible docs, then harmonic-mean F1."""
    precisions = [precision_score(*t) for t in tp_fp_fns if t[0] + t[1] > 0]
    recalls = [recall_score(*t) for t in tp_fp_fns if t[0] + t[2] > 0]
    precision = statistics.mean(precisions) if precisions else 0.0
    recall = statistics.mean(recalls) if recalls else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n": len(tp_fp_fns)}


# ---------------------------------------------------------------------------
# ROUGE-L: longest-common-subsequence based, so it rewards correct token ORDER
# (not just bag/shingle overlap). LCS computed with rapidfuzz (fast, C, bit-parallel).
# ---------------------------------------------------------------------------
# (lcs_len, n_ref_tokens, n_cand_tokens)
LCS_COUNTS = tuple[int, int, int]


def rouge_l_counts(true: str, pred: str) -> LCS_COUNTS:
    a = tokenize(true)
    b = tokenize(pred)
    lcs = LCSseq.similarity(a, b) if a and b else 0
    return lcs, len(a), len(b)


def doc_rouge_l(true: str, pred: str) -> dict[str, float]:
    """Per-document ROUGE-L precision/recall/F1. `true` is the reference."""
    lcs, n_ref, n_cand = rouge_l_counts(true, pred)
    if n_ref == 0 and n_cand == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "lcs": 0}
    p = lcs / n_cand if n_cand else 0.0
    r = lcs / n_ref if n_ref else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "lcs": lcs}


def levenshtein_sim(true: str, pred: str, max_chars: int = 20000) -> float:
    """Character-level Levenshtein similarity in [0,1] (1 - dist/max_len), matching
    the jusText repo's metric. Raw strings: newlines/spaces/case all count.

    max_chars caps each string before the O(n*m) edit-distance call so the metric
    stays fast at leaderboard scale (1000s of pairs). Identical to the uncapped
    value for the vast majority of extraction outputs (which are < 20k chars);
    only pathologically long strings are truncated.
    """
    from rapidfuzz.distance import Levenshtein

    a, b = (true or "")[:max_chars], (pred or "")[:max_chars]
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b)) or 1
    return 1.0 - Levenshtein.distance(a, b) / denom


def aggregate_rouge_l(counts: list[LCS_COUNTS]) -> dict[str, float]:
    """Corpus ROUGE-L, aggregated like the shingle metric: mean precision over
    docs with a non-empty candidate, mean recall over docs with a non-empty
    reference, then harmonic-mean F1. Both-empty docs are excluded from both."""
    precisions = [lcs / nc for lcs, nr, nc in counts if nc > 0]
    recalls = [lcs / nr for lcs, nr, nc in counts if nr > 0]
    precision = statistics.mean(precisions) if precisions else 0.0
    recall = statistics.mean(recalls) if recalls else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n": len(counts)}
