# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Text-similarity metrics for the extraction-router bake-off.

The router decides, per page, whether a cheap rule-based extractor reproduces the
8B extractor's output well enough to offload it. "Well enough" is measured here.

Primary metric: ROUGE-L (LCS-based F1) over **Llama-3.1-8B token-ID sequences** --
order-sensitive at the training-token granularity (what the LM actually consumes).
Secondary sanity metrics: word-level 4-gram shingle F1 and word-level ROUGE-L.

Labeling rule (centralised in :func:`offload_label`): a page is offloadable
(label 1) iff the 8B kept it AND the cheap output's F1 >= threshold; pages the 8B
abstained on are forced negatives (route to the 8B) regardless of F1.

All functions are pure (no I/O) so they are cheap to unit-test on hand-computed
cases. The expensive LCS uses rapidfuzz's C++ ``LCSseq`` when available and falls
back to a correct rolling-array DP otherwise (used in tests / minimal envs).
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")

# Both extractions are token-truncated before the O(n*m) LCS so a pathological
# boilerplate page can't blow up the bake-off. Generous vs the 8B's 6144-token
# output cap; precision counts truncated tokens as non-matches (documented).
DEFAULT_MAX_TOKENS = 16384

DEFAULT_SHINGLE_N = 4
DEFAULT_F1_THRESHOLD = 0.9


@dataclass(frozen=True)
class SimilarityScore:
    """Precision/recall/F1 of a candidate extraction against the 8B reference.

    Precision is over the candidate (how much of the cheap output is real 8B
    content); recall is over the reference (how much of the 8B output was
    recovered). ``overlap`` is the LCS length (token ROUGE-L) or intersection
    size (shingle F1); the ``*_size`` fields are the candidate/reference sizes.
    """

    precision: float
    recall: float
    f1: float
    overlap: int
    hyp_size: int
    ref_size: int


def _lcs_length(a: Sequence, b: Sequence) -> int:
    """Length of the longest common subsequence of two sequences of hashables."""
    if not a or not b:
        return 0
    try:
        from rapidfuzz.distance import LCSseq

        return LCSseq.similarity(a, b)
    except ImportError:
        return _lcs_length_pure(a, b)


def _lcs_length_pure(a: Sequence, b: Sequence) -> int:
    """Rolling-array DP LCS length. O(len(a)*len(b)) time, O(min) space.

    Reference implementation behind the rapidfuzz fast path; exercised directly
    by the unit tests so the metric is verifiable without the optional dep.
    """
    if len(b) > len(a):
        a, b = b, a
    cur = [0] * (len(b) + 1)
    for x in a:
        prev_diag = 0
        for j in range(1, len(b) + 1):
            tmp = cur[j]
            if x == b[j - 1]:
                cur[j] = prev_diag + 1
            elif cur[j - 1] > cur[j]:
                cur[j] = cur[j - 1]
            prev_diag = tmp
    return cur[-1]


def _f1_from_overlap(overlap: int, hyp_size: int, ref_size: int) -> SimilarityScore:
    """Build a :class:`SimilarityScore` from an overlap count and the two sizes.

    Edge rules: both sides empty -> perfect agreement (they agree there is
    nothing); exactly one side empty -> F1 0 (overlap is 0).
    """
    if hyp_size == 0 and ref_size == 0:
        return SimilarityScore(1.0, 1.0, 1.0, 0, 0, 0)
    precision = overlap / hyp_size if hyp_size else 0.0
    recall = overlap / ref_size if ref_size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return SimilarityScore(precision, recall, f1, overlap, hyp_size, ref_size)


def rouge_l(hyp: Sequence, ref: Sequence) -> SimilarityScore:
    """ROUGE-L (LCS-based F1) over two already-tokenized sequences."""
    overlap = _lcs_length(hyp, ref)
    return _f1_from_overlap(overlap, len(hyp), len(ref))


def token_rouge_l(
    hyp: str,
    ref: str,
    tokenize: Callable[[str], Sequence[int]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SimilarityScore:
    """Primary metric: ROUGE-L over Llama-3.1-8B token IDs.

    ``tokenize`` maps text to a list of token IDs (e.g. ``tokenizer.encode`` for
    the Llama-3.1-8B tokenizer). Both sides are truncated to ``max_tokens`` to
    bound the LCS cost; the raw text's case is preserved (the metric is about
    reproducing the 8B's actual output tokens).
    """
    hyp_ids = tokenize(hyp)[:max_tokens]
    ref_ids = tokenize(ref)[:max_tokens]
    return rouge_l(hyp_ids, ref_ids)


def _norm_words(text: str) -> list[str]:
    """Lowercase + whitespace-collapse, then split into words (secondary metrics)."""
    return _WS_RE.sub(" ", text).strip().lower().split()


def _word_shingles(text: str, n: int) -> set[str]:
    """Set of contiguous word n-grams of normalized ``text``.

    Mirrors ``marin.processing.classification.decon.extract_ngrams`` (stride 0)
    but normalizes first and, for texts shorter than ``n``, falls back to a
    single shingle of the whole text so short docs still compare.
    """
    words = _norm_words(text)
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def word_shingle_f1(hyp: str, ref: str, n: int = DEFAULT_SHINGLE_N) -> SimilarityScore:
    """Secondary sanity metric: set-based F1 over word n-gram shingles."""
    hyp_set = _word_shingles(hyp, n)
    ref_set = _word_shingles(ref, n)
    overlap = len(hyp_set & ref_set)
    return _f1_from_overlap(overlap, len(hyp_set), len(ref_set))


def word_rouge_l(hyp: str, ref: str) -> SimilarityScore:
    """Secondary sanity metric: ROUGE-L over normalized word sequences."""
    return rouge_l(_norm_words(hyp), _norm_words(ref))


def offload_label(score: SimilarityScore, is_abstain: bool, threshold: float = DEFAULT_F1_THRESHOLD) -> int:
    """The router's binary label for a page.

    ``1`` (offload to the cheap extractor) iff the 8B kept the page and the
    candidate's F1 >= ``threshold``; ``0`` (route to the 8B) for every abstained
    page and every kept page the cheap extractor reproduces poorly.
    """
    if is_abstain:
        return 0
    return 1 if score.f1 >= threshold else 0
