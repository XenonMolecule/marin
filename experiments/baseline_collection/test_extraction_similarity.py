# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for extraction_similarity: ROUGE-L (token + word), shingle F1, the
edge rules (empty/short docs), and the abstention-aware offload label. Cases are
hand-computed so a test fails on wrong behavior, not just changed implementation."""

from __future__ import annotations

import pytest

from experiments.baseline_collection.extraction_similarity import (
    SimilarityScore,
    _lcs_length,
    _lcs_length_pure,
    offload_label,
    rouge_l,
    token_rouge_l,
    word_rouge_l,
    word_shingle_f1,
)

# --- LCS core --------------------------------------------------------------


def test_lcs_pure_classic_example():
    # Textbook LCS("ABCBDAB", "BDCAB") = 4 (e.g. "BCAB").
    assert _lcs_length_pure(list("ABCBDAB"), list("BDCAB")) == 4


def test_lcs_pure_no_common_and_empty():
    assert _lcs_length_pure(list("abc"), list("xyz")) == 0
    assert _lcs_length_pure([], list("abc")) == 0
    assert _lcs_length_pure(list("abc"), []) == 0


@pytest.mark.parametrize(
    "a,b",
    [
        (list("ABCBDAB"), list("BDCAB")),
        (list("xmjyauz"), list("mzjawxu")),
        ([1, 2, 3, 4, 5, 1, 2], [2, 4, 1, 2, 5]),
        (list("aaaa"), list("aa")),
    ],
)
def test_lcs_fastpath_matches_pure(a, b):
    # If rapidfuzz is installed _lcs_length uses it; it must agree with the DP.
    assert _lcs_length(a, b) == _lcs_length_pure(a, b)


# --- rouge_l (already-tokenized) -------------------------------------------


def test_rouge_l_identical_is_perfect():
    s = rouge_l([1, 2, 3, 4], [1, 2, 3, 4])
    assert s.f1 == 1.0 and s.precision == 1.0 and s.recall == 1.0


def test_rouge_l_is_order_sensitive_unlike_bag_of_tokens():
    # Same multiset, reversed order -> LCS is just 1, so ROUGE-L is far below 1.
    s = rouge_l([1, 2, 3, 4], [4, 3, 2, 1])
    assert s.overlap == 1
    assert s.f1 == pytest.approx(2 * (1 / 4) * (1 / 4) / (1 / 4 + 1 / 4))  # = 0.25


def test_rouge_l_partial_precision_recall():
    s = rouge_l(list("ABCBDAB"), list("BDCAB"))  # LCS=4, hyp=7, ref=5
    assert s.overlap == 4 and s.hyp_size == 7 and s.ref_size == 5
    assert s.precision == pytest.approx(4 / 7)
    assert s.recall == pytest.approx(4 / 5)
    assert s.f1 == pytest.approx(2 * (4 / 7) * (4 / 5) / (4 / 7 + 4 / 5))


def test_rouge_l_edge_empty_rules():
    assert rouge_l([], []).f1 == 1.0  # both empty -> agree there is nothing
    assert rouge_l([1, 2], []).f1 == 0.0  # ref empty, hyp not -> 0
    assert rouge_l([], [1, 2]).f1 == 0.0  # hyp empty, ref not -> 0


# --- token_rouge_l ---------------------------------------------------------


def _char_tokenize(text: str) -> list[int]:
    return [ord(c) for c in text if not c.isspace()]


def test_token_rouge_l_uses_tokenizer():
    s = token_rouge_l("abcabc", "abc", _char_tokenize)  # ref=[a,b,c], hyp x2 -> LCS=3
    assert s.overlap == 3 and s.hyp_size == 6 and s.ref_size == 3
    assert s.recall == 1.0 and s.precision == pytest.approx(0.5)


def test_token_rouge_l_truncates_to_max_tokens():
    # hyp truncated to first 3 tokens "abc" -> identical to ref.
    s = token_rouge_l("abcdef", "abc", _char_tokenize, max_tokens=3)
    assert s.hyp_size == 3 and s.f1 == 1.0


# --- word shingle F1 -------------------------------------------------------


def test_word_shingle_f1_identical_and_disjoint():
    assert word_shingle_f1("the quick brown fox jumps", "the quick brown fox jumps").f1 == 1.0
    assert word_shingle_f1("the quick brown fox jumps", "a b c d e").f1 == 0.0


def test_word_shingle_f1_partial():
    # Each side has 2 four-grams; they share exactly "the quick brown fox".
    s = word_shingle_f1("the quick brown fox jumps", "the quick brown fox runs")
    assert s.overlap == 1 and s.hyp_size == 2 and s.ref_size == 2
    assert s.f1 == pytest.approx(0.5)


def test_word_shingle_f1_short_doc_fallback():
    # Fewer than n words -> whole text is one shingle.
    assert word_shingle_f1("hello world", "hello world").f1 == 1.0
    assert word_shingle_f1("hello world", "hello there").f1 == 0.0


# --- word_rouge_l normalization --------------------------------------------


def test_word_rouge_l_is_case_and_whitespace_insensitive():
    assert word_rouge_l("Hello   World\nFoo", "hello world foo").f1 == 1.0


# --- offload label (abstention-aware) --------------------------------------


def test_offload_label_abstention_is_forced_negative():
    perfect = SimilarityScore(1.0, 1.0, 1.0, 0, 0, 0)
    assert offload_label(perfect, is_abstain=True) == 0  # abstain -> route to 8B even if F1=1


def test_offload_label_threshold_is_inclusive():
    assert offload_label(SimilarityScore(0, 0, 0.9, 0, 0, 0), is_abstain=False) == 1
    assert offload_label(SimilarityScore(0, 0, 0.8999, 0, 0, 0), is_abstain=False) == 0
