# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure CPU stages."""

from __future__ import annotations

from experiments.baseline_collection.fasttext_useful_classifier import to_fasttext_text
from experiments.fast_curation import preprocess

SAMPLE_HTML = (
    "<html><head><title>T</title>"
    "<script>var x = 1; // SHOULD BE STRIPPED</script></head>"
    "<body><h1>Real Heading</h1>"
    "<p>This is the main article body with several informative sentences. "
    "It should survive boilerplate removal because it is the real content here.</p>"
    "<p>A second substantive paragraph continues the article discussion at length.</p>"
    "<nav>home about contact</nav></body></html>"
)


def test_ft_text_matches_classifier_golden():
    """preprocess.ft_text must be byte-identical to the classifier's to_fasttext_text(body_strip)."""
    assert preprocess.ft_text(SAMPLE_HTML) == to_fasttext_text(SAMPLE_HTML, "body_strip")


def test_ft_text_strips_script_lowercases_collapses():
    out = preprocess.ft_text(SAMPLE_HTML)
    assert "should be stripped" not in out  # <script> body removed
    assert out == out.lower()  # lowercased
    assert "  " not in out  # whitespace collapsed
    assert "real heading" in out  # body content kept


def test_fasttext_useful_prob_empty_is_zero():
    class _FakeModel:
        class f:
            @staticmethod
            def predict(text, k, threshold, mode):
                return []

    assert preprocess.fasttext_useful_prob(_FakeModel(), "") == 0.0


def test_fasttext_useful_prob_parses_both_tuple_orders():
    """The low-level predict can yield (prob, label) or (label, prob); both must parse."""

    class _ProbFirst:
        class f:
            @staticmethod
            def predict(text, k, threshold, mode):
                return [(0.91, "__label__useful"), (0.09, "__label__no_useful")]

    class _LabelFirst:
        class f:
            @staticmethod
            def predict(text, k, threshold, mode):
                return [("__label__no_useful", 0.09), ("__label__useful", 0.42)]

    assert preprocess.fasttext_useful_prob(_ProbFirst(), "hi") == 0.91
    assert preprocess.fasttext_useful_prob(_LabelFirst(), "hi") == 0.42


def test_justext_extracts_article_body():
    text = preprocess.justext_text(SAMPLE_HTML)
    assert text  # non-empty
    assert "informative sentences" in text.lower()


def test_justext_empty_on_garbage():
    assert preprocess.justext_text("") == ""


def test_tokenize_trunc_respects_max_length():
    tok = preprocess.load_tokenizer("answerdotai/ModernBERT-base")
    long_text = "word " * 5000
    ids = preprocess.tokenize_trunc(tok, long_text, 128)
    assert len(ids) <= 128
    assert all(isinstance(i, int) for i in ids)


def test_tokenize_trunc_batch_byte_identical_to_per_doc():
    """The batched tokenize must return ids byte-identical to the per-doc path — only faster.

    Covers empty text (special-tokens-only), short text, over-``max_length`` truncation, UTF-8, and
    over-the-``max_length*8``-char-cap input; ``batch_size=2`` forces the multi-sub-batch loop.
    """
    tok = preprocess.load_tokenizer("answerdotai/ModernBERT-base")
    max_length = 128
    # Heterogeneous long doc: distinct tokens at head vs tail, so a wrong truncation *direction*
    # (left instead of right) would change the ids and fail parity — a uniform "word "*N would not.
    heterogeneous = " ".join(f"tok{i}word{i * 7 % 13}" for i in range(4000))
    texts = [
        "",
        "short doc",
        "word " * 5000,  # exceeds max_length -> truncated
        heterogeneous,  # exceeds max_length with position-sensitive content
        "Some UTF-8: café π 漢字 emoji 🚀 mixed in.",
        "a" * (max_length * 8 + 50),  # exceeds the char cap
    ]
    per_doc = [preprocess.tokenize_trunc(tok, t, max_length) for t in texts]
    batched = preprocess.tokenize_trunc_batch(tok, texts, max_length, batch_size=2)
    assert batched == per_doc


def test_tokenize_trunc_batch_empty_list():
    tok = preprocess.load_tokenizer("answerdotai/ModernBERT-base")
    assert preprocess.tokenize_trunc_batch(tok, [], 128) == []
