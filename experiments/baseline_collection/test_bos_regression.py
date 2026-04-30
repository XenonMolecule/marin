# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local tests that reproduce the BOS regression in post-2026-04-10 caches.

Background
----------
`nemotron_full` (built 2026-04-11) and `llm_curated` (built 2026-04-19) were
tokenized without BOS (0.00% docs start with BOS). The other 4 baseline caches
were built 2026-04-06/07 and have 100% BOS. Between the two build windows, the
Levanter commit ``f03aa9ecb`` ("Cap homogeneous-run length when calling Rust BPE
tokenizers") changed ``MarinTokenizer.encode_batch`` to always pass
``add_special_tokens=False`` to the underlying Rust encoder, and manually
prepend BOS only when the caller passes ``add_special_tokens=True``.

`BatchTokenizer.__call__` (in ``lib/levanter/src/levanter/data/text/_batch_tokenizer.py``)
calls ``self.tokenizer.encode_batch(batch_text)`` without passing
``add_special_tokens`` — the default is ``False`` — so no BOS is added.

These tests reproduce the regression, verify the workaround (explicitly passing
``add_special_tokens=True``), and confirm that the older caches exhibited BOS
because pre-``f03aa9ecb`` the underlying Rust encoder's ``add_special_tokens``
default was effectively ``True`` (via post-processor) for Llama-3.1.

Run locally:

    uv run pytest experiments/baseline_collection/test_bos_regression.py -v
"""

from __future__ import annotations

import os

import pytest

# Keep tests offline-capable; if HF_TOKEN is absent, skip (Llama-3.1 is gated).
_HF_TOKEN = os.environ.get("HF_TOKEN")


@pytest.fixture(scope="module")
def llama31_tokenizer():
    if not _HF_TOKEN:
        pytest.skip("HF_TOKEN not set; cannot download gated Llama-3.1-8B tokenizer")
    from levanter.tokenizers import load_tokenizer

    return load_tokenizer("meta-llama/Meta-Llama-3.1-8B")


def test_llama31_bos_and_eos_ids(llama31_tokenizer):
    """Baseline sanity — Llama-3.1 exposes BOS (128000) and EOS (128001)."""
    assert llama31_tokenizer.bos_token_id == 128000
    assert llama31_tokenizer.eos_token_id == 128001


def test_tokenizer_encode_with_and_without_add_special_tokens(llama31_tokenizer):
    """Document the underlying encoder behavior w.r.t. BOS injection.

    The core question: does ``MarinTokenizer.encode("hi", add_special_tokens=True)``
    prepend BOS? The answer determines whether ``BatchTokenizer`` needs to
    manually add BOS.
    """
    tok = llama31_tokenizer
    bos = tok.bos_token_id

    # add_special_tokens=True should prepend BOS on Llama-3.1
    ids_true = tok.encode("hello world", add_special_tokens=True)
    assert ids_true[0] == bos, f"expected BOS at start when add_special_tokens=True, got {ids_true[:5]}"

    # add_special_tokens=False should NOT prepend BOS
    ids_false = tok.encode("hello world", add_special_tokens=False)
    assert ids_false[0] != bos, f"expected NO BOS when add_special_tokens=False, got {ids_false[:5]}"


def test_encode_batch_without_add_special_tokens_drops_bos(llama31_tokenizer):
    """Documents the raw behavior of `MarinTokenizer.encode_batch`: its default is
    ``add_special_tokens=False``, which produces NO BOS. This is the underlying
    cause of the nemotron_full / llm_curated missing-BOS regression — the old
    ``BatchTokenizer.__call__`` was calling encode_batch with this default.
    """
    tok = llama31_tokenizer
    bos = tok.bos_token_id
    ids_list = tok.encode_batch(["hello world", "goodbye moon"])
    for ids in ids_list:
        assert ids[0] != bos, (
            "Sanity check: encode_batch with default add_special_tokens=False must NOT return BOS at start. "
            "If this fails the default changed — BatchTokenizer's fix may need updating."
        )


def test_encode_batch_with_add_special_tokens_true_adds_bos(llama31_tokenizer):
    """The fix: explicitly pass add_special_tokens=True."""
    tok = llama31_tokenizer
    bos = tok.bos_token_id

    ids_list = tok.encode_batch(["hello world", "goodbye moon"], add_special_tokens=True)
    for ids in ids_list:
        assert ids[0] == bos, f"expected BOS at start with add_special_tokens=True, got {ids[:5]}"


def test_batch_tokenizer_end_to_end_produces_bos_and_eos(llama31_tokenizer):
    """End-to-end: with the fix applied, ``BatchTokenizer.__call__`` produces
    input_ids starting with BOS and ending with EOS — matching both the
    pre-2026-04-10 behavior AND the eval-cache format (paloma / uncheatable).

    This test PROTECTS AGAINST REGRESSION: if ``encode_batch`` is later called
    without ``add_special_tokens=True`` again, this test will fail.
    """
    from levanter.data.text._batch_tokenizer import BatchTokenizer

    bt = BatchTokenizer(llama31_tokenizer, enforce_bos=True, enforce_eos=True, text_field="text")
    bos = llama31_tokenizer.bos_token_id
    eos = llama31_tokenizer.eos_token_id

    # For Llama-3.1 the tokenizer handles BOS via the post-processor, so
    # ``_need_to_add_bos`` is False — we rely on the tokenizer, not on a
    # manual string prepend.
    assert bt._need_to_add_bos is False

    # Post-fix behavior: BOS prepended by tokenizer via add_special_tokens=True,
    # EOS appended manually via _need_to_add_eos path.
    out = bt([{"text": "hello world"}])
    ids = out[0]["input_ids"]
    assert ids[0] == bos, f"expected BOS at start, got {ids[:5]}"
    assert ids[-1] == eos, f"expected EOS at end, got {ids[-5:]}"
    # Sanity: no double BOS.
    assert ids[1] != bos, f"double BOS detected at positions 0,1: {ids[:5]}"


def test_batch_tokenizer_batch_of_multiple_docs(llama31_tokenizer):
    """Every doc in a batch should get its own BOS and EOS."""
    from levanter.data.text._batch_tokenizer import BatchTokenizer

    bt = BatchTokenizer(llama31_tokenizer, enforce_bos=True, enforce_eos=True, text_field="text")
    bos = llama31_tokenizer.bos_token_id
    eos = llama31_tokenizer.eos_token_id

    docs = [{"text": t} for t in ["alpha beta", "gamma delta epsilon", "zeta"]]
    out = bt(docs)
    for i, rec in enumerate(out):
        ids = rec["input_ids"]
        assert ids[0] == bos, f"doc {i}: expected BOS start, got {ids[:5]}"
        assert ids[-1] == eos, f"doc {i}: expected EOS end, got {ids[-5:]}"
        assert ids[1] != bos, f"doc {i}: double BOS at positions 0,1: {ids[:5]}"


def test_proposed_fix_for_batch_tokenizer(llama31_tokenizer):
    """Demonstrates the one-line fix: pass ``add_special_tokens=True``.

    If we patch BatchTokenizer.__call__ to pass add_special_tokens=True to
    encode_batch, BOS appears at every doc start (matching pre-4-10 behavior).
    """
    tok = llama31_tokenizer
    bos = tok.bos_token_id
    eos = tok.eos_token_id

    # Simulate fixed call site: encode_batch with add_special_tokens=True
    texts = ["hello world", "goodbye moon"]
    encoded = tok.encode_batch(texts, add_special_tokens=True)

    # Manually append EOS the way BatchTokenizer does:
    # (the real code prepends " " + eos as string so tokenizer splits correctly,
    # but for this test we just check BOS is at start)
    for ids in encoded:
        assert ids[0] == bos, f"fix failed: expected BOS start, got {ids[:5]}"
        # EOS would be appended manually by BatchTokenizer via string concat;
        # underlying encode_batch doesn't append EOS for Llama-3.1.
        _ = eos
