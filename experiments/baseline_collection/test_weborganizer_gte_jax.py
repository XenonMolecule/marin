# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity contract for the gte-base-en-v1.5 JAX port against the HF reference.

``weborganizer_gte_jax`` re-implements the forward pass by hand because the HF
model's fast kernels (``use_memory_efficient_attention``, ``unpad_inputs``) are
xformers/CUDA-only, so on CPU or XLA the reference falls back to a path that
scores an 8k-token document at 0.03 docs/s. The hand port is what makes labelling
tens of millions of documents possible, and this file is what makes it
trustworthy — the module docstring calls this test "the contract".

Several architecture details here would silently corrupt logits rather than
raise, which is exactly why parity is asserted end-to-end on real forward passes
instead of unit-testing pieces:

  * post-norm residuals, not pre-norm
  * ``NewGatedMLP`` splits ``up_gate_proj`` as up-FIRST-then-gate
  * ``hidden_act="gelu"`` is exact/erf gelu, not the tanh approximation
  * NTK-scaled rope, whose effective inv_freq is not expressible as a plain
    theta change (the port sidesteps this by baking HF's materialised
    ``cos_cached``/``sin_cached`` buffers in as constants)

Ragged batches are covered specifically because padding mask and long rope
positions interacting is where a subtle bug would hide: a batch of one short and
one long sequence exercises both at once.

Downloads the ~550 MB checkpoint from HF on first run.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from experiments.baseline_collection.weborganizer_gte_jax import extract_params, forward, to_device
from experiments.baseline_collection.weborganizer_topic_smoke import URL_MODEL, load_model

# Downloads a 550 MB checkpoint and runs a 137M-param model on CPU, so this is
# minutes rather than seconds and is excluded from the default run. Invoke with
# `pytest -m slow experiments/baseline_collection/test_weborganizer_gte_jax.py`.
pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]

# fp32 logits of scale ~9; the port previously measured max|diff| 3.3e-6 against
# the reference, so this leaves three orders of magnitude of headroom while still
# catching any structural error.
RTOL = 1e-3
ATOL = 1e-3

DOCUMENTS = [
    "http://example.com/a\n\nThe mitochondria is the powerhouse of the cell. " * 3,
    "http://code.example.org/b\n\ndef fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n" * 5,
    "http://news.example.net/c\n\nLocal council approves the new transit budget after a long debate. " * 8,
    "http://shop.example.com/d\n\nBuy shoes. Free shipping.",
]


@pytest.fixture(scope="module")
def reference():
    """The HF model, its tokenizer, and the extracted JAX params."""
    config, tokenizer, hf_model = load_model(URL_MODEL)
    params = to_device(extract_params(hf_model), jnp.float32)
    return config, tokenizer, hf_model, params


def _hf_logits(hf_model, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        out = hf_model(
            input_ids=torch.from_numpy(ids).long(),
            attention_mask=torch.from_numpy(mask).long(),
        )
    return out.logits.float().numpy()


def _encode(tokenizer, texts: list[str], max_length: int) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="np",
    )
    return encoded["input_ids"].astype(np.int32), encoded["attention_mask"].astype(np.int32)


@pytest.mark.parametrize("max_length", [128, 512, 2048])
def test_logits_match_hf_reference(reference, max_length):
    """The port must reproduce HF logits at every sequence length we run at."""
    _, tokenizer, hf_model, params = reference
    ids, mask = _encode(tokenizer, DOCUMENTS, max_length)

    ours = np.asarray(forward(params, jnp.asarray(ids), jnp.asarray(mask)), dtype=np.float32)
    theirs = _hf_logits(hf_model, ids, mask)

    assert ours.shape == theirs.shape
    np.testing.assert_allclose(ours, theirs, rtol=RTOL, atol=ATOL)


def test_ragged_batch_matches_hf_reference(reference):
    """Padding mask and long rope positions must be correct *together*.

    A uniform-length batch would not catch a mask bug that only manifests when
    real and padded positions coexist at high rope indices.
    """
    _, tokenizer, hf_model, params = reference
    texts = [DOCUMENTS[3], DOCUMENTS[2] * 4, DOCUMENTS[0]]  # very short, very long, medium
    ids, mask = _encode(tokenizer, texts, 1024)
    assert mask.sum(axis=1).min() < 40 < mask.sum(axis=1).max(), "batch is not actually ragged"

    ours = np.asarray(forward(params, jnp.asarray(ids), jnp.asarray(mask)), dtype=np.float32)
    np.testing.assert_allclose(ours, _hf_logits(hf_model, ids, mask), rtol=RTOL, atol=ATOL)


def test_argmax_label_agrees_with_reference(reference):
    """Beyond numeric parity: the predicted topic itself must match.

    This is the thing we actually store, so assert it directly rather than
    inferring it from the logit tolerance.
    """
    config, tokenizer, hf_model, params = reference
    ids, mask = _encode(tokenizer, DOCUMENTS, 512)

    ours = np.asarray(forward(params, jnp.asarray(ids), jnp.asarray(mask)), dtype=np.float32)
    theirs = _hf_logits(hf_model, ids, mask)

    assert ours.argmax(-1).tolist() == theirs.argmax(-1).tolist()
    assert config.num_labels == 24


def test_query_blocking_is_exact(reference):
    """Sequences longer than QUERY_BLOCK must match those shorter than it.

    Attention is computed in blocks of QUERY_BLOCK=1024 queries to bound peak
    memory (a single [B,H,T,T] tensor at T=8192 is ~103 GB and OOMs). Blocking
    the query axis is exact and needs no online-softmax rescaling, since each
    block still softmaxes over all keys — this asserts that claim by crossing the
    block boundary.
    """
    _, tokenizer, hf_model, params = reference
    long_text = DOCUMENTS[2] * 12
    ids, mask = _encode(tokenizer, [long_text], 2048)
    assert mask.sum() > 1024, "test sequence does not cross the query-block boundary"

    ours = np.asarray(forward(params, jnp.asarray(ids), jnp.asarray(mask)), dtype=np.float32)
    np.testing.assert_allclose(ours, _hf_logits(hf_model, ids, mask), rtol=RTOL, atol=ATOL)
