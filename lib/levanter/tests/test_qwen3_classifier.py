# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the Qwen3 discriminative classifier (no network / no TPU).

Validates the pieces that are easy to get wrong before spending TPU time: last-non-pad-token
pooling, the NO-token head warm-start, a finite forward/loss on a tiny model, and the prompt-wrapping
dataset's truncation + pool-position bookkeeping.
"""

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax
from haliax import Axis
from levanter.layers.attention import AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.main.train_decoder_classifier import DecoderPromptClassificationDataset
from levanter.models.modernbert import ClassificationExample
from levanter.models.qwen import Qwen3Config, Qwen3ForSequenceClassification, Qwen3LMHeadModel

DECISION = 3
VOCAB = 32


def _tiny_config() -> Qwen3Config:
    return Qwen3Config(
        max_seq_len=64,
        hidden_dim=32,
        intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        head_dim=8,
        tie_word_embeddings=True,
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
        tokenizer="Qwen/Qwen3-0.6B",
    )


def _tiny_model(warm_head: bool = True):
    config = _tiny_config()
    Vocab = Axis("vocab", VOCAB)
    Label = Axis("label", 2)
    lm = Qwen3LMHeadModel.init(Vocab, config, key=jrandom.PRNGKey(0))
    model = Qwen3ForSequenceClassification.from_lm_head_model(
        lm, config, Label, DECISION, key=jrandom.PRNGKey(1), warm_head=warm_head
    )
    return model, config, Vocab, Label


def _batch(config, pos_lengths):
    """Build a {Batch, Pos} token batch + causal+segment mask given per-row real lengths."""
    Pos = config.max_Pos
    b = len(pos_lengths)
    Batch = Axis("batch", b)
    ids = np.zeros((b, Pos.size), dtype=np.int32)
    seg = np.full((b, Pos.size), -1, dtype=np.int32)
    for r, n in enumerate(pos_lengths):
        ids[r, :n] = np.arange(1, n + 1) % VOCAB
        seg[r, :n] = 0
    tokens = hax.named(ids, (Batch, Pos))
    seg_named = hax.named(seg, (Batch, Pos))
    mask = AttentionMask(is_causal=True).with_segment_ids(seg_named, seg_named)
    return tokens, mask, Batch


def test_warm_start_head_uses_decision_token_row():
    model, _, _, Label = _tiny_model(warm_head=True)
    emb = model.lm.embeddings.token_embeddings.weight
    expected = np.asarray(emb["vocab", DECISION].array)
    w0 = np.asarray(model.classifier.weight["label", 0].array)
    w1 = np.asarray(model.classifier.weight["label", 1].array)
    np.testing.assert_allclose(w0, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(w1, 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(model.classifier.bias.array), 0.0, atol=1e-6)


def test_pool_selects_last_non_pad_position():
    model, config, _, _ = _tiny_model()
    Pos = config.max_Pos
    Embed = config.Embed
    Batch = Axis("batch", 2)
    # Distinct value per position so we can identify which one was pooled.
    hidden = hax.named(
        jnp.broadcast_to(jnp.arange(Pos.size, dtype=jnp.float32)[None, :, None], (2, Pos.size, Embed.size)),
        (Batch, Pos, Embed),
    )
    seg = np.full((2, Pos.size), -1, dtype=np.int32)
    seg[0, :5] = 0  # last real index 4
    seg[1, :10] = 0  # last real index 9
    seg_named = hax.named(seg, (Batch, Pos))
    mask = AttentionMask(is_causal=True).with_segment_ids(seg_named, seg_named)
    pooled = np.asarray(model._pool(hidden, mask).array)
    assert np.allclose(pooled[0], 4.0)
    assert np.allclose(pooled[1], 9.0)


def test_forward_and_loss_are_finite():
    model, config, _, Label = _tiny_model()
    tokens, mask, Batch = _batch(config, [12, 20])
    logits = model(tokens, mask)
    assert logits.axes == (Batch, Label) or set(a.name for a in logits.axes) == {"batch", "label"}
    assert np.all(np.isfinite(np.asarray(logits.array)))

    example = ClassificationExample.init(
        tokens=tokens, label=hax.named(np.array([0, 1], dtype=np.int32), Batch), attn_mask=mask
    )
    loss = model.compute_loss(example)
    assert np.isfinite(float(loss.array)) and float(loss.array) > 0


class _FakeTokenizer:
    """Char-count tokenizer: each whitespace-split word -> one id, truncatable."""

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (i % 5) for i, _ in enumerate(text.split())]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_dataset_assembles_prompt_and_pools_at_tail():
    Pos = Axis("position", 24)
    head_ids = [1, 2, 3]
    tail_ids = [7, 8, 58]  # ends at the decision token
    ds = DecoderPromptClassificationDataset(
        texts=["a b c d e f g h i j k l m n o p q r s t"],  # 20 words
        labels=[1],
        tokenizer=_FakeTokenizer(),
        Pos=Pos,
        pad_token_id=0,
        head_ids=head_ids,
        tail_ids=tail_ids,
    )
    assert ds.budget == 24 - 3 - 3  # 18
    ex = ds._encode_one(ds.texts[0], 1)
    toks = np.asarray(ex.tokens.array)
    seg = np.asarray(ex.attn_mask.segment_ids[0].array)
    n = 3 + 18 + 3  # head + truncated html (capped at budget 18) + tail = 24, fills Pos exactly
    assert n == 24
    assert toks[:3].tolist() == head_ids
    assert toks[-3:].tolist() == tail_ids  # tail (and the decision token) is the last real token
    assert seg[n - 1] == 0  # last real position is the decision token
    assert (seg >= 0).sum() == n
