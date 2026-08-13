# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from levanter.testing.helpers import use_test_mesh

import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionMask
from levanter.models.classification import ClassificationExample, build_classifier, save_eqx_classifier
from levanter.models.pooled_transformer import (
    PooledTransformerClassifier,
    PooledTransformerConfig,
    count_params,
    load_pooled_transformer_classifier,
)

VOCAB = Axis("vocab", 128)


def _tiny_config(**overrides) -> PooledTransformerConfig:
    defaults = dict(
        max_seq_len=64,
        pool_window=8,
        pool_kind="meanmaxmin",
        embed_dim=16,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.0,
        num_labels=2,
        pad_token_id=0,
    )
    defaults.update(overrides)
    return PooledTransformerConfig(**defaults)


def _model(config: PooledTransformerConfig, seed: int = 0) -> PooledTransformerClassifier:
    return PooledTransformerClassifier.init(VOCAB, config, key=random.PRNGKey(seed))


def _segment_mask(lengths: list[int], Batch: Axis, Pos: Axis) -> AttentionMask:
    """Pad-exclusion mask in the classifier-dataset convention: real = segment 0, pad = -1."""
    seg = np.full((Batch.size, Pos.size), -1, dtype=np.int32)
    for r, n in enumerate(lengths):
        seg[r, :n] = 0
    seg_named = hax.named(seg, (Batch, Pos))
    return AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)


def test_forward_and_loss_finite():
    config = _tiny_config()
    Batch = Axis("batch", 3)
    Pos = config.max_Pos
    model = _model(config)

    tokens = hax.random.randint(random.PRNGKey(1), (Batch, Pos), 0, VOCAB.size)
    mask = _segment_mask([64, 40, 9], Batch, Pos)
    logits = model(tokens, mask)

    assert logits.axes == (Batch, config.Label), logits.axes
    assert np.isfinite(np.asarray(logits.array)).all()

    labels = hax.named(np.array([0, 1, 1]), Batch)
    ex = ClassificationExample.init(tokens, labels, attn_mask=mask)
    loss = model.compute_loss(ex)
    assert loss.ndim == 0
    assert loss.dtype == jnp.float32
    assert np.isfinite(float(loss.array))


def test_unbatched_input():
    config = _tiny_config()
    Pos = config.max_Pos
    model = _model(config)
    tokens = hax.random.randint(random.PRNGKey(2), (Pos,), 0, VOCAB.size)
    logits = model(tokens)  # no mask: falls back to ids != pad_token_id
    assert logits.axes == (config.Label,)
    assert np.isfinite(np.asarray(logits.array)).all()


@pytest.mark.parametrize("pool_kind", ["mean", "max", "meanmaxmin", "attn"])
@pytest.mark.parametrize("final_pool", ["mean", "attn"])
def test_pad_invariance(pool_kind, final_pool):
    """Token values at pad positions (segment -1) must not affect the logits at all."""
    config = _tiny_config(pool_kind=pool_kind, final_pool=final_pool)
    Batch = Axis("batch", 2)
    Pos = config.max_Pos
    model = _model(config)

    lengths = [40, 5]
    mask = _segment_mask(lengths, Batch, Pos)
    ids = np.asarray(hax.random.randint(random.PRNGKey(3), (Batch, Pos), 0, VOCAB.size).array).copy()
    mutated = ids.copy()
    for r, n in enumerate(lengths):
        mutated[r, n:] = (mutated[r, n:] + 7) % VOCAB.size  # scribble over every pad position

    la = np.asarray(model(hax.named(ids, (Batch, Pos)), mask).array)
    lb = np.asarray(model(hax.named(mutated, (Batch, Pos)), mask).array)
    assert np.array_equal(la, lb), f"pad tokens leaked into logits ({pool_kind}/{final_pool}): {la} vs {lb}"


@pytest.mark.parametrize("pool_kind", ["mean", "max", "meanmaxmin", "attn"])
def test_all_pool_kinds_run(pool_kind):
    config = _tiny_config(pool_kind=pool_kind)
    Batch = Axis("batch", 2)
    Pos = config.max_Pos
    model = _model(config)
    tokens = hax.random.randint(random.PRNGKey(4), (Batch, Pos), 0, VOCAB.size)
    mask = _segment_mask([64, 20], Batch, Pos)
    logits = model(tokens, mask)
    assert np.isfinite(np.asarray(logits.array)).all()


def test_doc_shorter_than_one_window():
    """A doc shorter than pool_window leaves every later window empty -> still finite logits."""
    config = _tiny_config()
    Batch = Axis("batch", 1)
    Pos = config.max_Pos
    model = _model(config)
    tokens = hax.random.randint(random.PRNGKey(5), (Batch, Pos), 0, VOCAB.size)
    mask = _segment_mask([3], Batch, Pos)  # 3 real tokens < pool_window=8
    logits = model(tokens, mask)
    assert np.isfinite(np.asarray(logits.array)).all()


def test_fully_padded_doc_is_finite():
    config = _tiny_config()
    Batch = Axis("batch", 1)
    Pos = config.max_Pos
    model = _model(config)
    tokens = hax.random.randint(random.PRNGKey(6), (Batch, Pos), 0, VOCAB.size)
    mask = _segment_mask([0], Batch, Pos)
    logits = model(tokens, mask)
    assert np.isfinite(np.asarray(logits.array)).all()


def test_inference_is_deterministic():
    config = _tiny_config(dropout=0.5)  # dropout configured, but key=None -> inference, no dropout
    Batch = Axis("batch", 2)
    Pos = config.max_Pos
    model = _model(config)
    tokens = hax.random.randint(random.PRNGKey(7), (Batch, Pos), 0, VOCAB.size)
    mask = _segment_mask([64, 30], Batch, Pos)
    la = np.asarray(model(tokens, mask).array)
    lb = np.asarray(model(tokens, mask).array)
    assert np.array_equal(la, lb)


def test_dropout_key_path_finite():
    config = _tiny_config(dropout=0.5)
    Batch = Axis("batch", 2)
    Pos = config.max_Pos
    model = _model(config)
    tokens = hax.random.randint(random.PRNGKey(8), (Batch, Pos), 0, VOCAB.size)
    labels = hax.named(np.array([0, 1]), Batch)
    ex = ClassificationExample.init(tokens, labels, attn_mask=_segment_mask([64, 30], Batch, Pos))
    loss = model.compute_loss(ex, key=random.PRNGKey(9))
    assert np.isfinite(float(loss.array))


def test_half_padded_window_mean_pool_matches_real_half():
    """Port-semantics check: mean pooling a half-padded window equals the mean of the real half."""
    config = _tiny_config(pool_kind="mean")
    model = _model(config)
    w, e = config.pool_window, config.embed_dim
    emb = np.asarray(random.normal(random.PRNGKey(10), (1, config.max_seq_len, e)), dtype=np.float32)
    mask = np.zeros((1, config.max_seq_len), dtype=np.float32)
    mask[0, : w // 2] = 1.0  # first window: half real, half pad; all later windows empty

    pooled, valid = model._pool_windows(jnp.asarray(emb), jnp.asarray(mask))
    expected = emb[0, : w // 2].mean(axis=0)
    np.testing.assert_allclose(np.asarray(pooled)[0, 0], expected, rtol=1e-5, atol=1e-6)
    assert np.asarray(valid)[0].tolist() == [1.0] + [0.0] * (config.num_super_tokens - 1)


def test_registry_dispatch_and_named_jit():
    config = _tiny_config()
    with use_test_mesh():
        model = build_classifier(config, VOCAB, key=random.PRNGKey(0), warm_start=False)
        assert isinstance(model, PooledTransformerClassifier)

        Batch = Axis("batch", 2)
        Pos = config.max_Pos
        tokens = hax.random.randint(random.PRNGKey(11), (Batch, Pos), 0, VOCAB.size)
        mask = _segment_mask([64, 12], Batch, Pos)

        @hax.named_jit
        def _probs(m, ids, msk):
            logits = m(ids, msk).astype(jnp.float32)
            return hax.nn.softmax(logits, axis="label")["label", 1]

        probs = np.asarray(_probs(model, tokens, mask).array)
        assert probs.shape == (2,)
        assert np.isfinite(probs).all()


def test_warm_start_raises():
    with pytest.raises(ValueError, match="no pretrained weights"):
        build_classifier(_tiny_config(), VOCAB, key=random.PRNGKey(0), warm_start=True)


def test_save_load_roundtrip(tmp_path):
    config = _tiny_config()
    model = _model(config, seed=13)
    path = str(tmp_path / "clf")
    save_eqx_classifier(config, model, path)
    loaded = load_pooled_transformer_classifier(config, path, vocab_size=VOCAB.size)

    Pos = config.max_Pos
    Batch = Axis("batch", 2)
    tokens = hax.random.randint(random.PRNGKey(14), (Batch, Pos), 0, VOCAB.size)
    mask = _segment_mask([64, 33], Batch, Pos)
    la = np.asarray(model(tokens, mask).array)
    lb = np.asarray(loaded(tokens, mask).array)
    assert np.array_equal(la, lb)


def test_resize_vocab():
    config = _tiny_config()
    model = _model(config)
    grown = model.resize_vocab(VOCAB.size + 16, key=random.PRNGKey(15))
    assert grown.embed.shape == (VOCAB.size + 16, config.embed_dim)
    shrunk = model.resize_vocab(VOCAB.size - 16)
    assert shrunk.embed.shape == (VOCAB.size - 16, config.embed_dim)
    # original rows are preserved
    assert np.array_equal(np.asarray(grown.embed[: VOCAB.size]), np.asarray(model.embed))


def test_config_validation():
    with pytest.raises(ValueError, match="divisible by pool_window"):
        _tiny_config(max_seq_len=60)
    with pytest.raises(ValueError, match="pool_kind"):
        _tiny_config(pool_kind="bogus")
    with pytest.raises(ValueError, match="final_pool"):
        _tiny_config(final_pool="cls")
    with pytest.raises(ValueError, match="not divisible by num_heads"):
        _tiny_config(hidden_dim=30)


def test_flops_and_params_sanity():
    default = PooledTransformerConfig()
    fpt = default.flops_per_token(50368, 8192)
    assert 0 < fpt < 5e6  # the whole point of the architecture: ~1M flops/token
    model = _model(_tiny_config())
    assert count_params(model) > 0
    assert dataclasses.is_dataclass(default)
