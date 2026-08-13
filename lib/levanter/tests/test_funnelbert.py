# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import numpy as np
import pytest
from jax import random
from test_utils import skip_if_no_torch, use_test_mesh

import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionMask
from levanter.models.classification import ClassificationExample, build_classifier
from levanter.models.funnelbert import FunnelBertConfig, FunnelBertForSequenceClassification
from levanter.models.modernbert import ModernBertConfig


def _tiny_config(**overrides) -> FunnelBertConfig:
    kwargs = dict(
        max_seq_len=64,
        hidden_dim=64,
        intermediate_dim=128,
        num_full_layers=2,
        num_pooled_layers=2,
        pool_factor=4,
        num_heads=4,
        local_attention=16,
        num_labels=2,
        pad_token_id=0,
    )
    kwargs.update(overrides)
    return FunnelBertConfig(**kwargs)


def _pad_mask(Pos: Axis, num_real: int, batch: Axis | None = None) -> AttentionMask:
    seg = np.full((Pos.size,), -1, dtype=np.int32)
    seg[:num_real] = 0
    seg_named = hax.named(seg, Pos)
    if batch is not None:
        seg_named = seg_named.broadcast_axis(batch)
    return AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)


def test_config_validates_pool_factor_divisibility():
    with pytest.raises(ValueError, match="divisible by pool_factor"):
        _tiny_config(max_seq_len=66)


def test_forward_and_loss_finite():
    config = _tiny_config()
    Vocab = Axis("vocab", 1024)
    Batch = Axis("batch", 3)
    Pos = Axis("position", config.max_seq_len)

    model = FunnelBertForSequenceClassification.init(Vocab, config, key=random.PRNGKey(0))
    tokens = hax.random.randint(random.PRNGKey(1), (Batch, Pos), 0, Vocab.size)
    mask = _pad_mask(Pos, num_real=50, batch=Batch)

    logits = model(tokens, mask)
    assert logits.axes == (Batch, config.Label), logits.axes
    assert np.isfinite(np.asarray(logits.array)).all()

    labels = hax.named(np.array([0, 1, 1]), Batch)
    ex = ClassificationExample.init(tokens, labels, attn_mask=mask)
    loss = model.compute_loss(ex)
    assert loss.ndim == 0 and np.isfinite(float(loss.array))


def test_pooling_marks_all_pad_tail_windows_inactive():
    config = _tiny_config()
    Vocab = Axis("vocab", 1024)
    Pos = Axis("position", config.max_seq_len)
    model = FunnelBertForSequenceClassification.init(Vocab, config, key=random.PRNGKey(0))

    # 42 real tokens at pool_factor=4: windows 0..9 fully real, window 10 partial (2 real),
    # windows 11..15 all-pad -> inactive.
    num_real = 42
    mask = _pad_mask(Pos, num_real=num_real)
    x = hax.random.normal(random.PRNGKey(2), (Pos, config.Embed))
    pooled, pooled_mask, active = model._pool_windows(x, mask)

    active_np = np.asarray(active.array)
    assert active_np[:11].all()
    assert not active_np[11:].any()
    seg_np = np.asarray(pooled_mask.segment_ids[0].array)
    assert (seg_np[:11] == 0).all()
    assert (seg_np[11:] == -1).all()
    # inactive windows pool to exactly zero; the partial window averages only its real tokens
    pooled_np = np.asarray(pooled.array)
    assert (pooled_np[11:] == 0).all()
    want_partial = np.asarray(x.array)[40:42].mean(axis=0)
    assert np.allclose(pooled_np[10], want_partial, rtol=1e-5, atol=1e-5)


def test_pad_token_content_does_not_change_logits():
    config = _tiny_config()
    Vocab = Axis("vocab", 1024)
    Pos = Axis("position", config.max_seq_len)
    model = FunnelBertForSequenceClassification.init(Vocab, config, key=random.PRNGKey(0))

    num_real = 42
    mask = _pad_mask(Pos, num_real=num_real)
    tokens = np.asarray(hax.random.randint(random.PRNGKey(3), (Pos,), 0, Vocab.size).array).copy()
    scrambled = tokens.copy()
    scrambled[num_real:] = (scrambled[num_real:] + 17) % Vocab.size

    logits_a = np.asarray(model(hax.named(tokens, Pos), mask).array)
    logits_b = np.asarray(model(hax.named(scrambled, Pos), mask).array)
    assert np.allclose(logits_a, logits_b, rtol=1e-5, atol=1e-5)
    # sanity: changing REAL tokens does change logits
    real_changed = tokens.copy()
    real_changed[:num_real] = (real_changed[:num_real] + 17) % Vocab.size
    logits_c = np.asarray(model(hax.named(real_changed, Pos), mask).array)
    assert not np.allclose(logits_a, logits_c)


@skip_if_no_torch
def test_warm_start_keeps_embeddings_and_bottom_layers(local_gpt2_tokenizer_path):
    """Warm start loads the reference MLM at full depth and reuses embeddings + the first
    ``num_full_layers`` layers; pooled layers/head/classifier stay random."""
    import torch  # noqa: PLC0415
    from transformers.models.modernbert import modeling_modernbert  # noqa: PLC0415
    from transformers.models.modernbert.configuration_modernbert import (  # noqa: PLC0415
        ModernBertConfig as HfModernBertConfig,
    )

    hf_config = HfModernBertConfig(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=128,
        global_attn_every_n_layers=3,
        local_attention=16,
        tie_word_embeddings=True,
        pad_token_id=0,
        _attn_implementation="eager",
    )
    torch.random.manual_seed(0)
    torch_model = modeling_modernbert.ModernBertForMaskedLM(hf_config)

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        model_path = f"{tmpdir}/torch_model"
        torch_model.save_pretrained(model_path)
        funnel_config = _tiny_config(
            max_seq_len=128,
            reference_checkpoint=model_path,
            tokenizer=local_gpt2_tokenizer_path,
        )
        funnel = build_classifier(funnel_config, Axis("vocab", 1024), key=random.PRNGKey(0), warm_start=True)
        assert isinstance(funnel, FunnelBertForSequenceClassification)
        assert len(funnel.full_layers) == 2
        assert len(funnel.pooled_layers) == 2

        full_config = ModernBertConfig(
            max_seq_len=128,
            hidden_dim=64,
            intermediate_dim=128,
            num_layers=4,
            num_heads=4,
            local_attention=16,
            num_labels=2,
            pad_token_id=0,
            reference_checkpoint=model_path,
            tokenizer=local_gpt2_tokenizer_path,
        )
        full = build_classifier(full_config, Axis("vocab", 1024), key=random.PRNGKey(0), warm_start=True)

        assert np.allclose(
            np.array(funnel.embeddings.tok_embeddings.weight.array),
            np.array(full.model.embeddings.tok_embeddings.weight.array),
        )
        for i in range(2):
            got = np.array(funnel.full_layers[i].mlp.Wi.weight.array)
            want = np.array(full.model.layers[i].mlp.Wi.weight.array)
            assert np.allclose(got, want), f"layer {i} Wi mismatch after warm start"


def test_build_classifier_dispatches_to_funnelbert():
    config = _tiny_config()
    with use_test_mesh():
        model = build_classifier(config, Axis("vocab", 128), key=random.PRNGKey(0), warm_start=False)
    assert isinstance(model, FunnelBertForSequenceClassification)
    assert len(model.full_layers) == config.num_full_layers
    assert len(model.pooled_layers) == config.num_pooled_layers
    # subclass forces the inherited num_layers to the full-res depth
    assert config.num_layers == config.num_full_layers
