# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import numpy as np
import pytest
from jax import random
from levanter.testing.helpers import skip_if_no_torch, use_test_mesh

import haliax as hax
from haliax import Axis
from haliax.state_dict import to_torch_compatible_state_dict

from levanter.layers.attention import AttentionMask
from levanter.models.bert import BertConfig, BertForSequenceClassification, load_hf_bert_classifier
from levanter.models.classification import ClassificationExample, build_classifier


def _small_config(**kwargs) -> BertConfig:
    defaults = dict(
        max_seq_len=64,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        num_labels=2,
        pad_token_id=0,
    )
    defaults.update(kwargs)
    return BertConfig(**defaults)


def _pad_mask(Pos: Axis, num_real: int, *batch_axes: Axis) -> AttentionMask:
    """Segment mask marking the first ``num_real`` positions real (0) and the rest pad (-1)."""
    seg = hax.named(np.where(np.arange(Pos.size) < num_real, 0, -1).astype(np.int32), Pos)
    for ax in batch_axes:
        seg = seg.broadcast_axis(ax)
    return AttentionMask(is_causal=False).with_segment_ids(seg, seg)


@skip_if_no_torch
def test_classifier_hf_roundtrip(local_gpt2_tokenizer_path):
    """The weight-port oracle: a random HF BertForSequenceClassification round-trips into the
    Levanter port with matching logits on an input that includes pad tokens."""
    import torch  # noqa: PLC0415
    from transformers import BertConfig as HfBertConfig  # noqa: PLC0415
    from transformers import BertForSequenceClassification as HfBertForSequenceClassification  # noqa: PLC0415

    vocab_size = 512
    seq_len = 96
    num_real = 70

    hf_config = HfBertConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        type_vocab_size=2,
        pad_token_id=0,
        num_labels=2,
        attn_implementation="eager",
    )
    torch.random.manual_seed(0)
    torch_model = HfBertForSequenceClassification(hf_config)
    torch_model.eval()

    rng = np.random.default_rng(0)
    ids = rng.integers(1, vocab_size, size=seq_len).astype(np.int32)
    ids[num_real:] = 0  # pad tail
    attention_mask = (np.arange(seq_len) < num_real).astype(np.int64)

    with torch.no_grad():
        torch_out = (
            torch_model(
                input_ids=torch.from_numpy(ids).to(torch.int64).unsqueeze(0),
                attention_mask=torch.from_numpy(attention_mask).unsqueeze(0),
            )
            .logits[0]
            .numpy()
        )

    lev_config = BertConfig(
        max_seq_len=128,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        num_labels=2,
        pad_token_id=0,
        tokenizer=local_gpt2_tokenizer_path,
    )
    Pos = Axis("position", seq_len)
    input_ids = hax.named(ids, Pos)
    mask = _pad_mask(Pos, num_real)

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        model_path = f"{tmpdir}/torch_model"
        torch_model.save_pretrained(model_path)
        model = load_hf_bert_classifier(lev_config, model_path)

        @hax.named_jit
        def compute(m, tokens, mask):
            return m(tokens, mask)

        jax_out = np.asarray(compute(model, input_ids, mask).array)

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    assert np.isclose(torch_out, jax_out, rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"


def test_classifier_forward_and_loss():
    config = _small_config()
    Vocab = Axis("vocab", 512)
    Batch = Axis("batch", 3)
    Pos = Axis("position", config.max_seq_len)

    model = BertForSequenceClassification.init(Vocab, config, key=random.PRNGKey(0))
    tokens = hax.random.randint(random.PRNGKey(1), (Batch, Pos), 0, Vocab.size)
    mask = _pad_mask(Pos, 40, Batch)

    logits = model(tokens, mask)
    assert logits.axes == (Batch, config.Label), logits.axes
    assert np.isfinite(np.asarray(logits.array)).all()

    labels = hax.named(np.array([0, 1, 1]), Batch)
    ex = ClassificationExample.init(tokens, labels, attn_mask=mask)
    loss = model.compute_loss(ex)
    assert loss.ndim == 0 and np.isfinite(float(loss.array))


def test_pad_content_invariance():
    """Logits must not depend on the CONTENT of pad positions when they are masked via segment ids
    (CLS pooling + segment masking keeps pad values out of every real position's attention)."""
    config = _small_config()
    Vocab = Axis("vocab", 512)
    Pos = Axis("position", config.max_seq_len)
    num_real = 20

    model = BertForSequenceClassification.init(Vocab, config, key=random.PRNGKey(0))
    rng = np.random.default_rng(0)
    real = rng.integers(1, Vocab.size, size=Pos.size).astype(np.int32)

    padded = real.copy()
    padded[num_real:] = config.pad_token_id
    garbage = real.copy()
    garbage[num_real:] = rng.integers(1, Vocab.size, size=Pos.size - num_real)

    mask = _pad_mask(Pos, num_real)
    la = np.asarray(model(hax.named(padded, Pos), mask).array)
    lb = np.asarray(model(hax.named(garbage, Pos), mask).array)
    assert np.allclose(la, lb, rtol=1e-5, atol=1e-5), f"{la} != {lb}"


def test_build_classifier_dispatches_on_config_type():
    config = _small_config()
    with use_test_mesh():
        model = build_classifier(config, Axis("vocab", 128), key=random.PRNGKey(0), warm_start=False)
    assert isinstance(model, BertForSequenceClassification)
    assert len(model.bert.encoder.layer) == config.num_layers
    assert model.Label == config.Label
    logits = model(hax.random.randint(random.PRNGKey(1), (Axis("position", 16),), 0, 128))
    assert np.isfinite(np.asarray(logits.array)).all()


@skip_if_no_torch
def test_warm_start_partial_load(local_gpt2_tokenizer_path):
    """Warm-starting from a bare HF BertModel loads the encoder (and pooler) weights and keeps a
    random classifier head."""
    import torch  # noqa: PLC0415
    from transformers import BertConfig as HfBertConfig  # noqa: PLC0415
    from transformers import BertModel as HfBertModel  # noqa: PLC0415

    hf_config = HfBertConfig(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        type_vocab_size=2,
        pad_token_id=0,
    )
    torch.random.manual_seed(0)
    torch_model = HfBertModel(hf_config)
    torch_sd = {k: v.numpy() for k, v in torch_model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        model_path = f"{tmpdir}/torch_model"
        torch_model.save_pretrained(model_path)
        config = _small_config(reference_checkpoint=model_path, tokenizer=local_gpt2_tokenizer_path)
        model = build_classifier(config, Axis("vocab", 512), key=random.PRNGKey(0), warm_start=True)

    lev_sd = to_torch_compatible_state_dict(model)
    for hf_key in [
        "encoder.layer.0.attention.self.query.weight",
        "encoder.layer.1.output.dense.weight",
        "embeddings.word_embeddings.weight",
        "pooler.dense.weight",
    ]:
        np.testing.assert_allclose(lev_sd[f"bert.{hf_key}"], torch_sd[hf_key], rtol=1e-6, atol=1e-6)

    # classifier head is NOT in the base checkpoint: random init, correct shape, not zeros
    clf = lev_sd["classifier.weight"]
    assert clf.shape == (config.num_labels, config.hidden_dim)
    assert np.isfinite(clf).all() and not np.allclose(clf, 0.0)


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="not divisible"):
        BertConfig(hidden_dim=65, num_heads=4)
