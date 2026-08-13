# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import random
from levanter.testing.helpers import use_test_mesh

import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionMask
from levanter.models.bigdn import BigdnConfig, BigdnForSequenceClassification
from levanter.models.classification import ClassificationExample, build_classifier

VOCAB = Axis("vocab", 256)
PAD = 0


def _tiny_config() -> BigdnConfig:
    return BigdnConfig(
        max_seq_len=32,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=2,
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=8,
        head_v_dim=8,
        gdn_chunk_size=8,
        num_labels=2,
        vocab_size=VOCAB.size,
        pad_token_id=PAD,
    )


def _segment_mask(seg: np.ndarray, axes) -> AttentionMask:
    seg_named = hax.named(seg.astype(np.int32), axes)
    return AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)


def _padded_batch(config: BigdnConfig, key, n_real: list[int]):
    """Random real tokens (ids >= 1) right-padded with PAD; returns (tokens, mask, Batch, Pos)."""
    Batch = Axis("batch", len(n_real))
    Pos = Axis("position", config.max_seq_len)
    ids = np.asarray(hax.random.randint(key, (Batch, Pos), 1, VOCAB.size).array).copy()
    seg = np.zeros((Batch.size, Pos.size), dtype=np.int32)
    for r, n in enumerate(n_real):
        ids[r, n:] = PAD
        seg[r, n:] = -1
    return hax.named(ids, (Batch, Pos)), _segment_mask(seg, (Batch, Pos)), Batch, Pos


def test_forward_and_loss_finite():
    config = _tiny_config()
    model = BigdnForSequenceClassification.init(VOCAB, config, key=random.PRNGKey(0))
    tokens, mask, Batch, _ = _padded_batch(config, random.PRNGKey(1), [32, 20, 5])

    logits = model(tokens, mask)
    assert logits.axes == (Batch, config.Label)
    assert config.Label.name == "label"
    assert np.isfinite(np.asarray(logits.array)).all()
    # the eval path softmaxes over axis="label"
    probs = hax.nn.softmax(logits, axis="label")
    assert np.allclose(np.asarray(probs.array).sum(axis=-1), 1.0, atol=1e-5)

    labels = hax.named(np.array([0, 1, 1], dtype=np.int32), Batch)
    loss = model.compute_loss(ClassificationExample.init(tokens, labels, attn_mask=mask))
    assert loss.ndim == 0 and np.isfinite(float(loss.array))


def test_unbatched_input_is_normalized():
    config = _tiny_config()
    Pos = Axis("position", config.max_seq_len)
    model = BigdnForSequenceClassification.init(VOCAB, config, key=random.PRNGKey(0))
    tokens = hax.random.randint(random.PRNGKey(1), (Pos,), 1, VOCAB.size)
    logits = model(tokens, None)
    assert logits.axes == (config.Label,)
    assert np.isfinite(np.asarray(logits.array)).all()


def test_pad_invariance_is_exact():
    """Logits must not depend on token ids in the pad region (GDN zeroes pad inputs before
    mixing; pooling masks pads), so two runs differing only in pad content match exactly."""
    config = _tiny_config()
    model = BigdnForSequenceClassification.init(VOCAB, config, key=random.PRNGKey(0))
    tokens, mask, _, Pos = _padded_batch(config, random.PRNGKey(1), [12, 20])

    scribbled = np.asarray(tokens.array).copy()
    scribbled[0, 12:] = 123  # arbitrary garbage in the pad region
    scribbled[1, 20:] = 45
    tokens_scribbled = hax.named(scribbled, tokens.axes)

    la = np.asarray(model(tokens, mask).array)
    lb = np.asarray(model(tokens_scribbled, mask).array)
    np.testing.assert_array_equal(la, lb)


def _soften_decay(model: BigdnForSequenceClassification) -> BigdnForSequenceClassification:
    """Set every GDN decay gate to a mild value (A_log = log 0.05 => alpha ~= e^-0.065 per step).

    The random init draws A_log ~ log U(1e-6, 16), so some heads decay by ~e^-10 per position and a
    token's influence 31 positions away underflows float32 — which would make a connectivity test
    flaky about the INIT rather than the architecture."""
    import equinox as eqx

    def alogs(m):
        return [layer.gdn_fwd.A_log for layer in m.encoder.layers] + [
            layer.gdn_bwd.A_log for layer in m.encoder.layers
        ]

    return eqx.tree_at(alogs, model, [np.full(a.shape, np.log(0.05), dtype=np.float32) for a in alogs(model)])


def test_bidirectional_information_flow():
    """The first token must influence the last position's hidden state (forward direction) AND the
    last token must influence the first position's hidden state (backward direction) — a causal-only
    mixer fails the second check."""
    config = _tiny_config()
    Batch = Axis("batch", 1)
    Pos = Axis("position", config.max_seq_len)
    model = _soften_decay(BigdnForSequenceClassification.init(VOCAB, config, key=random.PRNGKey(0)))

    base = np.asarray(hax.random.randint(random.PRNGKey(2), (Batch, Pos), 1, VOCAB.size).array)

    def hidden(ids: np.ndarray) -> np.ndarray:
        return np.asarray(model.encoder(hax.named(ids, (Batch, Pos)), None).array)

    h_base = hidden(base)

    flip_first = base.copy()
    flip_first[0, 0] = (base[0, 0] % (VOCAB.size - 1)) + 1
    h_ff = hidden(flip_first)
    assert not np.allclose(h_base[0, -1], h_ff[0, -1]), "token 0 does not reach the last position"

    flip_last = base.copy()
    flip_last[0, -1] = (base[0, -1] % (VOCAB.size - 1)) + 1
    h_fl = hidden(flip_last)
    assert not np.allclose(h_base[0, 0], h_fl[0, 0]), "last token does not reach position 0 (no backward mixing)"

    # and both perturbations move the pooled logits
    mask = _segment_mask(np.zeros((Batch.size, Pos.size), dtype=np.int32), (Batch, Pos))
    l_base = np.asarray(model(hax.named(base, (Batch, Pos)), mask).array)
    l_fl = np.asarray(model(hax.named(flip_last, (Batch, Pos)), mask).array)
    assert not np.allclose(l_base, l_fl)


def test_registry_dispatch_and_no_warm_start():
    config = _tiny_config()
    with use_test_mesh():
        model = build_classifier(config, VOCAB, key=random.PRNGKey(0), warm_start=False)
    assert isinstance(model, BigdnForSequenceClassification)
    assert len(model.encoder.layers) == config.num_layers

    with pytest.raises(ValueError, match="no pretrained checkpoint"):
        build_classifier(config, VOCAB, key=random.PRNGKey(0), warm_start=True)


def test_eqx_save_load_roundtrip(tmp_path):
    from levanter.models.classification import load_eqx_classifier, save_classifier

    config = _tiny_config()
    model = BigdnForSequenceClassification.init(VOCAB, config, key=random.PRNGKey(0))
    save_classifier(config, model, str(tmp_path / "clf"))
    template = BigdnForSequenceClassification.init(VOCAB, config, key=random.PRNGKey(1))
    loaded = load_eqx_classifier(template, str(tmp_path / "clf"))

    tokens, mask, _, _ = _padded_batch(config, random.PRNGKey(3), [16, 32])
    np.testing.assert_array_equal(np.asarray(model(tokens, mask).array), np.asarray(loaded(tokens, mask).array))
