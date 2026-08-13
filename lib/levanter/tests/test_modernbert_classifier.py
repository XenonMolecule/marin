# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import tempfile

import numpy as np
import pytest
from jax import random
from test_utils import skip_if_module_missing, skip_if_no_torch, use_test_mesh

import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionMask
from levanter.main.train_classifier import (
    CHUNK_AGGREGATORS,
    ChunkedCachedClassificationDataset,
    ChunkedTextClassificationDataset,
    _load_token_cache,
    _save_token_cache,
    aggregate_sweep,
    chunk_starts,
    f1_sweep,
)
from levanter.store.tree_store import TreeStore
from levanter.models.modernbert import (
    ClassificationExample,
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
    load_hf_sequence_classifier,
)


def test_chunk_starts_non_overlap_and_overlap():
    assert chunk_starts(100, 10) == list(range(0, 100, 10))  # 10 non-overlapping tiles
    assert chunk_starts(100, 5) == list(range(0, 100, 5))  # 20 tiles at 50% overlap
    assert chunk_starts(10, 10) == [0]  # exactly one window
    assert chunk_starts(0, 10) == [0]  # empty doc still yields one (padded) chunk
    assert chunk_starts(15, 10) == [0, 10]  # short tail still becomes its own chunk


def _build_chunked(doc_token_lens, ctx, stride, max_chunks, labels=None, seed=0):
    # Pre-tokenized docs: token ids are just 0..len-1 so chunk content is checkable.
    doc_ids = [np.arange(n, dtype=np.int32) for n in doc_token_lens]
    labels = labels if labels is not None else [i % 2 for i in range(len(doc_ids))]
    return ChunkedTextClassificationDataset(
        doc_ids,
        labels,
        Axis("position", ctx),
        pad_token_id=999,
        max_chunks=max_chunks,
        stride=stride,
        seed=seed,
    )


def test_chunked_cached_dataset_streams_from_treecache(tmp_path):
    # Build a real TreeStore with docs of known token lengths, then verify the streaming chunked
    # dataset derives the chunk index from the jagged offsets and reads correct chunk content + labels.
    exemplar = {"input_ids": np.zeros((0,), np.int32), "label": np.zeros((), np.int32)}
    store = TreeStore.open(exemplar, str(tmp_path / "store"), mode="w")
    docs = [np.arange(5, dtype=np.int32), np.arange(100, 200, dtype=np.int32), np.arange(30, dtype=np.int32)]
    labels = [1, 0, 1]
    store.extend([{"input_ids": d, "label": np.int32(lbl)} for d, lbl in zip(docs, labels)])

    class _Cache:
        pass

    cache = _Cache()
    cache.store = store

    ds = ChunkedCachedClassificationDataset(
        cache, Axis("position", 10), pad_token_id=999, max_chunks=16, stride=10, seed=0
    )
    # lengths 5,100,30 at ctx=stride=10 → 1 + 10 + 3 chunks; lengths read from offsets, not token data
    assert ds.num_chunks == 1 + 10 + 3
    assert ds.labels == [1, 0, 1]
    # chunk index 2 = doc1 (the 100..199 doc) at start 10 → tokens 110..119, doc1 label 0
    ex = asyncio.run(ds.get_batch([2]))[0]
    assert np.array_equal(np.asarray(ex.tokens.array)[:10], np.arange(110, 120))
    assert int(ex.label.array) == 0
    # chunk index 11 = doc2 first chunk → tokens 0..9 (padded tail), doc2 label 1
    ex2 = asyncio.run(ds.get_batch([11]))[0]
    assert np.array_equal(np.asarray(ex2.tokens.array)[:10], np.arange(0, 10))
    assert int(ex2.label.array) == 1


def test_token_cache_roundtrip(tmp_path):
    # Variable-length docs + labels survive the flat (ids/offsets/labels) save→load exactly.
    doc_ids = [np.arange(5, dtype=np.int32), np.arange(100, 200, dtype=np.int32), np.zeros((0,), np.int32)]
    labels = [1, 0, 1]
    root = f"file://{tmp_path}/cache"
    _save_token_cache(root, doc_ids, labels)
    got_ids, got_labels = _load_token_cache(root)
    assert got_labels == labels
    assert len(got_ids) == len(doc_ids)
    for a, b in zip(got_ids, doc_ids, strict=True):
        assert np.array_equal(a, b)


def test_chunked_index_counts_and_cap():
    # doc lengths 5, 100 with ctx=stride=10: doc0 -> 1 chunk, doc1 -> 10 chunks (uncapped at max_chunks=16)
    ds = _build_chunked([5, 100], ctx=10, stride=10, max_chunks=16)
    assert ds.num_chunks == 1 + 10
    # cap at 4: doc1's 10 chunks become a random subset of 4; doc0 keeps its 1
    capped = _build_chunked([5, 100], ctx=10, stride=10, max_chunks=4)
    assert capped.num_chunks == 1 + 4
    doc1_starts = [st for di, st in capped.chunks if di == 1]
    assert len(doc1_starts) == 4
    assert doc1_starts == sorted(doc1_starts)  # kept in ascending order
    assert set(doc1_starts).issubset(set(range(0, 100, 10)))  # a genuine subset of all tiles


def test_chunked_sampling_is_deterministic():
    a = _build_chunked([200], ctx=10, stride=10, max_chunks=5, seed=7)
    b = _build_chunked([200], ctx=10, stride=10, max_chunks=5, seed=7)
    c = _build_chunked([200], ctx=10, stride=10, max_chunks=5, seed=8)
    assert a.chunks == b.chunks  # same seed -> identical subset
    assert a.chunks != c.chunks  # different seed -> (almost surely) different subset


def test_chunked_overlap_doubles_chunks():
    non = _build_chunked([100], ctx=10, stride=10, max_chunks=1000)
    ovl = _build_chunked([100], ctx=10, stride=5, max_chunks=1000)
    assert non.num_chunks == 10
    assert ovl.num_chunks == 20


def test_chunked_get_batch_content_and_label():
    ds = _build_chunked([35], ctx=10, stride=10, max_chunks=16, labels=[1])
    assert ds.num_chunks == 4  # ceil(35/10)
    examples = asyncio.run(ds.get_batch([0, 1, 3]))
    # chunk 1 starts at token 10 -> ids 10..19 in the first 10 positions
    assert np.array_equal(np.asarray(examples[1].tokens.array)[:10], np.arange(10, 20))
    # chunk 3 (start 30) has only 5 real tokens (30..34); rest padded to pad_token_id
    last = np.asarray(examples[2].tokens.array)
    assert np.array_equal(last[:5], np.arange(30, 35))
    assert (last[5:] == 999).all()
    assert all(int(ex.label.array) == 1 for ex in examples)  # every chunk inherits the doc label


def test_aggregate_sweep_prefers_separating_aggregator():
    # useful docs (label 1) have one high chunk but a LOW mean; non-useful (label 0) are uniformly
    # medium (higher mean than the useful docs). Only `max` separates them (F1=1.0); `mean` ranks
    # them backwards. This pins that the sweep actually picks the right aggregator, not just any tie.
    per_doc = [
        np.array([0.1, 0.1, 0.1, 0.95]),  # useful: mean=0.31, max=0.95
        np.array([0.1, 0.15, 0.1, 0.9]),  # useful: mean≈0.31, max=0.9
        np.array([0.5, 0.5, 0.5, 0.5]),  # non-useful: mean=0.5, max=0.5
        np.array([0.45, 0.55, 0.5, 0.5]),  # non-useful: mean=0.5, max=0.55
    ]
    labels = [1, 1, 0, 0]
    results, best_agg = aggregate_sweep(per_doc, labels)
    assert set(results) == set(CHUNK_AGGREGATORS)
    assert results["max"][0] == pytest.approx(1.0)
    assert best_agg == "max"
    assert results["mean"][0] < 1.0  # mean can't separate (non-useful mean > useful mean)


def test_f1_sweep_perfect_separation():
    probs = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    f1, t = f1_sweep(probs, labels)
    assert f1 == pytest.approx(1.0)
    assert 0.2 < t < 0.8


def _small_config(pooling: str = "cls") -> ModernBertConfig:
    return ModernBertConfig(
        max_seq_len=64,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=4,
        num_heads=4,
        local_attention=16,
        classifier_pooling=pooling,
        num_labels=2,
        tie_word_embeddings=True,
    )


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_classifier_forward_and_loss(pooling):
    config = _small_config(pooling)
    Vocab = Axis("vocab", 4096)
    Batch = Axis("batch", 3)
    Pos = Axis("position", config.max_seq_len)

    model = ModernBertForSequenceClassification.init(Vocab, config, key=random.PRNGKey(0))

    tokens = hax.random.randint(random.PRNGKey(1), (Batch, Pos), 0, Vocab.size)
    logits = model(tokens, AttentionMask.bidirectional_sliding_window(config.local_attention // 2))

    assert logits.axes == (Batch, config.Label), logits.axes
    assert np.isfinite(np.asarray(logits.array)).all()

    labels = hax.named(np.array([0, 1, 1]), Batch)
    ex = ClassificationExample.init(tokens, labels)
    loss = model.compute_loss(ex)
    assert loss.ndim == 0 and np.isfinite(float(loss.array))


def test_warm_start_from_masked_lm_shares_encoder_and_head():
    config = _small_config()
    Vocab = Axis("vocab", 4096)
    mlm = ModernBertForMaskedLM.init(Vocab, config, key=random.PRNGKey(2))
    clf = ModernBertForSequenceClassification.from_masked_lm(mlm, config, key=random.PRNGKey(3))

    # encoder + prediction head are the SAME pretrained weights (warm-started), classifier is fresh.
    assert clf.model is mlm.model
    assert clf.head is mlm.head
    assert clf.classifier.Out == config.Label
    enc_w = np.asarray(clf.model.embeddings.tok_embeddings.weight.array)
    assert np.array_equal(enc_w, np.asarray(mlm.model.embeddings.tok_embeddings.weight.array))


def test_classifier_logits_depend_on_input():
    """Sanity: different token sequences produce different logits (the head is wired to the encoder)."""
    config = _small_config()
    Vocab = Axis("vocab", 4096)
    Pos = Axis("position", config.max_seq_len)
    model = ModernBertForSequenceClassification.init(Vocab, config, key=random.PRNGKey(0))

    a = hax.random.randint(random.PRNGKey(10), (Pos,), 0, Vocab.size)
    b = hax.random.randint(random.PRNGKey(11), (Pos,), 0, Vocab.size)
    mask = AttentionMask.bidirectional_sliding_window(config.local_attention // 2)
    la = np.asarray(model(a, mask).array)
    lb = np.asarray(model(b, mask).array)
    assert not np.allclose(la, lb)


@skip_if_no_torch
@skip_if_module_missing("transformers.models.modernbert.modeling_modernbert")
@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_classifier_hf_roundtrip(pooling, local_gpt2_tokenizer_path):
    """The weight-port oracle: a fine-tuned HF ModernBertForSequenceClassification (encoder +
    head + classifier) round-trips into the Levanter classifier with matching logits."""
    import torch  # noqa: PLC0415
    from transformers.models.modernbert.configuration_modernbert import ModernBertConfig as HfModernBertConfig
    from transformers.models.modernbert import modeling_modernbert  # noqa: PLC0415

    Vocab = hax.Axis("vocab", 4096)
    hf_config = HfModernBertConfig(
        vocab_size=Vocab.size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=256,
        norm_eps=1e-5,
        norm_bias=False,
        attention_bias=False,
        mlp_bias=False,
        decoder_bias=True,
        classifier_bias=False,
        hidden_activation="gelu",
        classifier_activation="gelu",
        global_attn_every_n_layers=3,
        local_attention=16,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        tie_word_embeddings=True,
        pad_token_id=0,
        num_labels=2,
        classifier_pooling=pooling,
        _attn_implementation="eager",
    )
    lev_config = ModernBertConfig(
        max_seq_len=256,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=4,
        num_heads=4,
        local_attention=16,
        global_attn_every_n_layers=3,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        tie_word_embeddings=True,
        pad_token_id=0,
        num_labels=2,
        classifier_pooling=pooling,
        tokenizer=local_gpt2_tokenizer_path,
    )

    input_ids = hax.random.randint(random.PRNGKey(0), lev_config.max_Pos, 0, Vocab.size)
    input_torch = torch.from_numpy(np.array(input_ids.array)).to(torch.int64).unsqueeze(0)

    torch.random.manual_seed(0)
    torch_model = modeling_modernbert.ModernBertForSequenceClassification(hf_config)
    torch_model.eval()
    torch_out = torch_model(input_ids=input_torch).logits[0].detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        model_path = f"{tmpdir}/torch_model"
        torch_model.save_pretrained(model_path)
        model = load_hf_sequence_classifier(lev_config, model_path)

        @hax.named_jit
        def compute(m, ids):
            return m(ids)

        jax_out = np.array(compute(model, input_ids).array)

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    assert np.isclose(torch_out, jax_out, rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"


# --------------------------------------------------------------------------------------
# Architecture registry + pruned warm start
# --------------------------------------------------------------------------------------


def test_build_classifier_dispatches_on_config_type():
    from levanter.models.classification import build_classifier

    config = _small_config()
    with use_test_mesh():
        model = build_classifier(config, Axis("vocab", 128), key=random.PRNGKey(0), warm_start=False)
    assert isinstance(model, ModernBertForSequenceClassification)
    assert len(model.model.layers) == config.num_layers


def test_build_classifier_unregistered_config_raises():
    from dataclasses import dataclass

    from levanter.models.classification import build_classifier

    @dataclass(frozen=True)
    class NotRegistered:
        pass

    with pytest.raises(ValueError, match="no classifier builder"):
        build_classifier(NotRegistered(), Axis("vocab", 128), key=random.PRNGKey(0), warm_start=False)


def test_pruned_config_random_init_uses_pruned_depth():
    from levanter.models.modernbert import PrunedModernBertConfig

    config = PrunedModernBertConfig(
        max_seq_len=64,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        local_attention=16,
        num_labels=2,
        pad_token_id=0,
    )
    from levanter.models.classification import build_classifier

    with use_test_mesh():
        model = build_classifier(config, Axis("vocab", 128), key=random.PRNGKey(0), warm_start=False)
    assert len(model.model.layers) == 2


@skip_if_no_torch
def test_pruned_warm_start_keeps_bottom_layers(local_gpt2_tokenizer_path):
    """Warm-starting a pruned config loads the reference at full depth and keeps the bottom layers
    (weights identical to the full load's first N layers)."""
    import torch  # noqa: PLC0415
    from transformers.models.modernbert import modeling_modernbert  # noqa: PLC0415
    from transformers.models.modernbert.configuration_modernbert import ModernBertConfig as HfModernBertConfig

    from levanter.models.classification import build_classifier
    from levanter.models.modernbert import PrunedModernBertConfig

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
        pruned_config = PrunedModernBertConfig(
            max_seq_len=128,
            hidden_dim=64,
            intermediate_dim=128,
            num_layers=2,
            num_heads=4,
            local_attention=16,
            num_labels=2,
            pad_token_id=0,
            reference_checkpoint=model_path,
            tokenizer=local_gpt2_tokenizer_path,
        )
        pruned = build_classifier(pruned_config, Axis("vocab", 1024), key=random.PRNGKey(0), warm_start=True)
        assert len(pruned.model.layers) == 2

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
        for i in range(2):
            got = np.array(pruned.model.layers[i].mlp.Wi.weight.array)
            want = np.array(full.model.layers[i].mlp.Wi.weight.array)
            assert np.allclose(got, want), f"layer {i} Wi mismatch after pruning"
        assert np.allclose(
            np.array(pruned.model.embeddings.tok_embeddings.weight.array),
            np.array(full.model.embeddings.tok_embeddings.weight.array),
        )
