"""Offline integration test for the Levanter ModernBERT classifier training path.

Builds tiny pre-tokenized ClassificationExamples (no tokenizer download, no network) and runs
the full Trainer for two steps on CPU, validating the loss/optimizer/batch-stacking/mask
plumbing end to end. The TPU parity run uses the same `train_classifier` with real data.
"""

import haliax as hax
import jax
import numpy as np
from haliax import Axis
from levanter.data.dataset import ListAsyncDataset
from levanter.distributed import DistributedConfig
from levanter.layers.attention import AttentionMask
from levanter.main.train_classifier import train_classifier
from levanter.models.modernbert import ClassificationExample, ModernBertConfig
from levanter.optim import AdamConfig
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig


class _StubTokenizer:
    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size

    def __len__(self) -> int:
        return self._vocab_size


def _example(token_ids: list[int], label: int, Pos: Axis, pad_id: int) -> ClassificationExample:
    n = len(token_ids)
    ids = np.full((Pos.size,), pad_id, dtype=np.int32)
    ids[:n] = token_ids
    seg = np.full((Pos.size,), -1, dtype=np.int32)
    seg[:n] = 0
    seg_named = hax.named(seg, Pos)
    return ClassificationExample.init(
        tokens=hax.named(ids, Pos),
        label=hax.named(np.int32(label), ()),
        attn_mask=AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named),
    )


def test_train_classifier_smoke():
    vocab_size = 128
    config = ModernBertConfig(
        max_seq_len=32,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=4,
        num_heads=4,
        local_attention=8,
        num_labels=2,
        pad_token_id=0,
    )
    Pos = config.max_Pos
    rng = np.random.default_rng(0)
    # 16 examples: label 1 = longer token runs, label 0 = short runs (trivially learnable).
    data = []
    for k in range(16):
        label = k % 2
        length = 20 if label == 1 else 4
        ids = list(rng.integers(1, vocab_size, size=length))
        data.append(_example(ids, label, Pos, config.pad_token_id))
    dataset = ListAsyncDataset(data)

    import jmp

    trainer_config = TrainerConfig(
        id="modernbert-clf-test",
        num_train_steps=2,
        train_batch_size=len(jax.devices()),
        max_eval_batches=1,
        require_accelerator=False,
        tracker=NoopConfig(),
        distributed=DistributedConfig(initialize_jax_distributed=False),
        mp=jmp.get_policy("p=f32,c=f32"),
    )
    model = train_classifier(
        trainer_config=trainer_config,
        model_config=config,
        optimizer_config=AdamConfig(learning_rate=1e-4, warmup=0),
        train_dataset=dataset,
        tokenizer=_StubTokenizer(vocab_size),
        warm_start=False,
    )

    # The model runs a forward and produces 2-way logits over the label axis.
    ex = data[0]
    logits = model(ex.tokens.broadcast_axis(Axis("batch", 1)), ex.attn_mask)
    assert logits.resolve_axis("label").size == 2
    assert np.isfinite(np.asarray(logits.array)).all()
