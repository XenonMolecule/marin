# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the ModernBERT useful-vs-[NO_USEFUL_CONTENT] classifier on TPU via Levanter (JAX).

This is the JAX-native replacement for ``modernbert_tpu_smoke.py`` (PyTorch/torch_xla). It
reuses the same body_strip fastText-format data and the same frozen 7000-doc eval, but runs
through Levanter's ``Trainer`` (native multi-host/FSDP, Tensorstore checkpointing with
auto-resume, and the TPU splash kernel) using the ported ``ModernBertForSequenceClassification``.

The data format is the existing gzipped fastText files: one ``__label__useful <text>`` or
``__label__no_useful <text>`` line per document, where ``<text>`` is the body_strip-lowercased
document body. Label 1 = useful, 0 = no_useful.
"""

import dataclasses
import gzip
import logging
import random
from collections.abc import Sequence

import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning
from levanter.data.dataset import AsyncDataset
from levanter.distributed import DistributedConfig
from levanter.layers.attention import AttentionMask
from levanter.models.modernbert import (
    ClassificationExample,
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
)
from levanter.optim.config import AdamConfig
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode
from transformers import AutoTokenizer

import levanter

logger = logging.getLogger(__name__)

USEFUL_LABEL = "__label__useful"
MODEL_ID = "answerdotai/ModernBERT-base"


# --------------------------------------------------------------------------------------
# Data: read the existing gzipped fastText-format shards (label, text) per line.
# --------------------------------------------------------------------------------------


def _parse_line(line: str) -> tuple[int, str] | None:
    line = line.rstrip("\n")
    if not line or " " not in line:
        return None
    label_tok, text = line.split(" ", 1)
    if not label_tok.startswith("__label__") or not text:
        return None
    return (1 if label_tok == USEFUL_LABEL else 0, text)


def read_fasttext_shards(paths: Sequence[str], limit: int | None = None) -> tuple[list[str], list[int]]:
    """Read (texts, labels) from gzipped fastText-format shards in order, up to ``limit`` rows."""
    texts: list[str] = []
    labels: list[int] = []
    for path in paths:
        with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = _parse_line(line)
                if parsed is None:
                    continue
                labels.append(parsed[0])
                texts.append(parsed[1])
                if limit is not None and len(texts) >= limit:
                    return texts, labels
    return texts, labels


def read_frozen_test(paths: Sequence[str], rows: int, seed: int = 0) -> tuple[list[str], list[int]]:
    """Reproduce ``modernbert_tpu_smoke.read_test``: snapshot-stratified sample of the frozen test.

    Each shard is one snapshot; we take ``rows // n_shards`` per shard after a deterministic
    shuffle (random.Random(seed)), preserving the natural ~12:1 class ratio.
    """
    shards = list(paths)
    per = max(1, rows // max(1, len(shards)))
    texts: list[str] = []
    labels: list[int] = []
    for path in shards:
        shard_texts, shard_labels = read_fasttext_shards([path])
        idx = list(range(len(shard_texts)))
        random.Random(seed).shuffle(idx)
        for j in idx[:per]:
            texts.append(shard_texts[j])
            labels.append(shard_labels[j])
    return texts, labels


class TextClassificationDataset(AsyncDataset[ClassificationExample]):
    """In-memory (text, label) corpus tokenized lazily per batch to fixed-length ``Pos``.

    Holds raw strings in memory (cheap) and tokenizes on ``get_batch`` to avoid materializing
    padded token tensors for the whole corpus. Pad positions are masked out of attention via a
    segment mask (real positions = segment 0, pad positions = segment -1).
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, Pos: Axis, pad_token_id: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.Pos = Pos
        self.pad_token_id = pad_token_id

    async def async_len(self) -> int:
        return len(self.texts)

    def is_finite(self) -> bool:
        return True

    def _encode_one(self, text: str, label: int) -> ClassificationExample:
        seq_len = self.Pos.size
        ids = self.tokenizer(text, truncation=True, max_length=seq_len)["input_ids"]
        n = len(ids)
        padded = np.full((seq_len,), self.pad_token_id, dtype=np.int32)
        padded[:n] = np.asarray(ids, dtype=np.int32)
        seg = np.full((seq_len,), -1, dtype=np.int32)
        seg[:n] = 0  # real tokens share segment 0; pads (-1) are excluded from attention
        seg_named = hax.named(seg, self.Pos)
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        return ClassificationExample.init(
            tokens=hax.named(padded, self.Pos),
            label=hax.named(np.int32(label), ()),
            attn_mask=mask,
        )

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ClassificationExample]:
        return [self._encode_one(self.texts[i], self.labels[i]) for i in indices]


# --------------------------------------------------------------------------------------
# Scoring + F1 sweep (host-side, mirrors modernbert_tpu_smoke.f1_sweep)
# --------------------------------------------------------------------------------------


def score_texts(
    model: ModernBertForSequenceClassification,
    texts: list[str],
    tokenizer,
    Pos: Axis,
    pad_token_id: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Return P(useful) for each text. Runs a plain jitted forward; no training state."""
    model = inference_mode(model, True)

    @named_jit
    def _probs(m, tokens, mask):
        logits = m(tokens, mask).astype(jnp.float32)
        return hax.nn.softmax(logits, axis="label")["label", 1]

    out = np.zeros((len(texts),), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        ids = np.full((len(chunk), Pos.size), pad_token_id, dtype=np.int32)
        seg = np.full((len(chunk), Pos.size), -1, dtype=np.int32)
        for r, text in enumerate(chunk):
            enc = tokenizer(text, truncation=True, max_length=Pos.size)["input_ids"]
            ids[r, : len(enc)] = enc
            seg[r, : len(enc)] = 0
        bn = Axis("batch", len(chunk))
        tokens = hax.named(ids, (bn, Pos))
        seg_named = hax.named(seg, (bn, Pos))
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        probs = _probs(model, tokens, mask)
        out[start : start + len(chunk)] = np.asarray(probs.array)
    return out


def f1_sweep(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Best F1 over thresholds i/50, i in 1..49 (matches modernbert_tpu_smoke.f1_sweep)."""
    best_f1, best_t = 0.0, 0.5
    y = labels.astype(bool)
    for i in range(1, 50):
        t = i / 50
        pred = probs >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


# --------------------------------------------------------------------------------------
# Model construction (random init, or warm-start from the HF ModernBERT-base checkpoint)
# --------------------------------------------------------------------------------------


def build_model(
    config: ModernBertConfig,
    Vocab: Axis,
    *,
    key,
    warm_start: bool,
    axis_mapping=None,
    compute_dtype=None,
) -> ModernBertForSequenceClassification:
    if not warm_start:
        return ModernBertForSequenceClassification.init(Vocab, config, key=key)
    # Mirror HF AutoModelForSequenceClassification.from_pretrained(ModernBERT-base): load
    # encoder + prediction head from the checkpoint, randomly init only the classifier.
    converter = config.hf_checkpoint_converter()
    masked_lm = converter.load_pretrained(
        ModernBertForMaskedLM,
        ref=config.reference_checkpoint,
        config=config,
        axis_mapping=axis_mapping,
        dtype=compute_dtype,
    )
    return ModernBertForSequenceClassification.from_masked_lm(masked_lm, config, key=key)


def loss_fn(model: ModernBertForSequenceClassification, example: ClassificationExample, *, key=None):
    return model.compute_loss(example, key=key)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def train_classifier(
    *,
    trainer_config: TrainerConfig,
    model_config: ModernBertConfig,
    optimizer_config: AdamConfig,
    train_dataset: AsyncDataset[ClassificationExample],
    tokenizer,
    warm_start: bool,
    eval_texts: list[str] | None = None,
    eval_labels: list[int] | None = None,
) -> ModernBertForSequenceClassification:
    optimizer = optimizer_config.build(trainer_config.num_train_steps)
    with Trainer(trainer_config, optimizer, loss_fn) as trainer:
        model_key, training_key = jrandom.split(jrandom.PRNGKey(trainer_config.seed), 2)
        parameter_axis_mapping = trainer.parameter_axis_mapping
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)

        initial_model = build_model(
            model_config,
            Vocab,
            key=model_key,
            warm_start=warm_start,
            axis_mapping=parameter_axis_mapping,
            compute_dtype=trainer.mp.compute_dtype,
        )
        initial_model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(initial_model)
        state = trainer.initial_state(training_key, model=initial_model)
        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        train_loader = trainer.data_loader(train_dataset).iter_from_step(state.step)
        info = trainer.train(state, train_loader)
        final_model = inference_mode(info.state.model, True)

    if eval_texts is not None and eval_labels is not None:
        probs = score_texts(final_model, eval_texts, tokenizer, model_config.max_Pos, model_config.pad_token_id)
        best_f1, best_t = f1_sweep(probs, np.asarray(eval_labels))
        logger.info(f"[eval] best_f1={best_f1:.4f} @ t={best_t:.2f} on {len(eval_labels)} docs")
        levanter.tracker.log_summary({"eval/best_f1": best_f1, "eval/best_threshold": best_t})
    return final_model


def _tiny_smoke():
    """CPU integration smoke: a few synthetic docs through the full Trainer for 2 steps."""
    logging.basicConfig(level=logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = ModernBertConfig(
        max_seq_len=64,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=4,
        num_heads=4,
        local_attention=16,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    # Synthetic separable-ish data: "useful" docs vs short junk.
    texts = (["this is a long and genuinely useful article about science and history"] * 16) + (
        ["nav menu login click here"] * 16
    )
    labels = ([1] * 16) + ([0] * 16)
    ds = TextClassificationDataset(texts, labels, tokenizer, config.max_Pos, config.pad_token_id)

    trainer_config = TrainerConfig(
        id="modernbert-clf-smoke",
        num_train_steps=2,
        train_batch_size=len(jax.devices()),
        max_eval_batches=1,
        require_accelerator=False,
        tracker=NoopConfig(),
        distributed=DistributedConfig(initialize_jax_distributed=False),
        mp=jmp_policy("p=f32,c=f32"),
    )
    optimizer_config = AdamConfig(learning_rate=5e-5, warmup=0)
    train_classifier(
        trainer_config=trainer_config,
        model_config=config,
        optimizer_config=optimizer_config,
        train_dataset=ds,
        tokenizer=tokenizer,
        warm_start=False,
        eval_texts=texts,
        eval_labels=labels,
    )
    logger.info("smoke OK")


def jmp_policy(spec: str):
    import jmp

    return jmp.get_policy(spec)


# Existing body_strip fastText-format data (same as the torch run); see MEMORY feedback_llm_curated.
TRAIN_GLOB = "gs://marin-us-central2/classifiers/useful_fasttext/full_prep_body_strip/train/*.txt.gz"
TEST_GLOB = "gs://marin-us-central2/classifiers/useful_fasttext/full_prep_body_strip/test/*.txt.gz"


def run_real(args):
    """Parity / sweep training run on TPU. Reproduces the torch 1M recipe in Levanter.

    Targets a single-host TPU (e.g. v6e-4); ``levanter.initialize`` + DistributedConfig handle the
    JAX device mesh. Not runnable on CPU at scale — launch on a TPU worker.
    """
    logging.basicConfig(level=logging.INFO)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = ModernBertConfig(max_seq_len=args.max_seq_len, num_labels=2, pad_token_id=tokenizer.pad_token_id or 50283)

    fs = fsspec.core.url_to_fs(args.train_glob)[0]
    train_paths = sorted(f"gs://{p}" for p in fs.glob(args.train_glob))
    test_fs = fsspec.core.url_to_fs(args.test_glob)[0]
    test_paths = sorted(f"gs://{p}" for p in test_fs.glob(args.test_glob))
    logger.info(f"train shards={len(train_paths)} test shards={len(test_paths)}")

    train_texts, train_labels = read_fasttext_shards(train_paths, limit=args.train_rows)
    eval_texts, eval_labels = read_frozen_test(test_paths, rows=args.test_rows)
    logger.info(f"train docs={len(train_texts)} eval docs={len(eval_texts)}")

    dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, config.max_Pos, config.pad_token_id)

    eff_batch = args.batch_size
    steps_per_epoch = max(1, len(train_texts) // eff_batch)
    num_train_steps = steps_per_epoch * args.epochs

    trainer_config = TrainerConfig(
        id=args.run_id,
        num_train_steps=num_train_steps,
        train_batch_size=eff_batch,
        per_device_parallelism=args.per_device_parallelism,
        steps_per_eval=num_train_steps,  # F1 eval is done post-hoc on the frozen test
        mp=jmp_policy("p=f32,c=bfloat16"),
    )
    optimizer_config = AdamConfig(learning_rate=args.lr, warmup=args.warmup, max_grad_norm=1.0)

    levanter.initialize(trainer_config)
    model = train_classifier(
        trainer_config=trainer_config,
        model_config=dataclasses.replace(config, attn_backend=args.attn_backend),
        optimizer_config=optimizer_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        warm_start=not args.no_warm_start,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
    )

    if args.hf_out:
        converter = config.hf_checkpoint_converter()
        converter.save_pretrained(model, args.hf_out, save_tokenizer=True)
        logger.info(f"saved HF classifier to {args.hf_out}")


def _build_arg_parser():
    import argparse

    from levanter.layers.attention import AttentionBackend

    p = argparse.ArgumentParser(description="Train ModernBERT useful-classifier in Levanter (JAX/TPU).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke", help="CPU integration smoke (synthetic data, 2 steps).")

    r = sub.add_parser("real", help="Parity/sweep run on TPU.")
    r.add_argument("--run-id", required=True)
    r.add_argument("--train-glob", default=TRAIN_GLOB)
    r.add_argument("--test-glob", default=TEST_GLOB)
    r.add_argument("--train-rows", type=int, default=200_000)
    r.add_argument("--test-rows", type=int, default=7000)
    r.add_argument("--max-seq-len", type=int, default=8192)
    r.add_argument("--batch-size", type=int, default=256)
    r.add_argument("--per-device-parallelism", type=int, default=-1)
    r.add_argument("--epochs", type=int, default=1)
    r.add_argument("--lr", type=float, default=5e-5)
    r.add_argument("--warmup", type=int, default=200)
    r.add_argument("--attn-backend", type=AttentionBackend, default=AttentionBackend.VANILLA)
    r.add_argument("--no-warm-start", action="store_true", help="Random init instead of ModernBERT-base warm-start.")
    r.add_argument("--hf-out", default="", help="Optional gs:// path to export the final HF classifier.")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.cmd == "smoke":
        _tiny_smoke()
    else:
        run_real(args)
