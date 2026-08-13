# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a *decoder* (Qwen3) sequence classifier in Levanter (JAX/TPU).

The causal counterpart of ``train_classifier.py`` (which is ModernBERT/encoder-specific). The
backbone is a Qwen3 LM; a 2-way head reads the **last non-pad position** under a causal mask. Each
fastText ``__label__<name> <text>`` row is wrapped in a fixed prompt (``prompt_head`` + the text +
``prompt_tail``) so the pooled position is the generative "decision point" — the prompt strings are
supplied by the launcher (keeping this module free of experiment-side imports). ``label = 1`` iff
the label token equals ``useful_label``.

Reuses the generic fastText readers + F1 sweep from ``train_classifier.py``. Marin launches this via
``marin.training.training.run_levanter_train_decoder_classifier``.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import fsspec
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from transformers import AutoTokenizer

import haliax as hax
import levanter
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning
from levanter.data.dataset import AsyncDataset
from levanter.layers.attention import AttentionMask
from levanter.main.train_classifier import _expand_globs, f1_sweep, read_fasttext_shards, read_frozen_eval
from levanter.models.modernbert import ClassificationExample
from levanter.models.qwen import Qwen3Config, Qwen3ForSequenceClassification, Qwen3LMHeadModel
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode

logger = logging.getLogger(__name__)

NUM_LABELS = 2  # binary: 0 = no_useful (discard), 1 = useful (keep)


# --------------------------------------------------------------------------------------
# Data: fastText rows wrapped in a fixed decoder prompt, causal mask, last-token pool
# --------------------------------------------------------------------------------------


class DecoderPromptClassificationDataset(AsyncDataset[ClassificationExample]):
    """``(text, label)`` rows tokenized lazily into ``head_ids + truncated(text) + tail_ids``.

    Right-padded to ``Pos`` with a causal attention mask whose segment ids mark real (0) vs pad (-1)
    positions; the model pools the last real position (the final ``tail`` token)."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        Pos: Axis,
        pad_token_id: int,
        head_ids: Sequence[int],
        tail_ids: Sequence[int],
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.Pos = Pos
        self.pad_token_id = pad_token_id
        self.head_ids = list(head_ids)
        self.tail_ids = list(tail_ids)
        self.budget = Pos.size - len(self.head_ids) - len(self.tail_ids)
        if self.budget < 1:
            raise ValueError(
                f"Pos={Pos.size} too small for the fixed prompt "
                f"({len(self.head_ids) + len(self.tail_ids)} tokens); need >= "
                f"{len(self.head_ids) + len(self.tail_ids) + 1}."
            )

    async def async_len(self) -> int:
        return len(self.texts)

    def is_finite(self) -> bool:
        return True

    def _encode_one(self, text: str, label: int) -> ClassificationExample:
        seq_len = self.Pos.size
        html = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=self.budget)["input_ids"]
        ids = self.head_ids + html + self.tail_ids
        n = len(ids)
        padded = np.full((seq_len,), self.pad_token_id, dtype=np.int32)
        padded[:n] = np.asarray(ids, dtype=np.int32)
        seg = np.full((seq_len,), -1, dtype=np.int32)
        seg[:n] = 0
        seg_named = hax.named(seg, self.Pos)
        mask = AttentionMask(is_causal=True).with_segment_ids(seg_named, seg_named)
        return ClassificationExample.init(
            tokens=hax.named(padded, self.Pos),
            label=hax.named(np.int32(label), ()),
            attn_mask=mask,
        )

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ClassificationExample]:
        return [self._encode_one(self.texts[i], self.labels[i]) for i in indices]


@dataclass
class DecoderClassificationDataConfig:
    """Data source for decoder-classifier training (fastText line format + a fixed prompt wrapper)."""

    train_urls: list[str] = field(default_factory=list)
    validation_urls: list[str] = field(default_factory=list)
    tokenizer: str = "Qwen/Qwen3-0.6B"
    # Fixed prompt scaffolding around each document's text. Built by the launcher from the
    # extraction spec so this module stays free of experiment-side imports. ``prompt_tail`` must end
    # at the pooled "decision" token (e.g. the trailing ``[`` of ``[[ ## text ## ]]\n[``).
    prompt_head: str = ""
    prompt_tail: str = ""
    useful_label: str = "__label__useful"
    max_train_rows: Optional[int] = None
    eval_rows: int = 7000
    # Marin's _maybe_override_auto_build_caches reads this; classification has no cache so it is a no-op.
    auto_build_caches: bool = False

    @property
    def the_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer)

    def prompt_token_ids(self, tokenizer) -> tuple[list[int], list[int]]:
        head = tokenizer.encode(self.prompt_head, add_special_tokens=False)
        tail = tokenizer.encode(self.prompt_tail, add_special_tokens=False)
        if not tail:
            raise ValueError("prompt_tail tokenized to empty; it must end at the pooled decision token.")
        return head, tail

    def build_train(self, tokenizer, Pos: Axis, pad_token_id: int) -> DecoderPromptClassificationDataset:
        paths = _expand_globs(self.train_urls)
        logger.info(f"train shards: {len(paths)}")
        texts, labels = read_fasttext_shards(paths, self.useful_label, limit=self.max_train_rows)
        logger.info(f"train docs: {len(texts)}")
        head, tail = self.prompt_token_ids(tokenizer)
        logger.info(
            f"prompt: head={len(head)} tail={len(tail)} tokens; html budget={Pos.size - len(head) - len(tail)}"
        )
        return DecoderPromptClassificationDataset(texts, labels, tokenizer, Pos, pad_token_id, head, tail)

    def build_eval(self) -> tuple[list[str], list[int]]:
        if not self.validation_urls:
            return [], []
        paths = _expand_globs(self.validation_urls)
        return read_frozen_eval(paths, self.useful_label, rows=self.eval_rows)


# --------------------------------------------------------------------------------------
# Scoring (causal, last-token) -- runs inside the Trainer mesh
# --------------------------------------------------------------------------------------


def score_texts(
    model: Qwen3ForSequenceClassification,
    texts: list[str],
    tokenizer,
    Pos: Axis,
    pad_token_id: int,
    head_ids: Sequence[int],
    tail_ids: Sequence[int],
    batch_size: int = 8,
) -> np.ndarray:
    """Score texts -> P(useful). MUST run inside the Trainer's mesh (the model is sharded on it)."""
    model = inference_mode(model, True)
    Batch = Axis("batch", batch_size)
    budget = Pos.size - len(head_ids) - len(tail_ids)
    head_ids = list(head_ids)
    tail_ids = list(tail_ids)

    @hax.named_jit
    def _probs(m, tokens, mask):
        logits = m(tokens, mask).astype(jnp.float32)
        return hax.nn.softmax(logits, axis="label")["label", 1]

    out = np.zeros((len(texts),), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        n = len(chunk)
        ids = np.full((batch_size, Pos.size), pad_token_id, dtype=np.int32)
        seg = np.full((batch_size, Pos.size), -1, dtype=np.int32)
        for r, text in enumerate(chunk):
            html = tokenizer(text, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
            seq = head_ids + html + tail_ids
            m = len(seq)
            ids[r, :m] = seq
            seg[r, :m] = 0
        tokens = hax.named(ids, (Batch, Pos))
        seg_named = hax.named(seg, (Batch, Pos))
        mask = AttentionMask(is_causal=True).with_segment_ids(seg_named, seg_named)
        out[start : start + n] = np.asarray(_probs(model, tokens, mask).array)[:n]
    return out


# --------------------------------------------------------------------------------------
# Model construction + loss + save
# --------------------------------------------------------------------------------------


def build_model(
    config: Qwen3Config,
    Vocab: Axis,
    Label: Axis,
    decision_token_id: int,
    *,
    key,
    warm_start: bool,
    warm_head: bool,
    axis_mapping=None,
    compute_dtype=None,
) -> Qwen3ForSequenceClassification:
    if not warm_start:
        return Qwen3ForSequenceClassification.init(Vocab, config, Label, key=key)
    converter = config.hf_checkpoint_converter()
    lm = converter.load_pretrained(
        Qwen3LMHeadModel,
        ref=config.reference_checkpoint,
        config=config,
        axis_mapping=axis_mapping,
        dtype=compute_dtype,
    )
    return Qwen3ForSequenceClassification.from_lm_head_model(
        lm, config, Label, decision_token_id, key=key, warm_head=warm_head
    )


def classification_loss(model: Qwen3ForSequenceClassification, example: ClassificationExample, *, key=None):
    return model.compute_loss(example, key=key)


def save_classifier(
    model: Qwen3ForSequenceClassification, config: Qwen3Config, path: str, best_threshold: float
) -> None:
    """Save the fine-tuned Qwen3 backbone (standard HF) + the small classification head + metadata."""
    converter = config.hf_checkpoint_converter()
    converter.save_pretrained(model.lm, f"{path}/backbone", save_tokenizer=True)
    weight = np.asarray(model.classifier.weight.array)
    bias = (
        np.zeros((NUM_LABELS,), dtype=np.float32)
        if model.classifier.bias is None
        else np.asarray(model.classifier.bias.array)
    )
    with fsspec.open(f"{path}/classifier_head.npz", "wb") as f:
        np.savez(f, weight=weight, bias=bias)
    with fsspec.open(f"{path}/classifier_meta.json", "w") as f:
        json.dump(
            {
                "label_names": ["no_useful", "useful"],
                "positive_label": "useful (index 1)",
                "pooling": "last_non_pad_token",
                "best_threshold": float(best_threshold),
            },
            f,
            indent=2,
        )
    logger.info(f"saved decoder classifier (backbone + head) to {path}")


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def train_decoder_classifier(
    *,
    trainer_config: TrainerConfig,
    model_config: Qwen3Config,
    optimizer_config: OptimizerConfig,
    train_dataset: AsyncDataset[ClassificationExample],
    tokenizer,
    pad_token_id: int,
    head_ids: Sequence[int],
    tail_ids: Sequence[int],
    decision_token_id: int,
    warm_start: bool,
    warm_head: bool,
    eval_texts: Optional[list[str]] = None,
    eval_labels: Optional[list[int]] = None,
    hf_save_path: Optional[str] = None,
) -> Qwen3ForSequenceClassification:
    optimizer = optimizer_config.build(trainer_config.num_train_steps)
    Label = Axis("label", NUM_LABELS)
    with Trainer(trainer_config, optimizer, classification_loss) as trainer:
        model_key, training_key = jrandom.split(jrandom.PRNGKey(trainer_config.seed), 2)
        parameter_axis_mapping = trainer.parameter_axis_mapping
        Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), parameter_axis_mapping)

        initial_model = build_model(
            model_config,
            Vocab,
            Label,
            decision_token_id,
            key=model_key,
            warm_start=warm_start,
            warm_head=warm_head,
            axis_mapping=parameter_axis_mapping,
            compute_dtype=trainer.mp.compute_dtype,
        )
        initial_model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(initial_model)
        state = trainer.initial_state(training_key, model=initial_model)
        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        # Shuffle: the survivor shards are class-ordered (useful block then no_useful block); reading
        # in order gives class-homogeneous microbatches and a degenerate loss. (Same fix as ModernBERT.)
        train_dataset = train_dataset.shuffle(jrandom.PRNGKey(trainer_config.seed + 1))
        train_loader = trainer.data_loader(train_dataset).iter_from_step(state.step)
        info = trainer.train(state, train_loader)
        final_model = inference_mode(info.state.model, True)

        best_t = 0.5
        if eval_texts:
            probs = score_texts(
                final_model, eval_texts, tokenizer, model_config.max_Pos, pad_token_id, head_ids, tail_ids
            )
            best_f1, best_t = f1_sweep(probs, np.asarray(eval_labels))
            logger.info(f"[eval] best_f1={best_f1:.4f} @ t={best_t:.2f} on {len(eval_labels)} docs")
            levanter.tracker.log_summary({"eval/best_f1": best_f1, "eval/best_threshold": best_t})

        if hf_save_path:
            save_classifier(final_model, model_config, hf_save_path, best_t)
    return final_model


# --------------------------------------------------------------------------------------
# Config-driven entrypoint (the Fray/Iris entrypoint, submitted by marin)
# --------------------------------------------------------------------------------------


@dataclass
class TrainDecoderClassifierConfig:
    data: DecoderClassificationDataConfig = field(default_factory=DecoderClassificationDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: Qwen3Config = field(default_factory=Qwen3Config)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    warm_start: bool = True  # load the pretrained Qwen3 backbone (vs random init)
    warm_head: bool = True  # init the discard-class head row from the decision token's unembedding
    decision_token_id: int = 8996  # "NO" of [NO_USEFUL_CONTENT] under the Qwen3 tokenizer
    hf_save_path: Optional[str] = None


def main(config: TrainDecoderClassifierConfig):
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer
    pad_id = tokenizer.pad_token_id
    Pos = config.model.max_Pos

    train_dataset = config.data.build_train(tokenizer, Pos, pad_id)
    eval_texts, eval_labels = config.data.build_eval()
    head_ids, tail_ids = config.data.prompt_token_ids(tokenizer)

    train_decoder_classifier(
        trainer_config=config.trainer,
        model_config=config.model,
        optimizer_config=config.optimizer,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        pad_token_id=pad_id,
        head_ids=head_ids,
        tail_ids=tail_ids,
        decision_token_id=config.decision_token_id,
        warm_start=config.warm_start,
        warm_head=config.warm_head,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        hf_save_path=config.hf_save_path,
    )


if __name__ == "__main__":
    levanter.config.main(main)()
