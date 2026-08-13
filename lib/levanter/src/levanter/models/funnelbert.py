# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""FunnelBERT: a funnel-hybrid super-token classifier built on ModernBERT.

The first ``num_full_layers`` ModernBERT layers run at full resolution (warm-started from the
reference checkpoint, keeping their per-layer-index global/local attention assignment). The
sequence is then mask-aware mean-pooled into windows of ``pool_factor`` tokens ("super-tokens"),
and ``num_pooled_layers`` fresh global-attention layers run on the short sequence before a masked
mean-pool + prediction head + classifier. At 8192 context with ``pool_factor=8`` the pooled stage
sees only 1024 positions, so its full attention is cheap.

This architecture is not HF-round-trippable; it saves/loads via the generic equinox serializer
(:func:`levanter.models.classification.save_eqx_classifier`).
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Type

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call

from levanter.layers.attention import AttentionMask
from levanter.models.classification import (
    ClassificationExample,
    load_eqx_classifier,
    register_classifier_arch,
    save_eqx_classifier,
)
from levanter.models.lm_model import LmConfig
from levanter.models.modernbert import (
    ModernBertConfig,
    ModernBertEmbeddings,
    ModernBertEncoderLayer,
    ModernBertForMaskedLM,
    ModernBertPredictionHead,
)
from levanter.utils.flop_utils import lm_flops_per_token


@LmConfig.register_subclass("funnelbert")
@dataclass(frozen=True)
class FunnelBertConfig(ModernBertConfig):
    """ModernBERT config plus the funnel stage.

    ``num_full_layers`` full-resolution layers (warm-started), then mean-pool by ``pool_factor``
    into super-tokens, then ``num_pooled_layers`` fresh global-attention layers. ``num_layers`` is
    forced equal to ``num_full_layers`` so inherited full-resolution shape math stays consistent.
    """

    num_full_layers: int = 4
    pool_factor: int = 8
    num_pooled_layers: int = 4

    def __post_init__(self):
        object.__setattr__(self, "num_layers", self.num_full_layers)
        super().__post_init__()
        if self.pool_factor < 1:
            raise ValueError(f"pool_factor must be >= 1, got {self.pool_factor}")
        if self.max_seq_len % self.pool_factor != 0:
            raise ValueError(f"max_seq_len={self.max_seq_len} must be divisible by pool_factor={self.pool_factor}")

    @property
    def SuperPos(self) -> Axis:
        return Axis("position", self.max_seq_len // self.pool_factor)

    @property
    def model_type(self) -> Type["FunnelBertForSequenceClassification"]:  # pyrefly: ignore[bad-override]
        return FunnelBertForSequenceClassification

    def flops_per_token(self, vocab_size: int, context_length: int) -> Optional[float]:
        """Rough estimate: full-res stage at ``context_length`` plus the pooled stage's layer flops
        amortized over the ``pool_factor``-times-longer input sequence. ``vocab_size=0`` drops the
        LM-head term (the classifier head is negligible)."""
        full_stage = lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_full_layers,
            num_kv_heads=self.num_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=0,
            glu=True,
        )
        pooled_stage = lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_pooled_layers,
            num_kv_heads=self.num_heads,
            num_heads=self.num_heads,
            seq_len=max(context_length // self.pool_factor, 1),
            vocab_size=0,
            glu=True,
        )
        return full_stage + pooled_stage / self.pool_factor


class FunnelBertForSequenceClassification(eqx.Module):
    """Full-res ModernBERT prefix -> super-token pooling -> global pooled layers -> classifier."""

    config: FunnelBertConfig = eqx.field(static=True)
    embeddings: ModernBertEmbeddings
    full_layers: list  # list[ModernBertEncoderLayer], full resolution, warm-startable
    pooled_layers: list  # list[ModernBertEncoderLayer], global attention on super-tokens
    final_norm: hnn.LayerNorm
    head: ModernBertPredictionHead
    classifier: hnn.Linear

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Label(self) -> Axis:
        return self.classifier.Out if isinstance(self.classifier.Out, Axis) else self.config.Label

    @classmethod
    def init(cls, Vocab: Axis, config: FunnelBertConfig, *, key) -> "FunnelBertForSequenceClassification":
        k_emb, k_full, k_pooled, k_head, k_cls = jrandom.split(key, 5)
        embeddings = ModernBertEmbeddings.init(Vocab, config, key=k_emb)
        full_keys = jrandom.split(k_full, config.num_full_layers)
        full_layers = [ModernBertEncoderLayer.init(config, i, key=full_keys[i]) for i in range(config.num_full_layers)]
        # Pooled layers are all-global: a config copy with global_attn_every_n_layers=1 makes every
        # layer_idx global; layer_idx >= 1 so each pooled layer keeps its attention pre-norm.
        pooled_config = dataclasses.replace(config, global_attn_every_n_layers=1)
        pooled_keys = jrandom.split(k_pooled, config.num_pooled_layers)
        pooled_layers = [
            ModernBertEncoderLayer.init(pooled_config, i + 1, key=pooled_keys[i])
            for i in range(config.num_pooled_layers)
        ]
        final_norm = config.mk_LayerNorm(config.Embed)
        head = ModernBertPredictionHead.init(config, key=k_head)
        classifier = hnn.Linear.init(
            In=config.Embed, Out=config.Label, key=k_cls, use_bias=config.classifier_bias, out_first=True
        )
        return cls(config, embeddings, full_layers, pooled_layers, final_norm, head, classifier)

    def _pool_windows(
        self, x: NamedArray, attn_mask: AttentionMask | NamedArray | None
    ) -> tuple[NamedArray, AttentionMask, NamedArray]:
        """Mask-aware mean-pool ``position`` into super-tokens of ``pool_factor`` positions.

        Pad positions (segment id < 0, per the classifier convention of real=0/pad=-1) get zero
        weight; a window with no real tokens is inactive (pooled value 0, pooled segment id -1).
        Returns ``(pooled, pooled_mask, active)`` with pooled axes renamed back to ``position``.
        """
        Pos = x.resolve_axis("position")
        SuperPos = Axis("super_position", Pos.size // self.config.pool_factor)
        Window = Axis("window", self.config.pool_factor)

        if isinstance(attn_mask, AttentionMask) and attn_mask.segment_ids is not None:
            real = (attn_mask.segment_ids[0] >= 0).astype(x.dtype)
        else:
            real = hax.ones(Pos, dtype=x.dtype)

        x_win = hax.unflatten_axis(x, Pos, (SuperPos, Window))
        real_win = hax.unflatten_axis(real, "position", (SuperPos, Window))
        counts = real_win.sum(Window)
        pooled = (x_win * real_win).sum(Window) / hax.maximum(counts, 1.0)
        active = (counts > 0).rename({"super_position": "position"})
        pooled = pooled.rename({"super_position": "position"})
        pooled_seg = active.astype(jnp.int32) - 1  # active -> 0, inactive -> -1
        pooled_mask = AttentionMask(is_causal=False).with_segment_ids(pooled_seg, pooled_seg)
        return pooled, pooled_mask, active

    @named_call
    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        for layer in self.full_layers:
            x = layer(x, attn_mask, pos_ids=pos_ids)
        pooled, pooled_mask, active = self._pool_windows(x, attn_mask)
        for layer in self.pooled_layers:
            pooled = layer(pooled, pooled_mask)
        hidden = self.final_norm(pooled)
        weight = active.astype(hidden.dtype)
        denom = hax.maximum(weight.sum("position"), 1.0)
        doc = (hidden * weight).sum("position") / denom
        return self.classifier(self.head(doc))

    def compute_loss(
        self,
        example: ClassificationExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,  # pyrefly: ignore[bad-function-definition]
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> NamedArray:
        logits = self(example.tokens, example.attn_mask, key=key).astype(jnp.float32)
        target = hax.nn.one_hot(example.label, self.Label, dtype=logits.dtype)
        return hax.nn.cross_entropy_loss(logits, self.Label, target, reduction, reduction_axis=reduction_axis)

    def resize_vocab(self, new_size: int, key=None) -> "FunnelBertForSequenceClassification":
        new_embeddings = dataclasses.replace(
            self.embeddings, tok_embeddings=self.embeddings.tok_embeddings.resize_embeddings(new_size, key=key)
        )
        return dataclasses.replace(self, embeddings=new_embeddings)  # pyrefly: ignore[bad-specialization]


def _build_funnelbert_classifier(
    config: FunnelBertConfig, Vocab: Axis, *, key, warm_start: bool, axis_mapping=None, compute_dtype=None
) -> FunnelBertForSequenceClassification:
    """Random init, or warm-start embeddings + the bottom ``num_full_layers`` from the reference
    ModernBERT MLM at its own full depth. Pooled layers, head, and classifier stay random."""
    model = FunnelBertForSequenceClassification.init(Vocab, config, key=key)
    if not warm_start:
        return model
    base_fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(ModernBertConfig)}
    base_config = ModernBertConfig(**base_fields)
    converter = base_config.hf_checkpoint_converter()
    hf_config = converter.hf_config_from_hf_checkpoint(config.reference_checkpoint)
    if config.num_full_layers > hf_config.num_hidden_layers:
        raise ValueError(
            f"num_full_layers={config.num_full_layers} exceeds reference depth {hf_config.num_hidden_layers}"
        )
    full_config = dataclasses.replace(base_config, num_layers=hf_config.num_hidden_layers)
    masked_lm = converter.load_pretrained(
        ModernBertForMaskedLM,
        ref=config.reference_checkpoint,
        config=full_config,
        axis_mapping=axis_mapping,
        dtype=compute_dtype,
    )
    return dataclasses.replace(
        model,
        embeddings=masked_lm.model.embeddings,
        full_layers=masked_lm.model.layers[: config.num_full_layers],
    )


def load_funnelbert_classifier(
    config: FunnelBertConfig, path: str, *, vocab_size: int
) -> FunnelBertForSequenceClassification:
    """Load a classifier saved by the eqx saver into a fresh template (for inference)."""
    template = eqx.filter_eval_shape(
        FunnelBertForSequenceClassification.init, Axis("vocab", vocab_size), config, key=jrandom.PRNGKey(0)
    )
    return load_eqx_classifier(template, path)


register_classifier_arch(FunnelBertConfig, build=_build_funnelbert_classifier, save=save_eqx_classifier)


__all__ = [
    "FunnelBertConfig",
    "FunnelBertForSequenceClassification",
    "load_funnelbert_classifier",
]
