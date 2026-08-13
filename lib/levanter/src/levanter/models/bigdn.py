# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""BiGDN: a bidirectional Gated DeltaNet encoder for document classification.

A from-scratch sub-quadratic (linear-time) alternative to ModernBERT for long-document
classification. Each layer runs the in-tree chunkwise-parallel :class:`GatedDeltaNet`
token mixer (see ``levanter.layers.gated_deltanet``) twice — once over the sequence and
once over the reversed sequence, each direction with its own parameters — and sums the two
outputs (Hydra-style additive combine). Mixing cost is O(seq * chunk) per token instead of
O(seq) for global attention, and the kernel is pure matmul (TPU-friendly, no pretrained
weights: this architecture trains from scratch on labeled data).

Padding: examples arrive right-padded with a segment mask (real = segment 0, pad = -1;
see ``levanter.main.train_classifier``). Pad handling is EXACT, not approximate:

- ``GatedDeltaNet`` zeroes its input at pad positions (``attention_mask``) before the
  projections/conv, so pad content never enters the recurrent state. In the forward
  direction pads trail the real tokens and are causally invisible; in the backward
  (flipped) direction pads lead, but their zeroed inputs contribute no state updates
  (k = v = 0 ⇒ rank-1 delta update is 0, and decay of the zero state is zero).
- The MLP/norms are position-wise, so pad hidden states never mix into real positions.
- Pooling is a pad-masked mean, so pad hidden states get weight 0.

Consequently the logits are bitwise-independent of pad-region token ids (tested in
``tests/test_bigdn.py``).

The classifier follows the contract in ``levanter.models.classification`` and is
registered there, so ``levanter.main.train_classifier`` trains it by just setting
``model.type: bigdn`` (warm start is unsupported — there is no pretrained checkpoint).
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Type

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call

from levanter.layers.attention import AttentionMask
from levanter.layers.gated_deltanet import GatedDeltaNet, GatedDeltaNetConfig
from levanter.models.classification import ClassificationExample, register_classifier_arch, save_eqx_classifier
from levanter.models.llama import LlamaMlp
from levanter.models.lm_model import LmConfig
from levanter.utils.activation import ActivationFunctionEnum


@LmConfig.register_subclass("bigdn")
@dataclass(frozen=True)
class BigdnConfig(LmConfig):
    """Config for the BiGDN encoder classifier.

    Defaults target the ModernBERT-classifier deployment: 8192-token docs, the ModernBERT
    tokenizer (vocab 50368, pad 50283) so the existing tokenized TreeCaches are reused as-is.
    Head layout follows the GDN convention (head dim 128): ``num_k_heads * head_k_dim`` must
    equal ``num_v_heads * head_v_dim`` (rectangular states are allowed but keep it square
    unless there is a reason not to).
    """

    max_seq_len: int = 8192
    hidden_dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 12

    # GDN head layout (per direction). 4 heads x 128 = 512 = hidden_dim.
    num_k_heads: int = 4
    num_v_heads: int = 4
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_kernel_size: int = 4
    # Chunk length for the chunkwise-parallel kernel; 8192 tokens -> 128 sequential chunk steps.
    gdn_chunk_size: int = 64

    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    layer_norm_epsilon: float = 1e-6

    num_labels: int = 2
    vocab_size: int = 50368  # ModernBERT tokenizer
    pad_token_id: int = 50283  # ModernBERT [PAD]

    @property
    def model_type(self) -> Type["BigdnForSequenceClassification"]:  # pyrefly: ignore[bad-override]
        return BigdnForSequenceClassification

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def Mlp(self) -> Axis:
        return Axis("mlp", self.intermediate_dim)

    @property
    def Label(self) -> Axis:
        return Axis("label", self.num_labels)

    def __post_init__(self):
        if self.num_k_heads * self.head_k_dim != self.num_v_heads * self.head_v_dim:
            raise ValueError(
                f"key_dim ({self.num_k_heads}x{self.head_k_dim}) != value_dim "
                f"({self.num_v_heads}x{self.head_v_dim})"
            )
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be a multiple of num_k_heads={self.num_k_heads}")

    def gdn_config(self) -> GatedDeltaNetConfig:
        return GatedDeltaNetConfig(
            Embed=self.Embed,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            rms_norm_eps=self.layer_norm_epsilon,
        )

    def mk_norm(self, axis) -> hnn.RmsNorm:
        return hnn.RmsNorm.init(axis, eps=self.layer_norm_epsilon, use_bias=False)

    def flops_per_token(self, vocab_size: int, context_length: int) -> Optional[float]:
        """Analytic forward-pass FLOPs/token (matmul 2mn convention), independent of context length.

        Counts both directions of the GDN mixer (projections, depthwise conv, and the chunkwise
        kernel's intra-chunk triangular ops + cross-chunk state bridge) plus the SwiGLU MLP.
        """
        key_dim = self.num_k_heads * self.head_k_dim
        value_dim = self.num_v_heads * self.head_v_dim
        c = self.gdn_chunk_size
        proj = 2 * self.hidden_dim * (2 * key_dim + 2 * value_dim) + 2 * self.hidden_dim * 2 * self.num_v_heads
        conv = 2 * self.conv_kernel_size * (2 * key_dim + value_dim)
        # per V-head, per token: A_raw + forward-substitution + T@(betaV) + T@(betaK*decay)
        # + q k^T + attn@v_new + (v_prime, inter, state-update against S in R^{dk x dv})
        dk, dv = self.head_k_dim, self.head_v_dim
        kernel_per_head = (
            2 * c * dk + 2 * c * c + 2 * c * dv + 2 * c * dk + 2 * c * dk + 2 * c * dv + 3 * (2 * dk * dv)
        )
        kernel = self.num_v_heads * kernel_per_head
        out_proj = 2 * value_dim * self.hidden_dim
        gdn_one_direction = proj + conv + kernel + out_proj
        mlp = 2 * 3 * self.hidden_dim * self.intermediate_dim
        return self.num_layers * (2 * gdn_one_direction + mlp)


def _flip_positions(x: NamedArray) -> NamedArray:
    """Reverse a NamedArray along its ``position`` axis (lax.rev under the hood)."""
    i = x.axes.index(x.resolve_axis("position"))
    return hax.named(jnp.flip(x.array, axis=i), x.axes)


def _pad_weight(attn_mask: AttentionMask | NamedArray | None) -> Optional[NamedArray]:
    """Per-position real-token weight (1 real / 0 pad) from the classification segment mask.

    ``None`` (no mask, or no segment ids) means "no padding". Materialized [Pos, KeyPos] masks
    carry no per-position pad information a linear-time mixer can use, so they are rejected.
    """
    if attn_mask is None:
        return None
    if isinstance(attn_mask, AttentionMask):
        if attn_mask.segment_ids is None:
            return None
        return (attn_mask.segment_ids[0] >= 0).astype(jnp.float32)
    raise ValueError("bigdn requires an AttentionMask with segment_ids (or None), not a materialized mask")


class BigdnEncoderLayer(eqx.Module):
    """Pre-norm residual block: bidirectional GDN mixing, then a SwiGLU MLP.

    The two directions have independent parameters; the backward direction runs the causal
    GDN kernel over the position-reversed sequence and its output is reversed back before
    the additive combine.
    """

    chunk_size: int = eqx.field(static=True)
    gdn_norm: hnn.RmsNorm
    gdn_fwd: GatedDeltaNet
    gdn_bwd: GatedDeltaNet
    mlp_norm: hnn.RmsNorm
    mlp: LlamaMlp

    @staticmethod
    def init(config: BigdnConfig, *, key) -> "BigdnEncoderLayer":
        k_fwd, k_bwd, k_mlp = jrandom.split(key, 3)
        gdn_cfg = config.gdn_config()
        return BigdnEncoderLayer(
            chunk_size=config.gdn_chunk_size,
            gdn_norm=config.mk_norm(config.Embed),
            gdn_fwd=GatedDeltaNet.init(gdn_cfg, key=k_fwd),
            gdn_bwd=GatedDeltaNet.init(gdn_cfg, key=k_bwd),
            mlp_norm=config.mk_norm(config.Embed),
            mlp=LlamaMlp.init(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=False),
        )

    @named_call
    def __call__(self, x: NamedArray, pad_weight: Optional[NamedArray]) -> NamedArray:
        h = self.gdn_norm(x)
        fwd, _ = self.gdn_fwd(h, inference=False, chunk_size=self.chunk_size, attention_mask=pad_weight)
        h_rev = _flip_positions(h)
        w_rev = _flip_positions(pad_weight) if pad_weight is not None else None
        bwd_rev, _ = self.gdn_bwd(h_rev, inference=False, chunk_size=self.chunk_size, attention_mask=w_rev)
        x = x + fwd + _flip_positions(bwd_rev)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class BigdnEncoder(eqx.Module):
    config: BigdnConfig = eqx.field(static=True)
    embeddings: hnn.Embedding
    embed_norm: hnn.RmsNorm
    layers: list  # list[BigdnEncoderLayer]
    final_norm: hnn.RmsNorm

    @staticmethod
    def init(Vocab: Axis, config: BigdnConfig, *, key) -> "BigdnEncoder":
        k_emb, k_layers = jrandom.split(key, 2)
        layer_keys = jrandom.split(k_layers, config.num_layers)
        return BigdnEncoder(
            config=config,
            embeddings=hnn.Embedding.init(Vocab, config.Embed, key=k_emb),
            embed_norm=config.mk_norm(config.Embed),
            layers=[BigdnEncoderLayer.init(config, key=k) for k in layer_keys],
            final_norm=config.mk_norm(config.Embed),
        )

    @named_call
    def __call__(self, input_ids: NamedArray, pad_weight: Optional[NamedArray]) -> NamedArray:
        x = self.embed_norm(self.embeddings(input_ids))
        for layer in self.layers:
            x = layer(x, pad_weight)
        return self.final_norm(x)


class BigdnForSequenceClassification(eqx.Module):
    """BiGDN encoder + classification head: pad-masked mean pool -> dense+act+norm -> classifier.

    Satisfies the ``levanter.models.classification`` model contract: ``__call__`` returns logits
    with an axis literally named ``"label"``; ``compute_loss`` is softmax cross-entropy.
    """

    encoder: BigdnEncoder
    dense: hnn.Linear
    head_norm: hnn.RmsNorm
    classifier: hnn.Linear

    @property
    def config(self) -> BigdnConfig:
        return self.encoder.config

    @property
    def Vocab(self) -> Axis:
        return self.encoder.embeddings.Vocab

    @property
    def Label(self) -> Axis:
        return self.config.Label

    @classmethod
    def init(cls, Vocab: Axis, config: BigdnConfig, *, key) -> "BigdnForSequenceClassification":
        k_enc, k_dense, k_cls = jrandom.split(key, 3)
        encoder = BigdnEncoder.init(Vocab, config, key=k_enc)
        dense = hnn.Linear.init(
            In=config.Embed, Out=config.Embed.alias("head_embed"), key=k_dense, use_bias=False, out_first=True
        )
        classifier = hnn.Linear.init(In=config.Embed, Out=config.Label, key=k_cls, use_bias=False, out_first=True)
        return BigdnForSequenceClassification(encoder, dense, config.mk_norm(config.Embed), classifier)

    def _pool(self, hidden: NamedArray, pad_weight: Optional[NamedArray]) -> NamedArray:
        if pad_weight is None:
            return hidden.mean("position")
        w = pad_weight.astype(hidden.dtype)
        return (hidden * w).sum("position") / w.sum("position")

    @named_call
    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        # The GDN kernels require a "batch" axis; normalize unbatched inputs at this boundary.
        Batch1 = Axis("batch", 1)
        unbatched = "batch" not in (a.name for a in input_ids.axes)
        if unbatched:
            input_ids = input_ids.broadcast_axis(Batch1)
        pad_weight = _pad_weight(attn_mask)
        hidden = self.encoder(input_ids, pad_weight)
        pooled = self._pool(hidden, pad_weight)
        h = hnn.silu(self.dense(pooled)).rename({"head_embed": "embed"})
        logits = self.classifier(self.head_norm(h))
        if unbatched:
            logits = logits["batch", 0]
        return logits

    def compute_loss(
        self,
        example: ClassificationExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> NamedArray:
        logits = self(example.tokens, example.attn_mask, key=key).astype(jnp.float32)
        target = hax.nn.one_hot(example.label, self.Label, dtype=logits.dtype)
        return hax.nn.cross_entropy_loss(logits, self.Label, target, reduction, reduction_axis=reduction_axis)

    def resize_vocab(self, new_size: int, key=None) -> "BigdnForSequenceClassification":
        new_embeddings = self.encoder.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, encoder=dataclasses.replace(self.encoder, embeddings=new_embeddings))


def _build_bigdn_classifier(
    config: BigdnConfig, Vocab: Axis, *, key, warm_start: bool, axis_mapping=None, compute_dtype=None
) -> BigdnForSequenceClassification:
    if warm_start:
        raise ValueError("bigdn has no pretrained checkpoint; set warm_start=False to train from scratch")
    return BigdnForSequenceClassification.init(Vocab, config, key=key)


register_classifier_arch(BigdnConfig, build=_build_bigdn_classifier, save=save_eqx_classifier)
