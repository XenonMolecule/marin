# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Pooled super-token transformer classifier ("pooled_transformer").

A port of Marin's fast-transformer document-quality model (a pooled super-token
transformer regressor) to the Levanter classifier contract, with a 2-class
softmax head instead of the scalar regression head:

    token ids ──embed──▶ [B, T, E]
              ──mask-aware pool over windows of ``pool_window``──▶ [B, S, E_pool]   (S = T / w)
              ──input proj + learned super-token position──▶ [B, S, D]
              ──N pre-norm transformer layers over super-tokens──▶ [B, S, D]
              ──final masked pool over S──▶ [B, D]
              ──LayerNorm + linear head──▶ [B, num_labels]

Pooling at ``w``-token boundaries amortizes the transformer's per-token cost by
``w`` (~64x), which keeps the model around ~1M FLOPs/token while still running
real self-attention. ``pool_kind`` selects how a window of token embeddings
collapses to one super-token: plain ``mean`` / ``max``, the multi-statistic
``meanmaxmin`` concat (captures spread, not just centroid), or a learned
``attn`` pool.

Differences from the source model (beyond the classification head):

- Pad handling follows the Levanter classifier convention: real tokens come from
  ``attn_mask.segment_ids`` (real = 0, pad = -1) when an ``AttentionMask`` with
  segment ids is provided, falling back to ``ids != pad_token_id`` otherwise.
  The source's ``PAD_ID = 0`` is wrong here — ModernBERT's pad is 50283 and
  token id 0 is a real token.
- Inputs/outputs are ``NamedArray``s (axes ``(batch?, "position")`` in, batch
  axes + ``"label"`` out); internally the model runs the source's raw ``jnp``
  path with plain-array parameters (replicated, fine at this size).

Masked positions are inert everywhere: pooling ignores them, empty windows
become inactive super-tokens, and attention never attends to inactive
super-tokens. Matmuls run in bf16 with f32 accumulation (TPU MXU) regardless of
the trainer's mixed-precision policy, and the loss is computed in f32.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

import haliax as hax
from haliax import Axis, NamedArray

from levanter.layers.attention import AttentionMask
from levanter.models.classification import (
    ClassificationExample,
    load_eqx_classifier,
    register_classifier_arch,
    save_eqx_classifier,
)
from levanter.models.lm_model import LmConfig

POOL_KINDS = ("mean", "max", "meanmaxmin", "attn", "mean_minor", "matmul")
FINAL_POOLS = ("mean", "attn")
NEG_INF = -1e30
COMPUTE_DTYPE = jnp.bfloat16
MODERNBERT_VOCAB_SIZE = 50368  # answerdotai/ModernBERT tokenizer, shared with the ModernBERT caches
MODERNBERT_PAD_TOKEN_ID = 50283


@LmConfig.register_subclass("pooled_transformer")
@dataclass(frozen=True)
class PooledTransformerConfig(LmConfig):
    """Config for the pooled super-token transformer classifier."""

    max_seq_len: int = 4096
    pool_window: int = 64
    pool_kind: str = "meanmaxmin"
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    final_pool: str = "mean"
    num_labels: int = 2
    pad_token_id: int = MODERNBERT_PAD_TOKEN_ID

    def __post_init__(self) -> None:
        if self.max_seq_len % self.pool_window != 0:
            raise ValueError(f"max_seq_len={self.max_seq_len} must be divisible by pool_window={self.pool_window}")
        if self.pool_kind not in POOL_KINDS:
            raise ValueError(f"pool_kind={self.pool_kind} not in {POOL_KINDS}")
        if self.final_pool not in FINAL_POOLS:
            raise ValueError(f"final_pool={self.final_pool} not in {FINAL_POOLS}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} not divisible by num_heads={self.num_heads}")

    @property
    def num_super_tokens(self) -> int:
        return self.max_seq_len // self.pool_window

    @property
    def pool_out_dim(self) -> int:
        return self.embed_dim * 3 if self.pool_kind == "meanmaxmin" else self.embed_dim

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def Label(self) -> Axis:
        return Axis("label", self.num_labels)

    @property
    def model_type(self) -> Type["PooledTransformerClassifier"]:  # pyrefly: ignore[bad-override]
        return PooledTransformerClassifier

    def flops_per_token(self, vocab_size: int, context_length: int) -> float:
        """Forward FLOPs per *input* token (multiply-add counted as 2).

        Embedding lookup is a gather (~0 FLOPs). The dominant terms are the
        per-super-token linear layers (amortized by ``pool_window``) plus the
        (negligible) S^2 attention. This is the inference cost that matters when
        scoring a whole corpus.
        """
        d = self.hidden_dim
        s = max(1, context_length // self.pool_window)
        t = context_length
        d_ff = d * self.mlp_ratio
        proj = 2 * self.pool_out_dim * d * s  # input projection of pooled vectors
        attn_proj = 2 * (4 * d * d) * s  # qkv (3) + output (1) projections
        attn_scores = 2 * (2 * s * s * d)  # QK^T and AV
        mlp = 2 * (2 * d * d_ff) * s
        per_layer = attn_proj + attn_scores + mlp
        head = 2 * d * self.num_labels
        total = proj + self.num_layers * per_layer + head
        return total / t


def _glorot(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    fan_in, fan_out = shape[0], shape[-1]
    return jax.random.normal(key, shape) * math.sqrt(2.0 / (fan_in + fan_out))


def _matmul(x: Array, w: Array) -> Array:
    """``x @ w`` in bf16 (TPU MXU) with f32 accumulation/output."""
    out = jnp.matmul(x.astype(COMPUTE_DTYPE), w.astype(COMPUTE_DTYPE), preferred_element_type=jnp.float32)
    return out.astype(jnp.float32)


def _layer_norm(x: Array, gamma: Array, beta: Array) -> Array:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + 1e-5) * gamma + beta


def _dropout(x: Array, p: float, key: PRNGKeyArray | None) -> Array:
    """Inverted dropout; a ``None`` key (inference) or ``p == 0`` is the identity."""
    if p == 0.0 or key is None:
        return x
    keep = jax.random.bernoulli(key, 1.0 - p, x.shape)
    return jnp.where(keep, x / (1.0 - p), 0.0)


class PooledTransformerLayer(eqx.Module):
    """Batched masked pre-norm transformer block over super-tokens."""

    ln1_g: Array
    ln1_b: Array
    ln2_g: Array
    ln2_b: Array
    wqkv: Array  # [D, 3D]
    wo: Array  # [D, D]
    w1: Array  # [D, D_ff]
    w2: Array  # [D_ff, D]
    num_heads: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, dropout: float, *, key: PRNGKeyArray):
        kqkv, ko, k1, k2 = jax.random.split(key, 4)
        self.ln1_g = jnp.ones(dim)
        self.ln1_b = jnp.zeros(dim)
        self.ln2_g = jnp.ones(dim)
        self.ln2_b = jnp.zeros(dim)
        self.wqkv = _glorot(kqkv, (dim, 3 * dim))
        self.wo = _glorot(ko, (dim, dim))
        self.w1 = _glorot(k1, (dim, dim * mlp_ratio))
        self.w2 = _glorot(k2, (dim * mlp_ratio, dim))
        self.num_heads = num_heads
        self.dropout = dropout

    def __call__(self, x: Array, valid: Array, *, key: PRNGKeyArray | None) -> Array:
        b, s, d = x.shape
        h, hd = self.num_heads, d // self.num_heads
        ka, km = (None, None) if key is None else jax.random.split(key)

        normed = _layer_norm(x, self.ln1_g, self.ln1_b)
        qkv = _matmul(normed, self.wqkv).reshape(b, s, 3, h, hd)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [b, s, h, hd]
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(hd)
        scores = jnp.where(valid[:, None, None, :].astype(bool), scores, NEG_INF)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(b, s, d)
        x = x + _dropout(_matmul(ctx, self.wo), self.dropout, ka)

        normed = _layer_norm(x, self.ln2_g, self.ln2_b)
        mlp = _matmul(jax.nn.gelu(_matmul(normed, self.w1)), self.w2)
        x = x + _dropout(mlp, self.dropout, km)
        return x


class PooledTransformerClassifier(eqx.Module):
    """The pooled super-token transformer with a ``num_labels``-way classification head.

    Satisfies the duck-typed classifier contract in :mod:`levanter.models.classification`:
    ``__call__`` returns logits with a ``"label"`` axis and ``compute_loss`` is softmax CE.
    """

    config: PooledTransformerConfig = eqx.field(static=True)
    embed: Array  # [vocab, E]
    pool_query: Array  # [E]
    proj_w: Array  # [pool_out_dim, D]
    proj_b: Array  # [D]
    pos_embed: Array  # [S, D]
    layers: list[PooledTransformerLayer]
    final_query: Array  # [D]
    head_g: Array
    head_b: Array
    head_w: Array  # [D, num_labels]

    @classmethod
    def init(cls, Vocab: Axis, config: PooledTransformerConfig, *, key: PRNGKeyArray) -> "PooledTransformerClassifier":
        ke, kpq, kpr, kpos, klayers, kfq, khead = jax.random.split(key, 7)
        embed = jax.random.normal(ke, (Vocab.size, config.embed_dim)) * 0.02
        pool_query = jax.random.normal(kpq, (config.embed_dim,)) * 0.02
        proj_w = _glorot(kpr, (config.pool_out_dim, config.hidden_dim))
        proj_b = jnp.zeros(config.hidden_dim)
        pos_embed = jax.random.normal(kpos, (config.num_super_tokens, config.hidden_dim)) * 0.02
        layer_keys = jax.random.split(klayers, max(1, config.num_layers))
        layers = [
            PooledTransformerLayer(config.hidden_dim, config.num_heads, config.mlp_ratio, config.dropout, key=lk)
            for lk in layer_keys[: config.num_layers]
        ]
        final_query = jax.random.normal(kfq, (config.hidden_dim,)) * 0.02
        head_g = jnp.ones(config.hidden_dim)
        head_b = jnp.zeros(config.hidden_dim)
        head_w = _glorot(khead, (config.hidden_dim, config.num_labels))
        return cls(config, embed, pool_query, proj_w, proj_b, pos_embed, layers, final_query, head_g, head_b, head_w)

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.embed.shape[0])

    @property
    def Label(self) -> Axis:
        return self.config.Label

    def _real_token_mask(self, input_ids: NamedArray, attn_mask: AttentionMask | NamedArray | None) -> Array:
        """Per-position real-token mask [.., t] (1 = real, 0 = pad), axes ordered like ``input_ids``.

        Levanter's classification datasets mark pads via segment ids (real = 0, pad = -1); when no
        segment ids are present we fall back to ``ids != pad_token_id``.
        """
        if isinstance(attn_mask, AttentionMask) and attn_mask.segment_ids is not None:
            seg = attn_mask.segment_ids[0].rearrange(input_ids.axes)
            return (seg.array >= 0).astype(jnp.float32)
        return (input_ids.array != self.config.pad_token_id).astype(jnp.float32)

    def _pool_windows(self, emb: Array, mask: Array) -> tuple[Array, Array]:
        """Collapse windows of ``pool_window`` tokens. Returns (pooled, valid).

        Masked (pad) positions contribute nothing; a window with zero real tokens becomes an
        inactive super-token (``valid = 0``) that downstream attention and pooling skip.
        """
        cfg = self.config
        b, t, e = emb.shape
        s, w = t // cfg.pool_window, cfg.pool_window
        wemb = emb.reshape(b, s, w, e)
        wmask = mask.reshape(b, s, w)
        counts = wmask.sum(axis=2, keepdims=True)  # [b, s, 1]
        valid = (counts[..., 0] > 0).astype(jnp.float32)  # [b, s]
        denom = jnp.maximum(counts, 1.0)
        m3 = wmask[..., None]

        if cfg.pool_kind == "mean":
            pooled = (wemb * m3).sum(axis=2) / denom
        elif cfg.pool_kind == "mean_minor":
            # Same value as "mean", but the window axis is moved MINOR-most before the reduce.
            # Reducing over a major axis of a 4-D reshape is what XLA can lower as a spatial
            # convolution (whose TPU backward segfaults); a minor-axis reduce is a plain row reduce.
            pooled = (wemb * m3).transpose(0, 1, 3, 2).sum(axis=3) / denom
        elif cfg.pool_kind == "matmul":
            # Same value as "mean" with NO windowed reduce at all: contract the token axis against
            # a [t, s] block indicator, which is a dense MXU matmul on TPU.
            window_of = jnp.arange(t) // cfg.pool_window
            indicator = (window_of[:, None] == jnp.arange(s)[None, :]).astype(emb.dtype)  # [t, s]
            summed = jnp.einsum("bte,ts->bse", emb * mask[..., None], indicator)
            pooled = summed / denom
        elif cfg.pool_kind == "max":
            pooled = jnp.where(m3 > 0, wemb, NEG_INF).max(axis=2)
            pooled = jnp.where(valid[..., None] > 0, pooled, 0.0)
        elif cfg.pool_kind == "meanmaxmin":
            mean = (wemb * m3).sum(axis=2) / denom
            mx = jnp.where(valid[..., None] > 0, jnp.where(m3 > 0, wemb, NEG_INF).max(axis=2), 0.0)
            mn = jnp.where(valid[..., None] > 0, jnp.where(m3 > 0, wemb, -NEG_INF).min(axis=2), 0.0)
            pooled = jnp.concatenate([mean, mx, mn], axis=-1)
        else:  # attn: learned query, softmax over the window
            scores = (wemb @ self.pool_query) / math.sqrt(e)  # [b, s, w]
            scores = jnp.where(wmask > 0, scores, NEG_INF)
            attn = jax.nn.softmax(scores, axis=2)
            pooled = jnp.einsum("bsw,bswe->bse", attn, wemb)
            pooled = jnp.where(valid[..., None] > 0, pooled, 0.0)
        return pooled, valid

    def _forward(self, ids: Array, mask: Array, *, key: PRNGKeyArray | None) -> Array:
        """Raw path: [b, t] int ids + [b, t] real-token mask -> [b, num_labels] f32 logits."""
        cfg = self.config
        t = ids.shape[1]
        if t % cfg.pool_window != 0:
            raise ValueError(f"sequence length {t} must be divisible by pool_window={cfg.pool_window}")
        s = t // cfg.pool_window
        if s > cfg.num_super_tokens:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={cfg.max_seq_len}")

        emb = jnp.take(self.embed, ids, axis=0).astype(jnp.float32)  # [b, t, e]
        pooled, valid = self._pool_windows(emb, mask)  # [b, s, pool_out], [b, s]
        h = _matmul(pooled, self.proj_w) + self.proj_b + self.pos_embed[:s]  # [b, s, d]

        n = cfg.num_layers
        layer_keys = [None] * n if key is None else list(jax.random.split(key, n)) if n else []
        for layer, lk in zip(self.layers, layer_keys, strict=True):
            h = layer(h, valid, key=lk)

        if cfg.final_pool == "mean":
            pooled_doc = (h * valid[..., None]).sum(axis=1) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1.0)
        else:  # attn pool over super-tokens
            scores = (h @ self.final_query) / math.sqrt(cfg.hidden_dim)  # [b, s]
            scores = jnp.where(valid > 0, scores, NEG_INF)
            attn = jax.nn.softmax(scores, axis=1)
            pooled_doc = jnp.einsum("bs,bsd->bd", attn, h)

        normed = _layer_norm(pooled_doc, self.head_g, self.head_b)
        return _matmul(normed, self.head_w)  # [b, num_labels]

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NamedArray:
        Pos = input_ids.resolve_axis("position")
        batch_axes = tuple(ax for ax in input_ids.axes if ax.name != "position")
        if len(batch_axes) > 1:
            raise ValueError(f"expected at most one batch axis, got {input_ids.axes}")
        ordered = input_ids.rearrange((*batch_axes, Pos))
        ids = ordered.array
        mask = self._real_token_mask(ordered, attn_mask)
        if not batch_axes:
            ids, mask = ids[None], mask[None]
        logits = self._forward(ids, mask, key=key)
        if not batch_axes:
            logits = logits[0]
        return hax.named(logits, (*batch_axes, self.Label))

    def compute_loss(
        self,
        example: ClassificationExample,
        *,
        key: PRNGKeyArray | None = None,
        reduction: hax.ReductionFunction | None = hax.mean,  # pyrefly: ignore[bad-function-definition]
        reduction_axis: hax.AxisSelection | None = None,
    ) -> NamedArray:
        logits = self(example.tokens, example.attn_mask, key=key).astype(jnp.float32)
        target = hax.nn.one_hot(example.label, self.Label, dtype=logits.dtype)
        return hax.nn.cross_entropy_loss(logits, self.Label, target, reduction, reduction_axis=reduction_axis)

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "PooledTransformerClassifier":
        old_size, e = self.embed.shape
        if new_size == old_size:
            return self
        if new_size < old_size:
            new_embed = self.embed[:new_size]
        else:
            if key is not None:
                extra = jax.random.normal(key, (new_size - old_size, e)) * 0.02
            else:
                extra = jnp.zeros((new_size - old_size, e))
            new_embed = jnp.concatenate([self.embed, extra.astype(self.embed.dtype)], axis=0)
        return dataclasses.replace(self, embed=new_embed)  # pyrefly: ignore[bad-specialization]


def count_params(model: PooledTransformerClassifier) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))


def load_pooled_transformer_classifier(
    config: PooledTransformerConfig, path: str, *, vocab_size: int = MODERNBERT_VOCAB_SIZE
) -> PooledTransformerClassifier:
    """Load a classifier saved by :func:`levanter.models.classification.save_eqx_classifier`."""
    template = eqx.filter_eval_shape(
        PooledTransformerClassifier.init, Axis("vocab", vocab_size), config, key=jrandom.PRNGKey(0)
    )
    return load_eqx_classifier(template, path)


def _build_pooled_transformer_classifier(
    config: PooledTransformerConfig, Vocab: Axis, *, key, warm_start: bool, axis_mapping=None, compute_dtype=None
) -> PooledTransformerClassifier:
    if warm_start:
        raise ValueError("pooled_transformer has no pretrained weights; set warm_start=False")
    return PooledTransformerClassifier.init(Vocab, config, key=key)


register_classifier_arch(PooledTransformerConfig, build=_build_pooled_transformer_classifier, save=save_eqx_classifier)
