# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A JAX forward pass for `gte-base-en-v1.5` sequence classification (WebOrganizer Topic/Format).

Why this exists: the reference implementation is HF PyTorch and its fast paths (`xformers`
memory-efficient attention, `unpad_inputs`) are CUDA-only, so on TPU it falls back to a CPU-speed
eager path. Measured on CPU, an 8k-token doc scores at **0.03 docs/s** — labelling millions of docs
that way is hopeless. This module is ~200 lines of plain JAX that runs the same math on TPU at large
batch, with no torch_xla install hacks: it loads on the existing `--extra tpu` container.

Architecture (verified against the HF repo's `modeling.py`, not guessed):
  * 12 layers, 768 hidden, 12 heads (head_dim 64), vocab 30528, no token-type embeddings
    (`type_vocab_size=0`), no absolute position embeddings (`position_embedding_type="rope"`).
  * **Post-norm**: ``h = attn_ln(h + attn(h))`` then ``h = mlp_ln(h + mlp(h))``.
  * Packed QKV (`pack_qkv=true`): one `[3*768, 768]` weight, split into q/k/v.
  * `NewGatedMLP`: `up_gate_proj` 768->2*3072 (no bias) split **up FIRST then gate**;
    ``down_proj(gelu(gate) * up)`` with bias. `hidden_act="gelu"` is exact (erf) gelu, not tanh.
  * Pooler: CLS -> dense 768->768 -> tanh, then `classifier` 768->num_labels.
  * `logn_attention_scale=false`, so attention scale is the plain 1/sqrt(head_dim).

**RoPE is not re-derived here.** The checkpoint uses NTK scaling (`rope_theta=500000`,
`{"type":"ntk","factor":2.0}`), whose effective `inv_freq` is
``1/((theta*factor)**(2i/d)) / factor**(2/d)`` — easy to get subtly wrong. Instead
`extract_params` reads the materialised `cos_cached`/`sin_cached` buffers straight off the loaded HF
module and bakes them in as constants. That makes the rope bit-exact by construction and removes the
single most likely source of a silent numerical error.

`test_weborganizer_gte_jax.py` asserts parity against HF on real text; that test is the contract.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

NUM_HEADS = 12
HEAD_DIM = 64
INTERMEDIATE_SIZE = 3072
MASK_FILL = -1e9  # additive mask for padded keys, in fp32
QUERY_BLOCK = 1024  # attention is computed this many queries at a time; bounds peak memory


def extract_params(model) -> dict:
    """Pull every weight (and the materialised rope cache) off a loaded HF `NewForSequenceClassification`."""
    torch_model = model.new
    params: dict = {
        "word_embeddings": _np(torch_model.embeddings.word_embeddings.weight),
        "emb_ln_w": _np(torch_model.embeddings.LayerNorm.weight),
        "emb_ln_b": _np(torch_model.embeddings.LayerNorm.bias),
        # Bit-exact rope: the NTK-scaled cache as HF built it, rather than our re-derivation.
        "rope_cos": _np(torch_model.embeddings.rotary_emb.cos_cached),
        "rope_sin": _np(torch_model.embeddings.rotary_emb.sin_cached),
        "pooler_w": _np(torch_model.pooler.dense.weight),
        "pooler_b": _np(torch_model.pooler.dense.bias),
        "classifier_w": _np(model.classifier.weight),
        "classifier_b": _np(model.classifier.bias),
        "layers": [],
    }
    for layer in torch_model.encoder.layer:
        params["layers"].append(
            {
                "qkv_w": _np(layer.attention.qkv_proj.weight),
                "qkv_b": _np(layer.attention.qkv_proj.bias),
                "o_w": _np(layer.attention.o_proj.weight),
                "o_b": _np(layer.attention.o_proj.bias),
                "attn_ln_w": _np(layer.attn_ln.weight),
                "attn_ln_b": _np(layer.attn_ln.bias),
                "up_gate_w": _np(layer.mlp.up_gate_proj.weight),
                "down_w": _np(layer.mlp.down_proj.weight),
                "down_b": _np(layer.mlp.down_proj.bias),
                "mlp_ln_w": _np(layer.mlp_ln.weight),
                "mlp_ln_b": _np(layer.mlp_ln.bias),
            }
        )
    logger.info("extracted %d layers; rope cache %s", len(params["layers"]), params["rope_cos"].shape)
    return params


def _np(tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def to_device(params: dict, dtype) -> dict:
    """Move params to device arrays. LayerNorm/rope stay fp32 for numerical headroom."""
    keep_fp32 = {"emb_ln_w", "emb_ln_b", "rope_cos", "rope_sin", "attn_ln_w", "attn_ln_b", "mlp_ln_w", "mlp_ln_b"}

    def cast(key, value):
        return jnp.asarray(value, dtype=jnp.float32 if key in keep_fp32 else dtype)

    out = {k: cast(k, v) for k, v in params.items() if k != "layers"}
    out["layers"] = [{k: cast(k, v) for k, v in layer.items()} for layer in params["layers"]]
    return out


def _layer_norm(x, weight, bias, eps: float = 1e-12):
    x32 = x.astype(jnp.float32)
    mean = x32.mean(-1, keepdims=True)
    var = x32.var(-1, keepdims=True)
    normed = (x32 - mean) * jax.lax.rsqrt(var + eps)
    return (normed * weight + bias).astype(x.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(q, k, cos, sin):
    # q/k: [B, T, H, D]; cos/sin: [1, T, 1, D]
    cos = cos.astype(q.dtype)
    sin = sin.astype(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _attention(q, k, v, bias):
    """Exact attention, computed in blocks of queries to bound peak memory.

    A single `[B, H, T, T]` score tensor is not affordable here: gte-base attends globally at every
    layer, so at T=8192 with a token-budgeted batch of 32 that tensor is ~103 GB in fp32 and OOMs the
    chip. Blocking the QUERY axis caps live memory at `[B, H, QUERY_BLOCK, T]` while keeping the math
    identical — each block still softmaxes over all keys, so no online-softmax rescaling is needed.
    """
    # [B, T, H, D] -> [B, H, T, D]
    q, k, v = (jnp.swapaxes(t, 1, 2) for t in (q, k, v))
    seq_len = q.shape[2]
    block = min(QUERY_BLOCK, seq_len)
    scale = jnp.sqrt(jnp.asarray(HEAD_DIM, dtype=jnp.float32)).astype(q.dtype)
    chunks = []
    for start in range(0, seq_len, block):  # unrolled under jit; <=8 blocks at T=8192
        q_block = q[:, :, start : start + block]
        scores = jnp.einsum("bhqd,bhkd->bhqk", q_block, k) / scale
        scores = scores.astype(jnp.float32) + bias  # bias [B,1,1,T] broadcasts over the query block
        probs = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        chunks.append(jnp.einsum("bhqk,bhkd->bhqd", probs, v))
    context = jnp.concatenate(chunks, axis=2) if len(chunks) > 1 else chunks[0]
    return jnp.swapaxes(context, 1, 2)  # back to [B, T, H, D]


def forward(params: dict, input_ids: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
    """Return logits [batch, num_labels]. `attention_mask` is 1 for real tokens, 0 for padding."""
    batch, seq_len = input_ids.shape
    x = params["word_embeddings"][input_ids]
    x = _layer_norm(x, params["emb_ln_w"], params["emb_ln_b"])

    cos = params["rope_cos"][:seq_len][None, :, None, :]
    sin = params["rope_sin"][:seq_len][None, :, None, :]
    bias = (1.0 - attention_mask.astype(jnp.float32))[:, None, None, :] * MASK_FILL

    for layer in params["layers"]:
        qkv = jnp.einsum("btd,ed->bte", x, layer["qkv_w"]) + layer["qkv_b"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = (t.reshape(batch, seq_len, NUM_HEADS, HEAD_DIM) for t in (q, k, v))
        q, k = _apply_rope(q, k, cos, sin)
        context = _attention(q, k, v, bias).reshape(batch, seq_len, NUM_HEADS * HEAD_DIM)
        attn_out = jnp.einsum("btd,ed->bte", context, layer["o_w"]) + layer["o_b"]
        x = _layer_norm(x + attn_out, layer["attn_ln_w"], layer["attn_ln_b"])

        up_gate = jnp.einsum("btd,ed->bte", x, layer["up_gate_w"])  # no bias
        up, gate = up_gate[..., :INTERMEDIATE_SIZE], up_gate[..., INTERMEDIATE_SIZE:]
        hidden = jax.nn.gelu(gate, approximate=False) * up
        mlp_out = jnp.einsum("btd,ed->bte", hidden, layer["down_w"]) + layer["down_b"]
        x = _layer_norm(x + mlp_out, layer["mlp_ln_w"], layer["mlp_ln_b"])

    pooled = jnp.tanh(jnp.einsum("bd,ed->be", x[:, 0], params["pooler_w"]) + params["pooler_b"])
    return jnp.einsum("bd,ed->be", pooled, params["classifier_w"]) + params["classifier_b"]
