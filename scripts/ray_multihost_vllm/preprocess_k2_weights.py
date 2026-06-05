#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline weight preprocessing for Kimi K2-Instruct on TPU.

Runs the full FP8 blockwise dequant→requant→reorder pipeline ONCE on CPU,
saves pre-processed weights to new safetensors files. vLLM then loads these
with zero processing — pure I/O + device placement.

This eliminates the 35+ hour loading bottleneck for 384-expert MoE layers.

Requirements:
    - ~250GB RAM (runs on v5p-32 hosts which have 440GB)
    - CPU-only (JAX_PLATFORMS=cpu)
    - Access to K2-Instruct safetensors via gcsfuse or local path
    - tpu_inference package (available in vllm/vllm-tpu:nightly Docker image)

Usage:
    # Single host, 4 workers:
    JAX_PLATFORMS=cpu python preprocess_k2_weights.py \
        --input-dir /mnt/gcs-models \
        --output-dir /mnt/gcs-output \
        --tp-size 4 --workers 4

    # Specific layer range (for cross-host parallelism):
    JAX_PLATFORMS=cpu python preprocess_k2_weights.py \
        --input-dir /mnt/gcs-models \
        --output-dir /mnt/gcs-output \
        --tp-size 4 --workers 4 \
        --layer-start 1 --layer-end 15
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from multiprocessing import Pool

# Force CPU-only JAX before any JAX imports
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import ml_dtypes
import numpy as np

# tpu_inference imports (available inside vllm-tpu Docker image)
sys.path.insert(0, "/workspace/tpu_inference")
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights,
    MoEBackend,
    process_moe_weights,
    quantize_moe_weights,
)
from tpu_inference.layers.common.quantization import dequantize_tensor
from tpu_inference.layers.common.quantization.fp8 import (
    process_blockwise_fp8_linear_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
logger = logging.getLogger(__name__)
logger.handlers = [logging.StreamHandler(sys.stdout)]
logger.handlers[0].flush = lambda: sys.stdout.flush()

# ============================================================================
# Constants
# ============================================================================

# K2-Instruct architecture (from verified config.json inspection)
NUM_EXPERTS = 384
HIDDEN_SIZE = 7168
MOE_INTERMEDIATE_SIZE = 2048
DENSE_INTERMEDIATE_SIZE = 18432
NUM_HEADS = 64  # K2-Instruct (not 128 like DS-V3)
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
FIRST_K_DENSE_REPLACE = 1  # layer 0 is dense, 1-60 are MoE
NUM_LAYERS = 61
WEIGHT_BLOCK_SIZE = (128, 128)
W13_INTERLEAVE = False  # silu activation


# ============================================================================
# Expert weight loading
# ============================================================================


def load_expert_weights(shard_files, layer_idx, num_experts=NUM_EXPERTS):
    """Load all expert weights for one MoE layer from safetensors shards.

    Returns dict with numpy arrays:
        gate_weight: [E, 2048, 7168] float8_e4m3fn
        gate_scale:  [E, 16, 56] float32
        up_weight, up_scale: same shapes as gate
        down_weight: [E, 7168, 2048] float8_e4m3fn
        down_scale:  [E, 56, 16] float32
    """
    import torch
    from safetensors import safe_open

    prefix = f"model.layers.{layer_idx}.mlp.experts."

    # Pre-allocate per-expert storage
    result = {
        "gate_weight": [None] * num_experts,
        "gate_scale": [None] * num_experts,
        "up_weight": [None] * num_experts,
        "up_scale": [None] * num_experts,
        "down_weight": [None] * num_experts,
        "down_scale": [None] * num_experts,
    }

    proj_map = {
        ("gate_proj", "weight"): "gate_weight",
        ("gate_proj", "weight_scale_inv"): "gate_scale",
        ("up_proj", "weight"): "up_weight",
        ("up_proj", "weight_scale_inv"): "up_scale",
        ("down_proj", "weight"): "down_weight",
        ("down_proj", "weight_scale_inv"): "down_scale",
    }

    # Use safetensors index to only open relevant shards
    index_path = os.path.join(os.path.dirname(shard_files[0]), "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as idx_f:
            index = json.load(idx_f)
        weight_map = index.get("weight_map", {})
        # Find which shards contain this layer's expert weights
        relevant_shards = set()
        for key, shard_name in weight_map.items():
            if key.startswith(prefix):
                shard_path = os.path.join(os.path.dirname(shard_files[0]), shard_name)
                relevant_shards.add(shard_path)
        shard_list = sorted(relevant_shards)
        print(f"  Using index: {len(shard_list)} shards (of {len(shard_files)} total)", flush=True)
    else:
        shard_list = shard_files
        print(f"  No index file, scanning all {len(shard_list)} shards", flush=True)

    for sf_path in shard_list:
        with safe_open(sf_path, framework="torch") as f:
            for key in f.keys():
                if not key.startswith(prefix):
                    continue
                rest = key[len(prefix) :]
                parts = rest.split(".")
                if len(parts) != 3:
                    continue
                expert_id = int(parts[0])
                proj_name = parts[1]
                param_type = parts[2]

                map_key = proj_map.get((proj_name, param_type))
                if map_key is None:
                    continue

                t = f.get_tensor(key)
                if param_type == "weight":
                    arr = t.cpu().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
                else:
                    arr = t.cpu().float().numpy()

                result[map_key][expert_id] = arr

    # Verify all experts loaded
    for key, experts in result.items():
        loaded = sum(1 for e in experts if e is not None)
        if loaded != num_experts:
            raise ValueError(f"Layer {layer_idx}: only loaded {loaded}/{num_experts} for {key}")

    # Stack into [E, ...] arrays
    return {k: np.stack(v, axis=0) for k, v in result.items()}


# ============================================================================
# MoE layer processing
# ============================================================================


def process_moe_layer(expert_weights, tp_size):
    """Process one MoE layer's expert weights through the full pipeline.

    Returns dict of post-processing tensor name → numpy array.
    """
    # Fuse gate + up into w13: [E, 2*intermediate, hidden]
    w13 = jnp.array(np.concatenate([expert_weights["gate_weight"], expert_weights["up_weight"]], axis=1))
    s13 = jnp.array(np.concatenate([expert_weights["gate_scale"], expert_weights["up_scale"]], axis=1))
    w2 = jnp.array(expert_weights["down_weight"])
    s2 = jnp.array(expert_weights["down_scale"])

    # Step 1: Dequantize blockwise FP8 → float32
    w13_f32 = dequantize_tensor(w13, s13, (1, 2), jnp.float32, block_size=WEIGHT_BLOCK_SIZE)
    w2_f32 = dequantize_tensor(w2, s2, (1, 2), jnp.float32, block_size=WEIGHT_BLOCK_SIZE)

    # Step 2: Requantize to per-channel FP8
    fused = quantize_moe_weights(
        FusedMoEWeights(
            w13_weight=w13_f32,
            w13_weight_scale=None,
            w13_bias=None,
            w2_weight=w2_f32,
            w2_weight_scale=None,
            w2_bias=None,
        ),
        jnp.float8_e4m3fn,
        None,  # per-channel (no block size)
    )

    # Step 3: Reorder for GMM_TP kernel
    result = process_moe_weights(
        fused,
        moe_backend=MoEBackend.GMM_TP,
        w13_reorder_size=tp_size,
        w13_interleave=W13_INTERLEAVE,
    )

    return {
        "kernel_gating_upproj_EDF": np.asarray(result.w13_weight),
        "kernel_gating_upproj_EDF_weight_scale_inv": np.asarray(result.w13_weight_scale),
        "kernel_down_proj_EFD": np.asarray(result.w2_weight),
        "kernel_down_proj_EFD_weight_scale_inv": np.asarray(result.w2_weight_scale),
    }


# ============================================================================
# Dense linear processing
# ============================================================================


def load_dense_weight(shard_files, weight_name, weight_map=None):
    """Load a single dense weight + scale from safetensors shards."""
    import torch
    from safetensors import safe_open

    weight = None
    scale = None
    scale_key = weight_name.replace(".weight", ".weight_scale_inv")

    # Use index to find the right shard
    if weight_map:
        shard_dir = os.path.dirname(shard_files[0])
        shards_to_check = set()
        if weight_name in weight_map:
            shards_to_check.add(os.path.join(shard_dir, weight_map[weight_name]))
        if scale_key in weight_map:
            shards_to_check.add(os.path.join(shard_dir, weight_map[scale_key]))
        shard_list = sorted(shards_to_check)
    else:
        shard_list = shard_files

    for sf_path in shard_list:
        with safe_open(sf_path, framework="torch") as f:
            if weight_name in f.keys():
                t = f.get_tensor(weight_name)
                weight = t.cpu().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
            if scale_key in f.keys():
                t = f.get_tensor(scale_key)
                scale = t.cpu().float().numpy()

    return weight, scale


def process_dense_linear(weight, weight_scale):
    """Process one dense FP8 blockwise linear through the full pipeline.

    IMPORTANT: Weight must be in HF format (out_features, in_features) — NO transpose.
    The standard vLLM weight_loader uses permute_dims=(0,1) which is identity.
    output_sizes = (weight.shape[0],) and n_shards = 1 always (from QuantLinearConfig).
    """
    w = jnp.array(weight)
    s = jnp.array(weight_scale)

    result = process_blockwise_fp8_linear_weights(
        w,
        s,
        bias=None,
        weight_block_size=WEIGHT_BLOCK_SIZE,
        requant_block_size=None,
        output_sizes=(w.shape[0],),  # first dim = out_features in HF format
        requant_weight_dtype=jnp.float8_e4m3fn,
        fuse_matmuls=True,
        n_shards=1,  # always 1 from QuantLinearConfig
    )

    return {
        "weight": np.asarray(result.weight),
        "weight_scale_inv": np.asarray(result.weight_scale),
    }


# ============================================================================
# kv_b_proj special handling (MLA split)
# ============================================================================


def process_kv_b_proj(weight, weight_scale):
    """Replicate MLAEinsum.load_weights logic exactly: split kv_b_proj into k_up_proj + v_up_proj.

    IMPORTANT: Uses quantize_tensor from tpu_inference.kernels.quantized_matmul.util
    (the same one MLAEinsum uses), NOT from tpu_inference.layers.common.quantization.
    These have different APIs: kernels version uses (x, dtype, dim=), common version uses (dtype, tensor, axis=).

    Scale shapes are 3D with middle dim=1, matching what sharded_quantized_batched_matmul expects:
      k_scale: (N, 1, A) = (64, 1, 512)
      v_scale: (N, 1, V) = (64, 1, 128)
    """
    from tpu_inference.kernels.quantized_matmul.util import quantize_tensor as kernel_quantize

    w = jnp.array(weight)
    s = jnp.array(weight_scale)

    # Step 1: Dequantize (same as MLAEinsum line 525)
    dequantized = dequantize_tensor(w, s, (0, 1), block_size=None)
    dequantized = dequantized.T

    A = KV_LORA_RANK  # 512
    N = NUM_HEADS  # 64
    total_head_dim = QK_NOPE_HEAD_DIM + V_HEAD_DIM  # 256

    # Step 2: Reshape and split (same as MLAEinsum lines 536-541)
    dequantized = dequantized.reshape(A, N, total_head_dim)
    k_ANH, v_ANH = jnp.split(dequantized, [QK_NOPE_HEAD_DIM], axis=-1)

    # Step 3: Re-quantize using kernel's quantize_tensor (same as MLAEinsum lines 542-547)
    k_weight, k_scale = kernel_quantize(k_ANH, w.dtype, dim=-1)
    v_weight, v_scale = kernel_quantize(v_ANH, w.dtype, dim=0)

    # Step 4: Transpose scales (same as MLAEinsum lines 550-551)
    k_N1A_scale = k_scale.transpose(1, 2, 0)  # (A,N,1) → (N,1,A) = (64,1,512)
    v_N1H_scale = v_scale.transpose(1, 0, 2)  # (1,N,V) → (N,1,V) = (64,1,128)

    return {
        "k_up_proj.weight": np.asarray(k_weight),  # (512, 64, 128)
        "k_up_proj.weight_scale_inv": np.asarray(k_N1A_scale),  # (64, 1, 512)
        "v_up_proj.weight": np.asarray(v_weight),  # (512, 64, 128)
        "v_up_proj.weight_scale_inv": np.asarray(v_N1H_scale),  # (64, 1, 128)
    }


# ============================================================================
# Pass-through weight loading
# ============================================================================


def load_passthrough_weights(shard_files, key_names):
    """Load weights that don't need processing (norms, embeddings, router gates)."""
    import torch as _torch
    from safetensors import safe_open

    result = {}
    for sf_path in shard_files:
        with safe_open(sf_path, framework="torch") as f:
            for key in f.keys():
                if key in key_names:
                    t = f.get_tensor(key)
                    if t.dtype == _torch.bfloat16:
                        result[key] = t.cpu().float().numpy()  # BF16 → F32 for numpy compat
                    elif t.dtype == _torch.float32:
                        result[key] = t.cpu().numpy()
                    elif t.dtype == _torch.float8_e4m3fn:
                        result[key] = t.cpu().view(_torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
                    else:
                        result[key] = t.cpu().float().numpy()
    return result


# ============================================================================
# Layer processing (main per-layer function)
# ============================================================================


def get_attention_weight_names(layer_idx):
    """Get HF weight names for attention projections in a layer."""
    prefix = f"model.layers.{layer_idx}.self_attn."
    return [
        f"{prefix}q_a_proj.weight",
        f"{prefix}q_b_proj.weight",
        f"{prefix}kv_a_proj_with_mqa.weight",
        f"{prefix}kv_b_proj.weight",
        f"{prefix}o_proj.weight",
    ]


def get_shared_expert_weight_names(layer_idx):
    """Get HF weight names for shared experts in a MoE layer."""
    prefix = f"model.layers.{layer_idx}.mlp.shared_experts."
    return [
        f"{prefix}gate_proj.weight",
        f"{prefix}up_proj.weight",
        f"{prefix}down_proj.weight",
    ]


def get_dense_ffn_weight_names(layer_idx):
    """Get HF weight names for dense FFN layer (layer 0)."""
    prefix = f"model.layers.{layer_idx}.mlp."
    return [
        f"{prefix}gate_proj.weight",
        f"{prefix}up_proj.weight",
        f"{prefix}down_proj.weight",
    ]


def get_passthrough_names(layer_idx, is_moe):
    """Get weight names that pass through without processing."""
    prefix = f"model.layers.{layer_idx}."
    names = [
        f"{prefix}input_layernorm.weight",
        f"{prefix}post_attention_layernorm.weight",
        f"{prefix}self_attn.kv_a_layernorm.weight",
        f"{prefix}self_attn.q_a_layernorm.weight",
        f"{prefix}self_attn.rotary_emb.inv_freq",
    ]
    if is_moe:
        names.extend(
            [
                f"{prefix}mlp.gate.weight",
                f"{prefix}mlp.gate.e_score_correction_bias",
            ]
        )
    return names


def process_layer(args):
    """Process one layer completely. Designed for multiprocessing.Pool."""
    layer_idx, shard_files, output_dir, tp_size = args

    t0 = time.time()

    # Skip if already processed (makes script resumable)
    output_path = os.path.join(output_dir, f"model-layer-{layer_idx:04d}.safetensors")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[Layer {layer_idx}] Already exists ({os.path.getsize(output_path)/1e6:.0f}MB), skipping", flush=True)
        return layer_idx, 0.0

    is_moe = layer_idx >= FIRST_K_DENSE_REPLACE
    output_tensors = {}

    # Load index for fast shard lookup
    index_path = os.path.join(os.path.dirname(shard_files[0]), "model.safetensors.index.json")
    weight_map = None
    if os.path.exists(index_path):
        with open(index_path) as idx_f:
            weight_map = json.load(idx_f).get("weight_map", {})

    # 1. MoE expert weights (layers 1-60)
    if is_moe:
        print(f"[Layer {layer_idx}] Loading {NUM_EXPERTS} experts...", flush=True)
        t_load = time.time()
        expert_weights = load_expert_weights(shard_files, layer_idx)
        print(f"[Layer {layer_idx}] Loaded in {time.time()-t_load:.1f}s", flush=True)
        print(f"[Layer {layer_idx}] Processing MoE (dequant+requant+reorder)...", flush=True)
        t_proc = time.time()
        moe_result = process_moe_layer(expert_weights, tp_size)
        print(f"[Layer {layer_idx}] MoE done in {time.time()-t_proc:.1f}s", flush=True)
        prefix = f"model.layers.{layer_idx}.mlp.experts."
        for name, tensor in moe_result.items():
            output_tensors[f"{prefix}{name}"] = tensor
        del expert_weights, moe_result

    # 2. Attention projections (all layers)
    print(f"[Layer {layer_idx}] Processing attention projections...", flush=True)
    t_attn = time.time()
    attn_names = get_attention_weight_names(layer_idx)
    for wname in attn_names:
        weight, scale = load_dense_weight(shard_files, wname, weight_map)
        if weight is None:
            logger.warning(f"Layer {layer_idx}: Missing {wname}")
            continue

        if "kv_b_proj" in wname:
            # Special MLA handling
            kv_result = process_kv_b_proj(weight, scale)
            base = wname.replace("kv_b_proj.weight", "")
            for name, tensor in kv_result.items():
                output_tensors[f"{base}{name}"] = tensor
        else:
            # Standard dense linear processing
            # TODO: output_sizes and n_shards depend on the specific projection
            # For now, use simple single-shard processing
            result = process_dense_linear(weight, scale)
            output_tensors[wname] = result["weight"]
            output_tensors[wname.replace(".weight", ".weight_scale_inv")] = result["weight_scale_inv"]

    print(f"[Layer {layer_idx}] Attention done in {time.time()-t_attn:.1f}s", flush=True)

    # 3. Shared expert / dense FFN weights
    print(f"[Layer {layer_idx}] Processing FFN/shared experts...", flush=True)
    t_ffn = time.time()
    if is_moe:
        ffn_names = get_shared_expert_weight_names(layer_idx)
    else:
        ffn_names = get_dense_ffn_weight_names(layer_idx)

    for wname in ffn_names:
        weight, scale = load_dense_weight(shard_files, wname, weight_map)
        if weight is None:
            logger.warning(f"Layer {layer_idx}: Missing {wname}")
            continue
        result = process_dense_linear(weight, scale)
        output_tensors[wname] = result["weight"]
        output_tensors[wname.replace(".weight", ".weight_scale_inv")] = result["weight_scale_inv"]

    print(f"[Layer {layer_idx}] FFN done in {time.time()-t_ffn:.1f}s", flush=True)

    # 4. Pass-through weights (norms, router gates)
    passthrough_names = get_passthrough_names(layer_idx, is_moe)
    passthrough = load_passthrough_weights(shard_files, passthrough_names)
    output_tensors.update(passthrough)

    # 5. Save layer shard
    print(f"[Layer {layer_idx}] Saving {len(output_tensors)} tensors...", flush=True)
    t_save = time.time()
    save_layer(output_tensors, output_dir, layer_idx)
    print(f"[Layer {layer_idx}] Saved in {time.time()-t_save:.1f}s", flush=True)

    elapsed = time.time() - t0
    n_tensors = len(output_tensors)
    size_mb = sum(t.nbytes for t in output_tensors.values()) / 1e6
    logger.info(f"Layer {layer_idx}: Done in {elapsed:.1f}s " f"({n_tensors} tensors, {size_mb:.0f} MB)")
    return layer_idx, elapsed


# ============================================================================
# Saving
# ============================================================================


def save_layer(tensors, output_dir, layer_idx):
    """Save one layer's tensors to a safetensors file.

    Writes to a temp file first, then moves to output_dir.
    If output_dir is on GCS (via gcsfuse), the temp file is local to avoid
    filling up disk — we write, copy to GCS, then delete local.
    """
    from safetensors.numpy import save_file

    # Convert FP8/BF16 for safetensors compatibility
    save_dict = {}
    for key, arr in tensors.items():
        if hasattr(arr, "dtype") and arr.dtype == ml_dtypes.float8_e4m3fn:
            save_dict[key] = arr.view(np.uint8)
        elif hasattr(arr, "dtype") and arr.dtype == ml_dtypes.bfloat16:
            save_dict[key] = arr.astype(np.float32)
        elif isinstance(arr, np.ndarray):
            save_dict[key] = arr
        else:
            save_dict[key] = np.asarray(arr)

    filename = f"model-layer-{layer_idx:04d}.safetensors"
    final_path = os.path.join(output_dir, filename)

    # Write directly to output dir (no temp file — avoids disk space issues in Docker)
    save_file(save_dict, final_path, metadata={"fp8_dtype": "e4m3fn", "preprocessed": "true"})


def save_non_layer_weights(shard_files, output_dir):
    """Save non-layer weights (embeddings, final norm, lm_head)."""
    from safetensors.numpy import save_file

    non_layer_keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    tensors = load_passthrough_weights(shard_files, non_layer_keys)

    save_dict = {}
    for key, arr in tensors.items():
        if hasattr(arr, "dtype") and arr.dtype == ml_dtypes.float8_e4m3fn:
            save_dict[key] = arr.view(np.uint8)
        elif hasattr(arr, "dtype") and arr.dtype == ml_dtypes.bfloat16:
            save_dict[key] = arr.astype(np.float32)
        else:
            save_dict[key] = np.asarray(arr)

    path = os.path.join(output_dir, "model-non-layer.safetensors")
    save_file(save_dict, path, metadata={"preprocessed": "true"})
    logger.info(f"Saved non-layer weights: {list(tensors.keys())}")


def write_index(output_dir):
    """Write model.safetensors.index.json for the preprocessed checkpoint."""
    weight_map = {}
    total_size = 0

    for sf_path in sorted(glob.glob(os.path.join(output_dir, "*.safetensors"))):
        from safetensors import safe_open

        filename = os.path.basename(sf_path)
        with safe_open(sf_path, framework="numpy") as f:
            for key in f.keys():
                weight_map[key] = filename
                total_size += f.get_tensor(key).nbytes

    index = {
        "metadata": {
            "total_size": total_size,
            "preprocessed": True,
            "tp_size": 4,
            "moe_backend": "GMM_TP",
        },
        "weight_map": weight_map,
    }

    path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Index written: {len(weight_map)} keys, {total_size / 1e9:.1f} GB")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Preprocess K2-Instruct weights for TPU")
    parser.add_argument("--input-dir", required=True, help="Path to original safetensors")
    parser.add_argument("--output-dir", required=True, help="Path for preprocessed output")
    parser.add_argument("--tp-size", type=int, default=4, help="Tensor parallelism degree")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers per host")
    parser.add_argument("--layer-start", type=int, default=0, help="First layer to process")
    parser.add_argument("--layer-end", type=int, default=NUM_LAYERS, help="Last layer (exclusive)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    shard_files = sorted(glob.glob(os.path.join(args.input_dir, "*.safetensors")))
    logger.info(f"Input: {len(shard_files)} safetensors shards")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Layers: {args.layer_start}-{args.layer_end}, TP={args.tp_size}, " f"workers={args.workers}")

    # Copy config files — skip if output is on gcsfuse (we'll copy later via gcloud)
    # Only copy on layer_start=0 worker
    if args.layer_start == 0:
        for fname in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "tokenization_kimi.py",
            "tiktoken.model",
        ]:
            src = os.path.join(args.input_dir, fname)
            dst = os.path.join(args.output_dir, fname)
            if os.path.exists(src):
                try:
                    with open(src, "rb") as f_in:
                        data = f_in.read()
                    with open(dst, "wb") as f_out:
                        f_out.write(data)
                    logger.info(f"Copied {fname}")
                except Exception as e:
                    logger.warning(f"Could not copy {fname}: {e} (will copy later)")

    # Add preprocessing metadata to config
    if args.layer_start == 0:
        config_path = os.path.join(args.output_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                config["preprocessed_weights"] = {
                    "version": 1,
                    "tp_size": args.tp_size,
                    "moe_backend": "GMM_TP",
                    "w13_reorder_size": args.tp_size,
                    "w13_interleave": W13_INTERLEAVE,
                    "weight_block_size": list(WEIGHT_BLOCK_SIZE),
                }
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not patch config: {e}")

    # Process layers
    layer_range = range(args.layer_start, args.layer_end)
    layer_args = [(idx, shard_files, args.output_dir, args.tp_size) for idx in layer_range]

    t_total = time.time()
    if args.workers > 1:
        logger.info(f"Processing {len(layer_range)} layers with {args.workers} workers...")
        with Pool(processes=args.workers) as pool:
            results = pool.map(process_layer, layer_args)
    else:
        logger.info(f"Processing {len(layer_range)} layers sequentially...")
        results = [process_layer(a) for a in layer_args]

    # Save non-layer weights (only if we're processing from layer 0)
    if args.layer_start == 0:
        save_non_layer_weights(shard_files, args.output_dir)

    # Write index
    write_index(args.output_dir)

    elapsed = time.time() - t_total
    logger.info(f"COMPLETE: {len(results)} layers in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    for layer_idx, layer_time in results:
        logger.info(f"  Layer {layer_idx}: {layer_time:.1f}s")


if __name__ == "__main__":
    main()
