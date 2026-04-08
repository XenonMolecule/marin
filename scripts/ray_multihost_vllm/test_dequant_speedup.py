#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone test: compare JAX CPU vs NumPy vs TPU for FP8 dequant→requant.

Loads real K2-Instruct expert weights from safetensors, runs the dequant→requant
through three paths, validates numerical equivalence, and reports timing.

Usage (inside Docker container on a TPU host):
    python test_dequant_speedup.py [--num-experts 8] [--safetensors-path /mnt/gcs-models]

No vLLM, no Ray, no model initialization needed.
"""

import argparse
import itertools
import math
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

# ============================================================================
# Baseline: JAX CPU implementations (copied from tpu_inference source)
# ============================================================================


def dequantize_tensor_jax(tensor_q, scale, axis, out_dtype, block_size=None):
    """Exact copy of tpu_inference dequantize_tensor."""
    if axis is None:
        axis = [i for i in range(tensor_q.ndim)]
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor_q.shape
    if block_size is not None:
        pad_width = [[0, 0] for _ in range(tensor_q.ndim)]
        for ax, bs in zip(axis, block_size):
            pad_width[ax][1] = scale.shape[ax] * bs - tensor_q.shape[ax]
        tensor_q = jnp.pad(tensor_q, pad_width)

    aligned_shape = tensor_q.shape
    if tensor_q.ndim == scale.ndim:
        blocked_shape = [[i] for i in aligned_shape]
        for i in axis:
            num_blocks = scale.shape[i]
            calc_block_size = tensor_q.shape[i] // num_blocks
            blocked_shape[i] = (num_blocks, calc_block_size)
        axis = sorted([(i + tensor_q.ndim) % tensor_q.ndim for i in axis])
        axis = [1 + n + i for n, i in enumerate(axis)]
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor_q = tensor_q.reshape(blocked_shape)

    scale = jnp.expand_dims(scale, axis)
    tensor = (tensor_q.astype(jnp.float32) * scale).astype(out_dtype)
    tensor = tensor.reshape(aligned_shape)
    return jax.lax.slice(tensor, [0] * tensor.ndim, list(orig_shape))


def quantize_tensor_jax(dtype, tensor, axis=-1, block_size=None):
    """Exact copy of tpu_inference quantize_tensor."""
    if axis is None:
        axis = [i for i in range(tensor.ndim)]
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor.shape
    if block_size is not None:
        if isinstance(block_size, int):
            block_size = [block_size] * len(axis)
        blocked_shape = [[i] for i in orig_shape]
        for i, block in zip(axis, block_size):
            num_blocks = tensor.shape[i] // block
            blocked_shape[i] = (num_blocks, block)
        axis = sorted([i % tensor.ndim for i in axis])
        axis = [1 + n + i for n, i in enumerate(axis)]
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor = tensor.reshape(blocked_shape)

    dtype_info = jnp.finfo(dtype)
    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    abs_max = jnp.max(jnp.abs(tensor), axis=axis, keepdims=True)
    scale = abs_max / dtype_max
    scale_inv = jnp.nan_to_num(1 / scale, jnp.inf)
    tensor_q = jnp.clip(tensor * scale_inv, dtype_min, dtype_max)
    tensor_q = tensor_q.reshape(orig_shape)
    tensor_q = tensor_q.astype(dtype)
    scale = jnp.squeeze(scale, axis).astype(jnp.float32)
    return tensor_q, scale


@jax.jit(static_argnames=("block_size_0", "block_size_1"))
def baseline_dequant_requant_jax(weight_fp8, weight_scale, block_size_0, block_size_1):
    """Baseline: dequant blockwise FP8 → float32 → requant per-channel FP8.
    Mirrors what process_fp8_moe_weights does for one expert chunk."""
    # Dequantize
    f32 = dequantize_tensor_jax(weight_fp8, weight_scale, (1, 2), jnp.float32, block_size=(block_size_0, block_size_1))
    # Requant per-channel (block_size = contracting dim = last axis)
    q, s = quantize_tensor_jax(jnp.float8_e4m3fn, f32, axis=2, block_size=None)
    return q, s


# ============================================================================
# Plan A: Pure NumPy implementation
# ============================================================================


def dequantize_tensor_np(tensor_q_f32, scale, axis, block_size=None):
    """NumPy implementation of dequantize_tensor. Input is already float32."""
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor_q_f32.shape
    if block_size is not None:
        pad_width = [(0, 0)] * tensor_q_f32.ndim
        for ax, bs in zip(axis, block_size):
            pad_width[ax] = (0, scale.shape[ax] * bs - tensor_q_f32.shape[ax])
        tensor_q_f32 = np.pad(tensor_q_f32, pad_width)

    aligned_shape = tensor_q_f32.shape
    if tensor_q_f32.ndim == scale.ndim:
        blocked_shape = [[i] for i in aligned_shape]
        for i in axis:
            num_blocks = scale.shape[i]
            calc_block_size = tensor_q_f32.shape[i] // num_blocks
            blocked_shape[i] = (num_blocks, calc_block_size)
        axis_sorted = sorted([(i + tensor_q_f32.ndim) % tensor_q_f32.ndim for i in axis])
        expand_axes = [1 + n + i for n, i in enumerate(axis_sorted)]
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor_q_f32 = tensor_q_f32.reshape(blocked_shape)
    else:
        expand_axes = axis

    scale = np.expand_dims(scale, axis=expand_axes)
    result = tensor_q_f32 * scale
    result = result.reshape(aligned_shape)
    slices = tuple(slice(0, s) for s in orig_shape)
    return result[slices]


def quantize_tensor_np(tensor, axis=-1, block_size=None):
    """NumPy implementation of quantize_tensor. Returns (float32 quantized, scale).
    Final cast to FP8 must be done separately."""
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor.shape
    FP8_MAX = 448.0  # float8_e4m3fn max
    FP8_MIN = -448.0

    if block_size is not None:
        if isinstance(block_size, int):
            block_size = [block_size] * len(axis)
        blocked_shape = [[i] for i in orig_shape]
        for i, block in zip(axis, block_size):
            num_blocks = tensor.shape[i] // block
            blocked_shape[i] = (num_blocks, block)
        axis = sorted([i % tensor.ndim for i in axis])
        axis = [1 + n + i for n, i in enumerate(axis)]
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor = tensor.reshape(blocked_shape)

    abs_max = np.max(np.abs(tensor), axis=tuple(axis), keepdims=True)
    scale = abs_max / FP8_MAX

    with np.errstate(divide="ignore", invalid="ignore"):
        scale_inv = np.where(scale == 0, np.inf, 1.0 / scale)

    tensor_q = np.clip(tensor * scale_inv, FP8_MIN, FP8_MAX)
    tensor_q = tensor_q.reshape(orig_shape).astype(np.float32)
    scale = np.squeeze(scale, axis=tuple(axis)).astype(np.float32)
    return tensor_q, scale


def plan_a_numpy(weight_fp8_np, weight_scale_np, block_size):
    """Plan A: Full dequant→requant in numpy."""
    # Cast FP8 to float32 in numpy (view as uint8 → cast isn't direct,
    # but we can go through jnp for this one step)
    weight_f32 = np.asarray(jnp.asarray(weight_fp8_np).astype(jnp.float32))
    scale_f32 = np.asarray(weight_scale_np).astype(np.float32)

    # Dequantize
    deq = dequantize_tensor_np(weight_f32, scale_f32, (1, 2), block_size=block_size)

    # Requant per-channel
    q_f32, s = quantize_tensor_np(deq, axis=[2], block_size=None)

    # Final FP8 cast (only JAX step)
    q_fp8 = np.asarray(jnp.asarray(q_f32).astype(jnp.float8_e4m3fn))
    return q_fp8, s


# ============================================================================
# Plan B: TPU offload
# ============================================================================


def plan_b_tpu(weight_fp8_np, weight_scale_np, block_size):
    """Plan B: Run JAX dequant→requant on TPU instead of CPU.
    Pass numpy arrays — JAX auto-transfers to TPU."""
    # Convert to jnp arrays (will go to default device = TPU)
    weight = jnp.array(weight_fp8_np)
    scale = jnp.array(weight_scale_np)

    # Run the same JIT function but WITHOUT cpu_mesh — compiles for TPU
    q, s = baseline_dequant_requant_jax(weight, scale, block_size[0], block_size[1])

    # Block until done
    q.block_until_ready()
    s.block_until_ready()
    return np.asarray(q), np.asarray(s)


# ============================================================================
# Test harness
# ============================================================================


def load_expert_weights(safetensors_path, num_experts=8):
    """Load a few expert weights from the first safetensors shard."""
    from safetensors import safe_open
    import glob
    import os

    shard_files = sorted(glob.glob(os.path.join(safetensors_path, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors in {safetensors_path}")

    # Find expert weight keys in the first shard that has MoE weights
    gate_weights = []
    gate_scales = []
    for shard in shard_files:
        with safe_open(shard, framework="numpy") as f:
            keys = list(f.keys())
            expert_keys = [k for k in keys if "experts" in k and "gate_proj.weight" in k]
            if not expert_keys:
                continue

            print(f"Found {len(expert_keys)} expert keys in {os.path.basename(shard)}")
            for k in sorted(expert_keys)[:num_experts]:
                w = f.get_tensor(k)
                gate_weights.append(w)
                # Find corresponding scale
                scale_key = k.replace(".weight", ".weight_scale_inv")
                if scale_key in keys:
                    s = f.get_tensor(scale_key)
                    gate_scales.append(s)
                else:
                    print(f"  WARNING: no scale for {k}")
            break

    if not gate_weights:
        raise ValueError("No expert weights found")

    print(f"Loaded {len(gate_weights)} experts, shape={gate_weights[0].shape}, " f"dtype={gate_weights[0].dtype}")
    if gate_scales:
        print(f"Scales shape={gate_scales[0].shape}, dtype={gate_scales[0].dtype}")

    # Stack into [num_experts, dim1, dim2]
    weights = np.stack(gate_weights, axis=0)
    scales = np.stack(gate_scales, axis=0) if gate_scales else None
    return weights, scales


def create_synthetic_weights(num_experts=8, dim1=2048, dim2=7168, block_size=128):
    """Create synthetic FP8 weights + block scales for testing."""
    print(f"Creating synthetic weights: {num_experts} experts, [{dim1}, {dim2}], " f"block_size={block_size}")

    # Random float32, then quantize to simulate real FP8 data
    rng = np.random.RandomState(42)
    raw = rng.randn(num_experts, dim1, dim2).astype(np.float32) * 0.1

    # Quantize to FP8 with block scales [128, 128]
    scale_shape = (num_experts, math.ceil(dim1 / block_size), math.ceil(dim2 / block_size))
    scales = np.abs(rng.randn(*scale_shape).astype(np.float32)) * 0.01 + 0.001

    # Simulate FP8 quantized values (just clip to FP8 range)
    fp8_max = 448.0
    weight_f32 = np.clip(
        raw
        / np.expand_dims(scales, axis=(2, 4))
        .repeat(block_size, axis=2)
        .repeat(block_size, axis=4)
        .reshape(num_experts, scale_shape[1] * block_size, scale_shape[2] * block_size)[:, :dim1, :dim2],
        -fp8_max,
        fp8_max,
    )

    # Cast to FP8 via ml_dtypes
    weight_fp8 = weight_f32.astype(ml_dtypes.float8_e4m3fn)

    print(f"  weight: {weight_fp8.shape} {weight_fp8.dtype}")
    print(f"  scale:  {scales.shape} {scales.dtype}")
    return weight_fp8, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts to test with")
    parser.add_argument(
        "--safetensors-path", type=str, default=None, help="Path to model safetensors (uses synthetic if not set)"
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dim1", type=int, default=2048, help="Expert weight dim1 (for synthetic)")
    parser.add_argument("--dim2", type=int, default=7168, help="Expert weight dim2 (for synthetic)")
    parser.add_argument("--skip-tpu", action="store_true", help="Skip Plan B (TPU) test")
    args = parser.parse_args()

    block_size = (args.block_size, args.block_size)

    print("=" * 60)
    print("FP8 Dequant→Requant Speed Test")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print()

    # Load or create weights
    if args.safetensors_path:
        weight_fp8, scales = load_expert_weights(args.safetensors_path, args.num_experts)
    else:
        weight_fp8, scales = create_synthetic_weights(args.num_experts, args.dim1, args.dim2, args.block_size)

    print(f"\nTest input: {weight_fp8.shape} ({weight_fp8.nbytes / 1e6:.1f} MB FP8)")
    print(f"Block size: {block_size}")
    print()

    # ---- Baseline: JAX CPU ----
    print("=" * 60)
    print("BASELINE: JAX CPU (current code path)")
    print("=" * 60)
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        w_jax = jnp.array(weight_fp8)
        s_jax = jnp.array(scales)

        # Warmup (JIT compilation)
        print("  Compiling (first run)...")
        t0 = time.time()
        q_base, s_base = baseline_dequant_requant_jax(w_jax, s_jax, block_size[0], block_size[1])
        q_base.block_until_ready()
        compile_time = time.time() - t0
        print(f"  Compile + first run: {compile_time:.2f}s")

        # Timed run
        t0 = time.time()
        q_base, s_base = baseline_dequant_requant_jax(w_jax, s_jax, block_size[0], block_size[1])
        q_base.block_until_ready()
        baseline_time = time.time() - t0
        print(f"  Cached run:          {baseline_time:.2f}s")

    q_base_np = np.asarray(q_base)
    s_base_np = np.asarray(s_base)
    print(f"  Output: weight {q_base_np.shape} {q_base_np.dtype}, " f"scale {s_base_np.shape} {s_base_np.dtype}")

    # ---- Plan A: NumPy ----
    print()
    print("=" * 60)
    print("PLAN A: Pure NumPy")
    print("=" * 60)

    # Warmup
    t0 = time.time()
    q_a, s_a = plan_a_numpy(weight_fp8, scales, block_size)
    plan_a_time_first = time.time() - t0
    print(f"  First run: {plan_a_time_first:.2f}s")

    # Timed
    t0 = time.time()
    q_a, s_a = plan_a_numpy(weight_fp8, scales, block_size)
    plan_a_time = time.time() - t0
    print(f"  Second run: {plan_a_time:.2f}s")

    # Numerical comparison
    q_a_f32 = q_a.astype(np.float32)
    q_base_f32 = q_base_np.astype(np.float32)
    weight_max_diff = np.max(np.abs(q_a_f32 - q_base_f32))
    scale_max_diff = np.max(np.abs(s_a - s_base_np))
    weight_match = np.allclose(q_a_f32, q_base_f32, atol=1.0)  # FP8 has low precision
    scale_match = np.allclose(s_a, s_base_np, rtol=1e-5)
    print(f"  Weight max diff: {weight_max_diff:.4f} (match={weight_match})")
    print(f"  Scale max diff:  {scale_max_diff:.8f} (match={scale_match})")
    print(f"  Speedup vs baseline: {baseline_time / plan_a_time:.1f}x")

    # ---- Plan B: TPU ----
    if not args.skip_tpu and jax.default_backend() == "tpu":
        print()
        print("=" * 60)
        print("PLAN B: TPU offload")
        print("=" * 60)

        try:
            # Warmup (JIT compile for TPU)
            print("  Compiling for TPU (first run)...")
            t0 = time.time()
            q_b, s_b = plan_b_tpu(weight_fp8, scales, block_size)
            tpu_compile_time = time.time() - t0
            print(f"  Compile + first run: {tpu_compile_time:.2f}s")

            # Timed
            t0 = time.time()
            q_b, s_b = plan_b_tpu(weight_fp8, scales, block_size)
            plan_b_time = time.time() - t0
            print(f"  Cached run:          {plan_b_time:.2f}s")

            # Numerical comparison
            q_b_f32 = q_b.astype(np.float32)
            weight_max_diff_b = np.max(np.abs(q_b_f32 - q_base_f32))
            scale_max_diff_b = np.max(np.abs(s_b - s_base_np))
            weight_match_b = np.allclose(q_b_f32, q_base_f32, atol=1.0)
            scale_match_b = np.allclose(s_b, s_base_np, rtol=1e-5)
            print(f"  Weight max diff: {weight_max_diff_b:.4f} (match={weight_match_b})")
            print(f"  Scale max diff:  {scale_max_diff_b:.8f} (match={scale_match_b})")
            print(f"  Speedup vs baseline: {baseline_time / plan_b_time:.1f}x")
        except Exception as e:
            print(f"  FAILED: {e}")
    else:
        print("\nSkipping Plan B (no TPU or --skip-tpu)")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Experts tested:   {weight_fp8.shape[0]}")
    print(f"  Data size:        {weight_fp8.nbytes / 1e6:.1f} MB (FP8)")
    print(f"  Baseline (JAX CPU compile): {compile_time:.2f}s")
    print(f"  Baseline (JAX CPU cached):  {baseline_time:.2f}s")
    print(f"  Plan A (NumPy):             {plan_a_time:.2f}s  ({baseline_time/plan_a_time:.1f}x)")
    if not args.skip_tpu and jax.default_backend() == "tpu":
        try:
            print(f"  Plan B (TPU compile):        {tpu_compile_time:.2f}s")
            print(f"  Plan B (TPU cached):         {plan_b_time:.2f}s  ({baseline_time/plan_b_time:.1f}x)")
        except NameError:
            print("  Plan B: FAILED")
    print()
    print("Extrapolation to full K2-Instruct (384 experts, ~14 MoE layers per PP worker):")
    scale_factor = (384 / weight_fp8.shape[0]) * 14
    print(f"  Baseline: ~{baseline_time * scale_factor / 60:.0f} min")
    print(f"  Plan A:   ~{plan_a_time * scale_factor / 60:.0f} min")


if __name__ == "__main__":
    main()
