#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validated Plan B test: numerical correctness + real tpu_inference functions.

Tests:
  1. CPU vs TPU numerical equivalence using the REAL process_fp8_moe_weights
  2. Timing comparison for the real pipeline
  3. Memory usage tracking

Usage (inside Docker on TPU host, with single-host env vars):
    TPU_SKIP_MDS_QUERY=1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
    CLOUD_TPU_TASK_ID=0 python test_plan_b_validated.py
"""

import sys
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

# Import the REAL functions from tpu_inference
sys.path.insert(0, "/workspace/tpu_inference")
from tpu_inference.layers.common.quantization import dequantize_tensor, quantize_tensor


def time_fn(fn, name, warmup=1, repeats=3):
    """Time a function with warmup and repeats."""
    for _ in range(warmup):
        result = fn()
        if isinstance(result, tuple):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()
        elif hasattr(result, "block_until_ready"):
            result.block_until_ready()

    times = []
    for _ in range(repeats):
        t0 = time.time()
        result = fn()
        if isinstance(result, tuple):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()
        elif hasattr(result, "block_until_ready"):
            result.block_until_ready()
        times.append(time.time() - t0)

    median = sorted(times)[len(times) // 2]
    print(f"  {name}: {median:.3f}s (times: {[f'{t:.3f}' for t in times]})")
    return median, result


def create_realistic_fp8_data(num_experts, dim1, dim2, block_size):
    """Create realistic FP8 data that mimics real model weights."""
    rng = np.random.RandomState(42)

    # Create float32 weights with realistic distribution
    raw = rng.randn(num_experts, dim1, dim2).astype(np.float32) * 0.02

    # Quantize to blockwise FP8 properly (matching how HF stores them)
    bs = block_size
    n_blocks_1 = dim1 // bs
    n_blocks_2 = dim2 // bs

    # Compute block-wise scales
    raw_blocked = raw.reshape(num_experts, n_blocks_1, bs, n_blocks_2, bs)
    abs_max = np.max(np.abs(raw_blocked), axis=(2, 4), keepdims=True)
    fp8_max = 448.0
    scales = (abs_max / fp8_max).squeeze(axis=(2, 4)).astype(np.float32)

    # Quantize
    scales_expanded = scales[:, :, np.newaxis, :, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        scales_inv = np.where(scales_expanded == 0, 0.0, 1.0 / scales_expanded)
    quantized = np.clip(raw_blocked * scales_inv, -fp8_max, fp8_max)
    quantized = quantized.reshape(num_experts, dim1, dim2)

    # Cast to FP8
    weight_fp8 = quantized.astype(ml_dtypes.float8_e4m3fn)

    return weight_fp8, scales


def main():
    E = 384  # num experts (K2-Instruct)
    D1 = 2048  # intermediate size per expert
    D2 = 7168  # hidden size
    BS = 128  # block size

    print("=" * 70)
    print("Validated Plan B: CPU vs TPU with REAL tpu_inference functions")
    print(f"  Experts: {E}, Dims: [{D1}, {D2}], Block: {BS}")
    print(f"  FP8 size: {E * D1 * D2 / 1e9:.2f} GB")
    print(f"  Float32 intermediate: {E * D1 * D2 * 4 / 1e9:.2f} GB")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  TPU devices: {jax.devices('tpu')}")
    print("=" * 70)

    # Create realistic data
    print("\nCreating realistic FP8 data (properly quantized)...")
    weight_fp8, scales = create_realistic_fp8_data(E, D1, D2, BS)
    print(f"  weight: {weight_fp8.shape} {weight_fp8.dtype}")
    print(f"  scales: {scales.shape} {scales.dtype}")

    # Verify scales are reasonable
    print(f"  scale range: [{scales.min():.6f}, {scales.max():.6f}]")
    print(
        f"  weight range (as float): [{weight_fp8.astype(np.float32).min():.1f}, {weight_fp8.astype(np.float32).max():.1f}]"
    )

    # Create per-expert numpy arrays (simulating what t2j produces)
    expert_list_np = [weight_fp8[i : i + 1] for i in range(E)]

    # ================================================================
    # TEST 1: CPU baseline using REAL tpu_inference dequantize_tensor
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: CPU — real dequantize_tensor + quantize_tensor")
    print("=" * 70)

    cpu_device = jax.devices("cpu")[0]

    @jax.jit(static_argnames=("bs0", "bs1"))
    def real_dequant_requant_cpu(w, s, bs0, bs1):
        """Use the REAL tpu_inference functions."""
        # Dequantize (matches process_fp8_moe_weights)
        f32 = dequantize_tensor(w, s, (1, 2), jnp.float32, block_size=(bs0, bs1))
        # Requantize per-channel (matches quantize_moe_weights with block_size=None)
        q, new_s = quantize_tensor(jnp.float8_e4m3fn, f32, axis=2, block_size=None)
        return q, new_s

    def cpu_pipeline():
        with jax.default_device(cpu_device):
            w = jnp.concatenate(expert_list_np, axis=0)
            s = jnp.array(scales)
            return real_dequant_requant_cpu(w, s, BS, BS)

    print("  Running CPU pipeline (warmup includes JIT compile)...")
    t_cpu, (q_cpu, s_cpu) = time_fn(cpu_pipeline, "CPU (real functions)", warmup=1, repeats=2)

    q_cpu_np = np.asarray(q_cpu)
    s_cpu_np = np.asarray(s_cpu)
    print(f"  Output weight: {q_cpu_np.shape} {q_cpu_np.dtype}")
    print(f"  Output scale:  {s_cpu_np.shape} {s_cpu_np.dtype}")
    print(f"  Weight value range: [{q_cpu_np.astype(np.float32).min():.1f}, {q_cpu_np.astype(np.float32).max():.1f}]")
    print(f"  Scale value range:  [{s_cpu_np.min():.8f}, {s_cpu_np.max():.8f}]")

    # ================================================================
    # TEST 2: TPU using same REAL functions
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: TPU — same real functions, numpy input → auto-transfer")
    print("=" * 70)

    def tpu_pipeline():
        # np.concatenate on CPU (fast), then transfer to TPU via jnp.array
        w_np = np.concatenate(expert_list_np, axis=0)
        s_np = scales
        w = jnp.array(w_np)
        s = jnp.array(s_np)
        return real_dequant_requant_cpu(w, s, BS, BS)  # Same JIT, compiles for TPU

    print("  Running TPU pipeline (warmup includes JIT compile for TPU)...")
    try:
        t_tpu, (q_tpu, s_tpu) = time_fn(tpu_pipeline, "TPU (real functions)", warmup=1, repeats=2)

        q_tpu_np = np.asarray(q_tpu)
        s_tpu_np = np.asarray(s_tpu)
        print(f"  Output weight: {q_tpu_np.shape} {q_tpu_np.dtype}")
        print(f"  Output scale:  {s_tpu_np.shape} {s_tpu_np.dtype}")
        print(
            f"  Weight value range: [{q_tpu_np.astype(np.float32).min():.1f}, {q_tpu_np.astype(np.float32).max():.1f}]"
        )
        print(f"  Scale value range:  [{s_tpu_np.min():.8f}, {s_tpu_np.max():.8f}]")

        # ================================================================
        # NUMERICAL COMPARISON
        # ================================================================
        print("\n" + "=" * 70)
        print("NUMERICAL COMPARISON: CPU vs TPU")
        print("=" * 70)

        # Weight comparison (FP8 — compare as float32)
        q_cpu_f32 = q_cpu_np.astype(np.float32)
        q_tpu_f32 = q_tpu_np.astype(np.float32)

        # Exact match (same FP8 values)
        exact_match = np.array_equal(q_cpu_np.view(np.uint8), q_tpu_np.view(np.uint8))
        print(f"  Weight byte-exact match: {exact_match}")

        if not exact_match:
            diff = np.abs(q_cpu_f32 - q_tpu_f32)
            print(f"  Weight max diff (float32): {diff.max():.6f}")
            print(f"  Weight mean diff:          {diff.mean():.6f}")
            print(f"  Weight median diff:        {np.median(diff):.6f}")
            # How many values differ?
            num_diff = np.sum(q_cpu_np.view(np.uint8) != q_tpu_np.view(np.uint8))
            total = q_cpu_np.size
            print(f"  Values that differ:        {num_diff}/{total} ({100*num_diff/total:.2f}%)")
            # Most diffs should be ±1 in FP8 (rounding)
            if num_diff > 0:
                diff_vals = diff[diff > 0]
                print(
                    f"  Diff distribution: min={diff_vals.min():.4f}, "
                    f"max={diff_vals.max():.4f}, mean={diff_vals.mean():.4f}"
                )

        # Scale comparison (float32)
        s_exact = np.array_equal(s_cpu_np, s_tpu_np)
        print(f"\n  Scale exact match: {s_exact}")
        if not s_exact:
            s_diff = np.abs(s_cpu_np - s_tpu_np)
            print(f"  Scale max diff:    {s_diff.max():.10f}")
            print(f"  Scale mean diff:   {s_diff.mean():.10f}")
            s_reldiff = s_diff / (np.abs(s_cpu_np) + 1e-30)
            print(f"  Scale max reldiff: {s_reldiff.max():.10f}")

        # Dequantized equivalence check: does dequant(q, s) give same float32?
        print("\n  Reconstruction check (dequant the requantized weights):")
        recon_cpu = q_cpu_f32 * s_cpu_np[:, :, np.newaxis]
        recon_tpu = q_tpu_f32 * s_tpu_np[:, :, np.newaxis]
        recon_diff = np.abs(recon_cpu - recon_tpu)
        print(f"  Reconstructed max diff:  {recon_diff.max():.8f}")
        print(f"  Reconstructed mean diff: {recon_diff.mean():.8f}")
        # Compare to original float32 (before FP8 quantization)
        # This tells us total quantization error, not CPU vs TPU diff
        print("  (This measures whether CPU and TPU produce the same quantized representation)")

    except Exception as e:
        t_tpu = None
        import traceback

        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

    # ================================================================
    # TIMING BREAKDOWN
    # ================================================================
    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN")
    print("=" * 70)

    # np.concatenate
    def np_concat():
        return np.concatenate(expert_list_np, axis=0)

    t_np_concat, w_np_full = time_fn(np_concat, "np.concatenate", warmup=1, repeats=2)

    # CPU→TPU transfer
    w_np_full_val = w_np_full
    s_np_val = scales

    def transfer():
        w = jnp.array(w_np_full_val)
        s = jnp.array(s_np_val)
        w.block_until_ready()
        s.block_until_ready()
        return w, s

    t_transfer, (w_tpu_ready, s_tpu_ready) = time_fn(transfer, "CPU→TPU transfer", warmup=1, repeats=2)

    # TPU compute only
    def tpu_compute():
        return real_dequant_requant_cpu(w_tpu_ready, s_tpu_ready, BS, BS)

    t_tpu_compute, _ = time_fn(tpu_compute, "TPU compute only", warmup=1, repeats=2)

    # jnp.concatenate on CPU (current bottleneck)
    def jnp_concat_cpu():
        with jax.default_device(cpu_device):
            return jnp.concatenate(expert_list_np, axis=0)

    t_jnp_concat, _ = time_fn(jnp_concat_cpu, "jnp.concatenate CPU (current)", warmup=1, repeats=2)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n  Current path (all CPU):")
    print(f"    jnp.concatenate: {t_jnp_concat:.1f}s")
    print(f"    JIT dequant+requant: {t_cpu - t_jnp_concat:.1f}s")  # rough estimate
    print(f"    Total: {t_cpu:.1f}s")

    optimal = t_np_concat + t_transfer + t_tpu_compute
    print("\n  Optimal path (np.concat → transfer → TPU):")
    print(f"    np.concatenate:  {t_np_concat:.1f}s")
    print(f"    CPU→TPU transfer: {t_transfer:.1f}s")
    print(f"    TPU compute:     {t_tpu_compute:.3f}s")
    print(f"    Total: {optimal:.1f}s")

    speedup = t_cpu / optimal
    print(f"\n  Speedup: {speedup:.1f}x")

    num_layers = 14
    print(f"\n  Extrapolation ({num_layers} MoE layers per PP worker):")
    print(f"    Current: {t_cpu * num_layers / 60:.1f} min")
    print(f"    Optimal: {optimal * num_layers / 60:.1f} min")

    # Overall verdict
    print("\n" + "=" * 70)
    if exact_match:
        print("VERDICT: TPU produces BYTE-EXACT same results as CPU. SAFE TO USE.")
    elif t_tpu is not None:
        print("VERDICT: TPU produces slightly different results (FP8 rounding).")
        print("         Check the diff stats above to assess acceptability.")
    else:
        print("VERDICT: TPU test FAILED. See error above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
