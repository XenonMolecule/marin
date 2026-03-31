#!/usr/bin/env python3
"""Test Plan B: Run FP8 dequant→requant on TPU instead of CPU.

Compares CPU vs TPU for the full MoE weight processing pipeline:
  1. Concatenate 384 expert arrays
  2. Dequantize blockwise FP8 → float32
  3. Requantize float32 → per-channel FP8

Must run with TPU access (inside privileged Docker container).

Usage:
    python test_plan_b_tpu.py [--num-experts 384]
"""

import argparse
import itertools
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np


def time_fn(fn, name, warmup=1, repeats=3):
    """Time a function with warmup and repeats."""
    for _ in range(warmup):
        result = fn()
        if isinstance(result, tuple):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        elif hasattr(result, 'block_until_ready'):
            result.block_until_ready()

    times = []
    for _ in range(repeats):
        t0 = time.time()
        result = fn()
        if isinstance(result, tuple):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        elif hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        times.append(time.time() - t0)

    median = sorted(times)[len(times) // 2]
    print(f"  {name}: {median:.3f}s (times: {[f'{t:.3f}' for t in times]})")
    return median, result


FP8_MAX = 448.0
FP8_MIN = -448.0


@jax.jit(static_argnames=('bs',))
def full_pipeline_jit(w, s, bs):
    """Dequant blockwise FP8 → float32 → requant per-channel FP8."""
    n, h, k = w.shape
    # Dequant
    w_blocked = w.reshape(n, h // bs, bs, k // bs, bs)
    s_exp = s[:, :, jnp.newaxis, :, jnp.newaxis]
    f32 = (w_blocked.astype(jnp.float32) * s_exp).reshape(n, h, k)
    # Requant per-channel (axis=2, no block)
    abs_max = jnp.max(jnp.abs(f32), axis=2, keepdims=True)
    new_scale = abs_max / FP8_MAX
    scale_inv = jnp.nan_to_num(1 / new_scale, jnp.inf)
    q = jnp.clip(f32 * scale_inv, FP8_MIN, FP8_MAX).astype(jnp.float8_e4m3fn)
    new_scale = jnp.squeeze(new_scale, 2).astype(jnp.float32)
    return q, new_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--dim1", type=int, default=2048)
    parser.add_argument("--dim2", type=int, default=7168)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    E, D1, D2, BS = args.num_experts, args.dim1, args.dim2, args.block_size

    print("=" * 70)
    print("Plan B Test: CPU vs TPU for MoE Weight Processing")
    print(f"  Experts: {E}, Dims: [{D1}, {D2}], Block: {BS}")
    print(f"  FP8 size: {E * D1 * D2 / 1e9:.2f} GB")
    print(f"  Float32 intermediate: {E * D1 * D2 * 4 / 1e9:.2f} GB")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  CPU devices: {jax.devices('cpu')}")
    try:
        tpu_devices = jax.devices('tpu')
        print(f"  TPU devices: {tpu_devices}")
        has_tpu = True
    except RuntimeError:
        print("  TPU devices: NONE")
        has_tpu = False
    print("=" * 70)

    # Create synthetic expert weight list (numpy arrays, like our t2j patch produces)
    print("\nCreating synthetic data...")
    expert_list_np = []
    for i in range(E):
        w = np.random.randint(0, 255, (1, D1, D2), dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
        expert_list_np.append(w)
    scale = np.abs(np.random.randn(E, D1 // BS, D2 // BS).astype(np.float32)) * 0.01 + 0.001
    print(f"  Created {E} expert arrays, each {D1}x{D2} FP8")

    # ================================================================
    # Test 1: CPU — full pipeline (concat + dequant + requant)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: CPU (current code path)")
    print("=" * 70)

    cpu_device = jax.devices("cpu")[0]

    def cpu_pipeline():
        with jax.default_device(cpu_device):
            # Concat numpy arrays → JAX on CPU
            w = jnp.concatenate(expert_list_np, axis=0)
            s = jnp.array(scale)
            # Run JIT on CPU
            return full_pipeline_jit(w, s, BS)

    t_cpu, (q_cpu, s_cpu) = time_fn(cpu_pipeline, "CPU full pipeline", warmup=1, repeats=2)
    q_cpu_np = np.asarray(q_cpu)
    s_cpu_np = np.asarray(s_cpu)
    print(f"  Output: weight {q_cpu_np.shape} {q_cpu_np.dtype}, scale {s_cpu_np.shape}")

    # ================================================================
    # Test 2: TPU — numpy inputs, no cpu_mesh_context
    # ================================================================
    if has_tpu:
        print("\n" + "=" * 70)
        print("TEST 2: TPU (numpy inputs → auto-transfer)")
        print("=" * 70)

        def tpu_pipeline():
            # Concat numpy arrays → JAX (default device = TPU)
            w = jnp.concatenate(expert_list_np, axis=0)
            s = jnp.array(scale)
            return full_pipeline_jit(w, s, BS)

        try:
            t_tpu, (q_tpu, s_tpu) = time_fn(tpu_pipeline, "TPU full pipeline", warmup=1, repeats=2)
            q_tpu_np = np.asarray(q_tpu)
            s_tpu_np = np.asarray(s_tpu)

            # Numerical comparison
            w_diff = np.max(np.abs(q_tpu_np.astype(np.float32) - q_cpu_np.astype(np.float32)))
            s_diff = np.max(np.abs(s_tpu_np - s_cpu_np))
            print(f"  Output: weight {q_tpu_np.shape} {q_tpu_np.dtype}, scale {s_tpu_np.shape}")
            print(f"  vs CPU weight max diff: {w_diff:.4f}")
            print(f"  vs CPU scale max diff:  {s_diff:.8f}")
            print(f"  Numerical match (weight): {np.allclose(q_tpu_np.astype(np.float32), q_cpu_np.astype(np.float32), atol=1.0)}")
            print(f"  Numerical match (scale):  {np.allclose(s_tpu_np, s_cpu_np, rtol=1e-5)}")
        except Exception as e:
            t_tpu = None
            print(f"  FAILED: {type(e).__name__}: {e}")

        # ================================================================
        # Test 3: TPU — pre-concatenated on CPU, transfer, then process
        # ================================================================
        print("\n" + "=" * 70)
        print("TEST 3: TPU (pre-concat on CPU, transfer bulk, process on TPU)")
        print("=" * 70)

        def tpu_preconcat_pipeline():
            # Concat on CPU as numpy (fast, no JAX)
            w_np = np.concatenate(expert_list_np, axis=0)
            s_np = scale
            # Transfer to TPU
            w = jnp.array(w_np)
            s = jnp.array(s_np)
            # Process on TPU
            return full_pipeline_jit(w, s, BS)

        try:
            t_tpu_preconcat, _ = time_fn(tpu_preconcat_pipeline,
                                         "TPU (np.concat + transfer + process)",
                                         warmup=1, repeats=2)
        except Exception as e:
            t_tpu_preconcat = None
            print(f"  FAILED: {type(e).__name__}: {e}")

        # ================================================================
        # Test 4: Breakdown — measure transfer vs compute separately
        # ================================================================
        print("\n" + "=" * 70)
        print("TEST 4: Breakdown (transfer vs compute)")
        print("=" * 70)

        try:
            # Pre-concat as numpy
            w_np_full = np.concatenate(expert_list_np, axis=0)

            # Measure numpy concat alone
            def np_concat_only():
                return np.concatenate(expert_list_np, axis=0)
            t_np_concat, _ = time_fn(np_concat_only, "np.concatenate only", warmup=1, repeats=2)

            # Measure CPU→TPU transfer alone
            def transfer_only():
                w = jnp.array(w_np_full)
                s = jnp.array(scale)
                w.block_until_ready()
                s.block_until_ready()
                return w, s
            t_transfer, (w_tpu, s_tpu) = time_fn(transfer_only, "CPU→TPU transfer only", warmup=1, repeats=2)

            # Measure TPU compute alone (data already on TPU)
            def compute_only():
                return full_pipeline_jit(w_tpu, s_tpu, BS)
            t_compute, _ = time_fn(compute_only, "TPU compute only (dequant+requant)", warmup=1, repeats=2)

            # Measure jnp.concatenate on CPU (the current bottleneck)
            def jnp_concat_cpu():
                with jax.default_device(cpu_device):
                    return jnp.concatenate(expert_list_np, axis=0)
            t_jnp_concat, _ = time_fn(jnp_concat_cpu, "jnp.concatenate on CPU (current bottleneck)", warmup=1, repeats=2)

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            t_np_concat = t_transfer = t_compute = t_jnp_concat = None

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Per MoE layer ({E} experts):")
    print(f"    CPU full pipeline:     {t_cpu:.1f}s")
    if has_tpu and t_tpu:
        print(f"    TPU full pipeline:     {t_tpu:.1f}s  ({t_cpu/t_tpu:.1f}x speedup)")
    if has_tpu and t_tpu_preconcat:
        print(f"    TPU (preconcat):       {t_tpu_preconcat:.1f}s  ({t_cpu/t_tpu_preconcat:.1f}x speedup)")

    if has_tpu and t_np_concat and t_transfer and t_compute and t_jnp_concat:
        print(f"\n  Breakdown:")
        print(f"    jnp.concatenate (CPU): {t_jnp_concat:.1f}s  ← CURRENT BOTTLENECK")
        print(f"    np.concatenate:        {t_np_concat:.1f}s")
        print(f"    CPU→TPU transfer:      {t_transfer:.1f}s")
        print(f"    TPU compute:           {t_compute:.1f}s")
        optimal = t_np_concat + t_transfer + t_compute
        print(f"    Optimal total:         {optimal:.1f}s  ({t_cpu/optimal:.1f}x vs current)")

    num_layers = 14
    print(f"\n  Extrapolation ({num_layers} MoE layers per PP worker):")
    print(f"    CPU:     {t_cpu * num_layers / 60:.1f} min")
    if has_tpu and t_tpu_preconcat:
        print(f"    TPU:     {t_tpu_preconcat * num_layers / 60:.1f} min")
    if has_tpu and t_np_concat and t_transfer and t_compute:
        optimal = t_np_concat + t_transfer + t_compute
        print(f"    Optimal: {optimal * num_layers / 60:.1f} min")


if __name__ == "__main__":
    main()
