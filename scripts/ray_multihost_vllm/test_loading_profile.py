#!/usr/bin/env python3
"""Profile each phase of FP8 weight loading to find the real bottleneck.

Tests each component in isolation with synthetic data (no caching effects):
  1. t2j conversion (torch FP8 → JAX/numpy)
  2. jnp.concatenate (gathering experts)
  3. dequantize_tensor (blockwise FP8 → float32)
  4. quantize_tensor (float32 → per-channel FP8)
  5. Full pipeline: concat → dequant → requant

For each, measures both JAX CPU and numpy paths where applicable.

Usage (inside Docker, JAX_PLATFORMS=cpu for isolation):
    JAX_PLATFORMS=cpu python test_loading_profile.py [--num-experts 384]
"""

import argparse
import itertools
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch


def time_fn(fn, name, warmup=1, repeats=3):
    """Time a function with warmup and repeats. Returns median time."""
    for _ in range(warmup):
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
            result[0].block_until_ready()

    times = []
    for _ in range(repeats):
        t0 = time.time()
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple) and hasattr(result[0], 'block_until_ready'):
            result[0].block_until_ready()
        times.append(time.time() - t0)

    median = sorted(times)[len(times) // 2]
    print(f"  {name}: {median:.3f}s (times: {[f'{t:.3f}' for t in times]})")
    return median, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--dim1", type=int, default=2048,
                        help="MoE intermediate size per expert")
    parser.add_argument("--dim2", type=int, default=7168,
                        help="Hidden size")
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    E, D1, D2, BS = args.num_experts, args.dim1, args.dim2, args.block_size
    print("=" * 70)
    print(f"FP8 Weight Loading Profile")
    print(f"  Experts: {E}, Dims: [{D1}, {D2}], Block: {BS}")
    print(f"  FP8 size: {E * D1 * D2 / 1e9:.2f} GB")
    print(f"  Float32 size: {E * D1 * D2 * 4 / 1e9:.2f} GB")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  JAX devices: {jax.devices()}")
    print("=" * 70)

    # ================================================================
    # Phase 1: t2j conversion — torch FP8 tensor → JAX/numpy
    # ================================================================
    print("\n--- Phase 1: t2j (torch FP8 → JAX array) ---")
    print(f"  Single expert: [{D1}, {D2}] FP8 = {D1*D2/1e6:.1f} MB")

    # Create a torch FP8 tensor (simulating what safetensors gives us)
    torch_weight = torch.randint(0, 255, (D1, D2), dtype=torch.uint8).view(torch.float8_e4m3fn)

    # Method A: Current code path (jnp.array(bytes).view(fp8))
    def t2j_jax():
        bytes_np = torch_weight.cpu().view(torch.uint8).detach().numpy()
        return jnp.array(bytes_np).view(jnp.float8_e4m3fn)

    # Method B: Our numpy patch (np.view(ml_dtypes))
    def t2j_numpy():
        bytes_np = torch_weight.cpu().view(torch.uint8).detach().numpy()
        return bytes_np.view(ml_dtypes.float8_e4m3fn).reshape(torch_weight.shape)

    # Method C: torchax fallback (for non-contiguous tensors)
    def t2j_torch_contiguous():
        # Simulate permuted (non-contiguous) tensor
        t = torch_weight.T  # Now non-contiguous
        t = t.contiguous()  # Make contiguous first
        bytes_np = t.view(torch.uint8).detach().numpy()
        return bytes_np.view(ml_dtypes.float8_e4m3fn).reshape(t.shape)

    t_jax, r_jax = time_fn(t2j_jax, "JAX path (jnp.array.view)")
    t_np, r_np = time_fn(t2j_numpy, "NumPy path (np.view)")
    t_contig, _ = time_fn(t2j_torch_contiguous, "NumPy+contiguous (for permuted)")

    # Validate
    jax_f32 = np.asarray(r_jax).view(np.uint8)
    np_f32 = np.asarray(r_np).view(np.uint8)
    print(f"  Byte-exact match: {np.array_equal(jax_f32, np_f32)}")
    print(f"  Speedup (numpy vs jax): {t_jax/t_np:.1f}x")

    # Scale to 384 experts
    print(f"\n  Extrapolated to {E} experts:")
    print(f"    JAX path:   {t_jax * E:.1f}s")
    print(f"    NumPy path: {t_np * E:.1f}s")

    # ================================================================
    # Phase 2: jnp.concatenate — gathering experts
    # ================================================================
    print(f"\n--- Phase 2: jnp.concatenate ({E} experts) ---")

    # Create list of expert arrays (as numpy, simulating our patch)
    expert_list_np = [np.random.randint(0, 255, (1, D1, D2), dtype=np.uint8).view(
        ml_dtypes.float8_e4m3fn) for _ in range(E)]

    # Create list as JAX arrays (simulating original code)
    expert_list_jax = [jnp.array(e) for e in expert_list_np]

    def concat_from_numpy():
        return jnp.concatenate(expert_list_np, axis=0)

    def concat_from_jax():
        return jnp.concatenate(expert_list_jax, axis=0)

    t_concat_np, _ = time_fn(concat_from_numpy, "concat(numpy arrays)", warmup=1, repeats=2)
    t_concat_jax, _ = time_fn(concat_from_jax, "concat(jax arrays)", warmup=1, repeats=2)
    print(f"  Overhead of numpy→jax in concat: {t_concat_np - t_concat_jax:.3f}s")

    # ================================================================
    # Phase 3: dequantize_tensor — blockwise FP8 → float32
    # ================================================================
    print(f"\n--- Phase 3: dequantize ({E} experts, blockwise [{BS},{BS}] → float32) ---")

    # Create realistic FP8 weight + block scales
    weight_fp8 = np.random.randint(0, 255, (E, D1, D2), dtype=np.uint8).view(
        ml_dtypes.float8_e4m3fn)
    scale = np.abs(np.random.randn(E, D1 // BS, D2 // BS).astype(np.float32)) * 0.01 + 0.001

    weight_jax = jnp.array(weight_fp8)
    scale_jax = jnp.array(scale)

    @jax.jit(static_argnames=('bs',))
    def dequant_jit(w, s, bs):
        orig = w.shape
        aligned = w.shape  # assuming already aligned to block_size
        n, h, k = aligned
        w = w.reshape(n, h // bs, bs, k // bs, bs)
        s_exp = s[:, :, jnp.newaxis, :, jnp.newaxis]
        result = (w.astype(jnp.float32) * s_exp)
        return result.reshape(aligned)

    t_dequant, r_dequant = time_fn(
        lambda: dequant_jit(weight_jax, scale_jax, BS),
        "JAX JIT dequant", warmup=1, repeats=2)

    # ================================================================
    # Phase 4: quantize_tensor — float32 → per-channel FP8
    # ================================================================
    print(f"\n--- Phase 4: quantize ({E} experts, float32 → per-channel FP8) ---")

    # Use the dequantized result as input
    float32_weight = r_dequant

    FP8_MAX = 448.0
    FP8_MIN = -448.0

    @jax.jit
    def quant_jit(tensor):
        abs_max = jnp.max(jnp.abs(tensor), axis=2, keepdims=True)
        scale = abs_max / FP8_MAX
        scale_inv = jnp.nan_to_num(1 / scale, jnp.inf)
        tensor_q = jnp.clip(tensor * scale_inv, FP8_MIN, FP8_MAX)
        tensor_q = tensor_q.astype(jnp.float8_e4m3fn)
        scale = jnp.squeeze(scale, 2).astype(jnp.float32)
        return tensor_q, scale

    t_quant, _ = time_fn(
        lambda: quant_jit(float32_weight),
        "JAX JIT quantize", warmup=1, repeats=2)

    # ================================================================
    # Phase 5: Full pipeline — concat → dequant → requant
    # ================================================================
    print(f"\n--- Phase 5: Full pipeline (concat + dequant + requant) ---")

    @jax.jit(static_argnames=('bs',))
    def full_pipeline_jit(w, s, bs):
        # Dequant
        n, h, k = w.shape
        w_blocked = w.reshape(n, h // bs, bs, k // bs, bs)
        s_exp = s[:, :, jnp.newaxis, :, jnp.newaxis]
        f32 = (w_blocked.astype(jnp.float32) * s_exp).reshape(n, h, k)
        # Requant per-channel
        abs_max = jnp.max(jnp.abs(f32), axis=2, keepdims=True)
        new_scale = abs_max / FP8_MAX
        scale_inv = jnp.nan_to_num(1 / new_scale, jnp.inf)
        q = jnp.clip(f32 * scale_inv, FP8_MIN, FP8_MAX).astype(jnp.float8_e4m3fn)
        new_scale = jnp.squeeze(new_scale, 2).astype(jnp.float32)
        return q, new_scale

    def full_pipeline_from_numpy_experts():
        # Simulates: numpy expert arrays → concat → dequant → requant
        w = jnp.concatenate(expert_list_np, axis=0)
        s = jnp.array(scale)
        return full_pipeline_jit(w, s, BS)

    def full_pipeline_from_jax_experts():
        # Simulates: jax expert arrays → concat → dequant → requant
        w = jnp.concatenate(expert_list_jax, axis=0)
        s = scale_jax
        return full_pipeline_jit(w, s, BS)

    t_full_np, _ = time_fn(full_pipeline_from_numpy_experts,
                           "Full (numpy experts → concat → dequant → requant)",
                           warmup=1, repeats=2)
    t_full_jax, _ = time_fn(full_pipeline_from_jax_experts,
                            "Full (jax experts → concat → dequant → requant)",
                            warmup=1, repeats=2)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY — Time per MoE layer")
    print("=" * 70)
    print(f"  Phase 1 (t2j × {E} experts):")
    print(f"    JAX path:    {t_jax * E:.1f}s")
    print(f"    NumPy path:  {t_np * E:.1f}s")
    print(f"  Phase 2 (concatenate {E} experts):")
    print(f"    From numpy:  {t_concat_np:.1f}s")
    print(f"    From jax:    {t_concat_jax:.1f}s")
    print(f"  Phase 3 (dequant):     {t_dequant:.1f}s")
    print(f"  Phase 4 (requant):     {t_quant:.1f}s")
    print(f"  Phase 5 (full pipeline):")
    print(f"    NumPy experts: {t_full_np:.1f}s")
    print(f"    JAX experts:   {t_full_jax:.1f}s")

    total_np = t_np * E + t_full_np
    total_jax = t_jax * E + t_full_jax
    print(f"\n  TOTAL per MoE layer (t2j + pipeline):")
    print(f"    With numpy t2j patch: {total_np:.1f}s")
    print(f"    Without (original):   {total_jax:.1f}s")
    print(f"    Speedup: {total_jax / total_np:.1f}x")

    num_moe_layers = 14  # per PP worker
    print(f"\n  Extrapolation ({num_moe_layers} MoE layers per PP worker):")
    print(f"    With numpy t2j:  {total_np * num_moe_layers / 60:.1f} min")
    print(f"    Without:         {total_jax * num_moe_layers / 60:.1f} min")


if __name__ == "__main__":
    main()
