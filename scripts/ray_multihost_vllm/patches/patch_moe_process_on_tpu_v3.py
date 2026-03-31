# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move FP8 MoE weight processing to TPU with chunked expert processing.

v3: Processes experts in chunks to avoid TPU HBM OOM.
The float32 intermediate for 384 experts (~22.5GB) exceeds single-chip HBM.
Processing in chunks of 96 experts keeps peak at ~5.6GB float32.

Strategy:
  1. Concatenate expert weights on CPU (fast)
  2. Transfer to TPU via numpy → jnp.array
  3. Dequant+requant in chunks of CHUNK_SIZE experts on TPU
  4. Concatenate FP8 chunk results on TPU
  5. Run process_moe_weights (reordering) on the full FP8 tensor
"""

import os

FP8_PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/quantization/fp8.py"

with open(FP8_PATH) as f:
    code = f.read()

patched = False

# Need to add a chunked helper function at module level
# Find a good insertion point — after the imports section
HELPER_CODE = '''
# --- Chunked TPU processing helper (added by patch_moe_process_on_tpu_v3) ---
def _chunked_dequant_requant_on_tpu(input_weights, moe_backend, mesh, activation, weight_block_size, chunk_size=96):
    """Process MoE weights on TPU in chunks to avoid HBM OOM.

    Instead of calling process_fp8_moe_weights (which dequants all 384 experts
    to float32 at once = 22.5GB OOM), we:
    1. Dequant+requant in chunks of chunk_size experts
    2. Concatenate FP8 results
    3. Run process_moe_weights for GMM kernel reordering
    """
    import logging as _chunked_log
    import time as _chunked_time

    from tpu_inference.layers.common.quantization import dequantize_tensor, quantize_tensor
    from tpu_inference.layers.common.process_weights.moe_weights import (
        quantize_moe_weights, process_moe_weights, FusedMoEWeights,
        get_mesh_shape_product, ShardingAxisName, align_to,
    )
    from tpu_inference import envs
    from tpu_inference.utils import to_jax_dtype

    w13 = input_weights.w13_weight
    w13_s = input_weights.w13_weight_scale
    w2 = input_weights.w2_weight
    w2_s = input_weights.w2_weight_scale
    E = w13.shape[0]

    # Determine requant dtype and block size (matching process_fp8_moe_weights logic)
    if desired_quant_dtype_from_env := envs.MOE_REQUANTIZE_WEIGHT_DTYPE:
        desired_quant_dtype = to_jax_dtype(desired_quant_dtype_from_env)
    else:
        desired_quant_dtype = w13.dtype
    requant_block_size = None
    if requant_block_size_from_env := envs.MOE_REQUANTIZE_BLOCK_SIZE:
        requant_block_size = int(requant_block_size_from_env) if requant_block_size_from_env else None

    _chunked_log.warning(f"PROCESS_ON_TPU: Chunked processing {E} experts in chunks of {chunk_size}")
    _t0 = _chunked_time.time()

    # JIT function for one chunk — dequant + requant only (no reordering)
    @jax.jit(static_argnames=('bs', 'requant_bs'))
    def _dequant_requant_chunk(w13_chunk, w13_s_chunk, w2_chunk, w2_s_chunk, bs, requant_bs):
        # Dequant blockwise FP8 → float32
        w13_f32 = dequantize_tensor(w13_chunk, w13_s_chunk, (1, 2), jnp.float32, block_size=bs)
        w2_f32 = dequantize_tensor(w2_chunk, w2_s_chunk, (1, 2), jnp.float32, block_size=bs)

        # Determine per-channel block sizes (matching quantize_moe_weights)
        if requant_bs is None:
            w13_bs = w13_f32.shape[-1]
            w2_bs = w2_f32.shape[-1]
        else:
            w13_bs = w2_bs = requant_bs

        # Pad dimensions to align to block size
        _, orig_h, orig_i = w2_f32.shape
        h_aligned = orig_h + (w13_bs - orig_h % w13_bs) % w13_bs
        i_aligned = orig_i + (w2_bs - orig_i % w2_bs) % w2_bs

        w13_f32 = jnp.pad(w13_f32, [[0,0], [0, 2*(i_aligned - orig_i)], [0, h_aligned - orig_h]])
        w2_f32 = jnp.pad(w2_f32, [[0,0], [0, h_aligned - orig_h], [0, i_aligned - orig_i]])

        # Requant
        w13_q, w13_ns = quantize_tensor(desired_quant_dtype, w13_f32, 2, w13_bs)
        w2_q, w2_ns = quantize_tensor(desired_quant_dtype, w2_f32, 2, w2_bs)

        return w13_q, w13_ns, w2_q, w2_ns

    # Process in chunks
    w13_qs, w13_ss, w2_qs, w2_ss = [], [], [], []
    bs_tuple = weight_block_size if weight_block_size is not None else None

    for i in range(0, E, chunk_size):
        end = min(i + chunk_size, E)
        c_w13_q, c_w13_s, c_w2_q, c_w2_s = _dequant_requant_chunk(
            w13[i:end], w13_s[i:end], w2[i:end], w2_s[i:end],
            bs_tuple, requant_block_size)
        # Move results back to CPU numpy to free TPU HBM (prevents accumulation OOM)
        import numpy as _np_chunk
        w13_qs.append(_np_chunk.asarray(c_w13_q))
        w13_ss.append(_np_chunk.asarray(c_w13_s))
        w2_qs.append(_np_chunk.asarray(c_w2_q))
        w2_ss.append(_np_chunk.asarray(c_w2_s))
        del c_w13_q, c_w13_s, c_w2_q, c_w2_s
        _chunked_log.warning(f"PROCESS_ON_TPU:   Chunk {i}-{end} done in {_chunked_time.time()-_t0:.1f}s")

    # Concatenate FP8 results on TPU
    w13_weight = jnp.concatenate(w13_qs, axis=0)
    w13_weight_scale = jnp.concatenate(w13_ss, axis=0)
    w2_weight = jnp.concatenate(w2_qs, axis=0)
    w2_weight_scale = jnp.concatenate(w2_ss, axis=0)

    # Reorder for GMM kernel (operates on FP8 data — fits in memory)
    w13_interleave = activation == "swigluoai"
    w13_reorder_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)

    weights = process_moe_weights(
        FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=w13_weight_scale,
            w13_bias=None,
            w2_weight=w2_weight,
            w2_weight_scale=w2_weight_scale,
            w2_bias=None,
        ),
        moe_backend=moe_backend,
        w13_reorder_size=w13_reorder_size,
        w13_interleave=w13_interleave,
    )
    _chunked_log.warning(f"PROCESS_ON_TPU: Full MoE layer done in {_chunked_time.time()-_t0:.1f}s")
    return weights
# --- End chunked TPU processing helper ---
'''

# Insert the helper function BEFORE the Fp8FusedMoEMethod class (not inside it!)
INSERT_MARKER = "class Fp8FusedMoEMethod"
if INSERT_MARKER in code:
    idx = code.index(INSERT_MARKER)
    code = code[:idx] + HELPER_CODE + "\n" + code[idx:]
    print("INSERTED: _chunked_dequant_requant_on_tpu helper (before class)")
else:
    print("SKIP: Could not find Fp8FusedMoEMethod class")

# =============================================================================
# Patch 1: MoE weight processing — use chunked TPU path
# =============================================================================

old_moe = """            with cpu_mesh_context():
                w_gate = jnp.concatenate(
                    layer.kernel_gating_EDF._weights_to_load, axis=0)
                w_up = jnp.concatenate(
                    layer.kernel_up_proj_EDF._weights_to_load, axis=0)
                s_gate = jnp.concatenate(getattr(
                    layer, gating_scale_name)._weights_to_load,
                                         axis=0)
                s_up = jnp.concatenate(getattr(layer,
                                               up_scale_name)._weights_to_load,
                                       axis=0)
                w2_weight = jnp.concatenate(
                    layer.kernel_down_proj_EFD._weights_to_load, axis=0)
                w2_weight_scale = jnp.concatenate(getattr(
                    layer, down_scale_name)._weights_to_load,
                                                  axis=0)

                # Fuse the weights into w13: [Gate, Up]. w2 is expected to be
                # (num_experts, hidden_size, intermediate_size), w13 is expected to
                # be (num_experts, 2 * intermediate_size, hidden_size,)
                w13_weight = jnp.concatenate([w_gate, w_up], axis=1)
                w13_weight_scale = jnp.concatenate([s_gate, s_up], axis=1)

                # TODO (jacobplatin): we should support bias
                input_weights = FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=w13_weight_scale,
                    w13_bias=None,
                    w2_weight=w2_weight,
                    w2_weight_scale=w2_weight_scale,
                    w2_bias=None)

                weights = process_fp8_moe_weights(
                    input_weights,
                    moe_backend=layer.moe_backend,
                    mesh=layer.mesh,
                    activation=layer.activation,
                    # Source block size should be inferred from scale shape
                    weight_block_size=None,
                )"""

new_moe = """            # Step 1: Concatenate expert weights using np.concatenate (2.8x faster
            # than jnp.concatenate which triggers slow _convert_element_type per array)
            import numpy as _np_moe
            _to_np = lambda x: _np_moe.asarray(x) if not isinstance(x, _np_moe.ndarray) else x
            w_gate = _np_moe.concatenate(
                [_to_np(w) for w in layer.kernel_gating_EDF._weights_to_load], axis=0)
            w_up = _np_moe.concatenate(
                [_to_np(w) for w in layer.kernel_up_proj_EDF._weights_to_load], axis=0)
            s_gate = _np_moe.concatenate(
                [_to_np(w) for w in getattr(layer, gating_scale_name)._weights_to_load], axis=0)
            s_up = _np_moe.concatenate(
                [_to_np(w) for w in getattr(layer, up_scale_name)._weights_to_load], axis=0)
            w2_weight = _np_moe.concatenate(
                [_to_np(w) for w in layer.kernel_down_proj_EFD._weights_to_load], axis=0)
            w2_weight_scale = _np_moe.concatenate(
                [_to_np(w) for w in getattr(layer, down_scale_name)._weights_to_load], axis=0)
            w13_weight = _np_moe.concatenate([w_gate, w_up], axis=1)
            w13_weight_scale = _np_moe.concatenate([s_gate, s_up], axis=1)

            # Step 2: Transfer to TPU and process in chunks (avoids HBM OOM)
            input_weights = FusedMoEWeights(
                w13_weight=jnp.array(w13_weight),
                w13_weight_scale=jnp.array(w13_weight_scale),
                w13_bias=None,
                w2_weight=jnp.array(w2_weight),
                w2_weight_scale=jnp.array(w2_weight_scale),
                w2_bias=None)

            weights = _chunked_dequant_requant_on_tpu(
                input_weights,
                moe_backend=layer.moe_backend,
                mesh=layer.mesh,
                activation=layer.activation,
                weight_block_size=None,
                chunk_size=24,
            )"""

if old_moe in code:
    code = code.replace(old_moe, new_moe)
    patched = True
    print("PATCHED: MoE — chunked TPU processing (96 experts/chunk)")
else:
    print("SKIP: MoE pattern not found")

# =============================================================================
# Patch 2: Linear layer processing — same TPU approach (no chunking needed,
# individual linear layers are small enough)
# =============================================================================

old_linear = """        # Do the re-quant process on CPU to avoid OOM on device.
        with cpu_mesh_context():
            weight = layer.weight[...]
            weight_scale_inv = layer.weight_scale_inv[...]
            bias = layer.bias[...] if getattr(layer, 'bias',
                                              None) is not None else None
            if bias is not None:
                bias = bias.reshape(-1)
            weights = common_fp8.process_blockwise_fp8_linear_weights(
                weight,
                weight_scale_inv,
                bias=bias,
                weight_block_size=tuple(self.quant_config.weight_block_size),
                requant_block_size=self.linear_config.requant_block_size,
                output_sizes=tuple(self.linear_config.output_sizes),
                requant_weight_dtype=self.linear_config.requant_weight_dtype,
                fuse_matmuls=self.linear_config.fuse_matmuls,
                n_shards=self.linear_config.n_shards)
            delattr(layer, 'weight')
            delattr(layer, 'weight_scale_inv')
            delattr(layer, 'bias')

            if self.linear_config.enable_quantized_matmul_kernel:
                # The quantized_matmul_kernel expects weight scales shaped (n_out_features, 1, n_blocks) for blockwisze quantization.
                weights.weight_scale = jnp.expand_dims(
                    jnp.transpose(weights.weight_scale),
                    axis=1,
                )"""

new_linear = """        # Load weights on CPU, then process on TPU
        import numpy as _np_lin
        with cpu_mesh_context():
            weight = layer.weight[...]
            weight_scale_inv = layer.weight_scale_inv[...]
            bias = layer.bias[...] if getattr(layer, 'bias',
                                              None) is not None else None
            if bias is not None:
                bias = bias.reshape(-1)
        # Transfer to TPU via numpy conversion
        weight = jnp.array(_np_lin.asarray(weight))
        weight_scale_inv = jnp.array(_np_lin.asarray(weight_scale_inv))
        if bias is not None:
            bias = jnp.array(_np_lin.asarray(bias))
        weights = common_fp8.process_blockwise_fp8_linear_weights(
            weight,
            weight_scale_inv,
            bias=bias,
            weight_block_size=tuple(self.quant_config.weight_block_size),
            requant_block_size=self.linear_config.requant_block_size,
            output_sizes=tuple(self.linear_config.output_sizes),
            requant_weight_dtype=self.linear_config.requant_weight_dtype,
            fuse_matmuls=self.linear_config.fuse_matmuls,
            n_shards=self.linear_config.n_shards)
        delattr(layer, 'weight')
        delattr(layer, 'weight_scale_inv')
        delattr(layer, 'bias')

        if self.linear_config.enable_quantized_matmul_kernel:
            # The quantized_matmul_kernel expects weight scales shaped (n_out_features, 1, n_blocks) for blockwisze quantization.
            weights.weight_scale = jnp.expand_dims(
                jnp.transpose(weights.weight_scale),
                axis=1,
            )"""

if old_linear in code:
    code = code.replace(old_linear, new_linear)
    patched = True
    print("PATCHED: Linear — TPU processing")
else:
    print("SKIP: Linear pattern not found")

if patched:
    with open(FP8_PATH, "w") as f:
        f.write(code)
    print("\nPATCH COMPLETE v3. Chunked TPU processing for MoE (96 experts/chunk).")
else:
    print("\nNO CHANGES MADE.")
