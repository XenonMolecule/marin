# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move FP8 MoE weight processing from CPU to TPU.

Root cause of 40+ hour loading time: process_weights_after_loading runs
dequant→requant inside cpu_mesh_context(), forcing JAX JIT to compile and
execute on CPU. For 384-expert MoE layers, this is catastrophically slow.

Fix: When PROCESS_WEIGHTS_ON_TPU=1, keep concatenation on CPU (fast array
gathering) but run process_fp8_moe_weights under the default TPU mesh.
JAX automatically transfers CPU arrays to TPU at the JIT call boundary.

Memory safety: With PP=4 and TP=4, FP32 intermediate per chip is ~17GB
(well within 95GB HBM per v5p chip). Processing happens one layer at a
time, so peak memory is bounded.

Also patches linear layer processing (attention projections, shared expert)
for the same speedup.
"""

FP8_PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/quantization/fp8.py"

with open(FP8_PATH) as f:
    code = f.read()

patched = False

# =============================================================================
# Patch 1: MoE weight processing — move process_fp8_moe_weights to TPU
# =============================================================================
# Strategy: Split the single cpu_mesh_context() block into:
#   1. Concatenation under cpu_mesh_context (stays on CPU)
#   2. process_fp8_moe_weights under TPU mesh (when PROCESS_WEIGHTS_ON_TPU=1)

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

new_moe = """            # Step 1: Concatenate expert weights on CPU (fast array gathering)
            with cpu_mesh_context():
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
                w13_weight = jnp.concatenate([w_gate, w_up], axis=1)
                w13_weight_scale = jnp.concatenate([s_gate, s_up], axis=1)
                input_weights = FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=w13_weight_scale,
                    w13_bias=None,
                    w2_weight=w2_weight,
                    w2_weight_scale=w2_weight_scale,
                    w2_bias=None)

            # Step 2: Process (dequant→requant→reorder) — on TPU or CPU
            import os as _os
            _use_tpu = _os.environ.get("PROCESS_WEIGHTS_ON_TPU", "0") == "1"
            if _use_tpu:
                import time as _time
                _t0 = _time.time()
                import logging as _log
                _log.warning(f"PROCESS_ON_TPU: Starting MoE dequant+requant on TPU "
                             f"(w13={input_weights.w13_weight.shape}, w2={input_weights.w2_weight.shape})")
                # Convert CPU JAX arrays to numpy — JAX JIT accepts numpy inputs
                # and auto-places them on TPU (the default device under the TPU mesh).
                # This avoids the "incompatible devices" error from passing CPU JAX arrays.
                import numpy as _np
                input_weights = FusedMoEWeights(
                    w13_weight=_np.asarray(input_weights.w13_weight),
                    w13_weight_scale=_np.asarray(input_weights.w13_weight_scale),
                    w13_bias=None,
                    w2_weight=_np.asarray(input_weights.w2_weight),
                    w2_weight_scale=_np.asarray(input_weights.w2_weight_scale),
                    w2_bias=None)
                _log.warning(f"PROCESS_ON_TPU: Converted to numpy in {_time.time()-_t0:.1f}s, calling JIT...")
                weights = process_fp8_moe_weights(
                    input_weights,
                    moe_backend=layer.moe_backend,
                    mesh=layer.mesh,
                    activation=layer.activation,
                    weight_block_size=None,
                )
                _log.warning(f"PROCESS_ON_TPU: MoE layer done in {_time.time()-_t0:.1f}s")
            else:
                with cpu_mesh_context():
                    weights = process_fp8_moe_weights(
                        input_weights,
                        moe_backend=layer.moe_backend,
                        mesh=layer.mesh,
                        activation=layer.activation,
                        weight_block_size=None,
                    )"""

if old_moe in code:
    code = code.replace(old_moe, new_moe)
    patched = True
    print("PATCHED: MoE process_weights_after_loading — TPU processing path added")
else:
    print("SKIP: MoE pattern not found (may already be patched or code changed)")

# =============================================================================
# Patch 2: Linear layer processing — move process_blockwise_fp8_linear_weights to TPU
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
                n_shards=self.linear_config.n_shards)"""

new_linear = """        # Process on TPU (if requested) or CPU.
        # CPU avoids device OOM but is ~1000x slower for large models.
        import os as _os_lin
        _use_tpu_lin = _os_lin.environ.get("PROCESS_WEIGHTS_ON_TPU", "0") == "1"
        if _use_tpu_lin:
            with cpu_mesh_context():
                weight = layer.weight[...]
                weight_scale_inv = layer.weight_scale_inv[...]
                bias = layer.bias[...] if getattr(layer, 'bias',
                                                  None) is not None else None
                if bias is not None:
                    bias = bias.reshape(-1)
            # Convert CPU JAX arrays to numpy — JIT auto-places on TPU
            import numpy as _np_lin
            weight = _np_lin.asarray(weight)
            weight_scale_inv = _np_lin.asarray(weight_scale_inv)
            if bias is not None:
                bias = _np_lin.asarray(bias)
            # Run requant on TPU (default mesh)
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
        else:
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
                    n_shards=self.linear_config.n_shards)"""

if old_linear in code:
    code = code.replace(old_linear, new_linear)
    patched = True
    print("PATCHED: Linear process_weights_after_loading — TPU processing path added")
else:
    print("SKIP: Linear pattern not found (may already be patched or code changed)")

if patched:
    with open(FP8_PATH, "w") as f:
        f.write(code)
    print("\nPATCH COMPLETE. Set PROCESS_WEIGHTS_ON_TPU=1 to enable TPU processing.")
    print("Expected speedup: ~1000x for MoE layers (seconds vs 30+ min each)")
else:
    print("\nNO CHANGES MADE — patterns not found in source.")
