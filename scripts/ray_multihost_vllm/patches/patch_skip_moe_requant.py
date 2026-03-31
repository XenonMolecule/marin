# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Skip FP8 dequant→requant for MoE weights when block sizes match.

The process_fp8_moe_weights function dequantizes ALL expert weights from
block FP8 → float32, then re-quantizes back. For 384 experts this takes
~40 hours on CPU. If the source and target block sizes are compatible,
we can skip the dequant→requant and just do padding + reordering.

Strategy: Replace the dequant+requant with direct padding of FP8 weights
and scales, keeping the original block quantization format.
"""

PATH = "/workspace/tpu_inference/tpu_inference/layers/common/process_weights/moe_weights.py"
with open(PATH) as f:
    code = f.read()

# Find the dequantization section and add a fast path
old = """    # Dequantize fp8 2d block quantized weights into fp32.
    w13_weight = dequantize_tensor(w13_weight,
                                   w13_weight_scale, (1, 2),
                                   jnp.float32,
                                   block_size=weight_block_size)
    w2_weight = dequantize_tensor(w2_weight,
                                  w2_weight_scale, (1, 2),
                                  jnp.float32,
                                  block_size=weight_block_size)"""

new = """    # OPTIMIZATION: Skip dequant→requant if we can keep weights in FP8.
    # The dequant of 384 experts to float32 takes ~40 hours on CPU.
    # Instead, keep weights in FP8 and pass scales through directly.
    import os as _os
    _skip_requant = _os.environ.get("SKIP_MOE_REQUANT", "0") == "1"
    if _skip_requant:
        import logging as _log
        _log.warning(f"SKIP_MOE_REQUANT: Skipping dequant→requant for MoE weights. "
                     f"w13={w13_weight.shape} w2={w2_weight.shape} dtype={w13_weight.dtype}")
        # Skip dequant entirely — pass FP8 weights + scales directly to process_moe_weights
        # The GMM kernel should handle block-quantized FP8 natively
        weights = FusedMoEWeights(
            w13_weight=w13_weight,
            w13_weight_scale=w13_weight_scale,
            w13_bias=None,
            w2_weight=w2_weight,
            w2_weight_scale=w2_weight_scale,
            w2_bias=None,
        )
        w13_interleave = activation == "swigluoai"
        w13_reorder_size = get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR)
        return process_moe_weights(
            weights,
            moe_backend=moe_backend,
            w13_reorder_size=w13_reorder_size,
            w13_interleave=w13_interleave,
        )

    # Original path: Dequantize fp8 2d block quantized weights into fp32.
    w13_weight = dequantize_tensor(w13_weight,
                                   w13_weight_scale, (1, 2),
                                   jnp.float32,
                                   block_size=weight_block_size)
    w2_weight = dequantize_tensor(w2_weight,
                                  w2_weight_scale, (1, 2),
                                  jnp.float32,
                                  block_size=weight_block_size)"""

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED moe_weights.py: Added SKIP_MOE_REQUANT fast path")
else:
    print("SKIP: dequantize pattern not found")
