# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add INT4 group-quantized MoE support to TPU inference.

The tpu_inference compressed-tensors MoE path only supports FP8 and unquantized.
This patch adds W4A16 (INT4 weights, BF16 activations) for group-quantized MoE
layers, enabling models like unsloth/Kimi-K2.5 (INT4 via compressed-tensors QAT).

Strategy: Keep weights in packed INT4 format in HBM (~4x memory savings over BF16).
Dequantize to BF16 inside @jax.jit during forward pass. XLA may fuse the
dequant+matmul to avoid materializing the full BF16 weight tensor.

Weight format (from vLLM GPU CompressedTensorsWNA16MoEMethod):
  w13_weight_packed: int32 [num_experts, hidden//pack_factor, 2*inter]
  w2_weight_packed:  int32 [num_experts, inter//pack_factor, hidden]
  w13_weight_scale:  float [num_experts, hidden//group_size, 2*inter]
  w2_weight_scale:   float [num_experts, inter//group_size, hidden]
  (pack_factor = 32 // num_bits = 8 for INT4)
"""

import os
import textwrap

# =============================================================================
# Step 1: Patch compressed_tensors_moe.py to add INT4 MoE method
# =============================================================================

MOE_PATH = "/workspace/tpu_inference/tpu_inference/layers/vllm/quantization/compressed_tensors/compressed_tensors_moe.py"

with open(MOE_PATH) as f:
    code = f.read()

# Check if already patched
if "VllmCompressedTensorsW4A16IntMoEMethod" in code:
    print("SKIP: INT4 MoE method already patched")
else:
    # --- Add the new class at the end of the file ---
    new_class = textwrap.dedent('''

    class VllmCompressedTensorsW4A16IntMoEMethod(CompressedTensorsMoEMethod,
                                                  VllmQuantConfig):
        """INT4 weight-only group-quantized MoE on TPU.

        Keeps weights in packed INT4 format in HBM. Dequantizes to BF16
        inside @jax.jit during the forward pass.
        """

        def __init__(self,
                     weight_quant,
                     input_quant,
                     moe: FusedMoEConfig,
                     mesh: Mesh,
                     ep_axis_name: str = "model"):
            super().__init__(moe)
            self.mesh = mesh
            self.weight_quant = weight_quant
            self.num_bits = weight_quant.num_bits  # 4
            self.group_size = weight_quant.group_size  # 32
            self.pack_factor = 32 // self.num_bits  # 8 values per int32
            self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
            self.extra_backend_kwargs = {}
            if self.moe_backend == MoEBackend.FUSED_MOE:
                self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name)

        @property
        def is_monolithic(self) -> bool:
            return True

        def process_weights_after_loading(self, layer) -> None:
            """Convert packed INT4 weights to JAX arrays, keeping INT4 format.

            The GPU WNA16 path stores weights as:
              w13_weight_packed: int32 [E, hidden//8, 2*inter]
              w2_weight_packed:  int32 [E, inter//8, hidden]
              w13_weight_scale:  float [E, hidden//gs, 2*inter]
              w2_weight_scale:   float [E, inter//gs, hidden]

            We convert to JAX and shard across expert dimension.
            """
            import jax
            import jax.numpy as jnp
            from jax.sharding import NamedSharding, PartitionSpec as P
            from torchax.interop import torch_view
            from tpu_inference.utils import t2j
            from tpu_inference.layers.common.sharding import ShardingAxisName

            ep_sharding = NamedSharding(self.mesh, P(ShardingAxisName.EXPERT))

            # Convert packed weights to JAX (keep as int32)
            w13_packed = t2j(layer.w13_weight_packed, use_dlpack=False)
            w2_packed = t2j(layer.w2_weight_packed, use_dlpack=False)
            w13_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
            w2_scale = t2j(layer.w2_weight_scale, use_dlpack=False)

            # Free CPU memory
            for attr in ['w13_weight_packed', 'w2_weight_packed',
                         'w13_weight_scale', 'w2_weight_scale']:
                if hasattr(layer, attr):
                    getattr(layer, attr).untyped_storage().resize_(0)

            # Shard across expert dimension
            w13_packed = jax.device_put(w13_packed, ep_sharding)
            w2_packed = jax.device_put(w2_packed, ep_sharding)
            w13_scale = jax.device_put(w13_scale, ep_sharding)
            w2_scale = jax.device_put(w2_scale, ep_sharding)

            # Store as torch Parameters wrapping JAX arrays
            layer.w13_weight_packed = torch.nn.Parameter(
                torch_view(w13_packed), requires_grad=False)
            layer.w2_weight_packed = torch.nn.Parameter(
                torch_view(w2_packed), requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                torch_view(w13_scale), requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                torch_view(w2_scale), requires_grad=False)

            logger.info(
                f"INT4 MoE weights loaded: w13_packed={w13_packed.shape} "
                f"w2_packed={w2_packed.shape} "
                f"w13_scale={w13_scale.shape} w2_scale={w2_scale.shape}")

        def apply_monolithic(
            self,
            layer: FusedMoE,
            x: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass: dequant INT4 -> BF16, then standard MoE apply."""
            import jax
            import jax.numpy as jnp
            from torchax.interop import jax_view

            w13_packed = jax_view(layer.w13_weight_packed)
            w2_packed = jax_view(layer.w2_weight_packed)
            w13_scale = jax_view(layer.w13_weight_scale)
            w2_scale = jax_view(layer.w2_weight_scale)

            # Dequantize INT4 -> BF16 inside jit boundary
            # The GPU stores as [E, hidden//8, 2*inter] in int32
            # Each int32 holds 8 INT4 values along the "hidden" dimension
            @jax.named_call
            def _dequant_int4(packed, scale, pack_factor, group_size):
                """Unpack int32 -> 8x int4 -> bf16, apply group scales."""
                # packed: [E, dim_packed, out_dim] dtype=int32
                # scale:  [E, dim_packed*pack_factor//group_size, out_dim] dtype=float
                E, dim_packed, out_dim = packed.shape
                dim_full = dim_packed * pack_factor

                # Unpack: extract 8 int4 values from each int32
                # Shift and mask to get each 4-bit value
                shifts = jnp.arange(0, 32, 4, dtype=jnp.int32)  # [0,4,8,...,28]
                # packed[:,:,:,None] >> shifts[None,None,None,:] gives [E, dp, od, 8]
                unpacked = (packed[..., None] >> shifts) & 0xF
                # Sign-extend: values >= 8 are negative (two's complement for 4 bits)
                unpacked = jnp.where(unpacked >= 8, unpacked - 16, unpacked)
                # Reshape: [E, dim_packed, out_dim, 8] -> [E, dim_full, out_dim]
                unpacked = unpacked.reshape(E, dim_full, out_dim).astype(jnp.bfloat16)

                # Apply group-wise scales
                # scale: [E, num_groups, out_dim]
                # unpacked: [E, dim_full, out_dim]
                # Reshape for group application
                num_groups = dim_full // group_size
                grouped = unpacked.reshape(E, num_groups, group_size, out_dim)
                scaled = grouped * scale[:, :, None, :].astype(jnp.bfloat16)
                return scaled.reshape(E, dim_full, out_dim)

            # Dequantize both weight matrices
            w13_bf16 = _dequant_int4(w13_packed, w13_scale,
                                      self.pack_factor, self.group_size)
            w2_bf16 = _dequant_int4(w2_packed, w2_scale,
                                     self.pack_factor, self.group_size)

            # Transpose to match unquantized layout:
            # Unquantized: w13 = [E, 2*inter, hidden], w2 = [E, hidden, inter]
            # We have: w13 = [E, hidden, 2*inter], w2 = [E, inter, hidden]
            # So transpose last two dims
            w13_bf16 = jnp.swapaxes(w13_bf16, -2, -1)
            w2_bf16 = jnp.swapaxes(w2_bf16, -2, -1)

            # Now use standard unquantized MoE apply
            weights = FusedMoEWeights(
                w13_weight=w13_bf16,
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=w2_bf16,
                w2_weight_scale=None,
                w2_bias=None,
            )

            return vllm_moe_apply(layer=layer,
                                  weights=weights,
                                  quant_method_instance=self,
                                  x=x,
                                  router_logits=router_logits)
    ''')

    code += new_class

    # --- Patch get_moe_method to route INT4 to the new class ---
    old_raise = '''        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")'''

    new_check = '''        elif (weight_quant is not None
              and getattr(weight_quant, 'num_bits', None) == 4
              and getattr(weight_quant, 'type', None) == 'int'):
            return VllmCompressedTensorsW4A16IntMoEMethod(
                weight_quant, input_quant, layer.moe_config, quant_config.mesh)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")'''

    if old_raise in code:
        code = code.replace(old_raise, new_check)
        with open(MOE_PATH, "w") as f:
            f.write(code)
        print("PATCHED compressed_tensors_moe.py: added VllmCompressedTensorsW4A16IntMoEMethod")
    else:
        print("WARNING: could not find raise RuntimeError pattern to patch get_moe_method")
        # Still write the class addition
        with open(MOE_PATH, "w") as f:
            f.write(code)
        print("ADDED class but could not patch get_moe_method routing")

print("Done.")
