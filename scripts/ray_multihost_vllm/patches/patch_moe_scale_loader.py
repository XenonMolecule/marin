# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix: Add weight_loader to MoE FP8 scale params.

Fp8FusedMoEMethod.create_weights_jax creates scale params WITHOUT
weight_loader callbacks. JaxAutoWeightsLoader uses the default loader
(assign_and_shard_param) which puts the entire scale tensor into .value
instead of accumulating per-expert shards into _weights_to_load.

Fix: Set weight_loader on each scale param that accumulates per-expert
shards into _weights_to_load, matching Fp8FusedMoEMethod.load_weights.
"""

PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/quantization/fp8.py"
with open(PATH) as f:
    code = f.read()

# Add weight_loader to scale params after they're created
old = """                    setattr(
                        layer, f"{param_name}_{self.weight_scale_name}",
                        nnx.Param(scale_value,
                                  _weights_to_load=[None for _ in range(E)]))"""

new = """                    _scale_param = nnx.Param(scale_value,
                                  _weights_to_load=[None for _ in range(E)])
                    # FIX: Set weight_loader for scale params so JaxAutoWeightsLoader
                    # accumulates per-expert shards into _weights_to_load correctly.
                    from functools import partial as _partial
                    from tpu_inference.models.jax.utils.weight_utils import jax_array_from_reshaped_torch as _jart
                    def _scale_weight_loader(_param, _torch_weight, expert_id=None):
                        if expert_id is not None and hasattr(_param, '_weights_to_load'):
                            _jax_w = _jart(_torch_weight, reshape_dims=(1,) + _torch_weight.shape)
                            _param._weights_to_load[expert_id] = _jax_w
                        else:
                            _jax_w = _jart(_torch_weight)
                            from tpu_inference.models.jax.utils.weight_utils import assign_and_shard_param
                            assign_and_shard_param(_param, _jax_w, param_name=f"{param_name}_{self.weight_scale_name}")
                    _scale_param.set_metadata("weight_loader", _scale_weight_loader)
                    setattr(layer, f"{param_name}_{self.weight_scale_name}", _scale_param)"""

if old in code:
    code = code.replace(old, new, 1)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED fp8.py: Added weight_loader to MoE FP8 scale params")
else:
    print("SKIP: pattern not found")
