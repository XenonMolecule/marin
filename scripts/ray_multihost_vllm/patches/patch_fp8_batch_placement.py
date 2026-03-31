# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix FP8 batch_features weights stuck on CPU.

In fp8.py process_weights_after_loading, the batch_features path returns
early without moving weights from CPU to TPU. MoE expert weights use
batch_features=True because they have an expert batch dimension. These
weights stay on CPU mesh → XLA CompileTimeHbmOom (240GB/chip).

Fix: In the batch_features early return, shard weights onto TPU mesh.
"""

PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/quantization/fp8.py"

with open(PATH) as f:
    code = f.read()

old = """    def process_weights_after_loading(self, layer: JaxEinsum) -> bool:
        assert isinstance(layer, JaxEinsum)
        assert self.quant_config.weight_block_size is not None

        if self.batch_features:
            # Batched case: weight stays in FP8. No blockwise processing
            # needed — the batched matmul uses dot_general with FP8 natively.
            return True"""

new = """    def process_weights_after_loading(self, layer: JaxEinsum) -> bool:
        assert isinstance(layer, JaxEinsum)
        assert self.quant_config.weight_block_size is not None

        if self.batch_features:
            # Batched case: weight stays in FP8. No blockwise processing
            # needed — the batched matmul uses dot_general with FP8 natively.
            # FIX: Still need to move weights from CPU to TPU mesh for Ray PP.
            # Weights were loaded on CPU (via cpu_mesh metadata) and need to be
            # sharded onto the TPU mesh for compilation.
            from jax.sharding import get_mesh as _get_mesh, NamedSharding
            from jax.sharding import PartitionSpec as P
            _tpu_mesh = _get_mesh()
            if _tpu_mesh is not None and 'cpu' not in _tpu_mesh.axis_names:
                from tpu_inference.layers.common.utils import general_device_put
                import numpy as np
                _w_spec = self.weight_sharding if self.weight_sharding else P()
                _s_spec = P()  # scale is always replicated
                for attr_name, spec in [('weight', _w_spec), ('weight_scale_inv', _s_spec)]:
                    param = getattr(layer, attr_name, None)
                    if param is not None and hasattr(param, 'value'):
                        v = param.value
                        # Convert CPU array to numpy, then place on TPU
                        v_np = np.asarray(v) if hasattr(v, 'devices') else v
                        sharding = NamedSharding(_tpu_mesh, spec)
                        import jax
                        param.value = jax.make_array_from_callback(
                            v_np.shape, sharding, lambda idx, _v=v_np: _v[idx])
            return True"""

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED fp8.py: Move batch_features weights from CPU to TPU mesh")
else:
    print("SKIP: batch_features pattern not found (already patched?)")
