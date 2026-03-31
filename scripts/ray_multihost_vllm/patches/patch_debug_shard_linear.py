# Debug patch: log mesh and specs inside shard_linear_weights and process_weights_after_loading

import re

# Patch 1: shard_linear_weights — log mesh and specs
PATH1 = "/workspace/tpu_inference/tpu_inference/layers/common/process_weights/linear_weights.py"
with open(PATH1) as f:
    code1 = f.read()

old1 = '    mesh = mesh or meshlib.get_concrete_mesh()\n    if not transposed:'
new1 = '''    mesh = mesh or meshlib.get_concrete_mesh()
    import logging as _log
    _log.warning(f"SHARD_LINEAR | mesh={mesh.axis_names if mesh else 'NONE'} shape={mesh.shape if mesh else 'NONE'} | weight_p_spec={weight_p_spec} | bias_p_spec={bias_p_spec}")
    if not transposed:'''

if old1 in code1:
    code1 = code1.replace(old1, new1)
    with open(PATH1, "w") as f:
        f.write(code1)
    print("PATCHED linear_weights.py: debug logging in shard_linear_weights")
else:
    print("SKIP linear_weights.py")

# Patch 2: fp8.py — log what happens after process_weights_after_loading
PATH2 = "/workspace/tpu_inference/tpu_inference/layers/jax/quantization/fp8.py"
with open(PATH2) as f:
    code2 = f.read()

old2 = '''        # Put onto the device.
        weights = shard_linear_weights('''
new2 = '''        # Put onto the device.
        import logging as _log
        _log.warning(f"FP8_DEBUG | pre-shard weight.shape={weights.weight.shape} dtype={weights.weight.dtype} devices={weights.weight.devices() if hasattr(weights.weight, 'devices') else 'N/A'} | weight_sharding={self.linear_config.weight_sharding}")
        weights = shard_linear_weights('''

if old2 in code2:
    code2 = code2.replace(old2, new2)

    # Also add post-shard debug
    old3 = '''        if self.linear_config.fuse_matmuls:
            layer.weight = nnx.Param(weights.weight)'''
    new3 = '''        import logging as _log2
        _log2.warning(f"FP8_DEBUG | post-shard weight.shape={weights.weight.shape} dtype={weights.weight.dtype} devices={list(weights.weight.devices())[:4] if hasattr(weights.weight, 'devices') else 'N/A'} sharding={weights.weight.sharding if hasattr(weights.weight, 'sharding') else 'N/A'}")
        if self.linear_config.fuse_matmuls:
            layer.weight = nnx.Param(weights.weight)'''

    if old3 in code2:
        code2 = code2.replace(old3, new3)

    with open(PATH2, "w") as f:
        f.write(code2)
    print("PATCHED fp8.py: debug logging in process_weights_after_loading")
else:
    print("SKIP fp8.py")
