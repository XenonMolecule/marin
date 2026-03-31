# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nuclear fix: Force ALL weights from CPU to TPU after loading.

The FP8 weight loading pipeline under Ray multi-host has a broken
process_weights_after_loading path — it never runs, leaving ALL FP8
weights on CPU. This causes 240GB/chip CompileTimeHbmOom because XLA
plans to materialize CPU weights on TPU during execution.

Fix: After load_weights completes, iterate all model parameters and
move any CPU-resident weights to TPU mesh using their nnx sharding specs.
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"

with open(PATH) as f:
    code = f.read()

# Add a post-load CPU→TPU transfer after self.model.initialize_cache()
old = """        loaded = loader.load_weights(weights)

        self.model.initialize_cache()"""

new = """        loaded = loader.load_weights(weights)

        self.model.initialize_cache()

        # FIX: Force all CPU-resident weights to TPU mesh.
        # Under Ray multi-host, FP8 process_weights_after_loading never runs,
        # leaving weights on CPU. Move them to TPU using nnx sharding specs.
        import numpy as np
        from jax.sharding import get_mesh as _get_mesh, NamedSharding, SingleDeviceSharding
        from jax.sharding import PartitionSpec as _P
        _mesh = _get_mesh()
        _moved = 0
        _total = 0
        if _mesh is not None and 'cpu' not in _mesh.axis_names:
            for _name, _param in self.named_parameters():
                _total += 1
                _v = _param.value
                if not hasattr(_v, 'devices'):
                    continue
                _devs = list(_v.devices())
                if any('cpu' in str(d).lower() for d in _devs):
                    _spec = _param.get_metadata().get('sharding', ())
                    if isinstance(_spec, NamedSharding):
                        _spec = _spec.spec
                    elif isinstance(_spec, SingleDeviceSharding):
                        _spec = ()
                    if isinstance(_spec, tuple):
                        _spec = _P(*_spec)
                    try:
                        _v_np = np.asarray(_v)
                        _sharding = NamedSharding(_mesh, _spec)
                        import jax
                        _param.value = jax.make_array_from_callback(
                            _v_np.shape, _sharding, lambda idx, _d=_v_np: _d[idx])
                        _moved += 1
                    except Exception as e:
                        logger.warning(f"Failed to move {_name} to TPU: {e}")
            logger.warning(f"POST_LOAD_FIX: Moved {_moved}/{_total} weights from CPU to TPU mesh")"""

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED deepseek_v3.py: Force CPU weights to TPU after loading")
else:
    print("SKIP: load_weights pattern not found")
