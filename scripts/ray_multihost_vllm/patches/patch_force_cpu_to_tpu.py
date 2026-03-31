# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Force ALL CPU arrays to TPU after weight loading using nnx.state().

named_parameters() misses FP8 weights stored in _weights_to_load buffers.
nnx.state() captures the full graph state including all nested arrays.
Walk the state tree and move any CPU arrays to TPU mesh.
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"
with open(PATH) as f:
    code = f.read()

old = """        loaded = loader.load_weights(weights)

        self.model.initialize_cache()"""

new = """        loaded = loader.load_weights(weights)

        # FIX: Move ALL CPU arrays in model state to TPU mesh.
        # Under Ray, process_weights_after_loading never runs (quant_method
        # lost during serialization), leaving FP8 weights on CPU.
        import numpy as _np
        import jax as _jax
        from jax.sharding import NamedSharding as _NS, PartitionSpec as _PS
        from flax import nnx as _nnx
        _mesh = None
        try:
            from jax.sharding import get_mesh as _gm
            _mesh = _gm()
        except Exception:
            pass
        if _mesh is not None and 'cpu' not in _mesh.axis_names:
            _state = _nnx.state(self)
            _flat = _state.flat_state()
            _moved = 0
            _total = 0
            _cpu_count = 0
            for _key, _leaf in _flat:
                if not isinstance(_leaf, _nnx.Variable):
                    continue
                _v = _leaf.value
                _total += 1
                if not isinstance(_v, _jax.Array):
                    continue
                _on_cpu = any('cpu' in str(d).lower() for d in _v.devices())
                if not _on_cpu:
                    continue
                _cpu_count += 1
                # Get sharding spec from variable metadata
                _spec = _leaf.get_metadata('sharding', ())
                if isinstance(_spec, _NS):
                    _spec = _spec.spec
                if isinstance(_spec, tuple):
                    _spec = _PS(*_spec) if _spec else _PS(*([None] * _v.ndim))
                elif not isinstance(_spec, _PS):
                    _spec = _PS(*([None] * _v.ndim))
                try:
                    _v_np = _np.asarray(_v)
                    _ns = _NS(_mesh, _spec)
                    _leaf.value = _jax.make_array_from_callback(
                        _v_np.shape, _ns, lambda idx, _d=_v_np: _d[idx])
                    _moved += 1
                except Exception as _e:
                    _kstr = '.'.join(str(k) for k in _key) if isinstance(_key, tuple) else str(_key)
                    if _moved < 5 or _moved % 100 == 0:
                        logger.warning(f"CPU->TPU fail [{_kstr}]: {_e}")
            logger.warning(f"CPU_TO_TPU: {_moved}/{_cpu_count} CPU arrays moved to TPU (total={_total}, mesh={_mesh.shape})")
            # Write back
            _nnx.update(self, _state)

        self.model.initialize_cache()"""

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED deepseek_v3.py: nnx.state() CPU→TPU transfer")
else:
    print("SKIP: pattern not found (already patched?)")
