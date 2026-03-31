# Pure diagnostic: add comprehensive logging after weight loading.
# No fixes, just information gathering.

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"
with open(PATH) as f:
    code = f.read()

old = """        loaded = loader.load_weights(weights)

        self.model.initialize_cache()"""

new = """        loaded = loader.load_weights(weights)

        # === DIAGNOSTIC: dump weight placement info to file ===
        import jax as _jax
        from flax import nnx as _nnx
        with open("/tmp/weight_diagnostic.txt", "w") as _f:
            # 1. What model class am I?
            _f.write(f"MODEL_CLASS: {type(self).__name__}\\n")
            _f.write(f"MODEL_BASES: {[c.__name__ for c in type(self).__mro__[:5]]}\\n")

            # 2. Check mesh context
            try:
                from jax.sharding import get_mesh
                _m = get_mesh()
                _f.write(f"ACTIVE_MESH: axes={_m.axis_names} shape={_m.shape} devices={list(_m.devices.flatten())[:2]}\\n")
            except Exception as e:
                _f.write(f"ACTIVE_MESH: ERROR {e}\\n")

            # 3. Sample 10 nnx.state variables - check device and sharding
            _state = _nnx.state(self)
            _flat = _state.flat_state()
            _total = 0
            _on_tpu = 0
            _on_cpu = 0
            _replicated = 0
            _sharded = 0
            for _key, _leaf in _flat:
                if not isinstance(_leaf, _nnx.Variable):
                    continue
                _v = _leaf.value
                _total += 1
                if not isinstance(_v, _jax.Array):
                    continue
                _devs = list(_v.devices())
                _is_cpu = any('cpu' in str(d).lower() for d in _devs)
                _is_tpu = any('tpu' in str(d).lower() for d in _devs)
                if _is_cpu:
                    _on_cpu += 1
                if _is_tpu:
                    _on_tpu += 1
                # Check if sharded (different shard shapes) or replicated
                if hasattr(_v, 'sharding') and hasattr(_v.sharding, 'spec'):
                    _spec = _v.sharding.spec
                    _any_sharded = any(s is not None for s in _spec)
                    if _any_sharded:
                        _sharded += 1
                    else:
                        _replicated += 1
                # Log first 20 and every 50th
                if _total <= 20 or _total % 50 == 0:
                    _kstr = '.'.join(str(k) for k in _key) if isinstance(_key, tuple) else str(_key)
                    _f.write(f"VAR #{_total}: {_kstr} | shape={_v.shape} dtype={_v.dtype} | devices={_devs[:2]} | sharding={_v.sharding if hasattr(_v, 'sharding') else 'N/A'}\\n")

            _f.write(f"\\nSUMMARY: total={_total} on_cpu={_on_cpu} on_tpu={_on_tpu} replicated={_replicated} sharded={_sharded}\\n")

            # 4. Check HBM usage
            try:
                for d in _jax.local_devices()[:4]:
                    s = d.memory_stats()
                    _f.write(f"HBM {d}: {s['bytes_in_use']/1e9:.2f}GB / {s['bytes_limit']/1e9:.2f}GB\\n")
            except Exception as e:
                _f.write(f"HBM: ERROR {e}\\n")

            # 5. Check a specific MoE layer for quant_method and _weights_to_load
            if hasattr(self.model, 'layers'):
                for i, layer in enumerate(self.model.layers):
                    if hasattr(layer, '__class__') and 'PPMissing' not in type(layer).__name__:
                        _f.write(f"\\nFIRST_REAL_LAYER: index={i} type={type(layer).__name__}\\n")
                        if hasattr(layer, 'mlp'):
                            mlp = layer.mlp
                            _f.write(f"  MLP type={type(mlp).__name__}\\n")
                            _f.write(f"  MLP.quant_method={getattr(mlp, 'quant_method', 'MISSING')}\\n")
                            # Check kernel params
                            for attr in ['kernel_gating_EDF', 'kernel_up_proj_EDF', 'kernel_down_proj_EFD', 'kernel_gating_upproj_EDF']:
                                p = getattr(mlp, attr, None)
                                if p is not None:
                                    _wtl = getattr(p, '_weights_to_load', None)
                                    _f.write(f"  {attr}: type={type(p).__name__} shape={p.value.shape if hasattr(p, 'value') else 'N/A'} _wtl={'present' if _wtl else 'None'} _wtl_loaded={sum(1 for w in _wtl if w is not None) if _wtl else 0}/{len(_wtl) if _wtl else 0}\\n")
                        break

        logger.warning("DIAGNOSTIC: Written to /tmp/weight_diagnostic.txt")

        self.model.initialize_cache()"""

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED: Added comprehensive weight diagnostic")
else:
    print("SKIP: pattern not found")
