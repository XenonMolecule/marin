# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix slow jnp.concatenate in MoE _load_weights.

The _load_weights method in moe.py concatenates 384 expert weight arrays
using jnp.concatenate inside cpu_mesh_context(). When arrays are numpy
(from t2j patch), each triggers _convert_element_type = ~15s total.

Fix: use np.concatenate (1.7s) then jnp.array for the shard_put.
"""

PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/moe/moe.py"

with open(PATH) as f:
    code = f.read()

old = """            if all(w is not None for w in weights_to_load):
                with cpu_mesh_context():
                    weights = jnp.concatenate(param._weights_to_load, axis=0)"""

new = """            if all(w is not None for w in weights_to_load):
                import numpy as _np_moe_load
                _to_np = lambda x: _np_moe_load.asarray(x) if not isinstance(x, _np_moe_load.ndarray) else x
                with cpu_mesh_context():
                    weights = jnp.array(_np_moe_load.concatenate(
                        [_to_np(w) for w in param._weights_to_load], axis=0))"""

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED moe.py: np.concatenate in _load_weights (fixes W0 bottleneck)")
else:
    print("SKIP: _load_weights pattern not found")
