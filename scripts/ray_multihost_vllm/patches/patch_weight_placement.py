# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix weight placement under Ray multi-host.

In general_device_put (Ray multi-host path), CPU JAX arrays need to be
converted to numpy for reliable transfer via make_array_from_callback.

Fix: Convert CPU arrays to numpy before the callback.

NOTE: We do NOT discard CPU mesh in assign_and_shard_param. Dense linear
weights NEED CPU placement for process_weights_after_loading. MoE weights
bypass assign_and_shard_param entirely via our custom Fp8FusedMoEMethod.load_weights.
"""

import re

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/utils/weight_utils.py"

with open(PATH) as f:
    code = f.read()

print("SKIP: assign_and_shard_param not modified (dense linears need CPU placement)")

# Fix 2: general_device_put — convert CPU arrays to numpy for Ray transfer
PATH2 = "/workspace/tpu_inference/tpu_inference/layers/common/utils.py"

with open(PATH2) as f:
    code2 = f.read()

old_gdp = """        # NOTE: at here, num_global_devices != num_local_devices
        # meaning we are in multi-host setup. Each host will run the same process
        # and each process only need to handle the devices accessible to this host.
        ctx = nullcontext() if source_mesh is None else jax.set_mesh(
            source_mesh)
        # `t[i]` needs to be operated in the same mesh as `t`, which is provided as
        # `source_mesh`.
        with ctx:
            global_array = jax.make_array_from_callback(
                t.shape, sharding, lambda index: t[index])"""

new_gdp = """        # NOTE: at here, num_global_devices != num_local_devices
        # meaning we are in multi-host setup. Each host will run the same process
        # and each process only need to handle the devices accessible to this host.
        ctx = nullcontext() if source_mesh is None else jax.set_mesh(
            source_mesh)
        # `t[i]` needs to be operated in the same mesh as `t`, which is provided as
        # `source_mesh`.
        # FIX: When weights are loaded on CPU (via cpu_mesh_context), convert
        # to numpy before using as callback data. CPU JAX arrays don't transfer
        # correctly to TPU via make_array_from_callback under Ray multi-host.
        import numpy as np
        if source_mesh is None and isinstance(t, jax.Array):
            t_data = np.asarray(t)
        else:
            t_data = t
        with ctx:
            global_array = jax.make_array_from_callback(
                t.shape, sharding, lambda index: t_data[index])"""

if old_gdp in code2:
    code2 = code2.replace(old_gdp, new_gdp)
    with open(PATH2, "w") as f:
        f.write(code2)
    print("PATCHED utils.py: CPU→numpy conversion for Ray multi-host transfer")
else:
    print("SKIP: general_device_put pattern not found")

with open(PATH, "w") as f:
    f.write(code)
