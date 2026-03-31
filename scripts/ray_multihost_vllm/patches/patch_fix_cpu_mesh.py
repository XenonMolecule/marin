# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""REVERT the CPU mesh discard in assign_and_shard_param.

We discovered that FP8 weights NEED to be on CPU during loading because
process_weights_after_loading processes them on CPU before moving to TPU.
The CPU mesh in param metadata is INTENTIONAL for the loading phase.

The real fix is in patch_fp8_batch_placement.py which handles the
batch_features MoE weights that skip CPU processing.
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/utils/weight_utils.py"
with open(PATH) as f:
    code = f.read()

# Revert our broken fix back to the original
broken = """    # FIX: Discard CPU mesh from param metadata. FP8 JaxEinsum weights store
    # cpu_mesh in metadata from init; we always want the TPU mesh from get_mesh().
    _pm = jax_param.get_metadata().get("mesh")
    if _pm is not None and 'cpu' in _pm.axis_names:
        _pm = None
    param_mesh = _pm or mesh"""

original = '    param_mesh = jax_param.get_metadata().get("mesh") or mesh'

if broken in code:
    code = code.replace(broken, original)
    with open(PATH, "w") as f:
        f.write(code)
    print("REVERTED weight_utils.py: Restored original param_mesh logic (CPU mesh is intentional)")
elif original in code:
    print("SKIP: Already has original param_mesh logic")
else:
    print("SKIP: Neither broken nor original pattern found")
