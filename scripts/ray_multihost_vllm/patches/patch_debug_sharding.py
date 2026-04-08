# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Debug patch: Log detailed sharding info during weight loading.

Instruments assign_and_shard_param to log:
- The sharding spec from nnx metadata
- The mesh used (and whether it's from param metadata, passed arg, or get_mesh())
- The resulting sharding after shard_put
- Per-chip memory after each weight

This helps diagnose why TP sharding might not be distributing weights.
"""


PATH = "/workspace/tpu_inference/tpu_inference/models/jax/utils/weight_utils.py"

with open(PATH) as f:
    code = f.read()

old = '''def assign_and_shard_param(jax_param: nnx.Param,
                           jax_weight: jax.Array,
                           param_name: str = "Unknown",
                           mesh: Optional[Mesh] = None) -> None:
    """Distributes a JAX array across devices according to the `nnx.Param`'s sharding metadata, assigns it to the parameter, and marks it as loaded.

    Args:
        jax_param: The target nnx.Param to assign the weight to.
        jax_weight: The JAX array containing the weight data.
        param_name: The name of the parameter, used for error logging.
        mesh: The device mesh to shard the parameter on.
    """
    spec = jax_param.get_metadata().get("sharding", ())
    if isinstance(spec, NamedSharding):
        spec = spec.spec
    elif isinstance(spec, SingleDeviceSharding):
        spec = ()
    param_mesh = jax_param.get_metadata().get("mesh") or mesh
    try:
        jax_param.value = shard_put(jax_weight, spec, mesh=param_mesh)
        jax_param.set_metadata("_is_loaded", True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load weight '{param_name}' with shape {jax_weight.shape} into param with shape {jax_param.value.shape}"
        ) from e'''

new = '''def assign_and_shard_param(jax_param: nnx.Param,
                           jax_weight: jax.Array,
                           param_name: str = "Unknown",
                           mesh: Optional[Mesh] = None) -> None:
    """Distributes a JAX array across devices according to the `nnx.Param`'s sharding metadata, assigns it to the parameter, and marks it as loaded.

    Args:
        jax_param: The target nnx.Param to assign the weight to.
        jax_weight: The JAX array containing the weight data.
        param_name: The name of the parameter, used for error logging.
        mesh: The device mesh to shard the parameter on.
    """
    spec = jax_param.get_metadata().get("sharding", ())
    raw_spec = spec  # keep original for logging
    if isinstance(spec, NamedSharding):
        spec = spec.spec
    elif isinstance(spec, SingleDeviceSharding):
        spec = ()
    param_mesh = jax_param.get_metadata().get("mesh") or mesh
    mesh_source = "param_metadata" if jax_param.get_metadata().get("mesh") else ("arg" if mesh else "get_mesh()")

    # DEBUG: Log first 30 weights and every 100th after
    import os
    _debug_count = int(os.environ.get("_SHARD_DEBUG_COUNT", "0"))
    os.environ["_SHARD_DEBUG_COUNT"] = str(_debug_count + 1)
    if _debug_count < 30 or _debug_count % 100 == 0:
        _resolved_mesh = param_mesh if param_mesh is not None else get_mesh()
        logger.warning(
            f"SHARD_DEBUG | #{_debug_count} | {param_name} | "
            f"shape={jax_weight.shape} dtype={jax_weight.dtype} | "
            f"raw_spec={raw_spec} | resolved_spec={spec} | "
            f"mesh_source={mesh_source} | "
            f"mesh_axes={_resolved_mesh.axis_names if _resolved_mesh else 'NONE'} | "
            f"mesh_shape={_resolved_mesh.shape if _resolved_mesh else 'NONE'} | "
            f"weight_devices={jax_weight.devices() if hasattr(jax_weight, 'devices') else 'numpy'}"
        )

    try:
        jax_param.value = shard_put(jax_weight, spec, mesh=param_mesh)
        jax_param.set_metadata("_is_loaded", True)

        # DEBUG: Verify placement after sharding
        if _debug_count < 30 or _debug_count % 100 == 0:
            v = jax_param.value
            if hasattr(v, 'sharding'):
                logger.warning(
                    f"SHARD_RESULT | #{_debug_count} | {param_name} | "
                    f"result_sharding={v.sharding} | "
                    f"result_devices={list(v.devices())[:4]}... | "
                    f"shard_shape={v.addressable_shards[0].data.shape if v.addressable_shards else 'none'}"
                )
    except Exception as e:
        logger.error(
            f"SHARD_FAIL | {param_name} | shape={jax_weight.shape} | "
            f"spec={spec} | mesh={param_mesh} | error={e}"
        )
        raise RuntimeError(
            f"Failed to load weight '{param_name}' with shape {jax_weight.shape} into param with shape {jax_param.value.shape}"
        ) from e'''

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED weight_utils.py: Added detailed sharding debug logging")
else:
    print("SKIP: assign_and_shard_param pattern not found (already patched?)")
