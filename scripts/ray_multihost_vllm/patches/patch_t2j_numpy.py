# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix slow t2j dtype conversion by using numpy instead of JAX.

Root cause: t2j converts torch FP8/BF16 tensors via:
    bytes = t.view(uint8).numpy()
    return jnp.array(bytes).view(fp8_dtype)

The jnp.array().view() triggers JAX's _convert_element_type which blocks
on a futex for seconds per call. With 384 experts × multiple projections
per MoE layer, this adds up to minutes per shard.

Fix: Use numpy's ml_dtypes for the view instead of JAX. numpy.view() is
instant (just reinterprets the buffer). Downstream code (jnp.concatenate,
device_put, make_array_from_callback) all accept numpy arrays.
"""

PATH = "/workspace/tpu_inference/tpu_inference/utils.py"

with open(PATH) as f:
    code = f.read()

old = """    try:
        if t.dtype in _NUMPY_UNSUPPORTED_DTYPES:
            # This bit cast require t to be continguous and more than 1 dimension.
            if t.is_contiguous() and t.dim():
                bytes = t.cpu().view(torch.uint8).detach().numpy()
                return jnp.array(bytes).view(
                    _NUMPY_UNSUPPORTED_DTYPES[t.dtype])
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("t2j bit cast failed, falling back to torchax t2j: %s",
                       e)"""

new = """    try:
        if t.dtype in _NUMPY_UNSUPPORTED_DTYPES:
            # This bit cast require t to be continguous and more than 1 dimension.
            if t.is_contiguous() and t.dim():
                bytes = t.cpu().view(torch.uint8).detach().numpy()
                # OPTIMIZATION: Use numpy ml_dtypes for the view instead of
                # jnp.array().view() which triggers slow JAX _convert_element_type
                # on CPU (blocks on futex for seconds per call).
                import ml_dtypes as _ml_dtypes
                _NP_DTYPE_MAP = {
                    'torch.bfloat16': _ml_dtypes.bfloat16,
                    'torch.float8_e4m3fn': _ml_dtypes.float8_e4m3fn,
                    'torch.float8_e4m3fnuz': _ml_dtypes.float8_e4m3fnuz,
                    'torch.float8_e5m2': _ml_dtypes.float8_e5m2,
                    'torch.float8_e5m2fnuz': _ml_dtypes.float8_e5m2fnuz,
                }
                _np_dtype = _NP_DTYPE_MAP.get(str(t.dtype))
                if _np_dtype is not None:
                    return bytes.view(_np_dtype).reshape(t.shape)
                # Fallback to JAX path if dtype not in map
                return jnp.array(bytes).view(
                    _NUMPY_UNSUPPORTED_DTYPES[t.dtype])
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("t2j bit cast failed, falling back to torchax t2j: %s",
                       e)"""

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED t2j: numpy ml_dtypes path (bypasses slow JAX _convert_element_type)")
else:
    print("SKIP: t2j pattern not found")
