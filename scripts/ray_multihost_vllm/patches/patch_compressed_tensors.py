# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept compressed-tensors (INT4) quantization on TPU JAX path.

The tpu_inference JAX quantization layer only supports None (BF16) and fp8.
This patch adds compressed-tensors to the accepted methods, treating it the
same as fp8: accept the config but return None, causing the weight loader
to dequantize INT4 weights to BF16 at load time.

This enables serving models like unsloth/Kimi-K2.5 (INT4 quantized via QAT)
on TPU, at the cost of running in BF16 precision in HBM.
"""


PATH = "/workspace/tpu_inference/tpu_inference/layers/jax/quantization/__init__.py"

with open(PATH) as f:
    code = f.read()

# Add COMPRESSED_TENSORS import
old_import = "from tpu_inference.layers.common.quant_methods import FP8"
new_import = "from tpu_inference.layers.common.quant_methods import COMPRESSED_TENSORS, FP8"

if old_import in code and "COMPRESSED_TENSORS" not in code:
    code = code.replace(old_import, new_import)

    # Add compressed-tensors to the method_to_config dict
    old_dict_end = "        FP8: lambda _: None,\n    }"
    new_dict_end = "        FP8: lambda _: None,\n        COMPRESSED_TENSORS: lambda _: None,\n    }"

    if old_dict_end in code:
        code = code.replace(old_dict_end, new_dict_end)
        with open(PATH, "w") as f:
            f.write(code)
        print("PATCHED: added compressed-tensors to accepted JAX quant methods")
    else:
        print("SKIP: could not find method_to_config dict end")
elif "COMPRESSED_TENSORS" in code:
    print("SKIP: compressed-tensors already patched")
else:
    print("SKIP: import pattern not found")
