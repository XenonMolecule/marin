# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix mesh axis names for DeepSeek V3 JAX model compatibility.

The build_mesh function creates a 4-axis mesh ("data", "expert", "seq", "model")
but DeepSeek V3's JAX model expects 5 axes including "attn_dp" and "attn_dp_expert".
This patch adds the missing axes with size 1.
"""

PATH = "/workspace/tpu_inference/tpu_inference/layers/common/sharding.py"

with open(PATH) as f:
    code = f.read()

old = '''    axis_order = {
        "data": strategy.get("data_parallelism", 1),
        "expert": strategy.get("expert_parallelism", 1),
        "seq": strategy.get("sequence_parallelism", 1),
        "model": strategy.get("tensor_parallelism", 1),
    }'''

new = '''    axis_order = {
        "data": strategy.get("data_parallelism", 1),
        "attn_dp": strategy.get("attention_data_parallelism", 1),
        "attn_dp_expert": strategy.get("attention_data_expert_parallelism", 1),
        "expert": strategy.get("expert_parallelism", 1),
        "model": strategy.get("tensor_parallelism", 1),
    }'''

if old in code:
    code = code.replace(old, new)
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED: added attn_dp and attn_dp_expert axes to build_mesh")
else:
    print("SKIP: pattern not found")
