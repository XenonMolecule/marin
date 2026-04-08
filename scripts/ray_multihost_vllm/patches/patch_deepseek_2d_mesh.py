# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt DeepSeek V3 JAX model sharding for 2D mesh ("data", "model").

The original model uses 5D axis names. With a 2D mesh, we can only shard across
"data" and "model". For MoE specs that had two shard axes (e.g., expert + model),
we keep only one to avoid DuplicateSpecError.

Strategy:
  - Primary compute axes (ATTN_HEAD, MOE_TENSOR, MLP_TENSOR) -> "model"
  - Data parallelism axes (MLP_DATA, ATTN_DATA) -> "data"
  - Expert axes (ATTN_DATA_EXPERT, EXPERT) -> None (replicate, since 2D mesh
    can't shard experts separately from model parallelism)
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"

with open(PATH) as f:
    code = f.read()

# First, revert any previous patches (restore original axis names)
# We'll work from whatever state the file is in

# Replace axis names carefully to avoid duplicate specs
# The key insight: in 2D mesh, expert-related axes become None (replicate)
# and compute axes become "model"
replacements = [
    # Expert sharding -> replicate (can't do EP with only 2 axes)
    ("ShardingAxisName.ATTN_DATA_EXPERT", "None"),
    ("ShardingAxisName.EXPERT_DATA", "None"),
    ("ShardingAxisName.EXPERT", "None"),
    # Compute sharding -> "model"
    ("ShardingAxisName.ATTN_HEAD", '"model"'),
    ("ShardingAxisName.MOE_TENSOR", '"model"'),
    ("ShardingAxisName.MLP_TENSOR", '"model"'),
    ("ShardingAxisName.VOCAB", '"model"'),
    # Data parallelism -> "data"
    ("ShardingAxisName.ATTN_DATA", '"data"'),
    ("ShardingAxisName.MLP_DATA", '"data"'),
]

patched = 0
for old, new in replacements:
    if old in code:
        count = code.count(old)
        code = code.replace(old, new)
        patched += count

# Also handle the case where previous patch already replaced to "model"
# Fix any P(..., "model", "model") -> P(..., None, "model")
import re


# Find P() calls with duplicate "model"
def fix_duplicate_model(match):
    spec = match.group(0)
    # Count "model" occurrences
    parts = spec.split(",")
    model_count = sum(1 for p in parts if '"model"' in p)
    if model_count > 1:
        # Replace first "model" with None
        fixed = False
        new_parts = []
        for p in parts:
            if '"model"' in p and not fixed:
                new_parts.append(p.replace('"model"', "None"))
                fixed = True
            else:
                new_parts.append(p)
        return ",".join(new_parts)
    return spec


code = re.sub(r"P\([^)]+\)", fix_duplicate_model, code)


# Also fix P("data", "data", ...) duplicates
def fix_duplicate_data(match):
    spec = match.group(0)
    parts = spec.split(",")
    data_count = sum(1 for p in parts if '"data"' in p)
    if data_count > 1:
        fixed = False
        new_parts = []
        for p in parts:
            if '"data"' in p and not fixed:
                new_parts.append(p.replace('"data"', "None"))
                fixed = True
            else:
                new_parts.append(p)
        return ",".join(new_parts)
    return spec


code = re.sub(r"P\([^)]+\)", fix_duplicate_data, code)

if patched > 0:
    with open(PATH, "w") as f:
        f.write(code)
    print(f"PATCHED deepseek_v3.py: {patched} axis refs changed to 2D (no duplicates)")
else:
    print("SKIP: no axis names found (already patched?)")
