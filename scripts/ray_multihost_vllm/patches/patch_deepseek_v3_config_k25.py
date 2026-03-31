# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix hardcoded DeepSeek V3 JAX model parameters for Kimi K2.5.

Same architecture as K2-Instruct (384 experts, 163840 vocab, 64 heads)
but different rope_scaling: beta_fast=32 (same as DS V3), factor=64.

Key differences from DeepSeek V3:
  - 384 experts (vs 256)
  - 163840 vocab (vs 129280)
  - 64 attention heads (vs 128)
  - 1 expert group (vs 8)
  - rope_theta 50000 (vs 10000)
  - rope_scaling: factor=64, beta_fast=32 (same as DS V3 beta_fast)
  - first_k_dense_replace 1 (vs 3)
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"

with open(PATH) as f:
    code = f.read()

replacements = [
    # Model architecture (same as K2-Instruct)
    ("num_local_experts: int = 256", "num_local_experts: int = 384"),
    ("vocab_size: int = 129280", "vocab_size: int = 163840"),
    ("num_attention_heads: int = 128", "num_attention_heads: int = 64"),
    ("num_key_value_heads: int = 128", "num_key_value_heads: int = 64"),
    ("n_group: int = 8", "n_group: int = 1"),
    ("routed_scaling_factor: float = 2.5", "routed_scaling_factor: float = 2.827"),
    ("first_k_dense_replace: int = 3", "first_k_dense_replace: int = 1"),
    # RoPE — K2.5 specific: factor=64 (beta_fast=32 is already the DS V3 default)
    ("rope_theta = 10000", "rope_theta = 50000"),
    ('"factor": 40,', '"factor": 64.0,'),
    # MoE routing
    ("topk_groups=4,", "topk_groups=1,"),
]

patched = 0
for old, new in replacements:
    if old in code:
        code = code.replace(old, new)
        patched += 1

if patched > 0:
    with open(PATH, "w") as f:
        f.write(code)
    print(f"PATCHED deepseek_v3.py: {patched} values updated for Kimi K2.5")
else:
    print("SKIP: no patterns found (already patched?)")
