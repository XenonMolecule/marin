# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix hardcoded DeepSeek V3 JAX model parameters for Kimi K2-Instruct.

The tpu_inference JAX DeepSeek V3 model has hardcoded values matching the
original DeepSeek-V3 (256 experts, 129280 vocab, 128 heads, etc.). Kimi
K2-Instruct uses different values. This patch updates them to match.

Key differences from DeepSeek V3:
  - 384 experts (vs 256)
  - 163840 vocab (vs 129280)
  - 64 attention heads (vs 128)
  - 1 expert group (vs 8)
  - rope_theta 50000 (vs 10000)
  - rope_scaling: factor=32, beta_fast=1.0, beta_slow=1.0 (vs 40, 32, 1)
  - first_k_dense_replace 1 (vs 3)
"""

PATH = "/workspace/tpu_inference/tpu_inference/models/jax/deepseek_v3.py"

with open(PATH) as f:
    code = f.read()

replacements = [
    # Model architecture
    ("num_local_experts: int = 256", "num_local_experts: int = 384"),
    ("vocab_size: int = 129280", "vocab_size: int = 163840"),
    ("num_attention_heads: int = 128", "num_attention_heads: int = 64"),
    ("num_key_value_heads: int = 128", "num_key_value_heads: int = 64"),
    ("n_group: int = 8", "n_group: int = 1"),
    ("routed_scaling_factor: float = 2.5", "routed_scaling_factor: float = 2.827"),
    ("first_k_dense_replace: int = 3", "first_k_dense_replace: int = 1"),
    # RoPE
    ("rope_theta = 10000", "rope_theta = 50000"),
    ('"beta_fast": 32,', '"beta_fast": 1.0,'),
    ('"factor": 40,', '"factor": 32.0,'),
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
    print(f"PATCHED deepseek_v3.py: {patched} values updated for Kimi K2-Instruct")
else:
    print("SKIP: no patterns found (already patched?)")
