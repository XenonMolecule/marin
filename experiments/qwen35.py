# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5 model configurations for SFT experiments.

All values match the HuggingFace config.json for each model size.
"""

from levanter.layers.rotary import PartialRotaryEmbeddingsConfig
from levanter.models.qwen35 import Qwen35Config

_ROPE = PartialRotaryEmbeddingsConfig(theta=10_000_000.0, partial_rotary_factor=0.25)

QWEN35_TOKENIZER = "Qwen/Qwen3.5-0.8B"

qwen35_0_8b = Qwen35Config(
    max_seq_len=32768,
    hidden_dim=1024,
    intermediate_dim=3584,
    num_layers=24,
    num_heads=8,
    num_kv_heads=2,
    head_dim=256,
    linear_num_key_heads=16,
    linear_num_value_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    tie_word_embeddings=True,
    reference_checkpoint="Qwen/Qwen3.5-0.8B",
    rope=_ROPE,
)

qwen35_2b = Qwen35Config(
    max_seq_len=32768,
    hidden_dim=2048,
    intermediate_dim=6144,
    num_layers=24,
    num_heads=8,
    num_kv_heads=2,
    head_dim=256,
    linear_num_key_heads=16,
    linear_num_value_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    tie_word_embeddings=True,
    reference_checkpoint="Qwen/Qwen3.5-2B",
    rope=_ROPE,
)

qwen35_4b = Qwen35Config(
    max_seq_len=32768,
    hidden_dim=2560,
    intermediate_dim=9216,
    num_layers=32,
    num_heads=16,
    num_kv_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    tie_word_embeddings=True,
    reference_checkpoint="Qwen/Qwen3.5-4B",
    rope=_ROPE,
)

qwen35_9b = Qwen35Config(
    max_seq_len=32768,
    hidden_dim=4096,
    intermediate_dim=12288,
    num_layers=32,
    num_heads=16,
    num_kv_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    tie_word_embeddings=False,
    reference_checkpoint="Qwen/Qwen3.5-9B",
    rope=_ROPE,
)
