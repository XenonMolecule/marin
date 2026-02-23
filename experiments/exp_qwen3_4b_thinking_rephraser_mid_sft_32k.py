# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT fine-tuning Qwen3-4B-Thinking-2507 on the rephraser distillation dataset (mid, 32K context).

Dataset: MichaelR207/rephraser_mid_check_0219
  - Train: 601,502 rows
  - Validation: 100 rows
  - Format: multi-turn chat (system/user/assistant)

Model: Qwen/Qwen3-4B-Thinking-2507 (loaded from HuggingFace Hub)
  - Same architecture as Qwen3-4B (head_dim=128 override required)
  - Uses rope_theta=5M (vs 1M for base Qwen3-4B) for 262K context support
TPU: v5p-8 (4 chips, 95 GB HBM each)

This is the 32K context variant. See exp_qwen3_4b_thinking_rephraser_mid_sft.py for the
131K version (requires v5p-64).
"""

import dataclasses
import math
import os

from experiments.chat_templates.qwen3_thinking_chat_template import QWEN_3_THINKING_CHAT_TEMPLATE
from experiments.defaults import default_sft
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.qwen3 import qwen3_4b_hd128
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.execution.executor import ExecutorStep, ensure_versioned, executor_main, this_output_path
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
from marin.transform.filter_by_context_length import FilterByContextLengthConfig, filter_by_context_length

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID = "MichaelR207/rephraser_mid_check_0219"
# The Thinking-2507 models use a different chat template than the base Qwen3 models,
# so we must tokenize with the Thinking-2507 tokenizer (not the shared 0.6B one).
QWEN3_TOKENIZER = "Qwen/Qwen3-4B-Thinking-2507"
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
MAX_SEQ_LEN = 32_768

NUM_TRAIN_EXAMPLES = 601_502
TARGET_EPOCHS = 1
TRAIN_BATCH_SIZE = 64
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * NUM_TRAIN_EXAMPLES / TRAIN_BATCH_SIZE)

# ---------------------------------------------------------------------------
# 1. Transform both splits (train + validation)
# ---------------------------------------------------------------------------
train_dataset = get_instruction_dataset(DATASET_ID, splits=["train"])
val_dataset = get_instruction_dataset(DATASET_ID, splits=["validation"])

CHAT_FORMAT = ChatLmDatasetFormat(chat_template=QWEN_3_THINKING_CHAT_TEMPLATE)

# ---------------------------------------------------------------------------
# 2. Filter training data to remove examples whose user prompt exceeds the
#    context window, leaving no assistant tokens for the model to learn from.
#    Parameterized by (tokenizer, seq_len) so models sharing these values
#    reuse the same filtered output.
# ---------------------------------------------------------------------------
filtered_train = ExecutorStep(
    name=os.path.join("filtered", f"rephraser_mid_check_0219_qwen3_thinking_{MAX_SEQ_LEN // 1024}k"),
    description=f"Filter examples with <64 assistant tokens within {MAX_SEQ_LEN} context.",
    fn=filter_by_context_length,
    config=FilterByContextLengthConfig(
        input_path=train_dataset / "**/*.jsonl.gz",
        output_path=this_output_path(),
        tokenizer=QWEN3_TOKENIZER,
        seq_len=MAX_SEQ_LEN,
        chat_template=QWEN_3_THINKING_CHAT_TEMPLATE,
        min_completion_tokens=64,
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
    pip_dependency_groups=["cpu"],
    env_vars={
        "TRANSFORMERS_NO_TORCH": "1",
        "TRANSFORMERS_NO_TORCHVISION": "1",
        "USE_TORCH": "0",
        "TORCH_DISABLE_GLOBAL_DEPS": "1",
    },
)

# ---------------------------------------------------------------------------
# 3. Tokenize filtered train + unfiltered validation
#    Uses window_size_bytes=1 so each JSONL file becomes its own shard (avoids
#    bundling all files into 1 shard, which causes a single-worker bottleneck).
#    Merging both splits into one step avoids Zephyr worker contention on
#    TPU-only clusters where CPU resources are scarce.
# ---------------------------------------------------------------------------
tokenized = ExecutorStep(
    name=os.path.join("tokenized", f"rephraser_mid_check_0219_qwen3_4b_thinking_filtered_{MAX_SEQ_LEN // 1024}k"),
    description=f"Tokenize filtered data using the {QWEN3_TOKENIZER} tokenizer.",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[filtered_train / "**/*.jsonl.gz"],
        validation_paths=[val_dataset / "**/*.jsonl.gz"],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(QWEN3_TOKENIZER),
        format=CHAT_FORMAT,
        window_size_bytes=1,  # 1 byte -> each file becomes its own shard
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="16g"),
    pip_dependency_groups=["cpu"],
    env_vars={
        "TRANSFORMERS_NO_TORCH": "1",
        "TRANSFORMERS_NO_TORCHVISION": "1",
        "USE_TORCH": "0",
        "TORCH_DISABLE_GLOBAL_DEPS": "1",
    },
)

# ---------------------------------------------------------------------------
# 4. Data config
# ---------------------------------------------------------------------------
data_config = lm_data_config(
    training_set=tokenized,
    validation_sets={"rephraser_val": tokenized},
)

# ---------------------------------------------------------------------------
# 5. Model config
#    The base qwen3_4b_hd128 uses Llama3RotaryEmbeddingsConfig (theta=500K)
#    for from-scratch pretraining. The Thinking-2507 variant uses theta=5M
#    (higher than the base 4B's theta=1M) to support 262K context.
#    head_dim=128 is required because 2560/32 = 80, mismatching the HF checkpoint.
# ---------------------------------------------------------------------------
qwen3_model_config = dataclasses.replace(
    qwen3_4b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=5000000.0, factor=1.0),
    max_seq_len=MAX_SEQ_LEN,
)

# ---------------------------------------------------------------------------
# 6. SFT training config
# ---------------------------------------------------------------------------
sft_config = SimpleSFTConfig(
    # Hardware -- v5p-8 (4 chips x 95 GB HBM), fits at 32K context
    resources=ResourceConfig.with_tpu("v5p-8"),
    # Training
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    # Optimizer (slightly lower LR for 4B model)
    learning_rate=1e-5,
    lr_schedule="cosine",
    warmup=0.03,
    decay=0.97,  # fraction of steps for cosine decay (1.0 - warmup)
    weight_decay=0.01,
    max_grad_norm=1.0,
    # Model
    tokenizer=MODEL_ID,
    initialize_from_hf=MODEL_ID,
    pad_tokenizer_to_match_model=True,
    max_seq_len=MAX_SEQ_LEN,
    # Checkpointing & eval
    steps_per_eval=100,
    steps_per_checkpoint=250,
    steps_per_hf_export=250,
    seed=42,
    # Stability: z_loss penalizes large logits; skip_bad_steps skips anomalous batches.
    z_loss_weight=1e-5,
    skip_bad_steps=True,
    # Gradient accumulation: microbatch=4 (1 per device x 4 chips), 16 accum steps.
    # Logits tensor per device: 1 x 32768 x 151936 x 4B = 19 GB, fits in 95 GB v5p HBM.
    per_device_parallelism=1,
)

# ---------------------------------------------------------------------------
# 7. Create the training ExecutorStep
# ---------------------------------------------------------------------------
qwen3_4b_thinking_rephraser_mid_sft_32k_v1 = default_sft(
    name="qwen3-4b-thinking-rephraser-mid-sft-32k-v1",
    tokenized=data_config,
    model_config=qwen3_model_config,
    sft_config=sft_config,
    tags=["qwen3", "4b", "thinking", "sft", "rephraser", "mid", "32k"],
)

# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(steps=[qwen3_4b_thinking_rephraser_mid_sft_32k_v1])
