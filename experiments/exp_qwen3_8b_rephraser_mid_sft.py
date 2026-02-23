# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT fine-tuning Qwen3-8B on the rephraser distillation dataset (mid).

Dataset: MichaelR207/rephraser_mid_check_0219
  - Train: 601,502 rows
  - Validation: 100 rows
  - Format: multi-turn chat (system/user/assistant)

Model: Qwen/Qwen3-8B (loaded from HuggingFace Hub)
TPU: v5p-16 (8 chips, 95 GB HBM each)
"""

import math
import os

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.defaults import default_sft
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.qwen3 import qwen3_8b
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from levanter.data.text import ChatLmDatasetFormat
from marin.execution.executor import ExecutorStep, ensure_versioned, executor_main, this_output_path
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
from marin.transform.filter_by_context_length import FilterByContextLengthConfig, filter_by_context_length

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID = "MichaelR207/rephraser_mid_check_0219"
# All Qwen3 models share the same tokenizer; reuse the 0.6B tokenize step to avoid
# redundant tokenization across model sizes.
QWEN3_TOKENIZER = "Qwen/Qwen3-0.6B"
MODEL_ID = "Qwen/Qwen3-8B"
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

CHAT_FORMAT = ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE)

# ---------------------------------------------------------------------------
# 2. Filter training data to remove examples whose user prompt exceeds the
#    context window, leaving no assistant tokens for the model to learn from.
#    Parameterized by (tokenizer, seq_len) so models sharing these values
#    reuse the same filtered output.
# ---------------------------------------------------------------------------
filtered_train = ExecutorStep(
    name=os.path.join("filtered", f"rephraser_mid_check_0219_qwen3_{MAX_SEQ_LEN // 1024}k"),
    description=f"Filter examples with <64 assistant tokens within {MAX_SEQ_LEN} context.",
    fn=filter_by_context_length,
    config=FilterByContextLengthConfig(
        input_path=train_dataset / "**/*.jsonl.gz",
        output_path=this_output_path(),
        tokenizer=QWEN3_TOKENIZER,
        seq_len=MAX_SEQ_LEN,
        chat_template=QWEN_3_CHAT_TEMPLATE,
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
    name=os.path.join("tokenized", f"rephraser_mid_check_0219_qwen3_filtered_{MAX_SEQ_LEN // 1024}k"),
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
#    qwen3_8b already uses DefaultRotaryEmbeddingsConfig(theta=1M) and
#    max_seq_len=32768, matching the HF checkpoint. No replace needed.
#    head_dim = 4096/32 = 128, naturally correct.
# ---------------------------------------------------------------------------
qwen3_model_config = qwen3_8b

# ---------------------------------------------------------------------------
# 6. SFT training config
# ---------------------------------------------------------------------------
sft_config = SimpleSFTConfig(
    # Hardware -- v5p-16 (8 chips x 95 GB HBM)
    resources=ResourceConfig.with_tpu("v5p-16"),
    # Training
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    # Optimizer (lower LR for 8B model)
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
    # Gradient accumulation: microbatch=8 (1 per device x 8 chips), 8 accum steps.
    # Logits tensor per device: 1 x 32768 x 151936 x 4B = 20 GB, fits in 95 GB v5p HBM.
    per_device_parallelism=1,
)

# ---------------------------------------------------------------------------
# 7. Create the training ExecutorStep
# ---------------------------------------------------------------------------
qwen3_8b_rephraser_mid_sft_v1 = default_sft(
    name="qwen3-8b-rephraser-mid-sft-v1",
    tokenized=data_config,
    model_config=qwen3_model_config,
    sft_config=sft_config,
    tags=["qwen3", "8b", "sft", "rephraser", "mid"],
)

# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(steps=[qwen3_8b_rephraser_mid_sft_v1])
