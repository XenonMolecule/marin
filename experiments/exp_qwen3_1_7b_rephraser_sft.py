# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT fine-tuning Qwen3-1.7B on the rephraser distillation dataset.

Dataset: MichaelR207/rephraser_small_check_0213
  - Train: 84,389 rows
  - Validation: 100 rows
  - Format: multi-turn chat (system/user/assistant)

Model: Qwen/Qwen3-1.7B (loaded from HuggingFace Hub)
TPU: v5p-8 (4 chips, 95 GB HBM each)
"""

import dataclasses
import math
import os

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.defaults import default_sft
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.qwen3 import qwen3_1_7b
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.execution.executor import ExecutorStep, ensure_versioned, executor_main, this_output_path
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID = "MichaelR207/rephraser_small_check_0213"
QWEN3_TOKENIZER = "Qwen/Qwen3-1.7B"
MAX_SEQ_LEN = 32_768

NUM_TRAIN_EXAMPLES = 84_389
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
# 2. Single tokenize step for both train + validation
#    Uses window_size_bytes=1 so each JSONL file becomes its own shard (avoids
#    bundling all files into 1 shard, which causes a single-worker bottleneck).
#    Merging both splits into one step avoids Zephyr worker contention on
#    TPU-only clusters where CPU resources are scarce.
# ---------------------------------------------------------------------------
tokenized = ExecutorStep(
    name=os.path.join("tokenized", "rephraser_small_check_0213_qwen3_1_7b"),
    description=f"Tokenize raw text using the {QWEN3_TOKENIZER} tokenizer.",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[train_dataset / "**/*.jsonl.gz"],
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
# 3. Data config
# ---------------------------------------------------------------------------
data_config = lm_data_config(
    training_set=tokenized,
    validation_sets={"rephraser_val": tokenized},
)

# ---------------------------------------------------------------------------
# 4. Model config
#    The base qwen3_1_7b uses Llama3RotaryEmbeddingsConfig (theta=500K) which
#    is for from-scratch pretraining. For SFT from HF weights we need
#    theta=1M to match the pretrained RoPE.
#    head_dim = 2048/16 = 128, naturally correct (no explicit override needed).
# ---------------------------------------------------------------------------
qwen3_model_config = dataclasses.replace(
    qwen3_1_7b,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=MAX_SEQ_LEN,
)

# ---------------------------------------------------------------------------
# 5. SFT training config
# ---------------------------------------------------------------------------
sft_config = SimpleSFTConfig(
    # Hardware -- v5p-8 (4 chips x 95 GB HBM)
    resources=ResourceConfig.with_tpu("v5p-8"),
    # Training
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    # Optimizer
    learning_rate=2e-5,
    lr_schedule="cosine",
    warmup=0.03,
    decay=0.97,  # fraction of steps for cosine decay (1.0 - warmup)
    weight_decay=0.01,
    max_grad_norm=1.0,
    # Model
    tokenizer=QWEN3_TOKENIZER,
    initialize_from_hf="Qwen/Qwen3-1.7B",
    pad_tokenizer_to_match_model=True,
    max_seq_len=MAX_SEQ_LEN,
    # Checkpointing & eval
    steps_per_eval=100,
    steps_per_checkpoint=250,
    steps_per_hf_export=250,
    seed=42,
    # Gradient accumulation: microbatch=8 (2 per device x 4 chips), 8 accum steps.
    # Logits tensor per device: 2 x 32768 x 151936 x 4B = 40 GB, fits in 95 GB v5p HBM.
    per_device_parallelism=2,
)

# ---------------------------------------------------------------------------
# 6. Create the training ExecutorStep
# ---------------------------------------------------------------------------
qwen3_1_7b_rephraser_sft = default_sft(
    name="qwen3-1.7b-rephraser-sft",
    tokenized=data_config,
    model_config=qwen3_model_config,
    sft_config=sft_config,
    tags=["qwen3", "1.7b", "sft", "rephraser"],
)

# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(steps=[qwen3_1_7b_rephraser_sft])
