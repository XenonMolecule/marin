# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter grid sweep for Qwen3-1.7B SFT on the small rephraser dataset.

Uses the small dataset (84,389 train examples, ~1,319 steps at bs=64) for fast
iteration. Best configs can be scaled to the mid dataset afterward.

Grid: 3 learning rates x 2 weight decays x 2 beta2 values = 12 runs.
Existing baseline (lr=2e-5, wd=0.01, beta2=0.95) was run separately as
qwen3-1.7b-rephraser-sft-v4, giving 13 data points total.

Dataset: MichaelR207/rephraser_small_check_0213
  - Train: 84,389 rows
  - Validation: 100 rows

Model: Qwen/Qwen3-1.7B (loaded from HuggingFace Hub)
TPU: v5p-8 (4 chips, 95 GB HBM each) per run

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        --env_vars WANDB_API_KEY=${WANDB_API_KEY} \\
        -- python experiments/exp_qwen3_1_7b_rephraser_sweep.py --max_concurrent 12
"""

import dataclasses
import itertools
import math
import os

from fray.cluster import ResourceConfig
from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.execution.executor import ExecutorStep, ensure_versioned, executor_main, this_output_path
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
from marin.transform.filter_by_context_length import FilterByContextLengthConfig, filter_by_context_length

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.defaults import default_sft
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.qwen3 import qwen3_1_7b
from experiments.simple_sft_config import SimpleSFTConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID = "MichaelR207/rephraser_small_check_0213"
QWEN3_TOKENIZER = "Qwen/Qwen3-0.6B"
MODEL_ID = "Qwen/Qwen3-1.7B"
MAX_SEQ_LEN = 32_768

NUM_TRAIN_EXAMPLES = 84_389
TARGET_EPOCHS = 1
TRAIN_BATCH_SIZE = 64
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * NUM_TRAIN_EXAMPLES / TRAIN_BATCH_SIZE)

# ---------------------------------------------------------------------------
# Sweep grid
#
# Levanter's AdamConfig defaults to beta2=0.95, so None here means 0.95.
# ---------------------------------------------------------------------------
LEARNING_RATES = [1e-5, 2e-5, 5e-5]
WEIGHT_DECAYS = [0.005, 0.02]
BETA2_VALUES: list[float | None] = [None, 0.99]  # None = 0.95 (AdamConfig default)

# ---------------------------------------------------------------------------
# 1. Transform both splits (train + validation)
# ---------------------------------------------------------------------------
train_dataset = get_instruction_dataset(DATASET_ID, splits=["train"])
val_dataset = get_instruction_dataset(DATASET_ID, splits=["validation"])

CHAT_FORMAT = ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE)

# ---------------------------------------------------------------------------
# 2. Filter training data
#    Reuses the same step name as exp_qwen3_1_7b_rephraser_sft.py so the
#    executor skips it if already run.
# ---------------------------------------------------------------------------
filtered_train = ExecutorStep(
    name=os.path.join("filtered", f"rephraser_small_check_0213_qwen3_{MAX_SEQ_LEN // 1024}k"),
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
#    New step name (no pack=1) so it doesn't collide with the existing small
#    experiment's tokenized data.
# ---------------------------------------------------------------------------
tokenized = ExecutorStep(
    name=os.path.join("tokenized", f"rephraser_small_check_0213_qwen3_sweep_filtered_{MAX_SEQ_LEN // 1024}k"),
    description=f"Tokenize filtered data using the {QWEN3_TOKENIZER} tokenizer.",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[filtered_train / "**/*.jsonl.gz"],
        validation_paths=[val_dataset / "**/*.jsonl.gz"],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(QWEN3_TOKENIZER),
        format=CHAT_FORMAT,
        window_size_bytes=1,
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
# ---------------------------------------------------------------------------
qwen3_model_config = dataclasses.replace(
    qwen3_1_7b,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=MAX_SEQ_LEN,
)

# ---------------------------------------------------------------------------
# 6. Generate sweep training steps
# ---------------------------------------------------------------------------


def _beta2_label(beta2: float | None) -> str:
    """Human-readable label for beta2 (None = Levanter default of 0.95)."""
    return "0.95" if beta2 is None else str(beta2)


all_sft_steps: list[ExecutorStep] = []

for lr, wd, beta2 in itertools.product(LEARNING_RATES, WEIGHT_DECAYS, BETA2_VALUES):
    label = f"lr{lr:.0e}_wd{wd}_b2-{_beta2_label(beta2)}"
    name = f"qwen3-1.7b-rephraser-sweep/{label}"

    sft_config = SimpleSFTConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=lr,
        lr_schedule="cosine",
        warmup=0.03,
        decay=0.97,
        weight_decay=wd,
        beta2=beta2,
        max_grad_norm=1.0,
        tokenizer=MODEL_ID,
        initialize_from_hf=MODEL_ID,
        pad_tokenizer_to_match_model=True,
        max_seq_len=MAX_SEQ_LEN,
        steps_per_eval=100,
        steps_per_checkpoint=250,
        steps_per_hf_export=250,
        seed=42,
        z_loss_weight=1e-5,
        skip_bad_steps=True,
        per_device_parallelism=2,
    )

    step = default_sft(
        name=name,
        tokenized=data_config,
        model_config=qwen3_model_config,
        sft_config=sft_config,
        tags=["qwen3", "1.7b", "sft", "rephraser", "sweep", f"lr={lr}", f"wd={wd}", f"b2={_beta2_label(beta2)}"],
    )
    all_sft_steps.append(step)

# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(steps=all_sft_steps)
