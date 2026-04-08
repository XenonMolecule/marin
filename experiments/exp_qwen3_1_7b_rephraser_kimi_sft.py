# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT fine-tuning Qwen3-1.7B on the Kimi-distilled rephraser dataset.

Two variants: think (with <think> traces) and stripped (without).

Dataset: MichaelR207/rephraser_kimi_v1_0331 (57,799 train / 100 val)
Model: Qwen/Qwen3-1.7B
TPU: v5p-8 (4 chips)
"""

import dataclasses
import os

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.defaults import single_epoch_sft
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.qwen3 import qwen3_1_7b
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from levanter.data.text import ChatLmDatasetFormat
from marin.execution.executor import ExecutorStep, ensure_versioned, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
from marin.transform.filter_by_context_length import FilterByContextLengthConfig, filter_by_context_length
from marin.transform.strip_thinking import StripThinkingConfig, strip_thinking

DATASET_ID = "MichaelR207/rephraser_kimi_v1_0331"
MODEL_ID = "Qwen/Qwen3-1.7B"
TOKENIZER = "Qwen/Qwen3-1.7B"
MAX_SEQ_LEN = 32_768
TRAIN_BATCH_SIZE = 64

CHAT_FORMAT = ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE, pack=1)

_CPU_ENV = {
    "TRANSFORMERS_NO_TORCH": "1",
    "TRANSFORMERS_NO_TORCHVISION": "1",
    "USE_TORCH": "0",
    "TORCH_DISABLE_GLOBAL_DEPS": "1",
}

# --- Download ---
train_dataset = get_instruction_dataset(DATASET_ID, splits=["train"])
val_dataset = get_instruction_dataset(DATASET_ID, splits=["validation"])

# --- Strip thinking ---
stripped_train = ExecutorStep(
    name=os.path.join("stripped", "rephraser_kimi_v1_0331_no_think"),
    description="Strip <think> blocks from assistant messages.",
    fn=remote(
        strip_thinking,
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu"],
        env_vars=_CPU_ENV,
    ),
    config=StripThinkingConfig(
        input_path=train_dataset / "**/*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

# --- Filter ---
filtered_think = ExecutorStep(
    name=os.path.join("filtered", f"rephraser_kimi_v1_think_qwen3_{MAX_SEQ_LEN // 1024}k"),
    description=f"Filter think examples with <64 assistant tokens within {MAX_SEQ_LEN} context.",
    fn=remote(
        filter_by_context_length,
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu"],
        env_vars=_CPU_ENV,
    ),
    config=FilterByContextLengthConfig(
        input_path=train_dataset / "**/*.jsonl.gz",
        output_path=this_output_path(),
        tokenizer=TOKENIZER,
        seq_len=MAX_SEQ_LEN,
        chat_template=QWEN_3_CHAT_TEMPLATE,
        min_completion_tokens=64,
    ),
)

filtered_stripped = ExecutorStep(
    name=os.path.join("filtered", f"rephraser_kimi_v1_stripped_qwen3_{MAX_SEQ_LEN // 1024}k"),
    description=f"Filter stripped examples with <64 assistant tokens within {MAX_SEQ_LEN} context.",
    fn=remote(
        filter_by_context_length,
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu"],
        env_vars=_CPU_ENV,
    ),
    config=FilterByContextLengthConfig(
        input_path=stripped_train / "**/*.jsonl.gz",
        output_path=this_output_path(),
        tokenizer=TOKENIZER,
        seq_len=MAX_SEQ_LEN,
        chat_template=QWEN_3_CHAT_TEMPLATE,
        # No min_completion_tokens for stripped: short outputs like [NO_USEFUL_CONTENT]
        # are valid training signal (teaches the model when NOT to extract).
        min_completion_tokens=0,
    ),
)

# --- Tokenize ---
tokenized_think = ExecutorStep(
    name=os.path.join("tokenized", f"rephraser_kimi_v1_think_qwen3_{MAX_SEQ_LEN // 1024}k"),
    description=f"Tokenize think data using the {TOKENIZER} tokenizer.",
    fn=remote(
        tokenize,
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="16g"),
        pip_dependency_groups=["cpu"],
        env_vars=_CPU_ENV,
    ),
    config=TokenizeConfig(
        train_paths=[filtered_think / "**/*.jsonl.gz"],
        validation_paths=[val_dataset / "**/*.jsonl.gz"],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(TOKENIZER),
        format=CHAT_FORMAT,
    ),
)

tokenized_stripped = ExecutorStep(
    name=os.path.join("tokenized", f"rephraser_kimi_v1_stripped_qwen3_{MAX_SEQ_LEN // 1024}k"),
    description=f"Tokenize stripped data using the {TOKENIZER} tokenizer.",
    fn=remote(
        tokenize,
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="16g"),
        pip_dependency_groups=["cpu"],
        env_vars=_CPU_ENV,
    ),
    config=TokenizeConfig(
        train_paths=[filtered_stripped / "**/*.jsonl.gz"],
        validation_paths=[val_dataset / "**/*.jsonl.gz"],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(TOKENIZER),
        format=CHAT_FORMAT,
    ),
)

# --- Data configs ---
data_think = lm_data_config(training_set=tokenized_think, validation_sets={"rephraser_val": tokenized_think})
data_stripped = lm_data_config(training_set=tokenized_stripped, validation_sets={"rephraser_val": tokenized_stripped})

# --- SFT config ---
sft_config = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=TRAIN_BATCH_SIZE,
    learning_rate=2e-5,
    lr_schedule="cosine",
    warmup=0.03,
    decay=0.97,
    weight_decay=0.01,
    max_grad_norm=1.0,
    tokenizer=MODEL_ID,
    pad_tokenizer_to_match_model=True,
    initialize_from_hf=MODEL_ID,
    max_seq_len=MAX_SEQ_LEN,
    steps_per_eval=100,
    steps_per_checkpoint=250,
    steps_per_hf_export=250,
    seed=42,
    z_loss_weight=1e-5,
    per_device_parallelism=1,
)

# --- Training (num_train_steps auto-computed for 1 epoch) ---
qwen3_1_7b_think = single_epoch_sft(
    name="qwen3-1.7b-rephraser-kimi-think-sft",
    tokenized=data_think,
    model_config=dataclasses.replace(
        qwen3_1_7b,
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
        max_seq_len=MAX_SEQ_LEN,
    ),
    sft_config=sft_config,
    tags=["qwen3", "1.7b", "sft", "rephraser", "kimi", "think"],
)

qwen3_1_7b_stripped = single_epoch_sft(
    name="qwen3-1.7b-rephraser-kimi-stripped-sft",
    tokenized=data_stripped,
    model_config=dataclasses.replace(
        qwen3_1_7b,
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
        max_seq_len=MAX_SEQ_LEN,
    ),
    sft_config=sft_config,
    tags=["qwen3", "1.7b", "sft", "rephraser", "kimi", "stripped"],
)

# Launch via Iris:
# uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
#   --extra marin:tpu --tpu v5litepod-8 \
#   -e WANDB_API_KEY $WANDB_API_KEY -e HF_TOKEN $HF_TOKEN \
#   -- python experiments/exp_qwen3_1_7b_rephraser_kimi_sft.py
if __name__ == "__main__":
    executor_main(steps=[qwen3_1_7b_think, qwen3_1_7b_stripped])
