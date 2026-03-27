# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plain-text SFT on GSM8K train set (no chat template).

Same as gsm8k_sft.py but uses plain text "Q: ... A: ..." format for both
training and evaluation. This avoids the chat template issue where the model
generates empty output because it doesn't understand chat special tokens.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/gsm8k_sft_plaintext.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/gsm8k_sft_plaintext.py --dry_run true
"""

import json
import logging
import math
from dataclasses import dataclass
from datetime import timedelta

import fsspec
import jmp
from zephyr import Dataset, ZephyrContext, load_jsonl

from experiments.defaults import default_tokenize
from experiments.evals.evals import default_eval
from experiments.llama import compute_num_parameters
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.rephraser.rephraser_cooldown import _read_token_count, scaling_1e20_qwen3
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import lm_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# TODO: migrate to InputName.hardcoded() — see experiments/AGENTS.md. Deferred to avoid hash invalidation.
CHECKPOINT_PATH = (
    "gs://marin-us-central1/" "exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-44660"
)
LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
BATCH_SIZE = 64
SEQ_LEN = 4096


# ---------------------------------------------------------------------------
# Step 1: Get GSM8K and convert chat messages to plain text "Q: ... A: ..."
# ---------------------------------------------------------------------------
gsm8k_train = get_instruction_dataset("openai/gsm8k", splits=["train"])


@dataclass
class MessagesToPlainTextConfig:
    input_path: str
    output_path: str


def _messages_to_plaintext(records: list[dict]) -> list[dict]:
    """Convert {"messages": [...]} chat records to {"text": "Q: ... A: ..."}."""
    results = []
    for record in records:
        messages = record.get("messages", [])
        if len(messages) < 2:
            continue
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        if user_msg and asst_msg:
            results.append({"text": f"Q: {user_msg}\nA: {asst_msg}"})
    return results


def transform_messages_to_plaintext(config: MessagesToPlainTextConfig):
    """Convert chat messages JSONL to plain text JSONL."""
    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .map_shard(_messages_to_plaintext)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="messages-to-plaintext") as ctx:
        output_files = ctx.execute(pipeline)

    stats = {"output_files": len(output_files), "input_path": config.input_path}
    with fsspec.open(f"{config.output_path}/transform_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Transform complete: {len(output_files)} output shards")


plaintext_transform_step = ExecutorStep(
    name="documents/gsm8k_plaintext",
    description="Convert GSM8K chat messages to plain text Q/A format.",
    fn=transform_messages_to_plaintext,
    config=MessagesToPlainTextConfig(
        input_path=gsm8k_train / "**/*.jsonl.gz",
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
    pip_dependency_groups=["cpu"],
)

# ---------------------------------------------------------------------------
# Step 2: Tokenize as plain text (no chat template)
# ---------------------------------------------------------------------------
gsm8k_tokenized = default_tokenize(
    name="gsm8k_plaintext_sft",
    dataset=plaintext_transform_step / "**/*.jsonl.gz",
    tokenizer=LLAMA3_TOKENIZER,
    format=TextLmDatasetFormat(),
)

# ---------------------------------------------------------------------------
# Step 3: Data config
# ---------------------------------------------------------------------------
data_config = lm_data_config(
    training_set=gsm8k_tokenized,
    validation_sets={},
)


# ---------------------------------------------------------------------------
# Step 4: Single-epoch plain-text SFT
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SFTRunConfig:
    tokenized_path: str
    data_config: LMMixtureDatasetConfig
    output_path: str


def _run_single_epoch_sft(config: _SFTRunConfig):
    """Train for exactly 1 epoch, computing num_train_steps from the tokenized cache."""
    total_tokens = _read_token_count(config.tokenized_path, split="train")
    num_train_steps = math.ceil(total_tokens / (BATCH_SIZE * SEQ_LEN))

    logger.info("=== GSM8K Plain-Text Single-Epoch SFT ===")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Num train steps (1 epoch): {num_train_steps}")
    logger.info(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")

    inner_config = TrainLmConfig(
        data=config.data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=["gsm8k", "plaintext", "sft", "1e20", "qwen3", "single-epoch"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_train_steps,
            steps_per_eval=min(50, num_train_steps),
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=min(100, num_train_steps))],
            ),
            allow_nondivisible_batch_size=True,
            initialize_from=None,
        ),
        initialize_from_checkpoint_path=CHECKPOINT_PATH,
        initialize_from_hf=False,
        train_seq_len=SEQ_LEN,
        model=scaling_1e20_qwen3,
        optimizer=AdamConfig(
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup=0.03,
            decay=0.97,
            lr_schedule="cosine",
            max_grad_norm=1.0,
        ),
        hf_save_steps=num_train_steps,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu("v5p-8"),
        output_path=config.output_path,
    )

    run_levanter_train_lm(pod_config)


gsm8k_sft_step = ExecutorStep(
    name="checkpoints/gsm8k-plaintext-sft-1e20-qwen3",
    description=(
        f"Single-epoch plain-text SFT on GSM8K train set "
        f"({compute_num_parameters(scaling_1e20_qwen3, 128256):,} params)."
    ),
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=gsm8k_tokenized,
        data_config=data_config,
        output_path=this_output_path(),
    ),
)

# ---------------------------------------------------------------------------
# Step 5: Evaluate on GSM8K test set (8-shot CoT, plain text — no chat template)
# ---------------------------------------------------------------------------
GSM8K_EVAL = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
]

gsm8k_eval_step = default_eval(
    step=gsm8k_sft_step,
    evals=GSM8K_EVAL,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[gsm8k_sft_step, gsm8k_eval_step],
        description="GSM8K single-epoch plain-text SFT + GSM8K eval (no chat template).",
    )
