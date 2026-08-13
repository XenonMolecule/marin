# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plain-text SFT on mathhelpforum Q/R/A extracted math content (no chat template).

Same as mathhelpforum_qra_sft.py but formats Q/R/A content as plain text
"Q: ... A: ..." instead of chat messages. Both training and evaluation use
plain text, avoiding the empty-generation issue caused by chat template tokens.

Requires mathhelpforum_extract_qra.py to have completed first.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/mathhelpforum_qra_sft_plaintext.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/mathhelpforum_qra_sft_plaintext.py --dry_run true
"""

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import timedelta

import fsspec
import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import lm_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_jsonl

from experiments.defaults import default_tokenize
from experiments.evals.evals import default_eval
from experiments.llama import compute_num_parameters
from experiments.rephraser.mathhelpforum_extract_qra import postprocess_step_qra
from experiments.rephraser.rephraser_cooldown import _read_token_count, scaling_1e20_qwen3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "gs://marin-us-central1/" "exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-44660"
)
LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
BATCH_SIZE = 64
SEQ_LEN = 4096


# ---------------------------------------------------------------------------
# Step 1: Parse Q/R/A markdown into plain text "Q: ... A: ..."
# ---------------------------------------------------------------------------
@dataclass
class QRAToPlainTextConfig:
    input_path: str
    output_path: str


def _parse_qra_to_plaintext(text: str) -> list[str]:
    """Parse ## Question / ## Reasoning / ## Answer markdown into plain text Q/A pairs.

    Returns a list of "Q: ...\\nA: ..." strings, one per question block.
    """
    if not text or len(text.strip()) < 20:
        return []

    # Extract title (first # heading)
    title = ""
    title_match = re.match(r"^#\s+(.+?)$", text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    # Split into question blocks (accept optional numbered headings)
    parts = re.split(r"(?=^## Question(?:\s+\d+)?\s*\n)", text, flags=re.MULTILINE)

    results = []
    for part in parts:
        part = part.strip()
        if not re.match(r"^## Question(?:\s+\d+)?\s*\n", part):
            continue

        # Extract question content
        q_match = re.search(
            r"^## Question(?:\s+\d+)?\s*\n(.*?)(?=^## (?:Reasoning|Answer)\b|\Z)",
            part,
            re.MULTILINE | re.DOTALL,
        )
        if not q_match:
            continue
        question = q_match.group(1).strip()
        if not question:
            continue

        # Extract reasoning
        r_match = re.search(
            r"^## Reasoning(?:\s+\d+)?\s*\n(.*?)(?=^## Answer\b|\Z)",
            part,
            re.MULTILINE | re.DOTALL,
        )
        reasoning = r_match.group(1).strip() if r_match else ""

        # Extract answer
        a_match = re.search(
            r"^## Answer(?:\s+\d+)?\s*\n(.*)\Z",
            part,
            re.MULTILINE | re.DOTALL,
        )
        answer = a_match.group(1).strip() if a_match else ""

        # Build the answer text from reasoning + answer
        answer_parts = []
        if reasoning:
            answer_parts.append(reasoning)
        if answer:
            if reasoning:
                answer_parts.append(f"The answer is {answer}.")
            else:
                answer_parts.append(answer)

        if not answer_parts:
            continue

        q_text = question
        if title and not results:
            q_text = f"{title}\n\n{q_text}"

        results.append(f"Q: {q_text}\nA: {' '.join(answer_parts)}")

    return results


def _transform_to_plaintext(records: list[dict]) -> list[dict]:
    """Transform extracted Q/R/A records into plain text format."""
    results = []
    for record in records:
        text = record.get("text", "")
        qa_pairs = _parse_qra_to_plaintext(text)
        for qa in qa_pairs:
            results.append({"text": qa})
    return results


def transform_qra_to_plaintext(config: QRAToPlainTextConfig):
    """Parse Q/R/A mathhelpforum markdown into plain text Q/A JSONL."""
    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .map_shard(_transform_to_plaintext)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="qra-to-plaintext") as ctx:
        output_files = ctx.execute(pipeline)

    stats = {"output_files": len(output_files), "input_path": config.input_path}
    with fsspec.open(f"{config.output_path}/transform_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Transform complete: {len(output_files)} output shards")


plaintext_transform_step = ExecutorStep(
    name="documents/mathhelpforum_qra_plaintext",
    description="Parse mathhelpforum Q/R/A markdown into plain text Q/A JSONL.",
    fn=transform_qra_to_plaintext,
    config=QRAToPlainTextConfig(
        input_path=postprocess_step_qra / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

# v2: Fixed multi-line answer extraction (greedy .* with \Z instead of non-greedy .*? with $)
plaintext_transform_step_v2 = ExecutorStep(
    name="documents/mathhelpforum_qra_plaintext_v2",
    description="Parse mathhelpforum Q/R/A markdown into plain text Q/A JSONL (fixed multi-line answer).",
    fn=transform_qra_to_plaintext,
    config=QRAToPlainTextConfig(
        input_path=postprocess_step_qra / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

# ---------------------------------------------------------------------------
# Step 2: Tokenize as plain text (no chat template)
# ---------------------------------------------------------------------------
qra_tokenized = default_tokenize(
    name="mathhelpforum_qra_plaintext_sft",
    dataset=plaintext_transform_step / "**/*.jsonl.gz",
    tokenizer=LLAMA3_TOKENIZER,
    format=TextLmDatasetFormat(),
)

# ---------------------------------------------------------------------------
# Step 3: Data config
# ---------------------------------------------------------------------------
data_config = lm_data_config(
    training_set=qra_tokenized,
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

    logger.info("=== Mathhelpforum Q/R/A Plain-Text Single-Epoch SFT ===")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Num train steps (1 epoch): {num_train_steps}")
    logger.info(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")

    inner_config = TrainLmConfig(
        data=config.data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=["mathhelpforum", "qra", "plaintext", "sft", "1e20", "qwen3", "single-epoch"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_train_steps,
            steps_per_eval=min(50, num_train_steps),
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[],
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


qra_sft_step = ExecutorStep(
    name="checkpoints/mathhelpforum-qra-plaintext-sft-1e20-qwen3",
    description=(
        f"Single-epoch plain-text SFT on mathhelpforum Q/R/A extracted content "
        f"({compute_num_parameters(scaling_1e20_qwen3, 128256):,} params)."
    ),
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=qra_tokenized,
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
    step=qra_sft_step,
    evals=GSM8K_EVAL,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[qra_sft_step, gsm8k_eval_step],
        description="Mathhelpforum Q/R/A single-epoch plain-text SFT + GSM8K eval (no chat template).",
    )
