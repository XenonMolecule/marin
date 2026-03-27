# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continued pretraining on mathhelpforum.com plain text extracted via resiliparse.

Baseline comparison for mathhelpforum_sft.py: instead of using rephraser-extracted
structured markdown, we extract plain text from the same HTML using resiliparse
(the standard extraction method used by DCLM). No chat formatting — just continued
pretraining on the raw extracted text for 1 epoch, then evaluate on GSM8K.

Pipeline:
  consolidated HTML -> resiliparse text extraction -> tokenize (plain text) -> train -> GSM8K eval

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/mathhelpforum_resiliparse_sft.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/mathhelpforum_resiliparse_sft.py --dry_run true
"""

import logging
import math
from dataclasses import dataclass
from datetime import timedelta

import jmp

from experiments.defaults import default_tokenize
from experiments.evals.evals import default_eval
from experiments.llama import compute_num_parameters
from experiments.rephraser.mathhelpforum_extract import consolidate_step
from experiments.rephraser.rephraser_cooldown import _read_token_count, scaling_1e20_qwen3
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import lm_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.transform.extract_text_from_html import ExtractTextConfig, extract_text_from_html

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (shared across GSM8K / mathhelpforum / resiliparse SFT experiments)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "gs://marin-us-central1/" "exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-44660"
)
LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
BATCH_SIZE = 64
SEQ_LEN = 4096

# ---------------------------------------------------------------------------
# Step 1: Extract plain text from HTML using resiliparse
# ---------------------------------------------------------------------------
extract_text_step = ExecutorStep(
    name="processed/mathhelpforum_resiliparse_text",
    description="Extract plain text from mathhelpforum HTML using resiliparse.",
    fn=extract_text_from_html,
    config=ExtractTextConfig(
        input_path=consolidate_step / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
    pip_dependency_groups=["cpu"],
)

# ---------------------------------------------------------------------------
# Step 2: Tokenize as plain text (continued pretraining, not chat format)
# ---------------------------------------------------------------------------
resiliparse_tokenized = default_tokenize(
    name="mathhelpforum_resiliparse_sft",
    dataset=extract_text_step / "**/*.jsonl.gz",
    tokenizer=LLAMA3_TOKENIZER,
    format=TextLmDatasetFormat(),
)

# ---------------------------------------------------------------------------
# Step 3: Data config
# ---------------------------------------------------------------------------
data_config = lm_data_config(
    training_set=resiliparse_tokenized,
    validation_sets={},
)


# ---------------------------------------------------------------------------
# Step 4: Single-epoch continued pretraining (num_train_steps computed at runtime)
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

    logger.info("=== Mathhelpforum Resiliparse Single-Epoch Continued Pretraining ===")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Num train steps (1 epoch): {num_train_steps}")
    logger.info(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")

    inner_config = TrainLmConfig(
        data=config.data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=["mathhelpforum", "resiliparse", "continued-pretraining", "1e20", "qwen3", "single-epoch"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_train_steps,
            steps_per_eval=min(50, num_train_steps),
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=min(100, num_train_steps))],
            ),
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
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


resiliparse_sft_step = ExecutorStep(
    name="checkpoints/mathhelpforum-resiliparse-sft-1e20-qwen3",
    description=(
        f"Single-epoch continued pretraining on mathhelpforum resiliparse text "
        f"({compute_num_parameters(scaling_1e20_qwen3, 128256):,} params)."
    ),
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=resiliparse_tokenized,
        data_config=data_config,
        output_path=this_output_path(),
    ),
)

# ---------------------------------------------------------------------------
# Step 5: Evaluate on GSM8K test set (8-shot CoT)
# ---------------------------------------------------------------------------
GSM8K_EVAL = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
]

gsm8k_eval_step = default_eval(
    step=resiliparse_sft_step,
    evals=GSM8K_EVAL,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[resiliparse_sft_step, gsm8k_eval_step],
        description="Mathhelpforum resiliparse single-epoch continued pretraining + GSM8K eval.",
    )
