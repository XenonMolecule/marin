# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT fine-tuning on GSM8K train set using the exp2166 final checkpoint.

Takes the final checkpoint from the exp2166 nemotron 1e20 scaling ladder run
(~1.385B Qwen3 model pretrained with Llama3 tokenizer) and fine-tunes it on
the GSM8K train set (7,473 question/answer pairs with chain-of-thought reasoning)
for exactly 1 epoch. Evaluates on GSM8K test set using 8-shot CoT prompting.

Source run: exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0
  - Model: Qwen3Config(hidden=1792, layers=18, heads=14, intermediate=7168) ~1.385B params
  - Tokenizer: meta-llama/Meta-Llama-3.1-8B (Llama3)
  - Final checkpoint: step-44660

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/gsm8k_sft.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/gsm8k_sft.py --dry_run true
"""

import logging
import math
from dataclasses import dataclass
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import ChatLmDatasetFormat, LMMixtureDatasetConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import lm_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.chat_templates.llama3pt1_chat_template import LLAMA_3_1_CHAT_TEMPLATE
from experiments.defaults import default_tokenize
from experiments.evals.evals import default_eval
from experiments.llama import compute_num_parameters
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.rephraser.rephraser_cooldown import _read_token_count, scaling_1e20_qwen3

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
# 1. Transform GSM8K train split to chat format
# ---------------------------------------------------------------------------
gsm8k_train = get_instruction_dataset("openai/gsm8k", splits=["train"])

# ---------------------------------------------------------------------------
# 2. Tokenize with Llama3 chat template (matching pretraining tokenizer)
# ---------------------------------------------------------------------------
CHAT_FORMAT = ChatLmDatasetFormat(
    chat_template=LLAMA_3_1_CHAT_TEMPLATE,
    mask_user_turns=True,
    pack=True,
)

gsm8k_tokenized = default_tokenize(
    name="gsm8k_llama3_sft",
    dataset=gsm8k_train / "**/*.jsonl.gz",
    tokenizer=LLAMA3_TOKENIZER,
    format=CHAT_FORMAT,
)

# ---------------------------------------------------------------------------
# 3. Data config (single dataset, no validation set)
# ---------------------------------------------------------------------------
data_config = lm_data_config(
    training_set=gsm8k_tokenized,
    validation_sets={},
)


# ---------------------------------------------------------------------------
# 4. Single-epoch SFT training (num_train_steps computed at runtime)
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

    logger.info("=== GSM8K Single-Epoch SFT ===")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Num train steps (1 epoch): {num_train_steps}")
    logger.info(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")

    inner_config = TrainLmConfig(
        data=config.data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=["gsm8k", "sft", "1e20", "qwen3", "single-epoch"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_train_steps,
            steps_per_eval=min(50, num_train_steps),
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[],
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


gsm8k_sft_step = ExecutorStep(
    name="checkpoints/gsm8k-sft-1e20-qwen3",
    description=(
        f"Single-epoch SFT on GSM8K train set " f"({compute_num_parameters(scaling_1e20_qwen3, 128256):,} params)."
    ),
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=gsm8k_tokenized,
        data_config=data_config,
        output_path=this_output_path(),
    ),
)

# ---------------------------------------------------------------------------
# 5. Evaluate on GSM8K test set (8-shot CoT via lm-evaluation-harness)
# ---------------------------------------------------------------------------
GSM8K_EVAL = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
]

gsm8k_eval_step = default_eval(
    step=gsm8k_sft_step,
    evals=GSM8K_EVAL,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=True,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[gsm8k_sft_step, gsm8k_eval_step],
        description="GSM8K single-epoch SFT on exp2166 1e20 Qwen3 final checkpoint.",
    )
