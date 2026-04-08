# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 0.6B mathhelpforum SFT sweep.

Independent experiments (parallel DAG branches) comparing Qwen/Qwen3-0.6B
(instruct) and Qwen/Qwen3-0.6B-Base on math SFT data.

Instruct model experiments (1-8):
1. Baseline GSM8K eval (no training)
2. GSM8K chat SFT
3. GSM8K plaintext SFT
4. Q/R/A chat SFT (mathhelpforum rephraser-extracted)
5. Q/R/A plaintext SFT (mathhelpforum rephraser-extracted)
6. Resiliparse continued pretraining (mathhelpforum HTML -> plain text)
7. Q/R/A plaintext SFT v2 (fixed multi-line answer parser)
8. Raw Q/R/A markdown SFT (no parser, continued pretraining on structured markdown)

Base model experiments (9-13):
9. Baseline eval (Qwen3-0.6B-Base, no training)
10. GSM8K plaintext SFT (base)
11. Q/R/A plaintext SFT v2 (base)
12. Resiliparse continued pretraining (base)
13. Raw Q/R/A markdown SFT (base)

Data source steps are imported from existing scripts — the executor skips them
because their outputs already exist on GCS. Tokenized data is shared between
instruct and base variants (same tokenizer vocab).

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/qwen3_mathhelpforum_sft.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/qwen3_mathhelpforum_sft.py --dry_run true
"""

import dataclasses
import logging
import math
from dataclasses import dataclass
from datetime import timedelta

import jmp

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.rephraser.gsm8k_sft_plaintext import plaintext_transform_step as gsm8k_plaintext_step
from experiments.rephraser.mathhelpforum_qra_sft import chat_transform_step
from experiments.rephraser.mathhelpforum_extract_qra import postprocess_step_qra
from experiments.rephraser.mathhelpforum_qra_sft_plaintext import plaintext_transform_step as qra_plaintext_step
from experiments.rephraser.mathhelpforum_qra_sft_plaintext import plaintext_transform_step_v2 as qra_plaintext_step_v2
from experiments.rephraser.mathhelpforum_resiliparse_sft import extract_text_step
from experiments.rephraser.rephraser_cooldown import _read_token_count
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import ChatLmDatasetFormat, LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QWEN3_TOKENIZER = "Qwen/Qwen3-0.6B"
SEQ_LEN = 4096
BATCH_SIZE = 64

# Qwen3 0.6B architecture with head_dim=128 and theta=1M (matching HF weights)
qwen3_model_config = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=SEQ_LEN,
)

QWEN3_CHAT_FORMAT = ChatLmDatasetFormat(
    chat_template=QWEN_3_CHAT_TEMPLATE,
    mask_user_turns=True,
    pack=True,
)

MATH_EVALS = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
    EvalTaskConfig(name="hendrycks_math_algebra", num_fewshot=4, task_alias="hendrycks_math_algebra_4shot"),
    EvalTaskConfig(
        name="hendrycks_math_counting_and_prob", num_fewshot=4, task_alias="hendrycks_math_counting_and_prob_4shot"
    ),
    EvalTaskConfig(name="hendrycks_math_geometry", num_fewshot=4, task_alias="hendrycks_math_geometry_4shot"),
    EvalTaskConfig(
        name="hendrycks_math_intermediate_algebra", num_fewshot=4, task_alias="hendrycks_math_intermediate_algebra_4shot"
    ),
    EvalTaskConfig(name="hendrycks_math_num_theory", num_fewshot=4, task_alias="hendrycks_math_num_theory_4shot"),
    EvalTaskConfig(name="hendrycks_math_prealgebra", num_fewshot=4, task_alias="hendrycks_math_prealgebra_4shot"),
    EvalTaskConfig(name="hendrycks_math_precalc", num_fewshot=4, task_alias="hendrycks_math_precalc_4shot"),
]

MATH_EVALS_0SHOT = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=0, task_alias="gsm8k_cot_0shot"),
    EvalTaskConfig(name="hendrycks_math_algebra", num_fewshot=0, task_alias="hendrycks_math_algebra_0shot"),
    EvalTaskConfig(
        name="hendrycks_math_counting_and_prob", num_fewshot=0, task_alias="hendrycks_math_counting_and_prob_0shot"
    ),
    EvalTaskConfig(name="hendrycks_math_geometry", num_fewshot=0, task_alias="hendrycks_math_geometry_0shot"),
    EvalTaskConfig(
        name="hendrycks_math_intermediate_algebra", num_fewshot=0, task_alias="hendrycks_math_intermediate_algebra_0shot"
    ),
    EvalTaskConfig(name="hendrycks_math_num_theory", num_fewshot=0, task_alias="hendrycks_math_num_theory_0shot"),
    EvalTaskConfig(name="hendrycks_math_prealgebra", num_fewshot=0, task_alias="hendrycks_math_prealgebra_0shot"),
    EvalTaskConfig(name="hendrycks_math_precalc", num_fewshot=0, task_alias="hendrycks_math_precalc_0shot"),
]
GSM8K_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 1024}

# NOTE: All evals use evaluate_lm_evaluation_harness (vLLM), NOT the Levanter evaluator.
# The Levanter evaluator (evaluate_levanter_lm_evaluation_harness / default_eval) is designed
# for loglikelihood-based tasks (MMLU, perplexity). For generation tasks like GSM8K CoT and
# MATH, use the vLLM evaluator. The Levanter evaluator also has persistent TPU scheduling
# issues on the cluster ("No accelerator found") because it requires JAX to see the TPU at
# TrainerConfig init time, whereas vLLM handles device discovery more robustly via Docker.
# SFT checkpoints are accessible to vLLM via the HF export at output_path_of(step, "hf").


# ---------------------------------------------------------------------------
# Shared training function for all SFT experiments
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SFTRunConfig:
    tokenized_path: str
    data_config: LMMixtureDatasetConfig
    output_path: str
    tags: tuple[str, ...]


def _run_single_epoch_sft(config: _SFTRunConfig):
    """Train Qwen3-0.6B for exactly 1 epoch, computing steps from the tokenized cache."""
    total_tokens = _read_token_count(config.tokenized_path, split="train")
    num_train_steps = math.ceil(total_tokens / (BATCH_SIZE * SEQ_LEN))

    logger.info("=== Qwen3 0.6B Single-Epoch SFT ===")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Num train steps (1 epoch): {num_train_steps}")
    logger.info(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")
    logger.info(f"Tags: {config.tags}")

    inner_config = TrainLmConfig(
        data=config.data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=list(config.tags),
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
        initialize_from_hf="Qwen/Qwen3-0.6B",
        pad_tokenizer_to_match_model=True,
        train_seq_len=SEQ_LEN,
        model=qwen3_model_config,
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


# ===================================================================
# Experiment 1: Baseline eval (no training)
# ===================================================================
baseline_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-baseline-4shot",
    model_path="Qwen/Qwen3-0.6B",
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)


# ===================================================================
# Experiment 2: GSM8K chat SFT
# ===================================================================
gsm8k_train = get_instruction_dataset("openai/gsm8k", splits=["train"])

gsm8k_chat_tokenized = default_tokenize(
    name="qwen3_gsm8k_chat_sft",
    dataset=gsm8k_train / "**/*.jsonl.gz",
    tokenizer=QWEN3_TOKENIZER,
    format=QWEN3_CHAT_FORMAT,
)

gsm8k_chat_data = lm_data_config(training_set=gsm8k_chat_tokenized, validation_sets={})

gsm8k_chat_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-gsm8k-chat-sft",
    description="Single-epoch chat SFT on GSM8K train (Qwen3-0.6B).",
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=gsm8k_chat_tokenized,
        data_config=gsm8k_chat_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b", "gsm8k", "chat", "sft", "single-epoch"),
    ),
)

gsm8k_chat_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-gsm8k-chat-sft-vllm",
    model_path=output_path_of(gsm8k_chat_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=True,
)


# ===================================================================
# Experiment 3: GSM8K plaintext SFT
# ===================================================================
gsm8k_plain_tokenized = default_tokenize(
    name="qwen3_gsm8k_plaintext_sft",
    dataset=gsm8k_plaintext_step / "**/*.jsonl.gz",
    tokenizer=QWEN3_TOKENIZER,
    format=TextLmDatasetFormat(),
)

gsm8k_plain_data = lm_data_config(training_set=gsm8k_plain_tokenized, validation_sets={})

gsm8k_plain_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-gsm8k-plaintext-sft",
    description="Single-epoch plaintext SFT on GSM8K train (Qwen3-0.6B).",
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=gsm8k_plain_tokenized,
        data_config=gsm8k_plain_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b", "gsm8k", "plaintext", "sft", "single-epoch"),
    ),
)

gsm8k_plain_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-gsm8k-plaintext-sft-vllm",
    model_path=output_path_of(gsm8k_plain_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ===================================================================
# Experiment 4: Q/R/A chat SFT
# ===================================================================
qra_chat_tokenized = default_tokenize(
    name="qwen3_mathhelpforum_qra_chat_sft",
    dataset=chat_transform_step / "**/*.jsonl.gz",
    tokenizer=QWEN3_TOKENIZER,
    format=QWEN3_CHAT_FORMAT,
)

qra_chat_data = lm_data_config(training_set=qra_chat_tokenized, validation_sets={})

qra_chat_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-qra-chat-sft",
    description="Single-epoch chat SFT on mathhelpforum Q/R/A (Qwen3-0.6B).",
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=qra_chat_tokenized,
        data_config=qra_chat_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b", "mathhelpforum", "qra", "chat", "sft", "single-epoch"),
    ),
)

qra_chat_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-qra-chat-sft-vllm",
    model_path=output_path_of(qra_chat_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=True,
)


# ===================================================================
# Experiment 5: Q/R/A plaintext SFT
# ===================================================================
qra_plain_tokenized = default_tokenize(
    name="qwen3_mathhelpforum_qra_plaintext_sft",
    dataset=qra_plaintext_step / "**/*.jsonl.gz",
    tokenizer=QWEN3_TOKENIZER,
    format=TextLmDatasetFormat(),
)

qra_plain_data = lm_data_config(training_set=qra_plain_tokenized, validation_sets={})

qra_plain_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-qra-plaintext-sft",
    description="Single-epoch plaintext SFT on mathhelpforum Q/R/A (Qwen3-0.6B).",
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=qra_plain_tokenized,
        data_config=qra_plain_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b", "mathhelpforum", "qra", "plaintext", "sft", "single-epoch"),
    ),
)

qra_plain_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-qra-plaintext-sft-vllm",
    model_path=output_path_of(qra_plain_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ===================================================================
# Experiment 6: Resiliparse continued pretraining
# ===================================================================
resiliparse_tokenized = default_tokenize(
    name="qwen3_mathhelpforum_resiliparse_sft",
    dataset=extract_text_step / "**/*.jsonl.gz",
    tokenizer=QWEN3_TOKENIZER,
    format=TextLmDatasetFormat(),
)

resiliparse_data = lm_data_config(training_set=resiliparse_tokenized, validation_sets={})

resiliparse_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-resiliparse-sft",
    description="Single-epoch continued pretraining on mathhelpforum resiliparse text (Qwen3-0.6B).",
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=resiliparse_tokenized,
        data_config=resiliparse_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b", "mathhelpforum", "resiliparse", "continued-pretraining", "single-epoch"),
    ),
)

resiliparse_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-resiliparse-sft-vllm",
    model_path=output_path_of(resiliparse_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ===================================================================
# 0-shot evals (same models, 0-shot GSM8K + 0-shot MATH)
# ===================================================================
baseline_eval_0shot = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-baseline-0shot",
    model_path="Qwen/Qwen3-0.6B",
    evals=MATH_EVALS_0SHOT,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

gsm8k_plain_eval_0shot = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-gsm8k-plaintext-sft-0shot",
    model_path=output_path_of(gsm8k_plain_train_step, "hf"),
    evals=MATH_EVALS_0SHOT,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)

qra_plain_eval_0shot = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-qra-plaintext-sft-0shot",
    model_path=output_path_of(qra_plain_train_step, "hf"),
    evals=MATH_EVALS_0SHOT,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)

resiliparse_eval_0shot = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-resiliparse-sft-0shot",
    model_path=output_path_of(resiliparse_train_step, "hf"),
    evals=MATH_EVALS_0SHOT,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ===================================================================
# Experiment 7: Q/R/A plaintext SFT v2 (fixed multi-line answer parser)
# ===================================================================
qra_plain_v2_tokenized = default_tokenize(
    name="qwen3_mathhelpforum_qra_plaintext_sft_v2",
    dataset=qra_plaintext_step_v2 / "**/*.jsonl.gz",
    tokenizer=QWEN3_TOKENIZER,
    format=TextLmDatasetFormat(),
)

qra_plain_v2_data = lm_data_config(training_set=qra_plain_v2_tokenized, validation_sets={})

qra_plain_v2_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-qra-plaintext-sft-v2",
    description="Single-epoch plaintext SFT on mathhelpforum Q/R/A v2 — fixed multi-line answer (Qwen3-0.6B).",
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=qra_plain_v2_tokenized,
        data_config=qra_plain_v2_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b", "mathhelpforum", "qra", "plaintext", "sft", "v2", "fixed-answer", "single-epoch"),
    ),
)

qra_plain_v2_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-qra-plaintext-sft-v2-vllm",
    model_path=output_path_of(qra_plain_v2_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)

qra_plain_v2_eval_0shot = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-qra-plaintext-sft-v2-0shot",
    model_path=output_path_of(qra_plain_v2_train_step, "hf"),
    evals=MATH_EVALS_0SHOT,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ===================================================================
# Experiment 8: Raw Q/R/A markdown SFT (no parser, continued pretraining)
# ===================================================================
raw_qra_tokenized = default_tokenize(
    name="qwen3_mathhelpforum_raw_qra_markdown_sft",
    dataset=postprocess_step_qra / "**/*.jsonl.gz",
    tokenizer=QWEN3_TOKENIZER,
    format=TextLmDatasetFormat(),
)

raw_qra_data = lm_data_config(training_set=raw_qra_tokenized, validation_sets={})

raw_qra_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-raw-qra-markdown-sft",
    description="Single-epoch continued pretraining on raw Q/R/A markdown from rephraser (Qwen3-0.6B).",
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=raw_qra_tokenized,
        data_config=raw_qra_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b", "mathhelpforum", "raw-qra-markdown", "continued-pretraining", "single-epoch"),
    ),
)

raw_qra_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-raw-qra-markdown-sft-vllm",
    model_path=output_path_of(raw_qra_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)

raw_qra_eval_0shot = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-raw-qra-markdown-sft-0shot",
    model_path=output_path_of(raw_qra_train_step, "hf"),
    evals=MATH_EVALS_0SHOT,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ###################################################################
# Base model experiments (Qwen3-0.6B-Base)
#
# Separate config/function to avoid changing hashes of existing steps.
# Reuses the same tokenized data (same tokenizer vocab as instruct).
# ###################################################################
QWEN3_BASE = "Qwen/Qwen3-0.6B-Base"


@dataclass(frozen=True)
class _SFTBaseRunConfig:
    tokenized_path: str
    data_config: LMMixtureDatasetConfig
    output_path: str
    tags: tuple[str, ...]
    hf_model_name: str = QWEN3_BASE


def _run_single_epoch_sft_base(config: _SFTBaseRunConfig):
    """Train Qwen3-0.6B-Base for exactly 1 epoch."""
    total_tokens = _read_token_count(config.tokenized_path, split="train")
    num_train_steps = math.ceil(total_tokens / (BATCH_SIZE * SEQ_LEN))

    logger.info(f"=== Qwen3 0.6B-Base Single-Epoch SFT ({config.hf_model_name}) ===")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Num train steps (1 epoch): {num_train_steps}")
    logger.info(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")
    logger.info(f"Tags: {config.tags}")

    inner_config = TrainLmConfig(
        data=config.data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=list(config.tags),
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
        initialize_from_hf=config.hf_model_name,
        pad_tokenizer_to_match_model=True,
        train_seq_len=SEQ_LEN,
        model=qwen3_model_config,
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


# ===================================================================
# Experiment 9: Baseline eval (Qwen3-0.6B-Base, no training)
# ===================================================================
base_baseline_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-base-baseline-4shot",
    model_path=QWEN3_BASE,
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)


# ===================================================================
# Experiment 10: GSM8K plaintext SFT (base)
# ===================================================================
base_gsm8k_plain_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-base-gsm8k-plaintext-sft",
    description="Single-epoch plaintext SFT on GSM8K train (Qwen3-0.6B-Base).",
    fn=_run_single_epoch_sft_base,
    config=_SFTBaseRunConfig(
        tokenized_path=gsm8k_plain_tokenized,
        data_config=gsm8k_plain_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b-base", "gsm8k", "plaintext", "sft", "single-epoch"),
    ),
)

base_gsm8k_plain_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-base-gsm8k-plaintext-sft-vllm",
    model_path=output_path_of(base_gsm8k_plain_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ===================================================================
# Experiment 11: Q/R/A plaintext SFT v2 (base)
# ===================================================================
base_qra_plain_v2_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-base-qra-plaintext-sft-v2",
    description="Single-epoch plaintext SFT on mathhelpforum Q/R/A v2 (Qwen3-0.6B-Base).",
    fn=_run_single_epoch_sft_base,
    config=_SFTBaseRunConfig(
        tokenized_path=qra_plain_v2_tokenized,
        data_config=qra_plain_v2_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b-base", "mathhelpforum", "qra", "plaintext", "sft", "v2", "single-epoch"),
    ),
)

base_qra_plain_v2_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-base-qra-plaintext-sft-v2-vllm",
    model_path=output_path_of(base_qra_plain_v2_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ===================================================================
# Experiment 12: Resiliparse continued pretraining (base)
# ===================================================================
base_resiliparse_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-base-resiliparse-sft",
    description="Single-epoch continued pretraining on mathhelpforum resiliparse text (Qwen3-0.6B-Base).",
    fn=_run_single_epoch_sft_base,
    config=_SFTBaseRunConfig(
        tokenized_path=resiliparse_tokenized,
        data_config=resiliparse_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b-base", "mathhelpforum", "resiliparse", "continued-pretraining", "single-epoch"),
    ),
)

base_resiliparse_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-base-resiliparse-sft-vllm",
    model_path=output_path_of(base_resiliparse_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ===================================================================
# Experiment 13: Raw Q/R/A markdown SFT (base)
# ===================================================================
base_raw_qra_train_step = ExecutorStep(
    name="checkpoints/qwen3-0.6b-base-raw-qra-markdown-sft",
    description="Single-epoch continued pretraining on raw Q/R/A markdown (Qwen3-0.6B-Base).",
    fn=_run_single_epoch_sft_base,
    config=_SFTBaseRunConfig(
        tokenized_path=raw_qra_tokenized,
        data_config=raw_qra_data,
        output_path=this_output_path(),
        tags=("qwen3-0.6b-base", "mathhelpforum", "raw-qra-markdown", "continued-pretraining", "single-epoch"),
    ),
)

base_raw_qra_eval = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-base-raw-qra-markdown-sft-vllm",
    model_path=output_path_of(base_raw_qra_train_step, "hf"),
    evals=MATH_EVALS,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
)


# ===================================================================
# Entry point — all experiments run in parallel
# ===================================================================
all_steps: list[ExecutorStep] = [
    # Instruct model (experiments 1-8)
    baseline_eval,
    gsm8k_chat_eval,
    gsm8k_plain_eval,
    qra_chat_eval,
    qra_plain_eval,
    resiliparse_eval,
    # 0-shot evals
    baseline_eval_0shot,
    gsm8k_plain_eval_0shot,
    qra_plain_eval_0shot,
    resiliparse_eval_0shot,
    # Experiments 7-8 (fixed parser + raw markdown)
    qra_plain_v2_eval,
    qra_plain_v2_eval_0shot,
    raw_qra_eval,
    raw_qra_eval_0shot,
    # Base model (experiments 9-13)
    base_baseline_eval,
    base_gsm8k_plain_eval,
    base_qra_plain_v2_eval,
    base_resiliparse_eval,
    base_raw_qra_eval,
]

if __name__ == "__main__":
    executor_main(steps=all_steps, description="Qwen3 0.6B mathhelpforum SFT sweep (instruct + base)")
