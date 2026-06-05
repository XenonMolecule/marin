# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter sweep for v3 commented extraction SFT on Qwen3-14B-Base.

Phase 1: LR × Batch Size (15 runs)
  LR:    [1e-6, 2e-6, 5e-6, 1e-5, 5e-5]
  Batch: [16, 32, 64]

Uses v5p-32 for training (14B model needs more memory than v5p-8).
Reuses tokenized data from the 0.6B base experiment (same Qwen3 tokenizer).

Primary metric: MBPP 3-shot pass@1.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/code_extraction_sft_v3_14b_sweep.py
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.qwen3 import qwen3_14b
from experiments.rephraser.code_extraction_sft_v3_base import result as v3_result
from experiments.rephraser.extraction_sft_recipe import (
    _run_single_epoch_sft,
    _SFTRunConfig,
)

# ---------------------------------------------------------------------------
# Qwen3-14B-Base model config (with SFT-appropriate RoPE / seq_len)
# ---------------------------------------------------------------------------
qwen3_14b_with_rope = dataclasses.replace(
    qwen3_14b,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,  # Qwen3-14B-Base's actual RoPE capacity
)

# ---------------------------------------------------------------------------
# Sweep grid — Phase 1: LR × Batch Size
# ---------------------------------------------------------------------------
LEARNING_RATES = [1e-6, 2e-6, 5e-6, 1e-5, 5e-5]
BATCH_SIZES = [16, 32, 64]

# Fixed hyperparameters (defaults from extraction_sft_recipe)
WEIGHT_DECAY = 0.01
WARMUP = 0.03
DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096

# Training on v5p-32 (14B needs more memory; idle v5p-32 nodes available)
TRAIN_TPU_TYPE = "v5p-32"

# ---------------------------------------------------------------------------
# Eval tasks
# ---------------------------------------------------------------------------
EVAL_TASKS = [
    EvalTaskConfig(name="humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=0, task_alias="mbpp_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=3, task_alias="mbpp_3shot"),
]
# 8192 context fits all MBPP 3-shot prompts (up to ~4300 tokens) + generation.
EVAL_ENGINE_KWARGS = {"max_model_len": 8192, "max_gen_toks": 512}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Reuse tokenized data from the 0.6B v3 base experiment (same Qwen3 tokenizer)
# ---------------------------------------------------------------------------
v3_branch = v3_result.extraction_branches[0]  # "commented" extraction
tokenized_step = v3_branch.tokenize_step

# ---------------------------------------------------------------------------
# Build sweep grid
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

for lr in LEARNING_RATES:
    for bs in BATCH_SIZES:
        lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
        config_name = f"lr{lr_str}_bs{bs}"

        data_config = lm_data_config(training_set=tokenized_step, validation_sets={})

        train_step = ExecutorStep(
            name=f"checkpoints/code-v3-sweep-{config_name}-qwen3-14b-base",
            description=f"V3 extraction SFT sweep: {config_name} (Qwen3-14B-Base)",
            fn=_run_single_epoch_sft,
            config=_SFTRunConfig(
                tokenized_path=tokenized_step,
                data_config=data_config,
                output_path=this_output_path(),
                tags=("code", "v3-sweep", config_name, "extraction", "sft", "qwen3-14b-base"),
                model_config=qwen3_14b_with_rope,
                seq_len=SEQ_LEN,
                batch_size=bs,
                learning_rate=lr,
                weight_decay=WEIGHT_DECAY,
                warmup=WARMUP,
                decay=DECAY,
                lr_schedule=LR_SCHEDULE,
                max_grad_norm=MAX_GRAD_NORM,
                hf_model_name="Qwen/Qwen3-14B-Base",
                checkpoint_path=None,
                pad_tokenizer_to_match_model=True,
                train_tpu_type=TRAIN_TPU_TYPE,
            ),
        )

        eval_step = evaluate_lm_evaluation_harness(
            model_name=f"code-v3-sweep-{config_name}-qwen3-14b-base",
            model_path=output_path_of(train_step, "hf"),
            evals=EVAL_TASKS,
            engine_kwargs=EVAL_ENGINE_KWARGS,
            resource_config=EVAL_RESOURCE,
            apply_chat_template=False,
            discover_latest_checkpoint=True,
        )

        all_steps.append(eval_step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description=f"V3 extraction SFT sweep (14B): {len(LEARNING_RATES)} LRs × {len(BATCH_SIZES)} batch sizes = {len(all_steps)} runs",
    )
