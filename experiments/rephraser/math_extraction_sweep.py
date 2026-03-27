# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter sweep for math v2 extraction SFT on Qwen3-0.6B-Base.

Phase 1: LR x Batch Size (24 runs)
  LR:    [5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5]
  Batch: [16, 32, 64, 128]

All runs use the same tokenized v2 math extraction data (reused from the
v2 base experiment). Each trains for 1 epoch with different LR/batch combos,
then evaluates on minerva_math (algebra + prealgebra) and gsm8k_platinum_cot.

Fixed at high-reg weight decay (0.1) since that was best in the initial 3-config
comparison. Phase 2 can refine WD/warmup around the best Phase 1 config.

Primary metrics: minerva_math_algebra (4-shot), gsm8k_platinum_cot (8-shot).

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/math_extraction_sweep.py
"""

import dataclasses

from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.rephraser.extraction_sft_recipe import (
    _SFTRunConfig,
    _run_single_epoch_sft,
)
from experiments.rephraser.mathhelpforum_extraction_sft_v2_base import result as math_result
from fray.cluster import ResourceConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

# ---------------------------------------------------------------------------
# Sweep grid — Phase 1: LR x Batch Size
# ---------------------------------------------------------------------------
LEARNING_RATES = [5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5]
BATCH_SIZES = [16, 32, 64, 128]

# Fixed at high-reg values (best from initial comparison)
WEIGHT_DECAY = 0.1
WARMUP = 0.03
DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096

# ---------------------------------------------------------------------------
# Model config — with correct hf_max_position_embeddings for vLLM
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,  # Qwen3-0.6B-Base's actual RoPE capacity
)

# ---------------------------------------------------------------------------
# Eval tasks — compact subset for sweep (3 tasks instead of 8)
# ---------------------------------------------------------------------------
SWEEP_EVALS = [
    EvalTaskConfig(name="minerva_math_algebra", num_fewshot=4, task_alias="minerva_math_algebra_4shot"),
    EvalTaskConfig(name="minerva_math_prealgebra", num_fewshot=4, task_alias="minerva_math_prealgebra_4shot"),
    EvalTaskConfig(name="gsm8k_platinum_cot", num_fewshot=8, task_alias="gsm8k_platinum_cot_8shot"),
]
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Reuse tokenized data from the v2 base experiment
# ---------------------------------------------------------------------------
math_extraction_branch = math_result.extraction_branches[0]  # "unified" extraction
tokenized_step = math_extraction_branch.tokenize_step

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
            name=f"checkpoints/math-v2-sweep-{config_name}-qwen3-0.6b-base",
            description=f"Math extraction SFT sweep: {config_name}",
            fn=_run_single_epoch_sft,
            config=_SFTRunConfig(
                tokenized_path=tokenized_step,
                data_config=data_config,
                output_path=this_output_path(),
                tags=("math", "v2-sweep", config_name, "extraction", "sft", "qwen3-0.6b-base"),
                model_config=qwen3_0_6b_hd128_with_rope,
                seq_len=SEQ_LEN,
                batch_size=bs,
                learning_rate=lr,
                weight_decay=WEIGHT_DECAY,
                warmup=WARMUP,
                decay=DECAY,
                lr_schedule=LR_SCHEDULE,
                max_grad_norm=MAX_GRAD_NORM,
                hf_model_name="Qwen/Qwen3-0.6B-Base",
                checkpoint_path=None,
                pad_tokenizer_to_match_model=True,
            ),
        )

        eval_step = evaluate_lm_evaluation_harness(
            model_name=f"math-v2-sweep-{config_name}-qwen3-0.6b-base",
            model_path=output_path_of(train_step, "hf"),
            evals=SWEEP_EVALS,
            resource_config=EVAL_RESOURCE,
            apply_chat_template=False,
            discover_latest_checkpoint=True,
        )

        all_steps.append(eval_step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description=f"Math extraction SFT sweep: {len(LEARNING_RATES)} LRs x {len(BATCH_SIZES)} batch sizes = {len(all_steps)} runs",
    )
