# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 2: Weight Decay × Warmup sweep for Qwen3-14B-Base V3 extraction SFT.

Phase 1 winner: lr=2e-6, bs=32 (MBPP 3-shot: 72.8%)
Now sweep WD × Warmup (12 configs) at the winning LR/BS.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/code_extraction_sft_v3_14b_sweep_phase2.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.rephraser.code_extraction_sft_v3_base import result as v3_result
from experiments.rephraser.code_extraction_sft_v3_14b_sweep import qwen3_14b_with_rope
from experiments.rephraser.extraction_sft_recipe import (
    _SFTRunConfig,
    _run_single_epoch_sft,
)
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

# ---------------------------------------------------------------------------
# Fixed: lr=2e-6, bs=32 — Phase 1 winner (MBPP 3-shot: 72.8%)
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-6
BATCH_SIZE = 32

WEIGHT_DECAYS = [0.001, 0.01, 0.05, 0.1]
WARMUPS = [0.0, 0.03, 0.1]

DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096

# Training on v5p-32 (14B model)
TRAIN_TPU_TYPE = "v5p-32"

EVAL_TASKS = [
    EvalTaskConfig(name="humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=0, task_alias="mbpp_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=3, task_alias="mbpp_3shot"),
]
EVAL_ENGINE_KWARGS = {"max_model_len": 8192, "max_gen_toks": 512}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

v3_branch = v3_result.extraction_branches[0]
tokenized_step = v3_branch.tokenize_step

all_steps: list[ExecutorStep] = []

for wd in WEIGHT_DECAYS:
    for warmup in WARMUPS:
        wd_str = f"{wd}".replace(".", "p")
        warmup_str = f"{warmup}".replace(".", "p")
        config_name = f"wd{wd_str}_wu{warmup_str}"

        data_config = lm_data_config(training_set=tokenized_step, validation_sets={})

        train_step = ExecutorStep(
            name=f"checkpoints/code-v3-14b-p2-{config_name}-qwen3-14b-base",
            description=f"V3 14B Phase 2: lr=2e-6, bs=32, {config_name}",
            fn=_run_single_epoch_sft,
            config=_SFTRunConfig(
                tokenized_path=tokenized_step,
                data_config=data_config,
                output_path=this_output_path(),
                tags=("code", "v3-14b-phase2", config_name, "extraction", "sft", "qwen3-14b-base"),
                model_config=qwen3_14b_with_rope,
                seq_len=SEQ_LEN,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                weight_decay=wd,
                warmup=warmup,
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
            model_name=f"code-v3-14b-p2-{config_name}-qwen3-14b-base",
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
        description=f"V3 14B Phase 2: WD × Warmup at lr=2e-6, bs=32 ({len(all_steps)} configs)",
    )
