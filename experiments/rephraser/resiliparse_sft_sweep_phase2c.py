# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resiliparse Phase 2c: bs=32, Weight Decay × Warmup sweep around lr=2e-6, bs=64.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/resiliparse_sft_sweep_phase2c.py
"""

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.rephraser.code_extraction_sft_v3_base import (
    qwen3_0_6b_hd128_with_rope,
)
from experiments.rephraser.code_extraction_sft_v3_base import (
    result as v3_result,
)
from experiments.rephraser.extraction_sft_recipe import (
    _run_single_epoch_sft,
    _SFTRunConfig,
)

# ---------------------------------------------------------------------------
# Fixed: lr=2e-6, bs=64
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-6
BATCH_SIZE = 32

WEIGHT_DECAYS = [0.001, 0.01, 0.05, 0.1]
WARMUPS = [0.0, 0.03, 0.1]

DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096

EVAL_TASKS = [
    EvalTaskConfig(name="humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=0, task_alias="mbpp_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=3, task_alias="mbpp_3shot"),
]
EVAL_ENGINE_KWARGS = {"max_model_len": 8192, "max_gen_toks": 512}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# Resiliparse tokenized data (recreate — hashes to same thing)
resili_processed = v3_result.resiliparse_step
resili_tokenized = default_tokenize(
    name="code_resiliparse_qwen3-0.6b-base_sft",
    dataset=output_path_of(resili_processed) / "**/*.jsonl.gz",
    tokenizer="Qwen/Qwen3-0.6B-Base",
    format=TextLmDatasetFormat(),
)

all_steps: list[ExecutorStep] = []

for wd in WEIGHT_DECAYS:
    for warmup in WARMUPS:
        wd_str = f"{wd}".replace(".", "p")
        warmup_str = f"{warmup}".replace(".", "p")
        config_name = f"wd{wd_str}_wu{warmup_str}"

        data_config = lm_data_config(training_set=resili_tokenized, validation_sets={})

        train_step = ExecutorStep(
            name=f"checkpoints/code-resili-p2c-{config_name}-qwen3-0.6b-base",
            description=f"Resiliparse Phase 2c: bs=32, lr=2e-6, bs=64, {config_name}",
            fn=_run_single_epoch_sft,
            config=_SFTRunConfig(
                tokenized_path=resili_tokenized,
                data_config=data_config,
                output_path=this_output_path(),
                tags=("code", "resili-phase2c", config_name, "resiliparse", "sft", "qwen3-0.6b-base"),
                model_config=qwen3_0_6b_hd128_with_rope,
                seq_len=SEQ_LEN,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                weight_decay=wd,
                warmup=warmup,
                decay=DECAY,
                lr_schedule=LR_SCHEDULE,
                max_grad_norm=MAX_GRAD_NORM,
                hf_model_name="Qwen/Qwen3-0.6B-Base",
                checkpoint_path=None,
                pad_tokenizer_to_match_model=True,
            ),
        )

        eval_step = evaluate_lm_evaluation_harness(
            model_name=f"code-resili-p2c-{config_name}-qwen3-0.6b-base",
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
        description=f"Resiliparse Phase 2c: bs=32, WD × Warmup at lr=2e-6, bs=64 ({len(all_steps)} configs)",
    )
