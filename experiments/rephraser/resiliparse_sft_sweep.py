# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter sweep for resiliparse SFT on Qwen3-0.6B-Base.

Focused sweep based on v3 extraction findings: very low LRs work best.
Tests a focused grid at the promising LR range.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/resiliparse_sft_sweep.py
"""

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.rephraser.code_extraction_sft_v3_base import result as v3_result
from experiments.rephraser.extraction_sft_recipe import (
    _run_single_epoch_sft,
    _SFTRunConfig,
)

# ---------------------------------------------------------------------------
# Sweep grid — focused on promising LR range from v3 findings
# ---------------------------------------------------------------------------
CONFIGS = [
    # Very low LRs (best range from v3 extraction sweep)
    (1e-6, 32),
    (1e-6, 64),
    (2e-6, 32),
    (2e-6, 64),
    (3e-6, 64),
    (5e-6, 16),
    (5e-6, 32),
    (5e-6, 64),
    # Also test default for comparison
    (2e-5, 64),
]

WEIGHT_DECAY = 0.01
WARMUP = 0.03
DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096

# ---------------------------------------------------------------------------
# Eval tasks
# ---------------------------------------------------------------------------
EVAL_TASKS = [
    EvalTaskConfig(name="humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=0, task_alias="mbpp_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=3, task_alias="mbpp_3shot"),
]
EVAL_ENGINE_KWARGS = {"max_model_len": 8192, "max_gen_toks": 512}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Reuse resiliparse tokenized data from v3 base experiment
# ---------------------------------------------------------------------------
from levanter.data.text import TextLmDatasetFormat

from experiments.defaults import default_tokenize
from experiments.rephraser.code_extraction_sft_v3_base import qwen3_0_6b_hd128_with_rope

# Recreate the tokenize step — will hash to the same thing and be skipped
resili_processed = v3_result.resiliparse_step
resili_tokenized = default_tokenize(
    name="code_resiliparse_qwen3-0.6b-base_sft",
    dataset=output_path_of(resili_processed) / "**/*.jsonl.gz",
    tokenizer="Qwen/Qwen3-0.6B-Base",
    format=TextLmDatasetFormat(),
)

all_steps: list[ExecutorStep] = []

for lr, bs in CONFIGS:
    lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    config_name = f"lr{lr_str}_bs{bs}"

    data_config = lm_data_config(training_set=resili_tokenized, validation_sets={})

    train_step = ExecutorStep(
        name=f"checkpoints/code-resili-sweep-{config_name}-qwen3-0.6b-base",
        description=f"Resiliparse SFT sweep: {config_name}",
        fn=_run_single_epoch_sft,
        config=_SFTRunConfig(
            tokenized_path=resili_tokenized,
            data_config=data_config,
            output_path=this_output_path(),
            tags=("code", "resili-sweep", config_name, "resiliparse", "sft", "qwen3-0.6b-base"),
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
        model_name=f"code-resili-sweep-{config_name}-qwen3-0.6b-base",
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
        description=f"Resiliparse SFT sweep: {len(CONFIGS)} configs",
    )
