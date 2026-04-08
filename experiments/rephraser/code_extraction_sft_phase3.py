# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3: LR schedule sweep for both V3 extraction and resiliparse.

Tests cosine vs linear schedule with different decay fractions, for both
the lowreg and highreg HP settings.

Configs:
  HP settings: lowreg (lr=2e-6, bs=32, wd=0.001, wu=0.0)
               highreg (lr=2e-6, bs=64, wd=0.1, wu=0.03)
  Schedules:   cosine × decay=[0.9, 0.95, 1.0]
               linear × decay=[0.9, 0.95, 1.0]
  Data:        V3 extraction, resiliparse
  Total:       (6 schedules - 1 default) × 2 HP × 2 data = 20 new runs

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/code_extraction_sft_phase3.py
"""

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.rephraser.code_extraction_sft_v3_base import (
    qwen3_0_6b_hd128_with_rope,
    result as v3_result,
)
from experiments.rephraser.extraction_sft_recipe import (
    _SFTRunConfig,
    _run_single_epoch_sft,
)
from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

# ---------------------------------------------------------------------------
# HP settings
# ---------------------------------------------------------------------------
HP_SETTINGS = {
    "lowreg": dict(batch_size=32, learning_rate=2e-6, weight_decay=0.001, warmup=0.0),
    "highreg": dict(batch_size=64, learning_rate=2e-6, weight_decay=0.1, warmup=0.03),
}

# Schedule variants (skip cosine/0.97 which is the default already run)
SCHEDULE_CONFIGS = [
    ("cosine", 0.9),
    ("cosine", 0.95),
    ("cosine", 1.0),
    ("linear", 0.9),
    ("linear", 0.95),
    ("linear", 1.0),
]
DEFAULT_SCHEDULE = ("cosine", 0.97)

SEQ_LEN = 4096
MAX_GRAD_NORM = 1.0

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
# Data sources
# ---------------------------------------------------------------------------
# V3 extraction tokenized data
v3_branch = v3_result.extraction_branches[0]
v3_tokenized = v3_branch.tokenize_step

# Resiliparse tokenized data
resili_processed = v3_result.resiliparse_step
resili_tokenized = default_tokenize(
    name="code_resiliparse_qwen3-0.6b-base_sft",
    dataset=output_path_of(resili_processed) / "**/*.jsonl.gz",
    tokenizer="Qwen/Qwen3-0.6B-Base",
    format=TextLmDatasetFormat(),
)

DATA_SOURCES = {
    "v3": v3_tokenized,
    "resili": resili_tokenized,
}

# ---------------------------------------------------------------------------
# Build sweep
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

for data_name, tokenized_step in DATA_SOURCES.items():
    for hp_name, hp in HP_SETTINGS.items():
        for lr_schedule, decay in SCHEDULE_CONFIGS:
            # Skip the default (already run in Phase 2)
            if (lr_schedule, decay) == DEFAULT_SCHEDULE:
                continue

            sched_str = f"{lr_schedule}_d{decay}".replace(".", "p")
            config_name = f"{data_name}-p3-{hp_name}-{sched_str}"

            data_config = lm_data_config(training_set=tokenized_step, validation_sets={})

            train_step = ExecutorStep(
                name=f"checkpoints/code-{config_name}-qwen3-0.6b-base",
                description=f"Phase 3: {config_name}",
                fn=_run_single_epoch_sft,
                config=_SFTRunConfig(
                    tokenized_path=tokenized_step,
                    data_config=data_config,
                    output_path=this_output_path(),
                    tags=("code", "phase3", config_name, "sft", "qwen3-0.6b-base"),
                    model_config=qwen3_0_6b_hd128_with_rope,
                    seq_len=SEQ_LEN,
                    batch_size=hp["batch_size"],
                    learning_rate=hp["learning_rate"],
                    weight_decay=hp["weight_decay"],
                    warmup=hp["warmup"],
                    decay=decay,
                    lr_schedule=lr_schedule,
                    max_grad_norm=MAX_GRAD_NORM,
                    hf_model_name="Qwen/Qwen3-0.6B-Base",
                    checkpoint_path=None,
                    pad_tokenizer_to_match_model=True,
                ),
            )

            eval_step = evaluate_lm_evaluation_harness(
                model_name=f"code-{config_name}-qwen3-0.6b-base",
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
        description=f"Phase 3: Schedule sweep ({len(all_steps)} configs)",
    )
