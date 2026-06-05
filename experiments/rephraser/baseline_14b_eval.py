# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline evaluation of Qwen3-14B-Base (no SFT) on HumanEval and MBPP.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/baseline_14b_eval.py
"""

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

from experiments.evals.evals import evaluate_lm_evaluation_harness

EVAL_TASKS = [
    EvalTaskConfig(name="humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=0, task_alias="mbpp_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=3, task_alias="mbpp_3shot"),
]
EVAL_ENGINE_KWARGS = {"max_model_len": 8192, "max_gen_toks": 512}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

baseline_eval = evaluate_lm_evaluation_harness(
    model_name="code-qwen3-14b-base-baseline",
    model_path="Qwen/Qwen3-14B-Base",
    evals=EVAL_TASKS,
    engine_kwargs=EVAL_ENGINE_KWARGS,
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[baseline_eval],
        description="Baseline evaluation of Qwen3-14B-Base (no SFT)",
    )
