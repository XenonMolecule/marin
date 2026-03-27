# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone baseline 0-shot eval for Qwen3-0.6B.

Just the baseline 0-shot eval from qwen3_mathhelpforum_sft.py, extracted so it
can be launched independently on a different cluster.

Launch (us-east5-a):
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-east5-a --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/baseline_0shot_eval.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

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

baseline_eval_0shot = evaluate_lm_evaluation_harness(
    model_name="qwen3-0.6b-baseline-0shot",
    model_path="Qwen/Qwen3-0.6B",
    evals=MATH_EVALS_0SHOT,
    engine_kwargs=GSM8K_ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

if __name__ == "__main__":
    executor_main(steps=[baseline_eval_0shot], description="Qwen3 0.6B baseline 0-shot eval")
