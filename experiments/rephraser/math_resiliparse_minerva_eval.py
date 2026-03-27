# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run minerva_math + gsm8k_platinum_cot evals on resiliparse math SFT checkpoints.

Evaluates 3 resiliparse models:
- Default resiliparse SFT (lr=2e-5, bs=64)
- Low-reg resiliparse SFT (lr=2e-6, bs=32, wd=0.001)
- High-reg resiliparse SFT (lr=2e-6, bs=64, wd=0.1)

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/math_resiliparse_minerva_eval.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

# ---------------------------------------------------------------------------
# Eval tasks (same as extraction eval for comparison)
# ---------------------------------------------------------------------------
MINERVA_EVALS = [
    EvalTaskConfig(name="minerva_math_algebra", num_fewshot=4, task_alias="minerva_math_algebra_4shot"),
    EvalTaskConfig(name="minerva_math_prealgebra", num_fewshot=4, task_alias="minerva_math_prealgebra_4shot"),
    EvalTaskConfig(
        name="minerva_math_counting_and_prob",
        num_fewshot=4,
        task_alias="minerva_math_counting_and_prob_4shot",
    ),
    EvalTaskConfig(name="minerva_math_geometry", num_fewshot=4, task_alias="minerva_math_geometry_4shot"),
    EvalTaskConfig(
        name="minerva_math_intermediate_algebra",
        num_fewshot=4,
        task_alias="minerva_math_intermediate_algebra_4shot",
    ),
    EvalTaskConfig(name="minerva_math_num_theory", num_fewshot=4, task_alias="minerva_math_num_theory_4shot"),
    EvalTaskConfig(name="minerva_math_precalc", num_fewshot=4, task_alias="minerva_math_precalc_4shot"),
    EvalTaskConfig(name="gsm8k_platinum_cot", num_fewshot=8, task_alias="gsm8k_platinum_cot_8shot"),
]

EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Resiliparse checkpoint paths (config.json patched with max_position_embeddings=32768)
# ---------------------------------------------------------------------------
MODELS = {
    "minerva-math-resili-default-qwen3-0.6b-base": (
        "gs://marin-us-central1/checkpoints/" "math_multi_v2-resiliparse-qwen3-0.6b-base-sft-66e234/hf"
    ),
    "minerva-math-resili-lowreg-qwen3-0.6b-base": (
        "gs://marin-us-central1/checkpoints/" "math-resili-lowreg-qwen3-0.6b-base-b8490f/hf"
    ),
    "minerva-math-resili-highreg-qwen3-0.6b-base": (
        "gs://marin-us-central1/checkpoints/" "math-resili-highreg-qwen3-0.6b-base-0f1235/hf"
    ),
}

# ---------------------------------------------------------------------------
# Build eval steps
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

for model_name, model_path in MODELS.items():
    eval_step = evaluate_lm_evaluation_harness(
        model_name=model_name,
        model_path=model_path,
        evals=MINERVA_EVALS,
        resource_config=EVAL_RESOURCE,
        apply_chat_template=False,
        discover_latest_checkpoint=True,
    )
    all_steps.append(eval_step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Minerva MATH + GSM8K Platinum evals on resiliparse math SFT checkpoints (vLLM)",
    )
