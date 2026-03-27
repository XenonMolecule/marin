# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run minerva_math + gsm8k_platinum_cot evals on all math extraction v2 checkpoints.

Evaluates 4 models:
- Baseline (Qwen3-0.6B-Base, no SFT)
- Default extraction SFT (lr=2e-5, bs=64)
- Low-reg extraction SFT (lr=2e-6, bs=32, wd=0.001)
- High-reg extraction SFT (lr=2e-6, bs=64, wd=0.1)

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/math_extraction_minerva_eval.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

# ---------------------------------------------------------------------------
# Eval tasks
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
# Checkpoint paths (already trained, config.json patched with max_position_embeddings=32768)
# ---------------------------------------------------------------------------
MODELS = {
    "minerva-baseline-qwen3-0.6b-base": "Qwen/Qwen3-0.6B-Base",
    "minerva-math-v2-default-qwen3-0.6b-base": (
        "gs://marin-us-central1/checkpoints/" "math_multi_v2-extract-unified-qwen3-0.6b-base-sft-47198e/hf"
    ),
    "minerva-math-v2-lowreg-qwen3-0.6b-base": (
        "gs://marin-us-central1/checkpoints/" "math-v2-lowreg-qwen3-0.6b-base-4343be/hf"
    ),
    "minerva-math-v2-highreg-qwen3-0.6b-base": (
        "gs://marin-us-central1/checkpoints/" "math-v2-highreg-qwen3-0.6b-base-d22f28/hf"
    ),
}

# ---------------------------------------------------------------------------
# Build eval steps
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

for model_name, model_path in MODELS.items():
    is_hf_hub = not model_path.startswith("gs://")
    eval_step = evaluate_lm_evaluation_harness(
        model_name=model_name,
        model_path=model_path,
        evals=MINERVA_EVALS,
        resource_config=EVAL_RESOURCE,
        apply_chat_template=False,
        discover_latest_checkpoint=not is_hf_hub,
    )
    all_steps.append(eval_step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Minerva MATH + GSM8K Platinum evals on math extraction v2 checkpoints (vLLM)",
    )
