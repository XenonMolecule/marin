# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone re-run of the 2 failed 14B top3 extraction evals.

These evals failed in the main pipeline due to TPU preemption. Running them
independently so they don't queue behind resili-best-resili.

The eval step names match the original pipeline (math_14b_top3_sft.py), so
they write to the same eval dirs (which have FAILED status → auto-retry).

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/math_14b_top3_failed_evals.py
"""

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import InputName, executor_main

from experiments.evals.evals import evaluate_lm_evaluation_harness

# Same eval suite as the main experiment
MATH_14B_EVALS = [
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
EVAL_ENGINE_KWARGS = {"max_model_len": 8192, "max_gen_toks": 1024}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# Hardcoded paths to the already-trained HF checkpoints
EXTRACT_BEST_RESILI_PATH = InputName.hardcoded(
    "gs://marin-us-central1/checkpoints/math-14b-top3-extract-best-resili-qwen3-14b-base-3e275e/hf/step-1932"
)
EXTRACT_BEST_EXTRACT_PATH = InputName.hardcoded(
    "gs://marin-us-central1/checkpoints/math-14b-top3-extract-best-extract-qwen3-14b-base-981bb3/hf/step-1932"
)

# These use the same model_name as the original pipeline, so they'll
# produce the same eval dir hashes and retry the FAILED steps.
eval_extract_best_resili = evaluate_lm_evaluation_harness(
    model_name="math-14b-top3-extract-best-resili-qwen3-14b-base",
    model_path=EXTRACT_BEST_RESILI_PATH,
    evals=MATH_14B_EVALS,
    engine_kwargs=EVAL_ENGINE_KWARGS,
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

eval_extract_best_extract = evaluate_lm_evaluation_harness(
    model_name="math-14b-top3-extract-best-extract-qwen3-14b-base",
    model_path=EXTRACT_BEST_EXTRACT_PATH,
    evals=MATH_14B_EVALS,
    engine_kwargs=EVAL_ENGINE_KWARGS,
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[eval_extract_best_resili, eval_extract_best_extract],
        description="Re-run 2 failed 14B top3 extraction evals independently",
    )
