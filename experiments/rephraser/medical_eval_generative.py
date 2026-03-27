# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Medical evals using generative tasks to avoid vLLM TPU loglikelihood bug.

The vLLM TPU nightly-20260104 has a broken echo=True/max_tokens=0 codepath
that causes all loglikelihood-based MCQ tasks to fail with error "25".
Generative MMLU variants work around this by using generate_until instead.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-east5-a --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/medical_eval_generative.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")
EVAL_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 256}

# Generative MMLU medical subtasks (5-shot, matching standard MMLU fewshot)
MMLU_GENERATIVE_EVALS = [
    EvalTaskConfig(name="mmlu_anatomy_generative", num_fewshot=5, task_alias="mmlu_anatomy_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_clinical_knowledge_generative", num_fewshot=5, task_alias="mmlu_clinical_knowledge_gen_5shot",
    ),
    EvalTaskConfig(name="mmlu_college_medicine_generative", num_fewshot=5, task_alias="mmlu_college_medicine_gen_5shot"),
    EvalTaskConfig(name="mmlu_medical_genetics_generative", num_fewshot=5, task_alias="mmlu_medical_genetics_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_professional_medicine_generative", num_fewshot=5, task_alias="mmlu_professional_medicine_gen_5shot",
    ),
    EvalTaskConfig(name="mmlu_college_biology_generative", num_fewshot=5, task_alias="mmlu_college_biology_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_high_school_biology_generative", num_fewshot=5, task_alias="mmlu_high_school_biology_gen_5shot",
    ),
]

# Models to evaluate
MODELS = {
    "baseline": "Qwen/Qwen3-0.6B",
    "resiliparse": "gs://marin-us-east5/checkpoints/medical-resiliparse-qwen3-0.6b-sft-5bbd3e/hf/step-9282",
    "lowreg": "gs://marin-us-east5/checkpoints/medical-resili-lowreg-qwen3-0.6b-aaec7f/hf/step-18565",
    "highreg": "gs://marin-us-east5/checkpoints/medical-resili-highreg-qwen3-0.6b-0cf223/hf/step-9282",
}

all_steps: list[ExecutorStep] = []

for model_name, model_path in MODELS.items():
    # MMLU generative evals (test set)
    step = evaluate_lm_evaluation_harness(
        model_name=f"medical-{model_name}-mmlu-gen",
        model_path=model_path,
        evals=MMLU_GENERATIVE_EVALS,
        engine_kwargs=EVAL_ENGINE_KWARGS,
        resource_config=EVAL_RESOURCE,
        apply_chat_template=False,
        discover_latest_checkpoint=False,
    )
    all_steps.append(step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description=f"Medical generative evals — {len(MODELS)} models",
    )
