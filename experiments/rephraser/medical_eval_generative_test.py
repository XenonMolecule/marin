# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick test: generative medical evals on baseline Qwen3-0.6B only."""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")
EVAL_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 256}

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

step = evaluate_lm_evaluation_harness(
    model_name="medical-baseline-gen-test",
    model_path="Qwen/Qwen3-0.6B",
    evals=MMLU_GENERATIVE_EVALS,
    engine_kwargs=EVAL_ENGINE_KWARGS,
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

if __name__ == "__main__":
    executor_main(steps=[step], description="Medical generative eval test — baseline only")
