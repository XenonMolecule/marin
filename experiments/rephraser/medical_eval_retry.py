# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry medical evals with tasks split into small batches to survive TPU preemption.

Each eval step runs 2-3 tasks, completing in minutes instead of potentially
getting preempted during a long run.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-east5-a --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/medical_eval_retry.py
"""

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

from experiments.evals.evals import evaluate_lm_evaluation_harness

# Split evals into small fast batches
BATCH_1 = [  # ~7400 questions — the big ones
    EvalTaskConfig(name="medmcqa", num_fewshot=0, task_alias="medmcqa_0shot"),
    EvalTaskConfig(name="medqa_4options", num_fewshot=0, task_alias="medqa_0shot"),
]

BATCH_2 = [  # ~3200 questions
    EvalTaskConfig(name="pubmedqa", num_fewshot=0, task_alias="pubmedqa_0shot"),
    EvalTaskConfig(name="headqa_en", num_fewshot=0, task_alias="headqa_en_0shot"),
]

BATCH_3 = [  # ~550 questions — MMLU subtasks are small
    EvalTaskConfig(name="mmlu_anatomy", num_fewshot=0, task_alias="mmlu_anatomy_0shot"),
    EvalTaskConfig(name="mmlu_clinical_knowledge", num_fewshot=0, task_alias="mmlu_clinical_knowledge_0shot"),
    EvalTaskConfig(name="mmlu_college_medicine", num_fewshot=0, task_alias="mmlu_college_medicine_0shot"),
    EvalTaskConfig(name="mmlu_medical_genetics", num_fewshot=0, task_alias="mmlu_medical_genetics_0shot"),
    EvalTaskConfig(name="mmlu_professional_medicine", num_fewshot=0, task_alias="mmlu_professional_medicine_0shot"),
]

EVAL_BATCHES = [BATCH_1, BATCH_2, BATCH_3]

EVAL_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 1024}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# Models to evaluate
MODELS = {
    "baseline": "Qwen/Qwen3-0.6B",
    "resiliparse": "gs://marin-us-east5/checkpoints/medical-resiliparse-qwen3-0.6b-sft-5bbd3e/hf/step-9282",
    "lowreg": "gs://marin-us-east5/checkpoints/medical-resili-lowreg-qwen3-0.6b-aaec7f/hf/step-18565",
    "highreg": "gs://marin-us-east5/checkpoints/medical-resili-highreg-qwen3-0.6b-0cf223/hf/step-9282",
}

all_steps: list[ExecutorStep] = []

for model_name, model_path in MODELS.items():
    discover = False  # all paths point to exact checkpoint dirs
    for batch_idx, batch in enumerate(EVAL_BATCHES):
        step = evaluate_lm_evaluation_harness(
            model_name=f"medical-{model_name}-batch{batch_idx + 1}",
            model_path=model_path,
            evals=batch,
            engine_kwargs=EVAL_ENGINE_KWARGS,
            resource_config=EVAL_RESOURCE,
            apply_chat_template=False,
            discover_latest_checkpoint=discover,
        )
        all_steps.append(step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description=f"Medical eval retry — {len(MODELS)} models x {len(EVAL_BATCHES)} batches",
    )
