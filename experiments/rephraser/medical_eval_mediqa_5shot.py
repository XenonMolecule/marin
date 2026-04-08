# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test mediqa_qa2019_lite with 5-shot to see if fewshot improves generation quality.

The 0-shot results showed resiliparse models regressing vs baseline on ROUGE.
Hypothesis: 0-shot formatting hurts fine-tuned models, but medical knowledge
improves with fewshot examples that anchor the generation format.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-east5-a --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/medical_eval_mediqa_5shot.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")
EVAL_ENGINE_KWARGS = {"max_model_len": 16384, "max_gen_toks": 512}

# 5-shot prompts are ~11k tokens — need higher max_model_len
MEDIQA_5SHOT = [
    EvalTaskConfig(name="mediqa_qa2019_lite", num_fewshot=5, task_alias="mediqa_qa2019_lite_5shot"),
]

MODELS = {
    "baseline": "Qwen/Qwen3-0.6B",
    "resiliparse": "gs://marin-us-east5/checkpoints/medical-resiliparse-qwen3-0.6b-sft-5bbd3e/hf/step-9282",
    "lowreg": "gs://marin-us-east5/checkpoints/medical-resili-lowreg-qwen3-0.6b-aaec7f/hf/step-18565",
    "highreg": "gs://marin-us-east5/checkpoints/medical-resili-highreg-qwen3-0.6b-0cf223/hf/step-9282",
}

all_steps: list[ExecutorStep] = []

for model_name, model_path in MODELS.items():
    step = evaluate_lm_evaluation_harness(
        model_name=f"medical-{model_name}-mediqa-5shot",
        model_path=model_path,
        evals=MEDIQA_5SHOT,
        engine_kwargs=EVAL_ENGINE_KWARGS,
        resource_config=EVAL_RESOURCE,
        apply_chat_template=False,
        discover_latest_checkpoint=False,
    )
    all_steps.append(step)

if __name__ == "__main__":
    executor_main(steps=all_steps, description="Medical mediqa 5-shot test — 4 models")
