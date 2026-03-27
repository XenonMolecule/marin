# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""mediqa_qa2019_lite with 1-shot and 2-shot for baseline + best configs."""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")
EVAL_ENGINE_KWARGS = {"max_model_len": 16384, "max_gen_toks": 512}

MEDIQA_1SHOT = [
    EvalTaskConfig(name="mediqa_qa2019_lite", num_fewshot=1, task_alias="mediqa_qa2019_lite_1shot"),
]

MEDIQA_2SHOT = [
    EvalTaskConfig(name="mediqa_qa2019_lite", num_fewshot=2, task_alias="mediqa_qa2019_lite_2shot"),
]

MODELS = {
    "baseline": "Qwen/Qwen3-0.6B",
    "resili-best": (
        "gs://marin-us-east5/checkpoints/medical-resili-p1-lr7e-6_bs64-qwen3-0.6b-24b1e7/hf/step-9282"
    ),
    "extract-best": (
        "gs://marin-us-east5/checkpoints/medical-extract-p1-lr5e-6_bs32-qwen3-0.6b-c967ff/hf/step-3573"
    ),
}

all_steps: list[ExecutorStep] = []

for model_name, model_path in MODELS.items():
    for shot_evals, shot_label in [(MEDIQA_1SHOT, "1shot"), (MEDIQA_2SHOT, "2shot")]:
        step = evaluate_lm_evaluation_harness(
            model_name=f"medical-{model_name}-mediqa-{shot_label}",
            model_path=model_path,
            evals=shot_evals,
            engine_kwargs=EVAL_ENGINE_KWARGS,
            resource_config=EVAL_RESOURCE,
            apply_chat_template=False,
            discover_latest_checkpoint=False,
        )
        all_steps.append(step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="mediqa 1-shot + 2-shot — baseline + best resili + best extract",
    )
