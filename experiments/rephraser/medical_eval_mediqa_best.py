# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""mediqa_qa2019_lite eval on best resiliparse + best extraction configs."""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")
EVAL_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 512}

MEDIQA_EVALS = [
    EvalTaskConfig(name="mediqa_qa2019_lite", num_fewshot=0, task_alias="mediqa_qa2019_lite_0shot"),
]

MODELS = {
    "resili-best-lr7e6-bs64": (
        "gs://marin-us-east5/checkpoints/medical-resili-p1-lr7e-6_bs64-qwen3-0.6b-24b1e7/hf/step-9282"
    ),
    "extract-best-lr5e6-bs32": (
        "gs://marin-us-east5/checkpoints/medical-extract-p1-lr5e-6_bs32-qwen3-0.6b-c967ff/hf/step-3573"
    ),
}

all_steps: list[ExecutorStep] = []

for model_name, model_path in MODELS.items():
    step = evaluate_lm_evaluation_harness(
        model_name=f"medical-{model_name}-mediqa",
        model_path=model_path,
        evals=MEDIQA_EVALS,
        engine_kwargs=EVAL_ENGINE_KWARGS,
        resource_config=EVAL_RESOURCE,
        apply_chat_template=False,
        discover_latest_checkpoint=False,
    )
    all_steps.append(step)

if __name__ == "__main__":
    executor_main(steps=all_steps, description="mediqa eval — best resiliparse + best extraction")
