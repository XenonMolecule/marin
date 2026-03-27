# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Debug medical eval failures — test minimal configs to isolate the vLLM 500 error.

The error message is just {"message":"25"} which is cryptic. Test hypotheses:
1. Is it specific to 0-shot? Try 5-shot.
2. Is it specific to medical tasks? Try arc_easy (known working).
3. Is it the model? Try just baseline Qwen3-0.6B.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-east5-a --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/medical_eval_debug.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main

EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

all_steps: list[ExecutorStep] = []

# Test 1: arc_easy 0-shot (known working task) with baseline model
all_steps.append(evaluate_lm_evaluation_harness(
    model_name="debug-baseline-arc-0shot",
    model_path="Qwen/Qwen3-0.6B",
    evals=[EvalTaskConfig("arc_easy", 0, task_alias="arc_easy_0shot")],
    engine_kwargs={"max_model_len": 4096, "max_gen_toks": 1024},
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
))

# Test 2: mmlu_anatomy 0-shot (small medical task, 135 questions)
all_steps.append(evaluate_lm_evaluation_harness(
    model_name="debug-baseline-anatomy-0shot",
    model_path="Qwen/Qwen3-0.6B",
    evals=[EvalTaskConfig("mmlu_anatomy", 0, task_alias="mmlu_anatomy_0shot")],
    engine_kwargs={"max_model_len": 4096, "max_gen_toks": 1024},
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
))

# Test 3: mmlu_anatomy 5-shot (does fewshot fix it?)
all_steps.append(evaluate_lm_evaluation_harness(
    model_name="debug-baseline-anatomy-5shot",
    model_path="Qwen/Qwen3-0.6B",
    evals=[EvalTaskConfig("mmlu_anatomy", 5, task_alias="mmlu_anatomy_5shot")],
    engine_kwargs={"max_model_len": 4096, "max_gen_toks": 1024},
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
))

# Test 4: medmcqa 0-shot (the big one that always fails)
all_steps.append(evaluate_lm_evaluation_harness(
    model_name="debug-baseline-medmcqa-0shot",
    model_path="Qwen/Qwen3-0.6B",
    evals=[EvalTaskConfig("medmcqa", 0, task_alias="medmcqa_0shot")],
    engine_kwargs={"max_model_len": 4096, "max_gen_toks": 1024},
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
))

# Test 5: medmcqa with lower max_gen_toks (maybe 1024 gen tokens is the issue?)
all_steps.append(evaluate_lm_evaluation_harness(
    model_name="debug-baseline-medmcqa-lowgen",
    model_path="Qwen/Qwen3-0.6B",
    evals=[EvalTaskConfig("medmcqa", 0, task_alias="medmcqa_0shot")],
    engine_kwargs={"max_model_len": 4096, "max_gen_toks": 256},
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
))

if __name__ == "__main__":
    executor_main(steps=all_steps, description="Debug medical eval — 5 test configs")
