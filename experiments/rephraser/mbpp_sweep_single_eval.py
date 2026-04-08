# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single sweep config eval — isolate MBPP failure.

Evaluates ONLY lr=5e-6, bs=64 (the best HumanEval config) on
HumanEval + MBPP 0-shot + MBPP 3-shot.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/mbpp_sweep_single_eval.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

EVAL_TASKS = [
    EvalTaskConfig(name="humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=0, task_alias="mbpp_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=3, task_alias="mbpp_3shot"),
]

# The checkpoint has max_position_embeddings=4096 (from training seq_len), but
# the base model supports 32768. Use max_model_len=8192 to fit 3-shot MBPP
# prompts (up to ~4300 tokens) plus generation headroom.
ENGINE_KWARGS = {"max_model_len": 8192, "max_gen_toks": 512}

# Best sweep config: lr=5e-6, bs=64
eval_step = evaluate_lm_evaluation_harness(
    model_name="code-v3-sweep-lr5e-6-bs64-single-eval",
    model_path="gs://marin-us-central1/checkpoints/code-v3-sweep-lr5e-6_bs64-qwen3-0.6b-base-30eabe/hf",
    evals=EVAL_TASKS,
    engine_kwargs=ENGINE_KWARGS,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=False,
    discover_latest_checkpoint=True,
)

if __name__ == "__main__":
    executor_main(steps=[eval_step], description="Single sweep eval: lr=5e-6, bs=64")
