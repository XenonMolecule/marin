# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MBPP evaluation for the three base model conditions.

Uses HuggingFace Hub paths to bypass the broken runai_streamer on the cluster.

Runs MBPP 0-shot and 3-shot on:
  1. Baseline (Qwen3-0.6B-Base, no SFT)
  2. V3 commented extraction SFT (base)
  3. Resiliparse SFT (base)

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/mbpp_base_eval.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

MBPP_EVALS = [
    EvalTaskConfig(name="mbpp", num_fewshot=0, task_alias="mbpp_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=3, task_alias="mbpp_3shot"),
]

# HF checkpoints updated with max_position_embeddings=32768 (matching Qwen3-0.6B-Base).
# 8192 context leaves plenty of room for 3-shot MBPP prompts (~4k tokens) + generation.
ENGINE_KWARGS = {"max_model_len": 8192, "max_gen_toks": 512}
RESOURCE = ResourceConfig.with_tpu("v5p-8")

# 1. Baseline (Qwen3-0.6B-Base, no SFT)
baseline_eval = evaluate_lm_evaluation_harness(
    model_name="code-qwen3-0.6b-base-mbpp-baseline",
    model_path="Qwen/Qwen3-0.6B-Base",
    evals=MBPP_EVALS,
    engine_kwargs=ENGINE_KWARGS,
    resource_config=RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

# 2. V3 commented extraction SFT (base) — HF mirror of gs://marin-us-central1/checkpoints/code-extract-commented-qwen3-0.6b-base-sft-bedd7b/hf/step-994
v3_eval = evaluate_lm_evaluation_harness(
    model_name="code-extract-commented-qwen3-0.6b-base-mbpp-sft-hf",
    model_path="MichaelR207/code-extract-commented-qwen3-0.6b-base-sft",
    evals=MBPP_EVALS,
    engine_kwargs=ENGINE_KWARGS,
    resource_config=RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

# 3. Resiliparse SFT (base) — HF mirror of gs://marin-us-central1/checkpoints/code-resiliparse-qwen3-0.6b-base-sft-80285d/hf/step-3023
resili_eval = evaluate_lm_evaluation_harness(
    model_name="code-resiliparse-qwen3-0.6b-base-mbpp-sft-hf",
    model_path="MichaelR207/code-resiliparse-qwen3-0.6b-base-sft",
    evals=MBPP_EVALS,
    engine_kwargs=ENGINE_KWARGS,
    resource_config=RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[baseline_eval, v3_eval, resili_eval],
        description="MBPP eval (0-shot + 3-shot) for base model conditions",
    )
