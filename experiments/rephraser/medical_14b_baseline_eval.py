# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline evaluation of Qwen3-14B-Base (no SFT) on medical benchmarks.

Standalone job for debugging — gives direct view of eval logs.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/medical_14b_baseline_eval.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.rephraser.medical_extraction_sft_v2 import MEDICAL_EVALS
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

EVAL_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 256}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

baseline_eval = evaluate_lm_evaluation_harness(
    model_name="medical-14b-baseline-qwen3-14b-base",
    model_path="Qwen/Qwen3-14B-Base",
    evals=MEDICAL_EVALS,
    engine_kwargs=EVAL_ENGINE_KWARGS,
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[baseline_eval],
        description="Baseline evaluation of Qwen3-14B-Base on medical benchmarks",
    )
