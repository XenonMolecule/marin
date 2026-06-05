# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline eval for Qwen3-0.6B-Base on the same math evals as the top3 HP sweep.

Runs the untrained model through the exact same eval config (7 MINERVA MATH
subtasks 4-shot + GSM8K Platinum CoT 8-shot, no chat template, no
engine_kwargs) so the scores are directly comparable to the top3 sweep results.

Runs directly on the Iris TPU worker (no executor/step_runner) to avoid the
fray v1 LocalCluster threading issue where signal.alarm() in math_verify
crashes because it's not in the main thread.

Launch (Iris):
    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
        --tpu v5p-8 --memory 128GB \
        --extra marin:eval --extra marin:tpu --extra marin:vllm \
        -e MARIN_VLLM_MODE native \
        -- python experiments/rephraser/math_0_6b_baseline_eval.py
"""

import logging

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvaluationConfig
from marin.evaluation.run import evaluate

from experiments.rephraser.math_top3_hp_sweep import DOMAIN_MIX_EVALS

logging.basicConfig(level=logging.INFO)

OUTPUT_PATH = "gs://marin-us-central1/evaluation/lm_evaluation_harness/math-top3-baseline-qwen3-0.6b-base-direct"

config = EvaluationConfig(
    evaluator="lm_evaluation_harness",
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    model_name="math-top3-baseline-qwen3-0.6b-base",
    model_path="Qwen/Qwen3-0.6B-Base",
    evaluation_path=OUTPUT_PATH,
    evals=list(DOMAIN_MIX_EVALS),
    discover_latest_checkpoint=False,
    launch_with_ray=False,
    apply_chat_template=False,
)

if __name__ == "__main__":
    evaluate(config)
