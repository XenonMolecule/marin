# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the short rephraser cooldown checkpoint with lm-evaluation-harness.

Runs CORE_TASKS on the step-4999 HF export from the short rephraser cooldown
(spec d7d976d3, 1B nemotron + ~300M rephraser, 5000-step linear decay).

Compare against:
  - Pre-cooldown baseline (step 35,000): scaling_1e20_baseline_eval.py
  - Original nemotron cooldown (step 44,758): scaling_1e20_final_eval.py
  - DCLM-filtered cooldown: dclm_filtered_cooldown_eval.py

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/short_cooldown/eval_rephraser.py
"""

from fray.cluster import ResourceConfig

from experiments.evals.evals import default_eval
from marin.execution.executor import executor_main

MODEL_PATH = "gs://marin-us-central1/short-cooldown-rephraser-d7d976d3-v2-eebec0/hf/step-4999"

if __name__ == "__main__":
    eval_step = default_eval(
        step=MODEL_PATH,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        discover_latest_checkpoint=False,
    )
    executor_main(steps=[eval_step])
