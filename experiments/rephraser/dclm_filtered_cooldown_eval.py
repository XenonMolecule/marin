# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the DCLM-filtered cooldown checkpoint with lm-evaluation-harness.

Runs CORE_TASKS on the step-9758 HF export from the DCLM-filtered cooldown
training run. This model was trained by applying the full DCLM-Baseline
filtering pipeline to raw Common Crawl WARCs and mixing the surviving text
with NemotronCooldown data during the cooldown phase.

Compare against:
  - Pre-cooldown baseline (step 35,000): scaling_1e20_baseline_eval.py
  - Original nemotron cooldown (step 44,758): scaling_1e20_final_eval.py
  - Rephraser cooldown experiments: rephraser_cooldown.py

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/dclm_filtered_cooldown_eval.py
"""

from fray.cluster import ResourceConfig

from experiments.evals.evals import default_eval
from marin.execution.executor import executor_main

MODEL_PATH = "gs://marin-us-central1/cooldown-dclm-filtered-v1-54cdb7/hf/step-9758"

if __name__ == "__main__":
    eval_step = default_eval(
        step=MODEL_PATH,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        discover_latest_checkpoint=False,
    )
    executor_main(steps=[eval_step])
