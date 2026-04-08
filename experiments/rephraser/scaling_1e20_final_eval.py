# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the FINAL 1e20 scaling ladder checkpoint (with original nemotron cooldown).

Runs CORE_TASKS on the step-44758 HF export from the exp2166 nemotron 1e20 run.
This is the end-of-training checkpoint after the full WSD schedule including the
original nemotron-only cooldown. It tells us what lm-eval scores the model gets
when we DON'T intervene with new training data during cooldown.

Compare these numbers against:
  - Pre-cooldown baseline (step 35,000): scaling_1e20_baseline_eval.py
  - Rephraser cooldown experiments: rephraser_cooldown.py

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/scaling_1e20_final_eval.py
"""

from fray.cluster import ResourceConfig

from experiments.evals.evals import default_eval
from marin.execution.executor import executor_main

# Step-44758 HF export: final checkpoint after full WSD schedule + nemotron cooldown
MODEL_PATH = "gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/hf/step-44758"

if __name__ == "__main__":
    eval_step = default_eval(
        step=MODEL_PATH,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        discover_latest_checkpoint=False,
    )
    executor_main(steps=[eval_step])
