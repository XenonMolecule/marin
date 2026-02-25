# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline evaluation of Llama 3.2 1B on CORE_TASKS (no training).

Runs the exact same eval suite used in the rephraser spec sweep midtraining
so we have a pre-training reference point. Compare these numbers against
the midtrained checkpoints in W&B.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/llama_3_2_1b_baseline_eval.py
"""

from fray.cluster import ResourceConfig

from experiments.evals.evals import default_eval
from experiments.models import llama_3_2_1b
from marin.execution.executor import executor_main

if __name__ == "__main__":
    eval_step = default_eval(
        step=llama_3_2_1b,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        # CORE_TASKS is the default when evals=None — same suite used in rephraser_sweep.py
        discover_latest_checkpoint=False,
    )
    executor_main(steps=[eval_step])
