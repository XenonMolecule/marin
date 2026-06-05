# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-run for CPU inference cooldown: stage all prerequisites without running inference.

Runs the same prerequisite steps as rephraser_cooldown_cpu.py:
  0. Build llama.cpp (compile llama-server binary, upload to GCS)
  1. Download & extract WARCs (reuses cached output from other sweeps)
  1b. Filter HTML by token length (reuses cached output)
  5. Extract NemotronCooldown tokens (reuses cached output)

This ensures all dependencies are ready before the full pipeline is launched.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -- python experiments/rephraser/rephraser_cooldown_cpu_prerun.py
"""

from marin.execution.executor import executor_main

from experiments.rephraser.rephraser_cooldown_cpu import (
    build_llamacpp_step,
    download_warcs,
    extract_cooldown_step,
    filter_html,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            build_llamacpp_step,
            download_warcs,
            filter_html,
            extract_cooldown_step,
        ],
        description="Pre-run: stage all prerequisites for CPU inference cooldown (no inference).",
    )
