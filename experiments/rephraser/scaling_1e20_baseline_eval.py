# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline evaluation of the 1e20 scaling ladder checkpoint before cooldown.

Runs CORE_TASKS on the step-35000 checkpoint from the exp2166 nemotron 1e20 run.
Step 35,000 is right before cooldown begins (~step 35,808), giving us the
pre-cooldown reference point for comparing against rephraser cooldown experiments.

No HF export exists at step-35000 (available exports: 10k, 20k, 30k, 40k, 44758),
so we first convert the raw Levanter checkpoint to HF format, then evaluate.

Compare these numbers against:
  - Original run final eval (step 44,758): loss=2.828, bpb=0.966, paloma=1.1105
  - Rephraser cooldown experiments in rephraser_cooldown.py

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/scaling_1e20_baseline_eval.py
"""

from fray.cluster import ResourceConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import executor_main, output_path_of
from marin.export import convert_checkpoint_to_hf_step

from experiments.evals.evals import default_eval
from experiments.rephraser.rephraser_cooldown import scaling_1e20_qwen3

# Raw Levanter checkpoint at step-35000 (right before cooldown ~step 35,808)
CHECKPOINT_PATH = (
    "gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-35000"
)

# Step 1: Convert Levanter checkpoint to HF format (CPU mode avoids mesh/TPU issues)
# The model was trained with the Llama3 tokenizer (128,256 vocab), not Qwen3's default (32,000).
hf_export_step = convert_checkpoint_to_hf_step(
    name="hf/scaling-1e20-step-35000",
    checkpoint_path=CHECKPOINT_PATH,
    model=scaling_1e20_qwen3,
    trainer=TrainerConfig(),
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    use_cpu=True,
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
)

# Step 2: Evaluate the exported HF checkpoint
eval_step = default_eval(
    step=output_path_of(hf_export_step),
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    discover_latest_checkpoint=False,
)

if __name__ == "__main__":
    executor_main(steps=[hf_export_step, eval_step])
