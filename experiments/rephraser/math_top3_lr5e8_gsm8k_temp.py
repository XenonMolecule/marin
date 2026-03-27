# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-run GSM8K eval with temperature=0.01 to measure sampling variance.

The original and reproduction runs both got GSM8K_flex=64.27% with greedy
decoding (temperature=0.0). This script re-evaluates the same checkpoint
with temperature=0.01 to see how much variance exists.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/math_top3_lr5e8_gsm8k_temp.py
"""

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig
from marin.evaluation.run import evaluate
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path

# The already-trained model from the reproduction run
MODEL_PATH = InputName.hardcoded(
    "gs://marin-us-central1/checkpoints/math-top3-extract-lr5e-8_bs64-repro-qwen3-0.6b-base-2c9997/hf/step-966"
)

GSM8K_EVAL = [
    EvalTaskConfig(name="gsm8k_platinum_cot", num_fewshot=8, task_alias="gsm8k_platinum_cot_8shot"),
]

eval_step = ExecutorStep(
    name="evaluation/lm_evaluation_harness/math-top3-extract-lr5e-8_bs64-repro-gsm8k-temp001",
    fn=evaluate,
    config=EvaluationConfig(
        evaluator="lm_evaluation_harness",
        model_name="math-top3-extract-lr5e-8_bs64-repro-gsm8k-temp001",
        model_path=MODEL_PATH,
        evaluation_path=this_output_path(),
        evals=GSM8K_EVAL,
        launch_with_ray=True,
        discover_latest_checkpoint=False,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        apply_chat_template=False,
        generation_params={"temperature": 0.01},
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description="GSM8K eval with temperature=0.01 to measure sampling variance",
    )
