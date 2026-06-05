# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pass@k evaluation for Qwen3-0.6B-Base baseline (no SFT).

Resubmit of the baseline condition only — the extraction and resiliparse
conditions already succeeded in the prior run.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/code_extraction_passk_eval_baseline.py
"""

from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

from experiments.evals.evals import evaluate_lm_evaluation_harness

PASSK_EVALS = [
    EvalTaskConfig(name="humaneval_64", num_fewshot=0, task_alias="humaneval_64_0shot"),
]

ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 1024}
RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8")

baseline_eval = evaluate_lm_evaluation_harness(
    model_name="code-qwen3-0.6b-base-baseline-passk",
    model_path="Qwen/Qwen3-0.6B-Base",
    evals=PASSK_EVALS,
    engine_kwargs=ENGINE_KWARGS,
    resource_config=RESOURCE_CONFIG,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

if __name__ == "__main__":
    executor_main(steps=[baseline_eval], description="HumanEval pass@k eval — baseline only (resubmit)")
