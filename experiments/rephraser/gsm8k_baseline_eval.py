# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline GSM8K evaluation on the pretrained exp2166 checkpoint (no SFT).

Evaluates the exp2166 final checkpoint (~1.385B Qwen3, Llama3 tokenizer) on
GSM8K test set using 8-shot CoT prompting, WITHOUT any fine-tuning. This
serves as the baseline for comparison against the SFT experiments
(gsm8k_sft.py, mathhelpforum_sft.py, mathhelpforum_resiliparse_sft.py).

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/gsm8k_baseline_eval.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/gsm8k_baseline_eval.py --dry_run true
"""

from experiments.evals.evals import default_eval
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# HF export of the exp2166 final checkpoint (no post-training)
BASELINE_HF_PATH = (
    "gs://marin-us-central1/" "exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/hf/step-44758"
)

GSM8K_EVAL = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
]

gsm8k_baseline_eval_step = default_eval(
    step=BASELINE_HF_PATH,
    evals=GSM8K_EVAL,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    discover_latest_checkpoint=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[gsm8k_baseline_eval_step],
        description="Baseline GSM8K eval on exp2166 pretrained checkpoint (no SFT).",
    )
