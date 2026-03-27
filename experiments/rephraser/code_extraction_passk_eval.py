# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pass@k evaluation for code extraction SFT base model conditions.

Runs humaneval_64 (64 samples per prompt, temperature=0.2, top_p=0.95) on:
  1. Qwen3-0.6B-Base baseline (no SFT)
  2. Plaintext extraction SFT (step-978)
  3. Resiliparse SFT (step-3023)

This gives pass@k for k=[2,8,16,32,64] to determine whether the pass@1
differences (30.5% vs 28.7% vs 26.2%) are statistically significant.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/code_extraction_passk_eval.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \
        experiments/rephraser/code_extraction_passk_eval.py --dry_run true
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# ---------------------------------------------------------------------------
# humaneval_64: 64 samples per prompt, temperature=0.2, pass@k for k=[2,8,16,32,64]
# ---------------------------------------------------------------------------
PASSK_EVALS = [
    EvalTaskConfig(name="humaneval_64", num_fewshot=0, task_alias="humaneval_64_0shot"),
]

ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 1024}
RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Existing checkpoint paths (already trained and evaluated with pass@1)
# ---------------------------------------------------------------------------
BASELINE_MODEL = "Qwen/Qwen3-0.6B-Base"
EXTRACTION_CHECKPOINT = (
    "gs://marin-us-central1/checkpoints/code-extract-plaintext-qwen3-0.6b-base-sft-5bce58/hf/step-978"
)
RESILIPARSE_CHECKPOINT = "gs://marin-us-central1/checkpoints/code-resiliparse-qwen3-0.6b-base-sft-80285d/hf/step-3023"

# ---------------------------------------------------------------------------
# Eval steps
# ---------------------------------------------------------------------------
baseline_eval = evaluate_lm_evaluation_harness(
    model_name="code-qwen3-0.6b-base-baseline-passk",
    model_path=BASELINE_MODEL,
    evals=PASSK_EVALS,
    engine_kwargs=ENGINE_KWARGS,
    resource_config=RESOURCE_CONFIG,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

extraction_eval = evaluate_lm_evaluation_harness(
    model_name="code-extract-plaintext-qwen3-0.6b-base-sft-passk",
    model_path=EXTRACTION_CHECKPOINT,
    evals=PASSK_EVALS,
    engine_kwargs=ENGINE_KWARGS,
    resource_config=RESOURCE_CONFIG,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

resiliparse_eval = evaluate_lm_evaluation_harness(
    model_name="code-resiliparse-qwen3-0.6b-base-sft-passk",
    model_path=RESILIPARSE_CHECKPOINT,
    evals=PASSK_EVALS,
    engine_kwargs=ENGINE_KWARGS,
    resource_config=RESOURCE_CONFIG,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)

ALL_STEPS = [baseline_eval, extraction_eval, resiliparse_eval]

if __name__ == "__main__":
    executor_main(steps=ALL_STEPS, description="HumanEval pass@k eval — base model conditions (64 samples)")
