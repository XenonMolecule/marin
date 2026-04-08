# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-evaluate SFT checkpoints on GSM8K using vLLM-based evaluation.

GSM8K CoT is a generation task (generate_until), so we use the vLLM evaluator
rather than Levanter's inference engine (which is designed for loglikelihood tasks).

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/reeval_gsm8k.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

GSM8K_EVAL = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
]

# max_gen_toks=1024 (not 4096) to avoid exceeding max_model_len when combined with
# 8-shot GSM8K prompts (~700 tokens). vLLM's /completions endpoint returns 400 if
# max_tokens + prompt_tokens > max_model_len.
GSM8K_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 1024}

# Re-evaluate key checkpoints with plain text eval (no chat template).
CHECKPOINTS = {
    "reeval8-baseline": "gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0/hf",
    "reeval8-gsm8k-plaintext-sft": "gs://marin-us-central1/checkpoints/gsm8k-plaintext-sft-1e20-qwen3-a40a32/hf",
    "reeval8-qra-plaintext-sft": (
        "gs://marin-us-central1/checkpoints/mathhelpforum-qra-plaintext-sft-1e20-qwen3-8b0030/hf"
    ),
    "reeval8-resiliparse-sft": "gs://marin-us-central1/checkpoints/mathhelpforum-resiliparse-sft-1e20-qwen3-56a96f/hf",
}

eval_steps = []
for name, model_path in CHECKPOINTS.items():
    step = evaluate_lm_evaluation_harness(
        model_name=name,
        model_path=model_path,
        evals=GSM8K_EVAL,
        engine_kwargs=GSM8K_ENGINE_KWARGS,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        apply_chat_template=False,
        discover_latest_checkpoint=True,
    )
    eval_steps.append(step)

if __name__ == "__main__":
    executor_main(
        steps=eval_steps,
        description="Re-evaluate SFT checkpoints on GSM8K (vLLM).",
    )
