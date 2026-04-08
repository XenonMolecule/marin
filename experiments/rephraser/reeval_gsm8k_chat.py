# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-evaluate chat-template SFT checkpoints on GSM8K with apply_chat_template=True.

Same as reeval_gsm8k.py but uses chat template for both the GSM8K chat SFT and
the QRA chat SFT checkpoints. These were trained with chat-formatted data, so
the eval should use the chat template too.

Uses vLLM-based evaluation since GSM8K CoT is a generation task.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/reeval_gsm8k_chat.py
"""

from experiments.evals.evals import evaluate_lm_evaluation_harness
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

GSM8K_EVAL = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
]

# max_gen_toks=1024 to avoid exceeding max_model_len when combined with prompts.
GSM8K_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 1024}

# Chat-template SFT checkpoints, evaluated with apply_chat_template=True.
# Requires chat_template in tokenizer_config.json (added to both checkpoints).
CHECKPOINTS = {
    "reeval8-gsm8k-chat-sft": "gs://marin-us-central1/checkpoints/gsm8k-sft-1e20-qwen3-e89008/hf",
    "reeval8-qra-chat-sft": "gs://marin-us-central1/checkpoints/mathhelpforum-qra-sft-1e20-qwen3-c06b54/hf",
}

eval_steps = []
for name, model_path in CHECKPOINTS.items():
    step = evaluate_lm_evaluation_harness(
        model_name=name,
        model_path=model_path,
        evals=GSM8K_EVAL,
        engine_kwargs=GSM8K_ENGINE_KWARGS,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        apply_chat_template=True,
        discover_latest_checkpoint=True,
    )
    eval_steps.append(step)

if __name__ == "__main__":
    executor_main(
        steps=eval_steps,
        description="Re-evaluate chat-template SFT checkpoints on GSM8K (vLLM, chat template).",
    )
