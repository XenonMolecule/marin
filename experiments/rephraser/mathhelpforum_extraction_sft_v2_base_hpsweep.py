# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Math extraction SFT v2 on Qwen3-0.6B-Base — hyperparameter variants.

Runs the same math extraction pipeline with three hyperparameter settings
discovered from the code extraction sweep:

1. Default: lr=2e-5, bs=64, wd=0.01, wu=0.03
2. Low-reg (best for V3 extraction): lr=2e-6, bs=32, wd=0.001, wu=0.0
3. High-reg (best for resiliparse): lr=2e-6, bs=64, wd=0.1, wu=0.03

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/mathhelpforum_extraction_sft_v2_base_hpsweep.py
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness
from experiments.rephraser.extraction_sft_recipe import (
    TrainHyperparams,
    _run_single_epoch_sft,
    _SFTRunConfig,
)

# Import model config and evals from the math experiment
from experiments.rephraser.mathhelpforum_extraction_sft_v2 import (
    MATH_EVALS,
    qwen3_0_6b_hd128_with_rope,
)
from experiments.rephraser.mathhelpforum_extraction_sft_v2_base import result as math_result

# ---------------------------------------------------------------------------
# Hyperparameter settings to test
# ---------------------------------------------------------------------------
HP_CONFIGS = {
    "lowreg": TrainHyperparams(
        batch_size=32,
        learning_rate=2e-6,
        weight_decay=0.001,
        warmup=0.0,
    ),
    "highreg": TrainHyperparams(
        batch_size=64,
        learning_rate=2e-6,
        weight_decay=0.1,
        warmup=0.03,
    ),
}

EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Reuse tokenized data from the extraction branch
# ---------------------------------------------------------------------------
math_extraction_branch = math_result.extraction_branches[0]  # "unified" extraction
tokenized_step = math_extraction_branch.tokenize_step

# ---------------------------------------------------------------------------
# Build sweep runs
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

for hp_name, hp in HP_CONFIGS.items():
    data_config = lm_data_config(training_set=tokenized_step, validation_sets={})

    train_step = ExecutorStep(
        name=f"checkpoints/math-v2-{hp_name}-qwen3-0.6b-base",
        description=f"Math extraction SFT v2 ({hp_name})",
        fn=_run_single_epoch_sft,
        config=_SFTRunConfig(
            tokenized_path=tokenized_step,
            data_config=data_config,
            output_path=this_output_path(),
            tags=("math", f"v2-{hp_name}", "extraction", "sft", "qwen3-0.6b-base"),
            model_config=qwen3_0_6b_hd128_with_rope,
            seq_len=hp.seq_len,
            batch_size=hp.batch_size,
            learning_rate=hp.learning_rate,
            weight_decay=hp.weight_decay,
            warmup=hp.warmup,
            decay=hp.decay,
            lr_schedule=hp.lr_schedule,
            max_grad_norm=hp.max_grad_norm,
            hf_model_name="Qwen/Qwen3-0.6B-Base",
            checkpoint_path=None,
            pad_tokenizer_to_match_model=True,
        ),
    )

    # Use Levanter evaluator (JAX-native) to avoid vLLM Docker runai_streamer bug
    eval_step = evaluate_levanter_lm_evaluation_harness(
        model_name=f"math-v2-{hp_name}-qwen3-0.6b-base",
        model_path=output_path_of(train_step, "hf"),
        evals=MATH_EVALS,
        resource_config=EVAL_RESOURCE,
        apply_chat_template=False,
        discover_latest_checkpoint=True,
    )

    all_steps.append(eval_step)

# Also include the existing default run's eval steps for comparison
all_steps.extend(math_result.all_eval_steps)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description=f"Math extraction SFT v2 — HP variants ({len(HP_CONFIGS)} new + default)",
    )
