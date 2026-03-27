# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Medical resiliparse SFT on Qwen3-0.6B — hyperparameter variants.

Tests three HP settings on medical resiliparse data:
1. Default: lr=2e-5, bs=64, wd=0.01, wu=0.03 (already run by main experiment)
2. Low-reg: lr=2e-6, bs=32, wd=0.001, wu=0.0
3. High-reg: lr=2e-6, bs=64, wd=0.1, wu=0.03

This adds low-reg and high-reg configs. Default is handled by medical_extraction_sft.py.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/medical_resiliparse_sft_hpsweep.py
"""

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.rephraser.extraction_sft_recipe import (
    TrainHyperparams,
    _SFTRunConfig,
    _run_single_epoch_sft,
)
from experiments.rephraser.medical_extraction_sft import (
    MEDICAL_EVALS,
    qwen3_0_6b_hd128_with_rope,
    result as medical_result,
)
from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

# ---------------------------------------------------------------------------
# Hyperparameter settings
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

EVAL_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 1024}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Resiliparse tokenized data (reuse from medical experiment)
# ---------------------------------------------------------------------------
resili_processed = medical_result.resiliparse_step
resili_tokenized = default_tokenize(
    name="medical_resiliparse_qwen3-0.6b_sft",
    dataset=output_path_of(resili_processed) / "**/*.jsonl.gz",
    tokenizer="Qwen/Qwen3-0.6B",
    format=TextLmDatasetFormat(),
)

# ---------------------------------------------------------------------------
# Build runs
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

for hp_name, hp in HP_CONFIGS.items():
    data_config = lm_data_config(training_set=resili_tokenized, validation_sets={})

    train_step = ExecutorStep(
        name=f"checkpoints/medical-resili-{hp_name}-qwen3-0.6b",
        description=f"Medical resiliparse SFT ({hp_name})",
        fn=_run_single_epoch_sft,
        config=_SFTRunConfig(
            tokenized_path=resili_tokenized,
            data_config=data_config,
            output_path=this_output_path(),
            tags=("medical", f"resili-{hp_name}", "resiliparse", "sft", "qwen3-0.6b"),
            model_config=qwen3_0_6b_hd128_with_rope,
            seq_len=hp.seq_len,
            batch_size=hp.batch_size,
            learning_rate=hp.learning_rate,
            weight_decay=hp.weight_decay,
            warmup=hp.warmup,
            decay=hp.decay,
            lr_schedule=hp.lr_schedule,
            max_grad_norm=hp.max_grad_norm,
            hf_model_name="Qwen/Qwen3-0.6B",
            checkpoint_path=None,
            pad_tokenizer_to_match_model=True,
        ),
    )

    eval_step = evaluate_lm_evaluation_harness(
        model_name=f"medical-resili-{hp_name}-qwen3-0.6b",
        model_path=output_path_of(train_step, "hf"),
        evals=MEDICAL_EVALS,
        engine_kwargs=EVAL_ENGINE_KWARGS,
        resource_config=EVAL_RESOURCE,
        apply_chat_template=False,
        discover_latest_checkpoint=True,
    )

    all_steps.append(eval_step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description=f"Medical resiliparse SFT — HP variants ({len(HP_CONFIGS)} configs)",
    )
