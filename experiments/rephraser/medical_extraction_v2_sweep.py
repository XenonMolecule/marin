# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Medical extraction v2 SFT — full hyperparameter sweep.

Same grid as the v1 extraction sweep but on the v2 extracted data
(revised prompt with broader medical definition).

Grid:
  Phase 1 — LR x batch_size (fixed wd=0.01, wu=0.03):
    lr in {1e-6, 2e-6, 3e-6, 5e-6, 7e-6}, bs in {32, 64}

  Phase 2 — Weight decay x warmup (fixed lr=2e-6, bs=32):
    wd in {0.001, 0.01, 0.05, 0.1}, wu in {0.0, 0.03, 0.1}

Launch (only after tokenize completes):
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/medical_extraction_v2_sweep.py
"""

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.rephraser.extraction_sft_recipe import (
    _SFTRunConfig,
    _run_single_epoch_sft,
)
from experiments.rephraser.medical_extraction_sft_v2 import (
    MEDICAL_EVALS,
    qwen3_0_6b_hd128_with_rope,
    result as medical_v2_result,
)
from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096

EVAL_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 256}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Extraction v2 postprocessed data (reuse from medical v2 experiment)
# ---------------------------------------------------------------------------
extraction_branch = medical_v2_result.extraction_branches[0]
extraction_data = extraction_branch.postprocess_step

extraction_tokenized = default_tokenize(
    name="medical_extract-v2_qwen3-0.6b_sft",
    dataset=output_path_of(extraction_data) / "**/*.jsonl.gz",
    tokenizer="Qwen/Qwen3-0.6B",
    format=TextLmDatasetFormat(),
)

# ---------------------------------------------------------------------------
# Phase 1: LR x batch_size sweep
# ---------------------------------------------------------------------------
LR_BS_CONFIGS = [
    (1e-6, 64),
    (2e-6, 64),
    (3e-6, 64),
    (5e-6, 64),
    (7e-6, 64),
    (1e-6, 32),
    (2e-6, 32),
    (3e-6, 32),
    (5e-6, 32),
]

P1_WEIGHT_DECAY = 0.01
P1_WARMUP = 0.03

# ---------------------------------------------------------------------------
# Phase 2: Weight decay x warmup sweep
# ---------------------------------------------------------------------------
WEIGHT_DECAYS = [0.001, 0.01, 0.05, 0.1]
WARMUPS = [0.0, 0.03, 0.1]

P2_LEARNING_RATE = 2e-6
P2_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Build all runs
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

# Phase 1
for lr, bs in LR_BS_CONFIGS:
    lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    config_name = f"lr{lr_str}_bs{bs}"

    data_config = lm_data_config(training_set=extraction_tokenized, validation_sets={})

    train_step = ExecutorStep(
        name=f"checkpoints/medical-extractv2-p1-{config_name}-qwen3-0.6b",
        description=f"Medical extraction v2 SFT sweep P1 ({config_name})",
        fn=_run_single_epoch_sft,
        config=_SFTRunConfig(
            tokenized_path=extraction_tokenized,
            data_config=data_config,
            output_path=this_output_path(),
            tags=("medical", "extractv2-sweep-p1", config_name, "sft", "qwen3-0.6b"),
            model_config=qwen3_0_6b_hd128_with_rope,
            seq_len=SEQ_LEN,
            batch_size=bs,
            learning_rate=lr,
            weight_decay=P1_WEIGHT_DECAY,
            warmup=P1_WARMUP,
            decay=DECAY,
            lr_schedule=LR_SCHEDULE,
            max_grad_norm=MAX_GRAD_NORM,
            hf_model_name="Qwen/Qwen3-0.6B",
            checkpoint_path=None,
            pad_tokenizer_to_match_model=True,
        ),
    )

    eval_step = evaluate_lm_evaluation_harness(
        model_name=f"medical-extractv2-p1-{config_name}-qwen3-0.6b",
        model_path=output_path_of(train_step, "hf"),
        evals=MEDICAL_EVALS,
        engine_kwargs=EVAL_ENGINE_KWARGS,
        resource_config=EVAL_RESOURCE,
        apply_chat_template=False,
        discover_latest_checkpoint=True,
    )
    all_steps.append(eval_step)

# Phase 2
for wd in WEIGHT_DECAYS:
    for warmup in WARMUPS:
        wd_str = f"{wd}".replace(".", "p")
        warmup_str = f"{warmup}".replace(".", "p")
        config_name = f"wd{wd_str}_wu{warmup_str}"

        data_config = lm_data_config(training_set=extraction_tokenized, validation_sets={})

        train_step = ExecutorStep(
            name=f"checkpoints/medical-extractv2-p2-{config_name}-qwen3-0.6b",
            description=f"Medical extraction v2 SFT sweep P2 ({config_name})",
            fn=_run_single_epoch_sft,
            config=_SFTRunConfig(
                tokenized_path=extraction_tokenized,
                data_config=data_config,
                output_path=this_output_path(),
                tags=("medical", "extractv2-sweep-p2", config_name, "sft", "qwen3-0.6b"),
                model_config=qwen3_0_6b_hd128_with_rope,
                seq_len=SEQ_LEN,
                batch_size=P2_BATCH_SIZE,
                learning_rate=P2_LEARNING_RATE,
                weight_decay=wd,
                warmup=warmup,
                decay=DECAY,
                lr_schedule=LR_SCHEDULE,
                max_grad_norm=MAX_GRAD_NORM,
                hf_model_name="Qwen/Qwen3-0.6B",
                checkpoint_path=None,
                pad_tokenizer_to_match_model=True,
            ),
        )

        eval_step = evaluate_lm_evaluation_harness(
            model_name=f"medical-extractv2-p2-{config_name}-qwen3-0.6b",
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
        description=f"Medical extraction v2 SFT — full HP sweep ({len(all_steps)} configs)",
    )
