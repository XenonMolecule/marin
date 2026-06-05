# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Medical extraction SFT on Qwen3-14B-Base.

Tests whether the medical extraction data works better at 14B scale (where
extraction beat resiliparse for both code and math). Uses 3 HP configs from
the coding 14B sweep (default, best-resiliparse, best-extract) across 2 data
types (V2 extraction + resiliparse) plus a baseline eval = 7 configs.

HP configs (from coding 14B sweep):
  default:       lr=2e-5,  bs=64, wd=0.01, warmup=0.03
  best-resili:   lr=1e-6,  bs=32, wd=0.01, warmup=0.03
  best-extract:  lr=2e-6,  bs=32, wd=0.05, warmup=0.0

Data:
  V2 extraction: 560M tokens (revised prompt with broader medical definition)
  Resiliparse:   2.43B tokens (raw HTML text extraction)

Evals: 7 MMLU generative medical subtasks (5-shot).

Train: v5p-32 (14B needs more memory).  Eval: v5p-8.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/medical_14b_sft.py
"""

import dataclasses
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.qwen3 import qwen3_14b
from experiments.rephraser.extraction_sft_recipe import (
    _run_single_epoch_sft,
    _SFTRunConfig,
)
from experiments.rephraser.medical_extraction_sft_v2 import (
    MEDICAL_EVALS,
)
from experiments.rephraser.medical_extraction_sft_v2 import (
    result as medical_v2_result,
)

# ---------------------------------------------------------------------------
# Qwen3-14B-Base model config (with SFT-appropriate seq_len)
# ---------------------------------------------------------------------------
qwen3_14b_with_rope = dataclasses.replace(
    qwen3_14b,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)


# ---------------------------------------------------------------------------
# HP configs from coding 14B sweep
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HPConfig:
    name: str
    lr: float
    bs: int
    wd: float
    warmup: float


HP_CONFIGS = [
    HPConfig(name="default", lr=2e-5, bs=64, wd=0.01, warmup=0.03),
    HPConfig(name="best-resili", lr=1e-6, bs=32, wd=0.01, warmup=0.03),
    HPConfig(name="best-extract", lr=2e-6, bs=32, wd=0.05, warmup=0.0),
]

# Fixed across all configs
DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096
TRAIN_TPU_TYPE = "v5p-32"

# ---------------------------------------------------------------------------
# Eval config — same medical evals as 0.6B experiments
# ---------------------------------------------------------------------------
EVAL_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 256}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Reuse tokenized data from the 0.6B medical experiments (same tokenizer)
# ---------------------------------------------------------------------------
# V2 Extraction: postprocessed extraction data with revised prompt
extraction_branch = medical_v2_result.extraction_branches[0]
extraction_data = extraction_branch.postprocess_step

extraction_tokenized = default_tokenize(
    name="medical_extract-v2_qwen3-0.6b_sft",
    dataset=output_path_of(extraction_data) / "**/*.jsonl.gz",
    tokenizer="Qwen/Qwen3-0.6B",
    format=TextLmDatasetFormat(),
)

# Resiliparse: plain text from HTML
# Use the tokenize step from the resiliparse train config directly
# to ensure the hash matches the existing cached tokenized data (2061a9).
resili_train_step = medical_v2_result.shared_branches["resiliparse"][0]
resili_tokenized = resili_train_step.config.tokenized_path

DATA_SOURCES = {
    "extract": extraction_tokenized,
    "resili": resili_tokenized,
}

# ---------------------------------------------------------------------------
# Build all steps: 1 baseline + 6 training configs
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

# Baseline eval (no training)
baseline_eval = evaluate_lm_evaluation_harness(
    model_name="medical-14b-baseline-qwen3-14b-base",
    model_path="Qwen/Qwen3-14B-Base",
    evals=MEDICAL_EVALS,
    engine_kwargs=EVAL_ENGINE_KWARGS,
    resource_config=EVAL_RESOURCE,
    apply_chat_template=False,
    discover_latest_checkpoint=False,
)
all_steps.append(baseline_eval)

# 3 HP configs x 2 data types = 6 training + eval jobs
for data_name, tokenized_step in DATA_SOURCES.items():
    for hp in HP_CONFIGS:
        config_name = f"{data_name}-{hp.name}"

        data_config = lm_data_config(training_set=tokenized_step, validation_sets={})

        train_step = ExecutorStep(
            name=f"checkpoints/medical-14b-{config_name}-qwen3-14b-base",
            description=f"Medical 14B SFT: {config_name} (Qwen3-14B-Base)",
            fn=_run_single_epoch_sft,
            config=_SFTRunConfig(
                tokenized_path=tokenized_step,
                data_config=data_config,
                output_path=this_output_path(),
                tags=("medical", "14b", config_name, data_name, "sft", "qwen3-14b-base"),
                model_config=qwen3_14b_with_rope,
                seq_len=SEQ_LEN,
                batch_size=hp.bs,
                learning_rate=hp.lr,
                weight_decay=hp.wd,
                warmup=hp.warmup,
                decay=DECAY,
                lr_schedule=LR_SCHEDULE,
                max_grad_norm=MAX_GRAD_NORM,
                hf_model_name="Qwen/Qwen3-14B-Base",
                checkpoint_path=None,
                pad_tokenizer_to_match_model=True,
                train_tpu_type=TRAIN_TPU_TYPE,
            ),
        )

        eval_step = evaluate_lm_evaluation_harness(
            model_name=f"medical-14b-{config_name}-qwen3-14b-base",
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
        description=f"Medical 14B SFT: {len(HP_CONFIGS)} HP configs x {len(DATA_SOURCES)} data types + baseline = {len(all_steps)} configs",
    )
