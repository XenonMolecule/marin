# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Low-LR extension sweep for math SFT on top3 domains.

The HP sweep (math_top3_hp_sweep.py) found that lower LR consistently improves
math SFT quality, with 5e-7 as the best of the tested range. This sweep pushes
below 5e-7 to find where diminishing returns kick in.

Grid design (7 configs):
  Core LR ladder at bs=64 (5 configs):
    3e-7, 2e-7, 1e-7, 5e-8, 2e-8
  BS variation at promising LRs (2 configs):
    lr=2e-7/bs=16, lr=1e-7/bs=16

All configs use wd=0.10, warmup=0.03 (matching the sweep defaults).
Total runs: 14 train + 14 eval (7 HP × 2 data types).

Reference from prior sweep (extraction, best config):
  lr=5e-7, bs=64, wd=0.01: avg_minerva=30.3%, GSM8K_flex=60.3%

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/math_top3_low_lr_sweep.py
"""

import dataclasses

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.rephraser.extraction_sft_recipe import (
    TrainHyperparams,
    _SFTRunConfig,
    _run_single_epoch_sft,
)
from experiments.rephraser.mathhelpforum_extraction_sft_v2_base import result as math_result
from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config
from marin.transform.filter_by_domain import FilterByDomainConfig, filter_by_domain

# ---------------------------------------------------------------------------
# Model config (same as HP sweep)
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)

# ---------------------------------------------------------------------------
# Top 3 domains (V1 winner)
# ---------------------------------------------------------------------------
TOP_3 = (
    "brainly.com",
    "jiskha.com",
    "mathhelpforum.com",
)

# ---------------------------------------------------------------------------
# HP configs: push LR below 5e-7
# ---------------------------------------------------------------------------
HP_CONFIGS: dict[str, TrainHyperparams] = {
    # Core LR ladder at bs=64
    "lr3e-7_bs64": TrainHyperparams(
        batch_size=64,
        learning_rate=3e-7,
        weight_decay=0.1,
        warmup=0.03,
    ),
    "lr2e-7_bs64": TrainHyperparams(
        batch_size=64,
        learning_rate=2e-7,
        weight_decay=0.1,
        warmup=0.03,
    ),
    "lr1e-7_bs64": TrainHyperparams(
        batch_size=64,
        learning_rate=1e-7,
        weight_decay=0.1,
        warmup=0.03,
    ),
    "lr5e-8_bs64": TrainHyperparams(
        batch_size=64,
        learning_rate=5e-8,
        weight_decay=0.1,
        warmup=0.03,
    ),
    "lr2e-8_bs64": TrainHyperparams(
        batch_size=64,
        learning_rate=2e-8,
        weight_decay=0.1,
        warmup=0.03,
    ),
    # BS variation at promising LRs
    "lr2e-7_bs16": TrainHyperparams(
        batch_size=16,
        learning_rate=2e-7,
        weight_decay=0.1,
        warmup=0.03,
    ),
    "lr1e-7_bs16": TrainHyperparams(
        batch_size=16,
        learning_rate=1e-7,
        weight_decay=0.1,
        warmup=0.03,
    ),
}

# ---------------------------------------------------------------------------
# Evals — same as HP sweep for comparability
# ---------------------------------------------------------------------------
DOMAIN_MIX_EVALS = [
    EvalTaskConfig(name="minerva_math_algebra", num_fewshot=4, task_alias="minerva_math_algebra_4shot"),
    EvalTaskConfig(name="minerva_math_prealgebra", num_fewshot=4, task_alias="minerva_math_prealgebra_4shot"),
    EvalTaskConfig(
        name="minerva_math_counting_and_prob",
        num_fewshot=4,
        task_alias="minerva_math_counting_and_prob_4shot",
    ),
    EvalTaskConfig(name="minerva_math_geometry", num_fewshot=4, task_alias="minerva_math_geometry_4shot"),
    EvalTaskConfig(
        name="minerva_math_intermediate_algebra",
        num_fewshot=4,
        task_alias="minerva_math_intermediate_algebra_4shot",
    ),
    EvalTaskConfig(name="minerva_math_num_theory", num_fewshot=4, task_alias="minerva_math_num_theory_4shot"),
    EvalTaskConfig(name="minerva_math_precalc", num_fewshot=4, task_alias="minerva_math_precalc_4shot"),
    EvalTaskConfig(name="gsm8k_platinum_cot", num_fewshot=8, task_alias="gsm8k_platinum_cot_8shot"),
]
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Source data (reuse same filtered/tokenized artifacts from HP sweep)
# ---------------------------------------------------------------------------
extraction_branch = math_result.extraction_branches[0]  # "unified"
postprocessed_step = extraction_branch.postprocess_step
resiliparse_step = math_result.resiliparse_step

# ---------------------------------------------------------------------------
# Extraction top3: filter -> tokenize (reuses existing artifacts via same step names)
# ---------------------------------------------------------------------------
extract_filter_config = FilterByDomainConfig(
    input_path=postprocessed_step / "*.jsonl.gz",
    output_path=this_output_path(),
    blocked_domains=[],
    allowed_domains=list(TOP_3),
)
extract_filter_step = ExecutorStep(
    name="filtered/math_mix_top3",
    description="Domain filter: top3 (brainly + jiskha + mathhelpforum) — extraction",
    fn=filter_by_domain,
    config=extract_filter_config,
)

extract_tokenized = default_tokenize(
    name="math_mix_top3_qwen3-0.6b-base_sft",
    dataset=extract_filter_step / "**/*.jsonl.gz",
    tokenizer="Qwen/Qwen3-0.6B-Base",
    format=TextLmDatasetFormat(),
)

# ---------------------------------------------------------------------------
# Resiliparse top3: filter -> tokenize (reuses existing artifacts)
# ---------------------------------------------------------------------------
resili_filter_config = FilterByDomainConfig(
    input_path=output_path_of(resiliparse_step) / "*.jsonl.gz",
    output_path=this_output_path(),
    blocked_domains=[],
    allowed_domains=list(TOP_3),
)
resili_filter_step = ExecutorStep(
    name="filtered/math_resili_top3",
    description="Domain filter: top3 (brainly + jiskha + mathhelpforum) — resiliparse",
    fn=filter_by_domain,
    config=resili_filter_config,
)

resili_tokenized = default_tokenize(
    name="math_resili_top3_qwen3-0.6b-base_sft",
    dataset=resili_filter_step / "**/*.jsonl.gz",
    tokenizer="Qwen/Qwen3-0.6B-Base",
    format=TextLmDatasetFormat(),
)

# ---------------------------------------------------------------------------
# Build train + eval for each (data_type, hp_config) combination
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

DATA_SOURCES = {
    "extract": extract_tokenized,
    "resili": resili_tokenized,
}

for data_name, tokenized in DATA_SOURCES.items():
    for hp_name, hp in HP_CONFIGS.items():
        data_config = lm_data_config(training_set=tokenized, validation_sets={})

        run_name = f"math-top3-{data_name}-{hp_name}-qwen3-0.6b-base"

        train_step = ExecutorStep(
            name=f"checkpoints/{run_name}",
            description=f"Math top3 low-LR sweep: {data_name} {hp_name}",
            fn=_run_single_epoch_sft,
            config=_SFTRunConfig(
                tokenized_path=tokenized,
                data_config=data_config,
                output_path=this_output_path(),
                tags=("math", "top3-low-lr-sweep", data_name, hp_name, "sft", "qwen3-0.6b-base"),
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

        eval_step = evaluate_lm_evaluation_harness(
            model_name=run_name,
            model_path=output_path_of(train_step, "hf"),
            evals=DOMAIN_MIX_EVALS,
            resource_config=EVAL_RESOURCE,
            apply_chat_template=False,
            discover_latest_checkpoint=True,
        )

        all_steps.append(eval_step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description=(
            f"Math top3 low-LR sweep: {len(DATA_SOURCES)} data types × {len(HP_CONFIGS)} HP configs "
            f"= {len(all_steps)} runs (Qwen3-0.6B-Base)"
        ),
    )
