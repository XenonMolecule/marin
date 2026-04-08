# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Math extraction SFT on Qwen3-14B-Base with top3 domain filter.

Tests whether the top3 domain mix (brainly + jiskha + mathhelpforum) works
differently at 14B scale than the full 42-domain data. Uses the same 3 HP
configs as math_14b_sft.py across 2 data types (extraction + resiliparse)
plus a baseline eval = 7 configs.

The filter and tokenize steps reuse the same names/hashes as the 0.6B top3
experiment (math_top3_hp_sweep.py), so the filtered + tokenized data artifacts
on GCS are shared. The Qwen3-14B tokenizer is identical to Qwen3-0.6B.

HP configs (from coding 14B sweep):
  default:       lr=2e-5,  bs=64, wd=0.01, warmup=0.03
  best-resili:   lr=1e-6,  bs=32, wd=0.01, warmup=0.03
  best-extract:  lr=2e-6,  bs=32, wd=0.05, warmup=0.0

Evals: all 7 minerva_math subtasks (4-shot) + gsm8k_platinum_cot (8-shot).

Train: v5p-32 (14B needs more memory).  Eval: v5p-8.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/math_14b_top3_sft.py
"""

import dataclasses
from dataclasses import dataclass

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.qwen3 import qwen3_14b
from experiments.rephraser.extraction_sft_recipe import (
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
# Qwen3-14B-Base model config (with SFT-appropriate RoPE / seq_len)
# ---------------------------------------------------------------------------
qwen3_14b_with_rope = dataclasses.replace(
    qwen3_14b,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)

# ---------------------------------------------------------------------------
# Top 3 domains (V1 winner — same as math_top3_hp_sweep.py)
# ---------------------------------------------------------------------------
TOP_3 = (
    "brainly.com",
    "jiskha.com",
    "mathhelpforum.com",
)


# ---------------------------------------------------------------------------
# HP configs from coding 14B sweep (same as math_14b_sft.py)
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
# Eval tasks — all minerva_math subtasks + GSM8K (same as math_14b_sft.py)
# ---------------------------------------------------------------------------
MATH_14B_EVALS = [
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
EVAL_ENGINE_KWARGS = {"max_model_len": 8192, "max_gen_toks": 1024}
EVAL_RESOURCE = ResourceConfig.with_tpu("v5p-8")

# ---------------------------------------------------------------------------
# Source data — same upstream as 0.6B top3 (math_top3_hp_sweep.py)
# ---------------------------------------------------------------------------
extraction_branch = math_result.extraction_branches[0]  # "unified"
postprocessed_step = extraction_branch.postprocess_step
resiliparse_step = math_result.resiliparse_step

# ---------------------------------------------------------------------------
# Extraction top3: filter -> tokenize (reuses 0.6B artifacts via same step names/hashes)
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
# Resiliparse top3: filter -> tokenize (reuses 0.6B artifacts via same step names/hashes)
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
# Build all steps: 1 baseline + 6 training configs
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

DATA_SOURCES = {
    "extract": extract_tokenized,
    "resili": resili_tokenized,
}

# Baseline eval (no training) — same as math_14b_sft.py, will be skipped if already exists
baseline_eval = evaluate_lm_evaluation_harness(
    model_name="math-14b-baseline-qwen3-14b-base",
    model_path="Qwen/Qwen3-14B-Base",
    evals=MATH_14B_EVALS,
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
            name=f"checkpoints/math-14b-top3-{config_name}-qwen3-14b-base",
            description=f"Math 14B top3 SFT: {config_name} (Qwen3-14B-Base)",
            fn=_run_single_epoch_sft,
            config=_SFTRunConfig(
                tokenized_path=tokenized_step,
                data_config=data_config,
                output_path=this_output_path(),
                tags=("math", "14b", "top3", config_name, data_name, "sft", "qwen3-14b-base"),
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
            model_name=f"math-14b-top3-{config_name}-qwen3-14b-base",
            model_path=output_path_of(train_step, "hf"),
            evals=MATH_14B_EVALS,
            engine_kwargs=EVAL_ENGINE_KWARGS,
            resource_config=EVAL_RESOURCE,
            apply_chat_template=False,
            discover_latest_checkpoint=True,
        )

        all_steps.append(eval_step)

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description=(
            f"Math 14B top3 SFT: {len(HP_CONFIGS)} HP configs x {len(DATA_SOURCES)} data types + baseline "
            f"= {len(all_steps)} configs (Qwen3-14B-Base, top3 domain filter)"
        ),
    )
