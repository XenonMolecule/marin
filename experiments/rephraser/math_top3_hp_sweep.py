# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HP sweep on top3 domain mix for both extraction and resiliparse data.

Tests 20 HP configs on the top3 domain filter (brainly + jiskha + mathhelpforum)
for both extraction (LLM-extracted Q/R/A markdown) and resiliparse (plain text)
data pipelines. The top3 filter was identified as the best domain mix in V1.

Grid design:
  Core grid (15 configs): LR × BS at high regularization (wd=0.1, wu=0.03)
    LR: 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5
    BS: 16, 32, 64
    (Dropped: lr=1e-5/bs=16, lr=2e-5/bs=16, lr=2e-5/bs=32 — too aggressive)

  WD/warmup variation (5 configs): Test regularization sensitivity at best LRs
    16. lr=5e-7, bs=64, wd=0.01, wu=0.03  (math-best LR, low WD)
    17. lr=5e-7, bs=64, wd=0.05, wu=0.03  (math-best LR, mid WD)
    18. lr=1e-6, bs=32, wd=0.01, wu=0.03  (= code-best-resili)
    19. lr=2e-6, bs=32, wd=0.05, wu=0.0   (= code-best-extract)
    20. lr=2e-5, bs=64, wd=0.01, wu=0.03  (= code-default)

Total runs: 40 train + 40 eval (20 HP × 2 data types).
Each 0.6B train run is ~1000 steps on v5p-8 (~15-20 min).

V1 top3 reference scores (extraction, math-best HP):
  algebra_mv=50.8, prealg_mv=54.1, gsm8k_strict=54.7, gsm8k_flex=60.2

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/math_top3_hp_sweep.py
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
# Model config (same as domain mix sweeps)
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
# HP configs to sweep (20 total)
# ---------------------------------------------------------------------------
# Core grid: LR × BS at high regularization (wd=0.1, wu=0.03)
_CORE_LRS = [5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5]
_CORE_BSS = [16, 32, 64]

HP_CONFIGS: dict[str, TrainHyperparams] = {}

for lr in _CORE_LRS:
    for bs in _CORE_BSS:
        # Skip overly aggressive combos (high LR + small batch)
        if lr >= 1e-5 and bs == 16:
            continue
        if lr >= 2e-5 and bs == 32:
            continue
        lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
        name = f"lr{lr_str}_bs{bs}"
        HP_CONFIGS[name] = TrainHyperparams(
            batch_size=bs,
            learning_rate=lr,
            weight_decay=0.1,
            warmup=0.03,
        )

# WD/warmup variation at promising LR/BS combos
HP_CONFIGS["lr5e-7_bs64_wd01"] = TrainHyperparams(
    batch_size=64,
    learning_rate=5e-7,
    weight_decay=0.01,
    warmup=0.03,
)
HP_CONFIGS["lr5e-7_bs64_wd05"] = TrainHyperparams(
    batch_size=64,
    learning_rate=5e-7,
    weight_decay=0.05,
    warmup=0.03,
)
HP_CONFIGS["code-best-resili"] = TrainHyperparams(
    batch_size=32,
    learning_rate=1e-6,
    weight_decay=0.01,
    warmup=0.03,
)
HP_CONFIGS["code-best-extract"] = TrainHyperparams(
    batch_size=32,
    learning_rate=2e-6,
    weight_decay=0.05,
    warmup=0.0,
)
HP_CONFIGS["code-default"] = TrainHyperparams(
    batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup=0.03,
)

# ---------------------------------------------------------------------------
# Evals — same as domain mix sweeps for comparability
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
# Source data
# ---------------------------------------------------------------------------
extraction_branch = math_result.extraction_branches[0]  # "unified"
postprocessed_step = extraction_branch.postprocess_step
resiliparse_step = math_result.resiliparse_step

# ---------------------------------------------------------------------------
# Extraction top3: filter -> tokenize (reuses V1 artifacts via same step names)
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
    resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
    pip_dependency_groups=["cpu"],
)

extract_tokenized = default_tokenize(
    name="math_mix_top3_qwen3-0.6b-base_sft",
    dataset=extract_filter_step / "**/*.jsonl.gz",
    tokenizer="Qwen/Qwen3-0.6B-Base",
    format=TextLmDatasetFormat(),
)

# ---------------------------------------------------------------------------
# Resiliparse top3: filter -> tokenize (new — no prior resili top3 data)
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
    resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
    pip_dependency_groups=["cpu"],
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
            description=f"Math top3 HP sweep: {data_name} {hp_name}",
            fn=_run_single_epoch_sft,
            config=_SFTRunConfig(
                tokenized_path=tokenized,
                data_config=data_config,
                output_path=this_output_path(),
                tags=("math", "top3-hp-sweep", data_name, hp_name, "sft", "qwen3-0.6b-base"),
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
            f"Math top3 HP sweep: {len(DATA_SOURCES)} data types × {len(HP_CONFIGS)} HP configs "
            f"= {len(all_steps)} runs (Qwen3-0.6B-Base)"
        ),
    )
