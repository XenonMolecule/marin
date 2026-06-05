# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduction run for extract lr5e-8_bs64.

The low-LR sweep found that lr=5e-8 achieves 64.3% GSM8K flex (beating both
the untrained baseline at 62.8% and the prior sweep best at 60.3%) while
maintaining 29.4% avg_minerva. This is suspiciously good compared to neighbors
(lr=1e-7: 62.0%, lr=2e-8: 63.1%). Run a second training+eval to verify.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/math_top3_lr5e8_repro.py
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize import lm_data_config
from marin.transform.filter_by_domain import FilterByDomainConfig, filter_by_domain

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.rephraser.extraction_sft_recipe import (
    TrainHyperparams,
    _run_single_epoch_sft,
    _SFTRunConfig,
)
from experiments.rephraser.mathhelpforum_extraction_sft_v2_base import result as math_result

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)

TOP_3 = (
    "brainly.com",
    "jiskha.com",
    "mathhelpforum.com",
)

hp = TrainHyperparams(
    batch_size=64,
    learning_rate=5e-8,
    weight_decay=0.1,
    warmup=0.03,
)

# ---------------------------------------------------------------------------
# Evals — same as sweep for comparability
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
# Source data (reuses existing filtered/tokenized artifacts)
# ---------------------------------------------------------------------------
extraction_branch = math_result.extraction_branches[0]  # "unified"
postprocessed_step = extraction_branch.postprocess_step

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
# Train + eval (different step name forces fresh training run)
# ---------------------------------------------------------------------------
data_config = lm_data_config(training_set=extract_tokenized, validation_sets={})

run_name = "math-top3-extract-lr5e-8_bs64-repro-qwen3-0.6b-base"

train_step = ExecutorStep(
    name=f"checkpoints/{run_name}",
    description="Reproduction: extract lr5e-8_bs64 (verify GSM8K=64.3% is not a fluke)",
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=extract_tokenized,
        data_config=data_config,
        output_path=this_output_path(),
        tags=("math", "top3-lr5e8-repro", "extract", "lr5e-8_bs64", "sft", "qwen3-0.6b-base"),
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

if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description="Reproduction run: extract lr5e-8_bs64 (verify GSM8K result)",
    )
