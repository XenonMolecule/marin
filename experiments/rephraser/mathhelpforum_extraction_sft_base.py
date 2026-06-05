# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mathhelpforum extraction SFT on Qwen3-0.6B-Base.

Same pipeline as mathhelpforum_extraction_sft.py but fine-tunes the base model
(Qwen3-0.6B-Base) instead of the instruction-tuned model (Qwen3-0.6B).

Branches:
- Q/R/A extraction
- General/markdown extraction
- Resiliparse baseline
- GSM8K baseline
- No-training baseline eval

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/mathhelpforum_extraction_sft_base.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/mathhelpforum_extraction_sft_base.py --dry_run true
"""

import dataclasses

from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.rephraser.extraction_sft_recipe import (
    BaselineDataset,
    DomainSource,
    EvalSpec,
    ExtractionSpec,
    SFTModelSpec,
    UrlPattern,
    build_extraction_sft_experiment,
)
from experiments.rephraser.gsm8k_sft_plaintext import plaintext_transform_step as gsm8k_plaintext_step
from experiments.rephraser.mathhelpforum_extract import (
    EXTRACTION_SPEC as GENERAL_PROMPT,
)
from experiments.rephraser.mathhelpforum_extract import (
    REPHRASER_MODEL,
    REPHRASER_TOKENIZER,
    consolidate_step,
)
from experiments.rephraser.mathhelpforum_extract_qra import EXTRACTION_SPEC as QRA_PROMPT

# Qwen3 0.6B architecture with head_dim=128 and theta=1M (matching HF weights)
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
)

MATH_EVALS = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
    EvalTaskConfig(name="hendrycks_math_algebra", num_fewshot=4, task_alias="hendrycks_math_algebra_4shot"),
    EvalTaskConfig(
        name="hendrycks_math_counting_and_prob",
        num_fewshot=4,
        task_alias="hendrycks_math_counting_and_prob_4shot",
    ),
    EvalTaskConfig(name="hendrycks_math_geometry", num_fewshot=4, task_alias="hendrycks_math_geometry_4shot"),
    EvalTaskConfig(
        name="hendrycks_math_intermediate_algebra",
        num_fewshot=4,
        task_alias="hendrycks_math_intermediate_algebra_4shot",
    ),
    EvalTaskConfig(name="hendrycks_math_num_theory", num_fewshot=4, task_alias="hendrycks_math_num_theory_4shot"),
    EvalTaskConfig(name="hendrycks_math_prealgebra", num_fewshot=4, task_alias="hendrycks_math_prealgebra_4shot"),
    EvalTaskConfig(name="hendrycks_math_precalc", num_fewshot=4, task_alias="hendrycks_math_precalc_4shot"),
]

result = build_extraction_sft_experiment(
    domain="mathhelpforum",
    source=DomainSource(
        url_patterns=[UrlPattern("mathhelpforum.com")],
        html_data_override=consolidate_step,
    ),
    extractions=[
        ExtractionSpec(
            name="qra",
            prompt=QRA_PROMPT,
            model=REPHRASER_MODEL,
            model_tokenizer=REPHRASER_TOKENIZER,
        ),
        ExtractionSpec(
            name="general",
            prompt=GENERAL_PROMPT,
            model=REPHRASER_MODEL,
            model_tokenizer=REPHRASER_TOKENIZER,
        ),
    ],
    sft_model=SFTModelSpec(
        model_config=qwen3_0_6b_hd128_with_rope,
        tokenizer="Qwen/Qwen3-0.6B-Base",
        hf_model_name="Qwen/Qwen3-0.6B-Base",
        pad_tokenizer_to_match_model=True,
        short_name="qwen3-0.6b-base",
    ),
    eval_spec=EvalSpec(tasks=MATH_EVALS),
    baseline_datasets=[
        BaselineDataset(name="gsm8k", data_step=gsm8k_plaintext_step),
    ],
)

if __name__ == "__main__":
    executor_main(steps=result.all_steps, description="Mathhelpforum extraction SFT (Qwen3 0.6B-Base)")
