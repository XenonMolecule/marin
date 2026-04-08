# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Math extraction SFT v2 on Qwen3-0.6B-Base — multi-source, new prompt.

Same pipeline as mathhelpforum_extraction_sft_v2.py but fine-tunes the base model
(Qwen3-0.6B-Base) instead of the instruction-tuned model (Qwen3-0.6B).

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/mathhelpforum_extraction_sft_v2_base.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/mathhelpforum_extraction_sft_v2_base.py --dry_run true
"""

from experiments.rephraser.mathhelpforum_extraction_sft_v2 import (
    CRAWL_INDICES,
    DOMAIN_SOURCES,
    HOST_SOURCES,
    MATH_EVALS,
    MATH_EXTRACTION_PROMPT_V2,
    REPHRASER_MODEL,
    REPHRASER_TOKENIZER,
    qwen3_0_6b_hd128_with_rope,
)
from experiments.rephraser.extraction_sft_recipe import (
    BaselineDataset,
    DomainSource,
    EvalSpec,
    ExtractionSpec,
    SFTModelSpec,
    build_extraction_sft_experiment,
)
from experiments.rephraser.gsm8k_sft_plaintext import plaintext_transform_step as gsm8k_plaintext_step
from marin.execution.executor import executor_main

result = build_extraction_sft_experiment(
    domain="math_multi_v2",
    source=DomainSource(
        url_patterns=DOMAIN_SOURCES + HOST_SOURCES,
        crawl_indices=CRAWL_INDICES,
    ),
    extractions=[
        ExtractionSpec(
            name="unified",
            prompt=MATH_EXTRACTION_PROMPT_V2,
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
    executor_main(steps=result.all_steps, description="Math multi-source extraction SFT v2 (Qwen3 0.6B-Base)")
