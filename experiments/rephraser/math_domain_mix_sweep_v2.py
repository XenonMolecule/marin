# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Domain mix sweep V2: targeted mixes to find optimal combination.

V1 found that top3 (brainly+jiskha+mathhelpforum) leads on 5/7 MATH subtasks
but core_math leads on int_algebra, precalculus, and GSM8K. V2 tests targeted
combinations to isolate which additional domains help for advanced math and
GSM8K without hurting elementary math performance.

Mixes:
  1. top3_plus_mse:       top3 + math.stackexchange.com
                          (Tests if MSE alone provides advanced math boost)
  2. top3_plus_advanced:  top3 + mathoverflow + brilliant + MSE
                          (Tests if advanced sources recover int_alg/precalc)
  3. top3_plus_wordprob:  top3 + khanacademy + varsitytutors + openstax + mathplanet
                          (Tests if word-problem sites recover GSM8K)
  4. top3_plus_best5:     top3 + MSE + khanacademy + varsitytutors + openstax + brilliant
                          (Cherry-picks best from each category — predicted optimal)

All use best HP from sweep: lr=5e-7, bs=64, wd=0.1, warmup=0.03.

V1 reference scores:
  Untrained: algebra_mv=41.4, prealg_mv=49.0, gsm8k_strict=58.1, gsm8k_flex=62.8
  top3:      algebra_mv=50.8, prealg_mv=54.1, gsm8k_strict=54.7, gsm8k_flex=60.2
  core_math: algebra_mv=47.0, prealg_mv=51.0, gsm8k_strict=56.7, gsm8k_flex=60.5

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/math_domain_mix_sweep_v2.py
"""

import dataclasses
from dataclasses import dataclass

from experiments.defaults import default_tokenize
from experiments.evals.evals import evaluate_lm_evaluation_harness
from experiments.qwen3 import qwen3_0_6b_hd128
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
# Model config (same as V1)
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)

# ---------------------------------------------------------------------------
# Best HP from sweep (same as V1)
# ---------------------------------------------------------------------------
LR = 5e-7
BS = 64
WEIGHT_DECAY = 0.1
WARMUP = 0.03
DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096

# ---------------------------------------------------------------------------
# Evals — all minerva_math subtasks + GSM8K (same as V1)
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
# Source data: postprocessed extraction output (has url field for filtering)
# ---------------------------------------------------------------------------
extraction_branch = math_result.extraction_branches[0]  # "unified"
postprocessed_step = extraction_branch.postprocess_step


# ---------------------------------------------------------------------------
# Domain mix definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainMix:
    name: str
    description: str
    # Exactly one of these should be non-empty
    blocked_domains: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()


# Base: top 3 benchmark-relevant forums
TOP_3 = (
    "brainly.com",
    "jiskha.com",
    "mathhelpforum.com",
)

# Advanced math sources (university/competition level)
ADVANCED_MATH = (
    "mathoverflow.net",
    "brilliant.org",
    "math.stackexchange.com",
)

# Word-problem / applied-math educational sites
WORDPROBLEM_SITES = (
    "khanacademy.org",
    "varsitytutors.com",
    "openstax.org",
    "mathplanet.com",
)

MIXES = [
    DomainMix(
        name="top3_plus_mse",
        description="top3 + math.stackexchange.com (isolate MSE contribution)",
        allowed_domains=TOP_3 + ("math.stackexchange.com",),
    ),
    DomainMix(
        name="top3_plus_advanced",
        description="top3 + mathoverflow + brilliant + MSE (recover int_alg/precalc)",
        allowed_domains=TOP_3 + ADVANCED_MATH,
    ),
    DomainMix(
        name="top3_plus_wordprob",
        description="top3 + khanacademy + varsitytutors + openstax + mathplanet (recover GSM8K)",
        allowed_domains=TOP_3 + WORDPROBLEM_SITES,
    ),
    DomainMix(
        name="top3_plus_best5",
        description="top3 + MSE + khanacademy + varsitytutors + openstax + brilliant (predicted optimal)",
        allowed_domains=TOP_3
        + (
            "math.stackexchange.com",
            "khanacademy.org",
            "varsitytutors.com",
            "openstax.org",
            "brilliant.org",
        ),
    ),
]

# ---------------------------------------------------------------------------
# Build pipeline: filter -> tokenize -> train -> eval for each mix
# ---------------------------------------------------------------------------
all_steps: list[ExecutorStep] = []

for mix in MIXES:
    # Step 1: Domain filter
    filter_config = FilterByDomainConfig(
        input_path=postprocessed_step / "*.jsonl.gz",
        output_path=this_output_path(),
        blocked_domains=list(mix.blocked_domains),
        allowed_domains=list(mix.allowed_domains),
    )
    filter_step = ExecutorStep(
        name=f"filtered/math_mix_v2_{mix.name}",
        description=f"Domain filter V2: {mix.description}",
        fn=filter_by_domain,
        config=filter_config,
    )

    # Step 2: Tokenize filtered data
    tokenized = default_tokenize(
        name=f"math_mix_v2_{mix.name}_qwen3-0.6b-base_sft",
        dataset=filter_step / "**/*.jsonl.gz",
        tokenizer="Qwen/Qwen3-0.6B-Base",
        format=TextLmDatasetFormat(),
    )

    # Step 3: Train
    data_config = lm_data_config(training_set=tokenized, validation_sets={})

    train_step = ExecutorStep(
        name=f"checkpoints/math-mix-v2-{mix.name}-qwen3-0.6b-base",
        description=f"Math domain mix V2 SFT: {mix.name}",
        fn=_run_single_epoch_sft,
        config=_SFTRunConfig(
            tokenized_path=tokenized,
            data_config=data_config,
            output_path=this_output_path(),
            tags=("math", "domain-mix-v2", mix.name, "extraction", "sft", "qwen3-0.6b-base"),
            model_config=qwen3_0_6b_hd128_with_rope,
            seq_len=SEQ_LEN,
            batch_size=BS,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            warmup=WARMUP,
            decay=DECAY,
            lr_schedule=LR_SCHEDULE,
            max_grad_norm=MAX_GRAD_NORM,
            hf_model_name="Qwen/Qwen3-0.6B-Base",
            checkpoint_path=None,
            pad_tokenizer_to_match_model=True,
        ),
    )

    # Step 4: Eval
    eval_step = evaluate_lm_evaluation_harness(
        model_name=f"math-mix-v2-{mix.name}-qwen3-0.6b-base",
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
        description=f"Math domain mix V2 sweep: {len(MIXES)} mixes × 1 HP config = {len(all_steps)} runs",
    )
