# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Domain-level data mix sweep for math extraction SFT on Qwen3-0.6B-Base.

Tests 6 different domain filters on the extraction data, all using the best HP
config from the HP sweep (lr=5e-7, bs=64). The "full" baseline is the existing
lr5e-7_bs64 result from the HP sweep — no need to rerun.

Mixes (all use blocklist filtering on the postprocessed extraction data):

  Mix 1 — "drop_worst3":     Block geogebra, wolfram, brainmass
                              (~10% tokens removed)
  Mix 2 — "drop_offtopic":   Block geogebra, wolfram, brainmass, physicsforums
                              (~30% tokens removed, keeps mathoverflow)
  Mix 3 — "drop_offtopic_mo": Block above + mathoverflow
                              (~47% tokens removed)
  Mix 4 — "core_math":       Allowlist — top forums + mathoverflow + brilliant
                              + math.stackexchange + small educational sites
                              (drops only physics/wolfram/brainmass/geogebra/etc)
  Mix 5 — "top5_plus_small": Allowlist — mathhelpforum, jiskha, brainly,
                              mathisfunforum, mathforum.org + small educational
                              sites. Drops physicsforums, mathoverflow, wolfram,
                              brainmass, geogebra, brilliant, sparknotes, etc.
  Mix 6 — "top3":            Allowlist — brainly, jiskha, mathhelpforum only
                              (~28% of original tokens)

Comparison baseline (from existing HP sweep, NOT rerun):
  "full" = lr5e-7_bs64 on unfiltered extraction data

Evals: all 7 minerva_math subtasks (4-shot) + gsm8k_platinum_cot (8-shot).

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/math_domain_mix_sweep.py
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
# Model config (same as HP sweep)
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
    hf_max_position_embeddings=32768,
)

# ---------------------------------------------------------------------------
# Best HP from sweep
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
# Evals — all minerva_math subtasks + GSM8K
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


# Domains clearly identified as harmful/irrelevant in V2 report
WORST_3 = ("geogebra.org", "forums.wolfram.com", "brainmass.com")

# Off-topic: physics, not math
OFFTOPIC = WORST_3 + ("physicsforums.com",)

# Off-topic + graduate math (too advanced for benchmarks)
OFFTOPIC_PLUS_MO = OFFTOPIC + ("mathoverflow.net",)

# Core math domains: broader math allowlist — includes mathoverflow, brilliant,
# math.stackexchange (higher-level content) alongside the top forums + small sites.
# This tests whether keeping advanced math content helps via diversity.
CORE_MATH_ALLOW = (
    # Top benchmark-relevant forums (rated 3+/5)
    "mathhelpforum.com",
    "jiskha.com",
    "brainly.com",
    "mathisfunforum.com",
    "mathforum.org",
    # Higher-level math (keeping these to test diversity hypothesis)
    "mathoverflow.net",
    "brilliant.org",
    "math.stackexchange.com",
    # Smaller educational sites (generally clean)
    "symbolab.com",
    "khanacademy.org",
    "splashlearn.com",
    "varsitytutors.com",
    "mathcentre.ac.uk",
    "openstax.org",
    "purplemath.com",
    "mathway.com",
    "onlinemathlearning.com",
    "savemyexams.com",
    "softschools.com",
    "illustrativemathematics.org",
    "math-only-math.com",
    "helpingwithmath.com",
    "math-drills.com",
    "mathplanet.com",
    "mathsisfun.com",
    "coolmath.com",
    "aaamath.com",
    "homeschoolmath.net",
    "mathgoodies.com",
    "mathwarehouse.com",
    "engageny.org",
    "math.libretexts.org",
    "nrich.maths.org",
    "tutorial.math.lamar.edu",
)

# Top 5 forums + all small educational sites (drops physicsforums, mathoverflow,
# wolfram, brainmass, geogebra, brilliant, sparknotes, cliffsnotes, algebrahelp)
TOP_5_PLUS_SMALL_ALLOW = (
    # Top 5 benchmark-relevant forums
    "mathhelpforum.com",
    "jiskha.com",
    "brainly.com",
    "mathisfunforum.com",
    "mathforum.org",
    # Smaller educational sites
    "symbolab.com",
    "khanacademy.org",
    "splashlearn.com",
    "varsitytutors.com",
    "mathcentre.ac.uk",
    "openstax.org",
    "purplemath.com",
    "mathway.com",
    "onlinemathlearning.com",
    "savemyexams.com",
    "softschools.com",
    "illustrativemathematics.org",
    "math-only-math.com",
    "helpingwithmath.com",
    "math-drills.com",
    "mathplanet.com",
    "mathsisfun.com",
    "coolmath.com",
    "aaamath.com",
    "homeschoolmath.net",
    "mathgoodies.com",
    "mathwarehouse.com",
    "engageny.org",
    "math.libretexts.org",
    "nrich.maths.org",
    "tutorial.math.lamar.edu",
)

# Top 3 most benchmark-relevant domains
TOP_3_ALLOW = (
    "brainly.com",
    "jiskha.com",
    "mathhelpforum.com",
)

MIXES = [
    DomainMix(
        name="drop_worst3",
        description="Remove geogebra, wolfram, brainmass (~10% tokens)",
        blocked_domains=WORST_3,
    ),
    DomainMix(
        name="drop_offtopic",
        description="Remove worst3 + physicsforums (~30% tokens, keep mathoverflow)",
        blocked_domains=OFFTOPIC,
    ),
    DomainMix(
        name="drop_offtopic_mo",
        description="Remove worst3 + physicsforums + mathoverflow (~47% tokens)",
        blocked_domains=OFFTOPIC_PLUS_MO,
    ),
    DomainMix(
        name="core_math",
        description="Allowlist: top forums + mathoverflow + brilliant + small sites",
        allowed_domains=CORE_MATH_ALLOW,
    ),
    DomainMix(
        name="top5_plus_small",
        description="Allowlist: top 5 forums + small educational sites",
        allowed_domains=TOP_5_PLUS_SMALL_ALLOW,
    ),
    DomainMix(
        name="top3",
        description="Allowlist: brainly + jiskha + mathhelpforum only (~28% tokens)",
        allowed_domains=TOP_3_ALLOW,
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
        name=f"filtered/math_mix_{mix.name}",
        description=f"Domain filter: {mix.description}",
        fn=filter_by_domain,
        config=filter_config,
    )

    # Step 2: Tokenize filtered data
    tokenized = default_tokenize(
        name=f"math_mix_{mix.name}_qwen3-0.6b-base_sft",
        dataset=filter_step / "**/*.jsonl.gz",
        tokenizer="Qwen/Qwen3-0.6B-Base",
        format=TextLmDatasetFormat(),
    )

    # Step 3: Train
    data_config = lm_data_config(training_set=tokenized, validation_sets={})

    train_step = ExecutorStep(
        name=f"checkpoints/math-mix-{mix.name}-qwen3-0.6b-base",
        description=f"Math domain mix SFT: {mix.name}",
        fn=_run_single_epoch_sft,
        config=_SFTRunConfig(
            tokenized_path=tokenized,
            data_config=data_config,
            output_path=this_output_path(),
            tags=("math", "domain-mix", mix.name, "extraction", "sft", "qwen3-0.6b-base"),
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
        model_name=f"math-mix-{mix.name}-qwen3-0.6b-base",
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
        description=f"Math domain mix sweep: {len(MIXES)} mixes × 1 HP config = {len(all_steps)} runs",
    )
