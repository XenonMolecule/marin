# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Math extraction SFT v2 — multi-source, new prompt.

Expands from mathhelpforum-only to a broad collection of math sources discovered
via CDX probing (see math_sources.txt). Uses a unified extraction prompt that
handles both Q&A forums and tutorial/textbook pages.

Sources span 4 crawl indices:
- CC-MAIN-2013-48: Stack Exchange, mathforum, jiskha, mathoverflow, physicsforums
- CC-MAIN-2016-44: mathhelpforum (506k), brainly, brainmass, algebrahelp
- CC-MAIN-2018-47: last chance for physicsforums, engageny
- CC-MAIN-2025-47: modern sources (geogebra, symbolab, khanacademy, etc.)

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/mathhelpforum_extraction_sft_v2.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/mathhelpforum_extraction_sft_v2.py --dry_run true
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
    REPHRASER_MODEL,
    REPHRASER_TOKENIZER,
)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
)

# ---------------------------------------------------------------------------
# Extraction prompt v2 — unified Q&A + tutorial format
# ---------------------------------------------------------------------------
MATH_EXTRACTION_PROMPT_V2 = """\
Extract the mathematical content from this HTML page as clean Markdown. Follow all rules below.

1. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
   - The page is not about mathematics, statistics, or quantitative reasoning (e.g. history, social studies, literature, politics)
   - Index page, search results, listing page, category page, user profile, or wiki metadata page without substantive content
   - Software announcement, product news, or tool tutorial without a math problem
   - Login, signup, paywall, error page, or empty page
   - Not primarily in English

2. Remove all boilerplate: navigation, footers, sidebars, ads, user signatures, join dates, post counts, reaction buttons, and related thread links. Do not include any raw HTML tags or undecoded HTML entities in your output. Do not output [[ ## text ## ]] or similar framework markers.

3. Preserve all mathematical notation exactly: use $ for inline and $$ for display LaTeX. Do not alter variable names, drop terms, or change signs. Ensure every $ has a matching closing $, and every $$ has a matching closing $$. Do not place LaTeX commands like \\approx, \\left, or \\right outside of $ delimiters.

4. Write out every intermediate arithmetic and algebraic step explicitly. Do not skip computation. For example, write "$120/24 = 5$, and $5^2 = 25$" instead of "simplify to get 25". Show each substitution, each simplification, and each evaluation so the reader can follow without doing mental math.

5. Do not add information that is not on the page. Every claim, equation, and computation in your output must come from the source HTML. Do not solve exercises that have no answer on the page.

6. For Q&A pages (math forums, Stack Exchange, Brainly, pages with questions and replies), output in this format:
   - Start with the thread title as a top-level heading.
   - Use up to three sections: ## Question, ## Reasoning, ## Answer.
   - **Question**: The original poster's problem, preserving mathematical content faithfully.
   - **Reasoning**: Synthesize the useful replies into a coherent, detailed explanation written in an impersonal mathematical voice — do not address users by name, do not narrate the conversation ("Later you asked…", "The assistant replied…"), and do not attribute steps to specific posters. Reproduce the full derivations and step-by-step working from the replies — do not summarize or shorten them. When a reply shows how to integrate, factor, simplify, or solve, include every step. If the thread ends without resolving the question, state only what was established — do not add a concluding claim that goes beyond what was shown.
   - **Answer**: The final answer or solution, only if present in the thread. If no clear answer was reached, omit the ## Answer section entirely — do not include placeholder text like "Answer not provided". Just stop after ## Reasoning.
   - If the thread contains multiple distinct questions, use a separate ## Question, ## Reasoning, ## Answer block for each.

7. For all other pages (tutorials, textbooks, worksheets, documentation, lecture notes, reference):
   Output the text and math in reading order. Keep all explanatory text and worked examples with their full step-by-step solutions. For exercise sets, include the problem statement and its answer when an answer is provided on the page. If no answer is given for an exercise, include only the problem statement. Never insert placeholder text such as "No answer provided", "Answer not given", or similar — just move on to the next problem."""

# ---------------------------------------------------------------------------
# Crawl indices — 10 most important crawls for math content
# ---------------------------------------------------------------------------
# CDX dedup_by_url=True (default) ensures no duplicate URLs across crawls.
# Selected based on CDX probing results (see math_sources.txt).
CRAWL_INDICES = [
    "CC-MAIN-2013-48",  # Peak Q&A: math.stackexchange (148k), mathforum (389k), jiskha (258k), mathoverflow (202k)
    "CC-MAIN-2014-23",  # Strong Q&A continuation
    "CC-MAIN-2016-07",  # mathhelpforum building up, khanacademy (83k)
    "CC-MAIN-2016-44",  # mathhelpforum peak (506k!), brainly (203k), brainmass (114k), algebrahelp (58k)
    "CC-MAIN-2017-47",  # Transition year — catches sites before they block
    "CC-MAIN-2018-47",  # Last chance: physicsforums (52k), geogebra (104k), engageny (4.8k)
    "CC-MAIN-2020-50",  # Mid-era modern sources emerging
    "CC-MAIN-2022-49",  # symbolab growing (6k→43k), modern sites establishing
    "CC-MAIN-2024-46",  # Near-modern: splashlearn, savemyexams appearing
    "CC-MAIN-2025-47",  # Modern: symbolab (43k), khanacademy (35k), splashlearn (24k), geogebra (72k)
]

# ---------------------------------------------------------------------------
# URL patterns — all promising math sources from math_sources.txt
# ---------------------------------------------------------------------------
# Domain match sources
DOMAIN_SOURCES = [
    # === Modern large (>5k in 2025) ===
    UrlPattern("geogebra.org", "domain"),  # 72k modern, 104k in 2018
    UrlPattern("symbolab.com", "domain"),  # 43k modern
    UrlPattern("khanacademy.org", "domain"),  # 35k modern, 83k in 2016
    UrlPattern("splashlearn.com", "domain"),  # 24k modern (K-5)
    UrlPattern("varsitytutors.com", "domain"),  # 21k modern (K-12/SAT)
    UrlPattern("mathcentre.ac.uk", "domain"),  # 8.1k modern (UK university)
    UrlPattern("openstax.org", "domain"),  # 7.6k modern (OER textbooks)
    UrlPattern("purplemath.com", "domain"),  # 7.4k modern (algebra-precalc)
    UrlPattern("mathway.com", "domain"),  # 7.3k modern (solver)
    UrlPattern("sparknotes.com", "domain"),  # 7.2k modern (study guides)
    UrlPattern("mathisfunforum.com", "domain"),  # 3.1k modern (K-12 Q&A)
    UrlPattern("onlinemathlearning.com", "domain"),  # 3.3k modern (K-12 tutorials)
    UrlPattern("savemyexams.com", "domain"),  # 2.7k modern (exam prep)
    UrlPattern("softschools.com", "domain"),  # 1.9k modern (K-8)
    UrlPattern("illustrativemathematics.org", "domain"),  # 1.5k modern (K-12 curriculum)
    UrlPattern("desmos.com", "domain"),  # 1.2k modern (graphing)
    UrlPattern("math-only-math.com", "domain"),  # 1.1k modern, 3k in 2016
    UrlPattern("brilliant.org", "domain"),  # 1k modern (HS-university)
    UrlPattern("helpingwithmath.com", "domain"),  # 919 modern (K-8)
    UrlPattern("math-drills.com", "domain"),  # 4.7k modern (worksheets)
    UrlPattern("mathplanet.com", "domain"),  # 149 modern (HS structured)
    UrlPattern("mathsisfun.com", "domain"),  # 4.2k in 2013, clean HTML
    UrlPattern("coolmath.com", "domain"),  # 348 modern (K-8)
    # === Historical goldmines (best in 2013/2016) ===
    UrlPattern("mathhelpforum.com", "domain"),  # 506k in 2016!
    UrlPattern("mathforum.org", "domain"),  # 389k in 2013
    UrlPattern("jiskha.com", "domain"),  # 258k in 2013
    UrlPattern("brainly.com", "domain"),  # 203k in 2016
    UrlPattern("mathoverflow.net", "domain"),  # 202k in 2013
    UrlPattern("brainmass.com", "domain"),  # 114k in 2016
    UrlPattern("algebrahelp.com", "domain"),  # 58k in 2016
    UrlPattern("physicsforums.com", "domain"),  # 53k in 2013-2018
    UrlPattern("cliffsnotes.com", "domain"),  # 43k in 2013
    UrlPattern("aaamath.com", "domain"),  # 8.4k in 2016 (K-8)
    UrlPattern("homeschoolmath.net", "domain"),  # 2.2k in 2016 (K-8)
    UrlPattern("mathgoodies.com", "domain"),  # 3.4k in 2013
    UrlPattern("mathwarehouse.com", "domain"),  # 3.3k in 2018
    UrlPattern("engageny.org", "domain"),  # 4.8k in 2018 (K-12 curriculum)
]

# Host match sources (require exact hostname match)
HOST_SOURCES = [
    UrlPattern("math.libretexts.org", "host"),  # 7.4k modern (OER textbooks)
    UrlPattern("forums.wolfram.com", "host"),  # 4.5k modern (Mathematica Q&A)
    UrlPattern("nrich.maths.org", "host"),  # 1.4k modern (UK enrichment)
    UrlPattern("tutorial.math.lamar.edu", "host"),  # 818 modern (Paul's Online Notes)
    UrlPattern("math.stackexchange.com", "host"),  # 148k in 2013
    UrlPattern("physics.stackexchange.com", "host"),  # 574 in 2018
]

# ---------------------------------------------------------------------------
# Evals
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
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
        tokenizer="Qwen/Qwen3-0.6B",
        hf_model_name="Qwen/Qwen3-0.6B",
        pad_tokenizer_to_match_model=True,
        short_name="qwen3-0.6b",
    ),
    eval_spec=EvalSpec(tasks=MATH_EVALS),
    baseline_datasets=[
        BaselineDataset(name="gsm8k", data_step=gsm8k_plaintext_step),
    ],
)

if __name__ == "__main__":
    executor_main(steps=result.all_steps, description="Math multi-source extraction SFT v2 (Qwen3 0.6B)")
