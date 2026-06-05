# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Law extraction SFT — multi-source, starter prompt.

Broad collection of legal sources discovered via CDX probing (see
law_sources.txt). Uses a starter extraction prompt for legal Q&A forums,
case law, and legal reference pages. User will iterate on prompt independently.

Model: Qwen3-0.6B-Base (same as math/code/medical experiments).

Sources span multiple crawl indices:
- CC-MAIN-2013-48: Stack Exchange law topics, early forum content
- CC-MAIN-2016-44: Peak forum era (avvo, freeadvice)
- CC-MAIN-2018-47: Last chance for some forums before blocking
- CC-MAIN-2024-46: Modern legal reference (justia, nolo, cornell LII)
- CC-MAIN-2025-47: Latest crawl

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/law_extraction_sft.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/law_extraction_sft.py --dry_run true
"""

import dataclasses

from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.rephraser.extraction_sft_recipe import (
    DomainSource,
    EvalSpec,
    ExtractionSpec,
    SFTModelSpec,
    UrlPattern,
    build_extraction_sft_experiment,
)
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
# Extraction prompt — starter version for legal content
# ---------------------------------------------------------------------------
LAW_EXTRACTION_PROMPT = """\
Extract the legal content from this HTML page as clean Markdown. Follow all rules below.

1. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
   - The page is not about law, legal procedures, court decisions, legislation, or legal education
   - Index page, search results, listing page, lawyer directory without substantive content, or user profile
   - Advertisement for legal services without educational content
   - Login, signup, paywall, error page, or empty page
   - Not primarily in English

2. Remove all boilerplate: navigation, footers, sidebars, ads, user signatures, lawyer badges, rating widgets, related links, and cookie/privacy banners. Do not include any raw HTML tags or undecoded HTML entities in your output. Do not output [[ ## text ## ]] or similar framework markers.

3. Preserve all legal terminology, case citations, statute references, and regulatory codes exactly as written. Maintain standard legal citation format (e.g., "Brown v. Board of Education, 347 U.S. 483 (1954)"). Do not alter or abbreviate case names, statute numbers, or regulatory references.

4. For Q&A pages (legal forums, Q&A sites like Avvo, legal advice communities), output in this format:
   - Start with the thread title as a top-level heading.
   - Use up to three sections: ## Question, ## Analysis, ## Answer.
   - **Question**: The original poster's legal question or situation, preserving all relevant facts (jurisdiction, circumstances, specific legal issues).
   - **Analysis**: Synthesize the useful replies into a coherent legal analysis written in an impersonal voice — do not address users by name, do not narrate the conversation, and do not attribute analysis to specific posters. Reproduce the legal reasoning, relevant statutes, case law references, and practical advice from the replies. When a reply explains a legal principle, procedure, or defense, include the full explanation. If the thread ends without resolving the question, state only what was established. Include any jurisdiction-specific caveats mentioned.
   - **Answer**: The final answer or legal recommendation, only if present in the thread. If no clear answer was reached, omit the ## Answer section entirely.
   - If the thread contains multiple distinct legal questions, use separate ## Question, ## Analysis, ## Answer blocks for each.

5. For case law and court opinions:
   Output in this format:
   - Case name and citation as a top-level heading.
   - ## Facts: Key facts of the case.
   - ## Issue: The legal question(s) before the court.
   - ## Holding: The court's decision.
   - ## Reasoning: The court's legal reasoning, including precedents cited and statutory interpretation.
   If the page is a case summary rather than a full opinion, adapt the format accordingly while preserving the structure.

6. For reference pages (legal encyclopedias, statute texts, legal guides, educational content):
   Output the text in reading order. Keep all explanatory content, legal definitions, statutory text, procedural requirements, and examples. For statute pages, preserve section numbers and cross-references. For legal guides, preserve the structured format (elements, defenses, procedures, etc.).

7. Do not add information that is not on the page. Every legal claim, citation, procedural requirement, or recommendation in your output must come from the source HTML."""

# ---------------------------------------------------------------------------
# Crawl indices — cover historical forums through modern reference
# ---------------------------------------------------------------------------
CRAWL_INDICES = [
    "CC-MAIN-2013-48",  # Early forum content, Stack Exchange law topics
    "CC-MAIN-2014-23",  # Strong forum continuation
    "CC-MAIN-2016-07",  # Avvo growth period, legal forums building
    "CC-MAIN-2016-44",  # Peak forum era — avvo, freeadvice
    "CC-MAIN-2017-47",  # Transition year — catches sites before they block
    "CC-MAIN-2018-47",  # Last chance for some forums before blocking
    "CC-MAIN-2020-50",  # Mid-era, modern legal reference emerging
    "CC-MAIN-2022-49",  # Modern legal sites establishing
    "CC-MAIN-2024-46",  # Near-modern: justia, nolo, cornell LII
    "CC-MAIN-2025-47",  # Latest crawl
]

# ---------------------------------------------------------------------------
# URL patterns — legal sources from law_sources.txt
# ---------------------------------------------------------------------------
# Start with high-confidence sources. After CDX probing, expand this list.
DOMAIN_SOURCES = [
    # === Legal Q&A (highest extraction value) ===
    UrlPattern("avvo.com", "domain"),
    UrlPattern("freeadvice.com", "domain"),
    UrlPattern("legalbeagles.info", "domain"),
    UrlPattern("lawguru.com", "domain"),
    # === Legal encyclopedias / reference ===
    UrlPattern("justia.com", "domain"),
    UrlPattern("nolo.com", "domain"),
    UrlPattern("findlaw.com", "domain"),
    UrlPattern("law.cornell.edu", "domain"),
    UrlPattern("uslegal.com", "domain"),
    UrlPattern("legalmatch.com", "domain"),
    # === Case law ===
    UrlPattern("courtlistener.com", "domain"),
    UrlPattern("oyez.org", "domain"),
    # === Legal education ===
    UrlPattern("lawteacher.net", "domain"),
    UrlPattern("e-lawresources.co.uk", "domain"),
    UrlPattern("scotusblog.com", "domain"),
    UrlPattern("lawfaremedia.org", "domain"),
    # === Specialty legal ===
    UrlPattern("visajourney.com", "domain"),
    UrlPattern("divorcenet.com", "domain"),
    UrlPattern("criminaldefenselawyer.com", "domain"),
    UrlPattern("workplacefairness.org", "domain"),
    # === UK / international law ===
    UrlPattern("legislation.gov.uk", "domain"),
    UrlPattern("bailii.org", "domain"),
    UrlPattern("inbrief.co.uk", "domain"),
    UrlPattern("citizensadvice.org.uk", "domain"),
]

HOST_SOURCES = [
    UrlPattern("law.stackexchange.com", "host"),
    UrlPattern("law.justia.com", "host"),
    UrlPattern("supreme.justia.com", "host"),
    UrlPattern("wex.lii.cornell.edu", "host"),
]

# ---------------------------------------------------------------------------
# Evals — focused law benchmarks
# ---------------------------------------------------------------------------
LAW_EVALS = [
    # MMLU law subtasks
    EvalTaskConfig(name="mmlu_professional_law", num_fewshot=0, task_alias="mmlu_professional_law_0shot"),
    EvalTaskConfig(name="mmlu_jurisprudence", num_fewshot=0, task_alias="mmlu_jurisprudence_0shot"),
    EvalTaskConfig(name="mmlu_international_law", num_fewshot=0, task_alias="mmlu_international_law_0shot"),
    # AGIEval LSAT subtasks
    EvalTaskConfig(name="agieval_lsat_ar", num_fewshot=0, task_alias="agieval_lsat_ar_0shot"),
    EvalTaskConfig(name="agieval_lsat_lr", num_fewshot=0, task_alias="agieval_lsat_lr_0shot"),
    EvalTaskConfig(name="agieval_lsat_rc", num_fewshot=0, task_alias="agieval_lsat_rc_0shot"),
]

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
result = build_extraction_sft_experiment(
    domain="law",
    source=DomainSource(
        url_patterns=DOMAIN_SOURCES + HOST_SOURCES,
        crawl_indices=CRAWL_INDICES,
    ),
    extractions=[
        ExtractionSpec(
            name="starter",
            prompt=LAW_EXTRACTION_PROMPT,
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
    eval_spec=EvalSpec(tasks=LAW_EVALS),
)

if __name__ == "__main__":
    executor_main(steps=result.all_steps, description="Law extraction SFT (Qwen3 0.6B)")
