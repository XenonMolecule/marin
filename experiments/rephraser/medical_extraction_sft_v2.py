# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Medical extraction SFT v2 — revised prompt based on deep dive analysis.

Key changes from v1:
- Broadened medical definition to include nursing practice, clinical workflows
- Made Q/R/A format optional (plain Markdown for non-Q&A content)
- Fixed page 2+ handling
- Reduced over-filtering of allnurses clinical content
- "Do not summarize" instruction to preserve detail

CDX/WARC/resiliparse steps are cached from v1. Only the extraction step
(new prompt hash) and downstream train/eval will rerun.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/medical_extraction_sft_v2.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/medical_extraction_sft_v2.py --dry_run true
"""

import dataclasses

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
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
)

# ---------------------------------------------------------------------------
# Extraction prompt v2 — revised based on deep dive data analysis
# ---------------------------------------------------------------------------
MEDICAL_EXTRACTION_PROMPT_V2 = """Extract medical content from this HTML page as clean Markdown. The extracted text will train a language model on medical knowledge.

WHAT TO EXTRACT:
All content related to medicine, health, clinical practice, or patient care. This includes patient symptom discussions, clinical procedures, drug information and protocols, nursing practice discussions, scope of practice debates, diagnostic reasoning, care workflows, caregiver descriptions of patient conditions, and medical reference content. When in doubt about whether content is medical, extract it.

FILTER — output [NO_USEFUL_CONTENT] only for:
- Non-medical pages: login forms, user profiles, search results, sitemaps, advertising
- Directory and listing pages: business directories, pharmacy product/shop pages, residential home listings
- Tag listing pages, author archive pages, "who posted/liked" pages
- Pure career logistics with zero clinical content: salary numbers only, application deadlines only
- Content not in English (French, Spanish, German, etc.)
- Empty pages, error pages, pages with only navigation or boilerplate
- Forum board index pages that show only thread titles without actual post content

FORMAT:
- For forum threads with a clear question: use ## Question (original poster's words), ## Replies (content from all replies), ## Answer (only if clearly resolved, otherwise omit)
- For page 2+ of a thread, articles, reference pages, practice discussions, and any content that does not fit Q&A structure: extract in reading order as clean Markdown with appropriate headings
- Remove navigation, ads, signatures, footers, and boilerplate. No raw HTML tags.
- Preserve all medical terms, drug names, dosages, and lab values exactly as written.

CRITICAL RULES:
- Do not add any information not on the page.
- Do not summarize or compress. Preserve specific details from every reply.
- Do not add editorial commentary or clinical conclusions beyond what the page contains."""

# ---------------------------------------------------------------------------
# Crawl indices — cover historical forums through modern reference
# ---------------------------------------------------------------------------
CRAWL_INDICES = [
    "CC-MAIN-2013-48",  # Early forum content, Stack Exchange health topics
    "CC-MAIN-2014-23",  # Strong forum continuation
    "CC-MAIN-2016-07",  # Forums building up, medhelp/healthboards peak era
    "CC-MAIN-2016-44",  # Peak forum era — healthboards, studentdoctor, patient forums
    "CC-MAIN-2017-47",  # Transition year — catches sites before they block
    "CC-MAIN-2018-47",  # Last chance for some forums before blocking
    "CC-MAIN-2020-50",  # Mid-era, modern reference sites emerging
    "CC-MAIN-2022-49",  # Modern medical reference sites establishing
    "CC-MAIN-2024-46",  # Near-modern: mayo, cleveland clinic, drugs.com
    "CC-MAIN-2025-47",  # Latest crawl
]

# ---------------------------------------------------------------------------
# URL patterns — CDX-validated medical sources (probe_domain_sources.py, 2026-03-17)
# ---------------------------------------------------------------------------
# 13 sources validated across CC-MAIN-2013-48, CC-MAIN-2016-44, CC-MAIN-2025-47
# Grand total: ~2.85M estimated HTML 200 records
DOMAIN_SOURCES = [
    # === Medical Q&A forums — highest extraction value ===
    UrlPattern("healthboards.com", "domain"),  # 688k total, best 350k in 2016-44
    UrlPattern("allnurses.com", "domain"),  # 527k total, best 455k in 2016-44
    UrlPattern("medhelp.org", "domain"),  # 452k total, best 374k in 2016-44
    UrlPattern("healthunlocked.com", "domain"),  # 156k total, best 156k in 2016-44
    UrlPattern("patient.info", "domain"),  # 30k total, best 21k in 2025-47
    # === Medical reference — large, well-structured ===
    UrlPattern("webmd.com", "domain"),  # 360k total, best 231k in 2013-48
    UrlPattern("mayoclinic.org", "domain"),  # 273k total, best 219k in 2016-44
    UrlPattern("drugs.com", "domain"),  # 129k total, best 78k in 2016-44
    UrlPattern("clevelandclinic.org", "domain"),  # 35k total, best 20k in 2025-47
    UrlPattern("medlineplus.gov", "domain"),  # 31k total, best 27k in 2016-44
    UrlPattern("merckmanuals.com", "domain"),  # 9k total, best 9k in 2013-48
]

HOST_SOURCES = [
    UrlPattern("forums.studentdoctor.net", "host"),  # 37k total, best 25k in 2013-48
    UrlPattern("ncbi.nlm.nih.gov", "host"),  # 120k total, best 48k in 2013-48
]

# ---------------------------------------------------------------------------
# Evals — focused medical benchmarks
# ---------------------------------------------------------------------------
MEDICAL_EVALS = [
    # Generative MMLU medical subtasks (5-shot) — avoids vLLM TPU loglikelihood bug
    EvalTaskConfig(name="mmlu_anatomy_generative", num_fewshot=5, task_alias="mmlu_anatomy_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_clinical_knowledge_generative",
        num_fewshot=5,
        task_alias="mmlu_clinical_knowledge_gen_5shot",
    ),
    EvalTaskConfig(name="mmlu_college_medicine_generative", num_fewshot=5, task_alias="mmlu_college_medicine_gen_5shot"),
    EvalTaskConfig(name="mmlu_medical_genetics_generative", num_fewshot=5, task_alias="mmlu_medical_genetics_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_professional_medicine_generative",
        num_fewshot=5,
        task_alias="mmlu_professional_medicine_gen_5shot",
    ),
    EvalTaskConfig(name="mmlu_college_biology_generative", num_fewshot=5, task_alias="mmlu_college_biology_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_high_school_biology_generative",
        num_fewshot=5,
        task_alias="mmlu_high_school_biology_gen_5shot",
    ),
    # mediqa_qa2019_lite — generative medical QA (ROUGE), custom task in experiments/rephraser/custom_tasks/
    EvalTaskConfig(name="mediqa_qa2019_lite", num_fewshot=0, task_alias="mediqa_qa2019_lite_0shot"),
]

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
result = build_extraction_sft_experiment(
    domain="medical",
    source=DomainSource(
        url_patterns=DOMAIN_SOURCES + HOST_SOURCES,
        crawl_indices=CRAWL_INDICES,
    ),
    extractions=[
        ExtractionSpec(
            name="v2",
            prompt=MEDICAL_EXTRACTION_PROMPT_V2,
            model=REPHRASER_MODEL,
            model_tokenizer=REPHRASER_TOKENIZER,
            max_output_tokens=6144,
        ),
    ],
    sft_model=SFTModelSpec(
        model_config=qwen3_0_6b_hd128_with_rope,
        tokenizer="Qwen/Qwen3-0.6B",
        hf_model_name="Qwen/Qwen3-0.6B",
        pad_tokenizer_to_match_model=True,
        short_name="qwen3-0.6b",
    ),
    eval_spec=EvalSpec(tasks=MEDICAL_EVALS),
)

if __name__ == "__main__":
    # To run only resiliparse branch (no extraction), use:
    #   python medical_extraction_sft.py --resiliparse_only
    import sys

    if "--resiliparse_only" in sys.argv:
        sys.argv.remove("--resiliparse_only")
        _, resiliparse_eval = result.shared_branches["resiliparse"]
        executor_main(
            steps=[resiliparse_eval, result.baseline_eval, result.token_count_step],
            description="Medical v2 resiliparse-only SFT (Qwen3 0.6B)",
        )
    else:
        executor_main(steps=result.all_steps, description="Medical extraction SFT v2 (Qwen3 0.6B)")
