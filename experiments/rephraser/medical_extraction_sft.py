# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Medical extraction SFT — multi-source, starter prompt.

Broad collection of medical sources discovered via CDX probing (see
medical_sources.txt). Uses a starter extraction prompt for medical Q&A forums
and reference pages. User will iterate on prompt independently.

Model: Qwen3-0.6B-Base (same as math/code experiments).

Sources span multiple crawl indices:
- CC-MAIN-2013-48: Stack Exchange health topics, early forum content
- CC-MAIN-2016-44: Peak forum era (healthboards, medhelp, studentdoctor)
- CC-MAIN-2018-47: Last chance for some forums before blocking
- CC-MAIN-2024-46: Modern reference sites (mayo, cleveland clinic, drugs.com)
- CC-MAIN-2025-47: Latest crawl for current sites

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/medical_extraction_sft.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/medical_extraction_sft.py --dry_run true
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
# Extraction prompt — starter version for medical content
# ---------------------------------------------------------------------------
MEDICAL_EXTRACTION_PROMPT = """Extract the medical content from this HTML page as clean Markdown. Follow all rules below.

1. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply: the body text is not in English (e.g. French, Spanish, German); it is not about medicine or health; it is a directory, shop, product, login, search, profile, or tag page; or it is a personal blog about student life or career advice without discussing a medical condition or treatment. Patient forums discussing symptoms or treatments ARE medical content.

2. Remove navigation, ads, signatures, footers, and boilerplate. No raw HTML tags or undecoded entities.

3. Preserve all medical terminology, drug names, dosages, and lab values exactly as written.

4. For forum/Q&A pages, use this format:
   - ## Question: The original poster's full message, in their own words.
   - ## Reasoning: Synthesize the useful replies into a coherent, detailed explanation. Reproduce the full treatments, experiences, techniques, and reasoning from the replies — do not summarize or shorten them. When a reply describes a specific method or outcome, include the full detail. When a reply shares a personal experience with a treatment, include it. Do not attribute information to specific posters. If the thread ends without resolving the question, state only what was established — do not add editorial commentary or clinical conclusions beyond what replies contain.
   - ## Answer: The final answer or recommendation, only if one was clearly reached. Otherwise omit.

5. For reference pages, output the full text in reading order preserving all structure.

6. Do not add any information not on the page. This is important for preserving the faithfulness to the content."""

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
    UrlPattern("healthboards.com", "domain"),       # 688k total, best 350k in 2016-44
    UrlPattern("allnurses.com", "domain"),           # 527k total, best 455k in 2016-44
    UrlPattern("medhelp.org", "domain"),              # 452k total, best 374k in 2016-44
    UrlPattern("healthunlocked.com", "domain"),      # 156k total, best 156k in 2016-44
    UrlPattern("patient.info", "domain"),             # 30k total, best 21k in 2025-47
    # === Medical reference — large, well-structured ===
    UrlPattern("webmd.com", "domain"),                # 360k total, best 231k in 2013-48
    UrlPattern("mayoclinic.org", "domain"),           # 273k total, best 219k in 2016-44
    UrlPattern("drugs.com", "domain"),                # 129k total, best 78k in 2016-44
    UrlPattern("clevelandclinic.org", "domain"),      # 35k total, best 20k in 2025-47
    UrlPattern("medlineplus.gov", "domain"),          # 31k total, best 27k in 2016-44
    UrlPattern("merckmanuals.com", "domain"),         # 9k total, best 9k in 2013-48
]

HOST_SOURCES = [
    UrlPattern("forums.studentdoctor.net", "host"),  # 37k total, best 25k in 2013-48
    UrlPattern("ncbi.nlm.nih.gov", "host"),          # 120k total, best 48k in 2013-48
]

# ---------------------------------------------------------------------------
# Evals — focused medical benchmarks
# ---------------------------------------------------------------------------
MEDICAL_EVALS = [
    # Generative MMLU medical subtasks (5-shot) — avoids vLLM TPU loglikelihood bug
    EvalTaskConfig(name="mmlu_anatomy_generative", num_fewshot=5, task_alias="mmlu_anatomy_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_clinical_knowledge_generative", num_fewshot=5, task_alias="mmlu_clinical_knowledge_gen_5shot",
    ),
    EvalTaskConfig(name="mmlu_college_medicine_generative", num_fewshot=5, task_alias="mmlu_college_medicine_gen_5shot"),
    EvalTaskConfig(name="mmlu_medical_genetics_generative", num_fewshot=5, task_alias="mmlu_medical_genetics_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_professional_medicine_generative", num_fewshot=5, task_alias="mmlu_professional_medicine_gen_5shot",
    ),
    EvalTaskConfig(name="mmlu_college_biology_generative", num_fewshot=5, task_alias="mmlu_college_biology_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_high_school_biology_generative", num_fewshot=5, task_alias="mmlu_high_school_biology_gen_5shot",
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
            name="starter",
            prompt=MEDICAL_EXTRACTION_PROMPT,
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
            description="Medical resiliparse-only SFT (Qwen3 0.6B)",
        )
    else:
        executor_main(steps=result.all_steps, description="Medical extraction SFT (Qwen3 0.6B)")
