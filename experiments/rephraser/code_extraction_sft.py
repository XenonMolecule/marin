# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Code extraction SFT using the extraction_sft_recipe.

Runs the full extraction SFT pipeline for code-heavy web sources using fresh
Common Crawl data. Sources include Stack Exchange Q&A sites, programming language
documentation, code tutorials, and framework/API docs (see code_sources.txt).

Pipeline:
1. CDX query across all available crawl indices (grouped by match_type: host,
   prefix, domain — the recipe handles grouping and combining automatically)
2. WARC download of HTML pages
3. LLM extraction using the Qwen3-8B rephraser
4. Post-process, tokenize, single-epoch SFT on Qwen3 0.6B, evaluate

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/code_extraction_sft.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/code_extraction_sft.py --dry_run true
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
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import executor_main

# ---------------------------------------------------------------------------
# 8B extractor model (same rephraser used for mathhelpforum experiments)
# ---------------------------------------------------------------------------
# TODO: migrate to InputName.hardcoded() — see experiments/AGENTS.md. Deferred to avoid hash invalidation.
REPHRASER_MODEL = "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"

# ---------------------------------------------------------------------------
# Qwen3 0.6B SFT model (same config as mathhelpforum_extraction_sft.py)
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """Extract the content from this HTML page and reformat it as a source code file with comments.

Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
- Page contains no code blocks or programming examples (inline code references do not count)
- Question listing, search results, or index page without actual content
- Login, signup, paywall, error page, or empty page
- Not primarily in English

If the page has code, reformat it following these rules:

Format — clearly separate code from text:
- Write all explanatory text inside /* */ block comments. Group related paragraphs together.
- Use // for short inline notes only.
- Put all code examples inside ``` code fences.
- Start with a /* */ block summarizing the page topic.
- Use /* */ blocks as section separators between code examples.
- Do not use **bold**, ## headings, [links](url), or other markdown besides code fences.

Content — include everything from the page:
- Include every paragraph and code example. Do not shorten or omit content.
- Preserve all code exactly as written. Do not fix or improve it.
- Do not include any raw HTML tags or entities. Decode &amp; &lt; &gt; &quot; to their characters."""

# ---------------------------------------------------------------------------
# URL sources (from code_sources.txt, grouped by CDX match_type)
# ---------------------------------------------------------------------------
# Stack Exchange / Q&A sites
_STACKEXCHANGE_URLS = [
    UrlPattern("stackoverflow.com", "host"),
    UrlPattern("codereview.stackexchange.com", "host"),
    UrlPattern("codegolf.stackexchange.com", "host"),
    UrlPattern("cs.stackexchange.com", "host"),
    UrlPattern("cstheory.stackexchange.com", "host"),
    UrlPattern("math.stackexchange.com", "host"),
    UrlPattern("stats.stackexchange.com", "host"),
    UrlPattern("physics.stackexchange.com", "host"),
    UrlPattern("unix.stackexchange.com", "host"),
    UrlPattern("askubuntu.com", "host"),
    UrlPattern("softwareengineering.stackexchange.com", "host"),
    UrlPattern("dsp.stackexchange.com", "host"),
    UrlPattern("ai.stackexchange.com", "host"),
    UrlPattern("datascience.stackexchange.com", "host"),
    UrlPattern("electronics.stackexchange.com", "host"),
    UrlPattern("crypto.stackexchange.com", "host"),
    UrlPattern("security.stackexchange.com", "host"),
    UrlPattern("tex.stackexchange.com", "host"),
    UrlPattern("mathematica.stackexchange.com", "host"),
    UrlPattern("mathoverflow.net", "host"),
]

# Language documentation sites
_LANG_DOC_URLS = [
    UrlPattern("docs.python.org", "host"),
    UrlPattern("doc.rust-lang.org", "host"),
    UrlPattern("pkg.go.dev", "host"),
    UrlPattern("en.cppreference.com", "host"),
    UrlPattern("kotlinlang.org/docs", "prefix"),
    UrlPattern("docs.scala-lang.org", "host"),
    UrlPattern("docs.julialang.org", "host"),
    UrlPattern("hexdocs.pm", "host"),
    UrlPattern("www.php.net/manual", "prefix"),
    UrlPattern("ruby-doc.org", "host"),
    UrlPattern("docs.rs", "host"),
]

# Code tutorials / references
_TUTORIAL_URLS = [
    UrlPattern("rosettacode.org", "host"),
    UrlPattern("www.geeksforgeeks.org", "host"),
    UrlPattern("realpython.com", "host"),
    UrlPattern("www.w3schools.com", "host"),
    UrlPattern("www.tutorialspoint.com", "host"),
]

# Framework / API docs
_FRAMEWORK_URLS = [
    UrlPattern("developer.mozilla.org", "host"),
    UrlPattern("www.tensorflow.org", "host"),
    UrlPattern("api.flutter.dev", "host"),
    UrlPattern("doc.qt.io", "host"),
    UrlPattern("huggingface.co/docs", "prefix"),
    UrlPattern("llvm.org/docs", "prefix"),
    UrlPattern("gcc.gnu.org/onlinedocs", "prefix"),
    UrlPattern("www.boost.org/doc", "prefix"),
    UrlPattern("www.typescriptlang.org/docs/handbook", "prefix"),
    UrlPattern("readthedocs.io", "domain"),
    UrlPattern("devdocs.io", "host"),
]

ALL_CODE_URLS = _STACKEXCHANGE_URLS + _LANG_DOC_URLS + _TUTORIAL_URLS + _FRAMEWORK_URLS

# 10 most recent Common Crawl indices (late 2024 – early 2026). Dedup eliminates
# most cross-crawl duplicates for these high-frequency sites, so more crawls yield
# diminishing returns. Querying all 121 crawls via the CDX HTTP API is possible but
# slow (~2 QPS with rate limiting); the DuckDB columnar approach (cdx_query_columnar.py)
# could enable full-index queries in future once GCP→AWS network latency is addressed.
RECENT_CC_CRAWL_INDICES = [
    "CC-MAIN-2026-08",
    "CC-MAIN-2026-04",
    "CC-MAIN-2025-51",
    "CC-MAIN-2025-47",
    "CC-MAIN-2025-43",
    "CC-MAIN-2025-38",
    "CC-MAIN-2025-33",
    "CC-MAIN-2025-30",
    "CC-MAIN-2024-51",
    "CC-MAIN-2024-46",
]

# ---------------------------------------------------------------------------
# Evaluation benchmarks
# ---------------------------------------------------------------------------
CODE_EVALS = [
    EvalTaskConfig(name="humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
    # HumanEval hardcodes num_fewshot=0 in its lm-eval task YAML, so any
    # non-zero value here is silently ignored — the 4-shot config was removed
    # after confirming it produced identical prompts/results to 0-shot.
]

# ---------------------------------------------------------------------------
# Build experiment
# ---------------------------------------------------------------------------
result = build_extraction_sft_experiment(
    domain="code",
    source=DomainSource(
        url_patterns=ALL_CODE_URLS,
        crawl_indices=RECENT_CC_CRAWL_INDICES,
    ),
    extractions=[
        ExtractionSpec(
            name="general",
            prompt=EXTRACTION_PROMPT,
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
    eval_spec=EvalSpec(tasks=CODE_EVALS),
)

if __name__ == "__main__":
    executor_main(steps=result.all_steps, description="Code extraction SFT (Qwen3 0.6B)")
