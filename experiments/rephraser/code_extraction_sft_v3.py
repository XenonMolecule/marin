# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Code extraction SFT v3 — commented extraction prompt (instruct model).

Corresponds to prompt v29 in the other repo.

Same pipeline as v2 but with a new extraction prompt that adds inline reasoning
comments to multi-step code and structures Q&A forum pages as Q:/A: pairs.

All CDX/download/filter steps are reused from v1/v2 (same domain="code", same URLs).
Only the extraction, post-processing, tokenization, training, and eval are new.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/code_extraction_sft_v3.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/code_extraction_sft_v3.py --dry_run true
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
# 8B extractor model
# ---------------------------------------------------------------------------
# TODO: migrate to InputName.hardcoded() — see experiments/AGENTS.md. Deferred to avoid hash invalidation.
REPHRASER_MODEL = "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"

# ---------------------------------------------------------------------------
# Qwen3 0.6B SFT model (instruct)
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
)

# ---------------------------------------------------------------------------
# Extraction prompt v3 — commented code + Q/A routing
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """Extract the content from this HTML page as clean plain text. Follow all rules below.

1. Do not use ``` code fences, markdown headers, bold, or any formatting. Output plain text only. The characters ``` must not appear anywhere in your output.
2. Do not include raw HTML tags or entities. Decode &amp; to &, &lt; to <, &gt; to >, &quot; to "
3. Do not shorten, omit, or truncate content. Extract the full page.
4. Do not add text that is not on the page. Every sentence must come from the source.
5. Preserve all code exactly as written on the page. Do not modify, rewrite, or generate new code. You should fix broken indentation so code is syntactically valid, and you should add brief inline comments in the target language's comment style to uncommented multi-step logic.

6. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
   - The page is not about programming, software development, or computer science
   - Index page, search results, listing page, or wiki metadata page without substantive content
   - Login, signup, paywall, error page, or empty page

7. For Q&A pages (Stack Overflow, Stack Exchange, forums with questions and replies), output in this exact order:

   Q: <the question, including any code the asker provided>

   Reasoning: <synthesize the discussion from the thread into an explanation from the question to the answer. Explain why the solution works, step by step. Only include information from the thread.>

   A: <the top answer, extracted exactly as written from the page>

8. If no complete, substantive answer was provided, omit the A: section.
9. If the page explicitly discusses buggy or non-functioning code, add this line at the very top of your output: code_status: buggy

10. For all other pages (tutorials, documentation, blogs, reference):
    Output the text and code in reading order. Keep all explanatory text. Remove navigation, sidebars, headers, footers, and boilerplate.
"""

# ---------------------------------------------------------------------------
# URL sources (identical to v1/v2)
# ---------------------------------------------------------------------------
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

_TUTORIAL_URLS = [
    UrlPattern("rosettacode.org", "host"),
    UrlPattern("www.geeksforgeeks.org", "host"),
    UrlPattern("realpython.com", "host"),
    UrlPattern("www.w3schools.com", "host"),
    UrlPattern("www.tutorialspoint.com", "host"),
]

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
            name="commented",
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
    executor_main(steps=result.all_steps, description="Code extraction SFT v3 — commented (Qwen3 0.6B)")
