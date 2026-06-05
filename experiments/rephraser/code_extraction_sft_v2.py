# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Code extraction SFT v2 — plain text extraction prompt.

Same pipeline as code_extraction_sft.py but with a plain text extraction prompt
that avoids /* */ comments and ``` code fences. v1 showed format contamination:
the model learned to append code fences and comment blocks, causing syntax errors
on HumanEval even when the code itself was correct. This prompt outputs clean
plain text to match the evaluation format.

All CDX/download/filter steps are reused from v1 (same domain="code", same URLs).
Only the extraction, post-processing, tokenization, training, and eval are new.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/code_extraction_sft_v2.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/code_extraction_sft_v2.py --dry_run true
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

# ---------------------------------------------------------------------------
# 8B extractor model (same rephraser used for v1)
# ---------------------------------------------------------------------------
# TODO: migrate to InputName.hardcoded() — see experiments/AGENTS.md. Deferred to avoid hash invalidation.
REPHRASER_MODEL = "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"

# ---------------------------------------------------------------------------
# Qwen3 0.6B SFT model (same config as v1)
# ---------------------------------------------------------------------------
qwen3_0_6b_hd128_with_rope = dataclasses.replace(
    qwen3_0_6b_hd128,
    rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),
    max_seq_len=4096,
)

# ---------------------------------------------------------------------------
# Extraction prompt — plain text, no formatting artifacts
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """Extract the content from this HTML page as clean plain text.

Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
- The page is not about programming, software development, or computer science
- Index page, search results, or listing page without substantive content
- Login, signup, paywall, error page, or empty page
- Not primarily in English

Otherwise, extract the content following these rules:
- Output the text and code from the page in reading order.
- Preserve all code exactly as written.
- Keep all explanatory text that accompanies the code.
- Remove navigation menus, sidebars, headers, footers, and boilerplate.
- Do not add any formatting: no ``` code fences, no /* */ comments, no markdown.
- Do not include raw HTML tags or entities. Decode &amp; &lt; &gt; &quot; to their characters.
- Do not shorten or omit content."""

# ---------------------------------------------------------------------------
# URL sources (identical to v1)
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
            name="plaintext",
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
    executor_main(steps=result.all_steps, description="Code extraction SFT v2 — plain text (Qwen3 0.6B)")
