# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract math content from mathhelpforum.com HTML pages using the rephraser model.

Pipeline:
  1. Consolidate 530k tiny JSONL files into ~530 larger shards, filtering empty records
  2. Filter HTML by token length (skip documents exceeding model context)
  3. Run rephraser inference to extract clean Markdown from HTML
  4. Post-process the extraction output (strip [NO_USEFUL_CONTENT], etc.)

Input: mathhelpforum HTML pages downloaded from Common Crawl (530k pages)
       at gs://marin-us-central1/mathhelpforum/cc_download-5b0941/

Output: JSONL files with extracted Markdown math content

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -- python experiments/rephraser/mathhelpforum_extract.py

Dry run:
    python experiments/rephraser/mathhelpforum_extract.py --dry_run
"""

import hashlib
import json
import logging
from dataclasses import dataclass

import fsspec
from zephyr import Dataset, ZephyrContext, load_jsonl

from marin.execution.remote import remote
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    this_output_path,
)
from marin.generation.inference_v2 import InferenceV2Config, run_inference_v2
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

from experiments.rephraser.mathhelpforum_warc_scan import scan_step

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# TODO: migrate to InputName.hardcoded("checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318")
# to avoid hardcoded gs:// paths (see experiments/AGENTS.md). Deferred because changing
# this alters step hashes and forces full pipeline re-runs.
REPHRASER_MODEL = "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"

# ---------------------------------------------------------------------------
# Prompt configuration
# ---------------------------------------------------------------------------
EXTRACTION_SPEC = """\
Extract the main content from the provided HTML into clean Markdown.

First, check if the page should be rejected. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
- Not primarily in English
- Login, signup, account, checkout, paywall, or subscribe page
- Error page, captcha, cookie wall, bot check, or "session expired"
- Empty or near-empty page, directory index, or navigation-only page
- User profile, member page, or "who posted" page
- Image gallery or photo album listing without articles
- Search results page with no actual results
- Page where the main content is behind a login wall or paywall
- Product listing, gift card, or e-commerce page with prices/availability
- Social media post that is just an image or a single short caption
- Blog tag page, category page, or archive page that only lists post titles and teasers
- After removing boilerplate, the remaining useful text would be under ~100 words

If the page passes, extract with these rules:
- Output Markdown only. No commentary or analysis.
- Preserve original wording. Do not summarize or rewrite.
- Remove boilerplate: navbars, footers, sidebars, ads, share buttons, related links, breadcrumbs.
- Preserve all technical content exactly: code blocks verbatim with language tags, math/LaTeX using $ delimiters, chemical formulas, tables.
- Do not truncate or simplify content due to length.
- Include comments/replies only if they add real information (answers, corrections).
- Start with the page title as a top-level heading if available.
"""

SYSTEM_MESSAGE = (
    "Your input fields are:\n"
    "1. `html` (str): \n"
    "2. `extraction_spec` (str):\n"
    "Your output fields are:\n"
    "1. `text` (str):\n"
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in.\n\n"
    "[[ ## html ## ]]\n{html}\n\n"
    "[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n"
    "[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: \n"
    "        Extract the main content text from a given HTML document."
)

USER_TEMPLATE_FMT = (
    "[[ ## html ## ]]\n{{example}}\n\n"
    "[[ ## extraction_spec ## ]]\n{spec}\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)


def spec_hash(spec_text: str) -> str:
    """Stable 8-char ID from spec content."""
    return hashlib.sha256(spec_text.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Step 1: Consolidate tiny files into larger shards
# ---------------------------------------------------------------------------
# The WARC scan produced 530k individual files (one per CDX entry), many empty.
# This step reshards them into ~530 files with ~1000 records each, dropping
# empty records (content_length == 0) so downstream steps don't waste time.

NUM_OUTPUT_SHARDS = 530


@dataclass
class ConsolidateConfig:
    input_path: str
    output_path: str
    num_output_shards: int = NUM_OUTPUT_SHARDS


def consolidate_mathhelpforum(config: ConsolidateConfig):
    """Consolidate 530k tiny JSONL files into fewer larger shards, filtering empty records."""

    def _filter_nonempty(records: list[dict]) -> list[dict]:
        return [r for r in records if r.get("content_length", 0) > 0]

    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .reshard(config.num_output_shards)
        .map_shard(_filter_nonempty)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{config.num_output_shards:05d}.jsonl.gz")
    )

    with ZephyrContext(name="consolidate-mathhelpforum") as ctx:
        ctx.execute(pipeline)

    # Write consolidation stats
    non_empty_shards = 0
    fs = fsspec.filesystem("gcs")
    for i in range(config.num_output_shards):
        path = f"{config.output_path}/data-{i:05d}-of-{config.num_output_shards:05d}.jsonl.gz"
        try:
            info = fs.info(path)
            if info["size"] > 30:  # non-trivial gzip
                non_empty_shards += 1
        except FileNotFoundError:
            pass

    stats = {
        "num_output_shards": config.num_output_shards,
        "non_empty_shards": non_empty_shards,
    }
    with fsspec.open(f"{config.output_path}/consolidation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Consolidation complete: {config.num_output_shards} shards, " f"{non_empty_shards} non-empty")


consolidate_step = ExecutorStep(
    name="consolidated/mathhelpforum_html",
    description="Consolidate 530k mathhelpforum files into ~530 shards, filter empties.",
    fn=consolidate_mathhelpforum,
    config=ConsolidateConfig(
        input_path=scan_step / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

# ---------------------------------------------------------------------------
# Steps 2-4: Filter, Inference, Post-process
# ---------------------------------------------------------------------------
sid = spec_hash(EXTRACTION_SPEC)
user_template = USER_TEMPLATE_FMT.format(spec=EXTRACTION_SPEC)

# Step 2: Filter HTML by token length
filter_html = ExecutorStep(
    name="filtered/mathhelpforum_html",
    description=f"Filter mathhelpforum HTML documents exceeding {32768 - 4096} tokens.",
    fn=filter_by_token_length,
    config=FilterByTokenLengthConfig(
        input_path=consolidate_step / "*.jsonl.gz",
        output_path=this_output_path(),
        tokenizer=REPHRASER_TOKENIZER,
        text_column="html",
        max_tokens=32768 - 4096,
    ),
)

# Step 3: Inference with rephraser model
inference_step = ExecutorStep(
    name=f"documents/mathhelpforum_extract_{sid}",
    description=f"Run rephraser inference on mathhelpforum HTML (spec {sid}).",
    fn=remote(run_inference_v2, pip_dependency_groups=["vllm"]),
    config=InferenceV2Config(
        input_path=filter_html / "*.jsonl.gz",
        output_path=this_output_path(),
        model_name=REPHRASER_MODEL,
        input_format="jsonl.gz",
        output_format="jsonl.gz",
        engine_kwargs={
            "max_model_len": 32768,
            "enable_prefix_caching": True,
        },
        generation_kwargs={
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        system_message=SYSTEM_MESSAGE,
        template=user_template,
        prompt_column="html",
        apply_chat_template=True,
        max_doc_tokens=32768 - 4096,
        tensor_parallel_size=4,
        tpu_type="v5p-8",
        num_workers=16,
        records_per_shard=500,
    ),
)

# Step 4: Post-process extraction output
postprocess_step = ExecutorStep(
    name=f"processed/mathhelpforum_extract_{sid}",
    description=f"Post-process mathhelpforum extraction output (spec {sid}).",
    fn=postprocess_extraction,
    config=PostProcessExtractionConfig(
        input_path=inference_step / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[postprocess_step],
        description="Extract math content from mathhelpforum.com HTML pages.",
    )
