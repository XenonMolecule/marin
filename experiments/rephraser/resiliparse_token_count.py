# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count how many tokens resiliparse-extracted text produces from our 151 WARC files.

This runs resiliparse (plain text extraction, no quality filtering) on the same
HTML as the rephraser experiment, then tokenizes with Llama-3.1. Gives us the
"real text" baseline between raw HTML (~144B tokens) and rephraser output (~362M tokens).

Launch:
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/resiliparse_token_count.py
"""

import os

from levanter.data.text import TextLmDatasetFormat
from marin.datakit.download.commoncrawl.download_warc import WarcDownloadConfig, download_and_extract_warcs
from marin.execution.executor import (
    ExecutorStep,
    ensure_versioned,
    executor_main,
    this_output_path,
    versioned,
)
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.transform.extract_text_from_html import ExtractTextConfig, extract_text_from_html

from experiments.llama import llama3_tokenizer

WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")


def load_warc_paths(manifest_path: str) -> list[str]:
    with open(manifest_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


warc_paths = load_warc_paths(WARC_MANIFEST)

# Step 1: Download & Extract HTML from WARCs (reuses cached output)
download_warcs = ExecutorStep(
    name="raw/commoncrawl/rephraser_sweep_batch0",
    description="Download WARC files from Common Crawl and extract HTML.",
    fn=download_and_extract_warcs,
    config=WarcDownloadConfig(
        warc_paths=versioned(tuple(warc_paths)),
        output_path=this_output_path(),
    ),
)

# Step 2: Extract plain text with resiliparse (no quality filtering)
extract_text = ExecutorStep(
    name="processed/resiliparse_text_from_warcs",
    description="Extract plain text from HTML using resiliparse.",
    fn=extract_text_from_html,
    config=ExtractTextConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

# Step 3: Tokenize the extracted text with Llama-3.1
tokenize_resiliparse = ExecutorStep(
    name="tokenized/resiliparse_text_from_warcs",
    description="Tokenize resiliparse-extracted text with Llama-3.1.",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[extract_text / "*.jsonl.gz"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(llama3_tokenizer),
        format=TextLmDatasetFormat(),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[tokenize_resiliparse],
        description="Count resiliparse-extracted text tokens from rephraser WARC files.",
    )
