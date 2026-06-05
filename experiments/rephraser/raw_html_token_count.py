# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count how many tokens the raw HTML from our 150 WARC files produces when tokenized directly.

This skips all rephraser/extraction processing — just tokenizes the raw HTML with Llama-3.1.
Reuses the already-cached download_warcs step from rephraser_cooldown.py.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/raw_html_token_count.py
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

from experiments.llama import llama3_tokenizer

# Reuse the same WARC manifest as rephraser_cooldown.py
WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")


def load_warc_paths(manifest_path: str) -> list[str]:
    with open(manifest_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


warc_paths = load_warc_paths(WARC_MANIFEST)

# Step 1: Download & Extract HTML from WARCs (reuses cached output from rephraser_cooldown)
download_warcs = ExecutorStep(
    name="raw/commoncrawl/rephraser_sweep_batch0",
    description="Download WARC files from Common Crawl and extract HTML.",
    fn=download_and_extract_warcs,
    config=WarcDownloadConfig(
        warc_paths=versioned(tuple(warc_paths)),
        output_path=this_output_path(),
    ),
)

# Step 2: Tokenize raw HTML directly — text_key="html" reads the html field from JSONL
tokenize_raw_html = ExecutorStep(
    name="tokenized/raw_html_from_warcs",
    description="Tokenize raw HTML directly with Llama-3.1 to count tokens.",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[download_warcs / "*.jsonl.gz"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(llama3_tokenizer),
        format=TextLmDatasetFormat(text_key="html"),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[tokenize_raw_html],
        description="Count raw HTML tokens from rephraser WARC files.",
    )
