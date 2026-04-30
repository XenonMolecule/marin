# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expanded baseline collection pipeline for the full DCLM 400m-1x pool (10,363 WARCs).

Scope: Nemotron-only. The 3000-WARC subset is a strict subset of the 10,363
(verified via ``comm``), and the existing Nemotron filter logic is unchanged —
this pipeline just widens the input manifest.

Stages
------
1. ``download_warcs_incremental``: pull 10,363 WARCs from Common Crawl,
   extract HTML response records. Per-WARC resumable via filename hashing;
   ``skip_existing=True`` means nothing re-downloads on retry.
2. ``extract_warc_metadata``: emit per-record ``{warc_record_id, url, warc_file,
   snapshot}`` JSONL from the downloaded HTML.
3. ``filter_nemotron_full``: match Nemotron-CC v1 records (organic + 5 synthetic
   variants) against the URL set, scoped per-snapshot.
4. ``default_tokenize``: tokenize with the project default.

DCLM / FineWeb-Edu / resiliparse / raw-HTML are deliberately NOT run here — we
only need Nemotron for the 400m-1x expansion.

Usage (Iris, CPU only)::

    uv run iris --cluster marin job run --cpu 2 --memory 16GB \\
        --region us-central2 --no-wait \\
        --job-name baseline-nemotron-10k \\
        -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python experiments/baseline_collection/pipeline_10k.py
"""

from pathlib import Path

from fray.v2.types import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote

from experiments.baseline_collection.download_warcs import (
    IncrementalWarcDownloadConfig,
    download_warcs_incremental,
)
from experiments.baseline_collection.extract_warc_metadata import (
    ExtractWarcMetadataConfig,
    extract_warc_metadata,
)
from experiments.baseline_collection.filter_nemotron import (
    FilterNemotronFullConfig,
    filter_nemotron_full,
)
from experiments.defaults import default_tokenize

# --- Paths ---

WARC_MANIFEST = str(Path(__file__).resolve().parent.parent / "distill" / "dclm_400m_1x.txt")

NEMOTRON_BASE = "gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# --- Step 1: Download WARCs (incremental, per-file resumable) ---

download_warcs = ExecutorStep(
    name="raw/commoncrawl/dclm_400m_1x_10k",
    description="Download the full DCLM 400m-1x pool (10,363 WARCs) and extract HTML.",
    fn=download_warcs_incremental,
    config=IncrementalWarcDownloadConfig(
        warc_manifest_path=WARC_MANIFEST,
        output_path=this_output_path(),
    ),
)

# --- Step 2: Extract WARC metadata (record IDs, URLs, file paths) ---

extract_metadata = ExecutorStep(
    name="metadata/dclm_400m_1x_10k_warc_metadata",
    description="Extract per-record metadata from the 10k downloaded WARCs.",
    fn=extract_warc_metadata,
    config=ExtractWarcMetadataConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

# --- Step 3: Filter Nemotron-CC full (organic + synthetic rephraser variants) ---

filter_nemotron_10k_full = ExecutorStep(
    name="filtered/dclm_400m_1x_10k_nemotron_full",
    description="Filter Nemotron-CC v1 (actual + synthetic) records matching the 10k WARCs (join on URL).",
    fn=remote(filter_nemotron_full, resources=ResourceConfig(cpu=4, ram="16g")),
    config=FilterNemotronFullConfig(
        metadata_path=extract_metadata / "*.jsonl.gz",
        nemotron_base_path=NEMOTRON_BASE,
        output_path=this_output_path(),
    ),
)

# --- Step 4: Tokenize ---

tokenize_nemotron_10k_full = default_tokenize(
    name="dclm_400m_1x_10k_nemotron_full",
    dataset=filter_nemotron_10k_full / "*.jsonl.gz",
    tokenizer=TOKENIZER,
)

# --- Entry point ---

if __name__ == "__main__":
    executor_main(
        steps=[tokenize_nemotron_10k_full],
        description=(
            "10k Nemotron baseline: extract Nemotron-CC v1 actual+synthetic records for the "
            "full DCLM 400m-1x pool (10,363 WARCs) and tokenize."
        ),
    )
