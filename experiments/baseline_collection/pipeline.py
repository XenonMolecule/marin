# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline dataset collection pipeline.

Extracts matching records from Nemotron-CC, DCLM, and FineWeb-Edu for a fixed
set of 3000 Common Crawl WARC files. Also extracts raw text via resiliparse as
a fourth "everything" baseline.

All four subsets are tokenized with the standard Marin tokenizer for training
and token-count comparison.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait \
        -e WANDB_API_KEY ... -e HF_TOKEN ... \
        -- python experiments/baseline_collection/pipeline.py
"""

from pathlib import Path

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from fray.v2.types import ResourceConfig
from marin.transform.extract_text_from_html import (
    ExtractTextConfig,
    _extract_text,
    _is_non_empty,
)
from zephyr import Dataset, ZephyrContext

from experiments.baseline_collection.download_warcs import (
    IncrementalWarcDownloadConfig,
    download_warcs_incremental,
)
from experiments.baseline_collection.extract_warc_metadata import (
    ExtractWarcMetadataConfig,
    extract_warc_metadata,
)
from experiments.baseline_collection.filter_dclm import FilterDclmConfig, filter_dclm
from experiments.baseline_collection.filter_fineweb_edu import FilterFinewebEduConfig, filter_fineweb_edu
from experiments.baseline_collection.filter_nemotron import FilterNemotronConfig, filter_nemotron
from experiments.defaults import default_tokenize

# --- Paths ---

WARC_MANIFEST = str(Path(__file__).resolve().parent.parent / "distill" / "baseline_warcs_3000.txt")

NEMOTRON_BASE = "gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"
DCLM_BASE = (
    "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/" "mlfoundations/dclm-baseline-1.0/resolve/a3b142c"
)
FINEWEB_EDU_BASE = "gs://marin-us-central2/raw/fineweb-edu"

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# --- Step 1: Download WARCs (incremental, per-file resumable) ---

download_warcs = ExecutorStep(
    name="raw/commoncrawl/baseline_3000",
    description="Download 3000 WARC files from Common Crawl and extract HTML.",
    fn=download_warcs_incremental,
    config=IncrementalWarcDownloadConfig(
        warc_manifest_path=WARC_MANIFEST,
        output_path=this_output_path(),
    ),
)

# --- Step 2a: Extract WARC metadata (record IDs, URLs, file paths) ---

extract_metadata = ExecutorStep(
    name="metadata/baseline_warc_metadata",
    description="Extract per-record metadata (record_id, url, warc_file, snapshot) from downloaded WARCs.",
    fn=extract_warc_metadata,
    config=ExtractWarcMetadataConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

# --- Step 2b: Extract raw text via resiliparse (the "everything" baseline) ---


def extract_text_fast(config: ExtractTextConfig) -> None:
    """Wrapper around resiliparse extraction with 500 workers (upstream defaults to 128)."""
    pipeline = (
        Dataset.from_files(config.input_path)
        .load_file()
        .map(_extract_text)
        .filter(_is_non_empty)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )
    ctx = ZephyrContext(name="extract-text-resiliparse", max_workers=500)
    ctx.put("config", config)
    ctx.execute(pipeline)


extract_text = ExecutorStep(
    name="extracted/baseline_resiliparse",
    description="Extract all text from downloaded WARCs via resiliparse (main_content=True).",
    fn=remote(extract_text_fast, resources=ResourceConfig(cpu=4, ram="32g")),
    config=ExtractTextConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

# --- Step 3a: Filter Nemotron-CC v1 (join on URL per snapshot) ---

filter_nemotron_step = ExecutorStep(
    name="filtered/baseline_nemotron",
    description="Filter Nemotron-CC v1 records matching our 3000 WARCs (join on URL).",
    fn=remote(filter_nemotron, resources=ResourceConfig(cpu=4, ram="16g")),
    config=FilterNemotronConfig(
        metadata_path=extract_metadata / "*.jsonl.gz",
        nemotron_base_path=NEMOTRON_BASE,
        output_path=this_output_path(),
    ),
)

# --- Step 3b: Filter DCLM-baseline (join on WARC-Record-ID, full scan) ---

filter_dclm_step = ExecutorStep(
    name="filtered/baseline_dclm",
    description="Filter DCLM-baseline records matching our 3000 WARCs (join on WARC-Record-ID).",
    fn=remote(filter_dclm, resources=ResourceConfig(cpu=4, ram="24g")),
    config=FilterDclmConfig(
        metadata_path=extract_metadata / "*.jsonl.gz",
        dclm_base_path=DCLM_BASE,
        output_path=this_output_path(),
    ),
)

# --- Step 3c: Filter FineWeb-Edu (join on file_path per snapshot) ---

filter_fineweb_step = ExecutorStep(
    name="filtered/baseline_fineweb_edu",
    description="Filter FineWeb-Edu records matching our 3000 WARCs (join on file_path).",
    fn=remote(filter_fineweb_edu, resources=ResourceConfig(cpu=4, ram="32g")),
    config=FilterFinewebEduConfig(
        metadata_path=extract_metadata / "*.jsonl.gz",
        fineweb_base_path=FINEWEB_EDU_BASE,
        output_path=this_output_path(),
    ),
)

# --- Step 4: Tokenize all four subsets ---

tokenize_nemotron = default_tokenize(
    name="baseline_nemotron",
    dataset=filter_nemotron_step / "*.jsonl.gz",
    tokenizer=TOKENIZER,
)

# DCLM filter produces 27K+ shards, 90% empty (20 bytes each).
# The Levanter tokenizer crashes on empty files (IndexError).
# Consolidate into fewer non-empty shards first.


def consolidate_dclm(config: ExtractTextConfig) -> None:
    """Re-shard DCLM output into fewer non-empty files.

    The DCLM filter produces 27K+ shards (one per input), 90% empty.
    Reshard into 100 output files so the tokenizer doesn't choke on empties.
    """
    pipeline = (
        Dataset.from_files(config.input_path)
        .load_file()
        .reshard(100)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-00100.jsonl.gz")
    )
    ctx = ZephyrContext(name="consolidate-dclm", max_workers=100)
    ctx.execute(pipeline)


consolidate_dclm_step = ExecutorStep(
    name="filtered/baseline_dclm_resharded",
    description="Consolidate sparse DCLM filter output (27K shards, 90% empty) into fewer non-empty shards.",
    fn=consolidate_dclm,
    config=ExtractTextConfig(
        input_path=filter_dclm_step / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

tokenize_dclm = default_tokenize(
    name="baseline_dclm",
    dataset=consolidate_dclm_step / "*.jsonl.gz",
    tokenizer=TOKENIZER,
)

tokenize_fineweb = default_tokenize(
    name="baseline_fineweb_edu",
    dataset=filter_fineweb_step / "*.jsonl.gz",
    tokenizer=TOKENIZER,
)

tokenize_resiliparse = default_tokenize(
    name="baseline_resiliparse",
    dataset=extract_text / "*.jsonl.gz",
    tokenizer=TOKENIZER,
)

# --- Step 2c: Rename html→text for raw HTML tokenization ---


def rename_html_to_text(config: ExtractTextConfig) -> None:
    """Rename 'html' field to 'text' so the tokenizer can read it."""

    def _rename(record: dict) -> dict:
        return {"text": record.get("html", ""), "url": record.get("url", "")}

    pipeline = (
        Dataset.from_files(config.input_path)
        .load_file()
        .map(_rename)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )
    ctx = ZephyrContext(name="rename-html-to-text", max_workers=500)
    ctx.execute(pipeline)


rename_html = ExecutorStep(
    name="extracted/baseline_raw_html",
    description="Rename html→text field from downloaded WARCs for raw HTML tokenization.",
    fn=remote(rename_html_to_text, resources=ResourceConfig(cpu=4, ram="16g")),
    config=ExtractTextConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

tokenize_raw_html = default_tokenize(
    name="baseline_raw_html",
    dataset=rename_html / "*.jsonl.gz",
    tokenizer=TOKENIZER,
)

# --- Entry point ---

if __name__ == "__main__":
    executor_main(
        steps=[tokenize_nemotron, tokenize_dclm, tokenize_fineweb, tokenize_resiliparse, tokenize_raw_html],
        description="Baseline dataset collection: Nemotron/DCLM/FineWeb-Edu/resiliparse/raw-HTML for 3000 WARCs.",
    )
