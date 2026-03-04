# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract unique Common Crawl snapshot IDs (isPartOf) from the DCLM baseline dataset.

Scans the `warcinfo` field from all jsonl.zst shards in the raw DCLM baseline on GCS
and extracts unique `isPartOf` values (e.g. CC-MAIN-2014-10).

Output: a sorted, deduplicated text file with one crawl ID per line. Designed for
downstream set operations — e.g. filtering DCLM to specific crawls or computing set
differences against a list of s3 Common Crawl paths (where the crawl ID can be
extracted via r'CC-MAIN-\\d{4}-\\d{2}').

Usage:
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait \\
        -- python scripts/dclm/extract_dclm_crawl_ids.py
"""

import io
import logging
import os
import re
from dataclasses import dataclass

import fsspec
import ray
import zstandard
from google.cloud import storage

from marin.execution.executor import ExecutorStep, executor_main, this_output_path

logger = logging.getLogger("ray")

# The raw DCLM baseline lives on us-central2, hardcoded to avoid prefix mismatch
# if submitted from a different cluster.
DCLM_RAW_PATH = (
    "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/" "mlfoundations/dclm-baseline-1.0/resolve/a3b142c"
)

ISPARTOF_RE = re.compile(r"isPartOf:\s*(CC-MAIN-\d{4}-\d{2})")

# Process files in batches per Ray task to reduce scheduling overhead.
BATCH_SIZE = 50


@dataclass(frozen=True)
class ExtractCrawlIdsConfig:
    input_path: str = DCLM_RAW_PATH
    """GCS path to the raw DCLM baseline data."""

    output_path: str = ""
    """Where to write crawl_ids.txt. Set by executor."""


def _list_shard_files(input_path: str) -> list[str]:
    """List all jsonl.zst blob names under the DCLM input path."""
    if input_path.startswith("gs://"):
        path = input_path[len("gs://") :]
    else:
        path = input_path
    bucket_name, prefix = path.split("/", 1)

    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [f"gs://{bucket_name}/{b.name}" for b in blobs if b.name.endswith(".jsonl.zst")]


def _extract_ids_from_file(filepath: str) -> set[str]:
    """Stream-decompress a single jsonl.zst file and return unique isPartOf crawl IDs."""
    crawl_ids: set[str] = set()
    dctx = zstandard.ZstdDecompressor()
    try:
        with fsspec.open(filepath, "rb") as f:
            buf = io.BytesIO(f.read())
        reader = dctx.stream_reader(buf)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        for line in text_stream:
            m = ISPARTOF_RE.search(line)
            if m:
                crawl_ids.add(m.group(1))
    except Exception:
        logging.exception("Error processing %s", filepath)
    return crawl_ids


@ray.remote(num_cpus=0.5, memory=4 * 1024 * 1024 * 1024)
def _process_batch(filepaths: list[str]) -> list[str]:
    """Ray remote task: process a batch of files and return unique crawl IDs."""
    batch_ids: set[str] = set()
    for fp in filepaths:
        batch_ids.update(_extract_ids_from_file(fp))
    return list(batch_ids)


def extract_dclm_crawl_ids(config: ExtractCrawlIdsConfig) -> None:
    logger.info("Listing jsonl.zst files under %s ...", config.input_path)
    all_files = _list_shard_files(config.input_path)
    logger.info("Found %d jsonl.zst files", len(all_files))

    if not all_files:
        logger.warning("No files found! Check that input_path is correct.")
        return

    # Split files into batches and fan out as Ray tasks across the cluster.
    batches = [all_files[i : i + BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]
    logger.info("Dispatching %d Ray tasks (%d files per batch) across the cluster ...", len(batches), BATCH_SIZE)

    futures = [_process_batch.remote(batch) for batch in batches]

    all_crawl_ids: set[str] = set()
    completed = 0
    for result in ray.get(futures):
        ids = set(result)
        new_ids = ids - all_crawl_ids
        all_crawl_ids.update(ids)
        completed += 1
        if new_ids:
            logger.info(
                "[batch %d/%d] +%d new: %s  (total: %d)",
                completed,
                len(batches),
                len(new_ids),
                sorted(new_ids),
                len(all_crawl_ids),
            )
        elif completed % 50 == 0:
            logger.info("[batch %d/%d] total: %d", completed, len(batches), len(all_crawl_ids))

    sorted_ids = sorted(all_crawl_ids)

    output_file = os.path.join(config.output_path, "crawl_ids.txt")
    with fsspec.open(output_file, "w") as f:
        for crawl_id in sorted_ids:
            f.write(crawl_id + "\n")

    logger.info("Done — %d unique crawl IDs written to %s", len(sorted_ids), output_file)


extract_crawl_ids = ExecutorStep(
    name="dclm/extract_crawl_ids",
    description="Extract unique CC crawl IDs (isPartOf) from the raw DCLM baseline warcinfo field.",
    fn=extract_dclm_crawl_ids,
    config=ExtractCrawlIdsConfig(
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[extract_crawl_ids],
        description="Extract unique Common Crawl snapshot IDs from the DCLM baseline dataset.",
    )
