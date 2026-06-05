# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download mathhelpforum.com HTML pages from Common Crawl via byte-range requests.

Uses pre-computed CDX index results (stored on GCS) to download specific WARC
records containing mathhelpforum.com pages. Each record is fetched via a tiny
byte-range HTTP request (a few KB), not a full WARC download.

The CDX results were gathered by querying all CC crawl indices (2018-2022) for
mathhelpforum.com, deduplicating by URL (keeping the most recent capture).
Total: ~530k unique pages.

Supports checkpoint resumption: on startup, lists existing output files on GCS
and skips entries that already have output. Each worker writes directly to GCS
with the original shard index, so partial runs are safely resumable.

Output: JSONL files with {id, url, html, warc_file, timestamp, content_length}

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central2 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -- python experiments/rephraser/mathhelpforum_warc_scan.py

Dry run:
    python experiments/rephraser/mathhelpforum_warc_scan.py --dry_run
"""

import gzip
import io
import json
import logging
import os
import re
from dataclasses import dataclass

import fsspec
import requests
import warcio
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    this_output_path,
    versioned,
)
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)

CDX_MANIFEST_GCS_PATH = "gs://marin-us-central2/manifests/mathhelpforum_cdx_deduped.json"


# ---------------------------------------------------------------------------
# WARC record download with direct GCS write (runs in Zephyr workers)
# ---------------------------------------------------------------------------
def _write_jsonl_gz(path: str, record: dict | None):
    """Write a single record (or empty file) as gzipped JSONL to GCS."""
    with fsspec.open(path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb") as gz:
            if record is not None:
                gz.write((json.dumps(record) + "\n").encode("utf-8"))


def _download_and_write_record(cdx_entry: dict) -> list[dict]:
    """Download a single WARC record, write output directly to GCS with original shard index."""
    shard_idx = cdx_entry["_shard_idx"]
    total = cdx_entry["_total"]
    output_path = cdx_entry["_output_path"]
    output_file = f"{output_path}/data-{shard_idx:05d}-of-{total:05d}.jsonl.gz"

    filename = cdx_entry["filename"]
    offset = int(cdx_entry["offset"])
    length = int(cdx_entry["length"])

    url = f"https://data.commoncrawl.org/{filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    try:
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to download record from {filename} at offset {offset}: {e}")
        _write_jsonl_gz(output_file, None)
        return []

    stream = io.BytesIO(resp.content)
    for record in warcio.ArchiveIterator(stream):
        if record.rec_type != "response":
            continue

        http_headers = record.http_headers
        if http_headers:
            content_type = http_headers.get_header("Content-Type") or ""
            if "text/html" not in content_type.lower():
                _write_jsonl_gz(output_file, None)
                return []

        content = record.content_stream().read()
        html = content.decode("utf-8", errors="replace")
        target_url = record.rec_headers.get_header("WARC-Target-URI") or ""
        record_id = record.rec_headers.get_header("WARC-Record-ID") or ""

        result = {
            "id": record_id,
            "url": target_url,
            "html": html,
            "warc_file": f"s3://commoncrawl/{filename}",
            "timestamp": cdx_entry.get("timestamp", ""),
            "content_length": len(html),
        }

        _write_jsonl_gz(output_file, result)
        return []

    _write_jsonl_gz(output_file, None)
    return []


# ---------------------------------------------------------------------------
# Checkpoint: find already-completed shards
# ---------------------------------------------------------------------------
def _get_completed_shards(output_path: str) -> set[int]:
    """List existing output files on GCS and return set of completed shard indices."""
    fs = fsspec.filesystem("gcs")
    completed = set()
    try:
        paths = fs.ls(output_path, detail=False)
        for path in paths:
            basename = path.split("/")[-1]
            match = re.match(r"data-(\d+)-of-\d+\.jsonl\.gz$", basename)
            if match:
                completed.add(int(match.group(1)))
    except FileNotFoundError:
        pass
    return completed


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
@dataclass
class MathHelpForumDownloadConfig:
    cdx_manifest_gcs_path: str
    output_path: str


def download_mathhelpforum_pages(config: MathHelpForumDownloadConfig):
    """Load CDX results, download matching WARC records with checkpoint resumption."""

    logger.info(f"Loading CDX manifest from {config.cdx_manifest_gcs_path}...")
    with fsspec.open(config.cdx_manifest_gcs_path, "r") as f:
        cdx_entries = json.load(f)
    total = len(cdx_entries)
    logger.info(f"Loaded {total} CDX entries")

    # Checkpoint: find already-completed shards
    completed = _get_completed_shards(config.output_path)
    remaining = [(i, e) for i, e in enumerate(cdx_entries) if i not in completed]
    logger.info(f"Checkpoint: {len(completed)} shards done, {len(remaining)} remaining")

    if not remaining:
        logger.info("All entries already downloaded!")
        with fsspec.open(os.path.join(config.output_path, "stats.json"), "w") as f:
            json.dump({"total": total, "completed": total, "remaining": 0}, f, indent=2)
        return

    # Tag entries with metadata for the worker function
    for idx, entry in remaining:
        entry["_shard_idx"] = idx
        entry["_total"] = total
        entry["_output_path"] = config.output_path

    # Workers write directly to GCS with the original shard index as filename.
    # The pipeline's write_jsonl output is a dummy (workers return []).
    pipeline = (
        Dataset.from_list([e for _, e in remaining])
        .flat_map(_download_and_write_record)
        .write_jsonl(f"{config.output_path}/_pipeline_meta/shard-{{shard:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="mathhelpforum-download") as ctx:
        ctx.execute(pipeline)

    with fsspec.open(os.path.join(config.output_path, "stats.json"), "w") as f:
        json.dump(
            {
                "cdx_manifest": config.cdx_manifest_gcs_path,
                "total_cdx_entries": total,
                "completed_before_run": len(completed),
                "processed_this_run": len(remaining),
            },
            f,
            indent=2,
        )

    logger.info(f"Done. Processed {len(remaining)} entries (skipped {len(completed)} already done)")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
scan_step = ExecutorStep(
    name="mathhelpforum/cc_download",
    description="Download mathhelpforum.com pages from CC via byte-range WARC requests.",
    fn=download_mathhelpforum_pages,
    config=MathHelpForumDownloadConfig(
        cdx_manifest_gcs_path=versioned(CDX_MANIFEST_GCS_PATH),
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[scan_step],
        description="Download mathhelpforum.com pages from Common Crawl (530k unique URLs).",
    )
