# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download WARC records from Common Crawl via HTTP byte-range requests.

Given a CDX manifest JSON (from cdx_query.py), downloads each record's WARC
content via HTTP byte-range requests and writes consolidated JSONL shards.

Uses rate-limited concurrency and retry with backoff to avoid triggering
Common Crawl's 403 rate limits on data.commoncrawl.org.

Output: JSONL shards with {id, url, html, warc_file, timestamp, content_length}

Usage as executor step:
    step = ExecutorStep(
        name="downloaded/my_domain",
        fn=download_warc_records,
        config=WarcRecordDownloadConfig(
            cdx_manifest_path=cdx_step / "cdx_manifest.json",
            output_path=this_output_path(),
        ),
    )
"""

import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass

import fsspec
import requests
import warcio

from fray.v2.types import ResourceConfig as ZephyrResourceConfig
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)

DEFAULT_NUM_SHARDS = 500
DEFAULT_NUM_WORKERS = 64

# Common Crawl rate limit retry settings
MAX_DOWNLOAD_RETRIES = 3
RETRY_BASE_DELAY = 5.0  # seconds, doubles each retry


@dataclass
class WarcRecordDownloadConfig:
    """Configuration for downloading WARC records from Common Crawl.

    Args:
        cdx_manifest_path: Path to CDX manifest JSON (from cdx_query).
        output_path: Where to write consolidated HTML JSONL shards.
        num_output_shards: Number of output shards. If None, uses DEFAULT_NUM_SHARDS.
        num_workers: Number of Zephyr workers. Keep low (16) to avoid triggering
            Common Crawl's HTTP rate limits (403 errors at high concurrency).
    """

    cdx_manifest_path: str
    output_path: str
    num_output_shards: int | None = None
    num_workers: int | None = None


def _download_single_record(cdx_entry: dict, session: requests.Session) -> dict | None:
    """Download a single WARC record via HTTP byte-range request with retry.

    Retries on 403/429 (rate limit) with exponential backoff.
    Returns a dict with {id, url, html, warc_file, timestamp, content_length}
    or None if the download fails or the record isn't text/html.
    """
    filename = cdx_entry["filename"]
    offset = int(cdx_entry["offset"])
    length = int(cdx_entry["length"])

    url = f"https://data.commoncrawl.org/{filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    for attempt in range(MAX_DOWNLOAD_RETRIES):
        try:
            resp = session.get(url, headers=headers, timeout=120)
            if resp.status_code in (403, 429):
                delay = RETRY_BASE_DELAY * (2**attempt)
                if attempt < MAX_DOWNLOAD_RETRIES - 1:
                    logger.info(f"Rate limited ({resp.status_code}) on {filename}, retry {attempt + 1} in {delay:.0f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Rate limited ({resp.status_code}) on {filename} after {MAX_DOWNLOAD_RETRIES} attempts"
                    )
                    return None
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError:
            # Non-retryable HTTP error (already handled 403/429 above)
            logger.warning(f"Failed to download record from {filename} at offset {offset}: {resp.status_code}")
            return None
        except Exception as e:
            if attempt < MAX_DOWNLOAD_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.info(f"Download error on {filename}, retry {attempt + 1} in {delay:.0f}s: {e}")
                time.sleep(delay)
            else:
                logger.warning(f"Failed to download record from {filename} at offset {offset}: {e}")
                return None

    stream = io.BytesIO(resp.content)
    for record in warcio.ArchiveIterator(stream):
        if record.rec_type != "response":
            continue

        http_headers = record.http_headers
        if http_headers:
            content_type = http_headers.get_header("Content-Type") or ""
            if "text/html" not in content_type.lower():
                return None

        content = record.content_stream().read()
        html = content.decode("utf-8", errors="replace")
        target_url = record.rec_headers.get_header("WARC-Target-URI") or ""
        record_id = record.rec_headers.get_header("WARC-Record-ID") or ""

        return {
            "id": record_id,
            "url": target_url,
            "html": html,
            "warc_file": f"s3://commoncrawl/{filename}",
            "timestamp": cdx_entry.get("timestamp", ""),
            "content_length": len(html),
        }

    return None


def _download_shard(entries):
    """Download a shard of CDX entries, yielding non-empty results.

    Accepts an iterator (from Zephyr map_shard) and yields downloaded records.
    Creates one HTTP session per shard for connection pooling.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "marin-research-crawler/1.0 (academic research)"})
    successes = 0
    failures = 0
    for i, entry in enumerate(entries):
        record = _download_single_record(entry, session)
        if record is not None and record.get("content_length", 0) > 0:
            successes += 1
            yield record
        else:
            failures += 1
        if (i + 1) % 100 == 0:
            logger.info(f"  Shard progress: {i + 1} processed, {successes} ok, {failures} failed")
    logger.info(f"Shard complete: {successes + failures} entries -> {successes} records ({failures} failures)")


def _get_completed_shards(output_path: str) -> set[int]:
    """List existing output files and return set of completed shard indices."""
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
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


def download_warc_records(config: WarcRecordDownloadConfig):
    """Download WARC records via byte-range requests and write consolidated HTML JSONL.

    Reads the CDX manifest, downloads each record, and consolidates into sharded
    output files. Supports checkpoint resumption by skipping already-written shards.
    """
    logger.info(f"Loading CDX manifest from {config.cdx_manifest_path}...")
    cdx_entries: list[dict] = []
    if config.cdx_manifest_path.endswith(".jsonl") or config.cdx_manifest_path.endswith(".jsonl.gz"):
        with fsspec.open(config.cdx_manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    cdx_entries.append(json.loads(line))
    else:
        with fsspec.open(config.cdx_manifest_path, "r") as f:
            cdx_entries = json.load(f)
    total = len(cdx_entries)
    logger.info(f"Loaded {total} CDX entries")

    num_shards = config.num_output_shards or DEFAULT_NUM_SHARDS
    # Keep workers low to avoid triggering Common Crawl's HTTP rate limits.
    num_workers = config.num_workers or DEFAULT_NUM_WORKERS

    # Use Zephyr pipeline for parallel download + inline consolidation
    pipeline = (
        Dataset.from_list(cdx_entries)
        .reshard(num_shards)
        .map_shard(_download_shard)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{num_shards:05d}.jsonl.gz", skip_existing=True)
    )

    # WARC download is I/O-bound (HTTP requests), not CPU-bound.
    # Use fractional CPUs so workers aren't blocked by CPU scheduling limits.
    with ZephyrContext(
        name="download-warc-records",
        num_workers=num_workers,
        resources=ZephyrResourceConfig(cpu=0.1, ram="1g"),
    ) as ctx:
        output_files = ctx.execute(pipeline)

    stats = {
        "cdx_manifest": config.cdx_manifest_path,
        "total_cdx_entries": total,
        "num_output_shards": num_shards,
        "output_files": len(output_files),
    }
    with fsspec.open(os.path.join(config.output_path, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Done. {total} CDX entries -> {len(output_files)} shards at {config.output_path}")
