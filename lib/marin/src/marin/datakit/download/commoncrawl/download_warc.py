# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download WARC files from Common Crawl and extract HTML into JSONL.

Converts S3-style Common Crawl paths to HTTPS URLs, downloads each WARC file,
parses it with warcio, and extracts HTTP response records with HTML content.
Output records have fields: id, html, url, metadata.

Usage as an ExecutorStep (see experiments/rephraser/rephraser_sweep.py) or standalone:
    uv run zephyr --backend=ray -- python -m marin.download.commoncrawl.download_warc \\
        --warc_paths '["s3://commoncrawl/crawl-data/.../warc.gz"]' \\
        --output_path gs://bucket/output/
"""

import io
import logging
from dataclasses import dataclass

import requests
import warcio
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)


@dataclass
class WarcDownloadConfig:
    warc_paths: list[str] | tuple[str, ...]
    """S3-style Common Crawl paths (e.g. s3://commoncrawl/crawl-data/...)."""

    output_path: str
    """GCS or local path to write output JSONL files."""

    max_pages_per_warc: int | None = None
    """Optional cap on pages extracted per WARC file (useful for testing)."""

    http_timeout: int = 120
    """Timeout in seconds for downloading each WARC file."""


def _s3_to_https(s3_path: str) -> str:
    """Convert an S3-style Common Crawl path to an HTTPS download URL.

    s3://commoncrawl/crawl-data/X -> https://data.commoncrawl.org/crawl-data/X
    """
    if s3_path.startswith("s3://commoncrawl/"):
        return "https://data.commoncrawl.org/" + s3_path[len("s3://commoncrawl/") :]
    if s3_path.startswith("https://"):
        return s3_path
    return "https://data.commoncrawl.org/" + s3_path


def _extract_html_from_warc(
    warc_path: str,
    max_pages: int | None = None,
    http_timeout: int = 120,
) -> list[dict]:
    """Download a WARC file and extract HTML response records.

    Returns a list of dicts with keys: id, html, url, metadata.
    """
    url = _s3_to_https(warc_path)
    logger.info("Downloading WARC: %s", url)

    response = requests.get(url, stream=True, timeout=http_timeout)
    response.raise_for_status()

    records = []
    raw_bytes = io.BytesIO(response.content)

    for record in warcio.ArchiveIterator(raw_bytes):
        if record.rec_type != "response":
            continue

        # Only keep HTTP responses with HTML content
        http_headers = record.http_headers
        if http_headers is None:
            continue
        content_type = http_headers.get_header("Content-Type") or ""
        if "text/html" not in content_type.lower():
            continue

        content = record.content_stream().read()
        html = content.decode("utf-8", errors="replace")

        record_id = record.rec_headers.get_header("WARC-Record-ID") or ""
        target_uri = record.rec_headers.get_header("WARC-Target-URI") or ""

        records.append(
            {
                "id": record_id,
                "html": html,
                "url": target_uri,
                "metadata": {
                    "warc_file": warc_path,
                    "content_length": len(html),
                },
            }
        )

        if max_pages and len(records) >= max_pages:
            break

    logger.info("Extracted %d HTML pages from %s", len(records), warc_path)
    return records


def _process_warc(warc_path: str) -> list[dict]:
    """Zephyr-compatible wrapper: process a single WARC path, return extracted records."""
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    max_pages = ctx.get_shared("max_pages_per_warc")
    http_timeout = ctx.get_shared("http_timeout")
    return _extract_html_from_warc(warc_path, max_pages=max_pages, http_timeout=http_timeout)


def download_and_extract_warcs(config: WarcDownloadConfig) -> None:
    """Download WARC files from Common Crawl and extract HTML into JSONL.

    Parallelizes across WARC files using Zephyr's Dataset.from_list.
    Each WARC file produces multiple HTML records written to JSONL output.
    """
    warc_paths = list(config.warc_paths)
    logger.info("Processing %d WARC files", len(warc_paths))

    pipeline = (
        Dataset.from_list(warc_paths)
        .flat_map(_process_warc)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="download-warcs") as ctx:
        ctx.put("max_pages_per_warc", config.max_pages_per_warc)
        ctx.put("http_timeout", config.http_timeout)
        output_files = ctx.execute(pipeline)

    logger.info("Wrote %d output files to %s", len(output_files), config.output_path)
