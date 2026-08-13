# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Incremental WARC download with per-file resumability.

Downloads Common Crawl WARC files and extracts HTML response records. Unlike the
standard download_and_extract_warcs which hashes the full WARC list (making any
change re-download everything), this version:

- Reads WARC paths from a manifest file (not embedded in the config)
- Writes one output shard per WARC file with deterministic naming
- Uses skip_existing so adding 10 new WARCs doesn't re-download the existing 3000

Robustness:
- 600s timeout per WARC (1GB files over potentially slow connections)
- 5 retries with exponential backoff on connection errors, timeouts, HTTP 429/503
- Validation: warns if a WARC yields 0 HTML records
- Fails loudly if all retries exhausted (never silently drops a WARC)
"""

import hashlib
import io
import logging
import time
from dataclasses import dataclass

import fsspec
import requests
import warcio
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5.0  # seconds, doubles each retry
RETRYABLE_STATUS_CODES = {429, 503}
HTTP_TIMEOUT = 600  # 10 minutes — 1GB at 2MB/s = 500s


@dataclass
class IncrementalWarcDownloadConfig:
    warc_manifest_path: str
    """Path to text file with one WARC S3 path per line."""

    output_path: str
    """GCS or local path to write output JSONL files."""


def _s3_to_https(s3_path: str) -> str:
    """Convert s3://commoncrawl/... to https://data.commoncrawl.org/..."""
    if s3_path.startswith("s3://commoncrawl/"):
        return "https://data.commoncrawl.org/" + s3_path[len("s3://commoncrawl/") :]
    if s3_path.startswith("https://"):
        return s3_path
    return "https://data.commoncrawl.org/" + s3_path


def _warc_path_hash(warc_path: str) -> str:
    """Deterministic short hash of a WARC path for stable output filenames."""
    return hashlib.sha256(warc_path.encode()).hexdigest()[:12]


def _warc_cache_path(warc_path: str) -> str | None:
    """In-region GCS path for a WARC's cached raw ``.warc.gz``, or ``None`` (local runs).

    STRICTLY within-region: derived from ``marin_prefix()`` — the worker's OWN region
    bucket — and NEVER references another region. A WARC is multi-GB, so a cross-region
    read here would be paid egress; on any resolution error we return ``None`` and the
    caller falls back to Common Crawl. The ``tmp/ttl=2d/`` prefix is auto-expired by the
    bucket lifecycle rule (delete after 2 days), so the cache self-cleans — no code-side
    deletion, no bucket-config changes.
    """
    try:
        from rigging.filesystem import marin_prefix

        prefix = marin_prefix()
        if not prefix.startswith("gs://"):
            return None
        return f"{prefix}/tmp/ttl=2d/warc-cache/{_warc_path_hash(warc_path)}.warc.gz"
    except Exception:
        return None


def _download_warc_bytes_from_cc(warc_path: str) -> bytes:
    """Fetch a WARC's raw ``.warc.gz`` bytes from Common Crawl over HTTP.

    Retries on transient errors; raises ``RuntimeError`` on permanent failure.
    """
    url = _s3_to_https(warc_path)
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Downloading WARC from Common Crawl (attempt {attempt + 1}): {warc_path}")
            response = requests.get(url, stream=True, timeout=HTTP_TIMEOUT)

            if response.status_code in RETRYABLE_STATUS_CODES:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    f"Rate limited ({response.status_code}) on {warc_path}, "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s"
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response.content

        except requests.exceptions.RequestException as e:
            delay = RETRY_BASE_DELAY * (2**attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Download error on {warc_path} (attempt {attempt + 1}): {e}. Retrying in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to download {warc_path} after {MAX_RETRIES} attempts: {e}") from e

    raise RuntimeError(f"Failed to download {warc_path} after {MAX_RETRIES} attempts")


def _fetch_warc_bytes(warc_path: str) -> bytes:
    """Return a WARC's raw ``.warc.gz`` bytes: in-region GCS cache first, else Common
    Crawl (populating the cache on the way).

    Read-through, within-region cache under ``tmp/ttl=2d/`` (see ``_warc_cache_path``).
    Purpose: make preemption cheap — a resumed worker re-reads the local copy instead of
    re-pulling multi-GB from Common Crawl — and cut CC egress to one fetch per WARC per
    region. Correctness never depends on the cache: a miss or any read/write error falls
    back to Common Crawl. Cached bytes are the verbatim CC ``response.content``, so
    extraction output is byte-for-byte unchanged.
    """
    cache_path = _warc_cache_path(warc_path)

    if cache_path is not None:
        try:
            with fsspec.open(cache_path, "rb") as f:
                data = f.read()
            logger.info(f"WARC cache HIT (in-region): {warc_path}")
            return data
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"WARC cache read failed for {warc_path} ({e}) — using Common Crawl")

    data = _download_warc_bytes_from_cc(warc_path)

    if cache_path is not None:
        try:
            with fsspec.open(cache_path, "wb") as f:
                f.write(data)
            logger.info(f"WARC cache WRITE (in-region): {warc_path}")
        except Exception as e:
            # Best-effort: a race (another worker wrote it) or transient error is harmless.
            logger.info(f"WARC cache write skipped for {warc_path} ({e})")

    return data


def _download_one_warc(warc_path: str) -> list[dict]:
    """Fetch a single WARC (in-region cache or Common Crawl) and extract HTML response
    records.

    Returns list of {id, html, url, metadata} dicts. Raises on permanent download failure.
    """
    raw_bytes = io.BytesIO(_fetch_warc_bytes(warc_path))

    records = []
    parse_errors = 0
    for record in warcio.ArchiveIterator(raw_bytes):
        try:
            if record.rec_type != "response":
                continue

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
        except Exception as e:
            # Some WARC records have brotli decompression issues or other
            # corruption. Skip the individual record, not the whole file.
            parse_errors += 1
            if parse_errors <= 3:
                logger.warning(f"Skipping corrupt record in {warc_path}: {e}")

    if parse_errors > 0:
        logger.warning(f"Skipped {parse_errors} corrupt records in {warc_path}")

    if not records:
        logger.warning(f"WARC yielded 0 HTML records: {warc_path}")
    else:
        logger.info(f"Extracted {len(records)} HTML pages from {warc_path}")

    return records


def _load_manifest(manifest_path: str) -> list[str]:
    """Load WARC paths from a manifest file."""
    import fsspec

    with fsspec.open(manifest_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def download_warcs_incremental(config: IncrementalWarcDownloadConfig) -> None:
    """Download WARC files incrementally with per-file resumability."""
    warc_paths = _load_manifest(config.warc_manifest_path)
    logger.info(f"Loaded {len(warc_paths)} WARC paths from {config.warc_manifest_path}")

    def _output_path_fn(shard_idx: int, total_shards: int) -> str:
        warc_path = warc_paths[shard_idx]
        h = _warc_path_hash(warc_path)
        return f"{config.output_path}/data-{h}.jsonl.gz"

    pipeline = (
        Dataset.from_list(warc_paths)
        .reshard(len(warc_paths))
        .flat_map(_download_one_warc)
        .write_jsonl(_output_path_fn, skip_existing=True)
    )

    # Each worker loads a full WARC body (~1-1.3 GB compressed) into memory via
    # ``response.content`` and accumulates all decoded HTML records into a list.
    # HTML decompression expansion (5-10x) + Python string/dict overhead means
    # adversarial WARCs can peak well above 16 GB (one 1.3 GB WARC consistently
    # OOM'd on a 16 GiB worker, then cascaded the whole job). 24 GiB has wider
    # headroom for outliers.
    ctx = ZephyrContext(
        name="download-warcs",
        max_workers=500,
        resources=ResourceConfig(cpu=1, ram="24g"),
    )
    ctx.execute(pipeline)

    logger.info(f"Download complete. {len(warc_paths)} WARCs → {config.output_path}")
