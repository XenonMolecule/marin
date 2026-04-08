# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Query Common Crawl CDX API to build a manifest of WARC record pointers.

The CDX API (https://index.commoncrawl.org/CC-MAIN-{crawl_id}-index) accepts URL
patterns and returns JSONL with fields: filename, offset, length, timestamp, url,
status, mime. Each result is a pointer to a specific byte range in a WARC file.

This module generalizes the external CDX querying process used to create
mathhelpforum_cdx_deduped.json, making it programmatic and reusable.

The API paginates large result sets. For domains with many pages (e.g. forums),
a single un-paginated request can time out. This module always uses explicit
pagination via the ``showNumPages`` + ``page`` parameters to handle this reliably.

Usage as executor step:
    step = ExecutorStep(
        name="cdx/my_domain",
        fn=query_cdx,
        config=CDXQueryConfig(
            url_patterns=["example.com"],
            output_path=this_output_path(),
        ),
    )
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

import fsspec
import requests

logger = logging.getLogger(__name__)

CDX_COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
CDX_INDEX_URL_FMT = "https://index.commoncrawl.org/{crawl_id}-index"

# Default retry settings for CDX API requests. The API is flaky under load and
# returns 504s / drops connections frequently, especially from cloud VMs.
DEFAULT_MAX_RETRIES = 6
DEFAULT_RETRY_BACKOFF = 15  # seconds (multiplied by attempt number)
DEFAULT_REQUEST_TIMEOUT = 300  # seconds per HTTP request


@dataclass
class CDXQueryConfig:
    """Configuration for querying the Common Crawl CDX API.

    Args:
        url_patterns: URL patterns to query (e.g. ["mathhelpforum.com"]).
        output_path: Where to write the CDX manifest JSON.
        crawl_indices: Specific crawl IDs (e.g. ["CC-MAIN-2024-01"]). None queries all.
        match_type: CDX matchType parameter: "domain", "prefix", "exact", "host".
        dedup_by_url: Keep only the most recent capture per URL.
        status_filter: Only keep records with these HTTP status codes.
        mime_filter: Only keep records with these MIME types.
        request_delay: Seconds to wait between CDX API requests (rate limiting).
        max_retries: Max retries per CDX API request (the API returns 504s frequently).
        retry_backoff: Base backoff in seconds between retries (multiplied by attempt number).
        request_timeout: HTTP timeout in seconds for each CDX API request.
    """

    url_patterns: list[str]
    output_path: str
    crawl_indices: list[str] | None = None
    match_type: str = "domain"
    dedup_by_url: bool = True
    status_filter: list[str] = field(default_factory=lambda: ["200"])
    mime_filter: list[str] = field(default_factory=lambda: ["text/html"])
    request_delay: float = 0.5
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff: float = DEFAULT_RETRY_BACKOFF
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT


def get_available_crawl_indices() -> list[str]:
    """Fetch the list of available Common Crawl crawl indices.

    Returns:
        List of crawl IDs like ["CC-MAIN-2024-51", "CC-MAIN-2024-46", ...],
        sorted newest first.
    """
    resp = requests.get(CDX_COLLINFO_URL, timeout=30)
    resp.raise_for_status()
    indices = resp.json()
    return [idx["id"].replace("-index", "") for idx in indices]


def _cdx_request_with_retries(
    url: str,
    params: dict,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    label: str = "CDX request",
) -> requests.Response | None:
    """Make an HTTP GET to the CDX API with aggressive retries.

    Returns the Response on success, or None if the server returned 404.
    Raises RuntimeError after exhausting all retries.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=request_timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = retry_backoff * (attempt + 1)
                logger.warning(f"{label} failed (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
    raise RuntimeError(f"{label} failed after {max_retries} attempts") from last_exc


def _get_num_pages(
    index_url: str,
    url_pattern: str,
    match_type: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> int:
    """Ask the CDX API how many pages a query will return (with retries)."""
    params = {
        "url": url_pattern,
        "output": "json",
        "matchType": match_type,
        "showNumPages": "true",
    }
    resp = _cdx_request_with_retries(
        index_url,
        params,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        request_timeout=request_timeout,
        label="CDX showNumPages",
    )
    if resp is None:
        return 0
    info = resp.json()
    return info.get("pages", 0)


def _fetch_page(
    index_url: str,
    url_pattern: str,
    match_type: str,
    page: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> list[dict]:
    """Fetch a single page of CDX results with retries."""
    params = {
        "url": url_pattern,
        "output": "json",
        "matchType": match_type,
        "page": page,
    }
    resp = _cdx_request_with_retries(
        index_url,
        params,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        request_timeout=request_timeout,
        label=f"CDX page {page}",
    )
    if resp is None:
        return []
    results = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Skipping unparseable CDX line: {line[:200]}")
    return results


def query_single_index(
    url_pattern: str,
    crawl_id: str,
    match_type: str = "domain",
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> list[dict]:
    """Query a single CDX index for a URL pattern, handling pagination.

    For large domains, the CDX API paginates results. This function fetches
    the page count first, then retrieves all pages sequentially.

    Returns a list of CDX record dicts with keys: url, filename, offset, length,
    timestamp, status, mime (and other fields returned by the API).
    """
    index_url = CDX_INDEX_URL_FMT.format(crawl_id=crawl_id)
    retry_kwargs = dict(max_retries=max_retries, retry_backoff=retry_backoff, request_timeout=request_timeout)

    num_pages = _get_num_pages(index_url, url_pattern, match_type, **retry_kwargs)
    if num_pages == 0:
        return []

    all_results: list[dict] = []
    for page in range(num_pages):
        results = _fetch_page(index_url, url_pattern, match_type, page, **retry_kwargs)
        all_results.extend(results)
        if num_pages > 1:
            logger.info(f"  Page {page + 1}/{num_pages}: {len(results)} records")

    return all_results


def filter_cdx_records(
    records: list[dict],
    status_filter: list[str],
    mime_filter: list[str],
) -> list[dict]:
    """Filter CDX records by HTTP status code and MIME type."""
    filtered = []
    for r in records:
        status = str(r.get("status", ""))
        mime = str(r.get("mime", "")).lower()
        if status_filter and status not in status_filter:
            continue
        if mime_filter and not any(m.lower() in mime for m in mime_filter):
            continue
        filtered.append(r)
    return filtered


def dedup_cdx_records(records: list[dict]) -> list[dict]:
    """Deduplicate CDX records by URL, keeping the most recent timestamp."""
    by_url: dict[str, dict] = {}
    for r in records:
        url = r.get("url", "")
        ts = r.get("timestamp", "")
        if url not in by_url or ts > by_url[url].get("timestamp", ""):
            by_url[url] = r
    return list(by_url.values())


def _progress_key(pattern: str, crawl_id: str) -> str:
    """Build a filesystem-safe key for a (pattern, crawl_id) pair."""
    safe_pattern = re.sub(r"[^a-zA-Z0-9_.-]", "_", pattern)
    safe_crawl = re.sub(r"[^a-zA-Z0-9_.-]", "_", crawl_id)
    return f"{safe_pattern}__{safe_crawl}"


def _load_progress(output_path: str) -> dict[str, list[dict]]:
    """Load previously saved per-query progress files from ``{output_path}/.progress/``.

    Returns a dict mapping progress keys to their saved record lists.
    """
    progress_dir = os.path.join(output_path, ".progress")
    fs, _, _ = fsspec.get_fs_token_paths(progress_dir)
    saved: dict[str, list[dict]] = {}
    try:
        paths = fs.ls(progress_dir, detail=False)
    except FileNotFoundError:
        return saved
    for path in paths:
        basename = path.split("/")[-1]
        if not basename.endswith(".json"):
            continue
        key = basename[: -len(".json")]
        try:
            with fs.open(path, "r") as f:
                saved[key] = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping corrupt progress file {path}: {e}")
    return saved


def _save_progress(output_path: str, key: str, records: list[dict]) -> None:
    """Persist filtered records for a single (pattern, crawl_id) query."""
    progress_dir = os.path.join(output_path, ".progress")
    dest = os.path.join(progress_dir, f"{key}.json")
    with fsspec.open(dest, "w") as f:
        json.dump(records, f)


def query_cdx(config: CDXQueryConfig):
    """Query Common Crawl CDX API for all URL patterns, deduplicate, write manifest.

    Queries each (url_pattern, crawl_index) combination, filters by status and MIME,
    deduplicates by URL (keeping most recent), and writes the manifest JSON.

    Progress is checkpointed per (pattern, crawl_id) so that interrupted runs
    resume where they left off instead of starting from scratch.
    """
    crawl_indices = config.crawl_indices
    if crawl_indices is None:
        logger.info("Discovering available crawl indices...")
        crawl_indices = get_available_crawl_indices()
        logger.info(f"Found {len(crawl_indices)} crawl indices")

    saved_progress = _load_progress(config.output_path)
    if saved_progress:
        logger.info(f"Loaded progress for {len(saved_progress)} previously completed queries")

    all_records: list[dict] = []
    total_queries = len(config.url_patterns) * len(crawl_indices)
    query_num = 0
    skipped = 0

    for pattern in config.url_patterns:
        for crawl_id in crawl_indices:
            query_num += 1
            key = _progress_key(pattern, crawl_id)

            if key in saved_progress:
                all_records.extend(saved_progress[key])
                skipped += 1
                logger.info(
                    f"Skipping CDX [{query_num}/{total_queries}]: {pattern} @ {crawl_id} "
                    f"({len(saved_progress[key])} cached records)"
                )
                continue

            logger.info(f"Querying CDX [{query_num}/{total_queries}]: {pattern} @ {crawl_id}")
            try:
                records = query_single_index(
                    pattern,
                    crawl_id,
                    config.match_type,
                    max_retries=config.max_retries,
                    retry_backoff=config.retry_backoff,
                    request_timeout=config.request_timeout,
                )
                records = filter_cdx_records(records, config.status_filter, config.mime_filter)
                _save_progress(config.output_path, key, records)
                all_records.extend(records)
                logger.info(f"  Got {len(records)} records (total so far: {len(all_records)})")
            except Exception as e:
                logger.warning(f"  Failed to query {crawl_id} for {pattern}: {e}")

            if config.request_delay > 0:
                time.sleep(config.request_delay)

    if skipped:
        logger.info(f"Resumed from checkpoint: {skipped}/{total_queries} queries were cached")
    logger.info(f"Total records before dedup: {len(all_records)}")

    if config.dedup_by_url:
        all_records = dedup_cdx_records(all_records)
        logger.info(f"After dedup: {len(all_records)} unique URLs")

    # Write manifest in both JSON (backward compat) and JSONL (memory-efficient) formats
    with fsspec.open(f"{config.output_path}/cdx_manifest.json", "w") as f:
        json.dump(all_records, f)
    with fsspec.open(f"{config.output_path}/cdx_manifest.jsonl", "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    stats = {
        "url_patterns": config.url_patterns,
        "crawl_indices_queried": len(crawl_indices),
        "match_type": config.match_type,
        "total_records": len(all_records),
        "dedup_by_url": config.dedup_by_url,
    }
    with fsspec.open(f"{config.output_path}/cdx_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"CDX manifest written: {len(all_records)} records -> {config.output_path}/cdx_manifest.json")
