# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Query Common Crawl columnar index via DuckDB — fast replacement for cdx_query.

The CDX HTTP API at index.commoncrawl.org rate-limits to ~2 concurrent requests
per IP and bans at 8+. For bulk lookups (45 domains x 100+ crawls = 4500+ queries),
this takes days and fails frequently.

Common Crawl publishes the same data as Parquet files at
s3://commoncrawl/cc-index/table/cc-main/warc/, partitioned by crawl. DuckDB can
query these directly via HTTPS with predicate pushdown on Parquet row groups,
skipping most data. A single crawl query (300 Parquet files, ~300GB raw) typically
reads only a few GB after column projection + row group pruning.

This module is a drop-in replacement for query_cdx(): same CDXQueryConfig input,
same cdx_manifest.json + cdx_stats.json output. Crawls are queried in parallel
using a thread pool (one DuckDB connection per crawl), with per-crawl checkpointing.

Requires: pip install duckdb

Usage as executor step:
    step = ExecutorStep(
        name="cdx/my_domain",
        fn=query_cdx_columnar,
        config=CDXQueryConfig(
            url_patterns=["example.com"],
            output_path=this_output_path(),
        ),
    )
"""

import concurrent.futures
import gzip
import json
import logging
import os
import threading

import fsspec
import requests

from marin.download.commoncrawl.cdx_query import (
    CDXQueryConfig,
    dedup_cdx_records,
    get_available_crawl_indices,
)

logger = logging.getLogger(__name__)

CC_PATHS_URL_FMT = "https://data.commoncrawl.org/crawl-data/{crawl_id}/cc-index-table.paths.gz"
CC_DATA_BASE_URL = "https://data.commoncrawl.org/"

# Each DuckDB connection uses multiple threads internally for query execution.
# We limit per-connection threads to avoid oversubscription when running many
# crawls in parallel.
DUCKDB_THREADS_PER_CONNECTION = 4

# Default number of crawls to query in parallel. Each worker gets its own
# DuckDB connection; the bottleneck is network bandwidth, not CPU.
# Override via CDX_COLUMNAR_WORKERS env var.
DEFAULT_MAX_WORKERS = 10


def _get_parquet_urls(crawl_id: str) -> list[str] | None:
    """Fetch the list of Parquet file URLs for a crawl's warc subset.

    Returns None if this crawl doesn't have a columnar index (older crawls).
    """
    url = CC_PATHS_URL_FMT.format(crawl_id=crawl_id)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise
    paths = gzip.decompress(resp.content).decode().splitlines()
    warc_paths = [p for p in paths if "subset=warc" in p and p.endswith(".parquet")]
    if not warc_paths:
        return None
    return [f"{CC_DATA_BASE_URL}{p}" for p in warc_paths]


def _sql_quote(s: str) -> str:
    """Escape a string for SQL single-quote literals."""
    return "'" + s.replace("'", "''") + "'"


def _build_domain_filter(patterns: list[str], match_type: str) -> str:
    """Build a SQL WHERE clause that replicates CDX match_type semantics.

    CDX match types map to columnar index columns as follows:
      host   -> url_host_name = pattern
      domain -> url_host_registered_domain = pattern (captures all subdomains)
      prefix -> url_host_name = host_part AND url_path LIKE '/path_part%'
      exact  -> url = pattern
    """
    if match_type == "host":
        if len(patterns) == 1:
            return f"url_host_name = {_sql_quote(patterns[0])}"
        return f"url_host_name IN ({', '.join(_sql_quote(p) for p in patterns)})"

    elif match_type == "domain":
        if len(patterns) == 1:
            return f"url_host_registered_domain = {_sql_quote(patterns[0])}"
        return f"url_host_registered_domain IN ({', '.join(_sql_quote(p) for p in patterns)})"

    elif match_type == "prefix":
        conditions = []
        for p in patterns:
            if "/" in p:
                host, path = p.split("/", 1)
                conditions.append(
                    f"(url_host_name = {_sql_quote(host)} " f"AND url_path LIKE {_sql_quote('/' + path + '%')})"
                )
            else:
                conditions.append(f"url_host_name = {_sql_quote(p)}")
        return "(" + " OR ".join(conditions) + ")"

    elif match_type == "exact":
        if len(patterns) == 1:
            return f"url = {_sql_quote(patterns[0])}"
        return f"url IN ({', '.join(_sql_quote(p) for p in patterns)})"

    else:
        raise ValueError(f"Unknown match_type: {match_type}")


def _query_single_crawl(
    parquet_urls: list[str],
    domain_filter: str,
    status_filter: list[str],
    mime_filter: list[str],
) -> list[dict]:
    """Query one crawl's Parquet index and return CDX-compatible record dicts.

    Creates its own DuckDB connection so it can safely run in a thread pool.
    """
    status_clause = ""
    if status_filter:
        status_values = ", ".join(str(int(s)) for s in status_filter)
        status_clause = f"AND fetch_status IN ({status_values})"

    mime_clause = ""
    if mime_filter:
        # Substring match to handle "text/html; charset=utf-8" etc.
        mime_conditions = " OR ".join(
            f"LOWER(content_mime_detected) LIKE {_sql_quote('%' + m.lower() + '%')}" for m in mime_filter
        )
        mime_clause = f"AND ({mime_conditions})"

    import duckdb

    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {DUCKDB_THREADS_PER_CONNECTION};")
        con.execute("SET http_retries = 50;")
        con.execute("SET http_retry_wait_ms = 2000;")

        rel = con.read_parquet(parquet_urls, hive_partitioning=True)
        rel.create_view("_ccindex", replace=True)

        query = f"""
            SELECT
                url,
                CAST(fetch_status AS VARCHAR) AS status,
                content_mime_detected AS mime,
                strftime(fetch_time, '%Y%m%d%H%M%S') AS timestamp,
                warc_filename AS filename,
                CAST(warc_record_offset AS VARCHAR) AS "offset",
                CAST(warc_record_length AS VARCHAR) AS length
            FROM _ccindex
            WHERE {domain_filter}
            {status_clause}
            {mime_clause}
        """

        rows = con.execute(query).fetchall()
    finally:
        con.close()

    columns = ["url", "status", "mime", "timestamp", "filename", "offset", "length"]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _load_crawl_progress(output_path: str) -> dict[str, list[dict]]:
    """Load per-crawl checkpoint files from {output_path}/.progress_columnar/."""
    progress_dir = os.path.join(output_path, ".progress_columnar")
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


def _save_crawl_progress(output_path: str, crawl_id: str, records: list[dict]) -> None:
    """Save checkpoint for a completed crawl query."""
    progress_dir = os.path.join(output_path, ".progress_columnar")
    safe_crawl = crawl_id.replace("/", "_")
    dest = os.path.join(progress_dir, f"{safe_crawl}.json")
    with fsspec.open(dest, "w") as f:
        json.dump(records, f)


# Lock for thread-safe checkpoint writes and log counter updates
_progress_lock = threading.Lock()


def _worker(
    crawl_id: str,
    domain_filter: str,
    status_filter: list[str],
    mime_filter: list[str],
) -> tuple[str, list[dict]]:
    """Thread-pool worker: fetch file listing + query one crawl."""
    parquet_urls = _get_parquet_urls(crawl_id)
    if parquet_urls is None:
        return crawl_id, []
    records = _query_single_crawl(parquet_urls, domain_filter, status_filter, mime_filter)
    return crawl_id, records


def query_cdx_columnar(config: CDXQueryConfig):
    """Query Common Crawl columnar index via DuckDB — drop-in replacement for query_cdx.

    Crawls are queried in parallel using a thread pool. Each worker gets its own
    DuckDB connection (thread-safe, no shared state). The bottleneck is network
    bandwidth to CloudFront, which scales linearly with workers.

    Progress is checkpointed per crawl for preemption resilience. On restart,
    completed crawls are loaded from cache and only remaining crawls are queried.

    Control parallelism via CDX_COLUMNAR_WORKERS env var (default: 10).

    Output is identical to query_cdx(): cdx_manifest.json + cdx_stats.json.
    """
    crawl_indices = config.crawl_indices
    if crawl_indices is None:
        logger.info("Discovering available crawl indices...")
        # Retry crawl discovery since collinfo.json lives on the CDX API server
        # which may rate-limit or temporarily ban IPs.
        last_err = None
        for attempt in range(5):
            try:
                crawl_indices = get_available_crawl_indices()
                break
            except Exception as e:
                last_err = e
                wait = 10 * (attempt + 1)
                logger.warning(f"Crawl index discovery failed (attempt {attempt + 1}/5), retrying in {wait}s: {e}")
                import time

                time.sleep(wait)
        else:
            raise RuntimeError("Failed to discover crawl indices after 5 attempts") from last_err
        logger.info(f"Found {len(crawl_indices)} crawl indices")

    max_workers = int(os.environ.get("CDX_COLUMNAR_WORKERS", DEFAULT_MAX_WORKERS))

    domain_filter = _build_domain_filter(config.url_patterns, config.match_type)
    logger.info(f"Domain filter SQL: {domain_filter}")

    saved_progress = _load_crawl_progress(config.output_path)
    if saved_progress:
        logger.info(f"Loaded checkpoint for {len(saved_progress)} previously completed crawls")

    # Separate cached vs uncached crawls
    all_records: list[dict] = []
    crawls_to_query: list[str] = []

    for crawl_id in crawl_indices:
        if crawl_id in saved_progress:
            all_records.extend(saved_progress[crawl_id])
        else:
            crawls_to_query.append(crawl_id)

    if saved_progress:
        logger.info(
            f"Resumed: {len(saved_progress)} cached, "
            f"{len(crawls_to_query)} remaining, "
            f"{len(all_records)} records from cache"
        )

    if not crawls_to_query:
        logger.info("All crawls cached, skipping queries")
    else:
        logger.info(f"Querying {len(crawls_to_query)} crawls with {max_workers} parallel workers")

        completed = 0
        failed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_crawl = {
                pool.submit(
                    _worker,
                    crawl_id,
                    domain_filter,
                    config.status_filter,
                    config.mime_filter,
                ): crawl_id
                for crawl_id in crawls_to_query
            }

            for future in concurrent.futures.as_completed(future_to_crawl):
                crawl_id = future_to_crawl[future]
                try:
                    _, records = future.result()
                    with _progress_lock:
                        _save_crawl_progress(config.output_path, crawl_id, records)
                        all_records.extend(records)
                        completed += 1
                    logger.info(
                        f"[{completed + failed}/{len(crawls_to_query)}] {crawl_id}: "
                        f"{len(records)} records (total: {len(all_records)})"
                    )
                except Exception as e:
                    with _progress_lock:
                        failed += 1
                    logger.warning(f"[{completed + failed}/{len(crawls_to_query)}] {crawl_id}: " f"query failed: {e}")

        if failed:
            logger.warning(f"{failed} crawls failed (will be retried on next run)")

    logger.info(f"Total records before dedup: {len(all_records)}")

    if config.dedup_by_url:
        all_records = dedup_cdx_records(all_records)
        logger.info(f"After dedup: {len(all_records)} unique URLs")

    # Write manifest (same format as query_cdx)
    with fsspec.open(f"{config.output_path}/cdx_manifest.json", "w") as f:
        json.dump(all_records, f)

    stats = {
        "url_patterns": config.url_patterns,
        "crawl_indices_queried": len(crawl_indices),
        "crawl_indices_cached": len(crawl_indices) - len(crawls_to_query),
        "crawl_indices_failed": failed if crawls_to_query else 0,
        "max_workers": max_workers,
        "match_type": config.match_type,
        "total_records": len(all_records),
        "dedup_by_url": config.dedup_by_url,
        "method": "duckdb_columnar",
    }
    with fsspec.open(f"{config.output_path}/cdx_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"CDX manifest written: {len(all_records)} records -> {config.output_path}/cdx_manifest.json")
