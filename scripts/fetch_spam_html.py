#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch original HTML from Common Crawl for a list of URLs.

Uses the CC cluster.idx + CDX shard files to find WARC byte offsets,
completely bypassing both the CDX HTTP API and the heavy DuckDB Parquet path.

The approach:
  1. Download cluster.idx (~50MB text, sorted SURT URL ranges -> CDX shard numbers)
  2. Binary search to find which CDX shard files contain our URLs
  3. Download only those specific CDX shards (~200KB each, gzipped)
  4. Parse shards for exact URL matches -> get WARC filename/offset/length
  5. Fetch raw HTML via byte-range requests to data.commoncrawl.org

Usage:
    python scripts/fetch_spam_html.py \
        --url-file /tmp/spam_urls.txt \
        --crawls CC-MAIN-2024-10 CC-MAIN-2024-22 \
        --output /path/to/garbage_nemotron_html
"""

import argparse
import bisect
import gzip
import io
import json
import logging
import os
import re
import time
from urllib.parse import urlparse

import requests
import warcio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CDX_COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
CC_INDEX_BASE = "https://data.commoncrawl.org/cc-index/collections/{crawl_id}/indexes/"

MAX_RETRIES = 4
RETRY_BACKOFF = 5


def get_available_crawl_indices() -> list[str]:
    resp = requests.get(CDX_COLLINFO_URL, timeout=30)
    resp.raise_for_status()
    return [idx["id"].replace("-index", "") for idx in resp.json()]


def url_to_surt(url: str) -> str:
    """Convert a URL to SURT (Sort-friendly URI Rewriting Transform) format.

    https://hacktechnology.xyz/buy-cbd-oil/ -> xyz,hacktechnology)/buy-cbd-oil/
    http://delgrid.com/2021/10/03/foo/ -> com,delgrid)/2021/10/03/foo/
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    # Reverse domain components: www.example.com -> com,example,www
    parts = host.split(".")
    parts.reverse()
    surt_host = ",".join(parts)
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{surt_host}){path}{query}"


def _download_with_retry(url: str, timeout: int = 60) -> bytes | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (attempt + 1)
                logger.warning(f"  Retry {attempt + 1}: {e}, waiting {wait}s")
                time.sleep(wait)
            else:
                logger.error(f"  Failed after {MAX_RETRIES} attempts: {e}")
                return None
    return None


def load_cluster_idx(crawl_id: str) -> list[tuple[str, str, int, int]]:
    """Download and parse cluster.idx for a crawl.

    Returns sorted list of (surt_key, cdx_filename, offset, length).
    """
    url = CC_INDEX_BASE.format(crawl_id=crawl_id) + "cluster.idx"
    logger.info(f"  Downloading cluster.idx from {crawl_id}...")
    data = _download_with_retry(url, timeout=120)
    if data is None:
        return []

    entries = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        # Format: SURT_URL TIMESTAMP\tCDX_FILE\tOFFSET\tLENGTH\tSHARD
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        surt_key = parts[0].split(" ")[0]  # strip timestamp
        cdx_file = parts[1]
        offset = int(parts[2])
        length = int(parts[3])
        entries.append((surt_key, cdx_file, offset, length))

    logger.info(f"  Loaded {len(entries)} cluster.idx entries")
    return entries


def find_cdx_shards(surt_url: str, cluster_entries: list[tuple[str, str, int, int]]) -> list[tuple[str, int, int]]:
    """Find which CDX shard(s) might contain a given SURT URL.

    Returns list of (cdx_filename, offset, length) to check.
    """
    keys = [e[0] for e in cluster_entries]
    idx = bisect.bisect_right(keys, surt_url)
    # The URL falls in the shard at idx-1 (the last key <= our URL)
    # Also check idx in case of boundary conditions
    candidates = []
    for i in range(max(0, idx - 1), min(len(cluster_entries), idx + 1)):
        entry = cluster_entries[i]
        candidates.append((entry[1], entry[2], entry[3]))
    return candidates


def search_cdx_shard(crawl_id: str, cdx_file: str, offset: int, length: int, target_urls: set[str]) -> list[dict]:
    """Download a specific CDX shard slice and search for target URLs.

    CDX shard lines are: SURT_URL TIMESTAMP JSON_BLOB
    """
    base = f"https://data.commoncrawl.org/cc-index/collections/{crawl_id}/indexes/{cdx_file}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(base, headers=headers, timeout=120)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
            else:
                logger.warning(f"  Failed to download CDX shard {cdx_file}: {e}")
                return []

    try:
        decompressed = gzip.decompress(resp.content).decode("utf-8", errors="replace")
    except Exception:
        logger.warning(f"  Failed to decompress CDX shard {cdx_file}")
        return []

    results = []
    for line in decompressed.splitlines():
        # Format: SURT_URL TIMESTAMP JSON_BLOB
        # Find the JSON blob (starts with '{')
        json_start = line.find("{")
        if json_start == -1:
            continue
        try:
            record = json.loads(line[json_start:])
        except json.JSONDecodeError:
            continue

        record_url = record.get("url", "")
        if record_url in target_urls and str(record.get("status", "")) == "200":
            # Remap field names to match our download_warc_html expectations
            results.append(
                {
                    "url": record_url,
                    "filename": record.get("filename", ""),
                    "offset": record.get("offset", ""),
                    "length": record.get("length", ""),
                    "timestamp": line.split(" ")[1] if " " in line else "",
                    "status": str(record.get("status", "")),
                    "mime": record.get("mime", ""),
                }
            )

    return results


def cluster_lookup_batch(urls: list[str], crawl_ids: list[str]) -> dict[str, dict]:
    """Look up all URLs using the cluster.idx + CDX shard approach."""
    best: dict[str, dict] = {}
    target_set = set(urls)

    for crawl_id in crawl_ids:
        if len(best) == len(urls):
            break

        cluster = load_cluster_idx(crawl_id)
        if not cluster:
            continue

        # Find which shards to check for each remaining URL
        shards_to_check: dict[tuple[str, int, int], set[str]] = {}  # shard -> urls to look for
        remaining = target_set - set(best.keys())

        for url in remaining:
            surt = url_to_surt(url)
            candidates = find_cdx_shards(surt, cluster)
            for shard_key in candidates:
                if shard_key not in shards_to_check:
                    shards_to_check[shard_key] = set()
                shards_to_check[shard_key].add(url)

        logger.info(f"  Checking {len(shards_to_check)} CDX shards for {len(remaining)} remaining URLs")

        for (cdx_file, offset, length), shard_urls in shards_to_check.items():
            results = search_cdx_shard(crawl_id, cdx_file, offset, length, shard_urls)
            for record in results:
                url = record["url"]
                if url not in best or record.get("timestamp", "") > best[url].get("timestamp", ""):
                    best[url] = record
                    logger.info(f"    Found: {url}")

        found_count = len(best)
        logger.info(f"  After {crawl_id}: {found_count}/{len(urls)} URLs found")

    return best


def download_warc_html(cdx_entry: dict) -> str | None:
    """Download the raw HTML from a WARC record via byte-range request."""
    filename = cdx_entry["filename"]
    offset = int(cdx_entry["offset"])
    length = int(cdx_entry["length"])

    warc_url = f"https://data.commoncrawl.org/{filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(warc_url, headers=headers, timeout=120)
            if resp.status_code in (403, 429):
                wait = RETRY_BACKOFF * (2**attempt)
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Rate limited ({resp.status_code}), retry in {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"Rate limited after {MAX_RETRIES} attempts")
                    return None
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
            else:
                logger.error(f"Download failed: {e}")
                return None

    stream = io.BytesIO(resp.content)
    for record in warcio.ArchiveIterator(stream):
        if record.rec_type != "response":
            continue
        content = record.content_stream().read()
        return content.decode("utf-8", errors="replace")

    return None


def url_to_filename(url: str) -> str:
    """Convert a URL to a filesystem-safe filename."""
    name = re.sub(r"^https?://", "", url)
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    name = name.strip("_")
    if len(name) > 200:
        name = name[:200]
    return name + ".html"


def main():
    parser = argparse.ArgumentParser(description="Fetch original HTML from Common Crawl for specific URLs")
    parser.add_argument("--urls", nargs="+", help="URLs to fetch")
    parser.add_argument("--url-file", help="File with one URL per line")
    parser.add_argument("--output", required=True, help="Output directory for HTML files")
    parser.add_argument("--crawls", nargs="+", help="Specific CC crawl IDs to search")
    args = parser.parse_args()

    urls = list(args.urls or [])
    if args.url_file:
        with open(args.url_file) as f:
            urls.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    if not urls:
        parser.error("Provide --urls or --url-file")

    os.makedirs(args.output, exist_ok=True)

    if args.crawls:
        crawl_ids = args.crawls
    else:
        crawl_ids = get_available_crawl_indices()[:20]

    logger.info(f"Looking up {len(urls)} URLs across {len(crawl_ids)} crawl indices")
    cdx_results = cluster_lookup_batch(urls, crawl_ids)
    logger.info(f"Index lookup complete: {len(cdx_results)}/{len(urls)} URLs found")

    index = {}
    for i, url in enumerate(urls):
        cdx = cdx_results.get(url)
        if cdx is None:
            logger.warning(f"[{i + 1}/{len(urls)}] No index entry: {url}")
            index[url] = {"status": "not_found"}
            continue

        logger.info(f"[{i + 1}/{len(urls)}] Downloading: {url}")
        html = download_warc_html(cdx)
        if html is None:
            logger.warning("  Failed to download HTML")
            index[url] = {"status": "download_failed", "cdx": cdx}
            continue

        fname = url_to_filename(url)
        out_path = os.path.join(args.output, fname)
        with open(out_path, "w") as f:
            f.write(html)

        logger.info(f"  Saved {len(html)} chars -> {fname}")
        index[url] = {
            "status": "ok",
            "filename": fname,
            "html_length": len(html),
            "cdx_timestamp": cdx.get("timestamp", ""),
            "cdx_crawl": cdx.get("filename", "").split("/")[1] if "/" in cdx.get("filename", "") else "",
        }

    index_path = os.path.join(args.output, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    ok = sum(1 for v in index.values() if v["status"] == "ok")
    logger.info(f"\nDone: {ok}/{len(urls)} pages fetched -> {args.output}/")


if __name__ == "__main__":
    main()
