#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe Common Crawl CDX API across multiple crawl indices for a source list.

Generalized version of validate_cdx_sources.py. For each source x crawl index,
queries CDX page 0 and estimates total records. Outputs a structured JSON report
and a human-readable summary.

Usage:
    # Probe medical sources across default crawl indices
    uv run python experiments/rephraser/probe_domain_sources.py \
        --source_file experiments/rephraser/medical_sources.txt

    # Custom crawl indices and output
    uv run python experiments/rephraser/probe_domain_sources.py \
        --source_file experiments/rephraser/law_sources.txt \
        --output experiments/rephraser/law_sources_probed.json \
        --crawls CC-MAIN-2013-48 CC-MAIN-2016-44 CC-MAIN-2025-47

    # Probe a single domain quickly
    uv run python experiments/rephraser/probe_domain_sources.py \
        --domain avvo.com --match_type domain
"""

import argparse
import json
import logging
import sys
import time

import fsspec
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default crawl indices to probe — covers the historical goldmine era through modern.
DEFAULT_CRAWL_INDICES = [
    "CC-MAIN-2013-48",
    "CC-MAIN-2014-23",
    "CC-MAIN-2016-07",
    "CC-MAIN-2016-44",
    "CC-MAIN-2017-47",
    "CC-MAIN-2018-47",
    "CC-MAIN-2020-50",
    "CC-MAIN-2022-49",
    "CC-MAIN-2024-46",
    "CC-MAIN-2025-47",
]

CDX_URL_FMT = "https://index.commoncrawl.org/{crawl_id}-index"
REQUEST_TIMEOUT = 60
RATE_LIMIT_DELAY = 3.0  # seconds between requests to avoid CDX API rate limiting
MAX_RETRIES = 2
RETRY_BACKOFF = 5  # seconds, multiplied by attempt number


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _cdx_get(url: str, params: dict) -> requests.Response | None:
    """Make a CDX API request with retries and exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in RETRYABLE_STATUS_CODES:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * attempt
                    logger.warning(f"  HTTP {resp.status_code}, retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(f"  HTTP {resp.status_code} after {MAX_RETRIES} attempts, giving up")
                    return resp  # return the error response so caller can handle
            return resp
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                logger.warning(f"  Connection error (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.warning(f"  Connection failed after {MAX_RETRIES} attempts: {e}")
                return None
    return None


def _write_json(path: str, data: dict):
    """Write JSON to a local or GCS path."""
    with fsspec.open(path, "w") as f:
        json.dump(data, f, indent=2)


def _read_json(path: str) -> dict | None:
    """Read JSON from a local or GCS path. Returns None if not found."""
    try:
        with fsspec.open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def parse_source_file(path: str) -> list[tuple[str, str]]:
    """Parse a source file into (url_pattern, match_type) pairs.

    Supports two formats:
    1. math_sources.txt style: "domain.com match_type  # comment"
    2. code_sources.txt style: "domain.com match_type" (no leading comment marker)

    Lines starting with # are comments. Blank lines are skipped.
    Duplicate entries are deduplicated (keeps first occurrence).
    """
    sources = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            url_pattern, match_type = parts[0], parts[1]
            if match_type not in ("domain", "host", "prefix", "exact"):
                continue
            key = (url_pattern, match_type)
            if key not in seen:
                sources.append(key)
                seen.add(key)
    return sources


def get_num_pages(crawl_id: str, url_pattern: str, match_type: str) -> int | None:
    """Query CDX showNumPages to get total page count for a source."""
    url = CDX_URL_FMT.format(crawl_id=crawl_id)
    params = {
        "url": url_pattern,
        "matchType": match_type,
        "showNumPages": "true",
    }
    resp = _cdx_get(url, params)
    if resp is None:
        return None
    try:
        if resp.status_code == 404:
            return 0
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get("pages", 0)
        return int(data) if data else 0
    except Exception as e:
        logger.warning(f"  numPages parse error for {url_pattern} @ {crawl_id}: {e}")
        return None


def probe_page0(crawl_id: str, url_pattern: str, match_type: str) -> tuple[int, int]:
    """Query CDX page 0 and count raw + filtered (200/html) records.

    Returns (raw_count, html200_count). Returns (-1, -1) on error.
    """
    url = CDX_URL_FMT.format(crawl_id=crawl_id)
    params = {
        "url": url_pattern,
        "output": "json",
        "matchType": match_type,
        "page": 0,
    }
    resp = _cdx_get(url, params)
    if resp is None:
        return -1, -1
    try:
        if resp.status_code == 404:
            return 0, 0
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"  page0 status error for {url_pattern} @ {crawl_id}: {e}")
        return -1, -1

    raw = 0
    filtered = 0
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            raw += 1
            if str(r.get("status", "")) == "200" and "text/html" in str(r.get("mime", "")).lower():
                filtered += 1
        except json.JSONDecodeError:
            raw += 1
    return raw, filtered


def probe_source(url_pattern: str, match_type: str, crawl_indices: list[str]) -> dict:
    """Probe a single source across all crawl indices.

    Returns a dict with per-crawl results and summary statistics.
    """
    result = {
        "url_pattern": url_pattern,
        "match_type": match_type,
        "crawls": {},
        "best_crawl": None,
        "best_crawl_estimate": 0,
        "total_estimate": 0,
    }

    for crawl_id in crawl_indices:
        time.sleep(RATE_LIMIT_DELAY)

        num_pages = get_num_pages(crawl_id, url_pattern, match_type)
        time.sleep(RATE_LIMIT_DELAY)

        page0_raw, page0_html200 = probe_page0(crawl_id, url_pattern, match_type)

        if num_pages is None and page0_raw == -1:
            result["crawls"][crawl_id] = {
                "status": "ERROR",
                "num_pages": num_pages,
                "page0_raw": page0_raw,
                "page0_html200": page0_html200,
                "estimated_total": None,
            }
            continue

        effective_pages = num_pages if num_pages is not None else 0

        if page0_html200 > 0:
            # Best case: we have actual page0 data to extrapolate from
            estimated = page0_html200 * max(effective_pages, 1)
            status = "OK"
        elif page0_raw > 0:
            estimated = page0_raw * max(effective_pages, 1)
            status = "NO_HTML200"
        elif effective_pages > 0:
            # page0 failed but numPages says there's data — use ~3000 records/page heuristic
            estimated = effective_pages * 3000
            status = "PAGE0_FAILED"
        else:
            estimated = 0
            status = "EMPTY"

        result["crawls"][crawl_id] = {
            "status": status,
            "num_pages": num_pages,
            "page0_raw": page0_raw,
            "page0_html200": page0_html200,
            "estimated_total": estimated,
        }

        if estimated > result["best_crawl_estimate"]:
            result["best_crawl"] = crawl_id
            result["best_crawl_estimate"] = estimated
        result["total_estimate"] += estimated

    return result


def print_summary(results: list[dict], crawl_indices: list[str]):
    """Print a human-readable summary table."""
    # Header
    crawl_cols = "  ".join(f"{c.replace('CC-MAIN-', ''):>10}" for c in crawl_indices)
    print(f"\n{'Source':<45} {'Match':<8} {crawl_cols}  {'Best':>10} {'Total':>10}")
    print("-" * (45 + 8 + len(crawl_indices) * 12 + 22))

    # Sort by total estimate descending
    for r in sorted(results, key=lambda x: x["total_estimate"], reverse=True):
        cols = []
        for crawl_id in crawl_indices:
            crawl_data = r["crawls"].get(crawl_id, {})
            est = crawl_data.get("estimated_total")
            if est is None:
                cols.append(f"{'ERR':>10}")
            elif est == 0:
                cols.append(f"{'0':>10}")
            else:
                cols.append(f"{est:>10,}")

        crawl_str = "  ".join(cols)
        best = r["best_crawl_estimate"]
        total = r["total_estimate"]
        best_str = f"{best:>10,}" if best > 0 else f"{'0':>10}"
        total_str = f"{total:>10,}" if total > 0 else f"{'0':>10}"
        print(f"{r['url_pattern']:<45} {r['match_type']:<8} {crawl_str}  {best_str} {total_str}")

    # Tier summary
    tier1 = [r for r in results if r["best_crawl_estimate"] >= 5000]
    tier2 = [r for r in results if 1000 <= r["best_crawl_estimate"] < 5000]
    tier3 = [r for r in results if 0 < r["best_crawl_estimate"] < 1000]
    empty = [r for r in results if r["best_crawl_estimate"] == 0]

    print(f"\nTier 1 (>=5k best crawl):  {len(tier1)} sources")
    print(f"Tier 2 (1k-5k):           {len(tier2)} sources")
    print(f"Tier 3 (<1k):             {len(tier3)} sources")
    print(f"Empty/error:              {len(empty)} sources")
    print(f"Total sources probed:     {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Probe CDX API for domain sources across crawl indices.")
    parser.add_argument("--source_file", type=str, help="Path to source list file (medical_sources.txt format)")
    parser.add_argument("--domain", type=str, help="Single domain to probe (alternative to --source_file)")
    parser.add_argument("--match_type", type=str, default="domain", help="Match type for --domain mode")
    parser.add_argument("--output", type=str, help="Output JSON path (default: <source_file>_probed.json)")
    parser.add_argument(
        "--crawls",
        nargs="+",
        default=DEFAULT_CRAWL_INDICES,
        help="Crawl indices to probe",
    )
    args = parser.parse_args()

    if args.source_file:
        sources = parse_source_file(args.source_file)
        output_path = args.output or args.source_file.replace(".txt", "_probed.json")
    elif args.domain:
        sources = [(args.domain, args.match_type)]
        output_path = args.output or "/dev/stdout"
    else:
        parser.error("Provide either --source_file or --domain")
        return 1

    logger.info(f"Probing {len(sources)} sources across {len(args.crawls)} crawl indices")
    logger.info(f"Crawls: {', '.join(args.crawls)}")

    # Resume from partial results if output file already exists
    results = []
    completed_keys: set[tuple[str, str]] = set()
    if output_path != "/dev/stdout":
        partial = _read_json(output_path)
        if partial:
            for r in partial.get("sources", []):
                results.append(r)
                completed_keys.add((r["url_pattern"], r["match_type"]))
            if completed_keys:
                logger.info(f"Resumed: {len(completed_keys)} sources already probed, skipping them")

    for i, (url_pattern, match_type) in enumerate(sources):
        if (url_pattern, match_type) in completed_keys:
            logger.info(f"[{i + 1}/{len(sources)}] {url_pattern} ({match_type}) — already probed, skipping")
            continue

        logger.info(f"[{i + 1}/{len(sources)}] {url_pattern} ({match_type})")
        result = probe_source(url_pattern, match_type, args.crawls)
        results.append(result)
        logger.info(
            f"  -> best: {result['best_crawl'] or 'none'} "
            f"({result['best_crawl_estimate']:,} est.), "
            f"total: {result['total_estimate']:,}"
        )

        # Save incrementally after each source so progress is never lost
        if output_path != "/dev/stdout":
            _write_json(output_path, {"crawl_indices": args.crawls, "sources": results})

    print_summary(results, args.crawls)

    # Final write (for --domain /dev/stdout mode)
    if output_path == "/dev/stdout":
        json.dump({"crawl_indices": args.crawls, "sources": results}, sys.stdout, indent=2)
    else:
        logger.info(f"\nFull results written to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
