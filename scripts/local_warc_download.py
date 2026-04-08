#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local WARC downloader — runs on your laptop to bypass Common Crawl cloud rate limits.

Processes shards in REVERSE order (499 → 0) so it complements the cluster job
running forward. Upload completed shards to GCS as they finish and the cluster
will skip them via skip_existing.

Usage:
    # 1. Download the CDX manifest locally first (2 GB, one-time):
    gcloud storage cp gs://marin-us-central1/cdx/math_multi_v2_domain-98bdae/cdx_manifest.json /tmp/cdx_domain_manifest.json

    # 2. Run this script:
    python scripts/local_warc_download.py \
        --manifest /tmp/cdx_domain_manifest.json \
        --output /tmp/warc_download \
        --workers 64 \
        --reverse

    # 3. In another terminal, periodically sync completed shards to GCS:
    #    (only uploads new files, won't overwrite existing)
    while true; do
        gcloud storage cp --no-clobber /tmp/warc_download/data-*.jsonl.gz \
            gs://marin-us-central1/downloaded/math_multi_v2_domain_html-e7e65d/
        sleep 60
    done
"""

import argparse
import concurrent.futures
import gzip
import io
import json
import logging
import os
import time

import requests
import warcio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_SHARDS = 500
MAX_RETRIES = 4
RETRY_BASE_DELAY = 3.0


def download_single_record(cdx_entry: dict, session: requests.Session) -> dict | None:
    """Download a single WARC record via HTTP byte-range request with retry."""
    filename = cdx_entry["filename"]
    offset = int(cdx_entry["offset"])
    length = int(cdx_entry["length"])
    url = f"https://data.commoncrawl.org/{filename}"
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, headers=headers, timeout=120)
            if resp.status_code in (403, 429):
                delay = RETRY_BASE_DELAY * (2**attempt)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                    continue
                else:
                    return None
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError:
            return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2**attempt))
            else:
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


def process_shard(shard_idx: int, entries: list[dict], output_dir: str) -> tuple[int, int, int]:
    """Process a single shard: download all records and write output file.

    Returns (shard_idx, successes, failures).
    """
    outfile = os.path.join(output_dir, f"data-{shard_idx:05d}-of-{NUM_SHARDS:05d}.jsonl.gz")
    if os.path.exists(outfile):
        logger.info(f"Shard {shard_idx}: already exists, skipping")
        return (shard_idx, -1, 0)  # -1 = skipped

    session = requests.Session()
    session.headers.update({"User-Agent": "marin-research-crawler/1.0 (academic research)"})
    successes = 0
    failures = 0

    # Write to temp file, rename on completion (atomic)
    tmpfile = outfile + ".tmp"
    with gzip.open(tmpfile, "wt") as f:
        for i, entry in enumerate(entries):
            record = download_single_record(entry, session)
            if record is not None and record.get("content_length", 0) > 0:
                f.write(json.dumps(record) + "\n")
                successes += 1
            else:
                failures += 1
            if (i + 1) % 500 == 0:
                logger.info(f"  Shard {shard_idx}: {i + 1}/{len(entries)} processed, {successes} ok, {failures} failed")

    os.rename(tmpfile, outfile)
    logger.info(f"Shard {shard_idx} complete: {successes}/{successes + failures} records ({failures} failures)")
    return (shard_idx, successes, failures)


def main():
    parser = argparse.ArgumentParser(description="Local WARC downloader")
    parser.add_argument("--manifest", required=True, help="Path to CDX manifest JSON")
    parser.add_argument("--output", required=True, help="Local output directory")
    parser.add_argument("--workers", type=int, default=32, help="Number of concurrent shard workers")
    parser.add_argument("--reverse", action="store_true", help="Process shards in reverse order (499 → 0)")
    parser.add_argument("--start", type=int, default=None, help="Start shard (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End shard (inclusive)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info(f"Loading CDX manifest from {args.manifest}...")
    with open(args.manifest) as f:
        cdx_entries = json.load(f)
    logger.info(f"Loaded {len(cdx_entries)} CDX entries")

    # Shard the entries identically to Zephyr's reshard(500):
    # evenly divide into NUM_SHARDS chunks
    shard_size = (len(cdx_entries) + NUM_SHARDS - 1) // NUM_SHARDS
    shards: list[tuple[int, list[dict]]] = []
    for i in range(NUM_SHARDS):
        start = i * shard_size
        end = min(start + shard_size, len(cdx_entries))
        if start < len(cdx_entries):
            shards.append((i, cdx_entries[start:end]))

    # Filter to requested range
    if args.start is not None or args.end is not None:
        s = args.start if args.start is not None else 0
        e = args.end if args.end is not None else NUM_SHARDS - 1
        shards = [(idx, entries) for idx, entries in shards if s <= idx <= e]

    if args.reverse:
        shards.reverse()

    logger.info(f"Processing {len(shards)} shards with {args.workers} workers")
    logger.info(f"Order: {'reverse' if args.reverse else 'forward'}, range: {shards[0][0]}..{shards[-1][0]}")

    total_ok = 0
    total_fail = 0
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_shard, idx, entries, args.output): idx for idx, entries in shards}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                shard_idx, ok, fail = future.result()
                if ok >= 0:
                    total_ok += ok
                    total_fail += fail
                completed += 1
                if completed % 10 == 0:
                    logger.info(
                        f"Progress: {completed}/{len(shards)} shards done, {total_ok} records ok, {total_fail} failed"
                    )
            except Exception as e:
                logger.error(f"Shard {idx} raised exception: {e}")

    logger.info(f"Done! {completed} shards, {total_ok} records downloaded, {total_fail} failures")


if __name__ == "__main__":
    main()
