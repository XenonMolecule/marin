#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local CDX query helper — runs queries from your Mac and uploads progress files.

Queries the CDX API for (domain, crawl_id) pairs and uploads results directly
to the remote .progress/ directory so the running Ray job skips them.

Usage:
    python experiments/rephraser/local_cdx_helper.py
"""

import json
import subprocess
import sys
import tempfile

# Add the project to the path
sys.path.insert(0, "lib/marin/src")

from marin.datakit.download.commoncrawl.cdx_query import (
    filter_cdx_records,
    query_single_index,
    _progress_key,
)

PROGRESS_BASE = "gs://marin-us-central1/cdx/code_host-283157/.progress"

CRAWL_INDICES = [
    "CC-MAIN-2026-08",
    "CC-MAIN-2026-04",
    "CC-MAIN-2025-51",
    "CC-MAIN-2025-47",
    "CC-MAIN-2025-43",
    "CC-MAIN-2025-38",
    "CC-MAIN-2025-33",
    "CC-MAIN-2025-30",
    "CC-MAIN-2024-51",
    "CC-MAIN-2024-46",
]

# Domains to query, ordered so we start with those the remote job will hit LAST
DOMAINS_REVERSE = [
    "devdocs.io",
    "doc.qt.io",
    "api.flutter.dev",
    "www.tutorialspoint.com",
    # w3schools has 3/10 done, query the remaining 7
    "www.w3schools.com",
    # tensorflow has 7/10 done, query the remaining 3
    "www.tensorflow.org",
    # mozilla has 9/10 done, query the last 1
    "developer.mozilla.org",
]


def check_exists(domain: str, crawl_id: str) -> bool:
    """Check if a progress file already exists on GCS."""
    key = _progress_key(domain, crawl_id)
    dest = f"{PROGRESS_BASE}/{key}.json"
    result = subprocess.run(
        ["gcloud", "storage", "ls", dest],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


def upload_progress(domain: str, crawl_id: str, records: list[dict]) -> None:
    """Upload a progress file to GCS."""
    key = _progress_key(domain, crawl_id)
    dest = f"{PROGRESS_BASE}/{key}.json"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(records, f)
        tmp_path = f.name
    subprocess.run(
        ["gcloud", "storage", "cp", tmp_path, dest],
        capture_output=True,
        text=True,
        timeout=30,
    )


def main():
    total_queried = 0
    total_skipped = 0
    total_records = 0

    for domain in DOMAINS_REVERSE:
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        for crawl_id in CRAWL_INDICES:
            key = _progress_key(domain, crawl_id)

            # Skip if already done
            if check_exists(domain, crawl_id):
                print(f"  [{crawl_id}] SKIP (already exists)")
                total_skipped += 1
                continue

            print(f"  [{crawl_id}] Querying...", end="", flush=True)
            try:
                raw_records = query_single_index(
                    url_pattern=domain,
                    crawl_id=crawl_id,
                    match_type="host",
                    max_retries=4,
                    retry_backoff=10,
                    request_timeout=300,
                )
                # Filter for status=200, mime=text/html (same as the remote job)
                filtered = filter_cdx_records(raw_records, ["200"], ["text/html"])
                print(f" {len(raw_records)} raw -> {len(filtered)} filtered", end="", flush=True)

                # Upload
                upload_progress(domain, crawl_id, filtered)
                print(" -> UPLOADED")
                total_queried += 1
                total_records += len(filtered)

            except Exception as e:
                print(f" FAILED: {e}")

        print(
            f"  Domain done. Running totals: {total_queried} queried, {total_skipped} skipped, {total_records} records"
        )

    print(f"\n{'='*60}")
    print(f"DONE: {total_queried} queried, {total_skipped} skipped, {total_records} total records")


if __name__ == "__main__":
    main()
