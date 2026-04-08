# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that every WARC file has at least one Nemotron-CC match.

Reads Nemotron filtered output and metadata to verify complete WARC coverage.
Outputs a per-WARC match count to GCS.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py --cluster marin-big-run --no_wait \
        -- python experiments/baseline_collection/check_warc_coverage.py
"""

import json
import logging
from collections import Counter

import fsspec

from zephyr.readers import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NEMOTRON_OUTPUT = "gs://marin-us-central2/filtered/baseline_nemotron-037958"
METADATA_PATH = "gs://marin-us-central2/metadata/baseline_warc_metadata-d671da"
OUTPUT_PATH = "gs://marin-us-central2/raw/baseline-dataset-collection/warc_coverage.json"


def main():
    # Step 1: Collect all unique URLs from Nemotron output
    logger.info("Reading Nemotron output URLs...")
    fs = fsspec.filesystem("gcs")
    nemotron_dir = NEMOTRON_OUTPUT.replace("gs://", "")
    nemotron_files = [
        f"gs://{f}" for f in fs.ls(nemotron_dir, detail=False) if "CC-MAIN" in f and f.endswith(".jsonl.gz")
    ]
    logger.info(f"Found {len(nemotron_files)} Nemotron output files")

    nemotron_urls = set()
    for i, fpath in enumerate(nemotron_files):
        for record in load_file(fpath):
            url = record.get("url", "")
            if url:
                nemotron_urls.add(url)
        if (i + 1) % 100 == 0:
            logger.info(f"  Read {i + 1}/{len(nemotron_files)} files, {len(nemotron_urls):,} unique URLs")

    logger.info(f"Total Nemotron URLs: {len(nemotron_urls):,}")

    # Step 2: Read metadata to map URLs to WARC files
    logger.info("Reading metadata to map URLs → WARC files...")
    metadata_dir = METADATA_PATH.replace("gs://", "")
    metadata_files = sorted(f"gs://{f}" for f in fs.ls(metadata_dir, detail=False) if f.endswith(".jsonl.gz"))
    logger.info(f"Found {len(metadata_files)} metadata files")

    warc_match_counts = Counter()  # warc_file → number of nemotron matches
    warc_total_counts = Counter()  # warc_file → total records
    total_checked = 0

    for i, fpath in enumerate(metadata_files):
        for record in load_file(fpath):
            url = record.get("url", "")
            warc_file = record.get("warc_file", "")
            if warc_file:
                warc_total_counts[warc_file] += 1
                if url in nemotron_urls:
                    warc_match_counts[warc_file] += 1
            total_checked += 1
        if (i + 1) % 500 == 0:
            logger.info(f"  Read {i + 1}/{len(metadata_files)} metadata files, {total_checked:,} records checked")

    logger.info(f"Total metadata records: {total_checked:,}")
    logger.info(f"Unique WARC files in metadata: {len(warc_total_counts):,}")
    logger.info(f"WARC files with ≥1 Nemotron match: {len(warc_match_counts):,}")

    # Step 3: Find WARCs with zero matches
    zero_match = [wf for wf in warc_total_counts if wf not in warc_match_counts]
    logger.info(f"WARC files with ZERO Nemotron matches: {len(zero_match)}")
    if zero_match:
        for wf in zero_match[:20]:
            logger.warning(f"  MISSING: {wf} ({warc_total_counts[wf]} total records)")

    # Step 4: Write results
    result = {
        "total_warc_files": len(warc_total_counts),
        "warcs_with_matches": len(warc_match_counts),
        "warcs_without_matches": len(zero_match),
        "total_nemotron_urls": len(nemotron_urls),
        "total_metadata_records": total_checked,
        "zero_match_warcs": zero_match[:100],
        "match_count_distribution": {
            "min": min(warc_match_counts.values()) if warc_match_counts else 0,
            "max": max(warc_match_counts.values()) if warc_match_counts else 0,
            "mean": sum(warc_match_counts.values()) / len(warc_match_counts) if warc_match_counts else 0,
        },
    }

    with fs.open(OUTPUT_PATH.replace("gs://", ""), "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results written to {OUTPUT_PATH}")
    logger.info(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
