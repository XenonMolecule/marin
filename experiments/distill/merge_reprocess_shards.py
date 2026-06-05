# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge reprocessed dataset shards from multiple clusters/row ranges.

After running web_extraction_reprocess.py on multiple clusters (each handling
a row range), this script combines all reassembled outputs, deduplicates by
global_row_idx, verifies completeness, and writes the final dataset.

No assumptions about which cluster processed which rows — fully flexible.

Usage:
    python experiments/distill/merge_reprocess_shards.py \\
        --input_dirs \\
            gs://marin-us-central1/distill/.../reassembled_rows_0_200000-abc123/ \\
            gs://marin-us-east5/distill/.../reassembled_rows_200000_400000-def456/ \\
        --output_dir gs://marin-us-central1/distill/web_extraction_merged/ \\
        --expected_rows 818880
"""

import argparse
import gzip
import json
import logging
import os

import fsspec
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

FINAL_COLUMNS = ("messages", "warc_file", "doc_id", "spec_id", "spec", "model")


def merge_shards(
    input_dirs: list[str],
    output_dir: str,
    expected_rows: int,
    records_per_shard: int = 10_000,
) -> None:
    """Merge reassembled shards from multiple directories.

    1. Read all JSONL.gz files from all input directories.
    2. Deduplicate by global_row_idx (last-write-wins).
    3. Sort by global_row_idx for deterministic output.
    4. Verify completeness — warn on gaps.
    5. Drop global_row_idx and write final sharded JSONL.gz.
    """
    # Collect all records
    records_by_idx: dict[int, dict] = {}
    total_read = 0
    duplicates = 0

    # Aggregate token stats from all input directories
    agg_input_tokens = 0
    agg_output_tokens = 0
    agg_reasoning_tokens = 0
    for input_dir in input_dirs:
        # Each API inference run writes an api_inference_stats.json alongside its shards.
        # The reassemble step's output is one level down, so check the parent too.
        for stats_dir in [input_dir, os.path.dirname(input_dir.rstrip("/"))]:
            stats_path = os.path.join(stats_dir, "api_inference_stats.json")
            try:
                with fsspec.open(stats_path, "r") as f:
                    s = json.load(f)
                agg_input_tokens += s.get("total_input_tokens", 0)
                agg_output_tokens += s.get("total_output_tokens", 0)
                agg_reasoning_tokens += s.get("total_reasoning_tokens", 0)
                logger.info(
                    "Token stats from %s: %d in, %d out, %d reasoning",
                    stats_path,
                    s.get("total_input_tokens", 0),
                    s.get("total_output_tokens", 0),
                    s.get("total_reasoning_tokens", 0),
                )
                break
            except (FileNotFoundError, OSError):
                continue

    for input_dir in input_dirs:
        pattern = os.path.join(input_dir, "*.jsonl.gz")
        files = fsspec_glob(pattern)
        logger.info("Reading %d files from %s", len(files), input_dir)

        for file_path in files:
            with fsspec.open(file_path, "rb") as f:
                with gzip.open(f, "rt", encoding="utf-8") as gz:
                    for line in gz:
                        record = json.loads(line)
                        idx = record.get("global_row_idx")
                        if idx is None:
                            logger.warning("Record missing global_row_idx in %s, skipping", file_path)
                            continue
                        total_read += 1
                        if idx in records_by_idx:
                            duplicates += 1
                        records_by_idx[idx] = record

    logger.info(
        "Read %d records total, %d unique, %d duplicates (deduped)",
        total_read,
        len(records_by_idx),
        duplicates,
    )

    # Sort by global_row_idx
    sorted_indices = sorted(records_by_idx.keys())

    # Verify completeness
    expected_indices = set(range(expected_rows))
    present_indices = set(sorted_indices)
    missing = expected_indices - present_indices
    extra = present_indices - expected_indices

    if missing:
        logger.warning(
            "MISSING %d rows (expected 0-%d). First 20 missing: %s",
            len(missing),
            expected_rows - 1,
            sorted(missing)[:20],
        )
    if extra:
        logger.warning(
            "EXTRA %d rows beyond expected range. First 20: %s",
            len(extra),
            sorted(extra)[:20],
        )

    coverage_pct = len(present_indices & expected_indices) / max(expected_rows, 1) * 100
    logger.info("Coverage: %d / %d rows (%.1f%%)", len(present_indices & expected_indices), expected_rows, coverage_pct)

    # Write final output — drop global_row_idx, keep only FINAL_COLUMNS
    total_shards = max(1, (len(sorted_indices) + records_per_shard - 1) // records_per_shard)
    shard_idx = 0
    shard_records: list[dict] = []
    total_written = 0

    for idx in sorted_indices:
        record = records_by_idx[idx]
        # Keep only the final columns
        clean_record = {col: record[col] for col in FINAL_COLUMNS if col in record}
        shard_records.append(clean_record)

        if len(shard_records) >= records_per_shard:
            shard_path = f"{output_dir}/data-{shard_idx:05d}-of-{total_shards:05d}.jsonl.gz"
            _write_jsonl_gz(shard_path, shard_records)
            total_written += len(shard_records)
            shard_records = []
            shard_idx += 1

    # Flush remaining
    if shard_records:
        shard_path = f"{output_dir}/data-{shard_idx:05d}-of-{total_shards:05d}.jsonl.gz"
        _write_jsonl_gz(shard_path, shard_records)
        total_written += len(shard_records)
        shard_idx += 1

    # Write merge stats
    stats = {
        "input_dirs": input_dirs,
        "total_read": total_read,
        "total_unique": len(records_by_idx),
        "duplicates_deduped": duplicates,
        "total_written": total_written,
        "num_shards": shard_idx,
        "expected_rows": expected_rows,
        "coverage_pct": coverage_pct,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "total_input_tokens": agg_input_tokens,
        "total_output_tokens": agg_output_tokens,
        "total_reasoning_tokens": agg_reasoning_tokens,
    }
    with fsspec.open(f"{output_dir}/merge_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Merge complete: %d records in %d shards -> %s (%.1f%% coverage)",
        total_written,
        shard_idx,
        output_dir,
        coverage_pct,
    )


def _write_jsonl_gz(path: str, records: list[dict]) -> None:
    """Write a list of records to a gzipped JSONL file."""
    with fsspec.open(path, "wb") as f:
        with gzip.open(f, "wt", encoding="utf-8") as gz:
            for rec in records:
                gz.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Merge reprocessed dataset shards from multiple clusters.",
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="GCS paths to reassembled output directories.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to write the merged dataset.",
    )
    parser.add_argument(
        "--expected_rows",
        type=int,
        default=818_880,
        help="Total expected rows for completeness verification.",
    )
    parser.add_argument(
        "--records_per_shard",
        type=int,
        default=10_000,
        help="Number of records per output shard.",
    )
    args = parser.parse_args()

    merge_shards(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        expected_rows=args.expected_rows,
        records_per_shard=args.records_per_shard,
    )


if __name__ == "__main__":
    main()
