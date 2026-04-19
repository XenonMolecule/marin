# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge per-region inventories, resolve duplicates, emit canonical manifests.

Reads ``inventory_{region}.jsonl.gz`` for each region covered by the inventory
step, groups batch rows by (warc_hash, batch_idx), and for every key with
multiple copies picks a winner via this cascade:

    1. Prefer copies that decompress cleanly AND (when a ``.count`` sidecar
       exists) whose row count matches it.
    2. Among valid copies, pick the one with the most text characters
       (``sum_text_chars``). This rewards extractions that kept more records
       AND produced longer text per record.
    3. Tiebreak on higher ``num_records``, then on newer ``mtime``.
    4. If no copy validates, pick the best-ranked anyway (no key is ever left
       without a canonical pointer). All such cases are listed in the integrity
       report.

This script never deletes, moves, or overwrites batch data. It only writes
three small derived manifests:

    {out_prefix}/resolved/resolved.jsonl.gz      one row per (hash, batch_idx)
    {out_prefix}/resolved/duplicates.jsonl.gz    every observed collision
    {out_prefix}/resolved/integrity_report.json  summary counts + flagged cases
    {out_prefix}/resolved/missing_batches.jsonl.gz   hashes with gaps / no coverage

Intended to run in us-central1 — reads 5 small manifest files (~3 MB each
gzipped). Safe to run from a laptop or a tiny Iris CPU job in us-central1.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import defaultdict
from typing import Any

import fsspec

logger = logging.getLogger(__name__)

DEFAULT_REGIONS: tuple[str, ...] = ("us-central1", "us-east1", "us-east5", "us-west4", "europe-west4")
CONSOLIDATED_SUBDIR = "documents/baseline_llm_extraction_consolidated"
DEST_BUCKET = "marin-us-central1"
COMPLETED_REGISTRY_PREFIX = "marin-us-central1/documents/baseline_llm_extraction/_completed"


def _load_inventory(region: str, inventories_prefix: str) -> list[dict]:
    path = f"{inventories_prefix}/inventory_{region}.jsonl.gz"
    rows: list[dict] = []
    with fsspec.open(path, "rb") as f, gzip.open(f, "rt") as gz:
        for line in gz:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_completed_registry() -> set[str]:
    """Hashes with a central-registry marker written by ``_register_completed_warc``."""
    fs = fsspec.filesystem("gcs")
    try:
        paths = fs.ls(COMPLETED_REGISTRY_PREFIX)
    except FileNotFoundError:
        return set()
    hashes: set[str] = set()
    for p in paths:
        name = p.rsplit("/", 1)[-1]
        if name.startswith("data-"):
            hashes.add(name[len("data-") :])
    return hashes


def _count_ok(row: dict) -> bool:
    """True iff either no .count sidecar or sidecar matches decompressed row count."""
    cnt = row.get("count_sidecar")
    n = row.get("num_records")
    if cnt is None or n is None:
        return True  # can't contradict
    return cnt == n


def _is_valid(row: dict) -> bool:
    return bool(row.get("decompress_ok")) and _count_ok(row)


def _rank(row: dict) -> tuple:
    """Higher tuple = better candidate in the tiebreaker cascade."""
    return (
        1 if _is_valid(row) else 0,
        row.get("sum_text_chars") or 0,
        row.get("num_records") or 0,
        row.get("mtime") or "",
    )


def _candidate_summary(row: dict) -> dict:
    return {
        "region": row["region"],
        "path": row["path"],
        "num_records": row.get("num_records"),
        "sum_text_chars": row.get("sum_text_chars"),
        "size_bytes": row.get("size_bytes"),
        "mtime": row.get("mtime"),
        "has_count_sidecar": row.get("has_count_sidecar"),
        "count_sidecar": row.get("count_sidecar"),
        "has_tokens_sidecar": row.get("has_tokens_sidecar"),
        "decompress_ok": row.get("decompress_ok"),
        "count_matches": _count_ok(row),
        "is_valid": _is_valid(row),
        "error": row.get("error"),
    }


def _write_jsonl_gz(path: str, rows: list[dict]) -> None:
    with fsspec.open(path, "wb") as f, gzip.open(f, "wt", encoding="utf-8") as gz:
        for r in rows:
            gz.write(json.dumps(r, sort_keys=True) + "\n")


def _write_json(path: str, obj: Any) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def resolve(regions: list[str], out_prefix: str) -> dict:
    inventories_prefix = f"{out_prefix}/inventories"

    all_batch_rows: list[dict] = []
    all_warc_rows: list[dict] = []
    for region in regions:
        rows = _load_inventory(region, inventories_prefix)
        br = [r for r in rows if r.get("record_type") == "batch"]
        wr = [r for r in rows if r.get("record_type") == "warc"]
        logger.info("  region=%s batches=%d warcs=%d", region, len(br), len(wr))
        all_batch_rows.extend(br)
        all_warc_rows.extend(wr)

    logger.info("Merged: %d batch rows across %d regions", len(all_batch_rows), len(regions))

    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in all_batch_rows:
        by_key[(r["warc_hash"], r["batch_idx"])].append(r)

    resolved: list[dict] = []
    duplicates: list[dict] = []
    invalid_all: list[dict] = []

    for (h, idx), cands in by_key.items():
        ranked = sorted(cands, key=_rank, reverse=True)
        winner = ranked[0]
        entry = {
            "warc_hash": h,
            "batch_idx": idx,
            "path": winner["path"],
            "region": winner["region"],
            "num_records": winner.get("num_records"),
            "sum_text_chars": winner.get("sum_text_chars"),
            "is_valid": _is_valid(winner),
            "num_copies": len(cands),
        }
        resolved.append(entry)
        if len(cands) > 1:
            duplicates.append(
                {
                    "warc_hash": h,
                    "batch_idx": idx,
                    "chosen_region": winner["region"],
                    "chosen_path": winner["path"],
                    "num_copies": len(cands),
                    "candidates": [_candidate_summary(c) for c in ranked],
                    "char_delta_vs_runner_up": (
                        (winner.get("sum_text_chars") or 0) - (ranked[1].get("sum_text_chars") or 0)
                    ),
                    "records_delta_vs_runner_up": (winner.get("num_records") or 0) - (ranked[1].get("num_records") or 0),
                }
            )
        if not any(_is_valid(c) for c in cands):
            invalid_all.append(
                {
                    "warc_hash": h,
                    "batch_idx": idx,
                    "num_copies": len(cands),
                    "errors_by_region": {
                        c["region"]: c.get("error") or ("count_mismatch" if not _count_ok(c) else "decompress_not_ok")
                        for c in cands
                    },
                }
            )

    # Cross-check against the central completed-WARC registry.
    completed = _load_completed_registry()
    resolved_hashes = {r["warc_hash"] for r in resolved}
    warcs_with_done_anywhere = {r["warc_hash"] for r in all_warc_rows if r.get("has_done")}
    registry_missing_from_resolved = sorted(completed - resolved_hashes)
    registry_without_done_in_any_region = sorted(completed - warcs_with_done_anywhere)

    # Per-WARC gap check: for each hash we see, are batch indices contiguous from 0?
    # Gaps suggest lost batches (possible if both owner and stealer died before flush).
    by_hash_idxs: dict[str, set[int]] = defaultdict(set)
    for r in resolved:
        by_hash_idxs[r["warc_hash"]].add(r["batch_idx"])
    missing_batches: list[dict] = []
    for h, idxs in by_hash_idxs.items():
        max_idx = max(idxs)
        expected = set(range(max_idx + 1))
        gap = sorted(expected - idxs)
        if gap:
            missing_batches.append(
                {
                    "warc_hash": h,
                    "max_observed_idx": max_idx,
                    "missing_indices": gap,
                    "observed_count": len(idxs),
                }
            )

    # Write outputs.
    resolved_path = f"{out_prefix}/resolved/resolved.jsonl.gz"
    duplicates_path = f"{out_prefix}/resolved/duplicates.jsonl.gz"
    missing_path = f"{out_prefix}/resolved/missing_batches.jsonl.gz"
    report_path = f"{out_prefix}/resolved/integrity_report.json"

    _write_jsonl_gz(resolved_path, resolved)
    _write_jsonl_gz(duplicates_path, duplicates)
    _write_jsonl_gz(missing_path, missing_batches)

    report = {
        "regions": regions,
        "total_batch_copies_observed": len(all_batch_rows),
        "unique_keys_resolved": len(resolved),
        "duplicate_keys": len(duplicates),
        "invalid_on_all_copies": len(invalid_all),
        "warcs_with_done_in_any_region": len(warcs_with_done_anywhere),
        "completed_registry_size": len(completed),
        "completed_missing_from_resolved": len(registry_missing_from_resolved),
        "completed_without_done_anywhere": len(registry_without_done_in_any_region),
        "warcs_with_index_gaps": len(missing_batches),
        "sample_missing_from_resolved": registry_missing_from_resolved[:20],
        "sample_warcs_with_gaps": missing_batches[:20],
        "sample_invalid_all": invalid_all[:20],
        "outputs": {
            "resolved": resolved_path,
            "duplicates": duplicates_path,
            "missing_batches": missing_path,
            "integrity_report": report_path,
        },
    }
    _write_json(report_path, report)
    logger.info("Resolver summary:\n%s", json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        help="Regions with inventory files to merge.",
    )
    parser.add_argument(
        "--out-prefix",
        default=f"gs://{DEST_BUCKET}/{CONSOLIDATED_SUBDIR}",
        help="Root prefix for inventories/ and resolved/ (gs:// URI).",
    )
    args = parser.parse_args()
    resolve(args.regions, args.out_prefix)


if __name__ == "__main__":
    main()
