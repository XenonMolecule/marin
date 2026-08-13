# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Consolidate every region's ``kept_text/`` into the TARGET region's ``kept_text/``.

A multi-region extraction leaves each region holding only the WARCs its own workers claimed.
``dedup.py`` needs the whole pool in one place (fuzzy dedup is global), so the per-region trees must
be merged into the training region before dedup runs. This is the ONLY cross-region step in the
downstream, so it is also the only one that costs egress — run ``project_text_only.py`` in each
source region FIRST so the bytes moved are text-only (the ragged ``input_ids`` are ~2/3 of the kept
parquet and are useless downstream).

Shards are named ``data-{warc_hash}.parquet``, so a WARC processed in two regions (the claim
registry is per-region, so this happens) appears under the same basename in both. Copying by
basename therefore collapses those duplicates to one copy — the dedup input is one shard per
distinct WARC.

Copies go through the GCS rewrite API (``fs.copy`` on a single filesystem), so bytes move
server-side; nothing is pulled through this process. Idempotent: an already-present basename is
skipped, so a preempted run resumes for free.

    python -m experiments.fast_curation.consolidate_kept_text --spec lpv11_fastpipe_v1 \\
        --target-region us-east5 --source-regions us-central1 us-west4 us-east1 us-central2
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

import fsspec

from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)

REGION_TO_BUCKET: dict[str, str] = {
    "us-east5": "gs://marin-us-east5",
    "us-central1": "gs://marin-us-central1",
    "us-central2": "gs://marin-us-central2",
    "us-east1": "gs://marin-us-east1",
    "us-west4": "gs://marin-us-west4",
    "eu-west4": "gs://marin-eu-west4",
}


def _kept_text_shards(fs: fsspec.AbstractFileSystem, spec_id: str, region: str) -> dict[str, str]:
    """``{basename: gs://... path}`` for one region's kept_text tree (empty if the tree is absent)."""
    prefix = get_spec(spec_id).namespace(REGION_TO_BUCKET[region]) + "/kept_text"
    if not fs.exists(prefix.removeprefix("gs://")):
        return {}
    paths = [f"gs://{p}" for p in fs.ls(prefix.removeprefix("gs://")) if p.endswith(".parquet")]
    return {p.rsplit("/", 1)[1]: p for p in paths}


def consolidate(spec_id: str, target_region: str, source_regions: list[str], num_threads: int, dry_run: bool) -> dict:
    fs = fsspec.filesystem("gcs")
    target_prefix = get_spec(spec_id).namespace(REGION_TO_BUCKET[target_region]) + "/kept_text"
    present = _kept_text_shards(fs, spec_id, target_region)
    logger.info("target %s already holds %d shards", target_region, len(present))

    pending: dict[str, str] = {}  # basename -> source path (first region wins; duplicates collapse)
    duplicates = 0
    for region in source_regions:
        shards = _kept_text_shards(fs, spec_id, region)
        if not shards:
            raise FileNotFoundError(f"no kept_text tree in {region}; run project_text_only there first")
        new = 0
        for basename, path in shards.items():
            if basename in present:
                duplicates += 1
                continue
            if basename in pending:
                duplicates += 1
                continue
            pending[basename] = path
            new += 1
        logger.info("%-12s %5d shards, %5d new to copy", region, len(shards), new)

    logger.info(
        "copying %d shards -> %s (%d cross-region duplicate WARCs collapsed)",
        len(pending),
        target_prefix,
        duplicates,
    )
    if not pending:
        return {"copied": 0, "duplicates": duplicates, "total": len(present)}

    total_bytes = sum(fs.info(p)["size"] for p in pending.values())
    logger.info("bytes to move: %.1f GB", total_bytes / 1e9)
    if dry_run:
        logger.info("dry run: nothing copied")
        return {"copied": 0, "duplicates": duplicates, "bytes_to_move": total_bytes, "total": len(present)}

    def _copy_one(item: tuple[str, str]) -> None:
        basename, src = item
        fs.copy(src, f"{target_prefix}/{basename}")

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        for i, _ in enumerate(pool.map(_copy_one, pending.items()), start=1):
            if i % 200 == 0:
                logger.info("  copied %d/%d", i, len(pending))

    final = _kept_text_shards(fs, spec_id, target_region)
    logger.info("consolidated tree: %d shards under %s", len(final), target_prefix)
    return {
        "copied": len(pending),
        "duplicates": duplicates,
        "bytes_moved": total_bytes,
        "total": len(final),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--target-region", required=True, choices=sorted(REGION_TO_BUCKET))
    ap.add_argument("--source-regions", nargs="+", required=True, choices=sorted(REGION_TO_BUCKET))
    ap.add_argument("--num-threads", type=int, default=32)
    ap.add_argument("--expect-shards", type=int, default=None, help="Assert the final shard count (the WARC pool size).")
    ap.add_argument("--dry-run", action="store_true", help="Report the exact bytes that would move, copy nothing.")
    args = ap.parse_args()

    if args.target_region in args.source_regions:
        raise ValueError(f"target region {args.target_region} must not also be a source region")

    result = consolidate(args.spec, args.target_region, args.source_regions, args.num_threads, args.dry_run)
    logger.info("consolidation complete: %s", result)
    if args.expect_shards is not None and not args.dry_run and result["total"] != args.expect_shards:
        raise AssertionError(f"expected {args.expect_shards} shards after consolidation, got {result['total']}")


if __name__ == "__main__":
    main()
