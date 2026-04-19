# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy one region's extraction output into the us-central1 consolidated archive.

Runs inside a region-pinned Iris CPU job launched by ``launch_transfer.py``.
Copies ``gs://marin-{region_bucket}/documents/baseline_llm_extraction/`` into
``gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/{region}/``
via GCS server-side copy (the gcsfs ``copy`` method maps to the GCS
``rewriteObject`` API — bytes move bucket-to-bucket, not through the VM).

Idempotent: per-file ``skip_existing=True`` check via ``fs.exists`` before each
copy, so re-running after a straggler sweep is cheap.

NEVER deletes anything. The source bucket is strictly read-only from this
script's perspective.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

logger = logging.getLogger(__name__)

REGION_TO_BUCKET: dict[str, str] = {
    "us-central1": "marin-us-central1",
    "us-east1": "marin-us-east1",
    "us-east5": "marin-us-east5",
    "us-west4": "marin-us-west4",
    "europe-west4": "marin-eu-west4",
    "eu-west4": "marin-eu-west4",
}

SOURCE_SUBDIR = "documents/baseline_llm_extraction"
CONSOLIDATED_SUBDIR = "documents/baseline_llm_extraction_consolidated"
DEST_BUCKET = "marin-us-central1"


def _list_source_files(fs: fsspec.AbstractFileSystem, src_bucket: str) -> list[tuple[str, int]]:
    """List every object under source. Returns (bare_path, size) tuples."""
    prefix = f"{src_bucket}/{SOURCE_SUBDIR}"
    raw = fs.find(prefix, detail=True)  # dict: bare_path -> info
    return [(p, int(info.get("size", 0))) for p, info in raw.items()]


def _copy_one(
    fs: fsspec.AbstractFileSystem,
    src_bare: str,
    dst_bare: str,
    src_size: int,
) -> tuple[str, str, bool, str | None]:
    """Server-side copy src→dst. Skip if dst already exists with matching size.

    Returns (src, dst, copied, error_or_note).
    """
    try:
        # Check idempotency: skip if destination already exists with matching size.
        try:
            info = fs.info(dst_bare)
            if int(info.get("size", -1)) == src_size:
                return (src_bare, dst_bare, False, "skipped_exists")
        except (FileNotFoundError, Exception):
            pass  # destination missing; proceed
        fs.copy(src_bare, dst_bare)
        return (src_bare, dst_bare, True, None)
    except Exception as e:
        return (src_bare, dst_bare, False, f"{type(e).__name__}: {e}")


def transfer_region(region: str, max_workers: int = 32) -> int:
    if region not in REGION_TO_BUCKET:
        raise ValueError(f"Unknown region {region!r}")
    src_bucket = REGION_TO_BUCKET[region]
    src_prefix = f"{src_bucket}/{SOURCE_SUBDIR}"
    dst_prefix = f"{DEST_BUCKET}/{CONSOLIDATED_SUBDIR}/by_region/{region}"

    logger.info("Region=%s src=gs://%s/ dst=gs://%s/", region, src_prefix, dst_prefix)

    fs = fsspec.filesystem("gcs")

    t0 = time.monotonic()
    src_files = _list_source_files(fs, src_bucket)
    logger.info("Listed %d source files in %.1fs", len(src_files), time.monotonic() - t0)

    if not src_files:
        logger.warning("No source files found under %s", src_prefix)
        return 0

    # Compute destination mapping: strip src_prefix, prepend dst_prefix.
    copy_plan: list[tuple[str, str, int]] = []
    for src_bare, size in src_files:
        # src_bare looks like: "marin-us-east1/documents/baseline_llm_extraction/data-XXXX/batch_0000.jsonl.gz"
        assert (
            src_bare.startswith(src_prefix + "/") or src_bare == src_prefix
        ), f"unexpected src path {src_bare!r} (expected under {src_prefix!r})"
        rel = src_bare[len(src_prefix) + 1 :]
        dst_bare = f"{dst_prefix}/{rel}"
        copy_plan.append((src_bare, dst_bare, size))

    t0 = time.monotonic()
    copied = 0
    skipped = 0
    errors = 0
    first_errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_copy_one, fs, src, dst, size) for src, dst, size in copy_plan]
        for i, fut in enumerate(as_completed(futures), start=1):
            _src, _dst, did_copy, note = fut.result()
            if note == "skipped_exists":
                skipped += 1
            elif did_copy:
                copied += 1
            else:
                errors += 1
                if len(first_errors) < 10:
                    first_errors.append(f"{_src} -> {_dst}: {note}")
            if i % 2000 == 0:
                elapsed = time.monotonic() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(futures) - i) / max(rate, 1e-6)
                logger.info(
                    "Progress %d/%d (copied=%d skipped=%d errors=%d) %.0f/s ETA %.0fs",
                    i,
                    len(futures),
                    copied,
                    skipped,
                    errors,
                    rate,
                    eta,
                )

    elapsed = time.monotonic() - t0
    logger.info(
        "DONE region=%s: %d/%d files (copied=%d, skipped_exists=%d, errors=%d) in %.1fs",
        region,
        copied + skipped,
        len(copy_plan),
        copied,
        skipped,
        errors,
        elapsed,
    )
    if errors:
        logger.error("First errors (up to 10):")
        for e in first_errors:
            logger.error("  %s", e)
        return 1
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, choices=sorted(REGION_TO_BUCKET.keys()))
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()
    rc = transfer_region(args.region, max_workers=args.max_workers)
    sys.exit(rc)


if __name__ == "__main__":
    main()
