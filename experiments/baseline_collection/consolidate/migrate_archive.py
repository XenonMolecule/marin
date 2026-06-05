# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror the consolidated archive of one spec from us-central1 → another region.

Used when the destination region has cluster capacity (e.g. 64 GB worker
slots) that the source region lacks. Reads the consolidated archive under
``gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/{region}/{spec}/``
and writes the mirror under ``gs://marin-{dst-region}/...`` via GCS
server-side rewrite (no bytes traverse the worker VM, only API calls).

Also mirrors the per-spec resolved manifests so the dedup pipeline can
read everything intra-region.

Usage (run as an Iris CPU job pinned to the DESTINATION region for lower
API latency)::

    iris job run --priority interactive --no-wait \\
        --cpu 4 --memory 8GB --disk 5GB --extra cpu \\
        --enable-extra-resources --region us-east5 \\
        --job-name migrate-archive-high_quality \\
        -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python experiments/baseline_collection/consolidate/migrate_archive.py \\
           --spec high_quality --dst-region us-east5
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

ARCHIVE_REGIONS: tuple[str, ...] = ("us-central1", "us-east1", "us-east5", "us-west4", "europe-west4")
CONSOLIDATED_SUBDIR = "documents/baseline_llm_extraction_consolidated"
LEGACY_SPEC = "low_quality"


def _archive_subpath(region: str, spec: str) -> str:
    base = f"{CONSOLIDATED_SUBDIR}/by_region/{region}"
    return base if spec == LEGACY_SPEC else f"{base}/{spec}"


def _resolved_files(spec: str) -> list[str]:
    """Per-spec resolved/* filenames produced by resolve_duplicates.py."""
    if spec == LEGACY_SPEC:
        return [
            "resolved.jsonl.gz",
            "duplicates.jsonl.gz",
            "missing_batches.jsonl.gz",
            "integrity_report.json",
            "done_warcs.txt",
        ]
    return [
        f"resolved_{spec}.jsonl.gz",
        f"duplicates_{spec}.jsonl.gz",
        f"missing_batches_{spec}.jsonl.gz",
        f"integrity_report_{spec}.json",
        f"done_warcs_{spec}.txt",
    ]


def _copy_one(
    fs: fsspec.AbstractFileSystem,
    src_bare: str,
    dst_bare: str,
    src_size: int,
) -> tuple[str, bool, str | None]:
    """Server-side copy src→dst. Skip if dst already exists with matching size."""
    try:
        try:
            info = fs.info(dst_bare)
            if int(info.get("size", -1)) == src_size:
                return (dst_bare, False, "skipped_exists")
        except (FileNotFoundError, Exception):
            pass
        fs.copy(src_bare, dst_bare)
        return (dst_bare, True, None)
    except Exception as e:
        return (dst_bare, False, f"{type(e).__name__}: {e}")


def migrate_region(
    src_region: str,
    dst_region: str,
    spec: str,
    max_workers: int = 128,
) -> int:
    # The consolidated archive lives entirely under marin-us-central1 (the
    # destination of the original transfer step). by_region/<src_region>/ is
    # the per-source-region subtree within that single bucket — NOT a copy
    # in marin-<src_region>. Reading from marin-<src_region> would list 0
    # files and silently skip every non-central1 subtree.
    SRC_ARCHIVE_BUCKET = "marin-us-central1"
    dst_bucket = REGION_TO_BUCKET[dst_region]
    src_prefix = f"{SRC_ARCHIVE_BUCKET}/{_archive_subpath(src_region, spec)}"
    dst_prefix = f"{dst_bucket}/{_archive_subpath(src_region, spec)}"

    logger.info("Archive mirror %s: src=gs://%s/ dst=gs://%s/", src_region, src_prefix, dst_prefix)

    fs = fsspec.filesystem("gcs")

    t0 = time.monotonic()
    raw = fs.find(src_prefix, detail=True)
    src_files = [(p, int(info.get("size", 0))) for p, info in raw.items()]
    logger.info("  listed %d files in %.1fs", len(src_files), time.monotonic() - t0)

    if not src_files:
        logger.warning("  no files under %s", src_prefix)
        return 0

    copy_plan = []
    for src_bare, size in src_files:
        assert src_bare.startswith(src_prefix + "/"), f"unexpected {src_bare!r}"
        rel = src_bare[len(src_prefix) + 1 :]
        copy_plan.append((src_bare, f"{dst_prefix}/{rel}", size))

    t0 = time.monotonic()
    copied = 0
    skipped = 0
    errors = 0
    first_errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_copy_one, fs, src, dst, sz) for src, dst, sz in copy_plan]
        for i, fut in enumerate(as_completed(futs), start=1):
            _dst, did_copy, note = fut.result()
            if note == "skipped_exists":
                skipped += 1
            elif did_copy:
                copied += 1
            else:
                errors += 1
                if len(first_errors) < 10:
                    first_errors.append(f"{_dst}: {note}")
            if i % 5000 == 0:
                elapsed = time.monotonic() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(futs) - i) / max(rate, 1e-6)
                logger.info(
                    "  progress %d/%d (copied=%d skipped=%d errors=%d) %.0f/s ETA %.0fs",
                    i,
                    len(futs),
                    copied,
                    skipped,
                    errors,
                    rate,
                    eta,
                )

    elapsed = time.monotonic() - t0
    logger.info(
        "DONE archive %s→%s: %d/%d (copied=%d, skipped_exists=%d, errors=%d) in %.1fs",
        src_region,
        dst_region,
        copied + skipped,
        len(copy_plan),
        copied,
        skipped,
        errors,
        elapsed,
    )
    if errors:
        logger.error("First errors:")
        for e in first_errors:
            logger.error("  %s", e)
        return 1
    return 0


def migrate_resolved(spec: str, dst_region: str) -> None:
    """Mirror the small per-spec resolved/* files to the destination region."""
    src_bucket = "marin-us-central1"
    dst_bucket = REGION_TO_BUCKET[dst_region]
    src_prefix = f"{src_bucket}/{CONSOLIDATED_SUBDIR}/resolved"
    dst_prefix = f"{dst_bucket}/{CONSOLIDATED_SUBDIR}/resolved"

    fs = fsspec.filesystem("gcs")
    for fname in _resolved_files(spec):
        src = f"{src_prefix}/{fname}"
        dst = f"{dst_prefix}/{fname}"
        try:
            info = fs.info(src)
        except FileNotFoundError:
            logger.warning("resolved: missing %s — skipping", src)
            continue
        try:
            dinfo = fs.info(dst)
            if int(dinfo.get("size", -1)) == int(info.get("size", -1)):
                logger.info("resolved: %s already mirrored (%d bytes)", fname, info["size"])
                continue
        except (FileNotFoundError, Exception):
            pass
        fs.copy(src, dst)
        logger.info("resolved: copied %s (%d bytes)", fname, info["size"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dst-region", required=True, choices=sorted(REGION_TO_BUCKET))
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument(
        "--regions", nargs="+", default=list(ARCHIVE_REGIONS), help="Which per-region subtrees of by_region/ to mirror."
    )
    parser.add_argument(
        "--skip-resolved", action="store_true", help="Don't mirror the resolved/* manifests (they're tiny but useful)."
    )
    args = parser.parse_args()

    # Mirror resolved manifests first so dedup can read them if it races ahead.
    if not args.skip_resolved:
        migrate_resolved(args.spec, args.dst_region)

    rc = 0
    for r in args.regions:
        if migrate_region(r, args.dst_region, args.spec, args.max_workers) != 0:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
