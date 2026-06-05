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

SOURCE_SUBDIR_ROOT = "documents/baseline_llm_extraction"
CONSOLIDATED_SUBDIR = "documents/baseline_llm_extraction_consolidated"
DEST_BUCKET = "marin-us-central1"
LEGACY_SPEC = "low_quality"

# Default location of the done-WARC list written by resolve_duplicates.py.
# Transfer reads this file and skips batches whose warc_hash isn't in it,
# avoiding wasted copies of partial in-progress extractions.
_DEFAULT_DONE_WARCS_PATH_FMT = (
    "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/" "resolved/done_warcs{suffix}.txt"
)


def _default_done_warcs_path(spec: str) -> str:
    suffix = "" if spec == LEGACY_SPEC else f"_{spec}"
    return _DEFAULT_DONE_WARCS_PATH_FMT.format(suffix=suffix)


def _source_subdir(spec: str) -> str:
    if spec == LEGACY_SPEC:
        return SOURCE_SUBDIR_ROOT
    return f"{SOURCE_SUBDIR_ROOT}/{spec}"


def _load_done_hashes(path: str) -> set[str]:
    """Read newline-delimited WARC hashes (12-char hex) from a gs:// or local path."""
    with fsspec.open(path, "r") as f:
        hashes = {line.strip() for line in f if line.strip()}
    return hashes


def _dest_subdir(region: str, spec: str) -> str:
    """Destination subdir under DEST_BUCKET. Legacy spec keeps the flat layout."""
    if spec == LEGACY_SPEC:
        return f"{CONSOLIDATED_SUBDIR}/by_region/{region}"
    return f"{CONSOLIDATED_SUBDIR}/by_region/{region}/{spec}"


def _list_source_files(fs: fsspec.AbstractFileSystem, src_bucket: str, spec: str) -> list[tuple[str, int]]:
    """List every object under source. Returns (bare_path, size) tuples."""
    prefix = f"{src_bucket}/{_source_subdir(spec)}"
    raw = fs.find(prefix, detail=True)  # dict: bare_path -> info
    # For legacy spec, filter out paths that have a spec subdir (those are NEW specs'
    # data, not low_quality's).
    items = [(p, int(info.get("size", 0))) for p, info in raw.items()]
    if spec == LEGACY_SPEC:
        import re as _re

        spec_path_re = _re.compile(rf"{_re.escape(SOURCE_SUBDIR_ROOT)}/[a-z0-9_]+/data-[0-9a-f]+/")
        items = [(p, s) for p, s in items if not spec_path_re.search(p)]
    return items


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


def transfer_region(
    region: str,
    max_workers: int = 32,
    spec: str = LEGACY_SPEC,
    done_hashes_path: str | None = None,
) -> int:
    if region not in REGION_TO_BUCKET:
        raise ValueError(f"Unknown region {region!r}")
    src_bucket = REGION_TO_BUCKET[region]
    src_prefix = f"{src_bucket}/{_source_subdir(spec)}"
    dst_prefix = f"{DEST_BUCKET}/{_dest_subdir(region, spec)}"

    if done_hashes_path is None:
        done_hashes_path = _default_done_warcs_path(spec)
    done_hashes = _load_done_hashes(done_hashes_path)
    logger.info(
        "Loaded %d done WARC hashes from %s; only those will be transferred.",
        len(done_hashes),
        done_hashes_path,
    )

    logger.info("Region=%s spec=%s src=gs://%s/ dst=gs://%s/", region, spec, src_prefix, dst_prefix)

    fs = fsspec.filesystem("gcs")

    t0 = time.monotonic()
    src_files = _list_source_files(fs, src_bucket, spec)
    logger.info("Listed %d source files in %.1fs", len(src_files), time.monotonic() - t0)

    if not src_files:
        logger.warning("No source files found under %s", src_prefix)
        return 0

    # Compute destination mapping: strip src_prefix, prepend dst_prefix.
    # Drop any file whose data-{hash} directory is not in the done set so that
    # in-progress / partial WARCs don't get transferred. The done-set filter is
    # the primary guard against shipping incomplete data downstream; the
    # idempotent skip-on-size in _copy_one only catches identity, not freshness.
    import re as _re

    hash_re = _re.compile(r"/data-([0-9a-f]+)/")
    copy_plan: list[tuple[str, str, int]] = []
    skipped_partial = 0
    skipped_no_hash = 0
    for src_bare, size in src_files:
        # src_bare looks like: "marin-us-east1/documents/baseline_llm_extraction/data-XXXX/batch_0000.jsonl.gz"
        assert (
            src_bare.startswith(src_prefix + "/") or src_bare == src_prefix
        ), f"unexpected src path {src_bare!r} (expected under {src_prefix!r})"
        m = hash_re.search(src_bare)
        if m is None:
            skipped_no_hash += 1
            continue
        warc_hash = m.group(1)
        if warc_hash not in done_hashes:
            skipped_partial += 1
            continue
        rel = src_bare[len(src_prefix) + 1 :]
        dst_bare = f"{dst_prefix}/{rel}"
        copy_plan.append((src_bare, dst_bare, size))
    logger.info(
        "After done-hash filter: %d files to copy (%d partial-WARC files dropped, %d unhashed files dropped).",
        len(copy_plan),
        skipped_partial,
        skipped_no_hash,
    )

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
    parser.add_argument(
        "--spec",
        default=LEGACY_SPEC,
        help=(
            "Extraction spec. Legacy 'low_quality' uses unprefixed source + flat destination; "
            "others read from documents/baseline_llm_extraction/{spec}/ and write to "
            "by_region/{region}/{spec}/."
        ),
    )
    parser.add_argument(
        "--done-hashes-path",
        default=None,
        help=(
            "Path (gs:// or local) to a newline-delimited list of done WARC hashes. "
            "Only batches whose data-{hash}/ matches an entry are transferred. "
            "Defaults to the standard resolver output for this spec."
        ),
    )
    args = parser.parse_args()
    rc = transfer_region(
        args.region,
        max_workers=args.max_workers,
        spec=args.spec,
        done_hashes_path=args.done_hashes_path,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
