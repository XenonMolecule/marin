# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replicate a GCS prefix to another region and prove the copy is byte-identical.

Copies are server-side rewrites, so no object content passes through this process
— it only issues calls, and a small CPU box saturates the work.

Verification compares **CRC32C**, the checksum GCS computes over every object's
bytes and stores in its metadata. Comparing those two values is a genuine
byte-for-byte check that never downloads either side, so it costs no egress and
can run over the whole tree rather than a sample. Sizes are not enough: a
truncated-then-repadded rewrite, or an object copied from the wrong source, can
match on length.

A prior version of this check was written as a shell loop and reported every
object as mismatched — zsh does not word-split unquoted parameter expansions, so
the comparison was between a filename and an empty string. Keeping it in Python
is not a style preference.

    python -m experiments.baseline_collection.replicate_prefix \\
        --src gs://marin-us-central1/datakit/store/dclm_10k_gridv1 \\
        --dst gs://marin-us-east5/datakit/store/dclm_10k_gridv1
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import fsspec

logger = logging.getLogger(__name__)

# GCS rewrite is latency-bound from the caller's side, not bandwidth-bound.
COPY_PARALLELISM = 32
LIST_PAGE_LOG_EVERY = 20_000


@dataclass(frozen=True)
class ObjectInfo:
    """One object's identity as GCS reports it."""

    size: int
    crc32c: str


def _listing(fs, root: str) -> dict[str, ObjectInfo]:
    """Every object under ``root``, keyed by path relative to it.

    A bulk listing usually carries ``crc32c`` already; when it doesn't, the
    checksum is fetched per object rather than silently degrading the comparison
    to sizes, which is the check this module exists to avoid.
    """
    root = root.rstrip("/")
    # fsspec's find() returns keys stripped of the scheme (e.g. "gs://"), so the
    # offset for computing a relative key must be taken from the SAME
    # scheme-less form, not from `root` itself. Slicing by len(root) silently
    # truncated every key by len("gs://") == 5 characters.
    stripped_root = fs._strip_protocol(root)
    files = {path: info for path, info in fs.find(root, detail=True).items() if info.get("type") == "file"}
    needs_crc = [path for path, info in files.items() if not info.get("crc32c")]
    if needs_crc:
        logger.info("%s: fetching crc32c for %d objects the listing omitted", root, len(needs_crc))
        with ThreadPoolExecutor(max_workers=COPY_PARALLELISM) as pool:
            for path, info in zip(needs_crc, pool.map(fs.info, needs_crc), strict=True):
                files[path] = info

    listing = {}
    for path, info in files.items():
        crc = info.get("crc32c")
        if not crc:
            raise RuntimeError(f"{path}: no crc32c in object metadata; cannot verify byte-for-byte")
        listing[path[len(stripped_root) + 1 :]] = ObjectInfo(size=int(info["size"]), crc32c=crc)
    logger.info("%s: %d objects, %.2f GB", root, len(listing), sum(o.size for o in listing.values()) / 1e9)
    return listing


def _copy(fs, src_root: str, dst_root: str, keys: list[str]) -> None:
    """Server-side copy of each key, in parallel."""

    def one(key: str) -> None:
        fs.cp_file(f"{src_root}/{key}", f"{dst_root}/{key}")

    with ThreadPoolExecutor(max_workers=COPY_PARALLELISM) as pool:
        for done, _ in enumerate(pool.map(one, keys), start=1):
            if done % 500 == 0:
                logger.info("copied %d/%d", done, len(keys))


def replicate(src: str, dst: str, verify_only: bool) -> None:
    """Copy ``src`` to ``dst`` if needed, then verify every object's CRC32C.

    Raises:
        RuntimeError: If any object is missing, extra, or differs after copying.
            A partial replica is worse than none, so this never returns quietly.
    """
    src_root, dst_root = src.rstrip("/"), dst.rstrip("/")
    fs = fsspec.core.url_to_fs(src_root)[0]

    source = _listing(fs, src_root)
    if not source:
        raise FileNotFoundError(f"no objects under {src_root}")
    target = _listing(fs, dst_root) if fs.exists(dst_root) else {}

    todo = [k for k, info in source.items() if target.get(k) != info]
    if todo and verify_only:
        raise RuntimeError(f"{len(todo)} objects differ or are missing at {dst_root} (verify-only)")
    if todo:
        logger.info("copying %d objects (%d already match)", len(todo), len(source) - len(todo))
        _copy(fs, src_root, dst_root, todo)
        target = _listing(fs, dst_root)
    else:
        logger.info("destination already complete; verifying only")

    missing = sorted(set(source) - set(target))
    extra = sorted(set(target) - set(source))
    differing = sorted(k for k in source if k in target and source[k] != target[k])
    if missing or extra or differing:
        raise RuntimeError(
            f"replica mismatch: {len(missing)} missing, {len(extra)} extra, "
            f"{len(differing)} differing. First few: "
            f"missing={missing[:3]!r} extra={extra[:3]!r} differing={differing[:3]!r}"
        )
    logger.info(
        "VERIFIED byte-for-byte: %d objects, %.2f GB, every CRC32C matches -> %s",
        len(source),
        sum(o.size for o in source.values()) / 1e9,
        dst_root,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check an existing replica without copying anything. Use to re-confirm after a run.",
    )
    args = parser.parse_args()
    replicate(args.src, args.dst, args.verify_only)


if __name__ == "__main__":
    main()
