# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate every region's ``kept/`` into the TARGET region's fused ``kept/``.

Moves the source-of-truth kept parquet (one file per WARC, keyed ``data-{warc_hash}.parquet``)
into a single region so every later derivative — ``kept_text`` projection, quality-band
re-thresholds, dedup, tokenization — runs in-region with zero egress. Source files from the
two-phase v2.1 namespace (rung 1, shards 0-403) land in the SAME fused ``kept/`` tree: the
schemas are identical by the fused output contract, and one directory then holds the whole
2M-WARC corpus.

Server-side GCS rewrites (no client bandwidth), skip-existing so reruns resume, a per-file
size + crc32c check against its source, and a final basename check of the target listing.
Copies only. ``--verify-sources`` then checks each source region file by file against the target
(read-only); adding ``--delete-verified`` deletes exactly the source files whose target copy is
identical, leaving any missing or differing file in place.

    python -m experiments.fast_curation.consolidate_kept --target-region us-central2 \\
        --source-regions eu-west4 --limit 100   # then --dry-run / full runs, one region at a time
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import fsspec
from google.cloud import storage

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

SPEC_IDS = ("lpv11_fastpipe_v2_1_fused", "lpv11_fastpipe_v2_1")
COPY_BATCH = 2_000
DELETE_BATCH = 1_000


@dataclasses.dataclass(frozen=True, slots=True)
class KeptFile:
    name: str
    size: int
    crc32c: str


def _gcs() -> fsspec.AbstractFileSystem:
    return fsspec.filesystem("gcs", use_listings_cache=False, skip_instance_cache=True)


def _kept_prefix(spec_id: str, region: str) -> str:
    return get_spec(spec_id).namespace(REGION_TO_BUCKET[region]).removeprefix("gs://") + "/kept"


def _kept_shards(spec_id: str, region: str) -> dict[str, KeptFile]:
    """``{basename: KeptFile}`` for one region's kept tree (empty if absent).

    Streams the listing page by page with only name, size and crc32c: ``fs.ls`` materializes every
    full metadata dict at once and OOM-killed 16GB jobs once the target held 4.6M files.
    """
    bucket, prefix = _kept_prefix(spec_id, region).split("/", 1)
    blobs = storage.Client().list_blobs(
        bucket, prefix=prefix + "/", fields="items(name,size,crc32c),nextPageToken", page_size=5000
    )
    return {
        b.name.rsplit("/", 1)[1]: KeptFile(f"{bucket}/{b.name}", int(b.size), b.crc32c)
        for b in blobs
        if b.name.endswith(".parquet")
    }


def in_partition(basename: str, partition: tuple[int, int]) -> bool:
    """Stable split of file names into ``n`` disjoint parts, so parallel copy jobs never overlap."""
    index, count = partition
    return zlib.crc32(basename.encode()) % count == index


def consolidate(
    target_region: str,
    source_regions: list[str],
    num_threads: int,
    limit: int | None,
    dry_run: bool,
    partition: tuple[int, int],
) -> None:
    fs = _gcs()
    target_prefix = _kept_prefix("lpv11_fastpipe_v2_1_fused", target_region)
    present = set(_kept_shards("lpv11_fastpipe_v2_1_fused", target_region))
    logger.info("target %s already holds %d kept files", target_region, len(present))

    sources: dict[str, KeptFile] = {}  # basename -> source file (first wins; duplicates are benign copies)
    for spec_id in SPEC_IDS:
        # The target's own v2.1 (rung-1) tree is also a source: it merges into the fused kept/.
        for region in [*source_regions, *([target_region] if spec_id != "lpv11_fastpipe_v2_1_fused" else [])]:
            shards = _kept_shards(spec_id, region)
            new = {
                b: k for b, k in shards.items() if b not in present and b not in sources and in_partition(b, partition)
            }
            logger.info("%s %s: %d kept files, %d to copy", spec_id, region, len(shards), len(new))
            sources.update(new)

    todo = sorted(sources.items())[:limit]
    logger.info("partition %d/%d", *partition)
    logger.info("%d files, %.3f TB -> %s", len(todo), sum(k.size for _, k in todo) / 1e12, target_prefix)
    if dry_run:
        logger.info("dry run: nothing copied")
        return

    def copy_and_verify(item: tuple[str, KeptFile]) -> None:
        basename, a = item
        dst = f"{target_prefix}/{basename}"
        fs.copy(a.name, dst)
        b = fs.info(dst)
        if (a.size, a.crc32c) != (int(b["size"]), b["crc32c"]):
            raise RuntimeError(
                f"{dst} differs from {a.name}: size {b['size']} vs {a.size}, crc32c {b['crc32c']} vs {a.crc32c}"
            )

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        for start in range(0, len(todo), COPY_BATCH):
            batch = todo[start : start + COPY_BATCH]
            list(pool.map(copy_and_verify, batch))
            logger.info("copied and verified %d/%d", min(start + COPY_BATCH, len(todo)), len(todo))

    final = set(_kept_shards("lpv11_fastpipe_v2_1_fused", target_region))
    missing = {b for b, _ in todo} - final
    if missing:
        raise RuntimeError(f"{len(missing)} files missing after copy; sample {sorted(missing)[:5]}")
    logger.info("VERIFIED: %d copied files match their sources (size + crc32c); target holds %d", len(todo), len(final))


def verify_source(target_region: str, source_region: str) -> tuple[list[KeptFile], list[str]]:
    """Split a source region's kept files into those with an identical target copy and the rest.

    Compares fresh listings of both trees: a source file verifies when the target holds a file with
    the same name, size and crc32c. Returns (verified files, unverified paths); reads only.
    """
    target = _kept_shards("lpv11_fastpipe_v2_1_fused", target_region)
    sources = [k for spec_id in SPEC_IDS for k in _kept_shards(spec_id, source_region).values()]
    verified, bad = [], []
    for k in sources:
        t = target.get(k.name.rsplit("/", 1)[1])
        if t is not None and (t.size, t.crc32c) == (k.size, k.crc32c):
            verified.append(k)
        else:
            bad.append(k.name)
    logger.info(
        "%s: %d kept files, %.3f TB; %d verified, %d missing or different in %s %s",
        source_region,
        len(sources),
        sum(k.size for k in sources) / 1e12,
        len(verified),
        len(bad),
        target_region,
        bad[:20],
    )
    return verified, bad


def delete_verified(target_region: str, source_region: str, verified: list[KeptFile], num_threads: int) -> None:
    """Delete exactly the verified source files, after recording their paths in a manifest."""
    fs = _gcs()
    paths = sorted(k.name for k in verified)
    target_ns = get_spec("lpv11_fastpipe_v2_1_fused").namespace(REGION_TO_BUCKET[target_region])
    manifest = f"{target_ns}/_consolidated/deleted-{source_region}-{int(time.time())}.json"
    with fsspec.open(manifest, "w") as f:
        json.dump(paths, f)
    logger.info("manifest %s", manifest)
    batches = [paths[i : i + DELETE_BATCH] for i in range(0, len(paths), DELETE_BATCH)]
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        for i, _ in enumerate(pool.map(fs.rm, batches)):
            if i % 100 == 0:
                logger.info("deleted batch %d/%d", i, len(batches))
    left = sum(len(_kept_shards(spec_id, source_region)) for spec_id in SPEC_IDS)
    logger.info("DELETED %d verified files from %s; %d kept files remain there", len(paths), source_region, left)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-region", required=True, choices=sorted(REGION_TO_BUCKET))
    ap.add_argument("--source-regions", nargs="+", required=True, choices=sorted(REGION_TO_BUCKET))
    ap.add_argument("--num-threads", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="Copy (or delete) only the first N files (a test run).")
    ap.add_argument("--dry-run", action="store_true", help="Report the files and bytes that would move, copy nothing.")
    ap.add_argument(
        "--partition",
        default="0/1",
        help="I/N: copy only the I-th of N disjoint parts of the file names (parallel copy jobs).",
    )
    ap.add_argument(
        "--verify-sources", action="store_true", help="Instead of copying, check each source against the target."
    )
    ap.add_argument(
        "--delete-verified", action="store_true", help="With --verify-sources: delete the verified source files."
    )
    args = ap.parse_args()
    if args.target_region in args.source_regions:
        raise ValueError("target region cannot also be a source")
    if args.delete_verified and not args.verify_sources:
        raise ValueError("--delete-verified requires --verify-sources")
    if args.verify_sources:
        for region in args.source_regions:
            verified, _ = verify_source(args.target_region, region)
            if args.delete_verified:
                delete_verified(args.target_region, region, verified[: args.limit], args.num_threads)
        return
    index, count = (int(x) for x in args.partition.split("/"))
    if not 0 <= index < count:
        raise ValueError(f"bad --partition {args.partition}")
    consolidate(args.target_region, args.source_regions, args.num_threads, args.limit, args.dry_run, (index, count))


if __name__ == "__main__":
    main()
