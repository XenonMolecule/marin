# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror a corpus's shards into the compute region, 1:1 on basename.

Why this is its own step rather than "just read across regions": the scorer gets
re-run. It gets resumed after preemption, re-run when a threshold changes, and
re-run again when the second stage catches up months later. Reading the source
across regions makes every one of those re-runs pay egress again. Copying once
pays it exactly once, and regional storage is roughly two orders of magnitude
cheaper per month than the egress it avoids (37.8 GB for the priority wave is
about $0.76 either way, but only the mirror caps it at *once*).

Copies are server-side (GCS rewrite), so no shard content passes through this
worker — it only issues the calls. That means a small CPU box saturates it.

Idempotent at shard granularity: a shard whose destination already exists with a
matching byte count is skipped, so an interrupted mirror resumes for free.

    python -m experiments.baseline_collection.grid_mirror \\
        --dataset dclm_10k --num-chunks 8 --chunk-idx 0
"""

from __future__ import annotations

import argparse
import logging
import posixpath
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec

from experiments.baseline_collection.grid_corpora import (
    GRID_CORPORA,
    GridCorpus,
    assign_shards,
    list_shards,
    mirrored,
)

logger = logging.getLogger(__name__)

# GCS rewrite is latency-bound, not bandwidth-bound, from the caller's side.
COPY_PARALLELISM = 32


def _destination(corpus: GridCorpus, target: GridCorpus, shard: str) -> str:
    """Mirror path for ``shard``, preserving its basename exactly."""
    return f"{target.path}/{posixpath.basename(shard)}"


def _copy_one(src: str, dst: str) -> tuple[int, bool]:
    """Server-side copy ``src`` to ``dst`` unless already there at the same size.

    Returns:
        ``(bytes, skipped)`` — ``bytes`` is the source size either way, so the
        caller can report total mirrored volume and verify it against the source
        directory.
    """
    fs, src_path = fsspec.core.url_to_fs(src)
    _, dst_path = fsspec.core.url_to_fs(dst)
    size = int(fs.size(src_path) or 0)
    if size == 0:
        # A truly 0-byte object means a failed or in-flight write, so refuse it.
        # Note this is NOT the same as a shard containing no documents: a valid
        # empty gzip is ~20 bytes and is entirely normal for a filtered corpus
        # (63% of fineweb_cc's 21,531 shards are exactly that). Those copy fine
        # and produce zero-row outputs downstream.
        raise ValueError(f"source shard is 0 bytes — write failed or in flight: {src}")
    if fs.exists(dst_path) and int(fs.size(dst_path) or 0) == size:
        return size, True
    fs.copy(src_path, dst_path)
    return size, False


def run_mirror(dataset: str, num_chunks: int, chunk_idx: int) -> None:
    """Mirror this chunk's share of ``dataset`` into the compute region."""
    corpus = GRID_CORPORA[dataset]
    target = mirrored(dataset)
    if corpus.region == target.region:
        logger.info("%s is already in %s — nothing to mirror", dataset, target.region)
        return

    shards = assign_shards(list_shards(corpus), num_chunks, chunk_idx)
    logger.info(
        "mirroring %s: %d shards, %s -> %s",
        dataset,
        len(shards),
        corpus.region,
        target.region,
    )

    started = time.monotonic()
    total_bytes = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=COPY_PARALLELISM) as pool:
        results = pool.map(lambda s: _copy_one(s, _destination(corpus, target, s)), shards)
        for done, (size, was_skipped) in enumerate(results, start=1):
            total_bytes += size
            skipped += was_skipped
            if done % 500 == 0:
                logger.info("  %d/%d shards (%.1f GiB)", done, len(shards), total_bytes / 1024**3)

    logger.info(
        "mirrored %d shards (%d already present), %.2f GiB, %.0fs -> %s",
        len(shards),
        skipped,
        total_bytes / 1024**3,
        time.monotonic() - started,
        target.path,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(GRID_CORPORA))
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError(f"--chunk-idx {args.chunk_idx} out of range for --num-chunks {args.num_chunks}")
    run_mirror(args.dataset, args.num_chunks, args.chunk_idx)


if __name__ == "__main__":
    main()
