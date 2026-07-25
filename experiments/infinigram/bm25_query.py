# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open and query BM25 backup indices; smoke-test a freshly built one.

A BM25 index for a ``(dataset, collection)`` is one or more self-contained
``bm25s`` sub-indices (see :mod:`bm25_build`). :class:`Bm25Index` loads every
sub-index (memory-mapped) and answers ranked-retrieval ``search`` queries,
merging results across sub-indices by score. **Every hit carries the document's
stored metadata (url / warc ids / preview)** -- the whole point of the backup is
that a lookup returns provenance, not an opaque id.

Switching which dataset you search is a one-liner: :func:`open_bm25_index` maps a
``(dataset, collection)`` to its uploaded index and returns a ready object.
"""

import argparse
import gc
import heapq
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass

import fsspec
from marin.utils import fsspec_exists

# bm25s is an optional dep (installed only in build/query jobs). Guard the import
# so the coordinator, which imports MANIFEST_NAME from here, loads without it.
try:
    import bm25s
except ImportError:
    bm25s = None

from experiments.infinigram.bm25_build import bm25_index_dir
from experiments.infinigram.bm25_sources import get_bm25_target
from experiments.infinigram.gcs_io import download_dir
from experiments.infinigram.targets import Collection, IndexTarget

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"

# Probe for the post-build smoke test: a common phrase that must retrieve at least
# one document whose returned metadata is intact.
_SMOKE_QUERY = "the United States of America"


def _require_bm25s():
    if bm25s is None:
        raise RuntimeError("bm25s is not installed; a query/build job must include it in pip_packages.")
    return bm25s


@dataclass(frozen=True)
class Bm25Hit:
    """One retrieved document: its BM25 score and stored metadata (url, ids, ...)."""

    score: float
    metadata: dict


class Bm25Index:
    """A queryable BM25 index over one or more memory-mapped ``bm25s`` sub-indices."""

    def __init__(self, retrievers: list):
        if not retrievers:
            raise ValueError("Bm25Index needs at least one sub-index")
        self._retrievers = retrievers

    @property
    def num_docs(self) -> int:
        """Total documents across all sub-indices."""
        return sum(r.scores["num_docs"] for r in self._retrievers)

    def search(self, query: str, k: int = 10) -> list[Bm25Hit]:
        """Return the top-``k`` documents for ``query`` across all sub-indices.

        Each sub-index is queried for its own top-``k``; the pooled results are
        merged by score. Metadata (url / warc ids / preview) rides along on each
        hit -- ``bm25s`` returns the stored corpus row verbatim.
        """
        query_tokens = _require_bm25s().tokenize(query, stopwords="en", show_progress=False)
        pooled: list[Bm25Hit] = []
        for retriever in self._retrievers:
            n = retriever.scores["num_docs"]
            docs, scores = retriever.retrieve(query_tokens, k=min(k, n), show_progress=False)
            for meta, score in zip(docs[0], scores[0], strict=True):
                pooled.append(Bm25Hit(score=float(score), metadata=meta))
        return heapq.nlargest(k, pooled, key=lambda h: h.score)


class BatchedBm25Index:
    """Query many sub-indices with bounded disk AND memory.

    Downloading and loading every sub-index at once makes both disk and RAM scale
    with total index size -- infeasible for the large indexes (the 31 GiB / 39-sub-
    index resiliparse-300 OOM'd just localizing; the complete 10k index is ~2 TB,
    which no 100 GiB-disk worker can hold). Instead, for each batch of at most
    ``batch_size`` sub-indices we: mirror just that batch to local disk, load,
    query, then delete the batch's local files before moving on. Peak disk and RAM
    stay ~``batch_size`` sub-indices regardless of index size. Sub-indices are
    re-downloaded per ``search`` -- fine for interactive lookups; keep the object
    for a whole session or cache upstream for high throughput.
    """

    def __init__(
        self,
        dirs: list[str],
        *,
        batch_size: int = 2,
        mmap: bool = True,
        cache_root: str | None = None,
    ):
        if not dirs:
            raise ValueError("BatchedBm25Index needs at least one sub-index dir")
        self._dirs = dirs  # gs:// (mirrored per batch) or already-local
        self._batch_size = batch_size
        self._mmap = mmap
        self._cache_root = cache_root or tempfile.mkdtemp(prefix="bm25-batch-")

    def _local_dest(self, d: str) -> str:
        return os.path.join(self._cache_root, d.replace("gs://", "")) if d.startswith("gs://") else d

    def _mirror(self, batch: list[str]) -> list[str]:
        """Mirror a batch's gs:// dirs to local disk (already-local dirs pass through)."""
        local = []
        for d in batch:
            dest = self._local_dest(d)
            if d.startswith("gs://"):
                download_dir(d, dest)
            local.append(dest)
        return local

    def _cleanup(self, batch: list[str]) -> None:
        for d in batch:
            if d.startswith("gs://"):
                shutil.rmtree(self._local_dest(d), ignore_errors=True)

    def _batches(self):
        for i in range(0, len(self._dirs), self._batch_size):
            yield self._dirs[i : i + self._batch_size]

    @property
    def num_docs(self) -> int:
        """Total docs, one batch at a time WITHOUT loading corpus (cheap)."""
        total = 0
        for batch in self._batches():
            local = self._mirror(batch)
            rs = [_require_bm25s().BM25.load(d, load_corpus=False, mmap=True) for d in local]
            total += sum(r.scores["num_docs"] for r in rs)
            del rs
            self._cleanup(batch)
            gc.collect()
        return total

    def search(self, query: str, k: int = 10) -> list[Bm25Hit]:
        pooled: list[Bm25Hit] = []
        for batch in self._batches():
            local = self._mirror(batch)
            retrievers = [_require_bm25s().BM25.load(d, load_corpus=True, mmap=self._mmap) for d in local]
            pooled.extend(Bm25Index(retrievers).search(query, k))
            del retrievers
            self._cleanup(batch)  # free this batch's disk before the next
            gc.collect()
        return heapq.nlargest(k, pooled, key=lambda h: h.score)


def _manifest_url(index_dir: str) -> str:
    return f"{index_dir.rstrip('/')}/{MANIFEST_NAME}"


def shard_dirs_for(target: IndexTarget) -> list[str]:
    """Resolve a target's uploaded sub-index dirs from its ``manifest.json``."""
    manifest_url = _manifest_url(bm25_index_dir(target))
    if not fsspec_exists(manifest_url):
        raise FileNotFoundError(f"no BM25 index for {target.name} (missing {manifest_url}); build it first.")
    with fsspec.open(manifest_url, "r") as f:
        return list(json.load(f)["shard_dirs"])


def load_local_index(shard_dirs: list[str], *, mmap: bool = True) -> Bm25Index:
    """Open already-local sub-index dirs (used by the post-build smoke test)."""
    retrievers = [_require_bm25s().BM25.load(d, load_corpus=True, mmap=mmap) for d in shard_dirs]
    return Bm25Index(retrievers)


def open_bm25_index(
    dataset: str,
    collection: Collection = Collection.FULL,
    *,
    also: list[tuple[str, Collection]] | None = None,
    cache_root: str | None = None,
    mmap: bool = True,
    batch_size: int = 2,
) -> BatchedBm25Index:
    """Open the BM25 index for ``(dataset, collection)`` (optionally several) to query.

    Mirrors the uploaded sub-indices to local disk and returns a
    :class:`BatchedBm25Index`, which queries them in memory-bounded batches so RAM
    stays flat regardless of index size (needed for the multi-million-doc indexes).
    Passing ``also`` searches multiple corpora jointly.
    """
    targets = [get_bm25_target(dataset, collection)]
    for ds, col in also or []:
        targets.append(get_bm25_target(ds, col))

    dirs: list[str] = []
    for t in targets:
        dirs.extend(shard_dirs_for(t))

    logger.info(
        "Opening BM25 index over %d sub-index dir(s) [batch_size=%d, per-batch mirror]: %s",
        len(dirs),
        batch_size,
        [t.name for t in targets],
    )
    # Pass gs:// dirs straight through: BatchedBm25Index mirrors each batch to
    # local disk on demand and deletes it after, so disk/RAM stay bounded.
    return BatchedBm25Index(dirs, batch_size=batch_size, mmap=mmap, cache_root=cache_root)


def smoke_test_index(shard_dirs: list[str], *, doc_count: int) -> dict:
    """Validate a freshly built (local) BM25 index before upload.

    Asserts a common phrase retrieves at least one document and that the top hit
    carries usable metadata, and measures query latency. Returns a report; raises
    on the hard assertions.
    """
    index = load_local_index(shard_dirs, mmap=False)

    t0 = time.monotonic()
    hits = index.search(_SMOKE_QUERY, k=5)
    query_ms = (time.monotonic() - t0) * 1000

    if not hits:
        raise AssertionError(f"query {_SMOKE_QUERY!r} returned no hits; index looks empty or broken")
    top = hits[0]
    if not isinstance(top.metadata, dict) or "doc_id" not in top.metadata:
        raise AssertionError(f"top hit has no usable metadata: {top.metadata!r}")

    report = {
        "query": _SMOKE_QUERY,
        "num_hits": len(hits),
        "top_score": round(top.score, 4),
        "top_metadata_keys": sorted(top.metadata),
        "top_url": top.metadata.get("url"),
        "query_ms": round(query_ms, 2),
        "doc_count": doc_count,
    }
    logger.info("BM25 smoke test OK: %s", report)
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query an uploaded BM25 index (round-trip verification tool).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--collection", choices=[c.value for c in Collection], default=Collection.FULL.value)
    p.add_argument("--query", required=True, help="Free-text query.")
    p.add_argument("-k", type=int, default=5, help="Number of hits to return.")
    return p.parse_args()


def main() -> None:
    """Reopen an uploaded index from GCS and print ranked hits with their metadata.

    Proves an at-rest index is usable end-to-end (mirror from GCS -> mmap -> query
    -> metadata). Run as an in-region Iris job (needs GCS access + bm25s).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    index = open_bm25_index(args.dataset, Collection(args.collection))
    hits = index.search(args.query, k=args.k)
    logger.info("Query %r on %s-%s -> %d hits", args.query, args.dataset, args.collection, len(hits))
    for rank, hit in enumerate(hits):
        meta = hit.metadata
        logger.info(
            "  #%d score=%.4f url=%s warc_record_id=%s preview=%r",
            rank,
            hit.score,
            meta.get("url"),
            meta.get("warc_record_id"),
            (meta.get("preview") or "")[:120],
        )


if __name__ == "__main__":
    main()
