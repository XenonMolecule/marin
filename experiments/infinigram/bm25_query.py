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

import heapq
import json
import logging
import os
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
from experiments.infinigram.gcs_io import download_dir
from experiments.infinigram.targets import Collection, IndexTarget, get_target

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


def _manifest_url(index_dir: str) -> str:
    return f"{index_dir.rstrip('/')}/{MANIFEST_NAME}"


def shard_dirs_for(target: IndexTarget) -> list[str]:
    """Resolve a target's uploaded sub-index dirs from its ``manifest.json``."""
    manifest_url = _manifest_url(bm25_index_dir(target))
    if not fsspec_exists(manifest_url):
        raise FileNotFoundError(f"no BM25 index for {target.name} (missing {manifest_url}); build it first.")
    with fsspec.open(manifest_url, "r") as f:
        return list(json.load(f)["shard_dirs"])


def _localize(shard_dirs: list[str], cache_root: str) -> list[str]:
    """Mirror gs:// sub-index dirs to local disk (bm25s mmap needs local files)."""
    local: list[str] = []
    for d in shard_dirs:
        if not d.startswith("gs://"):
            local.append(d)
            continue
        dest = os.path.join(cache_root, d.replace("gs://", ""))
        download_dir(d, dest)
        local.append(dest)
    return local


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
) -> Bm25Index:
    """Open the BM25 index for ``(dataset, collection)`` (optionally several) to query.

    Mirrors the uploaded sub-indices locally, memory-maps them, and returns a
    ready :class:`Bm25Index`. Passing ``also`` searches multiple corpora jointly.
    """
    targets = [get_target(dataset, collection)]
    for ds, col in also or []:
        targets.append(get_target(ds, col))

    dirs: list[str] = []
    for t in targets:
        dirs.extend(shard_dirs_for(t))

    cache_root = cache_root or tempfile.mkdtemp(prefix="bm25-idx-")
    local_dirs = _localize(dirs, cache_root)
    logger.info("Opening BM25 index over %d sub-index dir(s): %s", len(local_dirs), [t.name for t in targets])
    return load_local_index(local_dirs, mmap=mmap)


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
