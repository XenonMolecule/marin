# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a BM25 retrieval index over a document corpus, as a backup to infini-gram.

infini-gram-mini answers *exact substring* n-gram queries via an FM-index that
needs a fragile prebuilt gcc-5/sdsl toolchain (see ``packaging.md``). BM25 answers
a different question -- *ranked bag-of-words relevance retrieval* -- but it is a
pure-pip ``bm25s`` index (numpy/scipy only, no native toolchain), so it is the
robust fallback for the document-retrieval half of the use case.

**Streaming design.** Cluster workers cap at 100 GiB disk, far below a 10k-WARC
corpus, so we never stage the whole corpus locally. :func:`stream_build` consumes
a ``(text, metadata)`` iterator (read shard-by-shard from GCS in-region by the
pipeline), buffers documents until their text reaches :data:`TEXT_BYTES_BUDGET`,
then flushes one *self-contained* ``bm25s`` sub-index. A caller-supplied callback
uploads each finished sub-index to GCS and deletes it locally, so both disk and
RAM stay bounded no matter how large the corpus is. The query layer
(:mod:`bm25_query`) loads every sub-index and merges by score; per-shard idf is
accepted (standard for distributed BM25 and immaterial for a backup retriever).

Every sub-index stores a row-aligned *metadata corpus* (url / warc ids / preview)
that ``bm25s`` returns verbatim on retrieval, so a lookup yields the document's
provenance -- not just an opaque id.
"""

import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# bm25s is an optional dep, installed only in the build jobs (via the launcher's
# pip_packages). Guard the import so this module -- and thus the coordinator that
# imports bm25_index_dir from it -- loads without bm25s present.
try:
    import bm25s
except ImportError:
    bm25s = None

from experiments.infinigram.provenance import PROVENANCE_FIELDS
from experiments.infinigram.targets import REGION_BUCKET, IndexTarget

logger = logging.getLogger(__name__)

# Canonical GCS home for BM25 indices, parallel to infinigram_indices/.
BM25_INDEX_ROOT_TEMPLATE = "{bucket}/bm25_indices/{collection}/{dataset}"

# Flush a sub-index once buffered *uncompressed* text reaches this. Peak build RAM
# is a few multiples of this (raw text + bm25s token-id lists + sparse arrays), so
# 2 GiB of text peaks around ~10-15 GiB -- safe on the 32-96 GiB workers we request
# while keeping the sub-index count (and query-time merge fan-out) modest.
TEXT_BYTES_BUDGET = 2 * 1024**3

# Per-document metadata stored in (and returned by) the index. ``doc_id`` is a
# corpus-global running index; provenance fields (url / warc ids) are recovered
# upstream; ``source_doc_id`` / ``modernbert_prob`` ride along when the source
# carries them (e.g. fastpipe bands, whose url is joinable later by doc_id). A
# short text preview makes hits human-inspectable and lets the smoke test assert
# on retrieved content without a second lookup.
METADATA_FIELDS = ("doc_id", *PROVENANCE_FIELDS, "id", "source_doc_id", "modernbert_prob")
PREVIEW_CHARS = 200

# bm25s tokenizer settings. English stopwords help precision; we deliberately skip
# stemming so the index needs no compiled Stemmer wheel (pure-pip fallback).
_STOPWORDS = "en"


@dataclass(frozen=True)
class Bm25ShardResult:
    """One built bm25s sub-index and its build metrics."""

    shard_dir: str
    doc_count: int
    index_bytes: int
    build_seconds: float


@dataclass(frozen=True)
class Bm25BuildResult:
    """A finished BM25 index: one or more sub-index shards."""

    shards: tuple[Bm25ShardResult, ...]
    doc_count: int
    index_bytes: int
    build_seconds: float


def bm25_index_dir(target: IndexTarget) -> str:
    """Canonical GCS output dir for ``target``'s BM25 index (always in-region)."""
    return BM25_INDEX_ROOT_TEMPLATE.format(
        bucket=REGION_BUCKET[target.region],
        collection=target.collection.value,
        dataset=target.dataset,
    )


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def build_subindex(texts: list[str], meta: list[dict], shard_dir: str) -> Bm25ShardResult:
    """Tokenize, index, and save one self-contained bm25s sub-index to ``shard_dir``.

    ``meta[i]`` is stored as the corpus row for ``texts[i]`` and returned verbatim
    on retrieval. Raw text is freed before the (larger) index arrays are built.
    """
    if bm25s is None:
        raise RuntimeError("bm25s is not installed; the build job must include it in pip_packages.")
    if not texts:
        raise ValueError(f"refusing to build empty sub-index at {shard_dir}")
    t0 = time.monotonic()
    tokens = bm25s.tokenize(texts, stopwords=_STOPWORDS, show_progress=False)
    doc_count = len(texts)
    texts.clear()
    retriever = bm25s.BM25(corpus=meta)
    retriever.index(tokens)
    os.makedirs(shard_dir, exist_ok=True)
    retriever.save(shard_dir, corpus=meta)

    build_seconds = time.monotonic() - t0
    index_bytes = _dir_bytes(shard_dir)
    logger.info(
        "Built bm25 sub-index %s: %d docs, %.2f MiB, %.1fs",
        shard_dir,
        doc_count,
        index_bytes / 1024**2,
        build_seconds,
    )
    return Bm25ShardResult(
        shard_dir=shard_dir, doc_count=doc_count, index_bytes=index_bytes, build_seconds=build_seconds
    )


def _make_meta(rec: dict, doc_id: int) -> dict:
    """Metadata row for a document: provenance fields present on ``rec`` + preview.

    The source's own ``doc_id`` (a stable join key, e.g. fastpipe's) is preserved
    as ``source_doc_id`` before ``doc_id`` is overwritten with the corpus-global
    running index.
    """
    text = rec.get("text") or ""
    row = {f: rec[f] for f in METADATA_FIELDS if rec.get(f) is not None}
    if rec.get("doc_id") is not None:
        row["source_doc_id"] = rec["doc_id"]
    row["doc_id"] = doc_id
    row["preview"] = text[:PREVIEW_CHARS]
    return row


def stream_build(
    shard_readers: "list[Callable[[], Iterable[dict]]]",
    save_dir: str,
    *,
    text_bytes_budget: int = TEXT_BYTES_BUDGET,
    on_flush: "Callable[[int, Bm25ShardResult, int, int], None] | None" = None,
    start_shard: int = 0,
    start_doc_id: int = 0,
    start_sub: int = 0,
) -> Bm25BuildResult:
    """Build a BM25 index by streaming input shards; flush a sub-index at *shard
    boundaries* once buffered text reaches ``text_bytes_budget``.

    ``shard_readers[i]()`` yields the docs (dicts with ``text``) of input shard i.
    Flushing at shard boundaries (rather than mid-shard) makes the build
    **resumable**: pass ``start_shard`` / ``start_doc_id`` / ``start_sub`` to
    continue after an interruption. ``on_flush(sub_num, result, shards_done,
    doc_id)`` runs after each sub-index -- the pipeline uploads it, deletes it
    locally, and checkpoints ``(shards_done, doc_id, sub_num+1)`` so a restart
    skips completed input shards. Assumes a single input shard fits in the budget
    (true for our datasets -- shards are far smaller than 2 GiB).
    """
    os.makedirs(save_dir, exist_ok=True)
    shards: list[Bm25ShardResult] = []
    texts: list[str] = []
    meta: list[dict] = []
    buffered_bytes = 0
    doc_id = start_doc_id
    sub_num = start_sub
    t0 = time.monotonic()

    def flush(shards_done: int) -> None:
        nonlocal texts, meta, buffered_bytes, sub_num
        if not texts:
            return
        result = build_subindex(texts, meta, os.path.join(save_dir, f"shard_{sub_num:03d}"))
        shards.append(result)
        if on_flush is not None:
            on_flush(sub_num, result, shards_done, doc_id)
        sub_num += 1
        texts, meta, buffered_bytes = [], [], 0

    for i in range(start_shard, len(shard_readers)):
        for rec in shard_readers[i]():
            text = rec.get("text")
            if not text:
                continue
            texts.append(text)
            meta.append(_make_meta(rec, doc_id))
            buffered_bytes += len(text)
            doc_id += 1
        # Flush only at the shard boundary, so a checkpoint == "shards 0..i done".
        if buffered_bytes >= text_bytes_budget:
            flush(i + 1)
    flush(len(shard_readers))

    # `shards` holds only THIS run's sub-indices (empty on a fully-resumed finalize);
    # the pipeline validates the cumulative uploaded set is non-empty.
    total = Bm25BuildResult(
        shards=tuple(shards),
        doc_count=doc_id,
        index_bytes=sum(s.index_bytes for s in shards),
        build_seconds=time.monotonic() - t0,
    )
    logger.info(
        "BM25 stream build (this run): %d docs total, %d new sub-index(es), %.2f MiB, %.1fs",
        total.doc_count,
        len(shards),
        total.index_bytes / 1024**2,
        total.build_seconds,
    )
    return total


def iter_shard_metrics(shards: Iterable[Bm25ShardResult]) -> list[dict]:
    """Per-shard efficiency records for the manifest."""
    return [
        {
            "shard": os.path.basename(s.shard_dir),
            "doc_count": s.doc_count,
            "index_bytes": s.index_bytes,
            "build_seconds": round(s.build_seconds, 2),
        }
        for s in shards
    ]
