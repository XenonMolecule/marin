# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a BM25 retrieval index over a staged corpus, as a backup to infini-gram.

infini-gram-mini answers *exact substring* n-gram queries via an FM-index that
needs a fragile prebuilt gcc-5/sdsl toolchain (see ``packaging.md``). BM25 answers
a different question -- *ranked bag-of-words relevance retrieval* -- but it is a
pure-pip ``bm25s`` index (numpy/scipy only, no native toolchain), so it is the
robust fallback for the document-retrieval half of the use case and never depends
on the infini-gram runtime.

Design: one *self-contained* ``bm25s`` sub-index per shard group. Each shard group
is a set of staged ``.jsonl.gz`` files bounded by :data:`SHARD_COMPRESSED_BYTES`
so a group's texts + tokens fit comfortably in a worker's RAM -- this is what lets
the 10k-WARC (~60 GiB) corpora index without a giant-memory node. Every sub-index
stores its own vocab and a row-aligned *metadata corpus* (url / warc ids), which
``bm25s`` returns verbatim on retrieval, so a lookup yields the document's
provenance -- not just an opaque id. The query layer (:mod:`bm25_query`) searches
every sub-index and merges by score; per-shard idf is accepted (standard for
distributed BM25 and immaterial for a backup retriever).

Separation of concerns: this module only *computes* the local index and returns
metrics. Upload lives in :mod:`bm25_pipeline`.
"""

import gzip
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

# bm25s is an optional dep, installed only in the build jobs (via the launcher's
# pip_packages). Guard the import so this module -- and thus the coordinator that
# imports bm25_index_dir from it -- loads without bm25s present.
try:
    import bm25s
except ImportError:
    bm25s = None

from experiments.infinigram.provenance import PROVENANCE_FIELDS
from experiments.infinigram.stage import StagedCorpus
from experiments.infinigram.targets import REGION_BUCKET, IndexTarget

logger = logging.getLogger(__name__)

# Canonical GCS home for BM25 indices, parallel to infinigram_indices/.
BM25_INDEX_ROOT_TEMPLATE = "{bucket}/bm25_indices/{collection}/{dataset}"

# One bm25s sub-index per group of staged shards whose *compressed* size sums to
# at most this. Peak build RAM is a LARGE multiple of the compressed size:
# jsonl.gz decompresses ~4-5x into the in-memory text list, and bm25s's Python
# token-id lists cost several times the text again -- empirically a 2 GiB group
# OOM-killed a 32 GiB worker. 512 MiB peaks around ~10 GiB, safe on the 48 GiB
# SMALL worker; FULL passes a larger budget (it has 128 GiB) to keep its
# sub-index count -- and thus build time -- down. Tune via --shard-bytes.
SHARD_COMPRESSED_BYTES = 512 * 1024**2

# Per-document metadata stored in (and returned by) the index. ``doc_id`` is a
# corpus-global running index; the rest is provenance recovered at stage time.
# A short text preview makes hits human-inspectable and lets the smoke test assert
# on retrieved content without a second lookup.
_METADATA_FIELDS = ("doc_id", *PROVENANCE_FIELDS, "id")
_PREVIEW_CHARS = 200

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
    """A finished local BM25 index: one or more sub-index dirs under ``save_dir``."""

    save_dir: str
    shards: tuple[Bm25ShardResult, ...]
    doc_count: int
    index_bytes: int
    build_seconds: float

    @property
    def shard_dirs(self) -> tuple[str, ...]:
        return tuple(s.shard_dir for s in self.shards)


def bm25_index_dir(target: IndexTarget) -> str:
    """Canonical GCS output dir for ``target``'s BM25 index (always in-region)."""
    return BM25_INDEX_ROOT_TEMPLATE.format(
        bucket=REGION_BUCKET[target.region],
        collection=target.collection.value,
        dataset=target.dataset,
    )


def _staged_shard_files(data_dir: str) -> list[str]:
    return sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl.gz"))


def _group_by_bytes(files: list[str], budget_bytes: int) -> list[list[str]]:
    """Greedily pack files into groups whose on-disk bytes sum to <= budget.

    A single file larger than the budget becomes its own group (never split a
    file across sub-indices -- doc ids must stay contiguous within a sub-index).
    """
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for f in files:
        size = os.path.getsize(f)
        if current and current_bytes + size > budget_bytes:
            groups.append(current)
            current, current_bytes = [], 0
        current.append(f)
        current_bytes += size
    if current:
        groups.append(current)
    return groups


def _read_docs(files: list[str], start_doc_id: int) -> tuple[list[str], list[dict]]:
    """Read (text, metadata) for every doc in ``files``, assigning global doc ids.

    Metadata carries provenance and a text preview; ``text`` itself is not stored
    (only tokenized into the index) to keep the metadata corpus compact.
    """
    texts: list[str] = []
    meta: list[dict] = []
    doc_id = start_doc_id
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                text = rec.get("text")
                if not text:
                    continue
                texts.append(text)
                row = {f: rec.get(f) for f in _METADATA_FIELDS if rec.get(f) is not None}
                row["doc_id"] = doc_id
                row["preview"] = text[:_PREVIEW_CHARS]
                meta.append(row)
                doc_id += 1
    return texts, meta


def _build_one_shard(files: list[str], shard_dir: str, start_doc_id: int) -> Bm25ShardResult:
    """Tokenize, index, and save a single self-contained bm25s sub-index."""
    if bm25s is None:
        raise RuntimeError("bm25s is not installed; the build job must include it in pip_packages.")
    t0 = time.monotonic()
    texts, meta = _read_docs(files, start_doc_id)
    if not texts:
        raise ValueError(f"shard group produced 0 docs from {files}")

    tokens = bm25s.tokenize(texts, stopwords=_STOPWORDS, show_progress=False)
    del texts  # free the raw text before the (larger) index arrays are allocated
    retriever = bm25s.BM25(corpus=meta)
    retriever.index(tokens)
    os.makedirs(shard_dir, exist_ok=True)
    retriever.save(shard_dir, corpus=meta)

    build_seconds = time.monotonic() - t0
    index_bytes = _dir_bytes(shard_dir)
    logger.info(
        "Built bm25 sub-index %s: %d docs, %.2f MiB, %.1fs",
        shard_dir,
        len(meta),
        index_bytes / 1024**2,
        build_seconds,
    )
    return Bm25ShardResult(
        shard_dir=shard_dir, doc_count=len(meta), index_bytes=index_bytes, build_seconds=build_seconds
    )


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def build_bm25_index(
    staged: StagedCorpus,
    save_dir: str,
    *,
    shard_bytes: int = SHARD_COMPRESSED_BYTES,
) -> Bm25BuildResult:
    """Build a (possibly multi-shard) BM25 index over ``staged`` into ``save_dir``.

    Groups staged shards by on-disk size so each sub-index fits in RAM, builds one
    self-contained bm25s index per group into ``save_dir/shard_NN``, and returns
    build metrics. Small corpora produce a single sub-index.
    """
    files = _staged_shard_files(staged.data_dir)
    if not files:
        raise ValueError(f"no staged .jsonl.gz shards under {staged.data_dir}")
    groups = _group_by_bytes(files, shard_bytes)
    logger.info("BM25 build: %d staged files -> %d sub-index shard(s)", len(files), len(groups))

    os.makedirs(save_dir, exist_ok=True)
    shards: list[Bm25ShardResult] = []
    next_doc_id = 0
    t0 = time.monotonic()
    for i, group in enumerate(groups):
        shard_dir = os.path.join(save_dir, f"shard_{i:03d}")
        result = _build_one_shard(group, shard_dir, next_doc_id)
        next_doc_id += result.doc_count
        shards.append(result)

    total = Bm25BuildResult(
        save_dir=save_dir,
        shards=tuple(shards),
        doc_count=next_doc_id,
        index_bytes=sum(s.index_bytes for s in shards),
        build_seconds=time.monotonic() - t0,
    )
    logger.info(
        "BM25 build done: %d docs across %d shard(s), %.2f MiB, %.1fs",
        total.doc_count,
        len(shards),
        total.index_bytes / 1024**2,
        total.build_seconds,
    )
    return total


def iter_shard_metrics(build: Bm25BuildResult) -> Iterator[dict]:
    """Per-shard efficiency records for the manifest."""
    for s in build.shards:
        yield {
            "shard_dir": os.path.basename(s.shard_dir),
            "doc_count": s.doc_count,
            "index_bytes": s.index_bytes,
            "build_seconds": round(s.build_seconds, 2),
        }
