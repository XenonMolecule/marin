# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the streaming BM25 backup index: build, shard, merge, metadata.

These exercise externally-observable behavior against a real (tiny) ``bm25s``
index on local disk -- no network, no toolchain. They assert both correctness
(the right document is retrieved and its provenance metadata comes back) and that
the streaming path shards by text budget and merges across sub-indices. ``bm25s``
is an optional dep, so the whole module skips cleanly if it is not installed.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("bm25s")

from experiments.infinigram.bm25_build import (
    Bm25ShardResult,
    bm25_index_dir,
    build_subindex,
    stream_build,
)
from experiments.infinigram.bm25_query import load_local_index, smoke_test_index
from experiments.infinigram.targets import Collection, get_target

# A tiny corpus where each doc is unambiguously the best match for one query term,
# and every doc carries a url so we can assert provenance round-trips.
_DOCS = [
    {"text": "the quick brown fox jumps over the lazy dog", "url": "http://example.com/fox"},
    {"text": "a fast red panda climbs the bamboo tree", "url": "http://example.com/panda"},
    {"text": "the united states of america declared independence in congress", "url": "http://example.com/usa"},
    {"text": "python is a popular programming language for data science", "url": "http://example.com/python"},
    {"text": "photosynthesis converts sunlight into energy in plant chloroplasts", "url": "http://example.com/plants"},
]


def _docs_with_ids():
    """Copies of _DOCS with a warc_record_id, so provenance round-trip is testable."""
    return [{**d, "warc_record_id": f"rec-{d['url'].rsplit('/', 1)[1]}"} for d in _DOCS]


def test_bm25_index_dir_is_in_region_and_namespaced():
    target = get_target("dclm", Collection.FULL)
    assert bm25_index_dir(target) == "gs://marin-us-central2/bm25_indices/full/dclm"


def test_build_subindex_returns_metrics_and_metadata(tmp_path):
    texts = [d["text"] for d in _DOCS]
    meta = [{"doc_id": i, "url": d["url"]} for i, d in enumerate(_DOCS)]
    result = build_subindex(texts, meta, str(tmp_path / "shard_000"))
    assert isinstance(result, Bm25ShardResult)
    assert result.doc_count == len(_DOCS)
    assert result.index_bytes > 0
    assert result.build_seconds >= 0
    # build_subindex consumes (clears) the texts list to free memory.
    assert texts == []

    index = load_local_index([result.shard_dir], mmap=False)
    assert index.num_docs == len(_DOCS)
    hits = index.search("independence of the united states", k=3)
    assert hits[0].metadata["url"] == "http://example.com/usa"
    assert hits[0].metadata["doc_id"] == 2


def _one_shard(docs):
    """A single input-shard reader over all docs."""
    return [lambda: iter(docs)]


def _per_doc_shards(docs):
    """One input-shard reader per doc (so a tiny budget flushes after each)."""
    return [(lambda d=d: iter([d])) for d in docs]


def test_stream_build_single_shard_returns_provenance(tmp_path):
    build = stream_build(_one_shard(_docs_with_ids()), str(tmp_path / "idx"))
    assert build.doc_count == len(_DOCS)
    assert len(build.shards) == 1
    index = load_local_index([s.shard_dir for s in build.shards], mmap=False)
    top = index.search("panda bamboo tree", k=1)[0]
    assert top.metadata["url"] == "http://example.com/panda"
    assert top.metadata["warc_record_id"] == "rec-panda"
    assert "preview" in top.metadata


@pytest.mark.parametrize(
    "query,expected_url",
    [
        ("brown fox", "http://example.com/fox"),
        ("machine data science programming", "http://example.com/python"),
        ("sunlight chloroplasts energy", "http://example.com/plants"),
    ],
)
def test_search_ranks_the_relevant_doc_first(tmp_path, query, expected_url):
    build = stream_build(_one_shard(_docs_with_ids()), str(tmp_path / "idx"))
    index = load_local_index([s.shard_dir for s in build.shards], mmap=False)
    assert index.search(query, k=1)[0].metadata["url"] == expected_url


def test_stream_build_flushes_at_shard_boundary_and_merges(tmp_path):
    # One input shard per doc + a tiny budget -> flush after every shard, so the
    # query path must merge across independently-built sub-indices.
    seen: list[tuple[int, int]] = []  # (sub_num, shards_done)
    build = stream_build(
        _per_doc_shards(_docs_with_ids()),
        str(tmp_path / "idx"),
        text_bytes_budget=1,
        on_flush=lambda sub, r, sd, did: seen.append((sub, sd)),
    )
    assert len(build.shards) == len(_DOCS)
    assert build.doc_count == len(_DOCS)
    # sub-index numbers and shards_done advance one per flush.
    assert [sub for sub, _ in seen] == list(range(len(_DOCS)))
    assert [sd for _, sd in seen] == list(range(1, len(_DOCS) + 1))
    index = load_local_index([s.shard_dir for s in build.shards], mmap=False)
    hits = index.search("united states independence", k=5)
    assert hits[0].metadata["url"] == "http://example.com/usa"
    assert sorted(h.metadata["doc_id"] for h in hits) == list(range(len(_DOCS)))


def test_stream_build_resume_continues_ids_and_shard_numbers(tmp_path):
    # First run builds shards 0-1; a "resume" builds the rest with continued
    # doc ids and sub-index numbers -> a complete, contiguous index.
    docs = _docs_with_ids()
    idx = str(tmp_path / "idx")
    b1 = stream_build(_per_doc_shards(docs)[:2], idx, text_bytes_budget=1)
    assert len(b1.shards) == 2
    b2 = stream_build(_per_doc_shards(docs), idx, text_bytes_budget=1, start_shard=2, start_doc_id=2, start_sub=2)
    assert len(b2.shards) == len(docs) - 2
    assert b2.doc_count == len(docs)  # doc ids continued from 2
    assert any(s.shard_dir.endswith("shard_002") for s in b2.shards)
    all_dirs = [os.path.join(idx, f"shard_{i:03d}") for i in range(len(docs))]
    index = load_local_index(all_dirs, mmap=False)
    hits = index.search("united states independence", k=len(docs))
    assert sorted(h.metadata["doc_id"] for h in hits) == list(range(len(docs)))


def test_stream_build_empty_corpus_returns_no_shards(tmp_path):
    build = stream_build(_one_shard([{"text": "hi there"}, {"text": ""}, {"no_text": 1}]), str(tmp_path / "idx"))
    assert build.doc_count == 1
    empty = stream_build(_one_shard([{"text": ""}, {"other": "x"}]), str(tmp_path / "empty"))
    assert empty.doc_count == 0
    assert len(empty.shards) == 0


def test_batched_index_bounds_memory_and_matches_eager(tmp_path):
    # Batched querying (one sub-index at a time) must return the same merged
    # top-k -- with url -- as loading all sub-indices at once.
    from experiments.infinigram.bm25_query import BatchedBm25Index

    build = stream_build(_per_doc_shards(_docs_with_ids()), str(tmp_path / "idx"), text_bytes_budget=1)
    dirs = [s.shard_dir for s in build.shards]
    assert len(dirs) == len(_DOCS)  # one sub-index per doc

    batched = BatchedBm25Index(dirs, batch_size=1, mmap=False)
    assert batched.num_docs == len(_DOCS)
    hits = batched.search("united states independence", k=5)
    assert hits[0].metadata["url"] == "http://example.com/usa"
    assert sorted(h.metadata["doc_id"] for h in hits) == list(range(len(_DOCS)))
    # Same top hit as the eager index.
    eager = load_local_index(dirs, mmap=False)
    assert batched.search("brown fox", k=1)[0].metadata["url"] == eager.search("brown fox", k=1)[0].metadata["url"]


def test_smoke_test_reports_metrics(tmp_path):
    build = stream_build(_one_shard(_docs_with_ids()), str(tmp_path / "idx"))
    report = smoke_test_index([s.shard_dir for s in build.shards], doc_count=build.doc_count)
    assert report["num_hits"] >= 1
    assert report["top_url"] is not None
    assert report["query_ms"] >= 0
    assert "url" in report["top_metadata_keys"]
