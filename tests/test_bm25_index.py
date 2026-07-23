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


def test_stream_build_single_shard_returns_provenance(tmp_path):
    build = stream_build(_docs_with_ids(), str(tmp_path / "idx"))
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
    build = stream_build(_docs_with_ids(), str(tmp_path / "idx"))
    index = load_local_index([s.shard_dir for s in build.shards], mmap=False)
    assert index.search(query, k=1)[0].metadata["url"] == expected_url


def test_stream_build_shards_by_text_budget_and_merges(tmp_path):
    # A budget below one doc's text forces a flush after every document, so the
    # query path must merge across independently-built sub-indices.
    seen: list[tuple[int, Bm25ShardResult]] = []
    build = stream_build(
        _docs_with_ids(),
        str(tmp_path / "idx"),
        text_bytes_budget=1,
        on_shard_built=lambda i, r: seen.append((i, r)),
    )
    assert len(build.shards) == len(_DOCS)
    assert build.doc_count == len(_DOCS)
    # Callback fired once per shard, with monotonic indices.
    assert [i for i, _ in seen] == list(range(len(_DOCS)))
    # doc_ids stay globally unique and contiguous across sub-indices.
    index = load_local_index([s.shard_dir for s in build.shards], mmap=False)
    hits = index.search("united states independence", k=5)
    assert hits[0].metadata["url"] == "http://example.com/usa"
    assert len({h.metadata["doc_id"] for h in hits}) == len(hits)
    assert sorted(h.metadata["doc_id"] for h in hits) == list(range(len(_DOCS)))


def test_stream_build_skips_empty_text_and_raises_on_empty_corpus(tmp_path):
    build = stream_build([{"text": "hello world"}, {"text": ""}, {"no_text": 1}], str(tmp_path / "idx"))
    assert build.doc_count == 1
    with pytest.raises(ValueError):
        stream_build([{"text": ""}, {"other": "x"}], str(tmp_path / "empty"))


def test_smoke_test_reports_metrics(tmp_path):
    build = stream_build(_docs_with_ids(), str(tmp_path / "idx"))
    report = smoke_test_index([s.shard_dir for s in build.shards], doc_count=build.doc_count)
    assert report["num_hits"] >= 1
    assert report["top_url"] is not None
    assert report["query_ms"] >= 0
    assert "url" in report["top_metadata_keys"]
