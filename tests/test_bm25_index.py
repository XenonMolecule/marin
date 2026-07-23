# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BM25 backup index: build, cross-shard query, and metadata return.

These exercise externally-observable behavior against a real (tiny) ``bm25s``
index on local disk -- no network, no toolchain. They assert both correctness
(the right document is retrieved and its provenance metadata comes back) and that
efficiency metrics are populated. ``bm25s`` is an optional dep, so the whole
module skips cleanly if it is not installed.
"""

from __future__ import annotations

import gzip
import json
import os

import pytest

pytest.importorskip("bm25s")

from experiments.infinigram.bm25_build import (
    _group_by_bytes,
    bm25_index_dir,
    build_bm25_index,
)
from experiments.infinigram.bm25_query import load_local_index, smoke_test_index
from experiments.infinigram.stage import StagedCorpus
from experiments.infinigram.targets import Collection, get_target

# A tiny corpus where each doc is unambiguously the best match for one query term,
# and every doc carries a url so we can assert provenance round-trips.
_DOCS = [
    ("the quick brown fox jumps over the lazy dog", "http://example.com/fox"),
    ("a fast red panda climbs the bamboo tree", "http://example.com/panda"),
    ("the united states of america declared independence in congress", "http://example.com/usa"),
    ("python is a popular programming language for data science and machine learning", "http://example.com/python"),
    ("photosynthesis converts sunlight into chemical energy in plant chloroplasts", "http://example.com/plants"),
]


def _write_staged(dir_path: str, docs: list[tuple[str, str]], *, shards: int = 1) -> StagedCorpus:
    """Write ``docs`` as ``shards`` local ``.jsonl.gz`` files, return a StagedCorpus."""
    os.makedirs(dir_path, exist_ok=True)
    per = [docs[i::shards] for i in range(shards)]
    for si, chunk in enumerate(per):
        with gzip.open(os.path.join(dir_path, f"shard-{si:05d}.jsonl.gz"), "wt", encoding="utf-8") as fh:
            for text, url in chunk:
                fh.write(json.dumps({"text": text, "url": url, "warc_record_id": f"rec-{url[-5:]}"}))
                fh.write("\n")
    return StagedCorpus(
        data_dir=dir_path,
        shard_count=shards,
        byte_count=0,
        doc_count=len(docs),
        matched_provenance=0,
        url_index_path=os.path.join(dir_path, "url_index.jsonl.gz"),
    )


def test_group_by_bytes_packs_and_never_splits_a_file(tmp_path):
    files = []
    for i in range(4):
        p = tmp_path / f"f{i}.gz"
        p.write_bytes(b"x" * 100)
        files.append(str(p))
    # Budget of 250 bytes -> groups of at most 2 files (2*100=200 <= 250 < 300).
    groups = _group_by_bytes(files, budget_bytes=250)
    assert [len(g) for g in groups] == [2, 2]
    # A file larger than the budget still becomes its own single-file group.
    big = tmp_path / "big.gz"
    big.write_bytes(b"x" * 500)
    groups2 = _group_by_bytes([str(big), files[0]], budget_bytes=250)
    assert [len(g) for g in groups2] == [1, 1]


def test_bm25_index_dir_is_in_region_and_namespaced():
    target = get_target("dclm", Collection.FULL)
    d = bm25_index_dir(target)
    assert d == "gs://marin-us-central2/bm25_indices/full/dclm"


def test_build_and_search_returns_correct_doc_with_metadata(tmp_path):
    staged = _write_staged(str(tmp_path / "data"), _DOCS)
    build = build_bm25_index(staged, str(tmp_path / "index"))

    assert build.doc_count == len(_DOCS)
    assert len(build.shards) == 1
    assert build.index_bytes > 0
    assert build.build_seconds >= 0

    index = load_local_index(list(build.shard_dirs), mmap=False)
    hits = index.search("independence of the united states", k=3)
    assert hits, "query returned no hits"
    top = hits[0]
    # Correct document retrieved...
    assert top.metadata["url"] == "http://example.com/usa"
    # ...and its provenance metadata round-tripped.
    assert top.metadata["warc_record_id"] == "rec-" + "http://example.com/usa"[-5:]
    assert "doc_id" in top.metadata
    assert top.score > 0


@pytest.mark.parametrize(
    "query,expected_url",
    [
        ("brown fox", "http://example.com/fox"),
        ("bamboo red panda", "http://example.com/panda"),
        ("machine learning programming", "http://example.com/python"),
        ("sunlight chloroplasts energy", "http://example.com/plants"),
    ],
)
def test_search_ranks_the_relevant_doc_first(tmp_path, query, expected_url):
    staged = _write_staged(str(tmp_path / "data"), _DOCS)
    build = build_bm25_index(staged, str(tmp_path / "index"))
    index = load_local_index(list(build.shard_dirs), mmap=False)
    hits = index.search(query, k=1)
    assert hits[0].metadata["url"] == expected_url


def test_multishard_build_merges_across_sub_indices(tmp_path):
    # Force one sub-index per staged file (budget below a single file's size), so
    # the query path must merge across independently-built bm25s indices.
    staged = _write_staged(str(tmp_path / "data"), _DOCS, shards=5)
    build = build_bm25_index(staged, str(tmp_path / "index"), shard_bytes=1)
    assert len(build.shards) == 5
    assert build.doc_count == len(_DOCS)
    # doc_ids stay globally unique and contiguous across sub-indices.
    index = load_local_index(list(build.shard_dirs), mmap=False)
    hits = index.search("united states independence", k=5)
    assert hits[0].metadata["url"] == "http://example.com/usa"
    assert len({h.metadata["doc_id"] for h in hits}) == len(hits)


def test_smoke_test_reports_efficiency_metrics(tmp_path):
    docs = [*_DOCS, ("the united states government sits in washington", "http://example.com/gov")]
    staged = _write_staged(str(tmp_path / "data"), docs)
    build = build_bm25_index(staged, str(tmp_path / "index"))
    report = smoke_test_index(list(build.shard_dirs), doc_count=build.doc_count)
    assert report["num_hits"] >= 1
    assert report["top_url"] is not None
    assert report["query_ms"] >= 0
    assert report["doc_count"] == len(docs)
    assert "url" in report["top_metadata_keys"]
