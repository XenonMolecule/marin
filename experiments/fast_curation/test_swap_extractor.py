# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Arm B swap worker: extraction call, doc selection, and the co-partitioned outputs."""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import generate_id

from experiments.baseline_collection import grid_corpora
from experiments.baseline_collection.grid_corpora import Stage, done_marker, stage_shard_output
from experiments.datakit.store.datakit_store import _ids_match, _load_cluster_table, _load_quality_table
from experiments.fast_curation import swap_extractor
from experiments.fast_curation.swap_extractor import (
    SwapResult,
    swap_records,
    write_attribute_tables,
    write_record_shard,
)
from experiments.fast_curation.upstream_resiliparse import MIN_TEXT_CHARS, upstream_resiliparse_text

BODY = (
    "This is the main article body with several informative sentences that should survive "
    "boilerplate removal because it is the real content of the page and it is long enough."
)
PAGE = (
    "<html><head><title>T</title></head><body>"
    "<nav><a href='/'>home</a> <a href='/about'>about</a> <a href='/contact'>contact</a></nav>"
    f"<main><h1>Real Heading</h1><p>{BODY}</p><img src='x.png' alt='ALTTEXTMARKER'>"
    "<noscript>NOSCRIPTMARKER</noscript></main>"
    "<footer>copyright footer links</footer></body></html>"
)
LABELS = [f"topic_{i}" for i in range(grid_corpora.NUM_TOPICS)]


def _allow_row(doc_id: str, **overrides) -> dict:
    row = {
        "doc_id": doc_id,
        "url": f"http://example.com/{doc_id}",
        "warc_hash": "abc123def456",
        "snapshot": "CC-MAIN-2020-05",
        "fasttext_score": 0.5,
        "pooled_prob": 0.9,
        "modernbert_prob": None,
        "cluster_24": 3,
        "topic_prob": 0.8,
        "token_length": 120,
        "quality_score": 0.55,
        "quality_bucket": 2,
        "a_shard": "data-00001-of-03042",
        "a_row": 7,
    }
    row.update(overrides)
    return row


def test_upstream_extraction_pins_the_baseline_call():
    """Main content only, alt texts off, noscript off, stripped -- the resiliparse_10k arm's call."""
    text = upstream_resiliparse_text(PAGE)
    assert "Real Heading" in text
    assert BODY.split(".")[0] in text
    assert "ALTTEXTMARKER" not in text
    assert "NOSCRIPTMARKER" not in text
    assert text == text.strip()
    assert upstream_resiliparse_text("") == ""


def test_swap_records_selects_by_doc_id_and_reports_short_and_missing():
    allow = [_allow_row("d1"), _allow_row("d2"), _allow_row("d3")]
    chunks = [
        [{"doc_id": "d2", "html": "<p>short</p>"}, {"doc_id": "extra", "html": "<p>not requested</p>"}],
        [{"doc_id": "d1", "html": "<p>long</p>"}],
    ]
    texts = {"<p>short</p>": "x" * (MIN_TEXT_CHARS - 1), "<p>long</p>": "y" * MIN_TEXT_CHARS}
    refreshes = []
    rows, result = swap_records(
        allow, chunks, lambda htmls: [texts[h] for h in htmls], refresh=lambda: refreshes.append(1)
    )
    assert [r["doc_id"] for r in rows] == ["d1"]
    assert rows[0]["text"] == "y" * MIN_TEXT_CHARS
    assert rows[0]["cluster_24"] == 3 and rows[0]["a_row"] == 7
    assert result == SwapResult(n_requested=3, n_written=1, n_short=1, missing=["d3"])
    assert len(refreshes) == 2  # once per chunk that held a hit


def test_swap_records_never_calls_extract_without_hits():
    calls = []

    def extract(htmls):
        calls.append(len(htmls))
        return ["z" * MIN_TEXT_CHARS] * len(htmls)

    rows, result = swap_records([_allow_row("d1")], [[{"doc_id": "other", "html": ""}]], extract)
    assert rows == [] and calls == [] and result.missing == ["d1"]


def test_record_shard_and_attribute_tables_round_trip_through_the_store_join(tmp_path, monkeypatch):
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path))
    dataset = "swap_test"
    shard = str(tmp_path / "deduped" / "data-abc123def456.jsonl.gz")
    rows = [
        {**_allow_row("d1", cluster_24=3, quality_bucket=2, quality_score=0.55), "text": "alpha " * 20},
        {**_allow_row("d2", cluster_24=11, quality_bucket=4, quality_score=0.95), "text": "beta " * 20},
    ]
    write_record_shard(shard, rows)
    write_attribute_tables(dataset, shard, rows, LABELS)

    with gzip.open(shard, "rt") as fh:
        records = [json.loads(line) for line in fh]
    assert [r["doc_id"] for r in records] == ["d1", "d2"]
    assert set(records[0]) == set(swap_extractor.RECORD_COLUMNS) | {"text"}
    assert records[0]["modernbert_prob"] is None

    cluster_ids, clusters = _load_cluster_table(stage_shard_output(dataset, Stage.TOPIC, shard), "cluster_24")
    quality_ids, buckets = _load_quality_table(stage_shard_output(dataset, Stage.QUALITY, shard))
    assert _ids_match(cluster_ids, quality_ids)
    assert clusters.tolist() == [3, 11] and buckets.tolist() == [2, 4]
    # The id is the content hash of B's text, which is what grid_tokenize will compute.
    assert cluster_ids.to_pylist() == [generate_id(r["text"]) for r in rows]
    for stage in Stage:
        with open(done_marker(dataset, stage, shard)) as fh:
            assert fh.read() == "2"
    with open(grid_corpora.counts_output(dataset, Stage.TOPIC, shard)) as fh:
        tally = json.load(fh)
    assert tally["n_docs"] == 2 and tally["counts"]["topic_3"] == 1 and tally["counts"]["topic_11"] == 1
    assert tally["n_tokens"] == 240
    with open(grid_corpora.counts_output(dataset, Stage.QUALITY, shard)) as fh:
        assert json.load(fh)["bucket_counts"] == [0, 0, 1, 0, 1]


def test_attribute_tables_handle_an_empty_warc(tmp_path, monkeypatch):
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path))
    shard = str(tmp_path / "deduped" / "data-empty.jsonl.gz")
    write_record_shard(shard, [])
    write_attribute_tables("swap_test", shard, [], LABELS)
    cluster_ids, _ = _load_cluster_table(stage_shard_output("swap_test", Stage.TOPIC, shard), "cluster_24")
    quality_ids, _ = _load_quality_table(stage_shard_output("swap_test", Stage.QUALITY, shard))
    assert len(cluster_ids) == 0 and _ids_match(cluster_ids, quality_ids)
    assert os.path.exists(shard)


def test_swap_one_warc_writes_catalog_and_skips_warcs_without_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path))
    root = str(tmp_path / "root")
    allow_dir = tmp_path / "root" / "allowlist"
    allow_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_allow_row("d1"), _allow_row("d9")]), allow_dir / "data-abc123def456.parquet")
    decoded = [[{"doc_id": "d1", "html": "<p>x</p>"}]]
    monkeypatch.setattr(swap_extractor, "_decode_one_warc_stream", lambda warc_path, chunk_records: iter(decoded))
    extract = lambda htmls: ["w" * MIN_TEXT_CHARS] * len(htmls)  # noqa: E731

    catalog = swap_extractor.swap_one_warc("s3://x/w.warc.gz", "abc123def456", root, extract, lambda: None)
    assert catalog["n_written"] == 1 and catalog["missing"] == ["d9"] and catalog["skipped"] is False
    with gzip.open(tmp_path / "root" / "deduped" / "data-abc123def456.jsonl.gz", "rt") as fh:
        assert [json.loads(line)["doc_id"] for line in fh] == ["d1"]
    with open(tmp_path / "root" / "catalog" / "data-abc123def456.json") as fh:
        assert json.load(fh)["missing"] == ["d9"]

    catalog = swap_extractor.swap_one_warc("s3://x/none.warc.gz", "nothere00000", root, extract, lambda: None)
    assert catalog["skipped"] is True and catalog["n_requested"] == 0
    assert not os.path.exists(tmp_path / "root" / "deduped" / "data-nothere00000.jsonl.gz")


@pytest.mark.parametrize("n", [0, 1, 3])
def test_bucket_counts_match_written_rows(tmp_path, monkeypatch, n):
    monkeypatch.setattr(grid_corpora, "OUTPUT_BASE", str(tmp_path))
    shard = str(tmp_path / "deduped" / f"data-{n}.jsonl.gz")
    rows = [{**_allow_row(f"d{i}", quality_bucket=i % 5), "text": f"doc {i} " * 20} for i in range(n)]
    write_attribute_tables("swap_test", shard, rows, LABELS)
    with open(grid_corpora.counts_output("swap_test", Stage.QUALITY, shard)) as fh:
        assert sum(json.load(fh)["bucket_counts"]) == n
    assert int(np.sum([1 for _ in rows])) == n
