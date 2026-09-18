# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allow-list builder: label attachment is verified row for row against arm A's tables."""

from __future__ import annotations

import gzip
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import generate_id

from experiments.fast_curation import build_swap_allowlist as bsa

DATASET = "a_test"
STEM = "data-00004-of-00010"


def _a_record(i: int, warc_hash: str) -> dict:
    return {
        "doc_id": f"doc-{i}",
        "url": f"http://e.com/{i}",
        "warc_hash": warc_hash,
        "snapshot": "CC-MAIN-2019-09",
        "fasttext_score": 0.1 * i,
        "pooled_prob": 0.5,
        "modernbert_prob": None if i % 2 else 0.7,
        "text": f"document number {i} " * 10,
    }


def _write_fixture(base, records: list[dict], *, native_ids=None, quality_ids=None, n_labels=None):
    shard = base / "deduped" / f"{STEM}.jsonl.gz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(shard, "wt") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    ids = [generate_id(r["text"]) for r in records]
    n = len(records) if n_labels is None else n_labels
    topic = pa.table(
        {
            "id": ids[:n],
            "native_id": (native_ids or [r["doc_id"] for r in records])[:n],
            "url": [r["url"] for r in records][:n],
            "cluster_24": pa.array([i % 24 for i in range(n)], pa.int32()),
            "topic_prob": pa.array([0.9] * n, pa.float32()),
            "token_length": pa.array([100 + i for i in range(n)], pa.int32()),
        }
    )
    quality = pa.table(
        {
            "source": [DATASET] * n,
            "id": (quality_ids or ids)[:n],
            "score": pa.array([0.3] * n, pa.float32()),
            "quality_bucket": pa.array([i % 5 for i in range(n)], pa.int8()),
        }
    )
    for path, table in (
        (bsa.topic_table_path(str(base), DATASET, STEM), topic),
        (bsa.quality_table_path(str(base), DATASET, STEM), quality),
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path)
    return str(shard)


def test_labeled_rows_attach_labels_by_position_and_verify_identity(tmp_path):
    records = [_a_record(i, "w1" if i < 2 else "w2") for i in range(4)]
    shard = _write_fixture(tmp_path, records)
    rows = list(bsa.labeled_rows(shard, str(tmp_path), DATASET))
    assert [r["doc_id"] for r in rows] == ["doc-0", "doc-1", "doc-2", "doc-3"]
    assert [r["cluster_24"] for r in rows] == [0, 1, 2, 3]
    assert [r["quality_bucket"] for r in rows] == [0, 1, 2, 3]
    assert [r["token_length"] for r in rows] == [100, 101, 102, 103]
    assert rows[3] == {**rows[3], "a_shard": STEM, "a_row": 3, "warc_hash": "w2", "doc_id": "doc-3"}
    assert "text" not in rows[0] and "url" not in rows[0]


def test_labeled_rows_rejects_a_native_id_mismatch(tmp_path):
    records = [_a_record(i, "w1") for i in range(3)]
    shard = _write_fixture(tmp_path, records, native_ids=["doc-0", "doc-X", "doc-2"])
    with pytest.raises(RuntimeError, match="native_id"):
        list(bsa.labeled_rows(shard, str(tmp_path), DATASET))


def test_labeled_rows_rejects_a_row_count_mismatch(tmp_path):
    records = [_a_record(i, "w1") for i in range(3)]
    shard = _write_fixture(tmp_path, records, n_labels=2)
    with pytest.raises(RuntimeError, match="co-partitioning"):
        list(bsa.labeled_rows(shard, str(tmp_path), DATASET))


def test_labeled_rows_rejects_a_label_computed_on_other_text(tmp_path):
    records = [_a_record(i, "w1") for i in range(2)]
    shard = _write_fixture(tmp_path, records)
    with gzip.open(shard, "wt") as fh:  # rewrite the second document's text after labelling
        fh.write(json.dumps(records[0]) + "\n")
        fh.write(json.dumps({**records[1], "text": "edited after the labels were computed"}) + "\n")
    with pytest.raises(RuntimeError, match="content hash"):
        list(bsa.labeled_rows(shard, str(tmp_path), DATASET))


def test_labeled_rows_rejects_topic_quality_id_disagreement(tmp_path):
    records = [_a_record(i, "w1") for i in range(2)]
    shard = _write_fixture(tmp_path, records, quality_ids=["0" * 32, "1" * 32])
    with pytest.raises(RuntimeError, match="topic/quality id mismatch"):
        list(bsa.labeled_rows(shard, str(tmp_path), DATASET))


def test_provenance_rows_need_no_label_tables(tmp_path):
    records = [_a_record(i, "w1") for i in range(3)]
    shard = tmp_path / "deduped" / f"{STEM}.jsonl.gz"
    shard.parent.mkdir(parents=True)
    with gzip.open(shard, "wt") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    rows = list(bsa.provenance_rows(str(shard)))
    assert [r["doc_id"] for r in rows] == ["doc-0", "doc-1", "doc-2"]
    assert rows[2] == {**rows[2], "a_shard": STEM, "a_row": 2, "url": "http://e.com/2", "modernbert_prob": 0.7}
    assert "text" not in rows[0] and "cluster_24" not in rows[0]


def test_write_warc_shard_round_trips_both_schemas(tmp_path):
    records = [_a_record(i, "w1") for i in range(2)]
    shard = _write_fixture(tmp_path, records)
    root = str(tmp_path / "root")
    prov = list(bsa.provenance_rows(shard))
    summary = bsa.write_warc_shard("w1", iter(prov), root, bsa.Stage.ALLOWLIST, bsa.ALLOWLIST_SCHEMA)
    assert summary == {"warc_hash": "w1", "n_rows": 2}
    table = pq.read_table(bsa.shard_output(root, bsa.Stage.ALLOWLIST, "w1"))
    assert table.schema.equals(bsa.ALLOWLIST_SCHEMA)
    assert table.column("doc_id").to_pylist() == ["doc-0", "doc-1"]
    assert table.column("modernbert_prob").to_pylist() == [pytest.approx(0.7), None]

    labels = list(bsa.labeled_rows(shard, str(tmp_path), DATASET))
    bsa.write_warc_shard("w1", iter(labels), root, bsa.Stage.LABELS, bsa.LABELS_SCHEMA)
    table = pq.read_table(bsa.shard_output(root, bsa.Stage.LABELS, "w1"))
    assert table.schema.equals(bsa.LABELS_SCHEMA)
    assert table.column("quality_bucket").to_pylist() == [0, 1]


def test_write_warc_shard_keeps_an_existing_file(tmp_path):
    records = [_a_record(i, "w1") for i in range(2)]
    shard = _write_fixture(tmp_path, records)
    root = str(tmp_path / "root")
    prov = list(bsa.provenance_rows(shard))
    bsa.write_warc_shard("w1", iter(prov), root, bsa.Stage.ALLOWLIST, bsa.ALLOWLIST_SCHEMA)
    path = bsa.shard_output(root, bsa.Stage.ALLOWLIST, "w1")
    before = open(path, "rb").read()
    summary = bsa.write_warc_shard("w1", iter(prov[:1]), root, bsa.Stage.ALLOWLIST, bsa.ALLOWLIST_SCHEMA)
    assert summary["n_rows"] == 1  # the count reflects this run's rows ...
    assert open(path, "rb").read() == before  # ... but the existing file is not rewritten
