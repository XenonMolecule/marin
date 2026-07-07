# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CPU->TPU on-disk contract."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pyarrow as pa

from experiments.fast_curation import batch_format


def _rows():
    return [
        {
            "doc_id": "a",
            "url": "http://a",
            "warc_hash": "deadbeef0001",
            "snapshot": "CC-MAIN-2013-20",
            "fasttext_score": 0.42,
            "text": "hello world",
            "input_ids": [1, 2, 3],
            "n_tokens": 3,
        },
        {
            "doc_id": "b",
            "url": "http://b",
            "warc_hash": "deadbeef0001",
            "snapshot": "CC-MAIN-2013-20",
            "fasttext_score": 0.99,
            "text": "x" * 100000,
            "input_ids": list(range(8192)),
            "n_tokens": 9001,
        },
    ]


def test_survivor_roundtrip(tmp_path):
    path = str(tmp_path / "data-deadbeef0001.parquet")
    batch_format.write_survivors(path, _rows())
    t = batch_format.read_table(path)
    assert t.schema.equals(batch_format.SURVIVOR_SCHEMA)
    assert t.num_rows == 2
    assert t.column("doc_id").to_pylist() == ["a", "b"]
    # ragged int32 token lists preserved exactly
    assert t.column("input_ids").to_pylist()[0] == [1, 2, 3]
    assert len(t.column("input_ids").to_pylist()[1]) == 8192
    # n_tokens records TRUE length even when input_ids was truncation-capped
    assert t.column("n_tokens").to_pylist() == [3, 9001]
    assert t.schema.field("fasttext_score").type == pa.float32()
    assert t.schema.field("input_ids").type == pa.list_(pa.int32())


def test_pad_batch_pads_and_masks():
    ids, seg = batch_format.pad_batch([[1, 2, 3], [4, 5], []], target_len=4, pad_token_id=50283)
    assert ids.dtype == np.int32 and seg.dtype == np.int32
    assert ids.shape == (3, 4) and seg.shape == (3, 4)
    np.testing.assert_array_equal(ids[0], [1, 2, 3, 50283])
    np.testing.assert_array_equal(seg[0], [0, 0, 0, -1])
    np.testing.assert_array_equal(seg[1], [0, 0, -1, -1])
    np.testing.assert_array_equal(seg[2], [-1, -1, -1, -1])  # empty row fully masked


def test_pad_batch_truncates_overlong():
    ids, seg = batch_format.pad_batch([[1, 2, 3, 4, 5]], target_len=3, pad_token_id=0)
    np.testing.assert_array_equal(ids[0], [1, 2, 3])
    np.testing.assert_array_equal(seg[0], [0, 0, 0])


def test_bucket_for():
    assert batch_format.bucket_for(50) == 128
    assert batch_format.bucket_for(700) == 1024
    assert batch_format.bucket_for(8192) == 8192
    assert batch_format.bucket_for(9000) == 8192  # capped at max_length
    assert batch_format.bucket_for(9000, max_length=4096) == 4096


def test_write_tombstones_roundtrip(tmp_path):
    path = str(tmp_path / "tomb.jsonl.gz")
    batch_format.write_tombstones(path, [("a", 0.01), ("b", 0.12345678)])
    with open(path, "rb") as fh:
        lines = gzip.decompress(fh.read()).decode().strip().split("\n")
    recs = [json.loads(x) for x in lines]
    assert recs[0] == {"doc_id": "a", "modernbert_prob": 0.01}
    assert recs[1]["doc_id"] == "b"
    assert abs(recs[1]["modernbert_prob"] - 0.123457) < 1e-6  # rounded to 6 dp


def test_write_kept_has_prob_column(tmp_path):
    survivor_path = str(tmp_path / "surv.parquet")
    batch_format.write_survivors(survivor_path, _rows())
    table = batch_format.read_table(survivor_path)
    probs = np.array([0.5, 0.9], dtype=np.float32)
    kept = table.append_column("modernbert_prob", pa.array(probs, type=pa.float32()))
    kept_path = str(tmp_path / "kept.parquet")
    batch_format.write_kept(kept_path, kept)
    back = batch_format.read_table(kept_path)
    assert back.schema.equals(batch_format.KEPT_SCHEMA)
    np.testing.assert_allclose(back.column("modernbert_prob").to_pylist(), [0.5, 0.9], atol=1e-6)


def _pre_rows():
    return [
        {
            "doc_id": "a",
            "url": "u",
            "warc_hash": "h",
            "snapshot": "s",
            "fasttext_score": 0.5,
            "html": "<p>x</p>",
            "input_ids": [1, 2, 3],
            "n_tokens": 3,
        },
        {
            "doc_id": "b",
            "url": "u2",
            "warc_hash": "h",
            "snapshot": "s",
            "fasttext_score": 0.9,
            "html": "<p>y</p>",
            "input_ids": [4, 5],
            "n_tokens": 2,
        },
    ]


def test_presurvivor_roundtrip_and_projection(tmp_path):
    path = str(tmp_path / "pre.parquet")
    batch_format.write_presurvivors(path, _pre_rows())
    t = batch_format.read_table(path)
    assert t.schema.equals(batch_format.PRESURVIVOR_SCHEMA)
    assert t.column("html").to_pylist() == ["<p>x</p>", "<p>y</p>"]
    # Phase B reads ONLY doc_id + input_ids (column projection avoids loading html).
    proj = batch_format.read_table(path, columns=["doc_id", "input_ids"])
    assert proj.column_names == ["doc_id", "input_ids"]
    assert proj.column("input_ids").to_pylist() == [[1, 2, 3], [4, 5]]


def test_presurvivor_empty(tmp_path):
    path = str(tmp_path / "pre_empty.parquet")
    batch_format.write_presurvivors(path, [])
    t = batch_format.read_table(path)
    assert t.num_rows == 0 and t.schema.equals(batch_format.PRESURVIVOR_SCHEMA)


def test_keeplist_roundtrip(tmp_path):
    path = str(tmp_path / "kl.parquet")
    batch_format.write_keeplist(path, ["a", "b"], [0.30, 0.05])
    t = batch_format.read_table(path)
    assert t.schema.equals(batch_format.KEEPLIST_SCHEMA)
    got = dict(zip(t.column("doc_id").to_pylist(), t.column("modernbert_prob").to_pylist(), strict=True))
    assert abs(got["a"] - 0.30) < 1e-6 and abs(got["b"] - 0.05) < 1e-6
