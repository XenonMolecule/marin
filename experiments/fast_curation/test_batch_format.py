# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CPU->TPU on-disk contract."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pyarrow as pa
import pytest

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


def test_pooled_keeplist_marks_unscored_docs_nan(tmp_path):
    """A doc the pooled pre-filter drops is never scored by ModernBERT, so it must carry NaN — and
    Phase C's ``prob >= threshold`` must then exclude it with no special-casing (NaN compares False).
    If this regressed, pooled-dropped docs would silently flow into the final corpus."""
    path = str(tmp_path / "keeplist.parquet")
    doc_ids = ["kept", "pooled_dropped", "mb_dropped"]
    mb = [0.9, float("nan"), 0.01]
    pooled = [0.8, 0.01, 0.8]
    batch_format.write_keeplist(path, doc_ids, mb, pooled_probs=pooled)

    table = batch_format.read_table(path)
    assert table.schema.names == ["doc_id", "modernbert_prob", "pooled_prob"]
    prob_of = dict(zip(table.column("doc_id").to_pylist(), table.column("modernbert_prob").to_pylist(), strict=True))
    threshold = 0.41
    keep = [d for d in doc_ids if prob_of[d] >= threshold]
    assert keep == ["kept"], "only the doc that passed BOTH stages survives"
    assert np.isnan(prob_of["pooled_dropped"]), "pooled-dropped must stay NaN, not 0.0"


def test_keeplist_without_pooled_keeps_the_legacy_schema(tmp_path):
    """The v1-v3 line must be byte-compatible: fastpipe_v3 is live with ~3,602 extracted WARCs."""
    path = str(tmp_path / "legacy.parquet")
    batch_format.write_keeplist(path, ["a", "b"], [0.9, 0.1])
    assert batch_format.read_table(path).schema.names == ["doc_id", "modernbert_prob"]


def test_pooled_kept_schema_is_conditional_and_survives_the_write(tmp_path):
    """`pooled_prob` must reach disk for a pooled spec, and must NOT widen the schema of the live
    v1-v3 line (KEPT_SCHEMA is shared, and Phase C concat_tables would mix schemas)."""
    from experiments.fast_curation.spec import get_spec

    assert batch_format.kept_schema_for(get_spec("fastpipe_v3")) is batch_format.KEPT_SCHEMA
    pooled_schema = batch_format.kept_schema_for(get_spec("lpv11_fastpipe_v1"))
    assert "pooled_prob" in pooled_schema.names

    row = {
        n: v
        for n, v in zip(
            batch_format.KEPT_SCHEMA.names,
            ["d1", "http://x", "h", "CC-MAIN-2020-05", 0.9, "text", [1, 2, 3], 3, 0.8],
            strict=True,
        )
    }
    table = pa.table({**{k: [v] for k, v in row.items()}, "pooled_prob": [0.55]}, schema=pooled_schema)
    path = str(tmp_path / "kept.parquet")
    batch_format.write_kept(path, table, pooled_schema)
    back = batch_format.read_table(path)
    assert back.schema.equals(pooled_schema), "the write must not silently narrow the schema"
    assert back.column("pooled_prob").to_pylist() == pytest.approx([0.55])  # stored float32


def test_drop_chunk_dir_is_safe_and_best_effort(tmp_path, monkeypatch):
    """Chunk cleanup must never fail a WARC that is otherwise complete."""
    # A nonexistent dir is a no-op, not an error.
    assert batch_format.drop_chunk_dir("gs://marin-us-east5/definitely/not/a/real/chunk/dir/xyz") == 0

    class Boom:
        def exists(self, p):
            raise RuntimeError("GCS is having a day")

    monkeypatch.setattr(batch_format.fsspec, "filesystem", lambda *a, **k: Boom())
    assert batch_format.drop_chunk_dir("gs://whatever") == 0, "a GCS failure must not propagate"
