# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the provenance recovery join and per-shard I/O."""

import gzip
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import generate_id

from experiments.fast_curation.recover_provenance import (
    _ids,
    _pass_of,
    compact_whitespace,
    content_ids,
    count_shard,
    ids_deduped_shard,
    index_kept_shard,
    match_survivors,
    materialize_shard,
)


def test_content_ids_match_datakit_generate_id():
    texts = ["hello", "", "über 🙂", "x" * 10_000]
    hi, lo = content_ids(texts)
    for t, h, lo_ in zip(texts, hi, lo, strict=True):
        assert format((int(h) << 64) | int(lo_), "032x") == generate_id(t)


def test_compact_whitespace_matches_normalize_cap():
    text = "a" + " " * 200 + "b" + "\n" * 128 + "c"
    out = compact_whitespace(text)
    assert out == "a" + " " * 128 + "b" + "\n" * 128 + "c"
    assert compact_whitespace(out) == out


def test_pass_of_partitions_every_id_into_range():
    hi = np.array([0, 2**64 - 1, 2**60, 2**63], dtype=np.uint64)
    assert _pass_of(hi, 16).tolist() == [0, 15, 1, 8]
    assert _pass_of(hi, 1).tolist() == [0, 0, 0, 0]


def test_match_survivors_picks_lowest_copy_and_reports_missing():
    # kept: id A appears in shard 2 row 0 and shard 0 row 5 (exact dup); id B only in shard 1 row 3.
    a, b, c = 11, 22, 33
    kept_ids = _ids(np.array([a, a, b], dtype=np.uint64), np.array([1, 1, 2], dtype=np.uint64))
    kept_shard = np.array([2, 0, 1], dtype=np.uint32)
    kept_row = np.array([0, 5, 3], dtype=np.uint32)
    surv = np.sort(_ids(np.array([a, b, c], dtype=np.uint64), np.array([1, 2, 3], dtype=np.uint64)))
    shard, row, unmatched = match_survivors(kept_ids, kept_shard, kept_row, surv)
    assert unmatched == 1  # c is not in kept
    assert sorted(zip(shard.tolist(), row.tolist(), strict=True)) == [(0, 5), (1, 3)]


def test_index_materialize_roundtrip_carries_every_column(tmp_path):
    kept = pa.table(
        {
            "doc_id": ["d0", "d1", "d2"],
            "url": ["http://a", "http://b", "http://c"],
            "warc_hash": ["w"] * 3,
            "text": ["alpha", "", "gamma" + " " * 300 + "delta"],
            "modernbert_prob": pa.array([0.1, 0.2, 0.3], type=pa.float32()),
        }
    )
    kept_path = tmp_path / "data-w.parquet"
    pq.write_table(kept, kept_path)

    # Empty-text rows are never candidates, so the index skips row 1 but keeps real row numbers.
    idx_path = tmp_path / "idx.parquet"
    assert index_kept_shard(str(kept_path), str(idx_path)) == 2
    assert pq.read_table(idx_path).column("row").to_pylist() == [0, 2]

    dedup_path = tmp_path / "data-00000-of-00001.jsonl.gz"
    with gzip.open(dedup_path, "wt") as fh:
        # The deduped tree holds normalize's compacted text.
        fh.write(json.dumps({"text": "gamma" + " " * 128 + "delta", "modernbert_prob": 0.3}) + "\n")
    ids_path = tmp_path / "ids.parquet"
    assert ids_deduped_shard(str(dedup_path), str(ids_path)) == 1
    hi, _ = content_ids(["gamma" + " " * 128 + "delta"])
    assert pq.read_table(ids_path).column("hi").to_pylist() == [int(hi[0])]
    # The kept index hashes the compacted text too, so the two sides agree on this doc's id.
    assert pq.read_table(idx_path).column("hi").to_pylist()[1] == int(hi[0])

    keeplist = tmp_path / "keep.parquet"
    pq.write_table(pa.table({"row": pa.array([2], type=pa.uint32())}), keeplist)
    out = tmp_path / "out.jsonl.gz"
    assert materialize_shard(str(kept_path), str(out), str(keeplist)) == 1
    with gzip.open(out, "rt") as fh:
        records = [json.loads(line) for line in fh]
    assert len(records) == 1
    record = records[0]
    assert {k: v for k, v in record.items() if k != "modernbert_prob"} == {
        "doc_id": "d2",
        "url": "http://c",
        "warc_hash": "w",
        "text": "gamma" + " " * 128 + "delta",
    }
    assert abs(record["modernbert_prob"] - 0.3) < 1e-6
    counts = tmp_path / "count"
    assert count_shard(str(out), str(counts)) == 1
