# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label attach: B's tables carry A's labels for exactly B's records, and disagreements are fatal."""

from __future__ import annotations

import gzip
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import generate_id

from experiments.baseline_collection import grid_corpora
from experiments.baseline_collection.grid_corpora import Stage, stage_shard_output
from experiments.datakit.store.datakit_store import _load_cluster_table, _load_quality_table
from experiments.fast_curation import attach_swap_labels as asl

LABELS = [f"topic_{i}" for i in range(grid_corpora.NUM_TOPICS)]
H = "abc123def456"


def _label(doc_id: str, cluster: int, bucket: int) -> dict:
    return {
        "doc_id": doc_id,
        "warc_hash": H,
        "cluster_24": cluster,
        "topic_prob": 0.9,
        "token_length": 50,
        "quality_score": 0.5,
        "quality_bucket": bucket,
        "a_shard": "data-00000-of-00001",
        "a_row": 0,
    }


def _record(doc_id: str) -> dict:
    return {"doc_id": doc_id, "url": f"http://e.com/{doc_id}", "warc_hash": H, "text": f"text of {doc_id} " * 5}


def _populate(tmp_path, records, labels, catalog):
    root = str(tmp_path / "root")
    os.makedirs(f"{root}/deduped")
    os.makedirs(f"{root}/labels")
    os.makedirs(f"{root}/catalog")
    with gzip.open(f"{root}/deduped/data-{H}.jsonl.gz", "wt") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    pq.write_table(pa.Table.from_pylist(labels), f"{root}/labels/data-{H}.parquet")
    with open(f"{root}/catalog/data-{H}.json", "w") as fh:
        json.dump({"warc_hash": H, "n_requested": len(labels), "skipped": False, **catalog}, fh)
    return root


def test_attach_writes_tables_in_shard_order_with_b_content_ids(tmp_path):
    records = [_record("d2"), _record("d1")]
    labels = [_label("d1", 4, 1), _label("d2", 7, 3), _label("d3", 0, 0)]  # d3 was dropped as short
    root = _populate(tmp_path, records, labels, {"n_written": 2, "n_short": 1, "missing": []})
    summary = asl.attach_one(H, root, "swap_test", str(tmp_path), LABELS)
    assert summary == {"warc_hash": H, "n_rows": 2, "n_unused_labels": 1}
    shard = f"{root}/deduped/data-{H}.jsonl.gz"
    ids, clusters = _load_cluster_table(stage_shard_output("swap_test", Stage.TOPIC, shard), "cluster_24")
    _, buckets = _load_quality_table(stage_shard_output("swap_test", Stage.QUALITY, shard))
    assert clusters.tolist() == [7, 4] and buckets.tolist() == [3, 1]
    assert ids.to_pylist() == [generate_id(r["text"]) for r in records]


def test_attach_rejects_a_record_without_a_label(tmp_path):
    root = _populate(
        tmp_path, [_record("d1"), _record("dX")], [_label("d1", 4, 1)], {"n_written": 2, "n_short": 0, "missing": []}
    )
    with pytest.raises(RuntimeError, match="no label"):
        asl.attach_one(H, root, "swap_test", str(tmp_path), LABELS)


def test_attach_rejects_unexplained_unused_labels(tmp_path):
    labels = [_label("d1", 4, 1), _label("d2", 7, 3)]
    root = _populate(tmp_path, [_record("d1")], labels, {"n_written": 1, "n_short": 0, "missing": []})
    with pytest.raises(RuntimeError, match="disagree"):
        asl.attach_one(H, root, "swap_test", str(tmp_path), LABELS)


def test_pending_skips_attached_and_skipped_warcs(tmp_path):
    root = _populate(tmp_path, [_record("d1")], [_label("d1", 4, 1)], {"n_written": 1, "n_short": 0, "missing": []})
    with open(f"{root}/catalog/data-skip00000000.json", "w") as fh:
        json.dump(
            {
                "warc_hash": "skip00000000",
                "n_requested": 0,
                "n_written": 0,
                "n_short": 0,
                "missing": [],
                "skipped": True,
            },
            fh,
        )
    assert asl._pending_warcs(root, "swap_test", str(tmp_path)) == [H]
    asl.attach_one(H, root, "swap_test", str(tmp_path), LABELS)
    assert asl._pending_warcs(root, "swap_test", str(tmp_path)) == []
