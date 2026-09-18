# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fast_curation.upstream_resiliparse import MIN_TEXT_CHARS
from experiments.fast_curation.verify_swap_shards import verify_one

H = "abc123def456"


def _populate(tmp_path, records, allowed, catalog):
    root = str(tmp_path / "root")
    for d in ("deduped", "allowlist", "catalog"):
        os.makedirs(f"{root}/{d}")
    with gzip.open(f"{root}/deduped/data-{H}.jsonl.gz", "wt") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    pq.write_table(pa.table({"doc_id": allowed}), f"{root}/allowlist/data-{H}.parquet")
    path = f"{root}/catalog/data-{H}.json"
    with open(path, "w") as fh:
        json.dump({"warc_hash": H, "skipped": False, **catalog}, fh)
    return root, path


def _rec(doc_id, text="x" * MIN_TEXT_CHARS):
    return {"doc_id": doc_id, "text": text}


def test_consistent_shard_passes(tmp_path):
    root, cat = _populate(
        tmp_path, [_rec("d1"), _rec("d2")], ["d1", "d2", "d3"], {"n_written": 2, "n_short": 1, "missing": []}
    )
    assert verify_one(cat, root) == {"warc_hash": H, "ok": True, "n_records": 2, "problems": []}


def test_truncated_shard_is_reported(tmp_path):
    root, cat = _populate(tmp_path, [_rec("d1")], ["d1", "d2"], {"n_written": 2, "n_short": 0, "missing": []})
    result = verify_one(cat, root)
    assert not result["ok"]
    assert any("records 1 != catalog n_written 2" in p for p in result["problems"])


def test_duplicates_and_foreign_ids_are_reported(tmp_path):
    root, cat = _populate(
        tmp_path, [_rec("d1"), _rec("d1"), _rec("zz")], ["d1", "d2"], {"n_written": 3, "n_short": 0, "missing": []}
    )
    problems = verify_one(cat, root)["problems"]
    assert any("duplicate" in p for p in problems)
    assert any("not in the allow-list" in p for p in problems)
