# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage gates read the on-disk artifacts and fail on the discrepancies that matter."""

from __future__ import annotations

import json
import os

from experiments.baseline_collection import grid_corpora
from experiments.baseline_collection.grid_corpora import Stage, write_done
from experiments.fast_curation import swap_status


def _write(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(json.dumps(payload) if not isinstance(payload, str) else payload)


def _populate(tmp_path, catalogs: list[dict], meta: dict | None = None):
    root = str(tmp_path / "root")
    grid_corpora.OUTPUT_BASE = str(tmp_path)
    if meta is not None:
        _write(f"{root}/allowlist/_meta.json", meta)
    for c in catalogs:
        h = c["warc_hash"]
        _write(f"{root}/catalog/data-{h}.json", c)
        _write(f"{root}/_completed/data-{h}", "")
        if not c["skipped"]:
            shard = f"{root}/deduped/data-{h}.jsonl.gz"
            _write(shard, "")
            write_done("swap_test", Stage.TOPIC, shard, c["n_written"])
            write_done("swap_test", Stage.QUALITY, shard, c["n_written"])
    return root


def _catalog(h: str, requested: int, written: int, short: int = 0, missing=(), skipped=False) -> dict:
    return {
        "warc_hash": h,
        "n_requested": requested,
        "n_written": written,
        "n_short": short,
        "missing": list(missing),
        "skipped": skipped,
    }


def test_consistent_run_passes_every_gate(tmp_path, monkeypatch):
    meta = {"n_rows": 30, "n_warcs": 2, "n_files": 2}
    root = _populate(
        tmp_path, [_catalog("aaa", 10, 9, short=1), _catalog("bbb", 20, 20), _catalog("ccc", 0, 0, skipped=True)], meta
    )
    summary, failures = swap_status.status(root, str(tmp_path), "swap_test", expect_warcs=3, expect_labels=True)
    assert failures == []
    assert summary["n_requested"] == 30 and summary["n_written"] == 29 and summary["n_short"] == 1
    assert summary["shards"] == 2 and summary["topic_done"] == 2 and summary["skipped_warcs"] == 1


def test_missing_ids_and_incomplete_fleet_fail(tmp_path, monkeypatch):
    meta = {"n_rows": 30, "n_warcs": 2, "n_files": 2}
    root = _populate(tmp_path, [_catalog("aaa", 10, 8, missing=["d1", "d2"])], meta)
    _, failures = swap_status.status(root, str(tmp_path), "swap_test", expect_warcs=3)
    assert any("missing" in f for f in failures)
    assert any("1 of 3 WARCs" in f for f in failures)


def test_shard_without_done_marker_fails(tmp_path, monkeypatch):
    root = _populate(tmp_path, [_catalog("aaa", 10, 10)], {"n_rows": 10, "n_warcs": 1, "n_files": 1})
    os.remove(grid_corpora.done_marker("swap_test", Stage.QUALITY, f"{root}/deduped/data-aaa.jsonl.gz"))
    _, failures = swap_status.status(root, str(tmp_path), "swap_test", expect_warcs=1, expect_labels=True)
    assert failures == ["quality _done: 0 present vs 1 catalogued WARCs with output"]
    _, failures = swap_status.status(root, str(tmp_path), "swap_test", expect_warcs=1)
    assert failures == []  # before attach, missing label tables are not a failure


def test_no_allowlist_is_reported(tmp_path):
    root = _populate(tmp_path, [])
    _, failures = swap_status.status(root, str(tmp_path), "swap_test", expect_warcs=None)
    assert failures == ["allow-list _meta.json missing (build_swap_allowlist has not completed)"]
