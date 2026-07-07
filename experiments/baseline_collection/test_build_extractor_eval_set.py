# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure per-record logic of the extractor eval-set builder: the
'actually extracted' rule, content-type flags, and body_strip'd record assembly."""

from __future__ import annotations

import gzip
import json

import pytest

from experiments.baseline_collection import build_extractor_eval_set as m
from experiments.baseline_collection.build_extractor_eval_set import (
    _is_extracted,
    collect_big,
    content_flags,
    to_record,
)


def test_is_extracted_rejects_empty_and_abstention():
    assert _is_extracted("real extracted text")
    assert not _is_extracted("")
    assert not _is_extracted(None)
    assert not _is_extracted("  [NO_USEFUL_CONTENT]  ")  # marker, whitespace-padded


def test_content_flags_detect_structure():
    assert content_flags("<TABLE><tr><td>x</td></tr></TABLE>")["has_table"]  # case-insensitive
    assert content_flags("<pre>code</pre>")["has_code"]
    assert content_flags("<ul><li>a</li></ul>")["has_list"]
    plain = content_flags("<p>just a paragraph</p>")
    assert not (plain["has_table"] or plain["has_code"] or plain["has_list"])


def test_to_record_applies_body_strip_and_keeps_gold():
    row = {
        "raw_html": (
            "<html><head><title>HEADMARK</title></head>"
            "<body><table><tr><td>cell</td></tr></table><p>BODYMARK</p>"
            "<script>EVIL()</script></body></html>"
        ),
        "final_output": "GOLD",
        "url": "https://example.com/a",
        "warc_record_id": "abc",
        "snapshot": "CC-MAIN-2017-13",
    }
    rec = to_record(row, "dev")
    assert "BODYMARK" in rec["html"]  # body kept
    assert "HEADMARK" not in rec["html"]  # head dropped by body_strip
    assert "EVIL" not in rec["html"]  # script stripped
    assert rec["final_output"] == "GOLD"  # gold passes through untouched
    assert rec["has_table"] and rec["split"] == "dev"
    assert rec["url"] == "https://example.com/a" and rec["snapshot"] == "CC-MAIN-2017-13"


def test_build_writes_transformed_schema(tmp_path, monkeypatch):
    """End-to-end: build() must write to_record'd records (body_strip'd html +
    flags + split), NOT the raw source rows. Regression guard for the missing
    to_record wiring."""
    shards = [f"shard{i}" for i in range(9)]
    snaps = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]  # 3 snapshots x 3 WARCs
    snap_of = dict(zip(shards, snaps, strict=True))

    def fake_read_shard(path: str) -> list[dict]:
        return [
            {
                "raw_html": (
                    "<html><head><title>HEADMARK</title></head>"
                    f"<body><p>BODY {path} row {j}</p><script>EVIL()</script></body></html>"
                ),
                "final_output": f"gold {path} {j}",
                "url": f"https://example.com/{path}/{j}",
                "warc_record_id": f"{path}-{j}",
                "snapshot": snap_of[path],
            }
            for j in range(2)
        ]

    monkeypatch.setattr(m, "fsspec_glob", lambda _pattern: shards)
    monkeypatch.setattr(m, "_shard_snapshots", lambda _shards: snaps)
    monkeypatch.setattr(m, "read_shard", fake_read_shard)

    m.build(str(tmp_path), n_train=3, n_dev=2, n_test=2, seed=0, k_holdout=1)

    expected_keys = {
        "html",
        "final_output",
        "url",
        "warc_record_id",
        "snapshot",
        "has_table",
        "has_code",
        "has_list",
        "split",
    }
    for split, n in (("train", 3), ("dev", 2), ("test", 2)):
        with gzip.open(tmp_path / f"{split}.jsonl.gz", "rt") as f:
            recs = [json.loads(line) for line in f]
        assert len(recs) == n
        for r in recs:
            assert set(r.keys()) == expected_keys  # transformed, not raw
            assert "raw_html" not in r
            assert "HEADMARK" not in r["html"] and "EVIL" not in r["html"]  # body_strip applied
            assert r["split"] == split

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 3, "dev": 2, "test": 2}


@pytest.fixture
def fake_corpus(monkeypatch):
    """50 fake WARCs = 5 snapshots x 10 WARCs, 4 extracted rows each, with distinct
    warc_record_ids. Enough that small-train consumes only a 32-WARC prefix, leaving
    a disjoint tail for big_train."""
    shards, snaps = [], []
    for s in range(5):
        for w in range(10):
            shards.append(f"snap{s}-warc{w}")
            snaps.append(f"S{s}")
    snap_of = dict(zip(shards, snaps, strict=True))

    def fake_read_shard(path: str) -> list[dict]:
        return [
            {
                "raw_html": f"<html><head><title>H</title></head><body><p>{path} {j}</p></body></html>",
                "final_output": f"gold {path} {j}",
                "url": f"http://{path}/{j}",
                "warc_record_id": f"{path}-{j}",
                "snapshot": snap_of[path],
            }
            for j in range(4)
        ]

    monkeypatch.setattr(m, "fsspec_glob", lambda _pattern: shards)
    monkeypatch.setattr(m, "_shard_snapshots", lambda _shards: snaps)
    monkeypatch.setattr(m, "read_shard", fake_read_shard)
    return shards


def test_collect_big_caps_per_warc_and_hits_target(fake_corpus):
    # 50 shards x 4 rows; cap=2 so each WARC contributes <=2 -> need >=10 WARCs for 20.
    rows, _n_read = collect_big(fake_corpus, target=20, per_warc_cap=2, seed=0)
    assert len(rows) == 20
    per_warc: dict[str, int] = {}
    for r in rows:
        warc = r["warc_record_id"].rsplit("-", 1)[0]
        per_warc[warc] = per_warc.get(warc, 0) + 1
    assert max(per_warc.values()) <= 2  # cap respected
    assert len(per_warc) >= 10  # spans many WARCs (diversity), not a handful


def test_big_train_is_warc_disjoint_from_existing_splits(tmp_path, fake_corpus):
    out = str(tmp_path)
    m.build(out, n_train=5, n_dev=2, n_test=2, seed=0, k_holdout=1)
    m.build_big_train(out, n=8, seed=0, reserve=32, per_warc_cap=10, k_holdout=1)

    def ids(path):
        with gzip.open(path, "rt") as f:
            return {json.loads(line)["warc_record_id"] for line in f}

    big = ids(tmp_path / "big_train.jsonl.gz")
    existing = ids(tmp_path / "train.jsonl.gz") | ids(tmp_path / "dev.jsonl.gz") | ids(tmp_path / "test.jsonl.gz")
    assert len(big) == 8
    assert big.isdisjoint(existing)  # the whole point

    with gzip.open(tmp_path / "big_train.jsonl.gz", "rt") as f:
        rec = json.loads(f.readline())
    assert rec["split"] == "big_train" and "html" in rec and "raw_html" not in rec
