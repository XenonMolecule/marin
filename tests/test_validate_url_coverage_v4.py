# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the resumable URL coverage validator (v4)."""

import gzip
import json
from unittest import mock

from experiments.baseline_collection.validate_url_coverage_v4 import (
    _aggregate_from_checkpoints,
    _checkpoint_name_for_nemotron,
    _scan_one_cdx,
)


def test_checkpoint_name_includes_partition_to_avoid_collisions():
    """The same part filename can appear under different quality/kind partitions —
    the checkpoint name must encode partition info so two files don't overwrite."""
    p1 = (
        "gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/"
        "quality=high/kind=actual/kind2=actual/CC-MAIN-2013-20-part-00000.jsonl.gz"
    )
    p2 = (
        "gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/"
        "quality=high/kind=synthetic/kind2=distill/CC-MAIN-2013-20-part-00000.jsonl.gz"
    )
    assert _checkpoint_name_for_nemotron(p1) != _checkpoint_name_for_nemotron(p2)
    assert "high__actual__actual" in _checkpoint_name_for_nemotron(p1)
    assert "high__synthetic__distill" in _checkpoint_name_for_nemotron(p2)


def _make_fake_cdx_bytes(records: list[dict]) -> bytes:
    lines = []
    for r in records:
        surt = "com,example)/"
        ts = "20130101000000"
        lines.append(f"{surt} {ts} {json.dumps(r)}\n".encode())
    return gzip.compress(b"".join(lines))


def test_scan_one_cdx_counts_hits():
    """v4 scanner has the same contract as v3: returns one chunk dict with scanned + hits."""
    nemotron_set = {"http://x/want1", "http://x/want2"}
    cdx_records = [
        {"url": "http://x/want1", "filename": "x.warc.gz"},
        {"url": "http://x/random", "filename": "x.warc.gz"},
        {"url": "http://x/want2", "filename": "x.warc.gz"},
    ]
    fake_body = _make_fake_cdx_bytes(cdx_records)

    worker_ctx = mock.MagicMock()
    worker_ctx.get_shared.return_value = nemotron_set

    with (
        mock.patch(
            "experiments.baseline_collection.validate_url_coverage_v4._http_get_bytes_with_retry",
            return_value=fake_body,
        ),
        mock.patch(
            "experiments.baseline_collection.validate_url_coverage_v4.zephyr_worker_ctx",
            return_value=worker_ctx,
        ),
    ):
        out = _scan_one_cdx({"cdx_rel": "fake/cdx-00000.gz"})

    assert len(out) == 1
    assert out[0]["scanned"] == 3
    assert sorted(out[0]["hits"]) == ["http://x/want1", "http://x/want2"]


def test_aggregate_from_checkpoints(tmp_path):
    """Aggregation reads back per-CDX hit shards (the format _scan_one_cdx writes via flat_map → write_jsonl)."""
    # Write two fake hit shards locally with the format that flat_map → write_jsonl produces:
    # one JSONL line per chunk, each containing {scanned, hits}
    chunks_a = [{"scanned": 100, "hits": ["http://x/a", "http://x/b"]}]
    chunks_b = [{"scanned": 200, "hits": ["http://x/b", "http://x/c"]}]  # 'b' overlaps

    shard_a = tmp_path / "data-00000-of-00002.jsonl.gz"
    shard_b = tmp_path / "data-00001-of-00002.jsonl.gz"

    def _write(path, chunks):
        with gzip.open(path, "wt") as gz:
            for c in chunks:
                gz.write(json.dumps(c) + "\n")

    _write(shard_a, chunks_a)
    _write(shard_b, chunks_b)

    # Mock fsspec to return our local files. Use file:// scheme with a real tmp_path.
    matched, total = _aggregate_from_checkpoints(f"file://{tmp_path}", nemotron_urls_size=10)

    # Set semantics: 'b' counted once. Total scanned sums.
    assert matched == {"http://x/a", "http://x/b", "http://x/c"}
    assert total == 300
