# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure helpers in build_hq_distill_dataset.

These cover the join/dedup logic that, if wrong, would silently corrupt the
dataset: content-id reproduction, reasoning extraction, record-id normalization,
manifest path remapping, and the per-WARC HTML join reducer.
"""

from __future__ import annotations

import pytest
from marin.datakit.normalize import generate_id

from experiments.baseline_collection.build_hq_distill_dataset import (
    MAX_FED_HTML_CHARS,
    NO_USEFUL_MARKER,
    _html_join,
    _join_metadata_file,
    _negatives_for_metadata_file,
    _warc_to_metadata,
    content_id,
    normalize_record_id,
    parse_reasoning,
    remap_to_consolidated,
    warc_hash_from_path,
)


def test_content_id_matches_generate_id_for_short_text():
    # No long whitespace run -> identical to a plain content hash.
    text = "Better late than never, I guess."
    assert content_id(text) == generate_id(text)


def test_content_id_compacts_long_whitespace_then_hashes():
    # A whitespace run > 128 chars is compacted to 128 before hashing, so the
    # id equals generate_id(compacted) and differs from generate_id(raw).
    raw = "a" + (" " * 200) + "b"
    compacted = "a" + (" " * 128) + "b"
    assert content_id(raw) == generate_id(compacted)
    assert content_id(raw) != generate_id(raw)


def test_content_id_keeps_short_whitespace_runs():
    text = "x" + (" " * 128) + "y"  # exactly at the cap -> untouched
    assert content_id(text) == generate_id(text)


def test_parse_reasoning_extracts_think_block():
    gt = "<think>\nWe weigh the options.\n</think>\n[[ ## text ## ]]\nThe answer."
    assert parse_reasoning(gt) == "We weigh the options."


def test_parse_reasoning_unclosed_think_falls_back_to_remainder():
    gt = "<think>\nTruncated mid-thought"
    assert parse_reasoning(gt) == "Truncated mid-thought"


def test_parse_reasoning_no_think_block_is_empty():
    assert parse_reasoning("Just an answer, no reasoning.") == ""
    assert parse_reasoning("") == ""


def test_parse_reasoning_first_block_only():
    gt = "<think>first</think>middle<think>second</think>"
    assert parse_reasoning(gt) == "first"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<urn:uuid:184ebb46-fdc3-47b1-b019-086ade1cccca>", "184ebb46-fdc3-47b1-b019-086ade1cccca"),
        ("184ebb46-fdc3-47b1-b019-086ade1cccca", "184ebb46-fdc3-47b1-b019-086ade1cccca"),
        ("  <urn:uuid:abc>  ", "abc"),
    ],
)
def test_normalize_record_id(raw, expected):
    assert normalize_record_id(raw) == expected


@pytest.mark.parametrize(
    "src,expected",
    [
        (
            "gs://marin-eu-west4/documents/baseline_llm_extraction/high_quality/data-03c97ce9fd4e/batch_0009.jsonl.gz",
            "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/europe-west4/"
            "high_quality/data-03c97ce9fd4e/batch_0009.jsonl.gz",
        ),
        (
            "gs://marin-us-east5/documents/baseline_llm_extraction/high_quality/data-000c699c170b/batch_0074.jsonl.gz",
            "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/us-east5/"
            "high_quality/data-000c699c170b/batch_0074.jsonl.gz",
        ),
    ],
)
def test_remap_to_consolidated(src, expected):
    assert remap_to_consolidated(src) == expected


def test_remap_to_consolidated_rejects_unknown_bucket():
    with pytest.raises(ValueError):
        remap_to_consolidated("gs://marin-mars-1/documents/baseline_llm_extraction/high_quality/data-x/batch_0.jsonl.gz")


def test_warc_hash_from_path():
    p = (
        "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/europe-west4/"
        "high_quality/data-03c97ce9fd4e/batch_0009.jsonl.gz"
    )
    assert warc_hash_from_path(p) == "03c97ce9fd4e"


def test_html_join_attaches_html_by_normalized_id(monkeypatch):
    # The reducer reads the WARC's HTML shard via load_jsonl; stub it.
    html_records = [
        {"id": "<urn:uuid:rec-1>", "html": "<html>one</html>"},
        {"id": "<urn:uuid:rec-2>", "html": "<html>two</html>"},
        {"id": "<urn:uuid:rec-unused>", "html": "<html>nope</html>"},
    ]
    monkeypatch.setattr(
        "experiments.baseline_collection.build_hq_distill_dataset.load_jsonl",
        lambda path: iter(html_records),
    )

    def meta(rid, final):
        return {
            "warc_record_id": rid,
            "reasoning_trace": f"reason-{rid}",
            "final_output": final,
            "url": f"http://x/{rid}",
            "warc_file": "s3://cc/w.warc.gz",
            "snapshot": "CC-MAIN-2015-18",
            "warc_hash": "0007d933a8ed",
        }

    rows = [meta("rec-1", "doc one"), meta("rec-2", "doc two")]
    out = sorted(_html_join("0007d933a8ed", iter(rows)), key=lambda r: r["warc_record_id"])

    assert [r["raw_html"] for r in out] == ["<html>one</html>", "<html>two</html>"]
    assert out[0]["final_output"] == "doc one"
    assert out[0]["reasoning_trace"] == "reason-rec-1"
    assert set(out[0].keys()) == {
        "raw_html",
        "reasoning_trace",
        "final_output",
        "url",
        "warc_file",
        "warc_record_id",
        "snapshot",
    }


def test_html_join_emits_null_html_when_record_missing(monkeypatch):
    monkeypatch.setattr(
        "experiments.baseline_collection.build_hq_distill_dataset.load_jsonl",
        lambda path: iter([{"id": "<urn:uuid:other>", "html": "<html/>"}]),
    )
    rows = [
        {
            "warc_record_id": "missing",
            "reasoning_trace": "r",
            "final_output": "f",
            "url": "u",
            "warc_file": "w",
            "snapshot": "s",
            "warc_hash": "0007d933a8ed",
        }
    ]
    out = list(_html_join("0007d933a8ed", iter(rows)))
    assert len(out) == 1
    assert out[0]["raw_html"] is None
    assert out[0]["final_output"] == "f"


def test_warc_to_metadata_dedups_within_warc_and_skips_empty(monkeypatch):
    # Two batches; the same text appears twice (exact dup) and one record has
    # empty text. Expect one row per distinct content, empties dropped.
    batches = {
        "b0": [
            {
                "text": "alpha",
                "generated_text": "<think>r1</think>x",
                "url": "u1",
                "warc_record_id": "1",
                "warc_file": "w",
                "snapshot": "s",
            },
            {
                "text": "",
                "generated_text": "<think>r</think>",
                "url": "u2",
                "warc_record_id": "2",
                "warc_file": "w",
                "snapshot": "s",
            },
        ],
        "b1": [
            {
                "text": "alpha",
                "generated_text": "<think>dup</think>x",
                "url": "u3",
                "warc_record_id": "3",
                "warc_file": "w",
                "snapshot": "s",
            },
            {
                "text": "beta",
                "generated_text": "<think>r2</think>y",
                "url": "u4",
                "warc_record_id": "4",
                "warc_file": "w",
                "snapshot": "s",
            },
        ],
    }
    monkeypatch.setattr(
        "experiments.baseline_collection.build_hq_distill_dataset.load_jsonl",
        lambda path: iter(batches[path]),
    )
    out = list(_warc_to_metadata({"warc_hash": "abc123abc123", "paths": ["b0", "b1"]}))
    finals = sorted(r["final_output"] for r in out)
    assert finals == ["alpha", "beta"]  # dup "alpha" collapsed, "" skipped
    assert {r["content_id"] for r in out} == {content_id("alpha"), content_id("beta")}
    assert all(r["warc_hash"] == "abc123abc123" for r in out)


def test_join_metadata_file_uses_warc_hash_from_rows(monkeypatch):
    meta_rows = [
        {
            "warc_record_id": "rec-1",
            "reasoning_trace": "r",
            "final_output": "f",
            "url": "u",
            "warc_file": "w",
            "snapshot": "s",
            "warc_hash": "feedfacefeed",
        },
    ]
    html_rows = [{"id": "<urn:uuid:rec-1>", "html": "<html>joined</html>"}]

    def fake_load(path):
        return iter(html_rows if path.endswith("data-feedfacefeed.jsonl.gz") else meta_rows)

    monkeypatch.setattr("experiments.baseline_collection.build_hq_distill_dataset.load_jsonl", fake_load)
    out = list(_join_metadata_file("gs://meta/data-00001.jsonl.gz"))
    assert len(out) == 1
    assert out[0]["raw_html"] == "<html>joined</html>"
    assert "warc_hash" not in out[0]  # stripped from final schema


def test_negatives_excludes_kept_toolong_and_dup_html(monkeypatch):
    meta_rows = [
        {
            "warc_record_id": "kept-1",
            "warc_hash": "abc123abc123",
            "warc_file": "s3://cc/w.warc.gz",
            "snapshot": "CC-MAIN-2016-07",
        },
    ]
    html_rows = [
        {"id": "<urn:uuid:kept-1>", "html": "<html>kept</html>", "url": "u-kept", "metadata": {"warc_file": "wf"}},
        {"id": "<urn:uuid:neg-1>", "html": "<html>abstained-A</html>", "url": "u1", "metadata": {"warc_file": "wf"}},
        {"id": "<urn:uuid:neg-2>", "html": "<html>abstained-A</html>", "url": "u2", "metadata": {"warc_file": "wf"}},
        {"id": "<urn:uuid:neg-3>", "html": "x" * (MAX_FED_HTML_CHARS + 1), "url": "u3", "metadata": {"warc_file": "wf"}},
        {"id": "<urn:uuid:neg-4>", "html": "<html>abstained-B</html>", "url": "u4", "metadata": {"warc_file": "wf"}},
    ]

    def fake_load(path):
        return iter(html_rows if path.endswith("data-abc123abc123.jsonl.gz") else meta_rows)

    monkeypatch.setattr("experiments.baseline_collection.build_hq_distill_dataset.load_jsonl", fake_load)
    out = list(_negatives_for_metadata_file("gs://meta/data-00000.jsonl.gz"))

    # kept-1 excluded (positive); neg-2 excluded (dup html of neg-1); neg-3 excluded (too long).
    urls = sorted(r["url"] for r in out)
    assert urls == ["u1", "u4"]
    assert all(r["final_output"] == NO_USEFUL_MARKER for r in out)
    assert all(r["reasoning_trace"] == "" for r in out)
    assert all(r["snapshot"] == "CC-MAIN-2016-07" for r in out)
    assert set(out[0].keys()) == {
        "raw_html",
        "reasoning_trace",
        "final_output",
        "url",
        "warc_file",
        "warc_record_id",
        "snapshot",
    }


def test_html_join_missing_shard_yields_null_html(monkeypatch):
    def _raise(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(
        "experiments.baseline_collection.build_hq_distill_dataset.load_jsonl",
        _raise,
    )
    rows = [
        {
            "warc_record_id": "rec-1",
            "reasoning_trace": "r",
            "final_output": "f",
            "url": "u",
            "warc_file": "w",
            "snapshot": "s",
            "warc_hash": "deadbeefcafe",
        }
    ]
    out = list(_html_join("deadbeefcafe", iter(rows)))
    assert len(out) == 1 and out[0]["raw_html"] is None
