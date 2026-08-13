# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure helpers in build_lpv11_distill_dataset.

Covers the logic that, if wrong, would silently corrupt labels: the
first-occurrence content dedup for positives, the UNdeduped kept-id set used by
the negatives set-difference (the hq build's leak, fixed here), and the
no-length-cap negatives criterion.
"""

from __future__ import annotations

from experiments.baseline_collection.build_lpv11_distill_dataset import (
    NO_USEFUL_MARKER,
    _join_metadata_file,
    _negatives_for_metadata_file,
    _warc_to_metadata,
    dedup_by_content,
    iter_negatives,
)


def _meta_row(cid: str, rid: str, text: str = "some text") -> dict:
    return {
        "content_id": cid,
        "final_output": text,
        "url": f"http://x/{rid}",
        "warc_file": "s3://commoncrawl/crawl-data/CC-MAIN-2019-43/x.warc.gz",
        "warc_record_id": rid,
        "snapshot": "CC-MAIN-2019-43",
        "warc_hash": "deadbeef1234",
        "html_path": "gs://bucket/pool/data-deadbeef1234.jsonl.gz",
    }


def test_dedup_by_content_keeps_first_occurrence():
    rows = [_meta_row("c1", "r1"), _meta_row("c2", "r2"), _meta_row("c1", "r3")]
    kept = dedup_by_content(rows)
    assert [r["warc_record_id"] for r in kept] == ["r1", "r2"]


def test_iter_negatives_no_length_cap_and_kept_excluded():
    huge = "z" * 1_000_000  # would have been length-filtered in the hq build
    records = [
        {"id": "<urn:uuid:kept-1>", "html": "<p>kept page</p>", "url": "u1", "metadata": {"warc_file": "w"}},
        {"id": "<urn:uuid:neg-1>", "html": huge, "url": "u2", "metadata": {"warc_file": "w"}},
        {"id": "<urn:uuid:neg-2>", "html": "", "url": "u3", "metadata": {}},  # empty -> skipped
    ]
    out = list(iter_negatives(records, {"kept-1"}, "wf", "CC-MAIN-2019-43"))
    assert [r["warc_record_id"] for r in out] == ["neg-1"]
    assert out[0]["raw_html"] == huge
    assert out[0]["final_output"] == NO_USEFUL_MARKER
    assert out[0]["reasoning_trace"] == ""


def test_iter_negatives_dedups_identical_html_within_warc():
    records = [
        {"id": "<urn:uuid:a>", "html": "<p>same</p>", "url": "u1", "metadata": {}},
        {"id": "<urn:uuid:b>", "html": "<p>same</p>", "url": "u2", "metadata": {}},
    ]
    out = list(iter_negatives(records, set(), "wf", "snap"))
    assert [r["warc_record_id"] for r in out] == ["a"]


def test_negatives_use_undeduped_kept_ids(monkeypatch):
    """A kept page whose text duplicates another kept page must NOT become a
    negative (the hq build's leak). The metadata file carries both rows; only
    the html-side set difference decides."""
    import experiments.baseline_collection.build_lpv11_distill_dataset as mod

    meta = [_meta_row("c1", "r1"), _meta_row("c1", "r2")]  # r2 = duplicate text of r1
    html = [
        {"id": "<urn:uuid:r1>", "html": "<p>page one</p>", "url": "u1", "metadata": {}},
        {"id": "<urn:uuid:r2>", "html": "<p>page two</p>", "url": "u2", "metadata": {}},
        {"id": "<urn:uuid:r3>", "html": "<p>page three</p>", "url": "u3", "metadata": {}},
    ]

    def fake_load(path):
        return iter(meta) if path.endswith("meta.jsonl.gz") else iter(html)

    monkeypatch.setattr(mod, "load_jsonl", fake_load)
    out = list(_negatives_for_metadata_file("meta.jsonl.gz"))
    assert [r["warc_record_id"] for r in out] == ["r3"]


def test_join_dedups_but_joins_all_first_occurrences(monkeypatch):
    import experiments.baseline_collection.build_lpv11_distill_dataset as mod

    meta = [_meta_row("c1", "r1"), _meta_row("c1", "r2"), _meta_row("c2", "r3")]
    html = [
        {"id": "<urn:uuid:r1>", "html": "<p>one</p>"},
        {"id": "<urn:uuid:r2>", "html": "<p>two</p>"},
        {"id": "<urn:uuid:r3>", "html": "<p>three</p>"},
    ]

    def fake_load(path):
        return iter(meta) if path.endswith("meta.jsonl.gz") else iter(html)

    monkeypatch.setattr(mod, "load_jsonl", fake_load)
    out = list(_join_metadata_file("meta.jsonl.gz"))
    assert {(r["warc_record_id"], r["raw_html"]) for r in out} == {("r1", "<p>one</p>"), ("r3", "<p>three</p>")}
    assert all(r["reasoning_trace"] == "" for r in out)


def test_warc_to_metadata_emits_all_rows_including_dup_text(monkeypatch):
    import experiments.baseline_collection.build_lpv11_distill_dataset as mod

    batch = [
        {"text": "same text", "url": "u1", "warc_record_id": "r1", "warc_file": "wf", "snapshot": "s"},
        {"text": "same text", "url": "u2", "warc_record_id": "r2", "warc_file": "wf", "snapshot": "s"},
        {"text": "", "url": "u3", "warc_record_id": "r3", "warc_file": "wf", "snapshot": "s"},
    ]
    monkeypatch.setattr(mod, "load_jsonl", lambda path: iter(batch))
    warc = {"warc_hash": "deadbeef1234", "paths": ["gs://x/batch_0000.jsonl.gz"], "html_path": "gs://x/h.jsonl.gz"}
    rows = list(_warc_to_metadata(warc))
    assert [r["warc_record_id"] for r in rows] == ["r1", "r2"]  # empty text dropped, dup text KEPT
    assert rows[0]["content_id"] == rows[1]["content_id"]
    assert all(r["html_path"] == "gs://x/h.jsonl.gz" for r in rows)
