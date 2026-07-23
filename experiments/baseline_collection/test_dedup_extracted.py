# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provenance-preservation tests for the LLM-extraction fuzzy-dedup path.

The dedup pipeline used to collapse every record to ``{text}`` at two points —
the reshape read and the fuzzy-apply write — so the final ``deduped`` tree lost
``url`` / ``warc_record_id`` / ``warc_file`` / ``snapshot``. These tests pin the
fix: provenance columns must survive a reshape → normalize → apply round-trip
(the real ``normalize_to_parquet`` engine passes non-id/text columns through
untouched), while ``text`` stays the sole dedup key and datakit-internal ids do
not leak downstream.
"""

import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fray import LocalClient, set_current_client
from marin.datakit.normalize import generate_id, normalize_to_parquet

from experiments.baseline_collection.dedup_extracted import _carry_provenance, _reshape_records


@pytest.fixture(autouse=True)
def local_backend():
    with set_current_client(LocalClient()):
        yield


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def _read_main_parquet(output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for pf in sorted((output_dir / "outputs" / "main").glob("*.parquet")):
        records.extend(pq.read_table(str(pf)).to_pylist())
    return records


def test_carry_provenance_from_raw_batch():
    """A raw extraction record keeps every provenance column; text is canonical, no id leaks."""
    raw = {
        "text": "extracted body",
        "url": "http://example.com/a",
        "warc_record_id": "rec-1",
        "warc_file": "CC-MAIN-x.warc.gz",
        "snapshot": "2024-10",
        "pipeline_id": "llm_pipeline_v1",
        "num_chunks": 3,
    }
    out = _carry_provenance(raw, raw["text"])
    assert out == {
        "url": "http://example.com/a",
        "warc_record_id": "rec-1",
        "warc_file": "CC-MAIN-x.warc.gz",
        "snapshot": "2024-10",
        "pipeline_id": "llm_pipeline_v1",
        "num_chunks": 3,
        "text": "extracted body",
    }


def test_carry_provenance_drops_datakit_internal_ids():
    """The apply read is over normalized parquet: id/source_id are datakit-internal and must not leak."""
    normalized = {
        "id": "deadbeef",
        "source_id": "orig-99",
        "text": "body",
        "url": "http://example.com/b",
        "warc_record_id": "rec-2",
    }
    out = _carry_provenance(normalized, normalized["text"])
    assert "id" not in out
    assert "source_id" not in out
    assert out == {"text": "body", "url": "http://example.com/b", "warc_record_id": "rec-2"}


def test_carry_provenance_generated_text_becomes_canonical_text():
    """When only ``generated_text`` is present it becomes ``text`` and is not duplicated."""
    raw = {"generated_text": "gen body", "url": "http://example.com/c", "warc_record_id": "rec-3"}
    out = _carry_provenance(raw, raw["generated_text"])
    assert out == {"text": "gen body", "url": "http://example.com/c", "warc_record_id": "rec-3"}


def test_reshape_records_carries_provenance_and_skips_empty(tmp_path: Path):
    """`_reshape_records` reads a batch and keeps provenance, skipping blank-text rows."""
    batch = tmp_path / "batch_0.jsonl.gz"
    _write_jsonl_gz(
        batch,
        [
            {"text": "doc one", "url": "http://a", "warc_record_id": "r1", "warc_file": "w1", "snapshot": "s1"},
            {"text": "", "url": "http://blank", "warc_record_id": "r2"},  # dropped: empty text
            {"generated_text": "doc three", "url": "http://c", "warc_record_id": "r3"},
        ],
    )
    out = list(_reshape_records(str(batch)))
    assert out == [
        {"text": "doc one", "url": "http://a", "warc_record_id": "r1", "warc_file": "w1", "snapshot": "s1"},
        {"text": "doc three", "url": "http://c", "warc_record_id": "r3"},
    ]


def test_provenance_survives_reshape_normalize_apply_roundtrip(tmp_path: Path):
    """End-to-end over the real normalize engine: reshape → normalize → apply keeps provenance.

    This is the regression guard for the bug: normalize passes non-id/text columns through, and
    both dedup drop-points (`_reshape_records` and the fuzzy-apply projection) now carry them.
    """
    raw_batch = tmp_path / "raw" / "batch_0.jsonl.gz"
    _write_jsonl_gz(
        raw_batch,
        [
            {"text": "alpha document", "url": "http://a", "warc_record_id": "r1", "warc_file": "w1", "snapshot": "s"},
            {"text": "beta document", "url": "http://b", "warc_record_id": "r2", "warc_file": "w1", "snapshot": "s"},
            # byte-identical text to the first row: EXACT dedup collapses it in normalize.
            {
                "text": "alpha document",
                "url": "http://a-dup",
                "warc_record_id": "r1b",
                "warc_file": "w2",
                "snapshot": "s",
            },
        ],
    )

    # Stage 1: reshape writes the normalize input tree (text + provenance).
    reshape_dir = tmp_path / "reshape"
    _write_jsonl_gz(reshape_dir / "data-00000.jsonl.gz", list(_reshape_records(str(raw_batch))))

    # Stage 2: the real normalize engine (EXACT dedup on text) — proves passthrough of provenance.
    normalize_dir = tmp_path / "normalize"
    normalize_to_parquet(input_path=str(reshape_dir), output_path=str(normalize_dir))
    normalized = _read_main_parquet(normalize_dir)
    assert len(normalized) == 2  # the duplicate "alpha document" was collapsed
    assert all("url" in r and "warc_record_id" in r for r in normalized)

    # Stage 3: the fuzzy-apply projection (here with no non-canonical drops) re-emits the deduped tree.
    final = [_carry_provenance(r, r["text"]) for r in normalized]
    by_text = {r["text"]: r for r in final}
    assert set(by_text) == {"alpha document", "beta document"}
    # Provenance survives to the final deduped record; datakit id does not leak.
    assert by_text["beta document"] == {
        "text": "beta document",
        "url": "http://b",
        "warc_record_id": "r2",
        "warc_file": "w1",
        "snapshot": "s",
    }
    assert "id" not in by_text["alpha document"]
    # Sanity: the surviving alpha record still carries provenance (one of the two colliding rows).
    assert by_text["alpha document"]["warc_record_id"] in {"r1", "r1b"}
    # text remains the dedup key: the collapsed id is the content hash of the text.
    assert generate_id("beta document") == next(r["id"] for r in normalized if r["text"] == "beta document")
