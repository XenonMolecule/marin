# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""lpv11 fork-extraction swap: WARC-hash derivation, selection, crash handling, label attach."""

from __future__ import annotations

import gzip
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import generate_id

from experiments.baseline_collection import grid_corpora
from experiments.baseline_collection.decode_warcs_clean import _warc_path_hash
from experiments.baseline_collection.grid_corpora import Stage, stage_shard_output
from experiments.datakit.store.datakit_store import _ids_match, _load_cluster_table, _load_quality_table
from experiments.fast_curation import attach_lpv11_fork_labels as attach
from experiments.fast_curation import build_lpv11_swap_allowlist as bsa
from experiments.fast_curation import lpv11_fork_extractor as fork
from experiments.fast_curation.upstream_resiliparse import MIN_TEXT_CHARS

WARC = "s3://commoncrawl/crawl-data/CC-MAIN-2017-30/segments/1500549426086.44/warc/CC-MAIN-x.warc.gz"
H = _warc_path_hash(WARC)
LABELS = [f"topic_{i}" for i in range(grid_corpora.NUM_TOPICS)]
LONG = "x" * MIN_TEXT_CHARS


def _allow(rid: str) -> dict:
    return {
        "warc_record_id": rid,
        "url": f"http://e.com/{rid}",
        "warc_file": WARC,
        "warc_hash": H,
        "snapshot": "CC-MAIN-2017-30",
        "src_shard": "data-00000-of-03193",
        "src_row": 0,
    }


def test_warc_hash_matches_the_decoder_convention():
    """The derived hash must equal what the per-WARC filenames elsewhere are built from."""
    assert bsa.warc_hash_of(WARC) == _warc_path_hash(WARC)
    assert bsa.warc_hash_of(WARC) == hashlib.sha256(WARC.encode()).hexdigest()[:12]


def test_select_and_extract_keys_on_warc_record_id():
    allow = [_allow("r1"), _allow("r2"), _allow("r3")]
    chunks = [
        [{"doc_id": "r2", "html": "<p>short</p>"}, {"doc_id": "other", "html": "<p>skip</p>"}],
        [{"doc_id": "r1", "html": "<p>long</p>"}],
    ]
    texts = {"<p>short</p>": "x" * (MIN_TEXT_CHARS - 1), "<p>long</p>": LONG}

    def extract(htmls, ids):
        return [texts[h] for h in htmls], []

    rows, stats = select_and_extract_helper(allow, chunks, extract)
    assert [r["warc_record_id"] for r in rows] == ["r1"]
    assert rows[0]["text"] == LONG and rows[0]["warc_hash"] == H
    assert stats["n_requested"] == 3 and stats["n_written"] == 1
    assert stats["n_short"] == 1 and stats["missing"] == ["r3"] and stats["n_crashed"] == 0


def select_and_extract_helper(allow, chunks, extract):
    return fork.select_and_extract(allow, chunks, extract)


def test_crashed_pages_are_reported_and_dropped_not_fatal():
    """A frameset page that kills its pool child yields '' and is recorded, never losing the WARC."""
    allow = [_allow("good"), _allow("frameset")]
    chunks = [[{"doc_id": "good", "html": "<p>ok</p>"}, {"doc_id": "frameset", "html": "<frameset>"}]]

    def extract(htmls, ids):
        # what ResiliparseRsPool returns after isolating a hard process death
        return [LONG if h == "<p>ok</p>" else "" for h in htmls], [i for i, h in zip(ids, htmls) if h == "<frameset>"]

    rows, stats = fork.select_and_extract(allow, chunks, extract)
    assert [r["warc_record_id"] for r in rows] == ["good"]
    assert stats["n_crashed"] == 1 and stats["crashed"] == ["frameset"]
    assert stats["n_short"] == 1  # the crashed page's "" also trips the length floor
    assert stats["missing"] == []


def test_record_shard_keeps_lpv11_column_names(tmp_path):
    shard = str(tmp_path / "data-x.jsonl.gz")
    fork.write_record_shard(shard, [{**_allow("r1"), "text": LONG}])
    with gzip.open(shard, "rt") as fh:
        rec = json.loads(fh.readline())
    assert set(rec) == set(fork.RECORD_COLUMNS) | {"text"}
    assert rec["warc_record_id"] == "r1" and "doc_id" not in rec


def _write_source(tmp_path, records, *, native_ids=None, n_labels=None):
    """One lpv11 source shard plus its co-partitioned topic/quality tables."""
    stem = "data-00000-of-03193"
    shard = tmp_path / "src" / f"{stem}.jsonl.gz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(shard, "wt") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    ids = [generate_id(r["text"]) for r in records]
    n = len(records) if n_labels is None else n_labels
    topic = pa.table(
        {
            "id": ids[:n],
            "native_id": (native_ids or [r["warc_record_id"] for r in records])[:n],
            "url": [r["url"] for r in records][:n],
            "cluster_24": pa.array([i % 24 for i in range(n)], pa.int32()),
            "topic_prob": pa.array([0.9] * n, pa.float32()),
            "token_length": pa.array([100 + i for i in range(n)], pa.int32()),
        }
    )
    quality = pa.table(
        {
            "source": ["lpv11_10k"] * n,
            "id": ids[:n],
            "score": pa.array([0.4] * n, pa.float32()),
            "quality_bucket": pa.array([i % 5 for i in range(n)], pa.int8()),
        }
    )
    for path, table in (
        (bsa.topic_table_path(str(tmp_path), "lpv11_10k", stem), topic),
        (bsa.quality_table_path(str(tmp_path), "lpv11_10k", stem), quality),
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path)
    return str(shard)


def _src_record(i: int) -> dict:
    return {
        "url": f"http://e.com/{i}",
        "warc_record_id": f"rid-{i}",
        "warc_file": WARC,
        "snapshot": "CC-MAIN-2017-30",
        "pipeline_id": "llm_pipeline_v1_1",
        "num_chunks": 1,
        "text": f"document {i} " * 12,
    }


def test_provenance_rows_derive_the_hash_and_need_no_labels(tmp_path):
    records = [_src_record(i) for i in range(3)]
    shard = _write_source(tmp_path, records)
    rows = list(bsa.provenance_rows(shard))
    assert [r["warc_record_id"] for r in rows] == ["rid-0", "rid-1", "rid-2"]
    assert {r["warc_hash"] for r in rows} == {H}
    assert "text" not in rows[0] and "cluster_24" not in rows[0]
    assert rows[2]["src_row"] == 2


def test_labeled_rows_verify_identity_against_lpv11_tables(tmp_path):
    records = [_src_record(i) for i in range(3)]
    shard = _write_source(tmp_path, records)
    rows = list(bsa.labeled_rows(shard, str(tmp_path), "lpv11_10k"))
    assert [r["cluster_24"] for r in rows] == [0, 1, 2]
    assert [r["quality_bucket"] for r in rows] == [0, 1, 2]
    assert {r["warc_hash"] for r in rows} == {H}


def test_labeled_rows_reject_a_native_id_mismatch(tmp_path):
    records = [_src_record(i) for i in range(3)]
    shard = _write_source(tmp_path, records, native_ids=["rid-0", "WRONG", "rid-2"])
    with pytest.raises(RuntimeError, match="native_id"):
        list(bsa.labeled_rows(shard, str(tmp_path), "lpv11_10k"))


def test_labeled_rows_reject_a_label_computed_on_other_text(tmp_path):
    records = [_src_record(i) for i in range(2)]
    shard = _write_source(tmp_path, records)
    with gzip.open(shard, "wt") as fh:
        fh.write(json.dumps(records[0]) + "\n")
        fh.write(json.dumps({**records[1], "text": "edited after labelling"}) + "\n")
    with pytest.raises(RuntimeError, match="content hash"):
        list(bsa.labeled_rows(shard, str(tmp_path), "lpv11_10k"))


def test_attach_writes_tables_in_shard_order(tmp_path):
    grid_corpora.OUTPUT_BASE = str(tmp_path)
    root = str(tmp_path / "root")
    os.makedirs(f"{root}/deduped")
    os.makedirs(f"{root}/labels")
    os.makedirs(f"{root}/catalog")
    recs = [{**_allow("rid-1"), "text": "alpha " * 12}, {**_allow("rid-0"), "text": "beta " * 12}]
    fork.write_record_shard(fork.shard_path(root, H), recs)
    labels = [
        {
            "warc_record_id": "rid-0",
            "warc_hash": H,
            "cluster_24": 4,
            "topic_prob": 0.9,
            "token_length": 50,
            "quality_score": 0.5,
            "quality_bucket": 1,
            "src_shard": "s",
            "src_row": 0,
        },
        {
            "warc_record_id": "rid-1",
            "warc_hash": H,
            "cluster_24": 7,
            "topic_prob": 0.8,
            "token_length": 60,
            "quality_score": 0.6,
            "quality_bucket": 3,
            "src_shard": "s",
            "src_row": 1,
        },
        {
            "warc_record_id": "rid-9",
            "warc_hash": H,
            "cluster_24": 0,
            "topic_prob": 0.1,
            "token_length": 10,
            "quality_score": 0.1,
            "quality_bucket": 0,
            "src_shard": "s",
            "src_row": 2,
        },
    ]
    pq.write_table(pa.Table.from_pylist(labels), f"{root}/labels/data-{H}.parquet")
    with open(f"{root}/catalog/data-{H}.json", "w") as fh:
        json.dump({"warc_hash": H, "n_written": 2, "n_short": 1, "n_crashed": 0, "missing": [], "skipped": False}, fh)

    summary = attach.attach_one(H, root, "fork_test", str(tmp_path), LABELS)
    assert summary == {"warc_hash": H, "n_rows": 2, "n_unused_labels": 1}
    shard = fork.shard_path(root, H)
    ids, clusters = _load_cluster_table(stage_shard_output("fork_test", Stage.TOPIC, shard), "cluster_24")
    qids, buckets = _load_quality_table(stage_shard_output("fork_test", Stage.QUALITY, shard))
    assert clusters.tolist() == [7, 4] and buckets.tolist() == [3, 1]  # shard order, not label order
    assert _ids_match(ids, qids)
    assert ids.to_pylist() == [generate_id(r["text"]) for r in recs]


def test_attach_rejects_a_catalog_that_disagrees_with_the_shard(tmp_path):
    grid_corpora.OUTPUT_BASE = str(tmp_path)
    root = str(tmp_path / "root")
    os.makedirs(f"{root}/deduped")
    os.makedirs(f"{root}/labels")
    os.makedirs(f"{root}/catalog")
    fork.write_record_shard(fork.shard_path(root, H), [{**_allow("rid-0"), "text": "alpha " * 12}])
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "warc_record_id": "rid-0",
                    "warc_hash": H,
                    "cluster_24": 4,
                    "topic_prob": 0.9,
                    "token_length": 50,
                    "quality_score": 0.5,
                    "quality_bucket": 1,
                    "src_shard": "s",
                    "src_row": 0,
                },
                {
                    "warc_record_id": "rid-5",
                    "warc_hash": H,
                    "cluster_24": 1,
                    "topic_prob": 0.2,
                    "token_length": 20,
                    "quality_score": 0.2,
                    "quality_bucket": 0,
                    "src_shard": "s",
                    "src_row": 1,
                },
            ]
        ),
        f"{root}/labels/data-{H}.parquet",
    )
    with open(f"{root}/catalog/data-{H}.json", "w") as fh:
        json.dump({"warc_hash": H, "n_written": 1, "n_short": 0, "n_crashed": 0, "missing": [], "skipped": False}, fh)
    with pytest.raises(RuntimeError, match="disagree"):
        attach.attach_one(H, root, "fork_test", str(tmp_path), LABELS)


def test_fork_worker_import_never_pulls_in_upstream_resiliparse():
    """The fork worker must be importable in a process where the Rust fork owns the name.

    ``cpu_phase_c._init_rs_worker`` puts the fork FIRST on ``sys.path`` in every pool child, and
    the fork has no ``resiliparse.extract.html2text``. If importing the worker drags that module
    in at module scope, every child dies on import and the pool reports it as a smoke-test
    SIGSEGV -- which is exactly what happened on 2026-10-02. Run in a subprocess so this measures
    a cold import, not whatever the test session already loaded.
    """
    import subprocess
    import sys

    probe = (
        "import importlib, sys;"
        "importlib.import_module('experiments.fast_curation.lpv11_fork_extractor');"
        "bad=[m for m in sys.modules if m.startswith('resiliparse.extract')];"
        "print('LOADED' if bad else 'CLEAN', bad)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-600:]
    assert out.stdout.startswith("CLEAN"), f"fork worker imported upstream resiliparse: {out.stdout.strip()}"


def test_attach_arithmetic_counts_a_crashed_page_once(tmp_path):
    """A crashed page is already inside n_short; adding n_crashed again would fail attach.

    Regression for the fleet's 10 frameset segfaults: the pool returns "" for a crashed page, the
    worker counts that as short, and the catalog ALSO records it under n_crashed.
    """
    grid_corpora.OUTPUT_BASE = str(tmp_path)
    root = str(tmp_path / "root")
    os.makedirs(f"{root}/deduped")
    os.makedirs(f"{root}/labels")
    os.makedirs(f"{root}/catalog")
    fork.write_record_shard(fork.shard_path(root, H), [{**_allow("rid-0"), "text": "alpha " * 12}])
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "warc_record_id": "rid-0",
                    "warc_hash": H,
                    "cluster_24": 4,
                    "topic_prob": 0.9,
                    "token_length": 50,
                    "quality_score": 0.5,
                    "quality_bucket": 1,
                    "src_shard": "s",
                    "src_row": 0,
                },
                {
                    "warc_record_id": "rid-crash",
                    "warc_hash": H,
                    "cluster_24": 1,
                    "topic_prob": 0.2,
                    "token_length": 20,
                    "quality_score": 0.2,
                    "quality_bucket": 0,
                    "src_shard": "s",
                    "src_row": 1,
                },
            ]
        ),
        f"{root}/labels/data-{H}.parquet",
    )
    with open(f"{root}/catalog/data-{H}.json", "w") as fh:
        # one page crashed -> "" -> also counted short. One unused label, not two.
        json.dump(
            {
                "warc_hash": H,
                "n_written": 1,
                "n_short": 1,
                "n_crashed": 1,
                "crashed": ["rid-crash"],
                "missing": [],
                "skipped": False,
            },
            fh,
        )
    assert attach.attach_one(H, root, "fork_test", str(tmp_path), LABELS)["n_unused_labels"] == 1
