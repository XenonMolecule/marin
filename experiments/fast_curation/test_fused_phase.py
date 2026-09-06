# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused single-phase contract: in-RAM rows through the v3 scoring path, v3-identical output."""

from __future__ import annotations

import functools
from concurrent.futures import Future

import fsspec
import numpy as np
import pyarrow.parquet as pq

from experiments.fast_curation import batch_format, fused_phase, preprocess, tpu_phase
from experiments.fast_curation import shard_worklist as sw
from experiments.fast_curation.spec import get_spec

SPEC = get_spec("lpv11_fastpipe_v2_1_fused")
WARCS = [f"s3://commoncrawl/crawl-data/CC-MAIN-2022-33/warc/{i:05d}.warc.gz" for i in range(8)]


class _InlineExecutor:
    """submit() runs the fn immediately and returns a resolved Future — deterministic pipelining."""

    def submit(self, fn, *a):
        f: Future = Future()
        f.set_result(fn(*a))
        return f


def test_fused_spec_shares_v3_output_contract():
    assert SPEC.storage_version == 4
    assert batch_format.kept_schema_for(SPEC).equals(batch_format.KEPT_V3_SCHEMA)
    v3 = get_spec("lpv11_fastpipe_v2_1")
    assert SPEC.subdir() != v3.subdir(), "fused must have its own namespace"
    for f in ("fasttext_threshold", "pooled_threshold", "pooled_hi", "modernbert_threshold"):
        assert getattr(SPEC, f) == getattr(v3, f), f"cascade semantics must match v3 ({f})"


def test_run_one_shard_pipelines_ram_rows_to_kept_and_catalog(tmp_path, monkeypatch):
    """END TO END minus real models: fake A returns in-RAM rows, fake chip scores them; the shard
    must produce kept/ files, per-WARC markers carrying A+B timing, a catalog, and a sentinel —
    with NO presurvivor files anywhere (the whole point of the fused contract)."""
    monkeypatch.setattr(sw, "CENTRAL_BUCKET", str(tmp_path / "central"))
    (tmp_path / "central").mkdir()
    bucket = str(tmp_path / "region")
    sw.build(SPEC, WARCS, n_shards=2, regions=["us-east5"], seed=0)

    def fake_a(warc_path, warc_hash):
        rows = [
            {
                "doc_id": f"{warc_hash}-hi",
                "url": "u",
                "warc_hash": warc_hash,
                "snapshot": "s",
                "fasttext_score": 0.5,
                "text": "confident text",
            },
            {
                "doc_id": f"{warc_hash}-lo",
                "url": "u",
                "warc_hash": warc_hash,
                "snapshot": "s",
                "fasttext_score": 0.5,
                "text": "rejected text",
            },
        ]
        return rows, {
            "a_n_in": 40,
            "a_n_prefilter_dropped": 1,
            "a_n_extract_empty": 0,
            "a_n_extract_crashed": 0,
            "a_decode_s": 1.0,
            "a_extract_s": 2.0,
            "a_fasttext_s": 0.5,
            "a_wall_s": 3.5,
        }

    monkeypatch.setattr(fused_phase, "a_extract_warc", fake_a)
    monkeypatch.setattr(
        tpu_phase,
        "score_survivors",
        lambda *a, **k: np.array([0.95, 0.01], dtype=np.float32)[: len(a[3])],
    )
    monkeypatch.setattr(tpu_phase, "_device_kind", lambda: "TPU test")
    monkeypatch.setattr(tpu_phase, "_device_count", lambda: 8)

    tokenizer = preprocess.load_tokenizer("answerdotai/ModernBERT-base")
    hb = type(
        "H", (), {"record_warc": lambda *a, **k: None, "tick": lambda *a, **k: None, "close": lambda *a, **k: None}
    )()
    for s in (0, 1):
        fused_phase._run_one_shard(
            SPEC,
            s,
            _InlineExecutor(),
            bucket=bucket,
            claim_path=f"{sw.claim_prefix(SPEC, 'b')}/shard-{s:05d}",
            queue_depth=3,
            batch_size=8,
            bucket_tokens=True,
            tokenize_batch=functools.partial(preprocess.tokenize_trunc_batch_arrow, tokenizer, max_length=64),
            score_fn=None,
            model=None,
            config=None,
            pooled=(None, None),
            hb=hb,
        )

    fs = fsspec.filesystem("file")
    sentinels = fs.ls(sw.sentinel_prefix(SPEC, "b"))
    assert len(sentinels) == 2, "both shards stamped complete"
    total = 0
    for s in (0, 1):
        with fsspec.open(sw.catalog_path(SPEC, s), "rb") as f:
            cat = pq.read_table(f).to_pylist()
        assert len(cat) == len(sw.load_shard(SPEC, s))
        for row in cat:
            assert row["n_kept"] == 1 and row["a_n_in"] == 40, "catalog rows carry fused A+B fields"
            kept = batch_format.read_table(row["kept_path"])
            assert kept.schema.equals(batch_format.KEPT_V3_SCHEMA)
            assert kept.num_rows == 1 and kept.to_pylist()[0]["doc_id"].endswith("-hi")
            total += kept.num_rows
    assert total == len(WARCS)
    assert not fs.exists(SPEC.presurvivors_prefix(bucket)), "fused writes NO presurvivors"


def test_run_one_shard_resumes_from_marks(tmp_path, monkeypatch):
    """A preempted fused worker must skip marked WARCs (their rows are gone from RAM) and still
    write a complete catalog from the surviving markers."""
    monkeypatch.setattr(sw, "CENTRAL_BUCKET", str(tmp_path / "central"))
    (tmp_path / "central").mkdir()
    bucket = str(tmp_path / "region")
    sw.build(SPEC, WARCS, n_shards=1, regions=["us-east5"], seed=0)
    pairs = sw.load_shard(SPEC, 0)
    # Pre-mark the first WARC as done by an earlier (preempted) attempt.
    pre_h = pairs[0][1]
    marks_prefix = f"{sw.CENTRAL_BUCKET}/{SPEC.subdir()}/_b_marks/shard-00000"
    tpu_phase._write_marker(f"{marks_prefix}/data-{pre_h}", {"warc_hash": pre_h, "n_kept": 7, "a_n_in": 1})

    seen = []

    def fake_a(warc_path, warc_hash):
        seen.append(warc_hash)
        return [], {
            "a_n_in": 0,
            "a_n_prefilter_dropped": 0,
            "a_n_extract_empty": 0,
            "a_n_extract_crashed": 0,
            "a_decode_s": 0,
            "a_extract_s": 0,
            "a_fasttext_s": 0,
            "a_wall_s": 0,
        }

    monkeypatch.setattr(fused_phase, "a_extract_warc", fake_a)
    monkeypatch.setattr(tpu_phase, "score_survivors", lambda *a, **k: np.array([], dtype=np.float32))
    monkeypatch.setattr(tpu_phase, "_device_kind", lambda: "TPU test")
    monkeypatch.setattr(tpu_phase, "_device_count", lambda: 8)
    tokenizer = preprocess.load_tokenizer("answerdotai/ModernBERT-base")
    hb = type(
        "H", (), {"record_warc": lambda *a, **k: None, "tick": lambda *a, **k: None, "close": lambda *a, **k: None}
    )()
    fused_phase._run_one_shard(
        SPEC,
        0,
        _InlineExecutor(),
        bucket=bucket,
        claim_path=f"{sw.claim_prefix(SPEC, 'b')}/shard-00000",
        queue_depth=2,
        batch_size=8,
        bucket_tokens=True,
        tokenize_batch=functools.partial(preprocess.tokenize_trunc_batch_arrow, tokenizer, max_length=64),
        score_fn=None,
        model=None,
        config=None,
        pooled=(None, None),
        hb=hb,
    )
    assert pre_h not in seen, "marked WARC must not be re-extracted"
    with fsspec.open(sw.catalog_path(SPEC, 0), "rb") as f:
        cat = {r["warc_hash"]: r for r in pq.read_table(f).to_pylist()}
    assert set(cat) == {h for _, h in pairs}
    assert cat[pre_h]["n_kept"] == 7, "pre-preemption marker row survives into the catalog"
