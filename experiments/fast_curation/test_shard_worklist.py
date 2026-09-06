# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V3 sharded work-list contract: deterministic layout, frozen-once, and the B shard flow."""

from __future__ import annotations

import functools
import os

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.fast_curation import batch_format, cpu_phase_a, preprocess, reaper, tpu_phase
from experiments.fast_curation import shard_worklist as sw
from experiments.fast_curation.spec import get_spec

SPEC = get_spec("lpv11_fastpipe_v2_1")
WARCS = [f"s3://commoncrawl/crawl-data/CC-MAIN-2022-33/warc/{i:05d}.warc.gz" for i in range(40)]


@pytest.fixture
def local_central(tmp_path, monkeypatch):
    """Route the central layout to a local dir so the contract is testable without GCS."""
    monkeypatch.setattr(sw, "CENTRAL_BUCKET", str(tmp_path / "central"))
    (tmp_path / "central").mkdir()
    return tmp_path


def test_shard_of_is_deterministic_and_stable():
    """The shard formula is a FROZEN partition key: any change re-shuffles a live run's claims and
    markers under it. Pin concrete values so a drive-by 'improvement' fails loudly."""
    assert sw.shard_of(WARCS[0], 8) == sw.shard_of(WARCS[0], 8)
    assert [sw.shard_of(w, 8) for w in WARCS[:4]] == [7, 2, 7, 4]


def test_build_layout_and_refuse_rebuild(local_central):
    sw.build(SPEC, WARCS, n_shards=4, regions=["us-east5", "us-east1"], seed=0)
    index = sw.load_index(SPEC)
    assert [e["shard"] for e in index] == [0, 1, 2, 3]
    assert sum(e["n_warcs"] for e in index) == len(WARCS)
    assert {e["region"] for e in index} == {"us-east5", "us-east1"}
    # membership matches the formula, exactly once per WARC
    seen = []
    for e in index:
        pairs = sw.load_shard(SPEC, e["shard"])
        assert len(pairs) == e["n_warcs"]
        for w, _h in pairs:
            assert sw.shard_of(w, 4) == e["shard"]
            seen.append(w)
    assert sorted(seen) == sorted(WARCS)
    # the layout is frozen: rebuilding (even identically) must refuse
    with pytest.raises(RuntimeError, match="FROZEN"):
        sw.build(SPEC, WARCS, n_shards=4, regions=["us-east5"], seed=0)


def test_process_warc_v3_tokenizes_from_text_and_writes_kept_v3(tmp_path, monkeypatch):
    """END TO END for the V3 contract: presurvivor carries only text; B must tokenize it, route the
    band, write KEPT_V3 (with n_tokens + both probs), and return a complete catalog row."""
    bucket = str(tmp_path)
    warc_hash = "beef03"
    rows = [
        {"doc_id": d, "url": "u", "warc_hash": warc_hash, "snapshot": "s", "fasttext_score": 0.5, "text": t}
        for d, t in [
            ("hi", "confident text the pooled gate accepts outright"),
            ("band_keep", "uncertain text the terminal model keeps"),
            ("band_drop", "uncertain text the terminal model rejects"),
            ("lo", "text the pooled gate drops"),
        ]
    ]
    batch_format.write_presurvivors_v3(f"{SPEC.presurvivors_prefix(bucket)}/data-{warc_hash}.parquet", rows)

    canned = iter(
        [
            np.array([0.95, 0.50, 0.40, 0.01], dtype=np.float32),  # pooled pass, all docs
            np.array([0.90, 0.10], dtype=np.float32),  # terminal pass, band only
        ]
    )
    monkeypatch.setattr(tpu_phase, "score_survivors", lambda *a, **k: next(canned))
    monkeypatch.setattr(tpu_phase, "_device_kind", lambda: "TPU test")
    monkeypatch.setattr(tpu_phase, "_device_count", lambda: 4)

    tokenizer = preprocess.load_tokenizer("answerdotai/ModernBERT-base")
    row = tpu_phase.process_warc_v3(
        SPEC,
        warc_hash,
        score_fn=None,
        model=None,
        config=None,
        bucket=bucket,
        batch_size=4,
        bucket_tokens=True,
        tokenize_batch=functools.partial(preprocess.tokenize_trunc_batch_arrow, tokenizer, max_length=128),
        pooled=(None, None),
    )

    kept = batch_format.read_table(f"{SPEC.kept_prefix(bucket)}/data-{warc_hash}.parquet")
    assert kept.schema.equals(batch_format.KEPT_V3_SCHEMA)
    by_id = {r["doc_id"]: r for r in kept.to_pylist()}
    assert set(by_id) == {"hi", "band_keep"}
    assert np.isnan(by_id["hi"]["modernbert_prob"]) and by_id["band_keep"]["modernbert_prob"] == pytest.approx(0.9)
    assert by_id["hi"]["n_tokens"] > 2, "n_tokens is computed at scoring time from real tokenization"
    assert (row["n_kept"], row["n_hi_accept"], row["n_band"]) == (2, 1, 2)
    assert row["kept_path"].endswith(f"kept/data-{warc_hash}.parquet") and row["device"] == "TPU test"


def test_a_shard_loop_marks_and_stamps_sentinels(local_central, monkeypatch):
    """The A shard loop must claim a shard, skip marked WARCs, mark new ones, and stamp _a_done."""
    sw.build(SPEC, WARCS, n_shards=2, regions=["us-east5"], seed=0)
    claims, registries = set(), {}
    monkeypatch.setattr(cpu_phase_a, "_claim_warc_atomic", lambda p, stale_hours: claims.add(p) or True)
    monkeypatch.setattr(
        cpu_phase_a, "_register_completed_warc", lambda h, prefix: registries.setdefault(prefix, set()).add(h)
    )
    monkeypatch.setattr(cpu_phase_a, "_load_completed_registry", lambda prefix: set(registries.get(prefix, set())))
    monkeypatch.setattr(cpu_phase_a, "_refresh_shard_claim", lambda p: None)
    monkeypatch.setattr(
        cpu_phase_a,
        "Heartbeat",
        lambda *a, **k: type(
            "H",
            (),
            {"record_warc": lambda *_a, **_k: None, "tick": lambda *_a, **_k: None, "close": lambda *_a, **_k: None},
        )(),
    )
    monkeypatch.setattr(cpu_phase_a, "region_from_bucket", lambda b: "us-east5")

    processed = []
    cpu_phase_a.run_shard_loop(
        SPEC,
        "gs://marin-us-east5",
        process_one=lambda w, h, refresh: processed.append(h)
        or {"docs_in": 1, "docs_out": 1, "wall_seconds": 0.0, "compute_seconds": {}},
        shuffle_seed=0,
        poll_seconds=0.01,
        max_idle_passes=1,
    )
    assert len(processed) == len(WARCS), "every WARC of every shard processed exactly once"
    sentinels = registries[sw.sentinel_prefix(SPEC, "a")]
    assert sentinels == {"00000", "00001"}
    marks = [k for k in registries if "_a_marks/shard-" in k]
    assert len(marks) == 2 and sum(len(registries[m]) for m in marks) == len(WARCS)


def test_reap_v3_verifies_catalog_then_deletes_shard_presurvivors(local_central, tmp_path, monkeypatch):
    """The v3 reaper must delete a shard's presurvivors ONLY after every catalog row's kept file
    verifies, and must leave shards with missing/corrupt kept output untouched."""
    sw.build(SPEC, WARCS[:8], n_shards=2, regions=["us-east5"], seed=0)
    bucket = str(tmp_path / "region")
    monkeypatch.setitem(sw.REGION_TO_BUCKET, "us-east5", bucket)
    monkeypatch.setattr(reaper, "_fs", lambda: fsspec.filesystem("file"))

    for s in (0, 1):
        rows = []
        for _, h in sw.load_shard(SPEC, s):
            pre = f"{SPEC.presurvivors_prefix(bucket)}/data-{h}.parquet"
            batch_format.write_presurvivors_v3(pre, [])
            kept = f"{SPEC.kept_prefix(bucket)}/data-{h}.parquet"
            batch_format.write_table_v3(kept, batch_format.KEPT_V3_SCHEMA.empty_table())
            rows.append({"warc_hash": h, "kept_path": kept, "n_kept": 0})
        with fsspec.open(sw.catalog_path(SPEC, s), "wb") as f:
            pq.write_table(pa.Table.from_pylist(rows), f)
        with fsspec.open(f"{sw.sentinel_prefix(SPEC, 'b')}/data-{s}", "w") as f:
            f.write("{}")
    # corrupt ONE kept file of shard 1 -> that whole shard must be left untouched
    victim = sw.load_shard(SPEC, 1)[0][1]
    with open(f"{SPEC.kept_prefix(bucket)}/data-{victim}.parquet", "wb") as f:
        f.write(b"not a parquet")

    stats = reaper.reap_v3(SPEC, lag_minutes=0, dry_run=False)
    assert stats["shards_reaped"] == 1 and stats["shards_unverified"] == 1
    for _, h in sw.load_shard(SPEC, 0):
        assert not os.path.exists(f"{SPEC.presurvivors_prefix(bucket)}/data-{h}.parquet")
    for _, h in sw.load_shard(SPEC, 1):
        assert os.path.exists(f"{SPEC.presurvivors_prefix(bucket)}/data-{h}.parquet"), "unverified shard untouched"
