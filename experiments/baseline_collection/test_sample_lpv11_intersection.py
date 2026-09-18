# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import uuid

import pytest

from experiments.baseline_collection import sample_lpv11_intersection as mod
from experiments.baseline_collection.download_warcs import _warc_path_hash

WARC_A = "s3://commoncrawl/crawl-data/CC-MAIN-2017-30/segments/1/warc/a.warc.gz"
WARC_B = "s3://commoncrawl/crawl-data/CC-MAIN-2016-50/segments/2/warc/b.warc.gz"
WARC_OTHER = "s3://commoncrawl/crawl-data/CC-MAIN-2019-09/segments/3/warc/c.warc.gz"


def _write_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _lpv11_row(warc, rid):
    return {
        "url": f"http://x/{rid}",
        "warc_record_id": rid,
        "warc_file": warc,
        "snapshot": mod._snapshot(warc),
        "pipeline_id": "llm_pipeline_v1_1",
        "num_chunks": 1,
        "text": f"text {rid}",
    }


def _fastpipe_row(warc, rid):
    return {
        "doc_id": rid,
        "url": f"http://x/{rid}",
        "warc_hash": _warc_path_hash(warc),
        "snapshot": "s",
        "fasttext_score": 0.5,
        "pooled_prob": 0.6,
        "modernbert_prob": None,
        "text": "t",
    }


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Two chosen WARCs plus one unchosen; lpv11 and fastpipe overlap partially."""
    ids = {w: [str(uuid.uuid4()) for _ in range(30)] for w in (WARC_A, WARC_B, WARC_OTHER)}
    lpv11_rows = [_lpv11_row(w, rid) for w in ids for rid in ids[w]]
    # fastpipe keeps only the first 20 of each WARC's 30 lpv11 docs, plus ids lpv11 never kept.
    fastpipe_rows = [_fastpipe_row(w, rid) for w in ids for rid in ids[w][:20]]
    fastpipe_rows += [_fastpipe_row(WARC_A, str(uuid.uuid4())) for _ in range(5)]

    monkeypatch.setattr(mod, "_CHOSEN", {_warc_path_hash(WARC_A): WARC_A, _warc_path_hash(WARC_B): WARC_B})
    monkeypatch.setattr(mod, "_SEED", 7)
    monkeypatch.setattr(mod, "_K", 4)
    return tmp_path, ids, lpv11_rows, fastpipe_rows


def _run_scan(tmp_path, lpv11_rows, fastpipe_rows, n_shards, monkeypatch):
    fp_shards = [_write_gz(tmp_path / f"fp{i}.jsonl.gz", fastpipe_rows[i::n_shards]) for i in range(n_shards)]
    lp_shards = [_write_gz(tmp_path / f"lp{i}.jsonl.gz", lpv11_rows[i::n_shards]) for i in range(n_shards)]
    parts = []
    fp_counts = mod.Counter()
    for p in fp_shards:
        keys, scores, c = mod._scan_fastpipe_shard(p)
        fp_counts.update(c)
        parts.append((keys, scores))
    monkeypatch.setattr(mod, "_IDS", mod.build_id_index(parts))
    running, n_l, n_b = {}, mod.Counter(), mod.Counter()
    for p in lp_shards:
        heaps, a, b = mod._scan_lpv11_shard(p)
        mod.merge_bottom_k(running, heaps, mod._K)
        n_l.update(a)
        n_b.update(b)
    return {h: [rec for _, rec in items] for h, items in running.items()}, fp_counts, n_l, n_b


def test_sample_is_bottom_k_of_intersection_and_shard_invariant(corpus, monkeypatch):
    tmp_path, ids, lpv11_rows, fastpipe_rows = corpus
    (tmp_path / "one").mkdir()
    (tmp_path / "many").mkdir()
    single, fp_counts, n_l, n_b = _run_scan(tmp_path / "one", lpv11_rows, fastpipe_rows, 1, monkeypatch)
    sharded, *_ = _run_scan(tmp_path / "many", lpv11_rows, fastpipe_rows, 5, monkeypatch)

    h_a, h_b, h_other = (_warc_path_hash(w) for w in (WARC_A, WARC_B, WARC_OTHER))
    assert h_other not in single and h_other not in fp_counts and h_other not in n_l
    assert fp_counts[h_a] == 25 and fp_counts[h_b] == 20
    assert n_l[h_a] == 30 and n_b[h_a] == 20 and n_b[h_b] == 20

    for h, warc in ((h_a, WARC_A), (h_b, WARC_B)):
        expected = sorted(ids[warc][:20], key=lambda rid: mod._priority(7, rid))[:4]
        assert [r["warc_record_id"] for r in single[h]] == expected
        assert [r["warc_record_id"] for r in sharded[h]] == expected
    row = single[h_a][0]
    assert row["final_output"] == f"text {row['warc_record_id']}"
    assert row["warc_path"] == WARC_A and row["pooled_prob"] == 0.6 and row["modernbert_prob"] is None


def test_select_warcs_is_seeded_and_hashes_paths():
    manifest = [WARC_A, WARC_B, WARC_OTHER]
    first = mod.select_warcs(manifest, 2, seed=3)
    assert first == mod.select_warcs(manifest, 2, seed=3)
    assert {w["warc_hash"] for w in first} == {_warc_path_hash(w["warc_path"]) for w in first}
    assert all(w["snapshot"].startswith("CC-MAIN-") for w in first)
