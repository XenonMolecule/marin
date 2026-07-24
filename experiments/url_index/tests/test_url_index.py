# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness + efficiency tests for the URL index.

Correctness runs the real build compute path (``_iter_rows`` → ``_write_unsorted``
→ ``_emit_artifacts``) on synthetic local shards, then consolidate → lookup →
coverage, and checks against brute-force Python set ops. Efficiency asserts
generous latency ceilings on a larger synthetic index so the mechanism (indexed
point/batch lookup, single-join coverage matrix) can't silently regress.
"""

import gzip
import json
import os
import time

import duckdb
import pytest

from experiments.infinigram.resolve import ResolvedTarget
from experiments.infinigram.targets import Collection, IndexSource, IndexTarget
from experiments.url_index import build, consolidate, coverage, layout, lookup
from experiments.url_index.keys import _normalize_record_id, domain_of, registrable_domain, url_key

# --------------------------------------------------------------------------- #
# Normalization units
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("http://www.google.com/", "google.com"),
        ("https://www.google.com/search?q=cats#top", "google.com/search?q=cats"),
        ("http://Maps.Google.com/dir/", "maps.google.com/dir"),
        ("https://example.org/a/b/?x=1&y=2", "example.org/a/b?x=1&y=2"),
        ("example.com", "example.com"),
    ],
)
def test_url_key(raw, expected):
    assert url_key(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://www.google.com/x", "google.com"),
        ("http://maps.google.com/", "google.com"),
        ("https://bbc.co.uk/news", "bbc.co.uk"),
        ("https://sub.example.com.au/p", "example.com.au"),
        ("https://foo.github.io/repo", "foo.github.io"),
    ],
)
def test_domain_etld1(raw, expected):
    assert domain_of(raw) == expected


def test_registrable_domain_bare_host():
    assert registrable_domain("a.b.google.com") == "google.com"
    assert registrable_domain("google.com") == "google.com"


def test_normalize_record_id():
    assert _normalize_record_id("<urn:uuid:ABC-123>") == "ABC-123"
    assert _normalize_record_id("urn:uuid:xyz") == "xyz"
    assert _normalize_record_id("plain") == "plain"


# --------------------------------------------------------------------------- #
# Fixtures / helpers: run the real build compute path on synthetic shards
# --------------------------------------------------------------------------- #


def _write_shard(path: str, docs: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d))
            f.write("\n")


def _build_local(tmp: str, dataset: str, docs: list[dict], provenance_globs: tuple = (), keys_only: bool = False) -> str:
    """Run _iter_rows/_write_unsorted/_emit_artifacts for one synthetic dataset.

    Returns the local output dir holding keys/meta/text parquet.
    """
    shard = os.path.join(tmp, f"{dataset}_shard.jsonl.gz")
    _write_shard(shard, docs)
    target = IndexTarget(
        dataset=dataset,
        collection=Collection.SMALL,
        region="us-central1",
        source=IndexSource.at("x"),
        provenance_globs=provenance_globs,
    )
    resolved = ResolvedTarget(target=target, shard_urls=(shard,), shard_bytes=(0,))

    prov_map: dict = {}
    if provenance_globs:
        wanted = build._collect_wanted_hashes(resolved.shard_urls)
        from experiments.infinigram.provenance import build_provenance_map

        prov_map = build_provenance_map(provenance_globs, wanted)

    staging = os.path.join(tmp, f"{dataset}_staging.parquet")
    build._write_unsorted(build._iter_rows(resolved, prov_map), staging)
    out_dir = os.path.join(tmp, f"{dataset}_out")
    os.makedirs(out_dir, exist_ok=True)
    build._emit_artifacts(staging, dataset, out_dir, keys_only=keys_only)
    return out_dir


def _doc(url, rid, text, snapshot="CC-MAIN-2020-05", warc="w.warc.gz"):
    return {"url": url, "warc_record_id": rid, "text": text, "snapshot": snapshot, "warc_file": warc}


@pytest.fixture
def two_datasets(tmp_path):
    """Datasets A and B sharing rids {r2, r3}; urls u2 in both."""
    tmp = str(tmp_path)
    a_docs = [
        _doc("http://u1.com/", "r1", "A extracted one"),
        _doc("http://u2.com/p", "r2", "A extracted two"),
        _doc("http://www.u3.com/x", "r3", "identical three body"),
    ]
    b_docs = [
        _doc("http://u2.com/p", "r2", "B extracted two DIFFERENT"),
        _doc("http://u3.com/x", "r3", "identical three body"),
        _doc("http://u4.com/", "r4", "B extracted four"),
    ]
    a_dir = _build_local(tmp, "dsA", a_docs)
    b_dir = _build_local(tmp, "dsB", b_docs)
    return tmp, a_dir, b_dir


# --------------------------------------------------------------------------- #
# Build correctness
# --------------------------------------------------------------------------- #


def test_build_emits_expected_columns_and_counts(two_datasets):
    _tmp, a_dir, _b = two_datasets
    con = duckdb.connect()
    n = con.execute(f"SELECT count(*) FROM read_parquet('{a_dir}/{layout.TEXT_NAME}')").fetchone()[0]
    assert n == 3
    row = con.execute(
        f"SELECT dataset, url_key, domain, warc_record_id, text FROM read_parquet('{a_dir}/{layout.TEXT_NAME}') "
        "WHERE url_key = 'u2.com/p'"
    ).fetchone()
    assert row == ("dsA", "u2.com/p", "u2.com", "r2", "A extracted two")
    # keys carries only the hash columns
    keycols = [
        c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{a_dir}/{layout.KEYS_NAME}')").fetchall()
    ]
    assert set(keycols) == {"dataset", "url_h", "rid_h", "text_h", "dom_h"}
    con.close()


def test_subset_filter_restricts_to_manifest_warcs(tmp_path):
    tmp = str(tmp_path)
    # A 10k-style tier carrying file_path; keep only docs from the manifest WARCs.
    docs = [
        {"url": "http://a.com/1", "warc_record_id": "r1", "text": "keep a", "file_path": "s3://commoncrawl/WARC-A.gz"},
        {"url": "http://b.com/2", "warc_record_id": "r2", "text": "drop b", "file_path": "s3://commoncrawl/WARC-B.gz"},
        {"url": "http://c.com/3", "warc_record_id": "r3", "text": "keep c", "file_path": "s3://commoncrawl/WARC-A.gz"},
    ]
    shard = os.path.join(tmp, "src.jsonl.gz")
    _write_shard(shard, docs)
    manifest = os.path.join(tmp, "manifest.txt")
    with open(manifest, "w") as f:
        f.write("# header\ns3://commoncrawl/WARC-A.gz\n")

    sub = build.build_subset_filter("file_path", manifest, None)
    assert sub.field == "file_path" and "s3://commoncrawl/WARC-A.gz" in sub.keys
    target = IndexTarget(dataset="ds", collection=Collection.SMALL, region="us-central1", source=IndexSource.at("x"))
    resolved = ResolvedTarget(target=target, shard_urls=(shard,), shard_bytes=(0,))
    rows = list(build._iter_rows(resolved, {}, subset=sub))
    assert {r["url_key"] for r in rows} == {"a.com/1", "c.com/3"}  # WARC-B dropped


def test_min_field_filters_quality_band(tmp_path):
    tmp = str(tmp_path)
    docs = [
        {"url": "http://a.com/1", "text": "low", "modernbert_prob": 0.2},
        {"url": "http://b.com/2", "text": "mid", "modernbert_prob": 0.5},
        {"url": "http://c.com/3", "text": "high", "modernbert_prob": 0.9},
    ]
    shard = os.path.join(tmp, "kt.jsonl.gz")
    _write_shard(shard, docs)
    target = IndexTarget(dataset="fp", collection=Collection.SMALL, region="us-east5", source=IndexSource.at("x"))
    resolved = ResolvedTarget(target=target, shard_urls=(shard,), shard_bytes=(0,))
    rows = list(build._iter_rows(resolved, {}, min_field="modernbert_prob", min_value=0.49377))
    assert {r["url_key"] for r in rows} == {"b.com/2", "c.com/3"}  # 0.2 dropped


def test_resumable_build_resumes_and_matches(tmp_path):
    # Two source shards; resumable build writes per-shard parts + concats. A second
    # run (parts present) must skip work and produce the identical output.
    tmp = str(tmp_path)
    out_dir = os.path.join(tmp, "out")
    s1 = os.path.join(tmp, "s1.jsonl.gz")
    s2 = os.path.join(tmp, "s2.jsonl.gz")
    _write_shard(s1, [_doc("http://a.com/1", "r1", "aa"), _doc("http://b.com/2", "r2", "bb")])
    _write_shard(s2, [_doc("http://c.com/3", "r3", "cc")])
    target = IndexTarget(dataset="ds", collection=Collection.SMALL, region="us-central1", source=IndexSource.at("x"))
    resolved = ResolvedTarget(target=target, shard_urls=(s1, s2), shard_bytes=(0, 0))
    n, up, names = build._resumable_emit(
        resolved,
        "ds",
        out_dir,
        keys_only=False,
        prov_map={},
        subset=None,
        min_field=None,
        min_value=0.0,
        overwrite=False,
    )
    assert n == 3 and up == 3 and set(names) == {"keys.parquet", "meta.parquet", "text.parquet"}
    con = duckdb.connect()
    urls = {r[0] for r in con.execute(f"SELECT url_key FROM read_parquet('{out_dir}/meta.parquet')").fetchall()}
    assert urls == {"a.com/1", "b.com/2", "c.com/3"}
    # second run: markers exist -> skips shard work, same result
    n2, _up2, _ = build._resumable_emit(
        resolved,
        "ds",
        out_dir,
        keys_only=False,
        prov_map={},
        subset=None,
        min_field=None,
        min_value=0.0,
        overwrite=False,
    )
    assert n2 == 3
    con.close()


def test_keys_only_skips_text_store(tmp_path):
    tmp = str(tmp_path)
    docs = [_doc("http://u1.com/", "r1", "raw universe text")]
    out_dir = _build_local(tmp, "big_raw", docs, keys_only=True)
    assert os.path.exists(f"{out_dir}/{layout.KEYS_NAME}")
    assert not os.path.exists(f"{out_dir}/{layout.TEXT_NAME}")
    assert not os.path.exists(f"{out_dir}/{layout.META_NAME}")


def test_text_only_tier_recovers_url_via_provenance(tmp_path):
    tmp = str(tmp_path)
    # Raw batch carries url + ids; deduped tier carries text only.
    raw = os.path.join(tmp, "raw_batch.jsonl.gz")
    _write_shard(raw, [_doc("http://recovered.com/here", "rid-9", "unique dedup text body")])
    text_only = [{"text": "unique dedup text body"}]
    out_dir = _build_local(tmp, "dsC", text_only, provenance_globs=(raw,))
    con = duckdb.connect()
    row = con.execute(
        f"SELECT url_key, domain, warc_record_id FROM read_parquet('{out_dir}/{layout.TEXT_NAME}')"
    ).fetchone()
    con.close()
    assert row == ("recovered.com/here", "recovered.com", "rid-9")


# --------------------------------------------------------------------------- #
# Coverage correctness vs brute-force set ops
# --------------------------------------------------------------------------- #


def test_coverage_matches_bruteforce(two_datasets):
    _tmp, a_dir, b_dir = two_datasets
    keys = [f"{a_dir}/{layout.KEYS_NAME}", f"{b_dir}/{layout.KEYS_NAME}"]
    datasets, sizes, rows = coverage.compute(keys, "rid_h")
    assert datasets == ["dsA", "dsB"]
    assert sizes == {"dsA": 3, "dsB": 3}

    a_rids, b_rids = {"r1", "r2", "r3"}, {"r2", "r3", "r4"}
    ab = next(r for r in rows if r["a"] == "dsA" and r["b"] == "dsB")
    assert ab["intersection"] == len(a_rids & b_rids) == 2
    assert ab["a_only"] == len(a_rids - b_rids) == 1
    assert ab["b_only"] == len(b_rids - a_rids) == 1
    assert ab["containment_a_in_b"] == pytest.approx(2 / 3)
    assert ab["jaccard"] == pytest.approx(2 / 4)


def test_coverage_url_key_works_without_record_id(tmp_path):
    # nemotron/fineweb_edu/resiliparse carry url but no warc_record_id -> url_h is
    # the universal coverage key; rid_h would be null and exclude them.
    tmp = str(tmp_path)
    a = _build_local(tmp, "no_rid", [{"url": "http://x.com/1", "text": "t1"}, {"url": "http://y.com/2", "text": "t2"}])
    b = _build_local(tmp, "with_rid", [_doc("http://x.com/1", "r9", "u"), _doc("http://z.com/3", "r10", "v")])
    keys = [f"{a}/{layout.KEYS_NAME}", f"{b}/{layout.KEYS_NAME}"]
    _ds, sizes, rows = coverage.compute(keys, "url_h")
    assert sizes == {"no_rid": 2, "with_rid": 2}
    ab = next(r for r in rows if r["a"] == "no_rid" and r["b"] == "with_rid")
    assert ab["intersection"] == 1  # x.com/1 shared
    # rid_h excludes the no_rid dataset entirely (all rid_h null)
    _ds2, sizes2, _ = coverage.compute(keys, "rid_h")
    assert "no_rid" not in sizes2 and sizes2.get("with_rid") == 2


def test_coverage_text_hash_disjoint_for_different_extractors(two_datasets):
    # Same source pages (rid), different extracted text -> text_h overlap is 0 except identical strings.
    _tmp, a_dir, b_dir = two_datasets
    keys = [f"{a_dir}/{layout.KEYS_NAME}", f"{b_dir}/{layout.KEYS_NAME}"]
    _ds, _sizes, rows = coverage.compute(keys, "text_h")
    ab = next(r for r in rows if r["a"] == "dsA" and r["b"] == "dsB")
    # only "... extracted three" matches byte-for-byte between A and B
    assert ab["intersection"] == 1


# --------------------------------------------------------------------------- #
# Lookup correctness (consolidate -> query)
# --------------------------------------------------------------------------- #


def _consolidate(tmp, dirs) -> str:
    metas = [f"{d}/{layout.META_NAME}" for d in dirs]
    db = os.path.join(tmp, "lookup.duckdb")
    consolidate.build_index(metas, Collection.SMALL, db)
    return db


def test_lookup_exact_url_hits_both_datasets(two_datasets):
    tmp, a_dir, b_dir = two_datasets
    db = _consolidate(tmp, [a_dir, b_dir])
    con = duckdb.connect(db, read_only=True)
    rows = lookup.lookup_url(con, "http://www.u2.com/p")
    con.close()
    assert {r["dataset"] for r in rows} == {"dsA", "dsB"}
    assert all(r["url_key"] == "u2.com/p" for r in rows)


def test_lookup_domain_returns_all_pages(two_datasets):
    tmp, a_dir, b_dir = two_datasets
    db = _consolidate(tmp, [a_dir, b_dir])
    con = duckdb.connect(db, read_only=True)
    rows = lookup.lookup_domain(con, "u3.com")
    con.close()
    assert {r["dataset"] for r in rows} == {"dsA", "dsB"}


def test_lookup_with_text_from_local_text_dir(two_datasets):
    tmp, a_dir, b_dir = two_datasets
    db = _consolidate(tmp, [a_dir, b_dir])
    # Lay out downloaded text.parquet copies as {text_dir}/{collection}/{dataset}/text.parquet
    text_dir = os.path.join(tmp, "textcache")
    for dataset, d in (("dsA", a_dir), ("dsB", b_dir)):
        dest = os.path.join(text_dir, "small", dataset)
        os.makedirs(dest, exist_ok=True)
        os.link(f"{d}/{layout.TEXT_NAME}", f"{dest}/{layout.TEXT_NAME}")
    con = duckdb.connect(db, read_only=True)
    rows = lookup.lookup_url(con, "u2.com/p")
    texts = lookup.fetch_texts(rows, Collection.SMALL, text_dir)
    con.close()
    assert texts[("dsA", "u2.com/p")] == "A extracted two"
    assert texts[("dsB", "u2.com/p")] == "B extracted two DIFFERENT"


# --------------------------------------------------------------------------- #
# Efficiency
# --------------------------------------------------------------------------- #


@pytest.fixture
def big_index(tmp_path):
    """~20k-doc lookup index for latency benchmarks."""
    tmp = str(tmp_path)
    docs = [_doc(f"http://site{i % 2000}.com/page{i}", f"rid-{i}", f"text {i}") for i in range(20_000)]
    d = _build_local(tmp, "big", docs)
    db = _consolidate(tmp, [d])
    return db, d


def test_point_lookup_latency(big_index):
    db, _d = big_index
    con = duckdb.connect(db, read_only=True)
    lookup.lookup_url(con, "http://site7.com/page7")  # warm
    t0 = time.perf_counter()
    rows = lookup.lookup_url(con, "http://site123.com/page4123")
    dt = time.perf_counter() - t0
    con.close()
    assert rows and dt < 0.05, f"point lookup took {dt * 1000:.1f}ms"


def test_batch_lookup_latency(big_index):
    db, _d = big_index
    con = duckdb.connect(db, read_only=True)
    targets = [f"http://site{i}.com/page{i}" for i in range(10)]
    for q in targets:  # warm
        lookup.lookup_url(con, q)
    t0 = time.perf_counter()
    for q in targets:
        lookup.lookup_url(con, q)
    dt = time.perf_counter() - t0
    con.close()
    assert dt < 0.3, f"batch of 10 took {dt * 1000:.1f}ms"


def test_coverage_matrix_latency(big_index):
    _db, d = big_index
    t0 = time.perf_counter()
    _ds, _sizes, _rows = coverage.compute([f"{d}/{layout.KEYS_NAME}"], "rid_h")
    dt = time.perf_counter() - t0
    assert dt < 5.0, f"coverage took {dt:.2f}s"
