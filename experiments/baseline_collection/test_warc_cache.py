# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the within-region read-through WARC cache in ``download_warcs``.

Verifies the three guarantees that matter: (1) the cache path is strictly the
worker's own region (never cross-region), (2) cache hit/miss round-trips are
byte-identical to a Common Crawl fetch — so extraction output is unchanged, and
(3) any cache miss/absence falls back to Common Crawl.
"""

import io
import os

from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

import experiments.baseline_collection.download_warcs as dw

WARC = "s3://commoncrawl/crawl-data/CC-MAIN-2099-99/segments/1/warc/CC-MAIN-test-00000.warc.gz"


def _make_synthetic_warc() -> bytes:
    """A tiny valid .warc.gz with two HTML response records (avoids downloading a real WARC)."""
    buf = io.BytesIO()
    writer = WARCWriter(buf, gzip=True)
    for url, html in [("http://a.example/", "<html>A</html>"), ("http://b.example/", "<html>B body here</html>")]:
        body = html.encode("utf-8")
        http_headers = StatusAndHeaders(
            "200 OK",
            [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))],
            protocol="HTTP/1.1",
        )
        record = writer.create_warc_record(
            url, "response", payload=io.BytesIO(body), length=len(body), http_headers=http_headers
        )
        writer.write_record(record)
    return buf.getvalue()


def test_cache_path_is_strictly_in_region(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")
    p = dw._warc_cache_path(WARC)
    assert p == f"gs://marin-us-east5/tmp/ttl=2d/warc-cache/{dw._warc_path_hash(WARC)}.warc.gz"
    # us-east5 only — never another region's bucket.
    assert "marin-us-east5" in p and p.count("gs://") == 1


def test_cache_path_uses_ttl2d_prefix(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-eu-west4")
    assert "/tmp/ttl=2d/warc-cache/" in dw._warc_cache_path(WARC)


def test_cache_disabled_for_local_prefix(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "/tmp/marin")
    assert dw._warc_cache_path(WARC) is None


def test_read_through_hit_miss_byte_identical(monkeypatch, tmp_path):
    warc_bytes = _make_synthetic_warc()
    cache_file = str(tmp_path / "cache" / f"{dw._warc_path_hash(WARC)}.warc.gz")
    monkeypatch.setattr(dw, "_warc_cache_path", lambda wp: cache_file)

    cc_calls = {"n": 0}

    def fake_cc(wp):
        cc_calls["n"] += 1
        return warc_bytes

    monkeypatch.setattr(dw, "_download_warc_bytes_from_cc", fake_cc)

    # First call: cache MISS -> Common Crawl -> populate cache.
    b1 = dw._fetch_warc_bytes(WARC)
    assert b1 == warc_bytes
    assert cc_calls["n"] == 1
    assert os.path.exists(cache_file), "cache should be populated on miss"
    assert open(cache_file, "rb").read() == warc_bytes, "cached bytes must be verbatim CC bytes"

    # Second call: cache HIT -> read local, CC NOT called again.
    b2 = dw._fetch_warc_bytes(WARC)
    assert b2 == warc_bytes
    assert cc_calls["n"] == 1, "cache hit must not re-fetch from Common Crawl"


def test_download_one_warc_identical_records_cc_vs_cache(monkeypatch, tmp_path):
    warc_bytes = _make_synthetic_warc()
    monkeypatch.setattr(dw, "_download_warc_bytes_from_cc", lambda wp: warc_bytes)

    # Path A: no cache (pure CC).
    monkeypatch.setattr(dw, "_warc_cache_path", lambda wp: None)
    recs_cc = dw._download_one_warc(WARC)

    # Path B: cache miss (writes) then hit (reads).
    monkeypatch.setattr(dw, "_warc_cache_path", lambda wp: str(tmp_path / f"{dw._warc_path_hash(WARC)}.warc.gz"))
    recs_write = dw._download_one_warc(WARC)  # miss -> write
    recs_read = dw._download_one_warc(WARC)  # hit -> read

    assert len(recs_cc) == 2, "two HTML response records expected"
    assert recs_cc == recs_write == recs_read, "records must be byte-for-byte identical regardless of source"


def test_fallback_to_cc_when_cache_unavailable(monkeypatch):
    monkeypatch.setattr(dw, "_warc_cache_path", lambda wp: None)
    calls = {"n": 0}

    def fake_cc(wp):
        calls["n"] += 1
        return _make_synthetic_warc()

    monkeypatch.setattr(dw, "_download_warc_bytes_from_cc", fake_cc)
    dw._fetch_warc_bytes(WARC)
    assert calls["n"] == 1


def test_fallback_to_cc_on_cache_read_error(monkeypatch, tmp_path):
    # Cache path points at a directory -> open('rb') raises a non-FileNotFound error;
    # the code must swallow it and fall back to Common Crawl rather than crash.
    bad = tmp_path / "adir"
    bad.mkdir()
    monkeypatch.setattr(dw, "_warc_cache_path", lambda wp: str(bad))
    calls = {"n": 0}

    def fake_cc(wp):
        calls["n"] += 1
        return _make_synthetic_warc()

    monkeypatch.setattr(dw, "_download_warc_bytes_from_cc", fake_cc)
    out = dw._fetch_warc_bytes(WARC)
    assert calls["n"] == 1 and out
