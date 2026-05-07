# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CDX-based WARC metadata extraction."""

import gzip
from unittest import mock

import pytest

from experiments.baseline_collection.extract_metadata_from_cc_cdx import (
    _group_warcs_by_snapshot,
    _list_cdx_files,
    _parse_cdx_line,
    _process_one_cdx,
    _to_relative,
)

# --- Unit tests (no network) --------------------------------------------------


def test_to_relative_strips_s3_prefix():
    assert (
        _to_relative("s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/x/warc/y.warc.gz")
        == "crawl-data/CC-MAIN-2013-20/segments/x/warc/y.warc.gz"
    )


def test_to_relative_passes_through_already_relative():
    assert _to_relative("crawl-data/CC-MAIN-2013-20/x.warc.gz") == "crawl-data/CC-MAIN-2013-20/x.warc.gz"


def test_group_warcs_by_snapshot_groups_and_normalizes(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/a/warc/a.warc.gz",
                "s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/b/warc/b.warc.gz",
                "s3://commoncrawl/crawl-data/CC-MAIN-2017-26/segments/c/warc/c.warc.gz",
                # Blank line and comment should be ignored
                "",
                "# a comment",
            ]
        )
    )
    result = _group_warcs_by_snapshot(str(manifest))
    assert set(result.keys()) == {"CC-MAIN-2013-20", "CC-MAIN-2017-26"}
    assert len(result["CC-MAIN-2013-20"]) == 2
    assert len(result["CC-MAIN-2017-26"]) == 1
    # Values are stored in relative (no s3://) form
    for snap, paths in result.items():
        for p in paths:
            assert not p.startswith("s3://")
            assert snap in p


def test_group_warcs_by_snapshot_rejects_unparseable_path(tmp_path):
    manifest = tmp_path / "bad.txt"
    manifest.write_text("s3://commoncrawl/some/path/without/snapshot/marker.warc.gz\n")
    with pytest.raises(ValueError, match="snapshot"):
        _group_warcs_by_snapshot(str(manifest))


def test_parse_cdx_line_handles_well_formed():
    line = (
        b"com,example)/page 20130101000000 "
        b'{"url": "http://example.com/page", "status": "200", '
        b'"filename": "crawl-data/CC-MAIN-2013-20/segments/x/warc/y.warc.gz"}\n'
    )
    meta = _parse_cdx_line(line)
    assert meta is not None
    assert meta["url"] == "http://example.com/page"
    assert meta["filename"].endswith(".warc.gz")


def test_parse_cdx_line_returns_none_on_malformed():
    assert _parse_cdx_line(b"") is None
    assert _parse_cdx_line(b"garbage with no json\n") is None
    assert _parse_cdx_line(b"key timestamp {not-valid-json\n") is None


def _make_fake_cdx_bytes(records: list[dict]) -> bytes:
    """Build a gzipped CDX blob from dict records."""
    import json

    lines: list[bytes] = []
    for r in records:
        surt = "com,example)/"
        ts = "20130101000000"
        lines.append(f"{surt} {ts} {json.dumps(r)}\n".encode())
    return gzip.compress(b"".join(lines))


def test_process_one_cdx_filters_by_filename_set():
    """End-to-end _process_one_cdx with a mocked HTTP CDX response."""
    wanted = "crawl-data/CC-MAIN-2013-20/segments/WANT/warc/w.warc.gz"
    unwanted = "crawl-data/CC-MAIN-2013-20/segments/SKIP/warc/s.warc.gz"

    fake_records = [
        {"url": "http://example.com/a", "filename": wanted, "status": "200"},
        {"url": "http://example.com/b", "filename": unwanted, "status": "200"},
        {"url": "http://example.com/c", "filename": wanted, "status": "200"},
        # Record with empty URL should be dropped
        {"url": "", "filename": wanted, "status": "200"},
    ]
    fake_body = _make_fake_cdx_bytes(fake_records)

    # Mock the Zephyr worker context to return our filename set.
    worker_ctx = mock.MagicMock()
    worker_ctx.get_shared.return_value = {"CC-MAIN-2013-20": {wanted}}

    with (
        mock.patch(
            "experiments.baseline_collection.extract_metadata_from_cc_cdx._http_get_bytes_with_retry",
            return_value=fake_body,
        ),
        mock.patch(
            "experiments.baseline_collection.extract_metadata_from_cc_cdx.zephyr_worker_ctx",
            return_value=worker_ctx,
        ),
    ):
        task = {"snapshot": "CC-MAIN-2013-20", "cdx_rel": "cc-index/collections/CC-MAIN-2013-20/indexes/cdx-00000.gz"}
        out = _process_one_cdx(task)

    # Two records match the wanted filename, one has empty URL (dropped), one is unwanted filename.
    assert len(out) == 2
    urls = {r["url"] for r in out}
    assert urls == {"http://example.com/a", "http://example.com/c"}
    # Schema conformance: every record has the four fields extract_warc_metadata emits.
    for r in out:
        assert set(r.keys()) == {"warc_record_id", "url", "warc_file", "snapshot"}
        assert r["snapshot"] == "CC-MAIN-2013-20"
        assert r["warc_file"].startswith("s3://commoncrawl/")
        assert r["warc_file"].endswith(wanted)


def test_output_schema_matches_extract_warc_metadata():
    """Regression guard: the JSONL fields match extract_warc_metadata.py exactly."""
    from experiments.baseline_collection.extract_warc_metadata import _extract_metadata

    sample_download_record = {
        "id": "<urn:uuid:abc-123>",
        "url": "http://x/y",
        "metadata": {"warc_file": "s3://commoncrawl/crawl-data/CC-MAIN-2013-20/warc/x.warc.gz"},
    }
    reference = _extract_metadata(sample_download_record)
    # Our CDX extractor emits the same four field names.
    expected_fields = {"warc_record_id", "url", "warc_file", "snapshot"}
    assert set(reference.keys()) == expected_fields


# --- Integration tests (real network; opt-in) --------------------------------


@pytest.mark.integration
def test_list_cdx_files_real_snapshot():
    """Verify we can list the 302 CDX files for a real snapshot."""
    paths = _list_cdx_files("CC-MAIN-2013-20")
    assert len(paths) >= 200  # always 302 in practice, but be lenient
    for p in paths:
        assert "/indexes/cdx-" in p
        assert p.endswith(".gz")


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_process_one_cdx_live_small_warc():
    """Hit a real CDX file for a known WARC in the 3000 manifest and expect matches.

    Slow: each CDX file is ~350 MB gzipped and CC's CDN throttles single-connection
    throughput to a few MB/s. Gated with ``--run-integration`` and a 10-minute timeout.
    """
    import json

    # Use the WARC we already know has records from earlier audit: shard 00100 in 2013-48.
    known_rel = (
        "crawl-data/CC-MAIN-2013-48/segments/1386163054096/warc/"
        "CC-MAIN-20131204131734-00071-ip-10-33-133-15.ec2.internal.warc.gz"
    )

    paths = _list_cdx_files("CC-MAIN-2013-48")
    # Scan the first few CDX files — the WARC's URLs are surt-scattered across all of them,
    # but ~40k records / 302 files ≈ 130 matches expected per CDX file on average.
    hits = 0
    for cdx_rel in paths[:3]:
        worker_ctx = mock.MagicMock()
        worker_ctx.get_shared.return_value = {"CC-MAIN-2013-48": {known_rel}}
        with mock.patch(
            "experiments.baseline_collection.extract_metadata_from_cc_cdx.zephyr_worker_ctx",
            return_value=worker_ctx,
        ):
            records = _process_one_cdx({"snapshot": "CC-MAIN-2013-48", "cdx_rel": cdx_rel})
        hits += len(records)
        for r in records:
            assert r["snapshot"] == "CC-MAIN-2013-48"
            assert r["warc_file"].endswith(known_rel)
            assert r["url"].startswith(("http://", "https://"))
            json.dumps(r)  # serializable
    assert hits > 0, "expected at least one matching record across 3 CDX files"
