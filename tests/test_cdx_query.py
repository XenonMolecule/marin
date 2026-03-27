# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CDX query and WARC record download modules."""

import json
import os
import tempfile

import pytest

from marin.datakit.download.commoncrawl.cdx_query import (
    CDXQueryConfig,
    _load_progress,
    _progress_key,
    dedup_cdx_records,
    filter_cdx_records,
    get_available_crawl_indices,
    query_cdx,
    query_single_index,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
def _make_cdx_record(
    url="http://example.com/page1",
    status="200",
    mime="text/html",
    timestamp="20240101120000",
    filename="crawl-data/CC-MAIN-2024-01/warc.gz",
    offset="1000",
    length="5000",
):
    return {
        "url": url,
        "status": status,
        "mime": mime,
        "timestamp": timestamp,
        "filename": filename,
        "offset": offset,
        "length": length,
    }


# ---------------------------------------------------------------------------
# Unit tests (no network)
# ---------------------------------------------------------------------------
class TestFilterCdxRecords:
    def test_status_filter_keeps_200(self):
        records = [
            _make_cdx_record(status="200"),
            _make_cdx_record(status="301", url="http://example.com/redirect"),
            _make_cdx_record(status="404", url="http://example.com/missing"),
        ]
        result = filter_cdx_records(records, status_filter=["200"], mime_filter=[])
        assert len(result) == 1
        assert result[0]["status"] == "200"

    def test_status_filter_multiple_codes(self):
        records = [
            _make_cdx_record(status="200"),
            _make_cdx_record(status="301", url="http://example.com/r"),
        ]
        result = filter_cdx_records(records, status_filter=["200", "301"], mime_filter=[])
        assert len(result) == 2

    def test_mime_filter(self):
        records = [
            _make_cdx_record(mime="text/html"),
            _make_cdx_record(mime="application/pdf", url="http://example.com/doc.pdf"),
            _make_cdx_record(mime="text/html; charset=utf-8", url="http://example.com/page2"),
        ]
        result = filter_cdx_records(records, status_filter=[], mime_filter=["text/html"])
        assert len(result) == 2

    def test_empty_filters_pass_all(self):
        records = [
            _make_cdx_record(status="404", mime="application/json"),
        ]
        result = filter_cdx_records(records, status_filter=[], mime_filter=[])
        assert len(result) == 1

    def test_empty_input(self):
        result = filter_cdx_records([], status_filter=["200"], mime_filter=["text/html"])
        assert result == []


class TestDedupCdxRecords:
    def test_keeps_most_recent(self):
        records = [
            _make_cdx_record(url="http://example.com/page", timestamp="20230101120000"),
            _make_cdx_record(url="http://example.com/page", timestamp="20240601120000"),
            _make_cdx_record(url="http://example.com/page", timestamp="20240101120000"),
        ]
        result = dedup_cdx_records(records)
        assert len(result) == 1
        assert result[0]["timestamp"] == "20240601120000"

    def test_different_urls_preserved(self):
        records = [
            _make_cdx_record(url="http://example.com/page1"),
            _make_cdx_record(url="http://example.com/page2"),
        ]
        result = dedup_cdx_records(records)
        assert len(result) == 2

    def test_empty_input(self):
        assert dedup_cdx_records([]) == []


class TestQueryCdx:
    def test_writes_manifest_to_local_path(self, monkeypatch):
        """Test that query_cdx writes output files correctly (mocked HTTP)."""
        # Mock query_single_index to avoid real HTTP
        mock_records = [
            _make_cdx_record(url="http://test.com/a"),
            _make_cdx_record(url="http://test.com/b", status="301"),
        ]

        def mock_query(pattern, crawl_id, match_type, **kwargs):
            return mock_records

        monkeypatch.setattr(
            "marin.download.commoncrawl.cdx_query.query_single_index",
            mock_query,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CDXQueryConfig(
                url_patterns=["test.com"],
                output_path=tmpdir,
                crawl_indices=["CC-MAIN-2024-01"],
                request_delay=0,
            )
            query_cdx(config)

            manifest_path = os.path.join(tmpdir, "cdx_manifest.json")
            assert os.path.exists(manifest_path)

            with open(manifest_path) as f:
                result = json.load(f)

            # Should have only the 200 record (301 filtered out)
            assert len(result) == 1
            assert result[0]["url"] == "http://test.com/a"

            # Stats file should exist
            stats_path = os.path.join(tmpdir, "cdx_stats.json")
            assert os.path.exists(stats_path)

    def test_dedup_across_indices(self, monkeypatch):
        """Records for the same URL across indices are deduplicated."""
        call_count = 0

        def mock_query(pattern, crawl_id, match_type, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [_make_cdx_record(url="http://test.com/same", timestamp="20230101000000")]
            else:
                return [_make_cdx_record(url="http://test.com/same", timestamp="20240101000000")]

        monkeypatch.setattr(
            "marin.download.commoncrawl.cdx_query.query_single_index",
            mock_query,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CDXQueryConfig(
                url_patterns=["test.com"],
                output_path=tmpdir,
                crawl_indices=["CC-MAIN-2023-01", "CC-MAIN-2024-01"],
                request_delay=0,
            )
            query_cdx(config)

            with open(os.path.join(tmpdir, "cdx_manifest.json")) as f:
                result = json.load(f)

            assert len(result) == 1
            assert result[0]["timestamp"] == "20240101000000"


# ---------------------------------------------------------------------------
# Checkpoint / resume tests
# ---------------------------------------------------------------------------
class TestProgressKey:
    def test_sanitizes_special_chars(self):
        key = _progress_key("stackoverflow.com/questions", "CC-MAIN-2024-30")
        assert key == "stackoverflow.com_questions__CC-MAIN-2024-30"

    def test_deterministic(self):
        a = _progress_key("example.com", "CC-MAIN-2024-01")
        b = _progress_key("example.com", "CC-MAIN-2024-01")
        assert a == b

    def test_different_patterns_differ(self):
        a = _progress_key("a.com", "CC-MAIN-2024-01")
        b = _progress_key("b.com", "CC-MAIN-2024-01")
        assert a != b


class TestCheckpointResumption:
    def test_resumes_from_checkpoint(self, monkeypatch):
        """Completed queries are loaded from .progress/ and not re-fetched."""
        call_count = 0

        def mock_query(pattern, crawl_id, match_type, **kwargs):
            nonlocal call_count
            call_count += 1
            return [_make_cdx_record(url=f"http://test.com/{crawl_id}")]

        monkeypatch.setattr(
            "marin.download.commoncrawl.cdx_query.query_single_index",
            mock_query,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CDXQueryConfig(
                url_patterns=["test.com"],
                output_path=tmpdir,
                crawl_indices=["CC-MAIN-2024-01", "CC-MAIN-2024-02"],
                request_delay=0,
            )

            # First run — both queries should be fetched
            query_cdx(config)
            assert call_count == 2

            # Progress files should exist
            progress_dir = os.path.join(tmpdir, ".progress")
            assert os.path.isdir(progress_dir)
            progress_files = os.listdir(progress_dir)
            assert len(progress_files) == 2

            # Second run — should skip both queries (loaded from cache)
            call_count = 0
            query_cdx(config)
            assert call_count == 0

            # Manifest should still be correct
            with open(os.path.join(tmpdir, "cdx_manifest.json")) as f:
                result = json.load(f)
            assert len(result) == 2

    def test_failed_queries_not_cached(self, monkeypatch):
        """Failed queries should NOT be saved to .progress/, so they get retried."""
        call_count = 0

        def mock_query(pattern, crawl_id, match_type, **kwargs):
            nonlocal call_count
            call_count += 1
            if crawl_id == "CC-MAIN-2024-02":
                raise RuntimeError("CDX API 503")
            return [_make_cdx_record(url=f"http://test.com/{crawl_id}")]

        monkeypatch.setattr(
            "marin.download.commoncrawl.cdx_query.query_single_index",
            mock_query,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CDXQueryConfig(
                url_patterns=["test.com"],
                output_path=tmpdir,
                crawl_indices=["CC-MAIN-2024-01", "CC-MAIN-2024-02"],
                request_delay=0,
            )

            query_cdx(config)
            assert call_count == 2

            # Only the successful query should be cached
            progress = _load_progress(tmpdir)
            assert len(progress) == 1

            # Second run should re-attempt the failed query
            call_count = 0
            query_cdx(config)
            # The successful one is cached (0 calls), the failed one retries (1 call)
            assert call_count == 1

    def test_partial_resume_extends_correctly(self, monkeypatch):
        """After partial completion, new queries extend the cached data."""
        call_count = 0

        def mock_query(pattern, crawl_id, match_type, **kwargs):
            nonlocal call_count
            call_count += 1
            return [_make_cdx_record(url=f"http://test.com/{crawl_id}", timestamp="20240101000000")]

        monkeypatch.setattr(
            "marin.download.commoncrawl.cdx_query.query_single_index",
            mock_query,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run with 1 index first
            config1 = CDXQueryConfig(
                url_patterns=["test.com"],
                output_path=tmpdir,
                crawl_indices=["CC-MAIN-2024-01"],
                request_delay=0,
            )
            query_cdx(config1)
            assert call_count == 1

            # Run again with 2 indices — first should be cached
            call_count = 0
            config2 = CDXQueryConfig(
                url_patterns=["test.com"],
                output_path=tmpdir,
                crawl_indices=["CC-MAIN-2024-01", "CC-MAIN-2024-02"],
                request_delay=0,
            )
            query_cdx(config2)
            assert call_count == 1  # only the new index queried

            with open(os.path.join(tmpdir, "cdx_manifest.json")) as f:
                result = json.load(f)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration tests (hit real CDX API)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestCdxQueryIntegration:
    def test_cdx_query_small_domain(self):
        """Query the real CDX API for example.com from a single crawl index."""
        records = query_single_index(
            url_pattern="example.com",
            crawl_id="CC-MAIN-2024-10",
            match_type="exact",
        )
        assert len(records) > 0
        # Each record should have the required fields
        for r in records:
            assert "url" in r
            assert "filename" in r
            assert "offset" in r
            assert "length" in r
            assert "timestamp" in r

    def test_get_available_indices(self):
        """Verify we can fetch the list of available crawl indices."""
        indices = get_available_crawl_indices()
        assert len(indices) > 10
        assert any("CC-MAIN-2024" in idx for idx in indices)

    def test_full_query_pipeline(self):
        """End-to-end: query, filter, dedup, write manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CDXQueryConfig(
                url_patterns=["example.com"],
                output_path=tmpdir,
                crawl_indices=["CC-MAIN-2024-10"],
                match_type="exact",
                request_delay=0,
            )
            query_cdx(config)

            manifest_path = os.path.join(tmpdir, "cdx_manifest.json")
            assert os.path.exists(manifest_path)

            with open(manifest_path) as f:
                result = json.load(f)

            assert len(result) > 0


@pytest.mark.integration
class TestWarcDownloadIntegration:
    def test_download_single_warc_record(self):
        """Download a single WARC record using a real CDX entry for example.com."""
        from marin.datakit.download.commoncrawl.download_warc_records import _download_single_record

        # First get a real CDX entry
        records = query_single_index(
            url_pattern="example.com",
            crawl_id="CC-MAIN-2024-10",
            match_type="exact",
        )
        assert len(records) > 0, "No CDX records found for example.com"

        filtered = filter_cdx_records(records, status_filter=["200"], mime_filter=["text/html"])
        assert len(filtered) > 0, "No 200/text/html records for example.com"

        entry = filtered[0]
        result = _download_single_record(entry)

        assert result is not None, f"Download failed for {entry['url']}"
        assert result["html"], "Downloaded HTML is empty"
        assert result["content_length"] > 0
        assert "example" in result["url"].lower()
