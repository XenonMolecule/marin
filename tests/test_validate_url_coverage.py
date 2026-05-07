# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Nemotron-CC URL coverage validator."""

import gzip
import json
from unittest import mock

from experiments.baseline_collection.validate_url_coverage import _scan_one_cdx


def _make_fake_cdx_bytes(records: list[dict]) -> bytes:
    lines = []
    for r in records:
        surt = "com,example)/"
        ts = "20130101000000"
        lines.append(f"{surt} {ts} {json.dumps(r)}\n".encode())
    return gzip.compress(b"".join(lines))


def test_scan_one_cdx_counts_hits_and_misses():
    """A CDX file with 4 lines, 2 of which are in the Nemotron set, should yield 2 hits."""
    nemotron_set = {"http://x/want1", "http://x/want2", "http://x/want3"}
    cdx_records = [
        {"url": "http://x/want1", "filename": "anyfile.warc.gz"},
        {"url": "http://x/random1", "filename": "anyfile.warc.gz"},
        {"url": "http://x/want2", "filename": "anyfile.warc.gz"},
        {"url": "", "filename": "anyfile.warc.gz"},  # empty URL skipped
    ]
    fake_body = _make_fake_cdx_bytes(cdx_records)

    worker_ctx = mock.MagicMock()
    worker_ctx.get_shared.return_value = nemotron_set

    with (
        mock.patch(
            "experiments.baseline_collection.validate_url_coverage._http_get_bytes_with_retry",
            return_value=fake_body,
        ),
        mock.patch(
            "experiments.baseline_collection.validate_url_coverage.zephyr_worker_ctx",
            return_value=worker_ctx,
        ),
    ):
        chunks = _scan_one_cdx({"cdx_rel": "fake/cdx-00000.gz"})

    assert len(chunks) == 1
    assert chunks[0]["scanned"] == 4
    assert sorted(chunks[0]["hits"]) == ["http://x/want1", "http://x/want2"]


def test_scan_one_cdx_dedupes_repeated_hits_only_at_aggregation():
    """``hits`` returns each hit individually; the validate() driver dedupes via set union.

    This guards the contract: if the same URL appears twice in CDX, _scan_one_cdx
    reports both. The driver collapses with set semantics.
    """
    nemotron_set = {"http://x/repeat"}
    cdx_records = [
        {"url": "http://x/repeat", "filename": "a.warc.gz"},
        {"url": "http://x/repeat", "filename": "b.warc.gz"},
    ]
    fake_body = _make_fake_cdx_bytes(cdx_records)

    worker_ctx = mock.MagicMock()
    worker_ctx.get_shared.return_value = nemotron_set

    with (
        mock.patch(
            "experiments.baseline_collection.validate_url_coverage._http_get_bytes_with_retry",
            return_value=fake_body,
        ),
        mock.patch(
            "experiments.baseline_collection.validate_url_coverage.zephyr_worker_ctx",
            return_value=worker_ctx,
        ),
    ):
        chunks = _scan_one_cdx({"cdx_rel": "fake/cdx-00000.gz"})

    # Worker reports both hit lines; driver-side set union collapses to one.
    assert chunks[0]["scanned"] == 2
    assert chunks[0]["hits"] == ["http://x/repeat", "http://x/repeat"]
