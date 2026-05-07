# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for CDX-vs-WARC parity validator."""

import gzip
import json
from unittest import mock

from experiments.baseline_collection.validate_cdx_warc_parity import (
    _collect_cdx_urls_for_warc,
)


def _make_fake_cdx_bytes(records: list[dict]) -> bytes:
    lines = []
    for r in records:
        surt = "com,example)/"
        ts = "20130101000000"
        lines.append(f"{surt} {ts} {json.dumps(r)}\n".encode())
    return gzip.compress(b"".join(lines))


def test_collect_cdx_urls_for_warc_filters_by_filename():
    """Given 3 CDX files where only 2 have records for our WARC, collect those records' URLs."""
    target_rel = "crawl-data/CC-MAIN-2013-20/segments/x/warc/TARGET.warc.gz"
    other_rel = "crawl-data/CC-MAIN-2013-20/segments/x/warc/OTHER.warc.gz"

    cdx1 = _make_fake_cdx_bytes(
        [
            {"url": "http://a.com/1", "filename": target_rel},
            {"url": "http://a.com/2", "filename": other_rel},  # different WARC, skipped
        ]
    )
    cdx2 = _make_fake_cdx_bytes(
        [
            {"url": "http://a.com/3", "filename": target_rel},
        ]
    )
    cdx3 = _make_fake_cdx_bytes(
        [
            {"url": "http://b.com/x", "filename": other_rel},  # none for our WARC
        ]
    )

    fake_cdx_files = ["idx/cdx-0.gz", "idx/cdx-1.gz", "idx/cdx-2.gz"]
    fake_bodies = {
        "https://data.commoncrawl.org/idx/cdx-0.gz": cdx1,
        "https://data.commoncrawl.org/idx/cdx-1.gz": cdx2,
        "https://data.commoncrawl.org/idx/cdx-2.gz": cdx3,
    }

    with (
        mock.patch(
            "experiments.baseline_collection.validate_cdx_warc_parity._list_cdx_files",
            return_value=fake_cdx_files,
        ),
        mock.patch(
            "experiments.baseline_collection.validate_cdx_warc_parity._http_get_bytes_with_retry",
            side_effect=lambda url: fake_bodies[url],
        ),
    ):
        urls = _collect_cdx_urls_for_warc(f"s3://commoncrawl/{target_rel}")

    assert urls == {"http://a.com/1", "http://a.com/3"}


def test_collect_cdx_urls_for_warc_rejects_unparseable_warc_path():
    import pytest

    with pytest.raises(ValueError, match="snapshot"):
        _collect_cdx_urls_for_warc("s3://commoncrawl/some/path/with/no/cc-main/marker.warc.gz")
