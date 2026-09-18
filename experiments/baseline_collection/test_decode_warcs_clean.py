# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WARC download integrity: a truncated 200 is retried, a complete one decodes."""

from __future__ import annotations

import io

import pytest
import requests
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from experiments.baseline_collection import decode_warcs_clean as dwc

WARC_PATH = "s3://commoncrawl/crawl-data/CC-MAIN-2020-05/segments/x/warc/CC-MAIN-x.warc.gz"


class _Response:
    def __init__(self, body: bytes, content_length: int | None):
        self.status_code = 200
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}
        self._body = body

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self._body

    @property
    def content(self):
        return self._body


def _warc_bytes(html: str) -> bytes:
    buf = io.BytesIO()
    writer = WARCWriter(buf, gzip=True)
    headers = StatusAndHeaders("200 OK", [("Content-Type", "text/html; charset=utf-8")], protocol="HTTP/1.0")
    record = writer.create_warc_record(
        "http://example.com/page",
        "response",
        payload=io.BytesIO(html.encode()),
        http_headers=headers,
        warc_headers_dict={"WARC-Record-ID": "<urn:uuid:abc-123>"},
    )
    writer.write_record(record)
    return buf.getvalue()


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(dwc.time, "sleep", lambda s: None)


def test_streamed_download_retries_then_fails_on_truncated_body(monkeypatch, no_sleep):
    body = _warc_bytes("<html><body>hello</body></html>")
    calls = []
    monkeypatch.setattr(dwc.requests, "get", lambda url, **kw: calls.append(url) or _Response(body[:-10], len(body)))
    with pytest.raises(RuntimeError, match="truncated"):
        list(dwc._decode_one_warc_stream(WARC_PATH))
    assert len(calls) == dwc.MAX_RETRIES


def test_streamed_download_decodes_a_complete_body(monkeypatch):
    body = _warc_bytes("<html><body><p>hello world</p></body></html>")
    monkeypatch.setattr(dwc.requests, "get", lambda url, **kw: _Response(body, len(body)))
    chunks = list(dwc._decode_one_warc_stream(WARC_PATH))
    (record,) = [r for chunk in chunks for r in chunk]
    assert record["doc_id"] == "abc-123"
    assert record["url"] == "http://example.com/page"
    assert "hello world" in record["html"]


def test_full_download_rejects_truncated_body(monkeypatch, no_sleep):
    body = _warc_bytes("<html><body>hello</body></html>")
    monkeypatch.setattr(dwc.requests, "get", lambda url, **kw: _Response(body[:-10], len(body)))
    with pytest.raises(RuntimeError, match="truncated"):
        dwc._decode_one_warc(WARC_PATH, with_text_body=False)


def test_check_content_length_ignores_encoded_transfers():
    resp = _Response(b"abc", 999)
    resp.headers["Content-Encoding"] = "gzip"
    dwc._check_content_length(resp, 3, WARC_PATH)  # no raise
    with pytest.raises(requests.exceptions.RequestException):
        dwc._check_content_length(_Response(b"abc", 999), 3, WARC_PATH)
