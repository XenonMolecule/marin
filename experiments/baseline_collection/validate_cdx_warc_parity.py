# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate that CDX-reported URLs for a WARC match the WARC's actual headers.

If this check passes at 100%, we have proof that our URL-matching chain
(WARC header -> CDX -> filter -> Nemotron) is losslessly linked at every
hop: CDX faithfully mirrors what the WARC contains.

Method
------
For ONE WARC file (chosen from the 3000 manifest):

1. Stream its CDX entries from Common Crawl's public CDX files for the
   same snapshot, keeping only lines whose ``filename`` matches our WARC.
   Collect the URL set ``A``.
2. Download the WARC itself (~1 GB), iterate response records, and collect
   every ``WARC-Target-URI`` whose HTTP status is 200 and whose
   Content-Type contains ``text/html``. Collect the URL set ``B``.
3. Compare. CDX indexes only successful HTML responses so we expect
   ``A == B``. Anything else is a defect.

Scoping rationale
-----------------
CDX's scope is "one row per successful HTML response". WARC files contain
ALL responses (including non-HTML, non-200, and sometimes request/metadata
records). Filtering to 200 text/html on the WARC side is what makes the
two sets comparable; without that filter the WARC side is a strict
superset and the comparison is meaningless.

Usage::

    uv run iris --cluster marin job run --cpu 2 --memory 16GB \\
        --enable-extra-resources --region us-central2 --no-wait \\
        --job-name validate-cdx-warc-parity \\
        -- python experiments/baseline_collection/validate_cdx_warc_parity.py \\
            --warc-path "s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/..." \\
            --output gs://marin-us-central2/scratch/cdx_warc_parity/
"""

import argparse
import gzip
import io
import json
import logging
import re
import time

import fsspec
import requests
import warcio

from experiments.baseline_collection.extract_metadata_from_cc_cdx import (
    _http_get_bytes_with_retry,
    _list_cdx_files,
    _parse_cdx_line,
    _to_relative,
)

logger = logging.getLogger(__name__)

CDX_FILE_URL_FMT = "https://data.commoncrawl.org/{relative_path}"
SNAPSHOT_RE = re.compile(r"CC-MAIN-\d{4}-\d{2}")
SAMPLE_DIFF = 20

HTTP_TIMEOUT = 600
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5.0


def _collect_cdx_urls_for_warc(warc_path: str) -> set[str]:
    """Scan every CDX file for the WARC's snapshot, keep entries matching the filename."""
    warc_rel = _to_relative(warc_path)
    m = SNAPSHOT_RE.search(warc_rel)
    if not m:
        raise ValueError(f"Could not extract snapshot from WARC path: {warc_path}")
    snapshot = m.group(0)

    cdx_files = _list_cdx_files(snapshot)
    logger.info(f"Scanning {len(cdx_files)} CDX files for {snapshot} looking for {warc_rel}")

    urls: set[str] = set()
    for i, cdx_rel in enumerate(cdx_files):
        body = _http_get_bytes_with_retry(CDX_FILE_URL_FMT.format(relative_path=cdx_rel))
        matched_here = 0
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
            for line in gz:
                meta = _parse_cdx_line(line)
                if meta is None:
                    continue
                if meta.get("filename") != warc_rel:
                    continue
                u = meta.get("url", "")
                if u:
                    urls.add(u)
                    matched_here += 1
        if matched_here or (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(cdx_files)} CDX files scanned, total matches: {len(urls):,}")

    logger.info(f"CDX URL set for {warc_rel}: {len(urls):,} URLs")
    return urls


def _collect_warc_urls(warc_path: str) -> set[str]:
    """Download the WARC, extract WARC-Target-URIs of 200 text/html response records."""
    url = warc_path
    if url.startswith("s3://commoncrawl/"):
        url = "https://data.commoncrawl.org/" + url[len("s3://commoncrawl/") :]

    logger.info(f"Downloading {url}...")
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT)
            if r.status_code in (429, 503):
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(f"Rate limited ({r.status_code}), retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s")
                time.sleep(delay)
                continue
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            delay = RETRY_BASE_DELAY * (2**attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Download error (attempt {attempt + 1}): {e}, retry in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts: {e}") from e
    else:
        raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts")

    urls: set[str] = set()
    total_records = 0
    response_records = 0
    text_html_200 = 0
    parse_errors = 0

    raw_bytes = io.BytesIO(r.content)
    for record in warcio.ArchiveIterator(raw_bytes):
        total_records += 1
        try:
            if record.rec_type != "response":
                continue
            response_records += 1
            http_headers = record.http_headers
            if http_headers is None:
                continue
            status = http_headers.get_statuscode() or ""
            if status != "200":
                continue
            content_type = http_headers.get_header("Content-Type") or ""
            if "text/html" not in content_type.lower():
                continue
            target_uri = record.rec_headers.get_header("WARC-Target-URI") or ""
            if target_uri:
                urls.add(target_uri)
                text_html_200 += 1
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 3:
                logger.warning(f"Skipping corrupt record: {e}")

    logger.info(
        f"WARC {warc_path}: total_records={total_records:,} response={response_records:,} "
        f"200/text-html={text_html_200:,} parse_errors={parse_errors} unique_urls={len(urls):,}"
    )
    return urls


def parity(warc_path: str, output_path: str) -> dict:
    """Run the parity check and return + write the result."""
    logger.info(f"=== CDX-WARC parity check for {warc_path} ===")

    # Kick off CDX side and WARC side sequentially; they're independent but memory
    # is not a concern (per-WARC URL sets are ~30-80k URLs = a few MB).
    cdx_urls = _collect_cdx_urls_for_warc(warc_path)
    warc_urls = _collect_warc_urls(warc_path)

    both = cdx_urls & warc_urls
    only_cdx = cdx_urls - warc_urls
    only_warc = warc_urls - cdx_urls

    result = {
        "warc_path": warc_path,
        "cdx_url_count": len(cdx_urls),
        "warc_url_count": len(warc_urls),
        "intersection": len(both),
        "only_in_cdx": len(only_cdx),
        "only_in_warc": len(only_warc),
        "cdx_coverage_pct": (len(both) / len(cdx_urls) * 100) if cdx_urls else 0.0,
        "warc_coverage_pct": (len(both) / len(warc_urls) * 100) if warc_urls else 0.0,
        "sample_only_cdx": sorted(only_cdx)[:SAMPLE_DIFF],
        "sample_only_warc": sorted(only_warc)[:SAMPLE_DIFF],
    }

    logger.info(
        f"=== Parity: CDX={len(cdx_urls):,} WARC={len(warc_urls):,} "
        f"both={len(both):,} only_cdx={len(only_cdx):,} only_warc={len(only_warc):,} ==="
    )
    logger.info(f"CDX coverage: {result['cdx_coverage_pct']:.4f}%  WARC coverage: {result['warc_coverage_pct']:.4f}%")

    # Sanitize filename from WARC path for the output
    slug = _to_relative(warc_path).rsplit("/", 1)[-1].removesuffix(".warc.gz")
    out_uri = f"{output_path.rstrip('/')}/parity_{slug}.json"
    with fsspec.open(out_uri, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Report written to {out_uri}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc-path", required=True, help="s3://commoncrawl/... or https://... WARC path")
    parser.add_argument("--output", required=True, help="GCS prefix for parity report JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parity(args.warc_path, args.output)


if __name__ == "__main__":
    main()
