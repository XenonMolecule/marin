# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract per-record metadata from Common Crawl's public CDX index.

Drop-in replacement for ``extract_warc_metadata.py`` that does not require
downloading WARC files. For a manifest of N WARCs, this pulls only the
gzipped CDX text files for the relevant snapshots and filters each line
by ``warc_filename``.

CDX file format (one record per line)::

    <surt_key> <timestamp> {"url": "...", "filename": "crawl-data/.../...warc.gz", ...}

One CDX file per-snapshot shard covers a surt-key range across all WARCs
in that snapshot; 302 CDX files per snapshot, ~350 MB gzipped each. To
find our subset's records we scan every CDX file in the relevant
snapshots and keep lines whose ``filename`` is in our WARC set.

Output schema matches ``extract_warc_metadata.py`` so downstream filters
run unchanged::

    {"warc_record_id": "", "url": "...", "warc_file": "s3://commoncrawl/...", "snapshot": "CC-MAIN-..."}

Note: ``warc_record_id`` is not carried by CDX (only offset + length).
That is fine for Nemotron filtering (URL-based) but not for the DCLM
filter (which joins on WARC-Record-ID). If DCLM output is needed for
the expanded manifest, use ``download_warcs`` instead.
"""

import gzip
import io
import json
import logging
import re
import time
from dataclasses import dataclass

import fsspec
import requests
from zephyr import Dataset, ZephyrContext
from zephyr.execution import zephyr_worker_ctx

logger = logging.getLogger(__name__)

CDX_PATHS_URL_FMT = "https://data.commoncrawl.org/crawl-data/{snapshot}/cc-index.paths.gz"
CDX_FILE_URL_FMT = "https://data.commoncrawl.org/{relative_path}"

SNAPSHOT_RE = re.compile(r"CC-MAIN-\d{4}-\d{2}")

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5.0
HTTP_TIMEOUT = 600


@dataclass
class ExtractMetadataFromCcCdxConfig:
    warc_manifest_path: str
    """Path to text file with one WARC path per line (s3://commoncrawl/... or relative)."""

    output_path: str
    """Output path for metadata JSONL files."""


def _to_relative(warc_path: str) -> str:
    """Normalize to the relative path used in CDX ``filename`` fields."""
    return warc_path.removeprefix("s3://commoncrawl/")


def _group_warcs_by_snapshot(manifest_path: str) -> dict[str, set[str]]:
    """Return ``{snapshot: {relative_warc_path, ...}}`` from a manifest file.

    Uses sets so the per-CDX filename lookup in workers is O(1).
    """
    with fsspec.open(manifest_path, "r") as f:
        paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    by_snap: dict[str, set[str]] = {}
    for p in paths:
        rel = _to_relative(p)
        m = SNAPSHOT_RE.search(rel)
        if not m:
            raise ValueError(f"Could not extract snapshot from WARC path: {p}")
        by_snap.setdefault(m.group(0), set()).add(rel)
    return by_snap


def _http_get_bytes_with_retry(url: str) -> bytes:
    """Download a URL's full body with exponential backoff on retryable errors.

    Returns the raw bytes. For gzipped CDX files we want the *compressed*
    body; we do not pass ``Accept-Encoding: gzip`` that would trigger
    transport-layer decompression on the server side.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # identity encoding to receive the gzipped bytes unmodified
            headers = {"Accept-Encoding": "identity"}
            r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code in (429, 503):
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    f"Rate limited ({r.status_code}) on {url}, retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s"
                )
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.content
        except requests.exceptions.RequestException as e:
            delay = RETRY_BASE_DELAY * (2**attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"HTTP error on {url} (attempt {attempt + 1}): {e}. Retrying in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed GET {url} after {MAX_RETRIES} attempts: {e}") from e
    raise RuntimeError(f"Failed GET {url} after {MAX_RETRIES} attempts")


def _list_cdx_files(snapshot: str) -> list[str]:
    """Return the list of CDX file relative paths for a snapshot.

    Pulls ``https://data.commoncrawl.org/crawl-data/{snapshot}/cc-index.paths.gz``
    which is a tiny gzipped text file (~10 KB).
    """
    url = CDX_PATHS_URL_FMT.format(snapshot=snapshot)
    body = _http_get_bytes_with_retry(url)
    paths = gzip.decompress(body).decode("utf-8").strip().splitlines()
    # Keep only cdx files (the paths.gz also includes other entries in some snapshots).
    return [p for p in paths if "/indexes/cdx-" in p and p.endswith(".gz")]


def _parse_cdx_line(line: bytes) -> dict | None:
    """Parse one CDX line into its JSON metadata dict, or None on malformed input."""
    try:
        s = line.decode("utf-8", errors="replace")
        # CDX: surt_key timestamp {json}
        i = s.find("{")
        if i < 0:
            return None
        return json.loads(s[i:])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _process_one_cdx(task: dict) -> list[dict]:
    """Stream one CDX file, filter by WARC filename, emit metadata records."""
    ctx = zephyr_worker_ctx()
    warcs_by_snap = ctx.get_shared("warcs_by_snap")

    snapshot = task["snapshot"]
    cdx_rel = task["cdx_rel"]
    warc_rel_set = warcs_by_snap[snapshot]

    cdx_url = CDX_FILE_URL_FMT.format(relative_path=cdx_rel)
    body = _http_get_bytes_with_retry(cdx_url)

    records: list[dict] = []
    total = 0
    matched = 0
    parse_errors = 0

    with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
        for line in gz:
            total += 1
            meta = _parse_cdx_line(line)
            if meta is None:
                parse_errors += 1
                continue
            fn = meta.get("filename", "")
            if fn not in warc_rel_set:
                continue
            url = meta.get("url", "")
            if not url:
                continue
            matched += 1
            records.append(
                {
                    "warc_record_id": "",
                    "url": url,
                    "warc_file": f"s3://commoncrawl/{fn}",
                    "snapshot": snapshot,
                }
            )

    logger.info(
        f"  {snapshot} {cdx_rel.rsplit('/', 1)[-1]}: {matched}/{total} matched " f"({parse_errors} parse errors)"
    )
    return records


def extract_metadata_from_cc_cdx(config: ExtractMetadataFromCcCdxConfig) -> None:
    """Extract WARC metadata for a manifest of WARCs via CC's public CDX index."""
    warcs_by_snap = _group_warcs_by_snapshot(config.warc_manifest_path)
    total_warcs = sum(len(v) for v in warcs_by_snap.values())
    logger.info(f"Manifest: {total_warcs} WARCs across {len(warcs_by_snap)} snapshots")

    # Build the task list: one task per (snapshot, cdx_file).
    tasks: list[dict] = []
    for snapshot in sorted(warcs_by_snap):
        cdx_files = _list_cdx_files(snapshot)
        logger.info(f"  {snapshot}: {len(warcs_by_snap[snapshot])} WARCs, {len(cdx_files)} CDX files")
        for cdx_rel in cdx_files:
            tasks.append({"snapshot": snapshot, "cdx_rel": cdx_rel})
    logger.info(f"Total CDX tasks: {len(tasks)}")

    pipeline = (
        Dataset.from_list(tasks)
        .flat_map(_process_one_cdx)
        .write_jsonl(
            f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
            skip_existing=True,
        )
    )

    ctx = ZephyrContext(name="extract-metadata-cc-cdx", max_workers=500)
    ctx.put("warcs_by_snap", warcs_by_snap)
    ctx.execute(pipeline)

    logger.info(f"CDX metadata extraction complete → {config.output_path}")
