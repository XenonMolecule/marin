# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate that every Nemotron-CC URL for a snapshot S maps back to a WARC in S.

Question this answers: "If we know a Nemotron-CC record's nemotron_url, can we
*always* find its origin WARC by searching CC's published headers for that
snapshot?" If yes, then our URL-based mapping in filter_nemotron is provably
lossless. If no, we have URL-canonicalization or scoping problems.

Method
------
1. Read all Nemotron-CC URLs for snapshot S from our existing GCS dataset.
   Iterates every ``CC-MAIN-S-part-*.jsonl.gz`` file across all qualities and
   kinds, projects ``metadata.nemotron_url``, dedupes into an in-memory set N.
2. Stream every CDX file for snapshot S from Common Crawl's public mirror
   (``https://data.commoncrawl.org/cc-index/collections/CC-MAIN-S/indexes/``).
   For each line, extract the ``url`` field and mark it found in N if present.
3. Report |N ∩ CC| / |N|. If < 99%, sample mismatched URLs.

Why low concurrency
-------------------
CC's CDX mirror throttles aggressively at high concurrency. We use 16 workers
(not 500) so the ~106 GB scan finishes in ~30-60 minutes without 429s.

Usage::

    uv run iris --cluster marin job run --cpu 2 --memory 32GB \\
        --region us-central2 --no-wait \\
        --job-name validate-nemotron-coverage \\
        -- python experiments/baseline_collection/validate_url_coverage.py \\
            --snapshot CC-MAIN-2013-20 \\
            --output gs://marin-us-central2/scratch/nemotron_coverage_validation/
"""

import argparse
import gzip
import io
import json
import logging
import re
import time
from dataclasses import dataclass

import fsspec
import requests
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext
from zephyr.execution import zephyr_worker_ctx

logger = logging.getLogger(__name__)

NEMOTRON_BASE = "gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"
CDX_PATHS_URL_FMT = "https://data.commoncrawl.org/crawl-data/{snapshot}/cc-index.paths.gz"
CDX_FILE_URL_FMT = "https://data.commoncrawl.org/{relative_path}"

QUALITY_LEVELS = ("high", "medium-high", "medium", "medium-low", "low")
SAMPLE_MISSES = 100  # how many "in N but not in CC" URLs to record for inspection

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5.0
HTTP_TIMEOUT = 600


@dataclass
class CoverageResult:
    snapshot: str
    nemotron_url_count: int
    cc_url_lines_scanned: int
    matched_count: int
    coverage_pct: float
    sample_missing: list[str]


def _http_get_bytes_with_retry(url: str) -> bytes:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers={"Accept-Encoding": "identity"}, timeout=HTTP_TIMEOUT)
            if r.status_code in (429, 503):
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(f"Rate limited ({r.status_code}) on {url}, retry in {delay:.0f}s")
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.content
        except requests.exceptions.RequestException as e:
            delay = RETRY_BASE_DELAY * (2**attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"HTTP error on {url} (attempt {attempt + 1}): {e}. Retry in {delay:.0f}s")
                time.sleep(delay)
            else:
                raise RuntimeError(f"GET {url} failed after {MAX_RETRIES} attempts: {e}") from e
    raise RuntimeError(f"GET {url} failed after {MAX_RETRIES} attempts")


def _list_cdx_files(snapshot: str) -> list[str]:
    body = _http_get_bytes_with_retry(CDX_PATHS_URL_FMT.format(snapshot=snapshot))
    paths = gzip.decompress(body).decode("utf-8").strip().splitlines()
    return [p for p in paths if "/indexes/cdx-" in p and p.endswith(".gz")]


def _load_nemotron_urls(snapshot: str) -> set[str]:
    """Read every ``metadata.nemotron_url`` for the snapshot across all partitions."""
    fs = fsspec.filesystem("gcs")
    base = NEMOTRON_BASE.replace("gs://", "")

    # Discover (quality, kind, kind2) partitions that contain this snapshot's parts.
    partition_dirs: list[str] = []
    for quality in QUALITY_LEVELS:
        partition_dirs.append(f"{base}/quality={quality}/kind=actual/kind2=actual")
        synthetic_root = f"{base}/quality={quality}/kind=synthetic"
        try:
            for kind2_dir in fs.ls(synthetic_root, detail=False):
                partition_dirs.append(kind2_dir.rstrip("/"))
        except FileNotFoundError:
            continue

    files: list[str] = []
    for d in partition_dirs:
        try:
            entries = fs.ls(d, detail=False)
        except FileNotFoundError:
            continue
        for f in entries:
            fname = f.rstrip("/").split("/")[-1]
            if fname.startswith(snapshot) and fname.endswith(".jsonl.gz"):
                files.append(f"gs://{f}")

    logger.info(f"Nemotron files for {snapshot}: {len(files)} across {len(partition_dirs)} partitions")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _extract_one(fpath: str) -> list[str]:
        out: list[str] = []
        with fsspec.open(fpath, "rb") as fh, gzip.open(fh, "rt", encoding="utf-8") as gz:
            for line in gz:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = r.get("metadata", {}).get("nemotron_url", "")
                if u:
                    out.append(u)
        return out

    urls: set[str] = set()
    done = 0
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [pool.submit(_extract_one, fp) for fp in files]
        for fut in as_completed(futures):
            urls.update(fut.result())
            done += 1
            if done % 100 == 0 or done == len(files):
                logger.info(f"  read {done}/{len(files)} Nemotron files, {len(urls):,} unique URLs so far")

    logger.info(f"Total unique Nemotron URLs for {snapshot}: {len(urls):,}")
    return urls


def _scan_one_cdx(task: dict) -> list[dict]:
    """Stream one CDX file, return ``[{matched: int, scanned: int, hit_urls: list[str]}]``.

    ``hit_urls`` lets the driver mark which Nemotron URLs were found, since we
    can't share a mutable ``found`` set across workers cheaply.
    """
    ctx = zephyr_worker_ctx()
    nemotron_urls: set[str] = ctx.get_shared("nemotron_urls")

    cdx_url = CDX_FILE_URL_FMT.format(relative_path=task["cdx_rel"])
    body = _http_get_bytes_with_retry(cdx_url)

    scanned = 0
    hits: list[str] = []
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
        for line in gz:
            scanned += 1
            try:
                s = line.decode("utf-8", errors="replace")
                i = s.find("{")
                if i < 0:
                    continue
                meta = json.loads(s[i:])
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            u = meta.get("url", "")
            if u and u in nemotron_urls:
                hits.append(u)

    logger.info(f"  {task['cdx_rel'].rsplit('/', 1)[-1]}: scanned={scanned:,} hits={len(hits):,}")
    return [{"scanned": scanned, "hits": hits}]


def validate(snapshot: str, output_path: str) -> CoverageResult:
    """Run the validation and write a coverage report to GCS."""
    logger.info(f"=== Validating Nemotron URL coverage for {snapshot} ===")

    nemotron_urls = _load_nemotron_urls(snapshot)
    if not nemotron_urls:
        raise RuntimeError(f"No Nemotron URLs found for {snapshot}")

    cdx_files = _list_cdx_files(snapshot)
    logger.info(f"Will scan {len(cdx_files)} CDX files at low concurrency to be gentle on CC")
    tasks = [{"cdx_rel": p} for p in cdx_files]

    pipeline = Dataset.from_list(tasks).flat_map(_scan_one_cdx).collect()

    # The Nemotron URL set for one snapshot is MUCH bigger than first estimated:
    # the driver observed 16M unique URLs after only 200/563 files (extrapolating
    # to 30-45M total). As a Python set that's 30-45 GB with hash-table + string
    # overhead. Each worker calls get_shared which cloudpickle.loads the whole set
    # into its own memory, AND the inner coordinator stages the serialized blob.
    # Overprovision aggressively so we don't keep OOMing; compute is free per the
    # user's GCP deal.
    ctx = ZephyrContext(
        name=f"validate-coverage-{snapshot}",
        max_workers=16,
        resources=ResourceConfig(cpu=1, ram="64g"),
        coordinator_resources=ResourceConfig(cpu=1, ram="64g"),
    )
    ctx.put("nemotron_urls", nemotron_urls)
    chunk_results = list(ctx.execute(pipeline))

    matched_set: set[str] = set()
    total_scanned = 0
    for chunk in chunk_results:
        total_scanned += chunk["scanned"]
        matched_set.update(chunk["hits"])

    matched = len(matched_set)
    coverage = matched / len(nemotron_urls) if nemotron_urls else 0.0
    sample_missing = list(nemotron_urls - matched_set)[:SAMPLE_MISSES]

    result = CoverageResult(
        snapshot=snapshot,
        nemotron_url_count=len(nemotron_urls),
        cc_url_lines_scanned=total_scanned,
        matched_count=matched,
        coverage_pct=coverage * 100,
        sample_missing=sample_missing,
    )

    out_uri = f"{output_path.rstrip('/')}/{snapshot}_coverage.json"
    with fsspec.open(out_uri, "w") as f:
        json.dump(
            {
                "snapshot": result.snapshot,
                "nemotron_url_count": result.nemotron_url_count,
                "cc_url_lines_scanned": result.cc_url_lines_scanned,
                "matched_count": result.matched_count,
                "coverage_pct": result.coverage_pct,
                "sample_missing": result.sample_missing,
            },
            f,
            indent=2,
        )

    logger.info(f"=== {snapshot}: {result.coverage_pct:.4f}% coverage ({matched:,}/{len(nemotron_urls):,}) ===")
    logger.info(f"Report written to {out_uri}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="e.g. CC-MAIN-2013-20")
    parser.add_argument("--output", required=True, help="GCS prefix for coverage report JSON")
    args = parser.parse_args()

    if not re.fullmatch(r"CC-MAIN-\d{4}-\d{2}", args.snapshot):
        raise SystemExit(f"--snapshot must look like CC-MAIN-YYYY-WW, got {args.snapshot!r}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate(args.snapshot, args.output)


if __name__ == "__main__":
    main()
