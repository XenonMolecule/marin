# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resumable Nemotron-CC URL coverage validator (v4).

Same goal as ``validate_url_coverage.py``: for snapshot S, check what fraction
of Nemotron-CC's URLs map back to URLs in CC's WARC headers. v4 differs only
in being **resilient to preemption / OOMs** by checkpointing intermediate
state to GCS at every step:

- **Per-Nemotron-file URL extraction**: each Nemotron jsonl.gz produces a
  gzipped text shard at ``{checkpoint_path}/nemotron_urls/{base}.txt.gz``.
  On restart, we list the directory and only process files without a shard.
- **Per-CDX-file scan results**: each scan task's hits are written to
  ``{checkpoint_path}/cdx_hits/`` via Zephyr's ``write_jsonl(skip_existing=True)``.
  On restart, completed shards are skipped.
- **Final aggregation**: reads hits back from GCS so the driver doesn't need
  to hold worker results in memory.

If the v3 driver OOMs partway through (URL set is bigger than estimated and
its 64 GiB driver may not survive), v4 picks up from wherever the
checkpoints landed.

Memory note: even with checkpointing, the *Final* Nemotron URL set must still
fit in the driver and each Zephyr worker's RAM during the CDX scan phase.
Provision aggressively for that — compute is free, OOMs are not.
"""

import argparse
import gzip
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import requests
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, zephyr_worker_ctx

logger = logging.getLogger(__name__)

NEMOTRON_BASE = "gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"
CDX_PATHS_URL_FMT = "https://data.commoncrawl.org/crawl-data/{snapshot}/cc-index.paths.gz"
CDX_FILE_URL_FMT = "https://data.commoncrawl.org/{relative_path}"

QUALITY_LEVELS = ("high", "medium-high", "medium", "medium-low", "low")
SAMPLE_MISSES = 100

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5.0
HTTP_TIMEOUT = 600


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


def _list_nemotron_files(snapshot: str) -> list[str]:
    """Discover all Nemotron files for a snapshot across all quality/kind partitions."""
    fs = fsspec.filesystem("gcs")
    base = NEMOTRON_BASE.replace("gs://", "")

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

    return files


def _checkpoint_name_for_nemotron(nemotron_path: str) -> str:
    """Stable basename for a Nemotron file's URL checkpoint.

    Encodes the partition info to avoid collisions across quality/kind partitions
    (the same ``CC-MAIN-S-part-NNNNN.jsonl.gz`` filename appears in many partitions).
    """
    # Path: .../quality=Q/kind=K/kind2=K2/CC-MAIN-S-part-NNNNN.jsonl.gz
    parts = nemotron_path.split("/")
    quality = next((p.removeprefix("quality=") for p in parts if p.startswith("quality=")), "unknown")
    kind = next((p.removeprefix("kind=") for p in parts if p.startswith("kind=")), "unknown")
    kind2 = next((p.removeprefix("kind2=") for p in parts if p.startswith("kind2=")), "unknown")
    base = parts[-1].removesuffix(".jsonl.gz")
    return f"{quality}__{kind}__{kind2}__{base}"


def _extract_one_nemotron_file_to_checkpoint(args: tuple[str, str]) -> tuple[str, int]:
    """Read one Nemotron file, extract URLs, write to checkpoint. Returns (checkpoint_path, count)."""
    fpath, ck_path = args
    urls: list[str] = []
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
                urls.append(u)
    # Atomic-ish write: write to .tmp then rename
    tmp_path = f"{ck_path}.tmp"
    with fsspec.open(tmp_path, "wb") as fh, gzip.open(fh, "wt", encoding="utf-8") as gz:
        for u in urls:
            gz.write(u + "\n")
    fs = fsspec.filesystem(tmp_path.split("://", 1)[0] if "://" in tmp_path else "file")
    fs.mv(tmp_path.replace("gs://", ""), ck_path.replace("gs://", ""))
    return ck_path, len(urls)


def _load_nemotron_urls_with_checkpoint(snapshot: str, checkpoint_path: str) -> set[str]:
    """Load Nemotron URLs with per-file checkpoints in GCS.

    Resume-safe: if a checkpoint exists for a file, skip that file. Process
    only the missing ones. Then read back all checkpoints to build the final set.
    """
    fs = fsspec.filesystem("gcs")
    nemotron_files = _list_nemotron_files(snapshot)
    logger.info(f"Nemotron files for {snapshot}: {len(nemotron_files)}")

    ck_dir = f"{checkpoint_path.rstrip('/')}/nemotron_urls"
    ck_dir_no_scheme = ck_dir.replace("gs://", "")
    try:
        existing = {f.rstrip("/").split("/")[-1] for f in fs.ls(ck_dir_no_scheme, detail=False)}
    except FileNotFoundError:
        existing = set()
    logger.info(f"Existing Nemotron URL checkpoints: {len(existing)}")

    # Build map: nemotron_file → checkpoint_path
    file_to_ck = {}
    for fp in nemotron_files:
        ck_name = f"{_checkpoint_name_for_nemotron(fp)}.txt.gz"
        file_to_ck[fp] = f"{ck_dir}/{ck_name}"

    # Determine missing files
    missing = [(fp, ck) for fp, ck in file_to_ck.items() if ck.split("/")[-1] not in existing]
    logger.info(f"Files to process: {len(missing)} (skipping {len(nemotron_files) - len(missing)} already checkpointed)")

    # Process missing files in parallel, each writes its own checkpoint
    if missing:
        done = 0
        with ThreadPoolExecutor(max_workers=64) as pool:
            futures = [pool.submit(_extract_one_nemotron_file_to_checkpoint, args) for args in missing]
            for fut in as_completed(futures):
                _ck_path, count = fut.result()
                done += 1
                if done % 50 == 0 or done == len(missing):
                    logger.info(f"Processed {done}/{len(missing)} missing files (most recent: {count} URLs)")

    # Read back ALL checkpoints into final set
    logger.info(f"Reading back {len(file_to_ck)} checkpoints into URL set...")
    urls: set[str] = set()
    done = 0

    def _read_one(ck_path: str) -> list[str]:
        out: list[str] = []
        with fsspec.open(ck_path, "rb") as fh, gzip.open(fh, "rt", encoding="utf-8") as gz:
            for line in gz:
                u = line.strip()
                if u:
                    out.append(u)
        return out

    with ThreadPoolExecutor(max_workers=64) as pool:
        for batch in pool.map(_read_one, file_to_ck.values()):
            urls.update(batch)
            done += 1
            if done % 100 == 0 or done == len(file_to_ck):
                logger.info(f"  read {done}/{len(file_to_ck)} checkpoints, {len(urls):,} unique URLs so far")

    logger.info(f"Total unique Nemotron URLs for {snapshot}: {len(urls):,}")
    return urls


def _scan_one_cdx(task: dict) -> list[dict]:
    """Stream one CDX file, return [{scanned, hits}]. Identical to v3 scanner."""
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


def _aggregate_from_checkpoints(cdx_hits_dir: str, nemotron_urls_size: int) -> tuple[set[str], int]:
    """Read back per-CDX hits from GCS (or any fsspec-supported FS) and aggregate."""
    scheme = cdx_hits_dir.split("://", 1)[0] if "://" in cdx_hits_dir else "file"
    fs = fsspec.filesystem(scheme)
    dir_no_scheme = cdx_hits_dir.split("://", 1)[1] if "://" in cdx_hits_dir else cdx_hits_dir
    raw_files = [f for f in fs.ls(dir_no_scheme, detail=False) if f.endswith(".jsonl.gz")]
    # Re-add scheme if needed so fsspec.open knows where to look
    files = sorted(f if "://" in f else f"{scheme}://{f}" for f in raw_files)
    logger.info(f"Aggregating {len(files)} per-CDX hit shards from {cdx_hits_dir}...")

    matched: set[str] = set()
    total_scanned = 0
    for i, fp in enumerate(files):
        with fsspec.open(fp, "rb") as fh, gzip.open(fh, "rt", encoding="utf-8") as gz:
            for line in gz:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_scanned += int(r.get("scanned", 0))
                for u in r.get("hits", []):
                    matched.add(u)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            logger.info(f"  aggregated {i + 1}/{len(files)} hit shards: matched={len(matched):,}/{nemotron_urls_size:,}")

    return matched, total_scanned


def validate(snapshot: str, output_path: str, checkpoint_path: str) -> dict:
    logger.info(f"=== [v4] Validating Nemotron URL coverage for {snapshot} ===")

    nemotron_urls = _load_nemotron_urls_with_checkpoint(snapshot, checkpoint_path)
    if not nemotron_urls:
        raise RuntimeError(f"No Nemotron URLs found for {snapshot}")

    cdx_files = _list_cdx_files(snapshot)
    logger.info(f"Will scan {len(cdx_files)} CDX files (per-CDX outputs are checkpointed)")
    tasks = [{"cdx_rel": p} for p in cdx_files]

    cdx_hits_dir = f"{checkpoint_path.rstrip('/')}/cdx_hits"
    pipeline = (
        Dataset.from_list(tasks)
        .flat_map(_scan_one_cdx)
        .write_jsonl(
            f"{cdx_hits_dir}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
            skip_existing=True,
        )
    )

    # Heavy memory ask: each worker holds the full Nemotron URL set + the inner
    # coordinator stages the cloudpickled blob. The set might be 50-80M URLs
    # for one snapshot (v3 logs show 35.7M at 300/563 files and growing).
    ctx = ZephyrContext(
        name=f"validate-coverage-v4-{snapshot}",
        max_workers=16,
        resources=ResourceConfig(cpu=1, ram="96g"),
        coordinator_resources=ResourceConfig(cpu=1, ram="96g"),
    )
    ctx.put("nemotron_urls", nemotron_urls)
    ctx.execute(pipeline)

    # Aggregate from per-CDX checkpoints (preemption-safe even if driver dies here)
    matched_set, total_scanned = _aggregate_from_checkpoints(cdx_hits_dir, len(nemotron_urls))

    coverage = len(matched_set) / len(nemotron_urls) if nemotron_urls else 0.0
    sample_missing = list(nemotron_urls - matched_set)[:SAMPLE_MISSES]
    result = {
        "snapshot": snapshot,
        "nemotron_url_count": len(nemotron_urls),
        "cc_url_lines_scanned": total_scanned,
        "matched_count": len(matched_set),
        "coverage_pct": coverage * 100,
        "sample_missing": sample_missing,
    }

    out_uri = f"{output_path.rstrip('/')}/{snapshot}_coverage.json"
    with fsspec.open(out_uri, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        f"=== {snapshot}: {result['coverage_pct']:.4f}% coverage ({len(matched_set):,}/{len(nemotron_urls):,}) ==="
    )
    logger.info(f"Report written to {out_uri}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="e.g. CC-MAIN-2013-20")
    parser.add_argument("--output", required=True, help="GCS prefix for final coverage report JSON")
    parser.add_argument("--checkpoint-path", required=True, help="GCS prefix for intermediate checkpoints")
    args = parser.parse_args()

    if not re.fullmatch(r"CC-MAIN-\d{4}-\d{2}", args.snapshot):
        raise SystemExit(f"--snapshot must look like CC-MAIN-YYYY-WW, got {args.snapshot!r}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate(args.snapshot, args.output, args.checkpoint_path)


if __name__ == "__main__":
    main()
