# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter FineWeb-Edu to records matching our selected WARC files.

Coverage argument
-----------------
Join key: file_path column in FineWeb-Edu parquet, which contains the S3 path to
the source WARC file (e.g., 'crawl-data/CC-MAIN-2022-49/segments/.../warc/...-00166.warc.gz').
This is the same path we use to download WARCs from Common Crawl.

Why this is the cleanest join: Unlike URL or record ID matching, this is a file-level
filter. Every FineWeb-Edu record knows exactly which WARC file it came from. We filter
to records where file_path matches one of our 3000 WARC files. No normalization edge
cases for the content — just path string matching.

Path normalization: Our WARC manifest uses 's3://commoncrawl/crawl-data/...' while
FineWeb-Edu file_path uses 'crawl-data/...' (no S3 prefix). We normalize both sides
by stripping the 's3://commoncrawl/' prefix.

Scoping: FineWeb-Edu is partitioned by snapshot (CC-MAIN-YYYY-WW/ directories). We
only read parquet files for snapshots present in our WARC set. Confirmed via our
overlap job: 99.95% of DCLM WARCs appear in FineWeb-Edu (10,359/10,364 — we excluded
the 5 missing ones when building our WARC manifest).

Coverage guarantee: If a document from our WARCs survived FineWeb-Edu's filtering
pipeline, its file_path will match one of our WARC paths. The parquet column projection
makes the initial filter fast (read only file_path column before loading full rows).
"""

import logging
import re
from dataclasses import dataclass

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, zephyr_worker_ctx

logger = logging.getLogger(__name__)


@dataclass
class FilterFinewebEduConfig:
    metadata_path: str
    """Glob pattern for WARC metadata JSONL files."""

    fineweb_base_path: str
    """GCS base path to FineWeb-Edu data (e.g., gs://marin-us-central2/raw/fineweb-edu)."""

    output_path: str

    manifest_path: str | None = None
    """Optional fast-path: a local .txt file with one WARC S3 path per line.

    When set, the driver skips the slow per-WARC metadata load (which fans out
    to 10k+ tiny GCS reads and takes 30-45 min) and instead derives the
    {snapshot: warc_files} dict directly from the manifest in <1 sec. Snapshot
    is extracted via the same regex as `_extract_snapshot`. The metadata_path
    field is ignored when this is set.
    """


def normalize_warc_path(path: str) -> str:
    """Strip s3://commoncrawl/ or https://data.commoncrawl.org/ prefix."""
    for prefix in ["s3://commoncrawl/", "https://data.commoncrawl.org/"]:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _extract_snapshot(warc_path: str) -> str:
    """Extract CC-MAIN-YYYY-WW from a WARC path."""
    m = re.search(r"CC-MAIN-\d{4}-\d{2}", warc_path)
    return m.group(0) if m else "unknown"


def _load_warc_files_by_snapshot(metadata_path: str) -> dict[str, set[str]]:
    """Load metadata and build {snapshot: set of normalized WARC file paths}."""
    import glob as globmod

    from zephyr.readers import load_file as zephyr_load_file

    by_snapshot: dict[str, set[str]] = {}

    if metadata_path.startswith("gs://"):
        fs = fsspec.filesystem("gcs")
        dir_path = (
            metadata_path.replace("gs://", "").rsplit("/", 1)[0]
            if "*" in metadata_path
            else metadata_path.replace("gs://", "")
        )
        files = [f"gs://{f}" for f in fs.ls(dir_path, detail=False) if f.endswith(".jsonl.gz")]
    else:
        files = globmod.glob(metadata_path)

    # Parallelize metadata loading — 3000 small GCS files sequentially is too slow
    from concurrent.futures import ThreadPoolExecutor

    def _read_one_file(fpath):
        results = []
        for record in zephyr_load_file(fpath):
            snap = record.get("snapshot", "")
            warc_file = normalize_warc_path(record.get("warc_file", ""))
            if snap and warc_file:
                results.append((snap, warc_file))
        return results

    with ThreadPoolExecutor(max_workers=64) as pool:
        for batch in pool.map(_read_one_file, files):
            for snap, warc_file in batch:
                by_snapshot.setdefault(snap, set()).add(warc_file)

    total = sum(len(v) for v in by_snapshot.values())
    logger.info(f"Loaded {total:,} unique WARC paths across {len(by_snapshot)} snapshots from {len(files)} files")
    return by_snapshot


def _list_fineweb_parquets(base_path: str, snapshots: set[str]) -> list[dict]:
    """List FineWeb-Edu parquet files for given snapshots."""
    fs = fsspec.filesystem("gcs") if base_path.startswith("gs://") else fsspec.filesystem("file")
    base = base_path.replace("gs://", "")
    tasks = []

    for snap in sorted(snapshots):
        snap_dir = f"{base}/{snap}"
        try:
            files = fs.ls(snap_dir, detail=False)
        except FileNotFoundError:
            logger.warning(f"Snapshot {snap} not found in FineWeb-Edu at {snap_dir}")
            continue

        for f in files:
            if f.endswith(".parquet"):
                tasks.append(
                    {
                        "parquet_path": f"gs://{f}" if base_path.startswith("gs://") else f,
                        "snapshot": snap,
                    }
                )

    logger.info(f"Found {len(tasks)} FineWeb-Edu parquet files across {len(snapshots)} snapshots")
    return tasks


def _process_fineweb_parquet(task: dict) -> list[dict]:
    """Process a single FineWeb-Edu parquet file, filtering by WARC file path."""
    ctx = zephyr_worker_ctx()
    warc_files_by_snapshot = ctx.get_shared("warc_files_by_snapshot")

    parquet_path = task["parquet_path"]
    snapshot = task["snapshot"]
    warc_file_set = warc_files_by_snapshot.get(snapshot, set())
    if not warc_file_set:
        return []

    # Read just the file_path column first for fast filtering
    with fsspec.open(parquet_path, "rb") as fh:
        file_path_table = pq.read_table(fh, columns=["file_path"])

    file_paths = file_path_table.column("file_path").to_pylist()
    matching_indices = [i for i, fp in enumerate(file_paths) if normalize_warc_path(fp) in warc_file_set]

    if not matching_indices:
        return []

    # Now read full rows for matches only
    with fsspec.open(parquet_path, "rb") as fh:
        full_table = pq.read_table(fh)

    results = []
    for idx in matching_indices:
        results.append(
            {
                "text": full_table.column("text")[idx].as_py(),
                "url": full_table.column("url")[idx].as_py(),
                "file_path": full_table.column("file_path")[idx].as_py(),
                "dump": full_table.column("dump")[idx].as_py() if "dump" in full_table.column_names else snapshot,
                "fineweb_score": full_table.column("score")[idx].as_py() if "score" in full_table.column_names else None,
                "fineweb_int_score": (
                    full_table.column("int_score")[idx].as_py() if "int_score" in full_table.column_names else None
                ),
            }
        )

    logger.info(f"  {snapshot} {parquet_path.split('/')[-1]}: " f"{len(results)}/{len(file_paths)} matched")
    return results


def _load_warc_files_from_manifest(manifest_path: str) -> dict[str, set[str]]:
    """Build {snapshot: set[warc_file]} directly from a manifest .txt file.

    Much faster than `_load_warc_files_by_snapshot` for large manifests because
    it skips the per-WARC metadata GCS reads — snapshot is derived from the
    WARC path itself via regex.
    """
    by_snapshot: dict[str, set[str]] = {}
    with open(manifest_path) as f:
        for line in f:
            warc_file = line.strip()
            if not warc_file:
                continue
            warc_file = normalize_warc_path(warc_file)
            snap = _extract_snapshot(warc_file)
            if snap and snap != "unknown":
                by_snapshot.setdefault(snap, set()).add(warc_file)
    total = sum(len(v) for v in by_snapshot.values())
    logger.info(f"Loaded {total:,} unique WARC paths across {len(by_snapshot)} snapshots from manifest {manifest_path}")
    return by_snapshot


def filter_fineweb_edu(config: FilterFinewebEduConfig) -> None:
    """Filter FineWeb-Edu records matching our WARC files."""
    if config.manifest_path:
        warc_files_by_snapshot = _load_warc_files_from_manifest(config.manifest_path)
    else:
        warc_files_by_snapshot = _load_warc_files_by_snapshot(config.metadata_path)
    snapshots = set(warc_files_by_snapshot.keys())
    tasks = _list_fineweb_parquets(config.fineweb_base_path, snapshots)

    pipeline = (
        Dataset.from_list(tasks)
        .flat_map(_process_fineweb_parquet)
        .write_jsonl(
            f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
            skip_existing=True,
        )
    )

    # 8 GiB workers: parquet reads load full tables into memory. FineWeb-Edu
    # parquets are ~1-2 GB compressed → 4-8 GB decoded as PyArrow tables, which
    # OOMs the 1 GiB Zephyr default. Observed on the 10k pool run: workers were
    # OOM-killed in a tight retry loop until this was bumped.
    ctx = ZephyrContext(
        name="filter-fineweb-edu",
        max_workers=500,
        resources=ResourceConfig(cpu=1, ram="8g"),
    )
    ctx.put("warc_files_by_snapshot", warc_files_by_snapshot)
    ctx.execute(pipeline)

    logger.info(f"FineWeb-Edu filter complete → {config.output_path}")
