# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter Nemotron-CC v1 to records matching our selected WARC files.

Coverage argument
-----------------
Join key: nemotron_url field matched against WARC-Target-URI from our WARC headers.

Why this works: Both values originate from the same WARC header (WARC-Target-URI).
Nemotron's text extraction pipeline (Justext) reads Common Crawl WARCs and preserves
the URL as-is in the nemotron_url metadata field. Our metadata extraction reads the
same WARC files and extracts the same header. Therefore, string equality is sufficient.

Scoping: Nemotron v1 data is partitioned by snapshot (filenames contain CC-MAIN-YYYY-WW).
We only scan files for snapshots present in our WARC set. Since Nemotron processes ALL
WARCs in each snapshot it covers (verified empirically — a random WARC from CC-MAIN-2025-18
produced matches in Nemotron-CC v2.1), every document from our WARCs that Nemotron kept
will appear in the snapshot's files.

Quality levels: We scan all 5 quality levels (high, medium-high, medium, medium-low, low)
for kind=actual. This gives complete coverage of Nemotron's organic data. Synthetic data
(kind=synthetic) is excluded — it has no URL mapping back to source WARCs.

Potential gap: URL normalization differences. Mitigated by comparing raw strings (both
from the same WARC header). Match rate is logged per snapshot for validation.

Memory optimization
-------------------
The full URL set across all snapshots is ~10GB (177M URLs), which exceeds the 31GB
node RAM with Python overhead. Instead of loading everything at once, we process one
snapshot at a time. Each snapshot has ~2-3M URLs (~300MB), well within node limits.
We serialize snapshot processing but parallelize across Nemotron files within each
snapshot via Zephyr flat_map.
"""

import gzip
import json
import logging
from dataclasses import dataclass

import fsspec

from zephyr import Dataset, ZephyrContext
from zephyr.execution import zephyr_worker_ctx
from zephyr.readers import load_file as zephyr_load_file

logger = logging.getLogger(__name__)

QUALITY_LEVELS = ("high", "medium-high", "medium", "medium-low", "low")


@dataclass
class FilterNemotronConfig:
    metadata_path: str
    """Glob pattern for WARC metadata JSONL files."""

    nemotron_base_path: str
    """GCS base path to Nemotron-CC v1 data-jsonl directory."""

    output_path: str


def _get_snapshots_from_metadata(metadata_path: str) -> dict[str, list[str]]:
    """Scan metadata to find which snapshots we have and which files contain them.

    Returns {snapshot: [metadata_file_paths]} so we can load URLs per-snapshot later.
    Only reads the first record of each file to determine its snapshot (all records
    in a metadata file share the same snapshot since they come from one WARC).
    """
    import glob as globmod

    if metadata_path.startswith("gs://"):
        fs = fsspec.filesystem("gcs")
        dir_path = (
            metadata_path.replace("gs://", "").rsplit("/", 1)[0]
            if "*" in metadata_path
            else metadata_path.replace("gs://", "")
        )
        files = sorted(f"gs://{f}" for f in fs.ls(dir_path, detail=False) if f.endswith(".jsonl.gz"))
    else:
        files = sorted(globmod.glob(metadata_path))

    snapshot_to_files: dict[str, list[str]] = {}
    for fpath in files:
        # Read first record to get snapshot
        for record in zephyr_load_file(fpath):
            snap = record.get("snapshot", "")
            if snap:
                snapshot_to_files.setdefault(snap, []).append(fpath)
            break  # only need first record

    total_files = sum(len(v) for v in snapshot_to_files.values())
    logger.info(f"Found {total_files} metadata files across {len(snapshot_to_files)} snapshots")
    return snapshot_to_files


def _load_urls_for_snapshot(snapshot: str, metadata_files: list[str]) -> set[str]:
    """Load URLs from metadata files for a single snapshot."""
    from concurrent.futures import ThreadPoolExecutor

    urls: set[str] = set()

    def _read_one(fpath):
        result = []
        for record in zephyr_load_file(fpath):
            url = record.get("url", "")
            if url:
                result.append(url)
        return result

    with ThreadPoolExecutor(max_workers=32) as pool:
        for batch in pool.map(_read_one, metadata_files):
            urls.update(batch)

    logger.info(f"  {snapshot}: loaded {len(urls):,} URLs from {len(metadata_files)} files")
    return urls


def _list_nemotron_files_for_snapshot(base_path: str, snapshot: str) -> list[dict]:
    """List all Nemotron files for a single snapshot across all quality levels."""
    fs = fsspec.filesystem("gcs") if base_path.startswith("gs://") else fsspec.filesystem("file")
    base = base_path.replace("gs://", "")
    tasks = []

    for quality in QUALITY_LEVELS:
        actual_dir = f"{base}/quality={quality}/kind=actual/kind2=actual"
        try:
            files = fs.ls(actual_dir, detail=False)
        except FileNotFoundError:
            continue

        for f in files:
            if not f.endswith(".jsonl.gz"):
                continue
            fname = f.split("/")[-1]
            if fname.startswith(snapshot):
                tasks.append(
                    {
                        "file_path": f"gs://{f}" if base_path.startswith("gs://") else f,
                        "quality": quality,
                        "snapshot": snapshot,
                    }
                )

    return tasks


def _process_nemotron_file(task: dict) -> list[dict]:
    """Process a single Nemotron file, filtering records whose URL is in our set."""
    ctx = zephyr_worker_ctx()
    url_set = ctx.get_shared("url_set")

    file_path = task["file_path"]
    quality = task["quality"]
    snapshot = task["snapshot"]

    results = []
    matched = 0
    total = 0

    with fsspec.open(file_path, "rb") as fh:
        with gzip.open(fh, "rt", encoding="utf-8") as gz:
            for line in gz:
                if not line.strip():
                    continue
                total += 1
                record = json.loads(line)
                url = record.get("metadata", {}).get("nemotron_url", "")
                if url in url_set:
                    matched += 1
                    results.append(
                        {
                            "text": record.get("text", ""),
                            "url": url,
                            "nemotron_quality": quality,
                            "nemotron_kind": "actual",
                            "nemotron_id": record.get("id", ""),
                        }
                    )

    logger.info(f"  {snapshot} quality={quality} {file_path.split('/')[-1]}: " f"{matched}/{total} matched")
    return results


def filter_nemotron(config: FilterNemotronConfig) -> None:
    """Filter Nemotron-CC v1 records matching our WARC files.

    Processes one snapshot at a time to keep memory under 31GB node limit.
    Each snapshot's URL set is ~300MB (vs 10GB for all snapshots at once).
    Within each snapshot, parallelizes across Nemotron files via Zephyr.
    """
    # Phase 1: Discover which snapshots we need (fast, reads 1 record per file)
    snapshot_to_files = _get_snapshots_from_metadata(config.metadata_path)

    # Track total output shard index across snapshots for unique filenames
    total_matched = 0
    total_scanned = 0

    # Phase 2: Process one snapshot at a time
    for snap_idx, (snapshot, meta_files) in enumerate(sorted(snapshot_to_files.items())):
        logger.info(f"[{snap_idx + 1}/{len(snapshot_to_files)}] Processing snapshot {snapshot}...")

        # Load URLs for just this snapshot (~300MB, fits in 31GB node)
        url_set = _load_urls_for_snapshot(snapshot, meta_files)
        if not url_set:
            logger.info(f"  {snapshot}: no URLs, skipping")
            continue

        # List Nemotron files for this snapshot
        tasks = _list_nemotron_files_for_snapshot(config.nemotron_base_path, snapshot)
        if not tasks:
            logger.info(f"  {snapshot}: no Nemotron files found, skipping")
            continue

        logger.info(f"  {snapshot}: {len(url_set):,} URLs, {len(tasks)} Nemotron files to scan")

        # Scan and filter in parallel within this snapshot
        pipeline = (
            Dataset.from_list(tasks)
            .flat_map(_process_nemotron_file)
            .write_jsonl(
                f"{config.output_path}/{snapshot}-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
                skip_existing=True,
            )
        )

        ctx = ZephyrContext(name=f"filter-nemotron-{snapshot}", max_workers=500)
        ctx.put("url_set", url_set)
        result = ctx.execute(pipeline)

        snap_files = len(list(result)) if result else 0
        total_scanned += len(tasks)
        logger.info(f"  {snapshot}: done, wrote {snap_files} output shards")

    logger.info(
        f"Nemotron filter complete: {len(snapshot_to_files)} snapshots, "
        f"{total_scanned} files scanned → {config.output_path}"
    )
