# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter Nemotron-CC v1 to records matching our selected WARC files.

Two filtering modes
-------------------
- ``filter_nemotron``: organic only. Scans ``kind=actual/kind2=actual`` across all 5
  quality levels.
- ``filter_nemotron_full``: organic plus rephraser-generated. Also scans
  ``kind=synthetic/kind2=*`` (distill, diverse_qa_pairs, extract_knowledge,
  knowledge_list, wrap_medium under quality=high; wrap_medium under quality=low).
  Each synthetic record carries ``metadata.nemotron_url`` pointing back at its source
  CC URL, so the same URL-based join works. The synthetic variants are what drive
  Nemotron-CC's headline token-count advantage over DCLM, so the "full" version is
  the right comparison against DCLM/FineWeb-Edu in apples-to-apples token counts.

Coverage argument
-----------------
Join key: nemotron_url field matched against WARC-Target-URI from our WARC headers.

Why this works: Both values originate from the same WARC header (WARC-Target-URI).
Nemotron's text extraction pipeline (Justext) reads Common Crawl WARCs and preserves
the URL as-is in the nemotron_url metadata field — synthetic records preserve it too,
because they are generated *from* an actual record and inherit its URL. Our metadata
extraction reads the same WARC files and extracts the same header. Therefore, string
equality is sufficient.

Scoping: Nemotron v1 data is partitioned by snapshot (filenames contain CC-MAIN-YYYY-WW).
We only scan files for snapshots present in our WARC set. Since Nemotron processes ALL
WARCs in each snapshot it covers (verified empirically — a random WARC from CC-MAIN-2025-18
produced matches in Nemotron-CC v2.1), every document from our WARCs that Nemotron kept
will appear in the snapshot's files.

Note: the same source URL can appear in *multiple* synthetic records (one per generator),
so a single WARC document may produce many output rows in the "full" mode. This is
correct — they're distinct documents in Nemotron-CC's training mix.

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
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext
from zephyr.execution import zephyr_worker_ctx
from zephyr.readers import load_file as zephyr_load_file

logger = logging.getLogger(__name__)

QUALITY_LEVELS = ("high", "medium-high", "medium", "medium-low", "low")


@dataclass
class FilterNemotronConfig:
    """Actual-only Nemotron-CC filter (scans kind=actual/kind2=actual)."""

    metadata_path: str
    """Glob pattern for WARC metadata JSONL files."""

    nemotron_base_path: str
    """GCS base path to Nemotron-CC v1 data-jsonl directory."""

    output_path: str


@dataclass
class FilterNemotronFullConfig:
    """Full Nemotron-CC filter (actual + all synthetic rephraser variants).

    Separate dataclass from ``FilterNemotronConfig`` so the two filters get
    distinct executor hashes and output directories even though the field
    shape is identical.
    """

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


def _enumerate_partitions(fs, base: str, include_synthetic: bool) -> list[tuple[str, str, str, str]]:
    """Enumerate ``(quality, kind, kind2, dir_path)`` tuples to scan.

    Always includes ``kind=actual/kind2=actual`` across all quality levels.
    When ``include_synthetic`` is True, also auto-discovers every
    ``kind=synthetic/kind2=*`` directory (e.g. distill, diverse_qa_pairs,
    extract_knowledge, knowledge_list, wrap_medium). The kind2 set varies
    per quality level, so we list it dynamically rather than hardcoding.
    """
    partitions: list[tuple[str, str, str, str]] = []
    for quality in QUALITY_LEVELS:
        partitions.append((quality, "actual", "actual", f"{base}/quality={quality}/kind=actual/kind2=actual"))
        if include_synthetic:
            synthetic_root = f"{base}/quality={quality}/kind=synthetic"
            try:
                kind2_dirs = fs.ls(synthetic_root, detail=False)
            except FileNotFoundError:
                continue
            for kind2_dir in kind2_dirs:
                kind2 = kind2_dir.rstrip("/").split("/")[-1].removeprefix("kind2=")
                partitions.append((quality, "synthetic", kind2, kind2_dir.rstrip("/")))
    return partitions


def _list_nemotron_files_for_snapshot(base_path: str, snapshot: str, include_synthetic: bool) -> list[dict]:
    """List Nemotron files for a single snapshot across all selected partitions."""
    fs = fsspec.filesystem("gcs") if base_path.startswith("gs://") else fsspec.filesystem("file")
    base = base_path.replace("gs://", "")
    tasks = []

    for quality, kind, kind2, dir_path in _enumerate_partitions(fs, base, include_synthetic):
        try:
            files = fs.ls(dir_path, detail=False)
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
                        "kind": kind,
                        "kind2": kind2,
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
    kind = task["kind"]
    kind2 = task["kind2"]
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
                            "nemotron_kind": kind,
                            "nemotron_kind2": kind2,
                            "nemotron_id": record.get("id", ""),
                        }
                    )

    logger.info(
        f"  {snapshot} quality={quality} kind={kind} kind2={kind2} "
        f"{file_path.split('/')[-1]}: {matched}/{total} matched"
    )
    return results


def _run_filter(
    metadata_path: str,
    nemotron_base_path: str,
    output_path: str,
    include_synthetic: bool,
    ctx_name_prefix: str,
) -> None:
    """Shared implementation for both the actual-only and full variants.

    Processes one snapshot at a time to keep memory under 31GB node limit.
    Each snapshot's URL set is ~300MB (vs 10GB for all snapshots at once).
    Within each snapshot, parallelizes across Nemotron files via Zephyr.
    """
    snapshot_to_files = _get_snapshots_from_metadata(metadata_path)

    total_scanned = 0

    for snap_idx, (snapshot, meta_files) in enumerate(sorted(snapshot_to_files.items())):
        logger.info(f"[{snap_idx + 1}/{len(snapshot_to_files)}] Processing snapshot {snapshot}...")

        url_set = _load_urls_for_snapshot(snapshot, meta_files)
        if not url_set:
            logger.info(f"  {snapshot}: no URLs, skipping")
            continue

        tasks = _list_nemotron_files_for_snapshot(nemotron_base_path, snapshot, include_synthetic)
        if not tasks:
            logger.info(f"  {snapshot}: no Nemotron files found, skipping")
            continue

        logger.info(f"  {snapshot}: {len(url_set):,} URLs, {len(tasks)} Nemotron files to scan")

        pipeline = (
            Dataset.from_list(tasks)
            .flat_map(_process_nemotron_file)
            .write_jsonl(
                f"{output_path}/{snapshot}-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
                skip_existing=True,
            )
        )

        # Each worker loads the shared url_set (up to ~2-3 GB with Python overhead
        # for the 10k manifest's larger snapshots) via get_shared, plus streams one
        # Nemotron shard (~500 MB gzipped, decompressed line-by-line). Overprovision
        # at 8 GiB to avoid tight-margin OOMs under adversarial inputs.
        ctx = ZephyrContext(
            name=f"{ctx_name_prefix}-{snapshot}",
            max_workers=500,
            resources=ResourceConfig(cpu=1, ram="8g"),
        )
        ctx.put("url_set", url_set)
        result = ctx.execute(pipeline)

        snap_files = len(result.results)
        total_scanned += len(tasks)
        logger.info(f"  {snapshot}: done, wrote {snap_files} output shards")

    logger.info(
        f"Nemotron filter complete: {len(snapshot_to_files)} snapshots, "
        f"{total_scanned} files scanned → {output_path}"
    )


def filter_nemotron(config: FilterNemotronConfig) -> None:
    """Filter Nemotron-CC v1 to organic (actual) records matching our WARC files."""
    _run_filter(
        metadata_path=config.metadata_path,
        nemotron_base_path=config.nemotron_base_path,
        output_path=config.output_path,
        include_synthetic=False,
        ctx_name_prefix="filter-nemotron",
    )


def filter_nemotron_full(config: FilterNemotronFullConfig) -> None:
    """Filter Nemotron-CC v1 to organic + rephraser-synthetic records matching our WARC files."""
    _run_filter(
        metadata_path=config.metadata_path,
        nemotron_base_path=config.nemotron_base_path,
        output_path=config.output_path,
        include_synthetic=True,
        ctx_name_prefix="filter-nemotron-full",
    )
