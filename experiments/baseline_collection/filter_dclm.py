# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter DCLM-baseline-1.0 to records matching our selected WARC files.

Coverage argument
-----------------
Join key: metadata.WARC-Record-ID in DCLM matched against WARC-Record-ID header
from our WARC files. Both are globally unique UUIDs assigned by Common Crawl when
the WARC file was written. There is zero ambiguity — one UUID identifies exactly
one HTTP response record in one WARC file.

Format: DCLM stores record IDs as '<urn:uuid:8eeff0ee-...>'. We normalize by
stripping the '<urn:uuid:>' wrapper. Our metadata stores the bare UUID '8eeff0ee-...'.

Why full scan: DCLM-baseline-1.0 is NOT partitioned by snapshot or WARC file. It
consists of 27,838 jsonl.zst shards mixed across all crawls. There is no way to
read "just the records from CC-MAIN-2022-49" without scanning everything. The
warcinfo.isPartOf field contains the snapshot but is inside the JSON payload, not
in the file structure.

Coverage guarantee: Every DCLM record has a metadata.WARC-Record-ID field (it is
part of the WARC headers that DCLM preserves verbatim). Every HTML record in our
WARCs has a WARC-Record-ID header. The intersection is exact. If a document from
our WARCs survived DCLM's filtering pipeline, we will find it.

The only records we WON'T find are ones DCLM filtered out (non-English, low quality,
deduplicated). This is expected and correct — it represents DCLM's curation decisions,
which is exactly what we want to measure.
"""

import json
import logging
from dataclasses import dataclass

import fsspec
import zstandard
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext
from zephyr.execution import zephyr_worker_ctx

logger = logging.getLogger(__name__)


@dataclass
class FilterDclmConfig:
    metadata_path: str
    """Glob pattern for WARC metadata JSONL files."""

    dclm_base_path: str
    """GCS base path to DCLM-baseline-1.0 data directory."""

    output_path: str


def _normalize_record_id(raw_id: str) -> str:
    """Strip <urn:uuid:...> wrapper to get bare UUID."""
    return raw_id.strip("<>").removeprefix("urn:uuid:")


def _load_record_id_set(metadata_path: str) -> set[str]:
    """Load metadata and build set of all WARC record IDs.

    For the 10k manifest the set is ~400M UUIDs which as Python strings is
    ~80-100 GB. The driver needs a >150 GB allocation; provisioned in the
    pipeline via the outer ``remote()`` wrapper.
    """
    import glob as globmod

    from zephyr.readers import load_file as zephyr_load_file

    record_ids: set[str] = set()

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

    from concurrent.futures import ThreadPoolExecutor

    def _read_one_file(fpath):
        return [r.get("warc_record_id", "") for r in zephyr_load_file(fpath) if r.get("warc_record_id")]

    with ThreadPoolExecutor(max_workers=64) as pool:
        for batch in pool.map(_read_one_file, files):
            record_ids.update(batch)

    logger.info(f"Loaded {len(record_ids):,} record IDs for DCLM filtering from {len(files)} files")
    return record_ids


def _list_dclm_shard_files(base_path: str) -> list[str]:
    """List all DCLM jsonl.zst shard files on GCS."""
    fs = fsspec.filesystem("gcs") if base_path.startswith("gs://") else fsspec.filesystem("file")
    base = base_path.replace("gs://", "")

    all_files = []
    # DCLM structure: global-shard_XX_of_10/local-shard_Y_of_10/shard_NNNN_processed.jsonl.zst
    for global_shard in fs.ls(base, detail=False):
        for local_shard in fs.ls(global_shard, detail=False):
            files = fs.ls(local_shard, detail=False)
            for f in files:
                if f.endswith(".jsonl.zst"):
                    all_files.append(f"gs://{f}" if base_path.startswith("gs://") else f)

    logger.info(f"Found {len(all_files):,} DCLM shard files")
    return all_files


def _process_dclm_file(file_path: str) -> list[dict]:
    """Process a single DCLM shard file, filtering by record ID set."""
    ctx = zephyr_worker_ctx()
    record_id_set = ctx.get_shared("record_id_set")

    results = []
    matched = 0
    total = 0

    with fsspec.open(file_path, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            import io

            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                if not line.strip():
                    continue
                total += 1
                record = json.loads(line)
                metadata = record.get("metadata", {})
                raw_id = metadata.get("WARC-Record-ID", "")
                normalized_id = _normalize_record_id(raw_id)

                if normalized_id in record_id_set:
                    matched += 1
                    results.append(
                        {
                            "text": record.get("text", ""),
                            "url": record.get("url", ""),
                            "warc_record_id": normalized_id,
                            "dclm_fasttext_score": record.get(
                                "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob"
                            ),
                            "dclm_language_score": record.get("language_id_whole_page_fasttext"),
                        }
                    )

    if matched > 0:
        logger.info(f"  {file_path.split('/')[-1]}: {matched}/{total} matched")
    return results


def filter_dclm(config: FilterDclmConfig) -> None:
    """Filter DCLM-baseline-1.0 records matching our WARC files.

    Full scan of all 27,838 shard files with hash-set lookup. This is the most
    expensive step (~30min at 500 workers) but guarantees complete coverage.
    """
    record_id_set = _load_record_id_set(config.metadata_path)
    dclm_files = _list_dclm_shard_files(config.dclm_base_path)

    pipeline = (
        Dataset.from_list(dclm_files)
        .flat_map(_process_dclm_file)
        .write_jsonl(
            f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
            skip_existing=True,
        )
    )

    # Each worker calls get_shared("record_id_set") which loads the full set
    # via cloudpickle.loads. For the 10k manifest's ~400M UUID strings the set
    # is ~80-100 GB. Each worker needs to fit it. Driver request is set in the
    # outer remote() wrapper in the pipeline file.
    ctx = ZephyrContext(
        name="filter-dclm",
        max_workers=100,
        resources=ResourceConfig(cpu=1, ram="128g"),
        coordinator_resources=ResourceConfig(cpu=1, ram="128g"),
    )
    ctx.put("record_id_set", record_id_set)
    ctx.execute(pipeline)

    logger.info(f"DCLM filter complete → {config.output_path}")
