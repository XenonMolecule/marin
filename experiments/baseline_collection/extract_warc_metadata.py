# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract per-record metadata from downloaded WARC HTML files.

Reads the HTML JSONL output from download_warcs and emits a lightweight metadata
record per document: {warc_record_id, url, warc_file, snapshot}. This is ~150MB
total for 3000 WARCs (vs ~2.2TB for the full HTML) and serves as the universal
join key for all three baseline filters:

- Nemotron-CC: joins on url (matched against nemotron_url)
- DCLM: joins on warc_record_id (matched against metadata.WARC-Record-ID)
- FineWeb-Edu: joins on warc_file (matched against file_path column)
"""

import logging
import re
from dataclasses import dataclass

from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)


@dataclass
class ExtractWarcMetadataConfig:
    input_path: str
    """Glob pattern for downloaded HTML JSONL files (e.g., download_step / '*.jsonl.gz')."""

    output_path: str
    """Output path for metadata JSONL files."""


def normalize_record_id(raw_id: str) -> str:
    """Strip <urn:uuid:...> wrapper to get bare UUID.

    WARC-Record-ID headers are formatted as '<urn:uuid:8eeff0ee-...>'.
    Nemotron-CC v2.1 and our metadata store just the UUID: '8eeff0ee-...'.
    """
    return raw_id.strip("<>").removeprefix("urn:uuid:")


def extract_snapshot(warc_path: str) -> str:
    """Extract CC-MAIN-YYYY-WW snapshot identifier from a WARC path."""
    m = re.search(r"CC-MAIN-\d{4}-\d{2}", warc_path)
    return m.group(0) if m else "unknown"


def _extract_metadata(record: dict) -> dict:
    """Transform a downloaded HTML record into a lightweight metadata record."""
    warc_file = record.get("metadata", {}).get("warc_file", "")
    return {
        "warc_record_id": normalize_record_id(record.get("id", "")),
        "url": record.get("url", ""),
        "warc_file": warc_file,
        "snapshot": extract_snapshot(warc_file),
    }


def extract_warc_metadata(config: ExtractWarcMetadataConfig) -> None:
    """Extract metadata from downloaded WARC HTML files."""
    pipeline = (
        Dataset.from_files(config.input_path)
        .load_file()
        .map(_extract_metadata)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    # load_file() reads the full HTML JSONL.gz shard (decoded HTML can be several
    # GB per shard for large WARCs). 8 GiB workers give comfortable headroom
    # against tight-margin OOMs.
    ctx = ZephyrContext(
        name="extract-warc-metadata",
        max_workers=500,
        resources=ResourceConfig(cpu=1, ram="8g"),
    )
    ctx.execute(pipeline)

    logger.info(f"Metadata extraction complete → {config.output_path}")
