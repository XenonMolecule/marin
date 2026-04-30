# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add a synthetic `id` field to a jsonl.gz tree so marin's dedupe can run on it.

`dedup_exact_document` and `dedup_fuzzy_document` both require an `id` column
(connected_components.py asserts `int` or `str`). Neither resiliparse nor
llm_curated extractions carry one, so we synthesize a stable per-file,
per-record id of the form ``"{shard:05d}_{record:08d}"``.

One input file → one output file (zephyr's `from_list().flat_map()` preserves
the 1:1 input→shard mapping). The output schema is ``{id, text, ...metadata}``
where ``text`` is the chosen dedupe column and the rest of the original record
is preserved unchanged for downstream re-tokenization after `apply_dedup`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

from fray.v2.types import ResourceConfig
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_file as zephyr_load_file

from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


@dataclass
class PrepConfig:
    input_path: str
    """Glob root for source jsonl.gz files."""

    output_path: str
    """Output root; one .jsonl.gz per input shard."""

    text_field: str = "text"
    """Source field used as the dedupe `text` column."""

    max_parallelism: int = 200


def prep_with_id(config: PrepConfig) -> dict:
    """Materialize a copy of the input with an added stable `id` field."""
    files = sorted(fsspec_glob(f"{config.input_path.rstrip('/')}/*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"No jsonl.gz files under {config.input_path}")

    indexed = list(enumerate(files))
    text_field = config.text_field
    logger.info("prep_with_id: %d input files, text_field=%r", len(files), text_field)

    def _read_with_id(item: tuple[int, str]) -> Iterator[dict]:
        shard_idx, path = item
        for record_idx, record in enumerate(zephyr_load_file(path)):
            text = record.get(text_field)
            if not text:
                continue
            out = {"id": f"{shard_idx:05d}_{record_idx:08d}", "text": text}
            for k, v in record.items():
                if k == "text":
                    continue
                out[k] = v
            yield out

    ctx = ZephyrContext(
        name="prep-with-id",
        max_workers=min(config.max_parallelism, len(files)),
        resources=ResourceConfig(cpu=1, ram="8g", disk="5g"),
    )

    pipeline = (
        Dataset.from_list(indexed)
        .flat_map(_read_with_id)
        .write_jsonl(
            f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
            skip_existing=True,
        )
    )
    result = ctx.execute(pipeline)
    written = len(result.results)
    logger.info("prep_with_id: wrote %d shards to %s", written, config.output_path)
    return {"success": True, "input_files": len(files), "output_files": written}
