# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter datasets to remove examples where a text column exceeds a token limit.

Simple token-count filter: tokenizes a specified column and drops rows that exceed
``max_tokens``. Useful as a pre-inference filter to avoid wasting compute on documents
that would be truncated anyway.

Uses a character-length pre-check to avoid expensive tokenizer.encode() calls on
documents that are obviously too long or obviously short enough. For typical
English/HTML text, 1 token ~ 3-4 characters, so documents with fewer characters
than max_tokens are always safe, and documents with more characters than
max_tokens * 6 are always too long.

Distributes work across the Ray cluster via Zephyr. Each input file becomes a shard
that Zephyr schedules on an independent worker node. With the character-length fast
path, most records are decided in microseconds (just ``len(text)``), so even
50k-record shards complete in seconds.

Example usage as an executor step::

    filtered = ExecutorStep(
        name="filtered/my_dataset",
        fn=filter_by_token_length,
        config=FilterByTokenLengthConfig(
            input_path=raw_data / "**/*.jsonl.gz",
            output_path=this_output_path(),
            tokenizer="Qwen/Qwen3-8B",
            text_column="html",
            max_tokens=28672,
        ),
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu"],
    )
"""

import dataclasses
import gzip
import json
import logging
from collections.abc import Iterator

import draccus
import fsspec
import transformers
from zephyr import Dataset, ZephyrContext, load_jsonl, zephyr_worker_ctx

from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

# Characters-per-token bounds for the fast pre-check.
# Most tokenizers produce 1 token per 3-5 characters for English/HTML.
# - Lower bound (1 char/token): any text with fewer chars than max_tokens
#   is guaranteed to have fewer tokens, so we can skip tokenization.
# - Upper bound (6 chars/token): any text with more chars than max_tokens * 6
#   is almost certainly over the limit. We use 6 as a conservative upper bound
#   to avoid false positives (some tokens like single punctuation are 1 char).
_CHARS_PER_TOKEN_UPPER = 6


@dataclasses.dataclass
class FilterByTokenLengthConfig:
    """Configuration for filtering examples by token count in a text column.

    Attributes:
        input_path: Glob pattern for input JSONL files.
        output_path: Directory to write filtered JSONL output.
        tokenizer: HuggingFace tokenizer name or path used for tokenizing.
        text_column: Name of the field to tokenize and check length against.
        max_tokens: Maximum number of tokens allowed. Rows exceeding this are dropped.
    """

    input_path: str
    output_path: str
    tokenizer: str
    text_column: str = "text"
    max_tokens: int = 28672


def _filter_records(*, config: FilterByTokenLengthConfig, records: Iterator[dict]) -> Iterator[dict]:
    """Worker function: check each record's text length and yield those within the limit.

    Uses a character-length fast path to skip tokenization for the vast majority
    of records: documents shorter than max_tokens characters are guaranteed safe
    (since every token is at least 1 character), and documents longer than
    max_tokens * 6 characters are guaranteed over the limit. Only borderline
    documents (between those thresholds) need actual tokenization.
    """
    tokenizer = zephyr_worker_ctx().get_shared("tokenizer")
    char_safe_threshold = config.max_tokens
    char_drop_threshold = config.max_tokens * _CHARS_PER_TOKEN_UPPER

    for record in records:
        text = record.get(config.text_column)
        if text is None:
            continue

        text_len = len(text)

        # Fast path: short documents are guaranteed safe.
        if text_len <= char_safe_threshold:
            yield record
            continue

        # Fast path: very long documents are definitely over the limit.
        if text_len > char_drop_threshold:
            continue

        # Borderline: need actual tokenization.
        token_ids = tokenizer.encode(text)
        if len(token_ids) <= config.max_tokens:
            yield record


def filter_by_token_length(config: FilterByTokenLengthConfig):
    """Filter a JSONL dataset, keeping only examples within the token limit.

    Distributes work via Zephyr: each input file is a shard processed on an
    independent worker across the cluster.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=True)

    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .map_shard(lambda records: _filter_records(config=config, records=records))
        .write_jsonl(f"{config.output_path}/{{shard:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="filter-by-token-length") as ctx:
        ctx.put("tokenizer", tokenizer)
        ctx.execute(pipeline)

    # Count output records and remove empty shards (downstream tokenize step
    # crashes on empty files).
    output_files = fsspec_glob(f"{config.output_path}/*.jsonl.gz")
    kept = 0
    empty_files: list[str] = []
    for path in output_files:
        shard_count = 0
        with fsspec.open(path, "rb") as f:
            with gzip.open(f, "rt", encoding="utf-8") as gz:
                for _ in gz:
                    shard_count += 1
        if shard_count == 0:
            empty_files.append(path)
        kept += shard_count

    for path in empty_files:
        fs, _ = fsspec.core.url_to_fs(path)
        fs.rm(path)
        logger.info(f"Removed empty shard: {path}")
    if empty_files:
        logger.info(f"Removed {len(empty_files)} empty shard(s) out of {len(output_files)} total.")

    logger.info("=" * 70)
    logger.info(
        f"FILTER COMPLETE: kept {kept} examples " f"(text_column={config.text_column!r}, max_tokens={config.max_tokens})"
    )
    logger.info("=" * 70)

    stats = {
        "kept": kept,
        "max_tokens": config.max_tokens,
        "text_column": config.text_column,
        "tokenizer": config.tokenizer,
    }
    stats_path = f"{config.output_path}/filter_stats.json"
    with fsspec.open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote filter stats to {stats_path}")


@draccus.wrap()
def main(config: FilterByTokenLengthConfig):
    filter_by_token_length(config)


if __name__ == "__main__":
    main()
