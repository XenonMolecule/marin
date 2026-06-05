# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-process rephraser inference output for use as training data.

Strips thinking tokens (<think>...</think>), removes DSPy-style field markers
([[ ## text ## ]], [[ ## completed ## ]]), filters out records matching exclusion
patterns (e.g. [NO_USEFUL_CONTENT]), and enforces minimum output length.

Writes cleaned records with a "text" field suitable for tokenization, plus a
postprocess_stats.json summary of kept/filtered counts.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import fsspec
from zephyr import Dataset, ZephyrContext, load_file

logger = logging.getLogger(__name__)

# Pre-compiled patterns for stripping inference artifacts
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_FIELD_MARKERS_RE = re.compile(
    r"\[\[\s*##\s*(text|completed)\s*##\s*\]\]",
    flags=re.IGNORECASE,
)


@dataclass
class PostProcessExtractionConfig:
    input_path: str
    """Glob pattern to inference output JSONL files."""

    output_path: str
    """Where to write cleaned JSONL output."""

    generated_text_column: str = "generated_text"
    """Column name containing the raw model output."""

    output_text_column: str = "text"
    """Column name for the cleaned text in output records (standard for tokenization)."""

    strip_thinking: bool = True
    """Whether to strip <think>...</think> blocks from model output."""

    filter_patterns: list[str] = field(
        default_factory=lambda: [
            r"\[NO_USEFUL_CONTENT\]",
        ]
    )
    """Regex patterns — if any matches the cleaned text, the record is dropped."""

    min_output_chars: int = 50
    """Minimum character length of cleaned text to keep a record."""


def _clean_text(raw_text: str, strip_thinking: bool) -> str:
    """Strip thinking tokens and DSPy field markers from raw model output."""
    text = raw_text
    if strip_thinking:
        text = _THINK_RE.sub("", text)
    text = _FIELD_MARKERS_RE.sub("", text)
    return text.strip()


def _process_record(record: dict[str, Any]) -> dict[str, Any]:
    """Clean a single inference output record."""
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: PostProcessExtractionConfig = ctx.get_shared("config")

    raw = record.get(config.generated_text_column, "")
    cleaned = _clean_text(raw, config.strip_thinking)
    record[config.output_text_column] = cleaned
    return record


def _should_keep(record: dict[str, Any]) -> bool:
    """Return True if the cleaned record passes all filters."""
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: PostProcessExtractionConfig = ctx.get_shared("config")
    compiled_patterns: list[re.Pattern] = ctx.get_shared("compiled_patterns")

    text = record.get(config.output_text_column, "")

    if len(text) < config.min_output_chars:
        return False

    for pattern in compiled_patterns:
        if pattern.search(text):
            return False

    return True


def postprocess_extraction(config: PostProcessExtractionConfig) -> None:
    """Clean inference output and filter low-quality records.

    Reads inference output files (parquet or JSONL), strips thinking tokens and
    field markers, filters records matching exclusion patterns or below minimum
    length, and writes cleaned JSONL with a "text" column ready for tokenization.
    """
    compiled_patterns = [re.compile(p) for p in config.filter_patterns]

    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_file)
        .map(_process_record)
        .filter(_should_keep)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="postprocess-extraction") as ctx:
        ctx.put("config", config)
        ctx.put("compiled_patterns", compiled_patterns)
        output_files = ctx.execute(pipeline)

    # Write stats summary for debugging
    stats = {
        "output_files": len(output_files),
        "input_path": config.input_path,
        "filter_patterns": config.filter_patterns,
        "min_output_chars": config.min_output_chars,
        "strip_thinking": config.strip_thinking,
    }
    stats_path = f"{config.output_path}/postprocess_stats.json"
    with fsspec.open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Post-processing complete: %d output files written to %s",
        len(output_files),
        config.output_path,
    )
