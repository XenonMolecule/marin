# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract and clean assistant text from rephraser chat-format datasets.

Reads parquet files with a "messages" column (OpenAI chat format),
extracts the last assistant response, strips thinking tokens and DSPy
field markers, filters short/empty records, and writes JSONL with a
plain "text" column ready for tokenization.

Uses the same cleaning regexes as postprocess_extraction.py.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import fsspec

from zephyr import Dataset, ZephyrContext, load_parquet

logger = logging.getLogger(__name__)

# Same patterns as postprocess_extraction.py
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_FIELD_MARKERS_RE = re.compile(
    r"\[\[\s*##\s*(text|completed)\s*##\s*\]\]",
    flags=re.IGNORECASE,
)


@dataclass
class CleanRephraserMessagesConfig:
    input_path: str
    """Glob pattern to parquet files with a 'messages' column."""

    output_path: str
    """Where to write cleaned JSONL output."""

    messages_column: str = "messages"
    """Column containing the chat messages list."""

    output_text_column: str = "text"
    """Column name for the cleaned text in output records."""

    strip_thinking: bool = True
    """Whether to strip <think>...</think> blocks."""

    filter_patterns: list[str] = field(
        default_factory=lambda: [
            r"\[NO_USEFUL_CONTENT\]",
        ]
    )
    """Regex patterns — if any matches the cleaned text, the record is dropped."""

    min_output_chars: int = 50
    """Minimum character length of cleaned text to keep a record."""


def _extract_assistant_text(messages: list[dict[str, str]]) -> str:
    """Extract the content of the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _clean_text(raw_text: str, strip_thinking: bool) -> str:
    """Strip thinking tokens and DSPy field markers from raw model output."""
    text = raw_text
    if strip_thinking:
        text = _THINK_RE.sub("", text)
    text = _FIELD_MARKERS_RE.sub("", text)
    return text.strip()


def _process_record(record: dict[str, Any]) -> dict[str, Any]:
    """Extract assistant text, clean it, and store in the output column."""
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: CleanRephraserMessagesConfig = ctx.get_shared("config")

    messages = record.get(config.messages_column, [])
    raw = _extract_assistant_text(messages)
    cleaned = _clean_text(raw, config.strip_thinking)
    record[config.output_text_column] = cleaned
    return record


def _should_keep(record: dict[str, Any]) -> bool:
    """Return True if the cleaned record passes all filters."""
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: CleanRephraserMessagesConfig = ctx.get_shared("config")
    compiled_patterns: list[re.Pattern] = ctx.get_shared("compiled_patterns")

    text = record.get(config.output_text_column, "")

    if len(text) < config.min_output_chars:
        return False

    for pattern in compiled_patterns:
        if pattern.search(text):
            return False

    return True


def clean_rephraser_messages(config: CleanRephraserMessagesConfig) -> None:
    """Extract and clean assistant text from rephraser chat-format data.

    Reads parquet files with a messages column, extracts the assistant
    response, strips thinking tokens and DSPy field markers, filters
    short/bad records, and writes cleaned JSONL with a "text" column.
    """
    compiled_patterns = [re.compile(p) for p in config.filter_patterns]

    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_parquet)
        .map(_process_record)
        .filter(_should_keep)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="clean-rephraser-messages") as ctx:
        ctx.put("config", config)
        ctx.put("compiled_patterns", compiled_patterns)
        output_files = ctx.execute(pipeline)

    stats = {
        "output_files": len(output_files),
        "input_path": config.input_path,
        "filter_patterns": config.filter_patterns,
        "min_output_chars": config.min_output_chars,
        "strip_thinking": config.strip_thinking,
    }
    stats_path = f"{config.output_path}/clean_stats.json"
    with fsspec.open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Cleaning complete: %d output files written to %s",
        len(output_files),
        config.output_path,
    )
