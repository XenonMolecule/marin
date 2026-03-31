# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strip <think>...</think> blocks from assistant messages in chat-format datasets.

Reads JSONL files with a "messages" column (OpenAI chat format), removes
thinking traces from assistant responses, and writes cleaned JSONL. The
messages structure is preserved -- only the content of assistant messages
is modified.
"""

import logging
import re
from dataclasses import dataclass

from zephyr import Dataset, ZephyrContext, load_jsonl, write_jsonl_file

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


@dataclass
class StripThinkingConfig:
    input_path: str
    """Glob pattern to JSONL files with a 'messages' column."""

    output_path: str
    """Where to write cleaned JSONL output."""

    messages_column: str = "messages"
    """Column containing the chat messages list."""


def _strip_thinking_from_record(record: dict) -> dict:
    """Strip <think>...</think> blocks from assistant messages in a record."""
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: StripThinkingConfig = ctx.get_shared("config")

    messages = record.get(config.messages_column, [])
    cleaned_messages = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            cleaned_content = _THINK_RE.sub("", content).strip()
            cleaned_messages.append({**msg, "content": cleaned_content})
        else:
            cleaned_messages.append(msg)

    record[config.messages_column] = cleaned_messages
    return record


def strip_thinking(config: StripThinkingConfig) -> None:
    """Strip thinking traces from assistant messages in chat-format data."""
    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .map(_strip_thinking_from_record)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="strip-thinking") as ctx:
        ctx.put("config", config)
        output_files = ctx.execute(pipeline)

    logger.info(
        "Strip-thinking complete: %d output files written to %s",
        len(output_files),
        config.output_path,
    )
