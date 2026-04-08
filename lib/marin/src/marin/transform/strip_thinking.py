# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace <think>...</think> blocks with Qwen3's native empty-think format.

Reads JSONL files with a "messages" column (OpenAI chat format), replaces
full thinking traces in assistant responses with ``<think>\\n\\n</think>\\n\\n``
(Qwen3's ``enable_thinking=False`` format). This preserves the model's
expected template structure while removing the actual reasoning content,
so the model can be served with the standard Qwen3 template using
``enable_thinking=False``.
"""

import logging
import re
from dataclasses import dataclass

from zephyr import Dataset, ZephyrContext, load_jsonl

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
# Qwen3's native no-think format: empty think block with two trailing newlines.
# Matches the output of apply_chat_template(..., enable_thinking=False).
_EMPTY_THINK = "<think>\n\n</think>\n\n"


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
            cleaned_content = _THINK_RE.sub(_EMPTY_THINK, content)
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

    ctx = ZephyrContext(name="strip-thinking")
    ctx.put("config", config)
    output_files = ctx.execute(pipeline)

    logger.info(
        "Strip-thinking complete: %d output files written to %s",
        len(output_files),
        config.output_path,
    )
