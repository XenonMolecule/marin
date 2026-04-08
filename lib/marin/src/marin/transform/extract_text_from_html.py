# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract plain text from HTML using resiliparse.

Converts HTML records (from WARC download) to plain text suitable for the DCLM
filtering pipeline. Uses resiliparse's ``extract_plain_text`` with main content
extraction enabled, matching what DCLM uses for text extraction.

Usage as an ExecutorStep::

    extract_step = ExecutorStep(
        name="processed/dclm_text_from_warcs",
        fn=extract_text_from_html,
        config=ExtractTextConfig(
            input_path=download_warcs / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu"],
    )
"""

import logging
from dataclasses import dataclass

from zephyr import Dataset, ZephyrContext, load_file

logger = logging.getLogger(__name__)


@dataclass
class ExtractTextConfig:
    input_path: str
    """Glob pattern for input JSONL files containing HTML records."""

    output_path: str
    """Where to write output JSONL files with extracted text."""

    html_column: str = "html"
    """Column name containing the HTML content."""

    url_column: str = "url"
    """Column name containing the page URL."""

    main_content: bool = True
    """Whether to use resiliparse's main content extraction."""

    min_text_length: int = 50
    """Minimum text length (chars) to keep a record."""


def _extract_text(record: dict) -> dict:
    """Extract plain text from an HTML record using resiliparse."""
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: ExtractTextConfig = ctx.get_shared("config")

    html = record.get(config.html_column, "")
    url = record.get(config.url_column, "")

    try:
        tree = HTMLTree.parse(html)
        text = extract_plain_text(
            tree,
            main_content=config.main_content,
            alt_texts=False,
            noscript=False,
        )
    except Exception:
        text = ""

    return {"text": text.strip(), "url": url}


def _is_non_empty(record: dict) -> bool:
    """Filter out empty or too-short text records."""
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: ExtractTextConfig = ctx.get_shared("config")
    return len(record.get("text", "")) >= config.min_text_length


def extract_text_from_html(config: ExtractTextConfig) -> None:
    """Extract plain text from HTML records using resiliparse.

    Reads HTML records from WARC download output, extracts plain text using
    resiliparse's main content extraction, and writes JSONL with ``text`` and
    ``url`` fields suitable for the DCLM filtering pipeline.
    """
    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_file)
        .map(_extract_text)
        .filter(_is_non_empty)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    ctx = ZephyrContext(name="extract-text-from-html")
    ctx.put("config", config)
    output_files = ctx.execute(pipeline)

    logger.info(
        "Text extraction complete: %d output files written to %s",
        len(output_files),
        config.output_path,
    )
