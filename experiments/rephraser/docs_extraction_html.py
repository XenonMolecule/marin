# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch HTML from technical documentation sites via Common Crawl CDX + WARC download.

Queries CDX for each documentation source, downloads WARC records, and writes
JSONL shards with raw HTML for downstream extraction testing.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -- python experiments/rephraser/docs_extraction_html.py
"""

from fray.cluster import ResourceConfig
from marin.datakit.download.commoncrawl.cdx_query import CDXQueryConfig, query_cdx
from marin.datakit.download.commoncrawl.download_warc_records import (
    WarcRecordDownloadConfig,
    download_warc_records,
)
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

# Recent crawl indices with good coverage
CRAWL_INDICES = ["CC-MAIN-2024-10", "CC-MAIN-2024-18"]

# ---- Sources grouped by CDX match type ----

# Host-level: entire subdomain is documentation
HOST_SOURCES = [
    "devdocs.io",
    "developer.mozilla.org",
    "www.tensorflow.org",
    "api.flutter.dev",
    "docs.nvidia.com",
    "tutorial.ponylang.io",
    "doc.qt.io",
]

# Prefix: specific paths within larger sites
PREFIX_SOURCES = [
    "www.kernel.org/doc/Documentation",
    "docs.swift.org/swift-book/documentation/the-swift-programming-language",
    "www.typescriptlang.org/docs/handbook",
    "www.newtonsoft.com/json/help/html",
    "docs.oracle.com/javase/tutorial/java",
    "qiskit.org/documentation",
    "learn.microsoft.com/en-us/azure/quantum/user-guide",
    "learn.microsoft.com/en-us/dotnet/csharp",
    "huggingface.co/docs",
    "llvm.org/docs",
    "gcc.gnu.org/onlinedocs",
    "www.mathworks.com/help/matlab",
    "www.boost.org/doc",
    "www.qemu.org/documentation",
    "docs.zephir-lang.com/0.12/en/introduction",
    "maxima.sourceforge.io/docs/manual/maxima_singlepage.html",
]

# ---- CDX query steps ----

host_cdx_step = ExecutorStep(
    name="docs_html/cdx_host",
    description="CDX query for documentation host sources.",
    fn=query_cdx,
    config=CDXQueryConfig(
        url_patterns=versioned(HOST_SOURCES),
        output_path=this_output_path(),
        crawl_indices=versioned(CRAWL_INDICES),
        match_type="host",
        request_delay=1.0,
    ),
    resources=ResourceConfig.with_cpu(cpu=2, ram="8g"),
    pip_dependency_groups=["cpu"],
)

prefix_cdx_step = ExecutorStep(
    name="docs_html/cdx_prefix",
    description="CDX query for documentation prefix sources.",
    fn=query_cdx,
    config=CDXQueryConfig(
        url_patterns=versioned(PREFIX_SOURCES),
        output_path=this_output_path(),
        crawl_indices=versioned(CRAWL_INDICES),
        match_type="prefix",
        request_delay=1.0,
    ),
    resources=ResourceConfig.with_cpu(cpu=2, ram="8g"),
    pip_dependency_groups=["cpu"],
)

# ---- Download steps ----

host_download_step = ExecutorStep(
    name="docs_html/download_host",
    description="Download WARC records for host doc sources.",
    fn=download_warc_records,
    config=WarcRecordDownloadConfig(
        cdx_manifest_path=host_cdx_step / "cdx_manifest.json",
        output_path=this_output_path(),
        num_output_shards=versioned(50),
    ),
    resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
    pip_dependency_groups=["cpu"],
)

prefix_download_step = ExecutorStep(
    name="docs_html/download_prefix",
    description="Download WARC records for prefix doc sources.",
    fn=download_warc_records,
    config=WarcRecordDownloadConfig(
        cdx_manifest_path=prefix_cdx_step / "cdx_manifest.json",
        output_path=this_output_path(),
        num_output_shards=versioned(50),
    ),
    resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
    pip_dependency_groups=["cpu"],
)

if __name__ == "__main__":
    executor_main(
        steps=[host_download_step, prefix_download_step],
        description="Fetch HTML from technical documentation sites via Common Crawl.",
    )
