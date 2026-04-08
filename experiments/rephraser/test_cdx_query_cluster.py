# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: run CDX query + WARC download on the cluster.

Tests mathhelpforum.com end-to-end: CDX query (with retries for flaky API) then
WARC byte-range downloads producing non-empty HTML JSONL shards.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central2 --no_wait \\
        -- python experiments/rephraser/test_cdx_query_cluster.py
"""

from marin.datakit.download.commoncrawl.cdx_query import CDXQueryConfig, query_cdx
from marin.datakit.download.commoncrawl.download_warc_records import (
    WarcRecordDownloadConfig,
    download_warc_records,
)
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

# CDX query for mathhelpforum.com from a single crawl index.
# CC-MAIN-2018-30 has real HTML captures with actual content.
cdx_step = ExecutorStep(
    name="test/cdx_mathhelpforum_v5",
    description="CDX query for mathhelpforum.com (CC-MAIN-2018-30).",
    fn=query_cdx,
    config=CDXQueryConfig(
        url_patterns=versioned(["mathhelpforum.com"]),
        output_path=this_output_path(),
        crawl_indices=versioned(["CC-MAIN-2018-30"]),
        match_type="domain",
        request_delay=1.0,
    ),
)

# Download WARC records into 10 shards. Uses num_workers=10 (one per shard)
# to avoid OOM from Zephyr's default 128 workers.
download_step = ExecutorStep(
    name="test/warc_mathhelpforum_v5",
    description="Download WARC records for mathhelpforum.com.",
    fn=download_warc_records,
    config=WarcRecordDownloadConfig(
        cdx_manifest_path=cdx_step / "cdx_manifest.json",
        output_path=this_output_path(),
        num_output_shards=versioned(10),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[download_step],
        description="Smoke test: CDX query + WARC download (mathhelpforum.com).",
    )
