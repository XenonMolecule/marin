# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resiliparse extraction for the full DCLM 400m-1x 10k pool.

The 10k raw HTML download was cleaned up 2026-04-28 (see
``.agents/projects/10k_warc_cleanup.md``) after building the DCLM /
Nemotron-full / FineWeb-Edu products — resiliparse was never run. This pipeline
reuses the **same** ``download_warcs`` step from ``pipeline_10k.py`` (identical
config → same ``raw/commoncrawl/dclm_400m_1x_10k`` prefix), so the executor
re-downloads the ~8 TB raw once and persists it; any other 10k product can then
piggyback without re-downloading. It then runs resiliparse main-content
extraction over all 10,364 WARCs.

Output: ``extracted/dclm_400m_1x_10k_resiliparse-<hash>/data-*-of-10364.jsonl.gz``
(schema ``{text, url}``) — the raw input for
``dedup_resiliparse_warc_scaling.py --n 10364 --extraction-name <name>
--total-shards 10364``.

CommonCrawl reads are free ingress (``download_warcs`` rewrites
``s3://commoncrawl`` → ``https://data.commoncrawl.org``); no AWS egress.

Usage (Iris, CPU, us-central2 where the raw pool + nemotron base live)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 50GB \\
        --priority interactive --extra cpu --enable-extra-resources \\
        --region us-central2 --job-name resiliparse-extract-10k \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/baseline_collection/pipeline_10k_resiliparse.py
"""

from fray.types import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.transform.extract_text_from_html import ExtractTextConfig

from experiments.baseline_collection.pipeline import extract_text_fast
from experiments.baseline_collection.pipeline_10k import download_warcs

# Resiliparse main-content extraction over the (re-downloaded) 10k WARCs. Reads
# the shared download step's HTML output; one output shard per input WARC.
extract_resiliparse_10k = ExecutorStep(
    name="extracted/dclm_400m_1x_10k_resiliparse",
    description="Resiliparse main-content extraction over the full DCLM 400m-1x 10k pool (10,364 WARCs).",
    fn=remote(extract_text_fast, resources=ResourceConfig(cpu=4, ram="32g")),
    config=ExtractTextConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[extract_resiliparse_10k],
        description=(
            "Resiliparse extraction for the 10k DCLM 400m-1x pool. Shares pipeline_10k's "
            "download_warcs step (re-downloads the cleaned-up ~8 TB raw, then extracts)."
        ),
    )
