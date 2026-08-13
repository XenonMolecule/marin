# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FineWeb-Edu filter for the full 10k DCLM 400m-1x WARC pool.

Companion to ``pipeline_10k.py`` (Nemotron) and ``pipeline_10k_dclm.py`` (DCLM).
Reuses the WARC metadata that ``pipeline_10k.py`` already produced — no
re-download, no re-extract. Just runs:

    filter_fineweb_edu  →  default_tokenize

against the 10,364 WARCs' file_path set, joined into FineWeb-Edu records via
the parquet ``file_path`` column. Same join logic as the 3k baseline pipeline.

This is the last derived dataset we want from the 10k pool before retiring the
raw HTML. Expected size: ~3-5 GB filtered text, ~2-3 B tokens (3.4× the 3k
pool's 0.82 B because the 10k pool is 3.4× larger).

Usage::

    uv run iris --config lib/iris/examples/marin.yaml job run \\
        --region us-central2 --extra cpu --enable-extra-resources \\
        --cpu 4 --memory 32GB --disk 50GB \\
        --priority batch --max-retries 20 \\
        --job-name baseline-fineweb-edu-10k --no-wait \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/baseline_collection/pipeline_10k_fineweb_edu.py
"""

from pathlib import Path

from fray.types import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote

from experiments.baseline_collection.filter_fineweb_edu import FilterFinewebEduConfig, filter_fineweb_edu
from experiments.defaults import default_tokenize

# --- Paths ---

# Reuse the metadata extracted by pipeline_10k.py (already complete on GCS).
# Hardcoded to share the existing output rather than re-extract.
METADATA_PATH = "gs://marin-us-central2/metadata/dclm_400m_1x_10k_warc_metadata-79158f/*.jsonl.gz"

# Fast path: load WARC paths directly from the manifest (snapshot derived from
# path via regex). Avoids 30-45 min of metadata-file GCS reads in the driver,
# which kept getting preempted before any filter work could start.
MANIFEST_PATH = str(Path(__file__).resolve().parent.parent / "distill" / "dclm_400m_1x.txt")

FINEWEB_EDU_BASE = "gs://marin-us-central2/raw/fineweb-edu"

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# --- Step 1: Filter FineWeb-Edu (parquet scan, join on file_path) ---

filter_fineweb_edu_10k = ExecutorStep(
    name="filtered/dclm_400m_1x_10k_fineweb_edu",
    description="Filter FineWeb-Edu records matching the 10k WARCs (join on file_path).",
    # Filter driver itself does almost nothing — manifest fast-path loads
    # {snapshot: warc_files} in 0.01s (~1 MB dict), then dispatches to Zephyr
    # workers (which have their own 8 GiB resources). The 4 CPU / 32 GB
    # config from the 3k pipeline was sized for the slow metadata-load that
    # the manifest fast-path replaces.
    fn=remote(filter_fineweb_edu, resources=ResourceConfig(cpu=2, ram="8g")),
    config=FilterFinewebEduConfig(
        metadata_path=METADATA_PATH,
        manifest_path=MANIFEST_PATH,
        fineweb_base_path=FINEWEB_EDU_BASE,
        output_path=this_output_path(),
    ),
)

# --- Step 2: Tokenize ---

tokenize_fineweb_edu_10k = default_tokenize(
    name="dclm_400m_1x_10k_fineweb_edu",
    dataset=filter_fineweb_edu_10k / "*.jsonl.gz",
    tokenizer=TOKENIZER,
)

# --- Entry point ---

if __name__ == "__main__":
    executor_main(
        steps=[tokenize_fineweb_edu_10k],
        description=(
            "10k FineWeb-Edu extraction: filter records matching the full DCLM 400m-1x "
            "pool (10,363 WARCs) by file_path, then tokenize. Last derived dataset before "
            "retiring the raw 10k HTML pool."
        ),
    )
