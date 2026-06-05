# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DCLM-baseline filter for the full 10k DCLM 400m-1x WARC pool.

Companion to ``pipeline_10k.py`` (which does Nemotron). Reuses the WARC
metadata that ``pipeline_10k.py`` already produced — no re-download, no
re-extract. Just runs:

    filter_dclm  →  consolidate_dclm  →  default_tokenize

against the 10,364 WARCs' record-ID set, joined into DCLM-baseline-1.0
records via ``metadata.WARC-Record-ID``.

Memory note: the record-ID set for 10,364 WARCs is ~250M UUIDs ≈ 30 GB
as a Python set. The driver and each Zephyr worker need to hold it.
``filter_dclm`` itself sets per-worker resources (48 GiB) for the
inner ZephyrContext; this pipeline file sets the *outer* ``remote()``
resources for ``filter_dclm``'s driver function.

Usage::

    uv run iris --cluster marin job run --cpu 2 --memory 2GB \\
        --region us-central2 --no-wait \\
        --job-name baseline-dclm-10k \\
        -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python experiments/baseline_collection/pipeline_10k_dclm.py
"""

from fray import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.transform.extract_text_from_html import ExtractTextConfig
from zephyr import Dataset, ZephyrContext

from experiments.baseline_collection.filter_dclm import FilterDclmConfig, filter_dclm
from experiments.defaults import default_tokenize

# --- Paths ---

# Reuse the metadata extracted by pipeline_10k.py (already complete on GCS).
# This is a hardcoded path because we deliberately want to share the existing
# output rather than re-extract.
METADATA_PATH = "gs://marin-us-central2/metadata/dclm_400m_1x_10k_warc_metadata-79158f/*.jsonl.gz"

DCLM_BASE = (
    "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/" "mlfoundations/dclm-baseline-1.0/resolve/a3b142c"
)

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# --- Step 1: Filter DCLM-baseline (full scan, join on WARC-Record-ID) ---

filter_dclm_10k = ExecutorStep(
    name="filtered/dclm_400m_1x_10k_dclm",
    description="Filter DCLM-baseline records matching the 10k WARCs (join on WARC-Record-ID).",
    # Driver loads ~400M UUID strings (~80-100 GB Python set), then ctx.put
    # serializes via cloudpickle (peak roughly 2x). 256 GB driver gives
    # comfortable headroom over the worst-case ~200 GB peak. The 64 GB driver
    # in the v1 attempt OOM'd; same string-set codepath as the 3000 pipeline.
    fn=remote(filter_dclm, resources=ResourceConfig(cpu=4, ram="256g")),
    config=FilterDclmConfig(
        metadata_path=METADATA_PATH,
        dclm_base_path=DCLM_BASE,
        output_path=this_output_path(),
    ),
)

# --- Step 2: Consolidate sparse output (90% of 27K shards are near-empty) ---


def consolidate_dclm(config: ExtractTextConfig) -> None:
    """Re-shard DCLM filter output into 100 dense files for the tokenizer."""
    pipeline = (
        Dataset.from_files(config.input_path)
        .load_file()
        .reshard(100)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-00100.jsonl.gz")
    )
    ctx = ZephyrContext(
        name="consolidate-dclm-10k",
        max_workers=100,
        resources=ResourceConfig(cpu=1, ram="8g"),
    )
    ctx.execute(pipeline)


consolidate_dclm_10k = ExecutorStep(
    name="filtered/dclm_400m_1x_10k_dclm_resharded",
    description="Consolidate sparse 10k-DCLM filter output (27K shards, 90% empty) into 100 dense shards.",
    fn=remote(consolidate_dclm, resources=ResourceConfig(cpu=2, ram="16g")),
    config=ExtractTextConfig(
        input_path=filter_dclm_10k / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

# --- Step 3: Tokenize ---

tokenize_dclm_10k = default_tokenize(
    name="dclm_400m_1x_10k_dclm",
    dataset=consolidate_dclm_10k / "*.jsonl.gz",
    tokenizer=TOKENIZER,
)

# --- Entry point ---

if __name__ == "__main__":
    executor_main(
        steps=[tokenize_dclm_10k],
        description=(
            "10k DCLM-baseline extraction: filter records matching the full DCLM 400m-1x "
            "pool (10,363 WARCs) by WARC-Record-ID, consolidate, and tokenize."
        ),
    )
