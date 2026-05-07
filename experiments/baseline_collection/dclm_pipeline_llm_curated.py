# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the full DCLM curation pipeline to llm_curated extracted text.

Stages:
  1. dclm_filter   — RefinedWeb heuristics + Gopher repetition + fastText quality
  2. bff_dedup     — DCLM-faithful 13-gram bloom filter dedup (--remove-type old-both)
  3. default_tokenize — llama3 tokenizer (reads ``text``)

Hypothesis: substituting llm_curated extraction for resiliparse extraction at the
front of the DCLM pipeline produces a higher-quality post-curation corpus, on the
same WARC pool that the published DCLM-baseline-1.0 was built from.

Output paths are NEW (do NOT collide with the source ``baseline_llm_curated-050243``).
The source dataset is read-only.

Usage::

    uv run iris --cluster marin job run --no-wait \\
        --cpu 4 --memory 4GB --disk 10GB \\
        --priority interactive \\
        --extra cpu --extra dclm \\
        --enable-extra-resources \\
        --region us-central1 \\
        --job-name dclm-pipeline-llm-curated \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/baseline_collection/dclm_pipeline_llm_curated.py
"""

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.datakit.download.download_url import (
    DownloadUrlToGcsConfig,
    download_url_to_gcs,
)
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    this_output_path,
    versioned,
)
from marin.transform.bff_dedup import BffDedupConfig, bff_dedup
from marin.transform.dclm_filter import DclmFilterConfig, dclm_filter

# ---------------------------------------------------------------------------
# Source: existing llm_curated documents (200 shards, ~800MB compressed each)
# ---------------------------------------------------------------------------

LLM_CURATED_DOCS = "gs://marin-us-central1/documents/baseline_llm_curated-050243"

# ---------------------------------------------------------------------------
# DCLM resources (mirror of dclm_filtered_cooldown.py setup; reused so the
# executor's content-hashing keeps the same model artifacts across runs).
# ---------------------------------------------------------------------------

download_lid_model = ExecutorStep(
    name="resources/dclm/lid_176",
    description="Download FastText language ID model (lid.176.bin).",
    fn=download_url_to_gcs,
    config=DownloadUrlToGcsConfig(
        url=versioned("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"),
        output_path=this_output_path(),
    ),
)

download_quality_model = ExecutorStep(
    name="resources/dclm/fasttext_oh_eli5",
    description="Download DCLM FastText quality classifier.",
    fn=download_url_to_gcs,
    config=DownloadUrlToGcsConfig(
        url=versioned(
            "https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/"
            "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
        ),
        output_path=this_output_path(),
        filename="fasttext_oh_eli5.bin",
    ),
)

# Ban lists are pre-staged on GCS (the curated domain list is 118MB, too large
# for GitHub). Versioning ensures the executor invalidates the filter step if
# the path changes.
BANLISTS_GCS_PATH = "gs://marin-us-central2/resources/dclm/banlists"

# ---------------------------------------------------------------------------
# Stage 1: DCLM filter (Zephyr-distributed; one ExecutorStep, scales by Zephyr)
# ---------------------------------------------------------------------------

filter_step = ExecutorStep(
    name="filtered/dclm_filter_llm_curated",
    description="Apply the full DCLM-Baseline filtering pipeline to llm_curated extraction.",
    fn=dclm_filter,
    config=DclmFilterConfig(
        input_path=f"{LLM_CURATED_DOCS}/*.jsonl.gz",
        output_path=this_output_path(),
        lid_model_path=download_lid_model,
        quality_model_path=download_quality_model,
        banlists_path=versioned(BANLISTS_GCS_PATH),
    ),
)

# ---------------------------------------------------------------------------
# Stage 2: bff dedup (Zephyr-driven; per-group bff invocations)
# ---------------------------------------------------------------------------
# DCLM's bff is single-process; we cannot run one global bloom over the whole
# corpus on iris (the cluster's only CPU pool is n2-highmem-2 = 2 vCPU). So
# we partition the input files into groups and run one bff per group. Each
# group has its own bloom filter — same multi-node strategy DCLM itself uses
# via ``--shard-num``/``--total-shards``. Cross-group near-duplicates are
# missed; within-group near-duplicates are caught.

dedup_step = ExecutorStep(
    name="deduped/bff_llm_curated",
    description="DCLM-faithful 13-gram bloom filter dedup over the dclm_filter output.",
    fn=bff_dedup,
    config=BffDedupConfig(
        input_path=filter_step / "*.jsonl.gz",
        output_path=this_output_path(),
        shards_per_group=versioned(10),
        # 10 input files × ~250M tokens × 30-50% post-filter ≈ 0.7-1.2B tokens
        # per group. 5B is a safe overestimate so the bloom filter is sized
        # large enough across groups of varying density.
        expected_ngram_count_per_group=versioned(5_000_000_000),
        fp_rate=versioned(0.01),
        min_ngram_size=versioned(13),
        max_ngram_size=versioned(13),
        filtering_threshold=versioned(0.8),
        remove_type=versioned("old-both"),
    ),
)

# ---------------------------------------------------------------------------
# Stage 3: tokenize (llama3, reads ``text`` field)
# ---------------------------------------------------------------------------

tokenize_step = default_tokenize(
    name="llm_curated_dclm_curated",
    dataset=dedup_step / "*.jsonl.gz",
    tokenizer=llama3_tokenizer,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(
        steps=[tokenize_step],
        description=(
            "DCLM curation pipeline applied to llm_curated extraction: "
            "dclm_filter → bff dedup (old-both, 13-gram, 0.8) → tokenize (llama3)."
        ),
    )
