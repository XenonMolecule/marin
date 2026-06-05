# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone DCLM curation pipeline for llm_curated.

Bypasses the marin executor's region inference (which forces the parent to
match the data's region) so that we can run the parent in any iris CPU pool
that has capacity. Workers stay pinned to us-central1 via ResourceConfig.

Pipeline stages:
  1. dclm_filter   — RefinedWeb heuristics + Gopher repetition + fastText quality
  2. bff_dedup     — DCLM 13-gram bloom filter dedup, per-group via Zephyr
  3. tokenize      — llama3 tokenizer (reads ``text``)

Pre-stage one-time resources via ``--stage prep`` before running ``--stage all``.

Usage::

    # one-time prep (downloads LID + quality models to GCS)
    iris job run --cpu 0.5 --memory 1GB --priority interactive \\
      --extra cpu --extra dclm \\
      -- python experiments/baseline_collection/run_dclm_pipeline_standalone.py --stage prep

    # full pipeline
    iris job run --cpu 0.5 --memory 1GB --priority interactive \\
      --extra cpu --extra dclm \\
      -- python experiments/baseline_collection/run_dclm_pipeline_standalone.py --stage all

Output paths are FIXED (no executor hash). Re-runs are idempotent: each stage
short-circuits if outputs already exist.
"""

from __future__ import annotations

import argparse
import json
import logging

import fsspec
from fray import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.transform.bff_dedup import BffDedupConfig, bff_dedup
from marin.transform.dclm_filter import DclmFilterConfig, dclm_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("dclm_pipeline_standalone")

# ---------------------------------------------------------------------------
# Fixed paths — change `_VERSION` to bust the output paths if you need a
# fresh run. Existing outputs are preserved (each stage skip-existing'd).
# ---------------------------------------------------------------------------

# Method name for the clean full-corpus run is ``llm_curated_dclm_filtered``
# (renamed from the broken ``llm_curated_dclm_curated`` v2 attempt that ran on
# only 30 of 200 shards). Bump ``_VERSION`` if you ever need to bust outputs.
_VERSION = "v1"
_METHOD = "llm_curated_dclm_filtered"

LLM_CURATED_DOCS = "gs://marin-us-central1/documents/baseline_llm_curated-050243"

# Reuse existing executor-deposited resources (saves ~3 GB redownload).
LID_DIR = "gs://marin-us-central1/resources/dclm/lid_176-951189"
QUALITY_DIR = "gs://marin-us-central1/resources/dclm/fasttext_oh_eli5-d828e9"
BANLISTS_DIR = "gs://marin-us-central2/resources/dclm/banlists"  # one-time ~120MB cross-region read

FILTER_OUT = f"gs://marin-us-central1/filtered/dclm_filter_{_METHOD}_{_VERSION}"
DEDUP_OUT = f"gs://marin-us-central1/deduped/bff_{_METHOD}_{_VERSION}"
TOKEN_OUT = f"gs://marin-us-central1/tokenized/{_METHOD}_{_VERSION}"

LLAMA3 = "meta-llama/Meta-Llama-3.1-8B"

# Workers can land in any US region; same-continent egress is ~$0.01/GB so
# the cross-region read cost on ~160GB compressed input is ~$1-2 — well within
# the data-transfer budget. Avoids being blocked on us-central1 capacity.
WORKER_REGIONS = [
    "us-central1",
    "us-central2",
    "us-east1",
    "us-east5",
    "us-west1",
    "us-west4",
]


# ---------------------------------------------------------------------------
# Resource pre-staging
# ---------------------------------------------------------------------------


def _file_exists_gcs(uri: str) -> bool:
    fs = fsspec.filesystem("gcs")
    try:
        return fs.exists(uri)
    except Exception:
        return False


def _download_to_gcs(url: str, gcs_dst: str) -> None:
    """Download a single file from URL to GCS, only if not already there."""
    if _file_exists_gcs(gcs_dst):
        logger.info("Already present, skipping: %s", gcs_dst)
        return
    import urllib.request

    logger.info("Downloading %s -> %s", url, gcs_dst)
    with urllib.request.urlopen(url) as src, fsspec.open(gcs_dst, "wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    logger.info("Done: %s", gcs_dst)


def stage_prep() -> None:
    """One-time: download lid.176.bin and fasttext_oh_eli5.bin to GCS."""
    _download_to_gcs(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        f"{LID_DIR}/lid.176.bin",
    )
    _download_to_gcs(
        "https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
        f"{QUALITY_DIR}/fasttext_oh_eli5.bin",
    )
    logger.info("Resource prep complete.")


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _has_outputs(gcs_dir: str) -> bool:
    """Return True if ``gcs_dir`` has at least one ``.jsonl.gz`` file."""
    fs = fsspec.filesystem("gcs")
    try:
        files = fs.ls(gcs_dir, detail=False)
    except Exception:
        return False
    return any(f.endswith(".jsonl.gz") for f in files)


def _has_tokenized(gcs_dir: str) -> bool:
    """Return True if a Levanter cache exists at ``gcs_dir`` (train/.stats.json)."""
    fs = fsspec.filesystem("gcs")
    return _file_exists_gcs(f"{gcs_dir}/train/.stats.json")


def stage_filter() -> None:
    if _has_outputs(FILTER_OUT):
        logger.info("Filter output already exists at %s; skipping.", FILTER_OUT)
        return
    # Patch worker resources via monkey-patching the ZephyrContext default.
    # dclm_filter creates its own ZephyrContext; we wrap it to inject regions.
    _run_filter_with_regions()


def _run_filter_with_regions() -> None:
    """Run dclm_filter, but swap ZephyrContext to one that pins workers to us-central1."""
    import marin.transform.dclm_filter as df

    original_ctx_cls = df.ZephyrContext

    def make_ctx(name: str, **kwargs):
        # Force regions onto whatever resources are passed in.
        kwargs.setdefault("max_workers", 50)
        kwargs.setdefault(
            "resources",
            ResourceConfig(cpu=2, ram="12g", regions=WORKER_REGIONS, preemptible=False),
        )
        # Override regions on user-supplied resources too.
        if isinstance(kwargs.get("resources"), ResourceConfig):
            import dataclasses

            kwargs["resources"] = dataclasses.replace(kwargs["resources"], regions=WORKER_REGIONS)
        return original_ctx_cls(name=name, **kwargs)

    df.ZephyrContext = make_ctx  # type: ignore[assignment]
    try:
        # Full 200-shard corpus = the 3000-WARC pool ≈ 56-60B raw tokens.
        # Do NOT subset — the comparison baselines (dclm, resiliparse,
        # nemotron_full_bos_fixed, llm_curated_bos_fixed, llm_curated_dedup)
        # are all built from the same full pool, so any subsetting invalidates
        # the FM scaling-law fit.
        dclm_filter(
            DclmFilterConfig(
                input_path=f"{LLM_CURATED_DOCS}/*.jsonl.gz",
                output_path=FILTER_OUT,
                lid_model_path=LID_DIR,
                quality_model_path=QUALITY_DIR,
                banlists_path=BANLISTS_DIR,
            )
        )
    finally:
        df.ZephyrContext = original_ctx_cls  # type: ignore[assignment]


def stage_dedup() -> None:
    if _has_outputs(DEDUP_OUT):
        logger.info("Dedup output already exists at %s; skipping.", DEDUP_OUT)
        return
    # bff_dedup also opens a ZephyrContext internally; same pin.
    import marin.transform.bff_dedup as bd

    original_ctx_cls = bd.ZephyrContext

    def make_ctx(name: str, **kwargs):
        kwargs.setdefault("max_workers", 100)
        kwargs.setdefault(
            "resources",
            ResourceConfig(cpu=2, ram="14g", disk="50g", regions=WORKER_REGIONS),
        )
        if isinstance(kwargs.get("resources"), ResourceConfig):
            import dataclasses

            kwargs["resources"] = dataclasses.replace(kwargs["resources"], regions=WORKER_REGIONS)
        return original_ctx_cls(name=name, **kwargs)

    bd.ZephyrContext = make_ctx  # type: ignore[assignment]
    try:
        bff_dedup(
            BffDedupConfig(
                input_path=f"{FILTER_OUT}/*.jsonl.gz",
                output_path=DEDUP_OUT,
                max_workers=50,
            )
        )
    finally:
        bd.ZephyrContext = original_ctx_cls  # type: ignore[assignment]


def stage_tokenize() -> None:
    if _has_tokenized(TOKEN_OUT):
        logger.info("Tokenized cache already exists at %s; skipping.", TOKEN_OUT)
        return
    # Tokenize uses Levanter's pipeline which talks to its own region scheduling.
    tokenize(
        TokenizeConfig(
            train_paths=[f"{DEDUP_OUT}/*.jsonl.gz"],
            validation_paths=[],
            cache_path=TOKEN_OUT,
            tokenizer=LLAMA3,
            format=TextLmDatasetFormat(),
        )
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prep", "filter", "dedup", "tokenize", "all"],
        default="all",
        help="Which stage(s) to run.",
    )
    args = parser.parse_args()

    if args.stage == "prep":
        stage_prep()
        return
    if args.stage in ("filter", "all"):
        stage_filter()
    if args.stage in ("dedup", "all"):
        stage_dedup()
    if args.stage in ("tokenize", "all"):
        stage_tokenize()

    # Print final summary
    summary = {
        "filter_out": FILTER_OUT,
        "dedup_out": DEDUP_OUT,
        "token_out": TOKEN_OUT,
        "filter_done": _has_outputs(FILTER_OUT),
        "dedup_done": _has_outputs(DEDUP_OUT),
        "tokenize_done": _has_tokenized(TOKEN_OUT),
    }
    logger.info("PIPELINE SUMMARY: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
