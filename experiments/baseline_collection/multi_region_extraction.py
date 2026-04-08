# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-region LLM extraction orchestrator.

Dispatches batches of WARC files to Iris jobs across multiple regions and TPU
types. Iris places each job wherever compute is available — no region pinning.
Each job downloads its assigned WARCs from CommonCrawl (free ingress) and writes
extracted text to the worker's local region bucket.

The orchestrator is **fully re-entrant**: all progress state lives in GCS output
files. Kill and restart at any time — it scans completed hashes and picks up
where it left off. In-flight Iris jobs continue running independently.

Usage::

    # Run from any machine with gcloud + iris auth
    python experiments/baseline_collection/multi_region_extraction.py
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field

import fsspec

from experiments.baseline_collection.download_warcs import (
    _load_manifest,
    _warc_path_hash,
)
from experiments.rephraser.extraction_sft_recipe import (
    DEFAULT_SYSTEM_MESSAGE,
    DEFAULT_USER_TEMPLATE_FMT,
)
from iris.marin_fs import REGION_TO_DATA_BUCKET
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fleet configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TpuFleetEntry:
    """A TPU type to dispatch work to. Iris places it in any region."""

    tpu_type: str
    """TPU variant for Iris ``--tpu`` flag (e.g. ``v5p-8``, ``v6e-8``)."""

    num_workers: int = 16
    """Zephyr workers per job."""

    batch_size: int = 50
    """WARCs per dispatch batch."""

    memory: str = "32GB"
    """Memory per worker."""


# All TPU types to use — Iris places them wherever available.
# Add/remove entries as compute opens up.
FLEET: list[TpuFleetEntry] = [
    TpuFleetEntry(tpu_type="v5p-8", num_workers=16, batch_size=50),
    TpuFleetEntry(tpu_type="v6e-8", num_workers=16, batch_size=50),
    # TpuFleetEntry(tpu_type="v6e-4", num_workers=32, batch_size=50),
]


# ---------------------------------------------------------------------------
# Extraction configuration
# ---------------------------------------------------------------------------

# Model weights pre-copied to each regional bucket.
# Keys are canonical GCP region names (matching iris.marin_fs).
_REPHRASER_CKPT = "checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"

MODEL_BY_REGION: dict[str, str] = {
    "us-central1": f"gs://marin-us-central1/{_REPHRASER_CKPT}",
    "europe-west4": f"gs://marin-eu-west4/{_REPHRASER_CKPT}",
    # Pre-copy to additional regions as needed:
    # "us-central2": f"gs://marin-us-central2/{_REPHRASER_CKPT}",
    # "us-east5": f"gs://marin-us-east5/{_REPHRASER_CKPT}",
}

# Output subdirectory (same across all regions).
OUTPUT_SUBDIR = "documents/baseline_llm_extraction"

# All regional output directories to scan for completed files.
ALL_OUTPUT_DIRS: list[str] = [f"gs://{bucket}/{OUTPUT_SUBDIR}" for bucket in REGION_TO_DATA_BUCKET.values()]

# --- Extraction prompt ---
# The system message is reused from the existing ExtractionSpec pattern.
# Only the extraction spec text needs to be customized per use case.
EXTRACTION_SYSTEM_MESSAGE = DEFAULT_SYSTEM_MESSAGE

# Extraction spec — iterating on this, will likely ship another version soon.
# Reference implementations for domain-specific prompts:
#   experiments/rephraser/mathhelpforum_extraction_sft_v2.py (math)
#   experiments/rephraser/code_extraction_sft_v3_base.py (code)
#   experiments/rephraser/medical_extraction_sft.py (medical)
EXTRACTION_SPEC = (
    "Extract the content from this HTML page as clean text. Follow all rules below.\n\n"
    "1. Extract the full page content in reading order. Keep all explanatory text, "
    "discussion, and comments that add substantive information.\n"
    "2. Remove boilerplate: navigation bars, footers, sidebars, ads, share buttons, "
    "related links, breadcrumbs, cookie banners, and user interface elements.\n"
    "3. Preserve all technical content exactly as written: code, math notation, "
    "formulas, tables, and data.\n"
    "4. Decode HTML entities to their plain characters (e.g. &amp; to &, &lt; to <, "
    "&gt; to >). Remove any raw HTML tags.\n"
    "5. Every sentence in your output must come from the source page. Do not add, "
    "invent, or embellish content.\n"
    "6. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:\n"
    "   - Login, signup, paywall, error page, or empty page\n"
    "   - Index page, directory listing, search results, or navigation-only page\n"
    "   - The page has under ~50 words of substantive content after removing boilerplate"
)

EXTRACTION_TEMPLATE = DEFAULT_USER_TEMPLATE_FMT.format(spec=EXTRACTION_SPEC)

# Master WARC manifest
WARC_MANIFEST = "experiments/distill/baseline_warcs_3000.txt"

# Batch manifests are written here (tiny files, write-once)
MANIFEST_STAGING_BUCKET = "gs://marin-us-central2"
MANIFEST_STAGING_DIR = f"{MANIFEST_STAGING_BUCKET}/tmp/extraction_manifests"


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def _scan_completed_hashes(output_dirs: list[str]) -> set[str]:
    """Scan all regional output directories for completed WARC hashes.

    Returns the set of 12-char hex hashes found in filenames like
    ``data-{hash}.jsonl.gz``.
    """
    completed: set[str] = set()
    for output_dir in output_dirs:
        try:
            files = fsspec_glob(f"{output_dir}/data-*.jsonl.gz")
        except Exception:
            # Bucket may not exist or be inaccessible
            continue
        for path in files:
            basename = os.path.basename(path)
            # data-{12-char-hash}.jsonl.gz
            if basename.startswith("data-") and basename.endswith(".jsonl.gz"):
                h = basename[len("data-") : -len(".jsonl.gz")]
                if len(h) == 12:
                    completed.add(h)
    return completed


# ---------------------------------------------------------------------------
# Batch manifest management
# ---------------------------------------------------------------------------


def _write_batch_manifest(warc_paths: list[str], batch_id: str) -> str:
    """Write a batch manifest to GCS. Returns the GCS path."""
    manifest_path = f"{MANIFEST_STAGING_DIR}/batch_{batch_id}.txt"
    content = "\n".join(warc_paths) + "\n"
    with fsspec.open(manifest_path, "w") as f:
        f.write(content)
    logger.info("Wrote batch manifest: %s (%d WARCs)", manifest_path, len(warc_paths))
    return manifest_path


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


def _build_config_json(
    manifest_path: str,
    num_workers: int,
) -> dict:
    """Build the DownloadAndExtractConfig as a JSON-serializable dict."""
    return {
        "warc_manifest_path": manifest_path,
        "output_subdir": OUTPUT_SUBDIR,
        "model_name_by_region": MODEL_BY_REGION,
        "template": EXTRACTION_TEMPLATE,
        "system_message": EXTRACTION_SYSTEM_MESSAGE,
        "prompt_column": "html",
        "generated_text_column": "generated_text",
        "apply_chat_template": True,
        "max_doc_tokens": 28672,
        "engine_kwargs": {"max_model_len": 32768, "enable_prefix_caching": True},
        "generation_kwargs": {"temperature": 0.0, "max_tokens": 4096},
        "strip_thinking": True,
        "filter_patterns": [r"\[NO_USEFUL_CONTENT\]"],
        "min_output_chars": 50,
        "num_workers": num_workers,
        "max_records_per_generate": 500,
        "http_timeout": 600,
        "max_retries": 5,
    }


def _submit_iris_job(
    entry: TpuFleetEntry,
    manifest_path: str,
    batch_id: str,
) -> str:
    """Submit an Iris job via CLI. Returns the job name for tracking."""
    config_dict = _build_config_json(manifest_path, entry.num_workers)

    # Write config to a temp GCS path
    config_gcs = f"{MANIFEST_STAGING_DIR}/config_{batch_id}.json"
    with fsspec.open(config_gcs, "w") as f:
        json.dump(config_dict, f, indent=2)

    job_name = f"extract-{batch_id}"

    # Submit via iris CLI — no --region flag, Iris places it wherever
    cmd = [
        "iris",
        "job",
        "run",
        "--tpu",
        entry.tpu_type,
        "--memory",
        entry.memory,
        "--no-wait",
        "--job-name",
        job_name,
        "--max-retries",
        "3",
        "--extra",
        "vllm",
        "--",
        "python",
        "-m",
        "experiments.baseline_collection.download_and_extract",
        "--config_path",
        config_gcs,
    ]

    logger.info("Submitting Iris job: %s (tpu=%s, %d WARCs)", job_name, entry.tpu_type, entry.batch_size)
    logger.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        logger.error("Job submission failed: %s", result.stderr[:500])
        raise RuntimeError(f"iris job run failed: {result.stderr[:500]}")

    logger.info("Submitted job %s: %s", job_name, result.stdout.strip()[:200])
    return job_name


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class DispatchState:
    """Tracks active jobs within a dispatch iteration."""

    active_jobs: list[str] = field(default_factory=list)
    jobs_submitted: int = 0


def dispatch_loop(
    manifest_path: str,
    fleet: list[TpuFleetEntry],
    poll_interval_seconds: int = 120,
    max_concurrent_jobs: int = 6,
) -> None:
    """Main orchestrator loop. Runs until all WARCs are extracted.

    Fully re-entrant: scans GCS for completed hashes on each iteration.
    Kill and restart at any time.
    """
    all_warcs = _load_manifest(manifest_path)

    logger.info("Loaded %d WARCs from manifest %s", len(all_warcs), manifest_path)
    logger.info("Fleet: %s", [(e.tpu_type, e.batch_size) for e in fleet])
    logger.info("Scanning %d regional output directories", len(ALL_OUTPUT_DIRS))

    iteration = 0
    state = DispatchState()

    while True:
        iteration += 1
        logger.info("--- Dispatch iteration %d ---", iteration)

        # 1. Scan ALL regional buckets for completed output files
        completed_hashes = _scan_completed_hashes(ALL_OUTPUT_DIRS)

        remaining = [w for w in all_warcs if _warc_path_hash(w) not in completed_hashes]
        logger.info(
            "Progress: %d/%d done (%.1f%%), %d remaining",
            len(all_warcs) - len(remaining),
            len(all_warcs),
            100 * (len(all_warcs) - len(remaining)) / len(all_warcs),
            len(remaining),
        )

        if not remaining:
            logger.info("All %d WARCs extracted! Orchestrator complete.", len(all_warcs))
            break

        # 2. Dispatch batches to fleet TPU types (round-robin)
        offset = 0
        jobs_this_round = 0

        for entry in fleet:
            if offset >= len(remaining):
                break
            if jobs_this_round >= max_concurrent_jobs:
                break

            batch = remaining[offset : offset + entry.batch_size]
            offset += len(batch)

            batch_id = f"{entry.tpu_type}_{int(time.time())}_{state.jobs_submitted}"
            manifest_gcs = _write_batch_manifest(batch, batch_id)

            try:
                job_name = _submit_iris_job(entry, manifest_gcs, batch_id)
                state.active_jobs.append(job_name)
                state.jobs_submitted += 1
                jobs_this_round += 1
            except Exception as e:
                logger.error("Failed to submit job for %s: %s", entry.tpu_type, e)
                continue

        if jobs_this_round == 0 and remaining:
            logger.warning(
                "No jobs submitted this round but %d WARCs remain. " "In-flight jobs may still be working. Waiting...",
                len(remaining),
            )

        # 3. Wait before next scan
        logger.info(
            "Waiting %ds before next scan (%d jobs submitted this round)...",
            poll_interval_seconds,
            jobs_this_round,
        )
        time.sleep(poll_interval_seconds)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dispatch_loop(
        manifest_path=WARC_MANIFEST,
        fleet=FLEET,
        poll_interval_seconds=120,
        max_concurrent_jobs=6,
    )


if __name__ == "__main__":
    main()
