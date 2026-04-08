#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal inference test to verify checkpoint recovery works.

Runs a tiny inference job (5 records), then re-runs to verify checkpointing
filters out already-completed records. Uses a small model (Qwen3-0.6B) on
a single TPU to keep the test fast.

Usage (us-central2-staging, v4 TPUs):
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central2-staging --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/test_checkpoint_recovery.py
"""

import json
import logging
import os

import fsspec
import ray
from fray.cluster import ResourceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_PATH = "gs://marin-us-central1/tmp/test_checkpoint_recovery_v2"
MODEL = "Qwen/Qwen3-0.6B"
CHECKPOINT_ID_COLUMN = "id"


def create_test_data(output_dir: str, n_records: int = 5) -> str:
    """Create a small test JSONL file with HTML snippets."""
    test_file = os.path.join(output_dir, "test_input.json")
    records = []
    for i in range(n_records):
        records.append(
            {
                "id": f"test-doc-{i:04d}",
                "html": f"<html><body><h1>Test Document {i}</h1><p>This is test content number {i}.</p></body></html>",
            }
        )

    with fsspec.open(test_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Wrote {n_records} test records to {test_file}")
    return test_file


def list_output_files(path: str, extension: str) -> list[str]:
    """List files matching a given extension in the output directory."""
    fs, fspath = fsspec.core.url_to_fs(path)
    pattern = os.path.join(fspath, f"**/*.{extension}")
    matched = fs.glob(pattern)
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    return [f"{protocol}://{p}" for p in matched]


def count_records(files: list[str]) -> int:
    """Count total records across files."""
    total = 0
    for f in files:
        with fsspec.open(f, "r") as fh:
            for line in fh:
                if line.strip():
                    total += 1
    return total


def main():
    ray.init(address="auto")

    input_dir = os.path.join(OUTPUT_PATH, "input")
    output_dir = os.path.join(OUTPUT_PATH, "output")

    # Create test data
    create_test_data(input_dir)

    # Check what's already in the output directory
    logger.info("=" * 60)
    logger.info("PRE-RUN: Checking output directory state")
    for ext in ["json", "jsonl", "jsonl.gz", "parquet"]:
        files = list_output_files(output_dir, ext)
        if files:
            n_records = count_records(files) if ext in ("json", "jsonl") else "?"
            logger.info(f"  *.{ext}: {len(files)} files, {n_records} records")
        else:
            logger.info(f"  *.{ext}: 0 files")

    # Import inference after ray.init
    from marin.generation.inference import TextGenerationInferenceConfig, run_inference

    # v4-8 has 4 chips -> tensor_parallel_size=4
    config = TextGenerationInferenceConfig(
        input_path=os.path.join(input_dir, "*.json"),
        output_path=output_dir,
        model_name=MODEL,
        engine_kwargs={
            "tensor_parallel_size": 4,
            "max_model_len": 4096,
        },
        generation_kwargs={
            "temperature": 0.0,
            "max_tokens": 128,
        },
        template="{example}",
        prompt_column="html",
        apply_chat_template=True,
        save_templated_prompt=False,
        num_instances=(1, 1),
        batch_size=5,
        tensor_parallel_size=4,
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        generated_text_column_name="generated_text",
        checkpoint_id_column=CHECKPOINT_ID_COLUMN,
        filetype="jsonl.gz",  # <-- THIS IS THE BUG: scanner looks for .jsonl.gz but write_json makes .json
    )

    # Run inference
    logger.info("=" * 60)
    logger.info("RUNNING INFERENCE (filetype='jsonl.gz' — the buggy config)")
    ray.get(run_inference.remote(config))

    # Check output
    logger.info("=" * 60)
    logger.info("POST-RUN: Checking output directory state")
    for ext in ["json", "jsonl", "jsonl.gz"]:
        files = list_output_files(output_dir, ext)
        if files:
            n_records = count_records(files) if ext in ("json", "jsonl") else "?"
            logger.info(f"  *.{ext}: {len(files)} files, {n_records} records")
        else:
            logger.info(f"  *.{ext}: 0 files")

    json_files = list_output_files(output_dir, "json")
    jsonl_gz_files = list_output_files(output_dir, "jsonl.gz")

    logger.info("=" * 60)
    if len(json_files) > 0 and len(jsonl_gz_files) == 0:
        logger.info("BUG CONFIRMED:")
        logger.info(f"  write_json() created {len(json_files)} .json files")
        logger.info("  but checkpoint scanner looks for *.jsonl.gz (finds 0)")
        logger.info("  A second run would reprocess everything from scratch!")
        logger.info("")
        logger.info("FIX: Change filetype='jsonl.gz' to filetype='json'")
        logger.info("  OR set output_filetype_override='json'")
    elif len(jsonl_gz_files) > 0:
        logger.info("Checkpoint files found with .jsonl.gz extension.")
        logger.info("The bug hypothesis was WRONG — investigate further.")
    else:
        logger.info("No output files found. Something else went wrong.")

    # Clean up
    logger.info("=" * 60)
    logger.info(f"Output left in {output_dir} for manual inspection.")
    logger.info(f"Clean up with: gcloud storage rm -r {OUTPUT_PATH}/")


if __name__ == "__main__":
    main()
