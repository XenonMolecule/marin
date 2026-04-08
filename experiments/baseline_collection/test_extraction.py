# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick test: submit a 2-WARC extraction job to verify the pipeline end-to-end."""

import json
import logging
import subprocess
import sys

import fsspec

from experiments.baseline_collection.download_warcs import _load_manifest
from experiments.baseline_collection.multi_region_extraction import (
    EXTRACTION_SYSTEM_MESSAGE,
    EXTRACTION_TEMPLATE,
    MODEL_BY_REGION,
    OUTPUT_SUBDIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Take first 2 WARCs for a quick test
MANIFEST_PATH = "experiments/distill/baseline_warcs_3000.txt"
NUM_TEST_WARCS = 2
TEST_TPU_TYPE = "v5p-8"
TEST_NUM_WORKERS = 1

STAGING_DIR = "gs://marin-us-central2/tmp/extraction_test"


def main() -> None:
    all_warcs = _load_manifest(MANIFEST_PATH)
    test_warcs = all_warcs[:NUM_TEST_WARCS]
    logger.info("Test WARCs: %s", test_warcs)

    # Write test manifest to GCS
    manifest_path = f"{STAGING_DIR}/test_manifest.txt"
    with fsspec.open(manifest_path, "w") as f:
        f.write("\n".join(test_warcs) + "\n")
    logger.info("Wrote test manifest: %s", manifest_path)

    # Build config
    config = {
        "warc_manifest_path": manifest_path,
        "output_subdir": f"{OUTPUT_SUBDIR}_test",
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
        "num_workers": TEST_NUM_WORKERS,
        "max_records_per_generate": 500,
        "http_timeout": 600,
        "max_retries": 5,
    }

    config_path = f"{STAGING_DIR}/test_config.json"
    with fsspec.open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Wrote test config: %s", config_path)

    # Submit via iris
    cmd = [
        "iris",
        "job",
        "run",
        "--tpu",
        TEST_TPU_TYPE,
        "--memory",
        "32GB",
        "--no-wait",
        "--job-name",
        "extract-test-2warc",
        "--max-retries",
        "1",
        "--extra",
        "vllm",
        "--",
        "python",
        "-m",
        "experiments.baseline_collection.download_and_extract",
        "--config_path",
        config_path,
    ]

    logger.info("Submitting: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        logger.error("Submission failed with rc=%d", result.returncode)
        sys.exit(1)

    logger.info("Job submitted! Monitor via: iris job list")
    logger.info("Check output at: gs://marin-*/documents/baseline_llm_extraction_test/")


if __name__ == "__main__":
    main()
