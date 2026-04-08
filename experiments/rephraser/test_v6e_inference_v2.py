#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal smoke test for inference_v2 on eu-west4-a with v6e-8 TPUs.

Creates a tiny synthetic dataset (10 HTML records), runs inference_v2 with
Zephyr/vLLM, and writes results.  Use this to verify that the new pipeline
spins up correctly before committing to a full sweep.

Launch (eu-west4-a):
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster eu-west4-a --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/test_v6e_inference_v2.py

Dry run (verify DAG, no execution):
    python experiments/rephraser/test_v6e_inference_v2.py --dry_run
"""

import json
import logging
from dataclasses import dataclass

import fsspec

from marin.execution.remote import remote
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.inference_v2 import InferenceV2Config, run_inference_v2

logger = logging.getLogger(__name__)

REPHRASER_MODEL = "gs://marin-eu-west4/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"

SYSTEM_MESSAGE = (
    "Your input fields are:\n"
    "1. `html` (str): \n"
    "2. `extraction_spec` (str):\n"
    "Your output fields are:\n"
    "1. `text` (str):\n"
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in.\n\n"
    "[[ ## html ## ]]\n{html}\n\n"
    "[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n"
    "[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: \n"
    "        Extract the main content text from a given HTML document."
)

SPEC = "Extract the main text content from this HTML page."
USER_TEMPLATE = (
    "[[ ## html ## ]]\n{example}\n\n"
    "[[ ## extraction_spec ## ]]\n" + SPEC + "\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)


@dataclass
class CreateTestDataConfig:
    output_path: str
    n_records: int = 10


def create_test_data(config: CreateTestDataConfig):
    """Create a small test JSONL file with HTML snippets."""
    test_file = f"{config.output_path}/test_input.jsonl.gz"
    records = []
    for i in range(config.n_records):
        records.append(
            {
                "id": f"test-v6e-v2-{i:04d}",
                "html": (
                    f"<html><head><title>Article {i}</title></head>"
                    f"<body><h1>Test Article {i}</h1>"
                    f"<p>This is paragraph one of test article {i}. "
                    f"It contains some interesting content about topic {i}.</p>"
                    f"<p>This is the second paragraph with more details.</p>"
                    f"<nav>Home | About | Contact</nav>"
                    f"<footer>Copyright 2026</footer></body></html>"
                ),
            }
        )

    with fsspec.open(test_file, "w", compression="gzip") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info("Wrote %d test records to %s", config.n_records, test_file)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

create_data = ExecutorStep(
    name="test_v6e_v2/input",
    description="Create tiny test HTML documents for inference_v2 smoke test.",
    fn=create_test_data,
    config=CreateTestDataConfig(output_path=this_output_path()),
)

inference = ExecutorStep(
    name="test_v6e_v2/inference",
    description="Run inference_v2 on v6e-8 TPU (eu-west4-a smoke test).",
    fn=remote(run_inference_v2, pip_dependency_groups=["vllm"]),
    config=InferenceV2Config(
        input_path=create_data / "*.jsonl.gz",
        output_path=this_output_path(),
        model_name=REPHRASER_MODEL,
        input_format="jsonl.gz",
        output_format="jsonl.gz",
        engine_kwargs={
            "max_model_len": 4096,  # Small context for test
            "enable_prefix_caching": True,
        },
        generation_kwargs={
            "temperature": 0.0,
            "max_tokens": 512,
        },
        system_message=SYSTEM_MESSAGE,
        template=USER_TEMPLATE,
        prompt_column="html",
        apply_chat_template=True,
        max_doc_tokens=4096 - 512,
        tensor_parallel_size=4,
        tpu_type="v6e-8",
        num_workers=1,  # Single worker for smoke test
        records_per_shard=10,  # All 10 records in one shard
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(
        steps=[inference],  # Executor auto-discovers create_data as upstream dep
        description="inference_v2 smoke test on eu-west4-a v6e-8.",
    )
