#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal inference test on eu-west4-a with v6e-8 TPUs.

Proof-of-concept: run the rephraser model on a v6e-8 node in eu-west4-a.
Uses the executor pattern (like Moo Jin's SDG experiments) to properly
handle TPU provisioning and vLLM dependency installation.

Launch (eu-west4-a):
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster eu-west4-a --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/test_v6e_inference.py
"""

import json
import logging
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.inference import TextGenerationInferenceConfig, run_inference

logger = logging.getLogger(__name__)

REPHRASER_MODEL = "gs://marin-eu-west4/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1000"

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
                "id": f"test-v6e-{i:04d}",
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

    logger.info(f"Wrote {config.n_records} test records to {test_file}")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

create_data = ExecutorStep(
    name="test_v6e/input",
    description="Create tiny test HTML documents for v6e inference test.",
    fn=create_test_data,
    config=CreateTestDataConfig(output_path=this_output_path()),
    resources=ResourceConfig.with_cpu(cpu=4, ram="8g"),
    pip_dependency_groups=["cpu"],
)

inference = ExecutorStep(
    name="test_v6e/inference",
    description="Run rephraser inference on v6e-8 TPU (eu-west4-a proof of concept).",
    fn=run_inference,
    config=TextGenerationInferenceConfig(
        input_path=create_data / "*.jsonl.gz",
        output_path=this_output_path(),
        model_name=REPHRASER_MODEL,
        engine_kwargs={
            "tensor_parallel_size": 4,
            "max_model_len": 4096,  # Small context for test
            "enable_prefix_caching": True,
            "load_format": "runai_streamer",  # Stream weights from GCS without FUSE
        },
        generation_kwargs={
            "temperature": 0.0,
            "max_tokens": 512,
        },
        system_message=SYSTEM_MESSAGE,
        template=USER_TEMPLATE,
        prompt_column="html",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=4096 - 512,
        num_instances=(1, 1),  # Single replica for test
        batch_size=256,
        tensor_parallel_size=4,
        resource_config=ResourceConfig.with_tpu("v6e-8"),
        generated_text_column_name="generated_text",
        checkpoint_id_column="id",
        filetype="jsonl.gz",
        output_filetype_override="parquet",
    ),
    pip_dependency_groups=["vllm"],
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(
        steps=[inference],  # Executor auto-discovers create_data as upstream dep
        description="v6e-8 inference proof-of-concept on eu-west4-a.",
    )
