# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 3: raw-text completion payload → engine.generate() dispatch.

Same model as test 2, but plain completion mode. Verifies the text path.
"""
from __future__ import annotations

from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference


def main() -> None:
    configure_logging()
    print("=== Test 3: Completion (text) payload ===")

    text_prompts = [
        "Roses are red, violets are",
        "The Pythagorean theorem states that",
        "In the beginning, the universe was",
        "A common Python list comprehension looks like",
        "The first three prime numbers are",
        "JavaScript's most surprising feature is",
    ]
    prompts = [{"id": f"t3-{i:03d}", "payload": {"kind": "text", "prompt": p}} for i, p in enumerate(text_prompts)]
    expected_ids = {p["id"] for p in prompts}

    cfg = InferenceConfig(
        regions=["us-central1"],
        results_region="us-central1",
        tpu_shapes=("v5p-8",),
        max_workers_per_region=1,
        shard_size=4,
        job_name="inf-integ-3-completion",
        sampling=SamplingParams(temperature=0.0, max_tokens=20),
    )
    model = ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4})

    print(f"Model: {model.model}")
    print(f"{len(prompts)} text completions")

    result = inference(model=model, dataset=prompts, config=cfg)
    records = assert_complete(result, expected_n=len(prompts), expected_ids=expected_ids)

    print(f"\nResults at: {result.results_uri}")
    print_sample(records, n=len(records))

    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
