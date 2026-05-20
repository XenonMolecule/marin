# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 1: HF model load, single region, text payloads.

Smallest end-to-end smoke. If this fails, the whole library is broken.
"""
from __future__ import annotations

from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference


def main() -> None:
    configure_logging()
    print("=== Test 1: HF model load, single region, text payloads ===")

    prompts = [
        {"id": f"t1-{i:03d}", "payload": {"kind": "text", "prompt": "The capital of France is"}} for i in range(8)
    ]
    expected_ids = {p["id"] for p in prompts}

    cfg = InferenceConfig(
        regions=["europe-west4"],
        results_region="europe-west4",
        tpu_shapes=("v6e-4",),
        max_workers_per_region=1,
        shard_size=4,
        job_name="inf-integ-1-hf-model",
        sampling=SamplingParams(temperature=0.0, max_tokens=16),
    )
    model = ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4})

    print(f"Model: {model.model}")
    print(f"Regions: {list(cfg.regions)}, results_region: {cfg.results_region}")
    print(f"{len(prompts)} prompts, shard_size={cfg.shard_size}")

    result = inference(model=model, dataset=prompts, config=cfg)
    records = assert_complete(result, expected_n=len(prompts), expected_ids=expected_ids)

    print(f"\nResults at: {result.results_uri}")
    print_sample(records)
    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
