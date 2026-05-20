# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 5: multi-shard run in a single region.

Verifies the worker actually processes multiple shards in sequence, with
the vLLM engine reused across them (no per-shard reload). If the engine is
NOT reused, we'd see a 3-5 min XLA compile delay per shard; with reuse, only
the first shard pays it. We can't measure that directly from outputs alone,
but we can verify all shards completed within a reasonable budget by
inspecting per-shard timestamps after the run (the Iris UI is the better
place for that — we just confirm completeness here).
"""
from __future__ import annotations

from collections import Counter

from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference


def main() -> None:
    configure_logging()
    print("=== Test 5: Multi-shard run in a single region ===")

    n_prompts = 60
    prompts = [
        {"id": f"t5-{i:03d}", "payload": {"kind": "text", "prompt": f"Question {i}: What is the meaning of"}}
        for i in range(n_prompts)
    ]
    expected_ids = {p["id"] for p in prompts}

    cfg = InferenceConfig(
        regions=["us-central1"],
        results_region="us-central1",
        tpu_shapes=("v5p-8",),
        max_workers_per_region=2,
        shard_size=10,  # 60 / 10 = 6 shards per worker pool
        job_name="inf-integ-5-many-shards",
        sampling=SamplingParams(temperature=0.0, max_tokens=20),
    )
    model = ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4})

    print(f"Model: {model.model}")
    print(f"{n_prompts} prompts, shard_size={cfg.shard_size}, max_workers={cfg.max_workers_per_region}")
    expected_shards = (n_prompts + cfg.shard_size - 1) // cfg.shard_size
    print(f"Expecting {expected_shards} shards across {cfg.max_workers_per_region} workers")

    result = inference(model=model, dataset=prompts, config=cfg)
    records = assert_complete(result, expected_n=n_prompts, expected_ids=expected_ids)

    shard_counts = Counter(r.shard for r in records)
    print(f"\nShard distribution: {dict(sorted(shard_counts.items()))}")
    if len(shard_counts) != expected_shards:
        raise AssertionError(f"expected {expected_shards} shards, got {len(shard_counts)}")

    # Every shard should have exactly shard_size records (last shard may be short).
    full_shards = [s for s, c in shard_counts.items() if c == cfg.shard_size]
    if len(full_shards) < expected_shards - 1:
        raise AssertionError(f"expected at least {expected_shards - 1} full shards, got {len(full_shards)}")

    print(f"\nResults at: {result.results_uri}")
    print_sample(records)

    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
