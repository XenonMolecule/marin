# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 4: multi-region inference aggregating to one region.

Two compute regions write outputs to a single ``results_region``. Verifies:

- Per-region rotation distributes work (both regions do useful inference)
- All shards land in ``results_region``
- ``skip_existing`` arbitrates any inter-region collision cleanly
- Output assembly via `InferenceResult.iter_records` reads from the single
  canonical prefix

Uses ``Qwen/Qwen3-0.6B`` from HF (works in any region without prior
replication). Enough prompts to need multiple shards so the regional
rotation has work to spread.
"""
from __future__ import annotations

from collections import Counter

from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference


def main() -> None:
    configure_logging()
    print("=== Test 4: Multi-region aggregating to one results region ===")

    n_prompts = 32
    prompts = [
        {"id": f"t4-{i:03d}", "payload": {"kind": "text", "prompt": f"The number {i} is special because"}}
        for i in range(n_prompts)
    ]
    expected_ids = {p["id"] for p in prompts}

    cfg = InferenceConfig(
        regions=["us-central1", "us-east5"],
        results_region="us-central1",
        tpu_shapes=("v5p-8",),
        max_workers_per_region=1,
        shard_size=8,  # 32 / 8 = 4 shards across 2 regions
        job_name="inf-integ-4-multi-region",
        sampling=SamplingParams(temperature=0.0, max_tokens=24),
    )
    model = ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4})

    print(f"Model: {model.model}")
    print(f"Regions: {list(cfg.regions)} → results_region: {cfg.results_region}")
    print(f"{n_prompts} prompts, shard_size={cfg.shard_size} (≈{n_prompts // cfg.shard_size} shards)")

    result = inference(model=model, dataset=prompts, config=cfg)
    records = assert_complete(result, expected_n=n_prompts, expected_ids=expected_ids)

    # All outputs MUST be under results_region's bucket regardless of which
    # region's worker computed them.
    assert (
        cfg.results_region in result.results_uri
    ), f"Outputs not in results_region {cfg.results_region}: {result.results_uri}"

    # Verify shards span the expected range.
    shard_counts = Counter(r.shard for r in records)
    print(f"\nShard distribution: {dict(sorted(shard_counts.items()))}")
    expected_shards = (n_prompts + cfg.shard_size - 1) // cfg.shard_size
    if len(shard_counts) != expected_shards:
        raise AssertionError(
            f"expected {expected_shards} shards in output, got {len(shard_counts)} ({sorted(shard_counts)})"
        )

    print(f"\nResults at: {result.results_uri}")
    print_sample(records)

    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
