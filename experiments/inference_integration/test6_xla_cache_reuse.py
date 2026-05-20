# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 6: XLA compile-cache reuse across runs.

Runs the same (model, engine_kwargs) twice in the same region. The first run
populates ``gs://marin-{region}/tmp/ttl=30d/vllm-cache/{model_hash}/``; the
second run should pull from it. We can't directly measure compile time from
inside our script, but we can:

1. Verify the cache prefix exists with non-trivial size after run 1.
2. Verify run 2 sees the same cache prefix (didn't get nuked).
3. Note wall-clock for inspection.

For a definitive measurement, watch the worker logs in the Iris UI: the
``Engine loaded in X.Xs`` line should drop from ~3-5 min (cold) to ~30-60s
(warm) between the two runs.
"""
from __future__ import annotations

import time

import fsspec
from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import (
    InferenceConfig,
    ModelSpec,
    SamplingParams,
    compile_cache,
    inference,
)


def _list_cache_files(cache_uri: str) -> list[str]:
    fs, fs_path = fsspec.core.url_to_fs(cache_uri)
    if not fs.exists(fs_path):
        return []
    try:
        return [p for p in fs.find(fs_path) if not p.endswith("/")]
    except Exception:
        return []


def main() -> None:
    configure_logging()
    print("=== Test 6: XLA compile-cache reuse across two runs ===")

    prompts = [{"id": f"t6-{i:03d}", "payload": {"kind": "text", "prompt": f"Sentence {i}:"}} for i in range(4)]
    expected_ids = {p["id"] for p in prompts}

    cfg = InferenceConfig(
        regions=["us-central1"],
        results_region="us-central1",
        tpu_shapes=("v5p-8",),
        max_workers_per_region=1,
        shard_size=4,
        job_name="inf-integ-6-xla-cache",
        sampling=SamplingParams(temperature=0.0, max_tokens=12),
    )
    model = ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4})

    cache_uri = compile_cache.resolve_cache_uri(model, "us-central1", template=None)
    print(f"Expected compile cache prefix: {cache_uri}")
    files_before = _list_cache_files(cache_uri)
    print(f"Cache files before run 1: {len(files_before)}")

    # --- Run 1: populate the cache ---
    print("\n--- Run 1 (cold; populates cache) ---")
    t0 = time.monotonic()
    result1 = inference(model=model, dataset=prompts, config=cfg)
    elapsed1 = time.monotonic() - t0
    print(f"Run 1 wall clock: {elapsed1:.1f}s")
    assert_complete(result1, expected_n=len(prompts), expected_ids=expected_ids)
    cleanup_run(result1)

    files_after_run1 = _list_cache_files(cache_uri)
    print(f"Cache files after run 1: {len(files_after_run1)}")
    if len(files_after_run1) == 0:
        raise AssertionError(
            f"Expected compile-cache files at {cache_uri} after run 1, found none. "
            "Either compile_cache.configure_env didn't fire, JAX/vLLM ignored the env, "
            "or the bucket lifecycle nuked them mid-run."
        )

    # --- Run 2: should reuse the cache ---
    print("\n--- Run 2 (warm; should reuse cache) ---")
    t0 = time.monotonic()
    result2 = inference(model=model, dataset=prompts, config=cfg)
    elapsed2 = time.monotonic() - t0
    print(f"Run 2 wall clock: {elapsed2:.1f}s")
    records2 = assert_complete(result2, expected_n=len(prompts), expected_ids=expected_ids)
    print_sample(records2, n=2)
    cleanup_run(result2)

    files_after_run2 = _list_cache_files(cache_uri)
    print(f"Cache files after run 2: {len(files_after_run2)}")
    print(f"\nWall clock: run1={elapsed1:.1f}s  run2={elapsed2:.1f}s  ratio={elapsed1 / max(elapsed2, 1):.2f}x")
    if elapsed2 >= elapsed1:
        print(
            "\nWARNING: run 2 was not faster than run 1. Compile cache may not be wired through; "
            "check worker logs for 'Engine loaded in X.Xs' on each run."
        )

    print(f"\nResults of run 2 at: {result2.results_uri}")

    # NOTE: We deliberately do NOT delete the compile-cache prefix —
    # subsequent runs of any test using the same model should benefit from
    # it, and it auto-expires in 30 days.
    print(f"\nNOTE: Compile cache retained at {cache_uri} (TTL=30d)")
    exit_pass()


if __name__ == "__main__":
    main()
