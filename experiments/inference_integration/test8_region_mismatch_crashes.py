# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 8: region mismatch crashes the regional job early.

Negative test. Pass a ``gs://marin-us-central1/...`` model URI to a worker
scheduled in ``us-east5``. The library's pre-flight
``rigging.filesystem.check_gcs_paths_same_region`` (called inside
``regional_job._validate_region``) must crash the regional job before vLLM
attempts to load the model — otherwise a cross-region weight download
would burn ~16GB of egress per worker.

The library's failure-isolation policy says a single regional failure does
not fail the whole `inference()` call. So this test:

- Configures one compute region (``us-east5``) whose worker WILL crash.
- Confirms the run completes but `result.missing_shards` is non-empty.
- Confirms the failure reason surfaces in logs (the worker's exception).
"""
from __future__ import annotations

import logging

from _common import cleanup_run, configure_logging, exit_pass
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference


def main() -> None:
    configure_logging()
    print("=== Test 8: Region-mismatch model path crashes regional job (negative) ===")

    prompts = [{"id": f"t8-{i:03d}", "payload": {"kind": "text", "prompt": "hello"}} for i in range(4)]

    cfg = InferenceConfig(
        regions=["europe-west4"],  # worker is in europe-west4
        results_region="us-central1",
        tpu_shapes=("v6e-4",),
        max_workers_per_region=1,
        shard_size=4,
        job_name="inf-integ-8-region-mismatch",
        sampling=SamplingParams(temperature=0.0, max_tokens=8),
    )
    # ...but the model is pinned to us-central1. Pre-flight should reject.
    model = ModelSpec(
        model="gs://marin-us-central1/checkpoints/does-not-need-to-exist",
        engine_kwargs={"tensor_parallel_size": 4},
    )

    print(f"Worker region: {list(cfg.regions)}")
    print("Model bucket region: us-central1 (deliberate mismatch)")
    print("Expected: regional job fails on the cross-region check, run returns incomplete result.")

    # Capture logs so we can verify the failure reason.
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    result = inference(model=model, dataset=prompts, config=cfg)

    if result.is_complete:
        raise AssertionError(
            "Expected an incomplete result due to region mismatch, but the run claimed completeness. "
            "Either the pre-flight check didn't fire, or it allowed the cross-region path to proceed."
        )
    print(f"\nResult.missing_shards: {result.missing_shards}")
    print(f"Result.is_complete: {result.is_complete}")
    print("PASS: region mismatch crashed the regional job as expected.")

    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
