# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 11: tpu_shapes multi-shape fallback.

``InferenceConfig.tpu_shapes`` is a sequence — the library allows passing
multiple TPU variants so the scheduler can land on whichever has capacity.
The constraint (enforced by ``fray.ResourceConfig.with_tpu``) is that all
listed variants must share the same ``vm_count`` AND ``chips_per_vm``. The
intent is to widen scheduling flexibility without changing the model's
tensor-parallel layout.

A few worked examples:

* ``("v6e-4", "v5litepod-4")`` — both 1 VM × 4 chips. Valid alternatives.
* ``("v6e-4", "v6e-8")`` — same chip family but 4-chip vs 8-chip. **Not**
  alternatives: ``chips_per_vm`` differs, so tensor-parallel layouts would
  need to differ too.
* ``("v6e-8", "v6e-4")`` swapped — same problem.

Tests 1-10 each pin a single tpu_shape; this test fills the gap by
exercising the multi-shape path against a real cluster.

What "PASS" means here:

1. ``InferenceConfig`` accepted the multi-shape tuple at init (no value-
   error from incompatible variants).
2. The scheduler landed the worker on one of the listed variants.
3. The full inference path completed (the test does NOT pin which variant
   was chosen; the point is that the library tolerates ``either``).
"""
from __future__ import annotations

from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference


def main() -> None:
    configure_logging()
    print("=== Test 11: tpu_shapes multi-shape fallback ===")

    prompts = [
        {"id": f"t11-{i:03d}", "payload": {"kind": "text", "prompt": "The capital of France is"}} for i in range(4)
    ]
    expected_ids = {p["id"] for p in prompts}

    alternatives = ("v6e-4", "v5litepod-4")
    cfg = InferenceConfig(
        regions=["europe-west4"],
        results_region="europe-west4",
        tpu_shapes=alternatives,
        max_workers_per_region=1,
        shard_size=4,
        job_name="inf-integ-11-shape-alternatives",
        sampling=SamplingParams(temperature=0.0, max_tokens=16),
    )
    model = ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4})

    print(f"Model: {model.model}")
    print(f"Region: {list(cfg.regions)}, results_region: {cfg.results_region}")
    print(f"tpu_shapes alternatives: {alternatives}")
    print(f"{len(prompts)} prompts, shard_size={cfg.shard_size}")

    result = inference(model=model, dataset=prompts, config=cfg)
    records = assert_complete(result, expected_n=len(prompts), expected_ids=expected_ids)

    print(f"\nResults at: {result.results_uri}")
    print_sample(records)

    # The selected variant is recorded in the worker job's resource constraints;
    # we don't pin it here — passing on either alternative is the desired
    # behavior. Inspect the Iris UI for the worker job's actual device-variant
    # if you want to know which one the scheduler picked this time.
    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
