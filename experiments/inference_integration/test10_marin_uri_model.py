# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 10: load a model from a `marin://` URI (regional bucket).

The library's `marin://path` URI resolves to `gs://marin-{region}/path` at
worker startup — every worker loads from its own region's replica, no HF
download. This is the production loading pattern for Marin-trained
checkpoints replicated across regions.

To run this test:
1. Set ``MARIN_MODEL_URI`` below to a path you know is replicated to the
   ``us-central1`` bucket. The default targets a Qwen3-8B rephraser
   checkpoint Michael has used; update if it has moved or you want to
   exercise a different one.
2. Optionally edit ``MARIN_MODEL_TP`` to match the checkpoint's expected
   tensor-parallel size (8B with tp=4 fits a v5p-8 slice).
3. If the constant is set to a path that doesn't exist, the test crashes
   with a clear "no such object" error from vLLM; that's the desired
   failure mode (not silently substituting a different model).
"""
from __future__ import annotations

from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference

# Edit this to point at a checkpoint replicated in `gs://marin-us-central1`.
# Resolved per-region at worker startup → `gs://marin-{worker_region}/<suffix>`.
MARIN_MODEL_URI = "marin://checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
MARIN_MODEL_TP = 4


def main() -> None:
    configure_logging()
    print("=== Test 10: Model loaded from marin:// URI (regional bucket) ===")

    prompts = [{"id": f"t10-{i:03d}", "payload": {"kind": "text", "prompt": f"Once upon a time, {i}"}} for i in range(4)]
    expected_ids = {p["id"] for p in prompts}

    cfg = InferenceConfig(
        regions=["europe-west4"],
        results_region="europe-west4",
        tpu_shapes=("v6e-4",),
        max_workers_per_region=1,
        shard_size=4,
        job_name="inf-integ-10-marin-uri",
        sampling=SamplingParams(temperature=0.0, max_tokens=20),
    )
    model = ModelSpec(
        model=MARIN_MODEL_URI,
        engine_kwargs={"tensor_parallel_size": MARIN_MODEL_TP, "max_model_len": 4096},
    )

    # Sanity-check the URI resolution before we submit (catches typos locally).
    resolved = model.resolve_for_region("europe-west4")
    print(f"marin:// URI: {MARIN_MODEL_URI}")
    print(f"Resolves to:  {resolved}")
    assert resolved.startswith(
        "gs://marin-eu-west4/"
    ), f"Expected resolved path under gs://marin-eu-west4/, got {resolved}"

    print(f"\n{len(prompts)} prompts, tp={MARIN_MODEL_TP}")
    result = inference(model=model, dataset=prompts, config=cfg)
    records = assert_complete(result, expected_n=len(prompts), expected_ids=expected_ids)

    print(f"\nResults at: {result.results_uri}")
    print_sample(records, n=len(records), max_chars=300)

    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
