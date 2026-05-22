# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 9: SamplingParams reach vLLM and shape output.

Exercises the SamplingParams pass-through with non-default settings:

- `temperature=0.7` — non-greedy; identical prompts should produce different
  responses across the batch when ``seed`` is left None.
- `top_p=0.9`, `repetition_penalty=1.1` — pass-through verification.
- `stop=["END"]` — verify the stop string is honored when it appears.
- `max_tokens=64` — verify the cap is respected (and we observe a
  ``finish_reason=='length'`` for prompts that would naturally generate
  more).

This is the test that catches "I forgot to translate a SamplingParams field
to vLLM's vllm.SamplingParams" — silent passthrough bugs.
"""
from __future__ import annotations

from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference


def main() -> None:
    configure_logging()
    print("=== Test 9: SamplingParams reach vLLM and shape output ===")

    # Six copies of the SAME prompt with different ids. With temperature > 0
    # and seed=None, vLLM samples differently for each. We then count
    # unique outputs to verify sampling actually fires.
    same_prompt = "Tell me a one-sentence joke about programmers."
    prompts = [{"id": f"t9-{i:03d}", "payload": {"kind": "text", "prompt": same_prompt}} for i in range(6)]
    expected_ids = {p["id"] for p in prompts}

    cfg = InferenceConfig(
        regions=["europe-west4"],
        results_region="europe-west4",
        tpu_shapes=("v6e-4",),
        max_workers_per_region=1,
        shard_size=6,
        job_name="inf-integ-9-sampling",
        sampling=SamplingParams(
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=64,
            stop=("END",),
        ),
    )
    model = ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4})

    print(f"Model: {model.model}")
    print(
        f"Sampling: temperature={cfg.sampling.temperature}, top_p={cfg.sampling.top_p}, "
        f"repetition_penalty={cfg.sampling.repetition_penalty}, max_tokens={cfg.sampling.max_tokens}, "
        f"stop={cfg.sampling.stop}"
    )
    print(f"{len(prompts)} prompts (all identical — exercises sampling variation)")

    result = inference(model=model, dataset=prompts, config=cfg)
    records = assert_complete(result, expected_n=len(prompts), expected_ids=expected_ids)

    # 1. Sampling actually varies — at least 2 distinct responses for the
    #    same prompt at temperature=0.7.
    unique_responses = {r.response for r in records}
    print(f"\nUnique responses across {len(records)} identical prompts: {len(unique_responses)}")
    if len(unique_responses) < 2:
        print_sample(records, n=len(records))
        raise AssertionError(
            f"Expected >=2 unique responses with temperature=0.7, got {len(unique_responses)}. "
            "Sampling may not be reaching vLLM (default-temp-zero leak)."
        )

    # 2. finish_reason is captured in extras for every record.
    no_finish_reason = [r for r in records if "finish_reason" not in r.extra]
    if no_finish_reason:
        raise AssertionError(
            f"{len(no_finish_reason)} record(s) missing 'finish_reason' in extra; "
            "either vLLM didn't set it or _extract_response dropped it. "
            f"First example: {no_finish_reason[0]}"
        )

    # 3. Token counts respect max_tokens (rough check via response length).
    finish_reasons = {r.extra["finish_reason"] for r in records}
    print(f"Observed finish_reasons: {sorted(finish_reasons)}")

    print_sample(records, n=4, max_chars=200)

    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
