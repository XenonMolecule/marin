# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 7: reasoning markers preserved end-to-end.

Many modern models emit ``<think>...</think>`` blocks before their final
answer. The library promises to pass model output through **verbatim** —
no stripping of reasoning markers, special tokens, or any other
model-emitted text. Downstream callers post-process as they see fit.

This test prompts a Qwen3 thinking variant and verifies the output
contains ``<think>`` markers in the JSONL. If a future refactor accidentally
strips them, this test fails.
"""
from __future__ import annotations

from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference


def main() -> None:
    configure_logging()
    print("=== Test 7: Reasoning markers preserved (no stripping) ===")

    prompts = [
        {
            "id": f"t7-{i:03d}",
            "payload": {
                "kind": "messages",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Think step by step: what is {i} + {i + 1}?",
                    }
                ],
            },
        }
        for i in range(4)
    ]
    expected_ids = {p["id"] for p in prompts}

    cfg = InferenceConfig(
        regions=["europe-west4"],
        results_region="europe-west4",
        tpu_shapes=("v6e-4",),
        max_workers_per_region=1,
        shard_size=4,
        job_name="inf-integ-7-reasoning",
        # Give the model enough room to think out loud.
        sampling=SamplingParams(temperature=0.0, max_tokens=512),
    )
    # Qwen3-0.6B's chat variant emits <think> blocks; if your cluster has a
    # different reasoning model you can substitute it here.
    model = ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4})

    print(f"Model: {model.model}")
    print(f"{len(prompts)} prompts (chat with explicit 'Think step by step' instruction)")

    result = inference(model=model, dataset=prompts, config=cfg)
    records = assert_complete(result, expected_n=len(prompts), expected_ids=expected_ids)

    # The library must NOT strip <think>/<reasoning> markers. We check that
    # at least one record contains them — if zero do, the model isn't
    # emitting reasoning blocks today and the test is inconclusive; print a
    # clear note in that case rather than silently passing.
    marker_present = [r for r in records if "<think>" in r.response or "<reasoning>" in r.response]
    if not marker_present:
        print(
            "\nWARNING: NONE of the responses contain <think> or <reasoning> markers. "
            "Either the model isn't emitting them (possible for non-thinking variants) "
            "or the library is silently stripping them. Inspect the sample below:"
        )
        print_sample(records, n=len(records), max_chars=600)
        raise AssertionError(
            "No reasoning markers in any output. Test inconclusive — verify with a known "
            "thinking model (e.g. Qwen3-0.6B-Thinking if available)."
        )

    print(f"\n{len(marker_present)} of {len(records)} responses contain reasoning markers.")
    print_sample(marker_present, n=2, max_chars=400)

    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
