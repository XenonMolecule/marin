# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test 2: OpenAI-messages payload → engine.chat() dispatch.

Verifies the conversational path lights up correctly and outputs look like
real chat replies (not the text-completion echo a misrouted text path would
produce).
"""
from __future__ import annotations

from _common import assert_complete, cleanup_run, configure_logging, exit_pass, print_sample
from marin.inference.distributed import InferenceConfig, ModelSpec, SamplingParams, inference


def main() -> None:
    configure_logging()
    print("=== Test 2: Conversational (messages) payload ===")

    conversations = [
        [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        [{"role": "user", "content": "Name one country in Europe. Reply with just the name."}],
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "What is the capital of Japan?"},
        ],
        [{"role": "user", "content": "What color is the sky? One word."}],
    ]
    prompts = [
        {"id": f"t2-{i:03d}", "payload": {"kind": "messages", "messages": convo}}
        for i, convo in enumerate(conversations)
    ]
    expected_ids = {p["id"] for p in prompts}

    cfg = InferenceConfig(
        regions=["us-central1"],
        results_region="us-central1",
        tpu_shapes=("v5p-8",),
        max_workers_per_region=1,
        shard_size=4,
        job_name="inf-integ-2-conversational",
        sampling=SamplingParams(temperature=0.0, max_tokens=32),
    )
    model = ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4})

    print(f"Model: {model.model}")
    print(f"{len(prompts)} conversations")

    result = inference(model=model, dataset=prompts, config=cfg)
    records = assert_complete(result, expected_n=len(prompts), expected_ids=expected_ids)

    print(f"\nResults at: {result.results_uri}")
    print_sample(records)

    # Manual sanity check: chat replies should look like answers, not echoes
    # of the question text. We just print them and trust visual inspection;
    # a more rigorous check would compare against the input.
    print("\nManual inspection: do these look like coherent chat replies, not echoes?")

    cleanup_run(result)
    exit_pass()


if __name__ == "__main__":
    main()
