# Distributed inference integration tests

One-off scripts that exercise `marin.inference.distributed.inference()`
against a real Iris cluster. These are **not** part of either upstream PR;
this directory lives only on `integration-tests/inference` in
`XenonMolecule/marin`.

## Goal

Catch bugs in the distributed inference library before opening the upstream
PR. Unit tests alone cannot exercise:

- Real vLLM-on-TPU compile and inference
- True multi-region scheduling via Iris
- GCS bucket interactions (lifecycle, atomic writes, list semantics)
- Per-region rotation under real worker count
- vLLM SamplingParams round-trip with a real engine
- XLA compile-cache fetch/push behavior across runs

## QA workflow

For each test below:

1. Launch on the real cluster (see "Running" section).
2. If it **passes**: move to the next test.
3. If it **fails**:
   a. Switch to the PR branch (`inference/distributed-library`).
   b. Write a **unit test** in `tests/inference_distributed/test_distributed_inference.py` that reproduces the failure mode with a stub engine / mocked GCS where possible. The test must fail before your fix.
   c. **Fix** the bug on the PR branch.
   d. Run the local unit tests (`uv run pytest tests/inference_distributed/`) — they must pass.
   e. Rebase `integration-tests/inference` onto the updated PR branch.
   f. Re-run the failing integration test on the real cluster.
4. Only after all tests pass do we open the upstream PR.

Every cluster-only bug results in a *permanent regression test* in the
upstream codebase, so the failure mode cannot silently return.

## The tests

| # | Script | What it exercises |
|---|---|---|
| 1 | `test1_hf_model.py` | Basic smoke — load a small public HF model on the cluster. Single region, text payloads. Quickest sanity check. |
| 2 | `test2_conversational.py` | OpenAI-messages payload → `engine.chat()` dispatch. Verifies the chat template path. |
| 3 | `test3_completion.py` | Raw-text payload → `engine.generate()` dispatch. |
| 4 | `test4_multi_region.py` | Two regions writing to a single `results_region`. Per-region rotation, cross-region dedup behavior, output assembly. |
| 5 | `test5_many_shards.py` | Enough prompts that the run requires multiple shards in a single region. Exercises shard checkpointing and worker reuse of vLLM across shards. |
| 6 | `test6_xla_cache_reuse.py` | Run twice back-to-back with the same model. First run populates `gs://marin-{region}/tmp/ttl=30d/vllm-cache/{model_hash}/`; second run pulls from it. Verifies the second run is materially faster (engine load < cold-start time). |
| 7 | `test7_reasoning_preserved.py` | Verify generated outputs contain `<think>` / `<reasoning>` markers verbatim. The library promises no stripping; downstream callers post-process as they need. |
| 8 | `test8_region_mismatch_crashes.py` | Negative test: pass a `gs://marin-us-central1/...` model URI to a worker scheduled in `us-east5`. The pre-flight `check_gcs_paths_same_region` must crash the regional job before vLLM loads (avoids a ~16 GB cross-region weight download). |
| 9 | `test9_sampling_params.py` | Set non-default `SamplingParams` (`temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.1`, `stop=["END"]`). Verify sampling actually fires (>=2 unique outputs from identical prompts) and `finish_reason` is captured in extras. |
| 10 | `test10_marin_uri_model.py` | Load a model from a `marin://` URI (per-region resolution to `gs://marin-{region}/...`). The production loading pattern for Marin-trained checkpoints. Edit `MARIN_MODEL_URI` at the top of the script if the default path has moved. |
| 11 | `test11_tpu_shape_alternatives.py` | `tpu_shapes` multi-shape fallback. Passes two topology-matched variants (`("v6e-4", "v5litepod-4")`); the scheduler may land on either. Fills the gap left by tests 1–10, which each pin a single TPU shape. |

Tests 1-3 are the smallest and fastest — run them first. Tests 4-5 need
~10 minutes wall clock. Test 6 needs ~20 minutes (two full runs).

## Running on the cluster

Each test is a standalone Python module submitted as an Iris job:

```bash
uv run iris --cluster marin job run --priority batch --no-preemptible \
    --memory 2GB --cpu 2 \
    --job-name infinttest-1 \
    -- python experiments/inference_integration/test1_hf_model.py
```

The parent (this script) runs as a small CPU job and submits per-region
Zephyr jobs internally via the library. Watch progress in the Iris UI; the
script prints a status line at the end.

For tests 4–6 you'll probably want a higher `--memory`/`--cpu` ceiling on
the parent and a longer run window:

```bash
uv run iris --cluster marin job run --priority batch --no-preemptible \
    --memory 4GB --cpu 2 \
    --job-name infinttest-6 \
    -- python experiments/inference_integration/test6_xla_cache_reuse.py
```

## Cleanup

**Every test deletes its outputs and inputs on success.** The teardown
happens after assertions pass. Each test prints what it removed.

If a test **crashes mid-run**, leftover files remain at the printed
`results_uri`. Run the cleanup helper afterward:

```bash
python experiments/inference_integration/cleanup.py --job-name <prefix>
```

Or just `gsutil rm -r gs://marin-<region>/<job_name>/`.

## Hardware

All tests use `tpu_shapes=("v5p-8",)` — single 4-chip v5p slice per worker.
Small models (≤1B params) fit comfortably; larger models may need explicit
overrides to `tensor_parallel_size=4` via `ModelSpec.engine_kwargs`.

## What each script asserts

Every script returns nonzero on failure. The assertions are intentionally
brittle — silent-success is worse than a noisy crash.

- `result.is_complete` (no `missing_shards`)
- `len(result.to_list()) == len(input_prompts)`
- Every input id has a corresponding output record (no id loss)
- Response text is non-empty
- Test-specific assertions (chat dispatch was used / reasoning markers
  present / XLA cache populated / etc.)

When a test asserts something subtle, it prints a sample record so you
can eyeball it.

## After all tests pass

1. Verify PR 1 (`https://github.com/marin-community/marin/pull/5875`) has merged.
2. Rebase the PR 2 branch onto upstream main.
3. Open the upstream PR for PR 2 (the library).
