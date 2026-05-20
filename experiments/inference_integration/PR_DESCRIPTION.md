# PR 2 — copy-paste body for upstream submission

Use this body when opening the PR for `inference/distributed-library`
against `marin-community/marin:main`, after PR 1 (#5875) has merged and
the branch has been rebased.

Title:

```
[inference] Distributed inference library (multi-region vLLM-on-TPU)
```

Suggested labels: `agent-generated`.

Suggested gh command:

```bash
gh pr create --repo marin-community/marin --base main \
    --head XenonMolecule:inference/distributed-library \
    --label agent-generated \
    --title "[inference] Distributed inference library (multi-region vLLM-on-TPU)" \
    --body "$(cat experiments/inference_integration/PR_DESCRIPTION.md | sed -n '/^---BODY---$/,$p' | tail -n +2)"
```

---BODY---
## Summary

Adds `marin.inference.distributed`, a multi-region distributed inference
library targeting vLLM-on-TPU. One stateless top-level entry point:

```python
from marin.inference.distributed import inference, InferenceConfig, ModelSpec, SamplingParams

result = inference(
    model=ModelSpec(model="marin://checkpoints/qwen3-8b/hf/step-1318"),
    dataset=[{"id": "1", "payload": {"kind": "text", "prompt": "Hello"}}],
    config=InferenceConfig(regions=["us-central1"], results_region="us-central1"),
)
print(list(result.iter_records()))
```

Under the hood, the library:

- Launches **one Zephyr job per region** in `config.regions`; each region's
  workers run vLLM on the configured TPU shape.
- Writes all shard outputs to a **single canonical `results_region`** so
  downstream ExecutorStep consumers pay no cross-region read egress.
- Uses **per-region deterministic input rotation** so workers in different
  regions mostly process disjoint shards; `skip_existing` on the shared
  output prefix arbitrates the rare race.
- Caches the **vLLM engine at module scope on each Zephyr worker** so it
  survives across shards (via `InlineRunner`), avoiding repeated XLA
  compile.
- Wires the **JAX/vLLM XLA compile cache** to
  `gs://marin-{region}/tmp/ttl=30d/vllm-cache/{model_hash}/` so workers in
  the same region share compiled artifacts across runs.
- Validates region safety via existing `rigging.filesystem.check_gcs_paths_same_region`
  (selective — exempts the intentionally cross-region `results_region`).
- Preserves model output **verbatim** (including `<think>` / `<reasoning>`
  markers) — downstream callers post-process as they see fit.

Depends on the per-context Zephyr config fields from #5875 (PR 1).

## How to run inference

Minimal example (single region, public HF model, text prompts):

```python
from marin.inference.distributed import (
    InferenceConfig, ModelSpec, SamplingParams, inference,
)

prompts = [
    {"id": f"p{i}", "payload": {"kind": "text", "prompt": f"Question {i}:"}}
    for i in range(100)
]

cfg = InferenceConfig(
    regions=["us-central1"],
    results_region="us-central1",
    tpu_shapes=("v5p-8",),
    max_workers_per_region=4,
    shard_size=500,
    sampling=SamplingParams(temperature=0.0, max_tokens=512),
    job_name="my-inference-run",
)

result = inference(
    model=ModelSpec(model="Qwen/Qwen3-0.6B", engine_kwargs={"tensor_parallel_size": 4}),
    dataset=prompts,
    config=cfg,
)
print(f"Outputs at: {result.results_uri}")
print(f"Missing shards: {result.missing_shards}")
for record in result.iter_records():
    print(record.id, record.response)
```

Multi-region with messages payload:

```python
cfg = InferenceConfig(
    regions=["us-central1", "europe-west4"],
    results_region="us-central1",          # canonical sink
    tpu_shapes=("v6e-4",),
    max_workers_per_region=8,
    shard_size=500,
    sampling=SamplingParams(temperature=0.7, max_tokens=2048, top_p=0.9),
)
prompts = [
    {"id": f"chat-{i}", "payload": {
        "kind": "messages",
        "messages": [{"role": "user", "content": "..."}]
    }}
    for i in range(N)
]
result = inference(model=..., dataset=prompts, config=cfg)
```

Model paths support three shapes:

- HF id (e.g. `"meta-llama/Llama-3-8B"`) — vLLM downloads at startup.
- `marin://checkpoints/...` — resolves per worker region to
  `gs://marin-{worker_region}/checkpoints/...`; use this for Marin-trained
  checkpoints replicated across regions.
- Explicit `gs://...` — hard-pinned; workers scheduled in a different
  region crash on the pre-flight `check_gcs_paths_same_region` check.

For long-running runs that exceed the upstream Zephyr defaults, raise the
per-context fields:

```python
cfg = InferenceConfig(
    ...,
    heartbeat_timeout=1800,          # default 120 — raise for cold XLA compile
    max_shard_infra_failures=200,    # default 20 — raise for preemption-heavy runs
)
```

vLLM throughput tuning belongs in `ModelSpec.engine_kwargs` (passed through
unchanged to `vllm.LLM(...)`):

```python
ModelSpec(
    model="...",
    engine_kwargs={
        "tensor_parallel_size": 4,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.92,
        "enable_prefix_caching": True,
        "max_model_len": 32768,
    },
)
```

## What's intentionally out of v1

- Multi-host TPU (v5p-16, v5p-32). Single-host shapes only.
- `SamplingParams.n > 1` (rejected up front so we don't silently drop
  completions). Multi-completion fans out via either nested-list or
  fan-out — landing in a follow-up.
- Levanter / remote / litellm backends. vLLM only.
- `token_ids` / `prompt_token_ids` in output extras (large; re-tokenize
  from `text` if needed). Add `capture_token_ids` config if a workflow
  demands it.

## Test plan

- [x] 29 / 29 unit tests under `tests/inference_distributed/` pass against
      a stub engine + `LocalClient` + local filesystem. Cover: config
      validation, model URI resolution, payload-kind validation, compile-cache
      resolution, per-region rotation determinism, response extraction
      (including reasoning-marker preservation), full pipeline end-to-end
      with stubbed engine, multi-region dedup, mixed-kind error surfacing.
- [x] `./infra/pre-commit.py --all-files` clean (ruff, black, pyrefly,
      license).
- [x] No regression in `lib/zephyr/tests/test_execution.py` (46 / 46
      pass; verifies the PR 1 changes still hold).
- [ ] Real-cluster integration tests pending separately on
      `XenonMolecule:integration-tests/inference` (capacity-limited
      overnight; will re-run before requesting review).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
