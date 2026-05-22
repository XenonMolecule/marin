# Day 2 integration-test run — morning report (2026-05-22)

**Run window**: 2026-05-21 ~01:46 PDT through 2026-05-21 ~21:53 PDT (~20h elapsed wall, ~7h productive).

## TL;DR

- **All 11 integration tests passed end-to-end on real Iris + TPU.** The suite covers the full distributed-inference path on a live cluster: HF model load, chat vs. completion dispatch, multi-region aggregation, multi-shard, XLA cache populate-and-reuse, reasoning-marker preservation, region-mismatch pre-flight, sampling params, marin:// URI loading, and (new today) `tpu_shapes` multi-shape fallback.
- **Two real bugs were caught by the integration suite and fixed on `inference/distributed-library`.** Both followed the documented fix-loop: failing cluster test → unit-test reproduction → fix → re-run on cluster.
- **PR 2 is ready to open upstream.** The feature branch has been rebased onto current `upstream/main` (PR 1 is in main; the redundant `[zephyr]` commit was dropped cleanly during rebase). `PR_DESCRIPTION.md` is unchanged and ready to paste.

## Test results

| # | Test | Region/TPU | Wall | Outcome |
|---|---|---|---|---|
| 1 | HF model, single region | us-central1 / v5p-8 | ~12 min | ✅ PASS |
| 2 | Conversational (chat dispatch) | us-central1 / v5p-8 | ~12 min | ✅ PASS |
| 3 | Completion (text dispatch) | us-central1 / v5p-8 | ~9 min | ✅ PASS |
| 4 | Multi-region aggregation | us-central1 + us-east5 / v5p-8 | ~11 min | ✅ PASS (us-east5 failed gracefully; us-central1 covered) |
| 5 | Many shards (60 prompts) | us-central1 / v5p-8 | ~8 min | ✅ PASS |
| 6 | XLA cache reuse | us-central1 / v5p-8 | ~3h total (2 runs) | ✅ PASS — **36.79× speedup** run1→run2 |
| 7 | Reasoning markers preserved | europe-west4 / v6e-4 | ~18 min | ✅ PASS — 4/4 responses preserve `<think>` blocks |
| 8 | Region-mismatch pre-flight crash | europe-west4 / (no TPU; pre-flight fails first) | ~1 min | ✅ PASS — `missing_shards=(0,)`, `is_complete=False` |
| 9 | SamplingParams pass-through | europe-west4 / v6e-4 | ~6 min | ✅ PASS — 6/6 unique outputs with `temperature=0.7` |
| 10 | `marin://` URI model | europe-west4 / v6e-4 | ~60 min (8B cold compile) | ✅ PASS — Qwen3-8B rephraser via `gs://marin-eu-west4/...` |
| 11 | `tpu_shapes` multi-shape fallback (**new**) | europe-west4 / `("v6e-4", "v5litepod-4")` | ~14 min | ✅ PASS — scheduler picked v6e-4 from the list |

Test 11 is new and fills a gap I noticed mid-run: tests 1–10 each pin a single TPU shape, leaving the library's documented multi-shape API completely unexercised. It's a 1-file addition under `experiments/inference_integration/`.

## Bugs surfaced and fixed

Both bugs followed the same fix-loop discipline: failing cluster test → unit-test reproduction on `inference/distributed-library` → fix → cluster re-run.

### Bug 1: `InferenceConfig.shard_size` was a dead config knob

**Surfaced by**: test 4 (multi-region). Cluster failure: `AssertionError: expected 4 shards in output, got 1 ([0])` — 32 prompts at shard_size=8 produced one output shard instead of four.

**Root cause**: `lib/marin/src/marin/inference/distributed/input.py` hard-coded `_INPUT_RECORDS_PER_FILE = 5000`. The materializer chunked records by that constant instead of `cfg.shard_size`. The library's pipeline maps one input file to one content shard (`assign_shard_ids`), so the documented + validated `shard_size` knob was effectively unused for inline-input calls.

Tests 1–3 passed despite the bug because their N (≤8) was smaller than every shard_size used; test 4 was the first to explicitly check the shard count.

**Fix**: commit `1cebc5fb7 [inference] Honor InferenceConfig.shard_size for inline input`. `api._prepare_input` now threads `cfg.shard_size` into `materialize_inline_input(..., records_per_file=shard_size)`. `input.py` requires the caller to pass `records_per_file` (no silent default). `config.py` docstring rewritten to accurately describe the knob's effect.

**Regression tests added** (`tests/inference_distributed/test_distributed_inference.py`):
- `test_materialize_inline_input_records_per_file_controls_file_count`
- `test_inference_inline_input_shard_size_controls_output_shard_count` (end-to-end mirror of the failing cluster test)
- `test_inference_inline_input_uneven_shard_size_rounds_up`

### Bug 2: XLA compile-cache env vars set on the wrong process

**Surfaced by**: test 6 (XLA cache reuse). Cluster failure: `AssertionError: Expected compile-cache files at gs://marin-us-central1/tmp/ttl=30d/vllm-cache/10914d595e06c937 after run 1, found none.`

**Root cause**: `regional_job.main` called `compile_cache.configure_env(cache_uri)` which mutated `os.environ` of the regional **CPU coordinator** process. The vLLM engine that needs `JAX_COMPILATION_CACHE_DIR` / `VLLM_XLA_CACHE_PATH` runs on a *separate* Zephyr TPU worker process, which doesn't inherit the coordinator's env.

This is structurally identical to the `worker_extras` bug Michael fixed yesterday: both required threading state through Fray's `EnvironmentConfig` to the worker submission rather than setting it on the dispatching process.

**Fix**: commit `dd7a95ba6 [inference] Thread XLA compile-cache env vars to the TPU worker process`. `regional_job._build_context` now builds the cache env-var dict via a new helper and passes it to `create_environment(env_vars=..., extras=...)`. `worker_environment` is created when extras OR env_vars are non-empty. `configure_env`'s `env` parameter type was tightened from `Mapping[str, str]` to `MutableMapping[str, str]` (it calls `setdefault`).

**Regression tests added**:
- `test_build_context_threads_compile_cache_env_vars_to_worker_environment` — pins that the env vars land on `worker_environment.env_vars`.
- `test_build_context_includes_compile_cache_env_vars_even_when_no_extras` — covers the cache-only case.
- `test_resolve_cache_uri_empty_template_disables_cache` — `template=""` sentinel for explicit opt-out.
- `test_build_context_omits_worker_environment_when_extras_empty` was renamed and adjusted (with the empty-template sentinel) to reflect that the env vars are now an independent driver of `worker_environment`.

**Cluster validation**: after the fix, test 6 produced 29 cache files in `gs://marin-us-central1/tmp/ttl=30d/vllm-cache/10914d595e06c937/` after run 1 (proving the env vars reached the TPU worker) and run 2 completed in 297.6s vs. run 1's 10948.7s — **36.79× speedup**.

## Cluster gotchas hit (operational notes)

These cost real wall-clock time today. Worth knowing for future runs.

1. **us-central1 has no non-preemptible v5p-8 pool at all.** The error message is explicit: `unschedulable: no non-preemptible group provides device tpu:v5p-8`. Setting `worker_preemptible=False` in us-central1 → instant fail-loop (~13s/attempt, retried ~12 times before I killed it).

2. **us-east5 CPU shortage at the coordinator level**, same condition the previous morning report flagged. The 0.5-core Zephyr coordinator container can't land; the TPU is irrelevant because we never reach the worker request.

3. **Calvin's `dm-proportional-controllabi...` job is hammering preemptible v5p-8.** Three preemptions in 3 hours on a fresh us-east5 schedule; the run never made progress. Switching to non-preemptible would have helped if (1) above didn't dead-end us in us-central1.

4. **`europe-west4` is the iris region name, not `eu-west4`.** I burned one whole launch on a config validation error. The GCS bucket suffix is `marin-eu-west4` (different convention from the iris region name) — so the InferenceConfig validation rejected `regions=["eu-west4"]` even though the bucket is `gs://marin-eu-west4/`. The `REGION_TO_DATA_BUCKET` mapping in `rigging.filesystem` makes this clear once you look for it.

5. **iris `job run --job-name X` attaches to existing job `X`.** Reusing a previously-used name (e.g., re-launching `infinttest-4` after a fix) tails the prior job's logs instead of creating a fresh one. The CLI will sit for many hours waiting for the prior already-terminal job's log stream to drain. Always bump the name on retry (`-4` → `-4b` → `-4c`). I lost ~7 hours of overnight runtime to this before catching on.

## State of branches at the end of this run

| Branch | Head | Pushed | Notes |
|---|---|---|---|
| `inference/distributed-library` | `dd7a95ba6` | yes (force-pushed) | PR 2 (ready, **not** opened). 5 commits, all `[inference]`. Rebased onto current `upstream/main` cleanly — no `[zephyr]` commit needed (PR 1 is in main). 39/39 unit tests pass; pre-commit + pyrefly clean. |
| `integration-tests/inference` | `c12fb3300` | yes (force-pushed) | Tests 7–10 region-shifted to europe-west4 / v6e-4; test 11 new. README updated. Rebased on top of the rebased feature branch. |
| `zephyr/per-context-limits` | (merged) | n/a | PR 1 was merged upstream as #5875 on 2026-05-20. No further work needed; can delete locally if you want. |

## What's left for you (recommended order)

1. **Open PR 2** upstream against current `marin-community/marin` main, using `PR_DESCRIPTION.md` (unchanged from yesterday's draft — the description is the right shape; the 2 new bug fixes are documented in commit messages and don't need to be called out in the PR body).
2. After PR 2 merges, the integration-tests branch can be archived. It deliberately does not get upstreamed — it lives only on the fork.
3. Optional follow-ups noted during the run:
   - README's `--memory 4GB --cpu 2` example for tests 4–6 is now incorrect — iris's `--enable-extra-resources` policy kicks in at ≥4 GB. Either drop the example to 2 GB or add the flag. Caught early in this run.
   - Test 6's "regional success but heartbeat noise during shutdown" race (mentioned in yesterday's report) reappeared in several tests as benign teardown spam. Worth filing a small Zephyr issue: coordinator should remain reachable long enough for workers' final heartbeat to land cleanly.

## Memories written during this run

Two saved to `/Users/michaelryan/.claude/projects/-Users-michaelryan-Documents-School-Stanford-Research-marin-fresh/memory/`:

- `feedback_autonomous_overnight.md` — never ask blocking questions in overnight/autonomous mode; find workarounds and defer to the morning report.
- `reference_iris_job_name_reuse.md` — the iris job-name-reuse gotcha above.

Sleep well — there is nothing on fire, and PR 2 is one `gh pr create` away from being filed.
