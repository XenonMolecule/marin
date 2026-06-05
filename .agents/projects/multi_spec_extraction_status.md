# Multi-spec extraction — status snapshot (auto-generated 2026-05-09 ~20:05)

## Plumbing complete
- `experiments/baseline_collection/extraction_specs.py` — registry with
  `low_quality` (legacy unprefixed path) and `med_quality` (quality_extraction_v8).
  `high_quality` slot reserved for the in-progress spec.
- `multi_region_extraction.py` — new flags: `--spec`, `--manifest`,
  `--exclude-manifest`, `--iris-priority`, `--iris-config`, `--fleet`,
  `--num-workers-override`, `--dry-run`. Replaced `fsspec`/`gcsfs` writes with
  `google-cloud-storage` (SSL cert workaround for local).
- `download_and_extract.py` — accepts `tpu_variant` from config so the entrypoint
  can run CPU-only (new Iris pattern). RAM bumped from 32g → 64g per worker.
- `run_extract_standalone.py` — `--spec` flag, spec-aware completed registry.
- `dashboard.py` + `dashboard.html` — `--specs` CLI, per-spec scanning, spec
  selector dropdown. Cache migrates legacy on-disk shape into the legacy slot.
- `lib/zephyr/src/zephyr/execution.py` — bumped Zephyr heartbeat timeout
  default 120s → 600s; was failing on every TPU vLLM job during engine compile.
  (Library-level change; flag if you'd like it reverted.)

## Iris CLI / API drift handled
- `--config` is a top-level flag (must precede `job run`).
- Entrypoint job can no longer carry `--tpu` without `--enable-extra-resources`;
  switched to CPU entrypoint + workers carry the TPU variant via Zephyr.
- `iris job` no longer has a `status` subcommand — using `summary` instead.
- `iris job list` has `--state`, `--prefix` only (no `--name-filter` / `--user`).
- `ZephyrContext.__init__` no longer accepts `chunk_size` (upstream merge);
  removed.
- `from fray.v2.types import ResourceConfig` is stale; the canonical path is
  `from fray.types import ResourceConfig` (renamed in commit 196a60557). Fixed
  in `download_warcs.py` and `download_and_extract.py`. There are ~15 other
  experiment files with the same stale import — left alone for now.

## Test state
- Job: `/michaelryan/extract-med_quality-v6e-4_1778382315_0` (current run #3
  after two failures: OOM at 32g RAM, then heartbeat timeout at 120s).
- Manifest: 5 WARCs from `experiments/distill/subsets/baseline_warcs_test1.txt`
  (first 5 of priority 100).
- Fleet: v6e-4 × num_workers=2 (small for cheap validation).
- Output target: `gs://marin-{region}/documents/baseline_llm_extraction/med_quality/data-{hash}.jsonl.gz`.

## Scale-up plan (after test confidence)
1. `--spec med_quality --manifest experiments/distill/subsets/baseline_warcs_100.txt
    --exclude-manifest experiments/distill/subsets/baseline_warcs_test1.txt
    --fleet v5p-8 --num-workers-override 4 --iris-priority batch`
2. Same with `--fleet v6e-16 --num-workers-override 4`
3. Watch progress for ~1-2h; if healthy, switch manifest to
   `experiments/distill/baseline_warcs_3000.txt` with the priority manifests
   excluded.

## Things to look at
- `experiments/distill/subsets/baseline_warcs_test1.txt` is a 5-WARC scratch
  manifest I created for the test. Delete or repurpose as needed.
- The dashboard's quick-scan logic is geared toward `run_extract_standalone.py`'s
  directory layout. `download_and_extract.py` writes flat `data-{hash}.jsonl.gz`,
  which the dashboard's deep-scan does pick up but the quick-scan does not.
  Pre-existing inconsistency; may want to unify.

## Status update (2026-05-09 ~20:50)

Bugs found and fixed during the run:
1. Iris top-level `--config` flag must precede `job run`. Added `--iris-config`
   CLI flag with default `lib/iris/examples/marin.yaml`.
2. Coordinator can't request `--tpu` or memory ≥4GB without
   `--enable-extra-resources`. Switched to CPU coordinator (cpu=2, memory=2GB);
   workers carry the TPU variant via Zephyr's ResourceConfig (config field
   `tpu_variant`).
3. Zephyr `chunk_size` kwarg removed upstream; dropped from ZephyrContext call.
4. `from fray.v2.types import ResourceConfig` is stale; the canonical path
   is `from fray.types import ResourceConfig`. Fixed in `download_warcs.py`
   and `download_and_extract.py`.
5. Per-worker RAM bumped from 32g → 64g (was OOMing on shard 3).
6. Zephyr heartbeat timeout default 120s was too short for vLLM TPU compile.
   Patched `lib/zephyr/src/zephyr/execution.py` defaults 120 → 600.
7. Orchestrator dedup: track in-process `submitted_hashes`, subtract from
   working set so iter 2 doesn't re-submit iter 1's WARCs.
8. Orchestrator was exiting on "all dispatched" instead of "all completed".
   Now polls until GCS confirms completion of every WARC.
9. Workers were getting unschedulable: `region=us-east1` constraint inherited
   from CPU parent, but `v5p-8` only exists in us-central1 and us-east5.
   Added `region` field to `TpuFleetEntry`; `--fleet v5p-8:us-east5` syntax
   pins parent's region.

Outstanding behavior to watch:
- Test (v6e-4) and v5p-8 batch both show outer-loop retries (workers-a0 → a1)
  after engine init, with no logs of progress. Cause unclear; could be Iris
  silent worker death, very slow `_filter_by_token_length` (tokenizing many
  records single-threaded), or another timeout.
- v5p-8:us-east5 batch on attempt #2; engines initialized at ~20:58; waiting
  for first generation log line.
- Test (v6e-4) on attempt #2; downloading WARCs again as of ~20:59.

Cost so far: ~$10 (rough — 2× v6e-4 + 4× v5p-8 over ~1.5h cumulative).

## Status update (2026-05-09 ~21:30) — BIG bug found and fixed

After rounds of "engine init succeeds, then silent failure", a worker traceback
finally revealed:

```
File "/app/experiments/baseline_collection/download_and_extract.py", line 276,
  in _format_prompts
    from vllm.inputs.data import TokensPrompt
ModuleNotFoundError: No module named 'vllm.inputs.data'
```

The vLLM upgrade in the upstream merge moved `TokensPrompt` out of
`vllm.inputs.data`. Fixed with a try/except chain over `vllm.inputs`,
`vllm.inputs.data`, and `vllm` (top-level).

This explains every prior failure: workers loaded the model successfully,
filtered records successfully, then crashed on the first `_format_prompts`
call when they hit the stale import. The crash was silent because
`_format_prompts` is called inside Zephyr's stage runner subprocess; the
parent only saw "Subprocess shard execution failed" with the traceback
buried in worker stdout.

Killed _0 (slow filter, OLD bundle) and _1 (fast filter, OLD bundle); both
had the broken import. Relaunched orchestrator on `baseline_warcs_100.txt`
(no exclusion now since the test is gone) with the fixed bundle:
`extract-med_quality-v5p-8_1778387395_0`.

If this run produces a "Batch:" log line, the entire path is healthy and we
can fan out to more orchestrators.
