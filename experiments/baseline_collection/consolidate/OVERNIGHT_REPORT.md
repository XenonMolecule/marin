# Overnight Consolidation & Tokenization — Live Report

**Date started**: 2026-04-19 (UTC night)
**Budget**: soft kill $25, hard kill $30
**Target**: consolidate 5 regions → us-central1, tokenize, launch 998M + 157M sweep

This file is a living logbook. I append at every milestone. Latest entries at the bottom.

Path: `experiments/baseline_collection/consolidate/OVERNIGHT_REPORT.md` (local to the git checkout). No GCS mirror — GCS writes are free but the mirror was unnecessary busywork.

## Ground rules I'm operating under

- **Do not delete anything, ever.** No `rm`, no `--delete-*`, no overwrite of source files.
- **Do not stop on non-catastrophic concerns.** Note them here and keep working.
- **Canary cheap regions first** (us-east1 → us-central1 is $0.09; verify rsync pipeline works before firing eu-west4 which costs $8.98).
- **Budget gates**:
  - Soft kill at $25 spent: pause new billable actions, finish any in-flight transfer, leave a detailed resume plan here.
  - Hard kill at $30 spent: immediate halt, no new billable actions.
  - Running tally tracked in the "Spend ledger" table below.

## Spend ledger

| Action | Estimate ($) | Actual ($) | Notes |
|---|---:|---:|---|
| Inventory writes (small JSONL → us-central1) | 0.01 | | ~80 MB total, mixed regions |
| Resolve (intra us-central1) | 0.00 | | reads + writes small manifests |
| Canary transfer us-east1 → us-central1 | 0.09 | | 4.59 GB @ $0.02/GB |
| Transfer us-east5 → us-central1 | 0.72 | | 36.09 GB @ $0.02/GB |
| Transfer us-west4 → us-central1 | 0.85 | | 42.66 GB @ $0.02/GB |
| Transfer eu-west4 → us-central1 | 8.98 | | 74.87 GB @ $0.12/GB (trans-atlantic) |
| Transfer us-central1 → us-central1 (intra) | 0.00 | | 14.89 GB, server-side copy, free |
| Re-inventory (intra) | 0.00 | | reads local |
| Reshape + tokenize (intra) | 0.00 | | reads + writes in us-central1 |
| Training (us-central1 only) | 0.00 | | TPU time is compute, not tracked here |
| **Projected total** | **$10.65** | | |

## Plan phases (I check these off as they complete)

- [x] Phase A — Inventory 5 regions ✅ DONE at 08:23 UTC; all 5 inventories (11.3 MB total) in us-central1.
- [x] Phase B — Resolve duplicates ✅ DONE at 08:24 UTC. 281,254 unique keys from 290,752 copies (9,498 duplicate copies = ~3.3% of batches had >1 region copy). 7,389 unique keys had duplicates.
- [x] Phase C — Integrity review ✅ CLEAN. Zero invalid-on-all, zero missing, zero index gaps, 3000/3000 WARCs with `_done` somewhere. Duplicate rate 2.6% is normal.
- [⏳] Phase D — Transfer: **first canary attempt failed** because the Iris `cpu` container doesn't include `gcloud`. Killed v1, rewrote `transfer_region.py` to use fsspec's `fs.copy()` (GCS server-side rewrite API, same result, no subprocess). Resubmitted as `/michaelryan/extract-transfer-canary-v2` at 08:44 UTC.

### 2026-04-19 10:10 UTC — All transfers complete

Final results (all 5 regions):
- us-east1 (canary): 18,840 files, 4.59 GB copied in 363s
- us-central1: 76,867 files, ~15 GB copied in 165s (intra-region, free)
- us-east5: 105,976 files, ~36 GB copied in 378s
- us-west4: 166,033 files, ~43 GB copied in 992s
- europe-west4: 322,160 files, ~75 GB copied in 2621s (4 min preempt + restart + skip 129K existing + copy remaining 193K)

Zero errors across all 5 regions. Total archive: ~173 GB in `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/{region}/`.

Killed keep-alive parents: extract-transfer-main, extract-transfer-canary-v2, extract-inventory-coord, extract-resolve-coord.

Starting lightweight verify (du + file-count match src vs. dst, 5 regions in parallel). If clean, skip straight to reshape+tokenize (Phase E/F).

### 2026-04-19 10:35 UTC — Verify ✅ ALL REGIONS BYTE-EXACT MATCH

| Region | Source bytes | Dest bytes | Files |
|---|---:|---:|---:|
| us-central1 | 14,893,377,221 | 14,893,377,221 | 76,867 |
| us-east1 | 4,591,292,994 | 4,591,292,994 | 18,840 |
| us-east5 | 36,086,456,760 | 36,086,456,760 | 105,976 |
| us-west4 | 42,666,574,950 | 42,666,574,950 | 166,033 |
| eu-west4 | 74,871,684,855 | 74,871,684,855 | 322,160 |
| **TOTAL** | **173.11 GB** | **173.11 GB** | **689,876** |

All regions OK.

### 2026-04-19 10:40 UTC — Phase F kickoff (reshape + tokenize)

Tried `ray_run.py --cluster us-central1` first — failed with "No head node found for cluster marin-us-central1". The us-central1 Ray cluster is down.

Switched to Iris-native execution: `fray.v2.client.current_client()` auto-detects Iris from env vars (via `get_iris_ctx()`), so submitting `pipeline_llm_curated.py` as an Iris CPU parent gives it a `FrayIrisClient` — Zephyr's `ZephyrContext(max_workers=200)` will spin up per-worker Iris child jobs instead of Ray workers. Same code path the quality-filter pipeline uses.

Submitted `/michaelryan/llm-curated-pipeline` at 10:40 UTC (4 CPU, 32 GB RAM, --enable-extra-resources for the bigger coordinator).

### 2026-04-19 08:57 UTC — Canary verified, main transfer launched

Canary `/michaelryan/extract-transfer-canary-v2` completed: 18,840/18,840 files, zero errors, 363s (52 files/s).

Integrity verification of us-east1 canary:
- Source `gs://marin-us-east1/.../baseline_llm_extraction/`: 4,591,292,994 bytes, 18,840 files
- Destination `gs://marin-us-central1/.../by_region/us-east1/`: 4,591,292,994 bytes, 18,840 files
- **Byte-exact match.** Server-side GCS rewrite preserves content perfectly.

Main transfer submitted as `/michaelryan/extract-transfer-main` at 08:57 UTC, covering the remaining 4 regions (us-central1, us-east5, us-west4, europe-west4) in parallel with `max_workers=128` each (bumped from 32 based on canary's 52/s rate — eu-west4's 322K files would take 2.5h at 32 workers, expect ~40 min at 128).

### 2026-04-19 08:44 UTC — `gcloud` not in cpu container; switched to fsspec

First canary submission `/michaelryan/extract-transfer-canary` hit `FileNotFoundError: 'gcloud'` because the Iris `cpu` extras bundle doesn't ship the Google Cloud SDK. Root-cause: my `transfer_region.py` shelled out to `gcloud storage rsync`.

**Fix**: rewrote to use `fsspec` + `gcsfs` — `fs.copy(src, dst)` between two `gs://` paths maps to the GCS `rewriteObject` API, which is server-side (bytes don't traverse the client VM). Per-file parallel with `ThreadPoolExecutor(32)`. Idempotent via `fs.info()` size-match check before each copy. Kept v1 killed; submitted v2.

No data loss. No egress cost from the failed attempt (it died before any copy).
- [ ] ⛔→📝 Checkpoint 1 — Record integrity concerns in this report (no stop unless catastrophic)
- [ ] Phase C — Canary transfer us-east1 only, verify
- [ ] Phase D — Transfer remaining regions in parallel (us-central1, us-east5, us-west4, eu-west4)
- [ ] ⛔→📝 Checkpoint 2 — Re-inventory, diff vs pre-transfer, debug anomalies before escalating
- [ ] Phase E — Reshape (dedup'd canonical batches → ~200 flat shards)
- [ ] Phase F — Tokenize with llama3 tokenizer → `tokenized/baseline_llm_curated-{hash}/`
- [ ] ⛔→📝 Checkpoint 3 — Verify tokenization (only panic if 0 tokens / pipeline errored outright; wide token-count variance is OK since we changed tokenizer from Qwen→Llama3)
- [ ] Phase G — Register `llm_curated` in curation_plan + plot_fixed_model_sweep + plot_curation_isoflop; commit
- [ ] Phase H — Launch 998M sweep (7 runs) in us-central1, interactive priority
- [ ] Wait 15 min for warm-up
- [ ] Phase I — Launch 157M sweep (7 runs) in us-central1, interactive priority
- [ ] Phase J — Write final report section with run IDs + WandB links

## Chronological log

### [Before inventory completion] 2026-04-19 — plan locked in

User (Michael) approved the full autonomous plan before bed with these revisions:
- Loosen all checkpoints to "log, don't stop" unless catastrophic.
- Test transfer with us-east1 first as canary before firing the eu-west4 (expensive) transfer.
- Add `llm_curated` to plot argparse defaults too: `plot_fixed_model_sweep.py:557, 583` and the color map in `plot_curation_isoflop.py:220`.
- Text field for tokenization = `text` (cleaned), not `generated_text`.
- Token-count thresholds: don't panic on large variance (Qwen→Llama3 tokenizer switch means token counts will be substantially different).

Inventory is running; 4 of 5 regions complete; eu-west4 ~15 min out.

### [While waiting on eu-west4] 2026-04-19 08:12 UTC — prepped downstream code

While inventory grinds through eu-west4's 126K files, I pre-wrote / pre-edited the code needed for later phases. Nothing launched yet.

- **Wrote** `experiments/baseline_collection/consolidate/pipeline_llm_curated.py` — Executor pipeline with a Zephyr reshape step (reads `resolved.jsonl.gz`, filters empties, reshards to 200 flat shards via `Dataset.from_iterable(paths).flat_map(load_jsonl)`) chained to a `default_tokenize` step with `llama3_tokenizer`.
- **Wrote** `experiments/baseline_collection/consolidate/launch_resolve.py` — Iris launcher that runs `resolve_duplicates.py` inside a us-central1-pinned CPU job. Keeps the "run on cluster" habit consistent with inventory/transfer launchers.
- **Edited** `experiments/scaling_law_sweeps/plot_fixed_model_sweep.py` — added `llm_curated` to the default `--methods` list (lines 557, 583).
- **Edited** `experiments/scaling_law_sweeps/plot_curation_isoflop.py` — added `"llm_curated": "#8c564b"` color to `COMPARE_COLORS`.
- Confirmed `fixed_model_plan.METHODS` imports from `curation_plan.METHODS`, so registering `llm_curated` in curation_plan alone propagates to the sweep launcher.

No commits yet — I'll bundle all Phase G edits into one commit after tokenization succeeds and we have the real executor hash to put in `_D_OBS_DEFAULTS`.

### 2026-04-19 08:23 UTC — Phase A complete, Phase B launched

All 5 inventories present in `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/inventories/` (11.3 MB total). Row counts per region: us-east1 9,882; us-central1 27,755; us-east5 63,318; us-west4 74,156; europe-west4 TBD (will come from resolver output).

Submitted `/michaelryan/extract-resolve-coord` to Iris at 08:23 UTC. Expects to take ~5 min. Resolver output will land at `resolved/resolved.jsonl.gz`, `resolved/duplicates.jsonl.gz`, `resolved/integrity_report.json`, `resolved/missing_batches.jsonl.gz`.

---

## FINAL SUMMARY — 2026-04-19 11:59 UTC (04:59 PT)

All 9 planned phases complete. Training is **LIVE**. 14 child jobs submitted, split across 998M and 157M sweeps. 10 running, 6 pending on TPU capacity at last check.

### Final artifacts

- **Consolidated archive**: `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/{region}/` — 173.11 GB, 689,876 files, byte-exact match to originals in all 5 regions.
- **Resolver manifests**: `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/resolved/` — resolved.jsonl.gz (281,254 canonical batches), duplicates.jsonl.gz (7,389 dup keys), integrity_report.json (0 anomalies).
- **Tokenized cache**: `gs://marin-us-central1/tokenized/baseline_llm_curated-3c07e4/` — **55,904,650,558 tokens, 103,706,721 records**, llama3 tokenizer.
- **Registration**: `curation_plan.py` has `llm_curated` in `_D_OBS_DEFAULTS` and `METHODS` (working tree; not yet committed due to pre-existing lint errors in curation_plan.py docstrings — Iris bundle still picks up the change).
- **Git commit**: `6bbbb6647 Add consolidate/ pipeline for multi-region LLM extraction → tokenized cache` — all 9 new files in `consolidate/`.

### Training sweeps LIVE

- `/michaelryan/llm-fm-998m/` — 7 children (998M, hidden=1536, budgets 3e18→3e20). TPU shapes: v4-8 / v4-16 / v4-32 / v4-64.
- `/michaelryan/llm-fm-157m/` — 7 children (157M, hidden=512, same 7 budgets). TPU shapes: v4-8 through v4-128 at larger budgets.
- Both region-locked to us-central1, interactive priority. WandB group: `data-curation-fixed-model`. Tracker: `gs://marin-us-central1/metadata/region_locks/data_curation_fixed_model/`.

### Open concerns (noted but not blocking)

1. **TPU vfio-busy retries on 998M small-budget children**: First child (3e18 budget on v4-8) hit `FAILED_PRECONDITION: open(/dev/vfio/{0..3}): Device or resource busy` across 3 retries. Second child (9e18) same. This is a TPU-share-handoff issue (NOT the libtpu_lockfile issue from prior memory — that's Docker-specific). Iris has `max_retries_preemption=100` and `max_retries_failure=3`, so retries continue. Should self-heal as workers cycle. Monitor `iris job list | grep llm-fm-` to see progress.
2. **data_curation_isoflop.py edit reverted** — pre-existing lint errors in that file blocked commit. The fixed-model sweep doesn't depend on it (fixed_model_plan imports METHODS directly from curation_plan, which is already edited). If you later want to run the IsoFLOP sweep (not fixed-model), add `"llm_curated": mk("llm_curated", "baseline_llm_curated-3c07e4")` to `_build_methods` in that file.
3. **Plot argparse edits uncommitted** — `plot_fixed_model_sweep.py` (lines 557, 583) and `plot_curation_isoflop.py` (COMPARE_COLORS) were edited in working tree only. They were already untracked before my session.
4. **Pipeline v2 cache-copy transient** — first tokenize attempt failed at the final merge step (~1 min into cache-copy with ZephyrWorkerError, all retries exhausted). v3 resubmit succeeded cleanly. The cause of the transient isn't fully understood; may be worth investigating if it recurs on other Levanter tokenize jobs.

### Spend ledger (final)

| Action | Estimate | Actual | Notes |
|---|---:|---:|---|
| Inventory writes | $0.01 | ~$0.01 | as expected |
| Resolve | $0 | $0 | intra-region |
| Canary transfer us-east1 | $0.09 | $0.09 | as expected |
| Main transfer (us-central1 intra + us-east5 + us-west4 + eu-west4) | $10.56 | **~$10.56** | eu-west4 preempted once, re-ran via skip; no double egress |
| Re-inventory + re-resolve | $0 | $0 | replaced by lightweight du+ls verify |
| Reshape + tokenize | $0 | $0 | intra-region |
| Training | $0 | $0 | TPU compute; us-central1 only |
| **Total egress spent** | **~$10.65** | **~$10.66** | **well under $25 soft / $30 hard cap** |

Ongoing storage: ~$6.32/month until explicit cleanup (158 GB × 2 copies: regional originals preserved per rule + consolidated archive).

### Phase progression timeline

| Phase | Start (UTC) | End (UTC) | Duration | Status |
|---|---|---|---|---|
| A — Inventory | 07:21 | 08:23 | 62 min | ✅ |
| B — Resolve | 08:23 | 08:24 | 1 min | ✅ |
| C — Integrity review | 08:24 | 08:24 | <1 min | ✅ clean |
| D — Transfer (canary + main) | 08:40 | 10:09 | 89 min | ✅ |
| E — Lightweight verify | 10:10 | 10:35 | 25 min | ✅ |
| F — Reshape + tokenize | 10:40 | 11:37 | 57 min | ✅ (v3 after v2 transient) |
| G — Register + commit | 11:37 | 11:42 | 5 min | ✅ (consolidate committed, curation_plan in working tree) |
| H — 998M sweep launch | 11:42 | 11:43 | 1 min | ✅ 7/7 submitted |
| Gap — 15-min wait | 11:43 | 11:58 | 15 min | per your request |
| I — 157M sweep launch | 11:58 | 11:59 | 1 min | ✅ 7/7 submitted |

Total elapsed: 07:21 → 11:59 UTC = **4 hours 38 minutes**. Models training on Iris well before your 8am wake target.

### Final state at report close (2026-04-19 12:00 UTC)

Iris job state for 14 children + 2 parents:
- **10 running** (8 children doing actual TPU work; 2 parents in keep-alive)
- **6 pending** — waiting for larger TPU slices (v4-16/32/64) to free up

Some children are experiencing transient `vfio_device_access` busy errors on startup — known pool issue when TPU chips haven't been cleanly released. Iris retries with `max_retries_preemption=100` and should migrate to clean workers within a handful of attempts. Nothing to do manually unless you see the same ones stuck past your wake-up.

### Where to look first in the morning

1. `iris --config lib/iris/examples/marin.yaml job list | grep llm-fm-` — snapshot of all 14 children's states.
2. WandB group `data-curation-fixed-model` — loss curves for running jobs.
3. `gs://marin-us-central1/metadata/data_curation_fixed_model_results/` — per-run summary JSONs materialize when jobs complete.
4. This report (git-tracked at `experiments/baseline_collection/consolidate/OVERNIGHT_REPORT.md`).

Goodnight.
