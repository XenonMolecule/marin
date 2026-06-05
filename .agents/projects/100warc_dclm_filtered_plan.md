# 100-WARC llm_curated_dclm_filtered launch plan (in progress)

User went out for a few hours (started 2026-05-09 ~14:55 PT).
"Do this rigorously and correct. Fix issues as they come up. Do not take shortcuts."

## Pipeline state at start
- coord5 (`/michaelryan/subset-coord-llm-dclm-filtered5`) running 4 children:
  filter+tokenize for n=100, 500, 1000, 2000.
  All 4 in state=3 with Zephyr `zephyr-filter-subset-*` workers spawned.
- bff dedup output (200 shards) at
  `gs://marin-us-central1/deduped/bff_llm_curated_dclm_filtered_v1/`
  is the source for the subset filter.

## Edits already applied
- `experiments/baseline_collection/subset_one.py`: added `llm_curated_dclm_filtered`
  to METHOD_SPEC; fixed `from fray import ResourceConfig`.
- `experiments/baseline_collection/launch_subsets_iris.py`: added new method to
  METHOD_REGIONS; fixed `Constraint.create(... value=region)`.
- `.gitignore`: changed `experiments/distill/subsets/` →
  `experiments/distill/subsets/*` + `!experiments/distill/subsets/baseline_warcs_*.txt`
  so iris workspace bundle includes the manifests.
- `experiments/scaling_law_sweeps/warc_scaling_plan.py`: added
  `llm_curated_dclm_filtered` to `WARC_METHOD_BASE_NAMES`.

## Remaining steps (DO IN ORDER)

1. **Wait for n=100 cache.** Background wait loop `bxndsx8tp` is watching for
   `gs://marin-us-central1/tokenized/baseline_llm_curated_dclm_filtered_100warcs/train/.stats.json`.

2. **Validate cache.** Read .stats.json. Confirm `total_tokens` field exists
   and is >0. Sanity check: should be ~1.5B tokens (vs llm_curated_bos_fixed
   at 1.86B; dclm_filter+bff drops ~15-20%).

3. **Mirror to other regions.** Same regions as the 3000-WARC mirror:
   `us-central2` and `us-east5`.
   Use `gcloud storage rsync gs://marin-us-central1/tokenized/baseline_llm_curated_dclm_filtered_100warcs/ gs://marin-{region}/tokenized/baseline_llm_curated_dclm_filtered_100warcs/ -r`
   (no `cp -r ... /` traps).
   After each mirror: `gcloud storage ls -l ... | wc -l` count parity vs source.

4. **Update curation_plan.py.**
   ```python
   _D_OBS_DEFAULTS["baseline_llm_curated_dclm_filtered_100warcs"] = <TOKEN_COUNT>
   "llm_curated_dclm_filtered_100": _method(
       "llm_curated_dclm_filtered_100",
       "baseline_llm_curated_dclm_filtered_100warcs",
       sampled_warcs=100,
   ),
   ```
   Also do for _500, _1000, _2000 *as soon as those caches land* (don't wait
   for all four — n=100 unblocks training).

5. **Dry-run the WARC launcher** to verify cell enumeration:
   ```
   uv run python experiments/scaling_law_sweeps/launch_warc_scaling_sweep.py \
       --methods llm_curated_dclm_filtered --n-warcs 100 \
       --extension-methods llm_curated_dclm_filtered --dry-run
   ```
   Expected: 18 cells = 2 hidden_sizes (256, 512) × (7 base + 2 extension budgets).

6. **Launch training.** Submit via Iris (CPU-only parent), batch priority for
   children. Use the same wandb-group / tracker-prefix / results-prefix as the
   FM sweep — those are class-level defaults in `launch_warc_scaling_sweep.py`.

7. **Verify decreasing loss.** Pick the smallest-budget cell, tail its wandb
   stream until 100+ steps land. If loss is going down, declare success and
   ramp wakeup interval back to 30+ min.

## Issues that have come up so far
- `from fray.v2.types import ResourceConfig` → moved to `from fray import ResourceConfig`
- `Constraint(key=..., value=...)` → must use `Constraint.create(key=..., op=..., value=...)`
- Subset manifest dir was gitignored, so iris bundle excluded it. Allow-listed `baseline_warcs_*.txt`.

## What NOT to do
- Don't subset shard counts. Pipeline runs over all 200 shards of the bff_dedup output.
- Don't mirror cross-region beyond the 3 known compute regions ($ matters).
- Don't launch training before mirroring — region-locked workers will fail.
- Don't claim success until at least one cell is showing a loss curve below init.
