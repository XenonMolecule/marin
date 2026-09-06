# 8M-WARC lpv11_fastpipe_v2_1 run — architecture plan (draft 2026-08-27)

Scaling the TEXT-line cascade (`lpv11_fastpipe_v2`, proven at 10,364 WARCs on 2026-08-27) to ~8M
WARCs. Grounded in measured rates: A=118 worker-s/WARC (8-cpu), B=31.5 s/WARC (4-chip slice),
~20MB/WARC final storage with the compressed schema. Timeline at realistic capacity (200 held
slices + right-sized CPU fleet): **~3 weeks end-to-end**; fixed bill ≈ 314k CPU-worker-hours +
78k slice-hours + ~$2.7k/mo storage during the active weeks (Coldline after).

## 1. Work-list: shards replace the flat manifest

The 10k run used a text manifest every worker loaded and shuffled, with one claim + one registry
object per WARC, and workers re-LISTing the registry each pass. At 8M that is an ~1GB manifest and
O(8M)-object listings per worker pass — the exact class-B explosion of the Jul-Aug incident.

Replace with a two-level layout:

- **Shard = the unit of claim and accounting.** Partition the 8M WARCs into ~8,000 shards of ~1,000
  WARCs each, deterministically: `shard = sha256(warc_path) % 8000` (pin the modulus — partition
  keys are frozen across resumes, per the standing rule). Shard membership is materialized once as
  8,000 parquet files (`shards/shard-{s:05d}.parquet`: warc_path, warc_hash), built from Common
  Crawl's published `warc.paths.gz` listings for the chosen snapshots. No worker ever loads the
  full 8M list.
- **A worker claims a SHARD**, then processes its ~1,000 WARCs with per-WARC done markers *inside
  the shard's own prefix* (`.../shard-{s}/_done/data-{h}` — only the shard owner lists it, ~1k
  objects). Preemption resume = re-claim shard, skip marked WARCs, resume banked chunks. Claim
  traffic drops from 8M×fleet to 8k total; steal/stale logic carries over at shard granularity.
- **Shard-done sentinel** when all of a shard's WARCs are done. Fleet-wide progress = LIST of 8k
  sentinels. B, the reaper, and the monitor read shard sentinels, never per-WARC registries.

## 2. Registry → catalog (the lookup)

Registry markers stop being empty files: a WARC's B-done marker records **where the output lives**
(`{region, kept_path, n_kept, n_presurvivors, device, ts}` as the marker's content). A **compactor**
(small looping job, like the reaper) rolls markers up into per-shard catalog parquets
(`catalog/shard-{s}.parquet`, one row per WARC) and deletes the compacted markers. The catalog is
the queryable index for: the dashboard, the reaper, downstream consolidation, dedup provenance, and
"where is WARC X?" lookups — no multi-region listing ever again. End state: 8k catalog files ≈ the
run's entire metadata, ~2GB.

## 3. Phase decoupling (the biggest operational lesson)

Run **A ahead, B trailing** rather than coupled fleets:

- A is CPU-abundant and stockout-immune: ~1,500 8-cpu workers finish all 8M in ~1 week. Requires
  the ops PR adding a right-sized CPU scale group (n2-highmem-8 class) — today big-CPU workers can
  only bin-pack onto TPU hosts (the on-demand group is e2-highmem-2, too small), which is what
  capped us at ~81 effective workers and caused the A/B host-memory contention (64GB vs 60.2GB).
- **Spread A across regions instead of mirroring.** Blanket presurvivor mirroring would be ~440TB
  of copies (~$8.8k same-continent egress + doubled transient storage) — rejected. Instead, shards
  are assigned round-robin across 3-4 regions at A time (A is region-agnostic: CC ingress is free
  everywhere, models are mirrored once), so each region's B scores its LOCAL share and no single
  stockout strands more than ~a quarter of the corpus. The fallback for a genuinely dry region is
  **on-demand rescue re-extraction** (production-proven this week): re-decode the stuck share into
  a TPU-healthy region for $0 — CPU time only, which is the abundant free resource. Dollars-optimal
  even though it spends calendar when weather is bad; the multi-region spread bounds how much.
  B grazes ANY healthy pool — v6e/v5p/v4/v5e across five regions, all generations measured.
- B fleets: reserved v4 baseline (stockout-immune) + preemptible everything else, 48GB memory,
  short claim-stale (0.2h), long endgame drain, device attribution already in timing records.

## 4. Storage (measured plan)

- Schema v3: **drop `input_ids`** (42% of bytes; B re-tokenizes from `text` via gigatoken arrow for
  ~4% of B wall) and write text at **zstd level 12** (-21% for 5s/WARC CPU). Kept ≈ 20MB/WARC →
  ~160TB active (~$2.7k/mo standard — shrinks further after dedup), lifecycle to Coldline when the
  downstream has consumed it (~$530/mo).
- **Reaper loops hourly** behind B (incremental, verify-gated, namespace-guarded — shipped and
  proven on the 10k run). Presurvivor float between passes ≈ $7/day. Sharding makes it cheaper
  still: reap per shard-done sentinel.
- Tombstones stay (re-thresholding audit); kept rows carry all three classifier scores.

## 5. Spec & namespace

`lpv11_fastpipe_v2_1`: same models/thresholds as v2 (identical corpus semantics), new namespace for
the sharded on-disk contract + schema change. The 10k v2 corpus stays untouched as the reference.
Hash-pin in test_spec.py as usual.

## 6. Remaining pre-launch checklist

| item | status |
|---|---|
| Reaper (verify-gated, incremental) | DONE, proven on 10k |
| Device-attributed B timing | DONE |
| CPU scale group ops PR (n2-highmem-8/16) | TODO — gates A at >100 workers |
| Sharded work-list + shard claims + catalog compactor | TODO — the main build |
| Schema v3 (drop input_ids, zstd-12) + B tokenize-on-read | TODO (small) |
| Multi-region shard assignment in A (round-robin) | TODO (small) |
| Endgame auto-finisher (shard-level reassignment) | TODO (small; manual twice now) |
| CC download behavior ≥600 workers | validate during shakeout |
| Reserved TPU ask (~200 slices baseline) | user/capacity decision |
| **100k-WARC shakeout run** of all of the above | before the 8M button |

## 7. Risks

- Regional TPU stockouts: mitigated by multi-gen/multi-region B + the A-side regional spread, with
  on-demand rescue as the $0 fallback (measured: the 10k survived two of them); residual risk only
  to the calendar, not correctness.
- CC throttling at fleet width: retries/backoff exist; stagger worker starts; untested past ~100
  concurrent downloads.
- GCS class-B budget: sharding + catalog reduce ops by ~10^3 vs the 10k design; compactor keeps
  marker counts bounded.
- Scores of A-dropped docs are not recorded (destructive gate, by design, threshold 0.0048 keeps
  ~81% so little is lost); `--preserve-dropped-scores` sidecar exists if that ever changes.
