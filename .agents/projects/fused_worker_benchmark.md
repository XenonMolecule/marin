# Fused single-phase worker: fair benchmark vs two-phase (A+B)

Decide whether the merged worker (Phase A on the TPU host's CPUs, presurvivors in RAM,
chip fed directly) beats the current two-phase contract. The decision metric must charge
each architecture for everything it consumes, on the resource that actually constrains the
8M run: **held TPU slices**.

## Decision metric

**Steady-state WARCs per held-slice-hour, per chip type, all overheads included.**

Two-phase's CPU is only "free" while it bin-packs into someone's TPU hosts — the 100k run
showed it crowding B out of v5e/v6e host RAM, i.e. it *does* spend the constrained
resource. So the comparison is made at the host level:

- **Two-phase**: for a reference host (e.g. v6e-8: 8 chips, ~180 vCPU), compute achievable
  WARCs/host-hour from the measured primitives: B chip-seconds/WARC (chips) and A
  CPU-core-seconds/WARC (cores), with the host packing as many A workers as CPU/RAM allow
  next to its B worker. Fleet-pooling of A output is allowed (that is the design), so the
  arithmetic is aggregate: hosts_per_WARC/hour = chips-limited B + cores-limited A.
- **Merged**: measured directly — WARCs/host-hour of the fused worker, everything on-node.

Secondary metrics (diagnose *why* one wins):
- chip duty cycle: fraction of wall the chip is scoring (target >85% for merged;
  two-phase B fleet duty cycle = Σ catalog `wall_s` / Σ Iris B-task wall).
- per-stage seconds/WARC: decode / extract / fastText / tokenize / score / write.
- GCS traffic and ops per WARC (two-phase: ~36MB presurvivor written + read + deleted,
  measured 2026-08-29 over 300 files; merged: kept/ + catalog only).
- startup overhead: job start → first WARC done (includes XLA compile), amortized per shard.
- preemption cost: merged loses in-RAM presurvivors → re-decode; measure WARCs lost per
  preemption × observed preemption rate from the 100k run.
- RAM high-water on the merged worker (producer queue depth × ~36MB/WARC uncompressed×k).

## Fairness controls

1. **Same chip type per comparison cell.** v6e-8 primary (merged wants big-CPU single-host
   slices; pools were healthier than v6e-4 on 2026-08-29). Report v6e-4 as a second cell.
2. **Same spec semantics.** Fused runs the identical lpv11_fastpipe_v2_1 thresholds/models.
   Give it `storage_version=4` (distinct namespace hash — no presurvivors is a different
   storage contract; also prevents kept/ collisions with the live run).
3. **Exchangeable WARC sets.** Shards are sha256-hash partitions of the CC pool, so any two
   shard sets are statistically exchangeable samples. Use *disjoint virgin* shards
   (ids ≥ 404, untouched by the 100k ladder): ≥4 shards (~1000 WARCs) per arm, interleaved
   ids (e.g. two-phase 404,406,408,410; merged 405,407,409,411). Report mean ± bootstrap CI
   over per-WARC times. Both arms' kept output counts toward the 8M run — nothing wasted.
4. **Correctness gate before speed.** Run ONE shared shard through the fused worker into a
   scratch namespace and diff its kept parquets against the two-phase output of the same
   shard (same chip type; pipeline is deterministic per chip). Scores/doc sets must match
   before any timing claim.
5. **Steady state.** Exclude each worker's first WARC (compile) from per-WARC stats; report
   it separately under startup overhead.
6. **Same weather window.** Run both arms concurrently in the same region so preemption and
   GCS conditions are shared.

## What to capture from the CURRENTLY RUNNING two-phase 100k (already durable)

- `timing_a/data-{hash}.json` (regional buckets): per-WARC A stage seconds
  (decode/extract/fasttext/tokenize), wall, docs in/out. Survives the reaper.
- `catalog/shard-*.parquet` (central): per-WARC B wall_s, device, funnel counts. Survives.
- Iris controller DB (`iris query`, task_attempts): per-task start/finish → worker-hours,
  startup overheads, preemption counts. **Snapshot at end of run** (history can age out):
  A worker-hours, B slice-hours by device_variant, idle-exit waste.
- Presurvivor bytes/WARC: mean 36.0MB (n=300 sample, 2026-08-29) — reaper deletes the
  files, number recorded here.
- Fleet duty cycle at run end: Σ catalog wall_s vs Σ B task wall, per device_variant.

## Merged-worker implementation notes (for the experiment build)

- Producer/consumer inside one process: A pool (extract-procs sized to host cores minus
  infeed/XLA threads) feeding a bounded RAM queue; chip loop consumes. The queue bound is
  the RAM/backpressure knob.
- One claim, one sentinel, one catalog row per shard; no `_a_done`, no `_claims_a`, no
  reaper, no presurvivor storage.
- On preemption: shard reclaimed whole (in-RAM work lost); per-WARC done markers in the
  shard prefix keep the loss to in-flight WARCs only, same as today.

## Measured two-phase numbers from the 100k run (2026-08-30, fill in as run completes)

- **Run wall-clock**: first production A worker submitted 2026-08-29T09:11Z; first A shard
  sentinel 17:30Z same day; **finish 2026-09-01T03:31:16Z — 66.3h total** for 99,638 WARCs
  (404 shards). Includes two self-inflicted stalls (submit-concurrency wedge ~1h; the
  us-central2 v4 stranding + reprocess ~12h) and one infra event (east5 worker freeze at
  11:55Z Aug 31, ~4h) — the honest number for 8M extrapolation is the whole wall, since
  events like these are the norm at scale, not the exception.

- Phase A fleet total: 477 tasks, 4,640.9 worker-hours (8-cpu) for 100,364 WARCs
  = **166.5 worker-s/WARC all-in** (= 1,332 core-s/WARC) vs ~118 worker-s/WARC clean
  per-WARC time → **~29% overhead** (startup, claim/poll, idle-exit waste). Snapshot B
  slice-hours by device_variant the same way when B completes.

## First fused results (2026-09-01, shard 664, 230 WARCs, v6e-8, rung 2)

- **Parity**: presurv/WARC 22,689 vs 22,456 baseline (+1.0%); hi 0.1858 vs 0.1830; band
  0.3467 vs 0.3457; kept 0.3349 vs 0.3309 — all within shard noise; KEPT_V3 schema exact.
- **Throughput**: 26.2 slice-s/WARC ALL-IN (claim->sentinel wall), B median 23.3s, chip duty
  ~89% — vs two-phase v2.1's 65.5 all-in / 28.5 clean → **~2.5x WARCs per held slice**.
- Pilot kinks fixed en route: concurrent extractor unpack race (per-PID dest), 100GB OOM
  (queue-depth WARCs hold full decoded records; 350GB default), gigatoken CPU starvation at
  a_procs=11 (fleet runs a_procs=8). CC 1MiB-truncated docs crash-drop at ~1e-6 (same
  semantics as v2.1's pool).

## Shard granularity (lesson from the 100k run, 2026-08-29)

Shard claims are worker-exclusive, so Phase A latency per shard = shard_size ×
per-WARC wall on ONE worker (~8h for a 250-WARC shard on an 8-cpu A worker; a
1000-WARC 8M shard would be ~33h). Adding workers beyond #shards does nothing, and
the completion tail is long. Mitigations to weigh in the benchmark: (a) fused worker
throws a whole host's cores at its shard (v6e-8 host ≈ 180 vCPU → ~20x shorter shard
wall); (b) if two-phase survives, shrink shards (more, smaller) or add intra-shard
claim splitting for A.

## Verdict rule

Adopt the merged worker for the 8M run iff, at equal held-slice-hours on the primary cell
(v6e-8), it completes ≥10% more WARCs than the two-phase host-level number AND passes the
correctness gate. If within ±10%, prefer merged anyway for the operational simplification
(no cross-phase machinery, no reaper, no presurvivor storage bill) unless its preemption
loss rate is materially worse.
