# Fast-curation production dashboard (CPU+TPU, multi-region, 7M-scalable)

Goal: a localhost dashboard — in the style of `experiments/baseline_collection/dashboard.py` —
to monitor the fast_curation cascade (Phase A CPU decode/fastText/tokenize → Phase B TPU
ModernBERT → Phase C CPU JustText) across regions, **designed from the get-go to scale to the
7M-WARC production run**. User's two headline needs:

1. **Predictability** — trustworthy per-run, per-phase ETAs ("done in ~6h", and the wall-clock date).
2. **Oversight of what's in flight** — every run × version × phase × region, at a glance.

## The scalability constraint (the design driver)

The reference dashboard derives progress by **listing GCS** — the central `_completed` registry
(one blob/WARC) and per-WARC output dirs. That is **O(WARCs)** per refresh. At 7M WARCs ×3 phases
that is ~21M list ops/refresh: minutes of latency and real GCS cost (~$35 per full registry list at
$0.005/1k Class-A ops). **Unusable at production scale.**

**Solution: worker-written heartbeats. The dashboard reads O(workers), never O(WARCs).**

## Telemetry layer (worker-side heartbeats)

Each worker writes a small rolling JSON heartbeat to a central, bounded-cardinality path:

```
gs://marin-us-central1/{subdir}/_heartbeats/{phase}/{region}-seed{N}.json
# subdir = documents/fast_curation/{spec_id}-{version}
```

Cardinality = number of workers (hundreds now, low thousands at production) — NOT WARCs.

### Heartbeat schema
```json
{
  "spec_id": "fastpipe_v3", "version": "6855733850",
  "phase": "a", "region": "us-east5", "seed": 12, "kind": "cpu",   // kind: cpu|tpu
  "started_at": <epoch>, "updated_at": <epoch>, "status": "running", // running|idle|draining|done
  "cumulative": {            // MONOTONIC across restarts (see below)
    "warcs_done": 47,
    "docs_in": 2600000, "docs_out": 740000,
    "wall_seconds": 8123.0,
    "compute_seconds": {"decode":.., "fasttext":.., "tokenize":.., "modernbert":.., "justext":..}
  },
  "recent": { "warc_wall_seconds": [.. last 20 ..], "last_warc": "abc123", "last_warc_at": <epoch> }
}
```

### Monotonicity across preemption (critical)
A preemptible worker (slot = `{region}-seed{N}`) restarts as a new process. On startup it **reads
its own prior heartbeat** and seeds `cumulative` from it, then keeps incrementing. So each slot's
`warcs_done` is monotonic, and `Σ slots warcs_done` = total done — accurate and O(workers). Central
claims guarantee each WARC is processed by exactly one slot, so no double-count. Worst case a crash
loses < 1 heartbeat-interval of counts; the registry remains the correctness backstop.

### Write cadence
On each WARC completion (and at least every 30s), plus a final write on graceful drain. ~1KB/write.
Over a 7M run that is 7M tiny writes total — negligible.

## Rate & ETA (predictability)

The dashboard keeps a short **in-memory time-series** of aggregate `warcs_done` per (run, phase).
- `rate_recent` = slope over a sliding window (e.g. last 15 min), robust and cheap.
- Cross-check: Σ over active workers of `1 / mean(recent.warc_wall_seconds)` = instantaneous capacity.
- **Per-phase ETA** = `(manifest_total − phase_done) / rate_recent`.
- **Run ETA** = `max` over phases of per-phase ETA (A→B→C pipeline → the bottleneck phase gates
  completion; max() captures it). Shown as both a duration and a wall-clock date.

## Worker liveness (in-flight oversight)
A slot is **active** if `updated_at` within a staleness window (5 min). Per (run, phase, region):
active vs total slots, recent rate, docs processed, compute-time mix. Stale slots flagged
(preempted/dead). This *is* the "what's in flight" view.

## Backend (Flask) — reuse reference structure
- Tabs: **Runs** (heartbeat-aggregated; replaces quick/deep GCS scans), **Cluster** (iris
  autoscaler — reuse), **Jobs** (iris job list filtered to `fastcur-*` + kill — reuse).
- **AUTO-refresh** the Runs tab (heartbeat reads are cheap, unlike the reference's behind-button
  scans) → live, no clicking. Optional on-demand "registry reconcile" (O(WARCs), clearly marked
  expensive) for ground-truth verification — never on the hot path.
- Endpoints: `/api/runs` (active runs + per-phase/region aggregates + ETA + sparkline series),
  `/api/cluster`, `/api/jobs`, `/api/jobs/kill`.
- Run discovery: list `documents/fast_curation/` (delimiter) for namespaces with recent
  heartbeats; `--specs` filters. So multiple concurrent runs/versions show side by side.

## Frontend (the experience) — dark theme + tabs, matching reference
1. **Run overview cards** (top): per run, big glanceable % complete, run ETA + wall-clock date,
   WARCs done/total, aggregate throughput, active-worker count.
2. **Per-phase columns A | B | C**: progress bar, rate (WARCs/hr), ETA, active workers (CPU for
   A/C, TPU for B), docs in→out, compute mix. Makes CPU vs TPU progress explicit.
3. **Per-region rows** within each phase: active workers, rate, done — regional balance.
4. **Pipeline funnel** A→B→C (presurvivors→keeplist→kept) showing the bottleneck.
5. **Throughput sparklines** from the in-memory series — trend + predictability.

## Build order
1. `telemetry.py` — `Heartbeat` helper (read-own-prior, accumulate, throttled write).
2. Wire into `cpu_phase.run_claim_loop` (A, C) and `tpu_phase.run_worker` (B). One helper, both sites.
3. Fix `tpu_phase._assert_ckpt_in_region` to use canonical `REGION_TO_DATA_BUCKET` (so
   region `europe-west4` ↔ bucket `marin-eu-west4` passes; current naive regex rejects it).
4. `fc_dashboard.py` backend + `fc_dashboard.html` frontend.
5. Relaunch all 3 regions (us-east5, us-east1, **europe-west4**) with heartbeat-instrumented
   workers + the region fix. Registry resume → no lost work.

## Status / notes
- v3 = `fastpipe_v3-6855733850` (cap 50MB + 60s timeout). us-east5 + us-east1 live and producing;
  europe-west4 failed first launch (wrong region string `eu-west4`; correct is `europe-west4`).
- The heartbeat relaunch is required (running workers predate the telemetry) — cheap registry resume.
