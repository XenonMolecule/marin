# ModernBERT→Levanter migration — launch tracker

Live scratchpad for jobs launched during the migration. Check this when resuming / working M4 in
parallel. Plan: `~/.claude/plans/okay-yeah-can-we-staged-galaxy.md`. Memory:
`project_modernbert_levanter_migration`.

## Code state (all local, NOT committed)
- **levanter**: `models/modernbert.py` (classifier + `load_hf_sequence_classifier`), `layers/attention.py`
  (bidirectional window), `main/train_classifier.py` (entrypoint + data + train core), `tests/`.
- **marin**: `lib/marin/.../training/training.py` → `TrainClassifierOnPodConfig` + `run_levanter_train_classifier`.
- **launcher**: `experiments/baseline_collection/launch_modernbert_levanter.py` (coordinator; `--smoke`, `--no-submit`).
- **tests green (local)**: levanter modernbert 8/8 (incl. HF roundtrip oracle rtol 1e-4), marin offline Trainer 1/1.

## Launched jobs

| date | run-id | what | region/tpu | status | where to look |
|---|---|---|---|---|---|
| 2026-06-24 15:31 | mb-clf-smoke-e5a | SMOKE: 1024 ctx, 1024 rows, 32 steps, VANILLA, warm-start | us-east5 / v6e-4 | RUNNING — child TPU job `train_classifier` allocated+Trainer up by 15:33; compiling/training. Launch path (b) validated: region-check passed, current_client→Iris backend, Fray submit OK | coord+child logs: `iris --cluster marin job logs /michaelryan/mb-clf-smoke-coord`; wandb: https://wandb.ai/marin-community/modernbert-useful/runs/mb-clf-smoke-e5a ; ckpt: `gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-smoke-e5a/` |

| 2026-06-24 15:47 | mb-clf-50k-surv-e5 | 50k SURVIVOR, 8192, SPLASH, batch256/pdp=-1 | us-east5 / v6e-4 | **FAILED @ ~15min — HBM OOM** (not a SPLASH bug). pdp=-1 → per-device batch 64 @ 8192 → QKV/MLP activations need 186G > 32G/chip. **SPLASH kernel COMPILED+RAN (splash_mha_fwd_segmented_residuals in HLO) → SPLASH@8192 works.** Fix: per_device_parallelism=2 (microbatch, grad-accum to 256). | — |
| 2026-06-24 16:06 | mb-clf-50k-surv-e5b | 50k SURVIVOR, 8192, SPLASH, batch256 **pdp=2** (microbatch 8, accum 32) | us-east5 / v6e-4 | LAUNCHED — coordinator `/michaelryan/mb-clf-50k-coord-b` (OOM-fix relaunch) | logs: `iris --cluster marin job logs /michaelryan/mb-clf-50k-coord-b/train_classifier`; wandb: https://wandb.ai/marin-community/modernbert-useful/runs/mb-clf-50k-surv-e5b ; ckpt: `gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-50k-surv-e5b/` |

| 2026-06-24 16:07 | mb-clf-splashcheck | M4 equivalence @2048 | (CPU, no mesh) | **FAILED — harness bug, not SPLASH**: launched as CPU job (no --tpu) + script lacked a JAX mesh → "Splash requires non-empty mesh" + 4GB container OOM. Fixed: script now builds a `Mesh(devices,("data",))` + batch sharded; relaunch as TPU job. | — |
| 2026-06-24 16:17 | mb-clf-splashcheck-b | M4 equivalence: VANILLA vs SPLASH logits + throughput @ **2048**, batch 4 (mesh-fixed) | us-east5 / v6e-4 (--tpu) | LAUNCHED `/michaelryan/mb-clf-splashcheck-b` | logs: `iris --cluster marin job logs /michaelryan/mb-clf-splashcheck-b/0`; grep `[correctness]`/`[throughput]`/`PASS`/`FAIL`. TPU job flags: `--tpu v6e-4 --enable-extra-resources --extra tpu --memory 32GB`. |

Canonical job-name note: `iris job logs` needs `/michaelryan/<job>` (leading /user/), not the bare name.
Memory-config note: at 8192 ctx, `per_device_parallelism` MUST be small (2) — default `-1` makes per-device batch = batch/num_devices = 64 → HBM OOM (186G). Launcher default is now 2.

| 2026-06-24 16:20 | mb-clf-50k-surv-e5b | (update) | us-east5 / v6e-4 | **SPLASH@8192 TRAINS** — step 0 done in 97.3s, stepping 1/195. Run is SLOW (~hours at 8192; lower-ctx M5 runs far faster). | wandb mb-clf-50k-surv-e5b |
| 2026-06-24 16:19 | mb-clf-splashcheck-b | M4 equiv @2048 (fp32-vanilla vs bf16-splash) | us-east5 / v6e-4 | **"FAIL" max_abs=0.027 — but it's PRECISION, not a bug**: compared fp32-vanilla vs bf16-splash. CPU check shows the bf16 FLOOR alone is 0.070 (>0.027). A masking bug would be orders larger. Fixed test → matched-precision. | — |
| 2026-06-24 16:26 | mb-clf-splashcheck-c | M4 equiv MATCHED bf16 | us-east5 / v6e-4 | **✅ M4 PASS** — backend_diff=7.03e-2, bf16_floor=9.64e-2, **ratio=0.73 (<1!)** → SPLASH==VANILLA within bf16 noise, NOT a masking bug. Throughput@2048: splash 0.59× (slower — flash overhead only pays at long ctx; 8192 is where VANILLA OOMs). | done |
| 2026-06-24 16:33 | mb-clf-ctx1024/2048/4096-surv-e5 | **M5 SURVIVOR CONTEXT SWEEP** — 50k survivor, SPLASH, ctx∈{1024,2048,4096} (8192 point = e5b) | us-east5 / v6e-4 ×3 | LAUNCHED — coords `/michaelryan/mb-clf-ctx{1024,2048,4096}-coord` | wandb runs mb-clf-ctx{N}-surv-e5; collect best_f1 per ctx → F1-vs-context curve |

**🐞 BUG FOUND+FIXED (loss=0 root cause):** raw `useful_cascade_survivors/parts` shards are CLASS-ORDERED
(surv_00300: useful lines 1–3856, then no_useful) and the reader read in order with NO shuffle →
class-homogeneous microbatches → degenerate gradients → loss→0. **Fix (two parts):** (1) train on the
PRE-SHUFFLED, IN-REGION preshard `gs://marin-us-east5/.../presharded_survivor_1M_w4/train_shard_*` (mixed
~20% useful, zero cross-region egress, = torch 0.705 parity data); (2) `train_dataset.shuffle(seed)` in
train_classifier. Stopped the 4 degenerate runs (ctx{1024,2048,4096}-surv-e5 + 50k-surv-e5b).

| 2026-06-24 16:50 | mb-clf-ctx{1024,2048,4096,8192}-surv-f | **M5 sweep, FIXED data+shuffle** — 50k preshard survivor, SPLASH, ctx∈{1024,2048,4096,8192} | us-east5 / v6e-4 ×4 | LAUNCHED — coords `/michaelryan/mb-clf-ctx{N}-coord-f` | wandb mb-clf-ctx{N}-surv-f. EARLY SANITY: train/loss should now be a NORMAL CE value (~0.1–0.6), NOT ~0. Then eval/best_f1 ≈0.6–0.7. |

**✅ DATA FIX VALIDATED (17:07):** all 4 -f runs show NORMAL train/loss ~0.43–0.53 (was degenerate ~0).
Mixed-class microbatches → real training. Stepping cleanly (ctx1024 @37, 2048 @43, 4096 @30, 8192 @18 of 195).
Now running to completion for eval F1. Slow start per run = ~8min cross-region eval-read + warm-start + SPLASH compile.

### F1-vs-context curve @ 50k survivor (survivor 1M baseline=0.705) — EVAL FIX #2 CONFIRMED (F1s logged)
| ctx | best_f1 | threshold |
|---|---|---|
| 1024 | 0.238 | 0.22 |
| 2048 | 0.327 | 0.14 |
| 4096 | 0.451 | 0.34 |
| 8192 | **0.564** | (ctx8192-50k) |

**✅ 50k CONTEXT CURVE COMPLETE: 0.238 → 0.327 → 0.451 → 0.564 (1024→8192).** Clean monotonic; 8192 at
only 50k docs already nears the 1M baseline 0.705. Context is a strong lever (truncation of long docs).

### ✅ @200k F1-vs-context (data-scaling check; first CLEAN end-to-end runs — HF-save fix VERIFIED, state=succeeded)
| ctx | 200k F1 | 50k F1 | Δ(more data) |
|---|---|---|---|
| 1024 | **0.355** (t=0.22) | 0.238 | +0.117 |
| 2048 | **0.441** (t=0.30) | 0.327 | +0.114 |
| 4096 | **0.546** (t=0.32) | 0.451 | +0.095 |
| 8192 | **0.645** (t=0.40) | 0.564 | +0.081 |

@200k curve COMPLETE — all 4 SUCCEEDED. Monotonic in ctx; 200k>50k everywhere; data-scaling gain larger
at short ctx (more data helps most where truncation hurts). 8/8 sweep runs saved cleanly (HF-save verified).

## 🆕 5M/10M RANDOM SURVIVOR DATASET PIPELINE (user "go" — random-3k, unbiased)
Random-3000 pool `gs://marin-us-central2/raw/commoncrawl/baseline_3000_random-34884d` is teacher-extracted
(its WARCs are in done_warcs_high_quality = the 10,364 extracted) + HTML ready → ~48M potential survivors
(ample for 5M/10M, unbiased random sample of the 10k draw). Retargeted `build_hq_distill_dataset.py` with
ADDITIVE flags (`--html-dir`, `--restrict-to-html-dir`; defaults unchanged, lint+parse OK). PIPELINE:
1. assemble `--html-dir <random> --restrict-to-html-dir --tag random` (us-central1; reads extraction, writes
   ~40GB metadata→central2 one-time) → htmljoin `--html-dir <random> --tag random` (us-central2) →
   negatives `--html-dir <random> --tag random` (us-central2) = dataset high_quality_3000_distill_random.
2. cascade_survivor_filter random, target 5M, **WARC-disjoint from frozen test** (us-central2).
3. mirror survivors → us-east5; 4. train 5M then 10M via launch_modernbert_levanter.
**LAUNCHED 22:44:** smoke assemble `hq-distill-rand-assemble-smoke` (10 WARCs, tag random_smoke) to validate
retargeting before the full assemble. Coord: `/michaelryan/hq-distill-rand-assemble-smoke`.

**HF-SAVE FIX VERIFIED (20:5x):** mb-clf-200k-surv-f + mb-clf-200k-ctx4096-surv-f both state=SUCCEEDED — first
fully clean train→eval→save runs (all 3 mesh bugs resolved). Data scaling works (200k > 50k at matching
ctx); context trend holds. 200k@8192=0.645 on trajectory toward 1M baseline 0.705.

**MULTI-HOST: v6e-16 QUOTA-BLOCKED** (19:35) — 4-host gang stuck pending: "No workers match constraints
device-type/variant/region; autoscaler tier_blocked: quota-pool tier monotonicity". Single-host v6e-4
schedules fine; larger gangs blocked. Stopped the v6e-16 test. Single-host 1M (mb-clf-1M-surv-f) continues
(~10h). Probing v6e-8 (2 hosts, ~5h) to see if a smaller gang clears the quota wall.
→ v6e-8 ALSO pending ("Insufficient TPUs need 8 available 0; tier_blocked quota-pool tier monotonicity").
CONCLUSION: ANY multi-host gang (>v6e-4) is quota-blocked in us-east5 now. ~3h needs a QUOTA BUMP for the
larger v6e pool (admin action) — or a different region (but data is in us-east5 → would need re-mirror,
cross-region). Stopped both probes. **Single-host v6e-4 1M (~10h) is the available path; it continues.**
Decision for user: request quota bump for multi-host, or accept single-host ~10h.
→ v6e-32 (8 hosts) ALSO tier_blocked. ALL multi-host gangs blocked. LEFT the v6e-32 (coord
mb-clf-50k-mh32-coord, run mb-clf-50k-mh32) PENDING as a lottery ticket (costs nothing; user says big
allocs sometimes land). Heartbeat watches pending→running; if it lands, move 1M onto it (~2h) + stop the
single-host 1M. If still blocked after ~2 cycles, clear it.

**Result reading:** clean MONOTONIC rise with context. Low absolute at short ctx is EXPECTED: each run
evals test docs TRUNCATED to its own max_seq_len, and these are long web pages — ctx1024 sees only the
first 1024 tokens. So the sweep measures "how much does seeing more of the doc help" → context helps a
lot. ctx8192 (full docs) should be highest / closest to baseline.

**🐞 EVAL BUG #3 (HF save, fixed 18:30):** runs logged F1 then crashed at converter.save_pretrained
(in main(), OUTSIDE Trainer mesh → "with_sharding_constraint requires non-empty mesh"). SAME root cause.
FIX: HF save moved INSIDE the Trainer block (train_classifier now takes hf_save_path). Lint+offline pass.
The 50k sweep F1s were logged BEFORE this crash → results safe. (3 mesh-context bugs total: eval, save,
+ the score_texts own-mesh attempt — all "operate on Trainer-sharded model only inside its mesh".)

**Non-blocking optimization for user:** each run re-reads the 35-shard frozen eval set cross-region
(us-central2→us-east5, ~8min). Mirroring full_prep_body_strip/test to us-east5 once would remove this.

**🐞 EVAL-MESH BUG (found 17:27, fixed 17:50):** ctx1024/2048/4096-surv-f all FAILED at step 194 —
trained fine (loss ~0.4) but crashed at the post-training eval. Cause: score_texts ran the SPLASH
model via named_jit OUTSIDE the Trainer's mesh context ("Splash requires non-empty mesh") — same class
of bug as the splash-check harness. CPU smoke missed it (VANILLA needs no mesh). FIX: score_texts now
builds its own Mesh(devices,("data",)) + pads each chunk to batch_size + shards batch:data (CPU-sanity OK:
probs finite, f1_sweep works). Lint clean.

| 2026-06-24 17:50 | mb-clf-200k-surv-f | **200k PARITY** (user ASAP), 8192, SPLASH, fixed eval | us-east5/v6e-4 | LAUNCHED `/michaelryan/mb-clf-200k-coord` | wandb mb-clf-200k-surv-f |
| 2026-06-24 17:50 | mb-clf-1M-surv-f | **1M PARITY** (user ASAP), 8192, SPLASH, fixed eval — target 0.705 | us-east5/v6e-4 | LAUNCHED `/michaelryan/mb-clf-1M-coord` (long, ~3906 steps) | wandb mb-clf-1M-surv-f |
| 2026-06-24 17:51 | mb-clf-ctx{1024,2048,4096}-surv-f | M5 sweep RESUME (fixed eval), coords -f2 | us-east5/v6e-4 ×3 | RELAUNCHED `/michaelryan/mb-clf-ctx{N}-coord-f2` (resume from ~step194 ckpt → eval) | wandb mb-clf-ctx{N}-surv-f |

| 2026-06-24 17:55 | mb-clf-200k-ctx{1024,2048,4096}-surv-f | **200k CONTEXT SWEEP** (user wants ctx sweep at 200k too; 8192@200k = mb-clf-200k-surv-f) | us-east5/v6e-4 ×3 | LAUNCHED `/michaelryan/mb-clf-200k-ctx{N}-coord` | wandb mb-clf-200k-ctx{N}-surv-f |

**CAPACITY GAUGE:** all 6 earlier jobs got TPU workers (no queueing) → ≥6 concurrent v6e-4 confirmed. Using whether the 200k-sweep jobs schedule promptly as the test for whether to ALSO launch the 1M context sweep (4 more). If they queue → capacity ~limited, hold 1M sweep.
**EVAL-READ EGRESS:** with ~10 runs each re-reading the 35-shard frozen test cross-region (~8min + egress each), mirroring full_prep_body_strip/test → us-east5 ONCE would cut repeated egress. DO THIS before launching the 1M context sweep (4 more runs).

**🐞 EVAL BUG #2 (device-order, fixed 18:05):** the first eval fix (score_texts making its OWN
Mesh(devices,("data",)) = device order [0,1,2,3]) crashed because the model is sharded on the
TRAINER's mesh (device order [0,1,3,2]) → "Received incompatible devices" in jit. REAL FIX: run the
eval INSIDE the `with Trainer` block (same mesh the model lives on); score_texts no longer makes its
own mesh. Lint+offline pass. Stopped all 6 broken-code runs; relaunched with fix (coords -f3):
- 50k sweep ctx{1024,2048,4096,8192}-surv-f (resume step-194 → eval = FAST CANARY for the fix)
- 200k parity mb-clf-200k-surv-f, 1M parity mb-clf-1M-surv-f
**200k CONTEXT SWEEP (3 jobs) is HELD** until the canary confirms eval works (don't relaunch the
sweep onto a 3rd unverified eval). 1M context sweep also pending capacity + canary.
Lesson: eval/inference of a Trainer-sharded model must use the Trainer's mesh, not a fresh one.

**🚨 CROSS-REGION FIX (18:18) — user's #1 cost driver:** the eval set (full_prep_body_strip/test =
10.2GB / 35 shards) was in us-central2, so EVERY run read 10.2GB cross-region in build_eval. Across
~15 runs today ≈ 100-150GB inter-region reads (~$2-3 at GCP US inter-region ~$0.02/GB). A us-east5
MIRROR of the test set ALREADY EXISTED (gs://marin-us-east5/classifiers/useful_fasttext/full_prep_body_strip/test,
35 shards) → fixed by pointing TEST_GLOB there (no copy needed). NOW: TRAIN_GLOB + TEST_GLOB BOTH
us-east5 → future runs do ZERO cross-region reads. The 6 currently-running runs already paid their
one-time read (build_eval runs once per run; not re-read). Frozen-7000 sample is deterministic +
shard names identical → F1 still comparable to 0.705.
(Further speed-only optimization, deferred: pre-sample the frozen 7000 to a tiny in-region file so
runs don't read 10GB in-region each — but cost is now solved.)

**🧪 MULTI-HOST TEST (19:29):** mb-clf-50k-mh (coord mb-clf-50k-mh-coord) — 50k @ 8192 SPLASH on
**v6e-16 (4 hosts / 16 chips)** to see if we can push the 1M from ~10h → ~3h. RISKY: multi-host gang
scheduling historically FAILED for this model (gang-coschedule / broadcast hangs / requeue). Watch:
(1) does iris gang-schedule 4 hosts (capacity)? (2) jax.distributed init across hosts OK (vs single-host
'skipping')? (3) step rate vs single-host 9.1s/step (want ~3-4×)? (4) eval+save clean? If all good →
launch 1M on v6e-16. If gang/distributed fails → stick with single-host 1M (already running, ~10h).

**⏱️ THROUGHPUT @8192 (measured 18:48, steady-state):** Levanter SPLASH = **~9.1 s/opt-step** (256 docs/step)
= ~28 docs/sec. → 200k ≈ 2.0h, 1M ≈ 9.9h. Torch_xla baseline: 1M took ~2 days (~48h) = ~44 s/opt-step
→ Levanter ~4.8× wall-clock. CAVEAT: torch's 48h included heavy preemption/restart thrash (~14min/restart,
preempt every 15-40min) + poison-batch stalls, so the PURE kernel+framework speedup is likely less (~2-3×);
the rest is reliability (Levanter resumes cleaner). NOT an isolated same-precision kernel benchmark. Also:
SPLASH is SLOWER than VANILLA at short ctx (2048 bench = 0.59×) — its win is enabling/accelerating LONG ctx.

**⚠️ PERF BUG — data-loader starvation at short ctx (19:55):** TextClassificationDataset tokenizes 256
docs PER BATCH on the host; at short ctx the TPU step is ~1s so the loader starves → "Data loader stalled
~240s" → effective ~50-72 s/it. Affects 200k ctx1024/2048 badly, ctx4096 intermittently; ctx8192 + the 1M
(8192) are FINE (slow compute hides it). The 200k sweep will finish (valid F1s) but slowly (hours).
DECISION: do NOT churn/relaunch — the 1M headline is unaffected and the 50k curve is already complete. Real
fix = pre-tokenize once (or parallelize the loader); needs profiling to confirm the 240s stall root cause
(240s >> raw tokenize time, so may be loader-async interaction, not just tokenize speed). Deferred.
v6e-32 lottery CLEARED (tier-blocked ~17min). Multi-host needs a quota bump (admin).

**RELAUNCHED with BOTH mesh fixes (18:32, coords -f4):** mb-clf-200k-surv-f (200k@8192 parity),
mb-clf-1M-surv-f (1M@8192 parity, HEADLINE→0.705), mb-clf-200k-ctx{1024,2048,4096}-surv-f (200k context
sweep). These should now FINISH cleanly (state=finished) with saved HF models + logged F1. ctx8192-50k
still running (gives 50k 8192 point; may crash at HF-save on old code but F1 is kept). Fleet ≈6 v6e-4.

**🛡️ REGION GUARD added (18:24):** `launch_modernbert_levanter.assert_data_in_region({train,test globs}, region)`
runs at launch (before submit) and RAISES if any gs:// data path isn't in the run region (uses
`rigging.filesystem.region_from_prefix`). Verified: passes for in-region globs, fails loudly for a
us-central2 test glob. So an accidental cross-region data read can no longer be launched.
**NOTE:** sweep runs are 50k docs (small, fast signal); F1 will be BELOW the 1M=0.705 baseline (general 50k≈0.611). Sweep measures CONTEXT effect at fixed 50k. The 200k→1M runs are the real parity push toward 0.705.

**SMOKE RESULT (mb-clf-smoke-e5a):** path VALIDATED — warm-start load (ModernBERT-base from HF) ✓,
data ✓, compile ✓, **step 0 done in 77.3s on v6e-4** (VANILLA, 1024 ctx). Training loop executes
end-to-end on TPU. (Smoke used full_prep data — path test only, F1 not meaningful.)

## Ready-to-run launch commands

**Smoke (1024 ctx, 1024 rows, ~32 steps, VANILLA, warm-start) — validates the path:**
```
uv run iris --cluster marin job run --region us-east5 \
  --cpu 4 --memory 8GB --disk 10GB --priority interactive --no-wait \
  --job-name mb-clf-smoke-coord \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.baseline_collection.launch_modernbert_levanter \
       --run-id mb-clf-smoke-e5a --smoke --tpu-type v6e-4 --region us-east5
```

**200k parity (8192 ctx, SPLASH) — after smoke is green:**
```
... -- python -m experiments.baseline_collection.launch_modernbert_levanter \
       --run-id mb-clf-200k-e5 --train-rows 200000 --max-seq-len 8192 \
       --attn-backend splash --tpu-type v6e-4 --region us-east5
```
Target F1 >= 0.644 (torch 200k baseline). Then `--run-id mb-clf-1M --train-rows 1000000` → >= 0.667.

## Open risks / watch-items
- Launch mechanics (coordinator→Fray submit of TPU job) not yet validated end-to-end on cluster.
- warm-start `load_pretrained(ModernBERT-base)` on TPU + sharding: untested at scale.
- VANILLA is O(seq^2) → only safe at short ctx; 8192 REQUIRES `--attn-backend splash` (validate in M4).
- Monitor checkpoint/step ADVANCEMENT (wandb summary._step + GCS checkpoints), not just job state
  ([[feedback_monitor_step_advancement_not_job_state]]).
- Output checkpoints: `gs://marin-us-east5/checkpoints/modernbert-useful/<run-id>/`.

## Autonomous gating sequence to M5 (do in order; no shortcuts)
1. **50k @ 8192 SPLASH runs** (mb-clf-50k-surv-e5) — first train step completing = SPLASH compiles
   at 8192 (VANILLA can't do 8192, so this is the gate). [IN FLIGHT]
2. **M4 equivalence** — once (1) compiles, launch `modernbert_splash_check.py` on TPU:
   `... -- python -m experiments.baseline_collection.modernbert_splash_check --max-seq-len 8192 --batch 8`
   Must PASS (splash logits == vanilla, rtol/atol 1e-3) BEFORE trusting any SPLASH result. Also gives
   the 8192 throughput speedup number. Script CPU-validated (vanilla-vs-vanilla self-consistency OK).
3. **M5 context sweep on SURVIVOR** — only after M4 PASS: launch the launcher at ctx ∈ {1024,2048,4096}
   (8192 = the 50k run / scale-up) at a fixed train size (start 50k for speed), e.g.
   `... --run-id mb-clf-ctx{N}-surv-e5 --train-rows 50000 --max-seq-len {N} --attn-backend splash ...`
   Collect best_f1 per ctx → F1-vs-context curve (the queued experiment, never run before).
4. **M3 scale-up** (parallel): 100k/200k/1M survivor @ 8192 SPLASH (run-ids mb-clf-{100k,200k,1M}-surv-e5).
   Baseline to beat: 200k(general)=0.644, 1M(survivor)=0.705.

M4 script: `experiments/baseline_collection/modernbert_splash_check.py` (correctness + throughput,
configurable --backend-a/--backend-b; splash needs TPU).

---
## 2026-06-24 22:58 heartbeat — 1M parity + 5M/10M random data pipeline

**A) 1M parity (mb-clf-1M-surv-f / coord mb-clf-1M-coord-f4):** HEALTHY, advancing.
- ~1.45k/3.91k, 9.3 s/it, loss ~0.24, ~6.3h remaining → headline parity vs torch 0.705.
- Was PREEMPTED ~05:42 (new worker ...20260625-0304-...worker-0); auto-resumed from 15-min
  checkpoint (~step 1.36k), recompiled, back to 9.3 s/it. Preempt→auto-resume working as designed.
  NOT a stall. Context sweeps DONE: 50k 0.238/0.327/0.451/0.564; 200k 0.355/0.441/0.546/0.645.

**B) 5M/10M RANDOM survivor data pipeline:**
- Smoke assemble `hq-distill-rand-assemble-smoke` (10 random WARCs, --restrict-to-html-dir):
  **SUCCEEDED** — 10 metadata shards written to
  `gs://marin-us-central2/datasets/high_quality_3000_distill_random_smoke/_staging_metadata/`.
  Retargeting at random pool `baseline_3000_random-34884d` VALIDATED.
- FULL assemble launched: `/michaelryan/hq-distill-rand-assemble` (us-central1, all ~3000 random
  WARCs, --restrict-to-html-dir --tag random). Output:
  `gs://marin-us-central2/datasets/high_quality_3000_distill_random/`.
- NEXT stages (drive in sequence as each completes): htmljoin → negatives (us-central2) →
  cascade_survivor_filter (random, target 5M, WARC-disjoint from frozen test) → mirror to us-east5
  → train 5M then 10M. Gate the final 5M *training* launch on user; drive data stages autonomously.

## 2026-06-24 23:18 heartbeat
- A) 1M: advancing 1.56k/3.91k, 9.3 s/it, loss ~0.24, ~6h left. Healthy.
- B) Full assemble `/michaelryan/hq-distill-rand-assemble` SUCCEEDED — 3001 metadata shards
  at gs://marin-us-central2/datasets/high_quality_3000_distill_random/_staging_metadata/.
  HTMLJOIN launched: `/michaelryan/hq-distill-rand-htmljoin` (us-central2, --html-dir
  baseline_3000_random-34884d --tag random). NEXT after htmljoin: negatives → cascade survivor (5M).

## 2026-06-24 23:36 heartbeat
- A) 1M: advancing 1.68k/3.91k, 9.1 s/it, loss ~0.33, ~5.6h left. Healthy.
- B) htmljoin SUCCEEDED (data/ parquet written). NEGATIVES launched:
  `/michaelryan/hq-distill-rand-negatives` (us-central2, --html-dir baseline_3000_random-34884d
  --tag random). NEXT after negatives: cascade survivor filter (random, 5M, test-disjoint) → mirror.

## 2026-06-25 00:0x heartbeat — survivor-filter (fastText stage-1) PREPPED, leakage guard solved
- A) 1M: advancing ~1.78k+/3.91k, 9.2 s/it, loss healthy, ~5h left.
- B) negatives at ~2464/3000 (running). Survivor filter reads BOTH data/ AND data_no_useful/
  → must wait for negatives to finish before launching.

### LEAKAGE GUARD (CRITICAL, solved)
The survivor filter's index>=300 disjointness was built for the TIME-SORTED 3k (val/test in shards
0-299). The RANDOM pool has no index↔split relation. Computed via metadata only (no data egress):
- _split_manifest.json: val=35 + test=35 = 70 held-out WARCs (positional in original sorted pool).
- orig pool (baseline_3000-265ff5) ∩ random pool (baseline_3000_random-34884d) = 888 WARC overlap.
- 21 of the 70 held-out WARCs ALSO appear in the random pool → at random shard indices:
  `1,8,21,34,40,41,44,53,58,59,62,74,76,105,111,122,146,161,259,275,330`
- NOTE index>=300 would have LEAKED shard 330 (and wasted 280 clean WARCs). So index-range is WRONG
  for the random pool — must exclude by HASH-mapped indices.
- Mapping is sound: load_warcs sorts by hash; HTML files named data-<hash>.jsonl.gz; gcloud ls is
  lexicographic → shard i ↔ sorted(hashes)[i] in BOTH pools.
- Added `--exclude-shards` to cascade_survivor_filter.py (compiles; 21-index parse verified).

### STAGE-1 MODEL (pinned)
`gs://marin-us-central2/classifiers/useful_fasttext/body_strip_natratio200kpos_mc500/model.bin`
(0.96 GB — exact match to documented cascade stage-1; threshold default 0.0121 calibrated for it).

### SURVIVOR-FILTER LAUNCH (run AFTER negatives done; us-central2; ~48M survivors possible → 10M easy)
uv run iris --cluster marin job run --region us-central2 --cpu 16 --memory 64GB --disk 50GB \
  --priority interactive --no-wait --extra cpu --enable-extra-resources \
  --job-name hq-rand-survivor-10m \
  -e WANDB_API_KEY ... -e HF_TOKEN ... -- \
  python experiments/baseline_collection/cascade_survivor_filter.py \
    --model gs://marin-us-central2/classifiers/useful_fasttext/body_strip_natratio200kpos_mc500/model.bin \
    --data-base gs://marin-us-central2/datasets/high_quality_3000_distill_random \
    --out gs://marin-us-central2/classifiers/useful_fasttext/survivor_random/parts \
    --shard-start 0 --shard-end 3000 --target-survivors 10000000 --workers 16 \
    --exclude-shards 1,8,21,34,40,41,44,53,58,59,62,74,76,105,111,122,146,161,259,275,330
Then preshard 5M + 10M (sample_preshard_survivors.py pattern) → mirror to us-east5 → ASK USER before
5M training launch.

## 2026-06-25 00:18 heartbeat
- A) 1M: advancing 1.95k/3.91k, 9.1 s/it, loss ~0.22, ~5h left. Healthy.
- B) Negatives SUCCEEDED (3000 data_no_useful shards). SURVIVOR FILTER LAUNCHED:
  `/michaelryan/hq-rand-survivor-10m` (us-central2, model body_strip_natratio200kpos_mc500,
  target 10M, exclude 21 leaking shards). Verify "EXCLUDING 21 shards" + scoring next tick.
  Output parts: gs://marin-us-central2/classifiers/useful_fasttext/survivor_random/parts.
  NEXT: preshard 5M + 10M → mirror us-east5 → ASK USER before 5M training.

## 2026-06-25 00:35 heartbeat — survivor filter FAILED (fasttext missing) → relaunched
- A) 1M: 2.06k/3.91k, 9.2 s/it, loss ~0.23, ~4.7h left. Healthy.
- B) Survivor filter FAILED first try: ModuleNotFoundError 'fasttext' — `cpu` extra lacks it.
  fasttext-wheel is in the `dclm` extra. RELAUNCHED with `--extra cpu --extra dclm`
  (same job-name hq-rand-survivor-10m, model/exclude/target unchanged). Verify scoring next tick.

## 2026-06-25 00:52 heartbeat — survivor filter SCORING (healthy)
- A) 1M: 2.17k/3.91k, 9.1 s/it, loss ~0.25, ~4.4h left.
- B) Survivor filter RUNNING (dclm extra fixed it). Denominator (N/2979) confirms 21-shard exclusion
  applied. cum=1.25M survivors @ 73 shards (~17k/shard), surv-rate 45.4%, useful-frac 10.1% (stable).
  → 10M in ~585 shards, ETA ~2-2.5h. 75 parts written to survivor_random/parts.
- COMPARABILITY NOTE: natural useful-frac = 10.1% (NOT the ~20% in old notes — imprecise). Same
  fastText model + threshold 0.0121 + sibling subset of same 10k extraction as the 1M baseline →
  distributions match by construction. train_classifier shuffles, so class-ordered parts are fine.
- PRESHARD CAUTION (next): parts in us-central2; original presharded set in us-east5. Producing
  presharded 5M/10M in us-east5 = cross-region GB write. Stage in us-central2 + ASK USER re mirror.

## 2026-06-25 03:31 — ModernBERT-LARGE arm (new experiment) + cluster/egress findings
USER REQUEST: train ModernBERT-large @ 8192 ctx across dataset sizes 50k/200k/1M/5M/10M (the
data-scaling curve, bigger model). Base @8192 baseline: 0.564(50k)/0.645(200k)/~0.705(1M).

### Cluster TPU families (lib/iris/examples/marin.yaml) + EGRESS
- v4: us-central2-b, 32GB/chip, RESERVED+preempt. v5p: us-central1-a/us-east5-a, 95GB/chip, preempt.
  v6e: ew4-a/us-east1-d/us-east5-b 32GB. v5e: ew4-b/us-west4-a 16GB.
- Survivor data is in us-central2 → train on v4-us-central2 = $0 egress (and reserved/no-preempt).
- v5p (95GB, large fits at high pdp/faster) is NOT in us-central2 → ~$6-9 one-time mirror of the
  ~72GB 10M presharded set. RECOMMEND v4-us-central2 (bigger slice v4-16/32) unless v4 too slow.
- Saved to memory reference_cluster_tpu_types_egress.md.

### Prep DONE (launch_modernbert_levanter.py)
- Added MODEL_PRESETS {base,large} + `--model-size` flag (sets hidden/inter/layers/heads +
  reference_checkpoint; tokenizer+pad shared). Verified: --no-submit builds large config OK.
  large dims: hidden 1024 / inter 2624 / 28 layers / 16 heads / vocab 50368 / 8192 (395M).
- FIT-SMOKE launched: `/michaelryan/mb-clf-large-smoke-coord` (run-id mb-clf-large-fit-smoke),
  v6e-4 us-east5 (32GB/chip = same as v4), 8192 ctx, pdp=1, splash, 2560 rows. Tests fit + warm-start
  load of large HF weights. CHECK next tick: compiles? fits at pdp=1? else enable grad-checkpoint.

### USER DECISIONS
- 5M/10M-large compute: user asked re v5p/egress (answered above; awaiting pick on v4 vs v5p).
- Use the new 10M random survivor set, sliced via --train-rows, for ALL 5 sizes (consistent pool).

## 2026-06-25 03:41 — 10M survivor data DONE + preshard launched
- SURVIVOR FILTER `/michaelryan/hq-rand-survivor-10m`: SUCCEEDED. 552 parts, 67GB, ~10M survivors
  (stopped at 552/2979 shards = hit 10M target). Leakage-guarded (21 shards excluded).
- sample_preshard_survivors.py PARAMETERIZED (argparse: --parts/--out-prefix/--n/--world/--total/--seed;
  was hardcoded 1M→us-east5). Now writes IN-REGION us-central2.
- PRESHARD launched: `/michaelryan/hq-rand-preshard` (us-central2; keep-all ~10M, world 40 →
  gs://marin-us-central2/classifiers/useful_fasttext/presharded_survivor_random/train_shard_*.txt.gz).
- KEY: ONE presharded ~10M set serves ALL sizes (50k/200k/1M/5M/10M) via --train-rows (no separate files).
- TRAINING REGION: data is us-central2. Train on v4-us-central2 = $0 egress (recommend), or mirror
  presharded set (~$9) to v5p region. Awaiting user pick. Then update launcher TRAIN_GLOB + region.

## 2026-06-25 03:58 — LARGE FIT-SMOKE PASSED (key result) + 1M near done + preshard running
- B) LARGE FIT-SMOKE (mb-clf-large-fit-smoke, v6e-4, 8192, pdp=1, splash): **PASSED**.
  Completed 10/10 steps, warm-start loaded large HF weights (no dim mismatch), checkpoint saved.
  **large@8192 FITS at pdp=1 on 32GB (v4/v6e) — NO grad-checkpoint needed.**
  Steady-state **~15.7 s/it** (batch 256) = ~1.7x base (9.2 s/it). ETAs per size on a single v6e-4/v4-4:
    50k ~0.85h, 200k ~3.4h, 1M ~17h, 5M ~85h (~3.5d), 10M ~170h (~7d).
  → 50k/200k/1M-large feasible single-chip; 5M/10M-large want a bigger slice (v4-16/32 us-central2,
    zero egress → ~4x faster: 5M ~21h, 10M ~42h).
- A) 1M parity: 3.38k/3.91k, 9.2 s/it, loss 0.224, ~1.3h left. NEAR DONE → F1 headline imminent.
- C) Preshard running (40 shards finalize at completion; 0 visible mid-run is expected).

## 2026-06-25 05:34 — ★ 1M PARITY LANDED: PASS ★ (base migration headline COMPLETE)
- mb-clf-1M-surv-f: state=finished, step 3905, **best_f1=0.6969 @ thr=0.36**.
- PARITY vs torch 0.705: Δ=0.008 (~1%) → **AT PARITY** (within F1 noise on 516-pos test).
- HF checkpoint saved: gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-1M-surv-f/hf/
  (config.json + model.safetensors + tokenizer).
- => Levanter/JAX ModernBERT classifier REPRODUCES torch_xla. M1-M5 done; base migration validated.
  Data-scaling @8192 (Levanter, survivor): 50k 0.564 / 200k 0.645 / 1M 0.697(≈0.705).
- Throughput: ~9.2 s/it @ pdp=2 (base, v6e-4); preempt→auto-resume worked all run (was the torch pain point).

## 2026-06-25 07:01 — DATA READY (parallel preshard) + v4 SWEEP LAUNCHED
- Parallel preshard DONE: 40 shards at gs://marin-us-central2/classifiers/useful_fasttext/presharded_survivor_random_par/
  (8 jobs g0-g7, ~46min vs ~7h single-process; ~10M survivors, useful-frac ~10%). Slow fallback stopped.
- V4SHORT LAUNCHED (v4-8 us-central2, @8192, zero egress): mb-clf-large-{50k,200k,1M}-surv + mb-clf-base-1M-rand. 4 coords submitted.
- MIRROR to us-east5 in progress (bg, ~67GB, pre-approved). v5plong (10 runs: large@5M + base@5M @8192,
  + base&large@10M ctx sweep {1024,2048,4096,8192}) launches after mirror verifies 40 shards.
- Full sweep = 14 runs. Existing base curve (old pool): 50k 0.564/200k 0.645/1M 0.697.

## 2026-06-25 07:05 — ALL 14 RUNS LAUNCHED ★ sweep live
Mirror to us-east5 verified (40 shards, no double-nest). v5plong submitted (10 runs).
FULL SWEEP (all @random pool, pdp=1, splash, warm-start):
- v4-8 us-central2 @8192: mb-clf-large-50k-surv, -200k-, -1M-surv, mb-clf-base-1M-rand
- v5p-8 us-east5: mb-clf-large-5M-surv, mb-clf-base-5M-surv (@8192),
  + 10M CONTEXT SWEEP: mb-clf-{large,base}-10M-c{1024,2048,4096,8192}
Now: crash+throughput watch; record best_f1 per run → base-vs-large data curve (50k→10M) + 10M ctx curve.
WATCH short-ctx 10M (c1024/c2048/c4096) for data-loader starvation (>30s/it → flag).

## MORNING SUMMARY (live, updated 2026-06-25 07:27)
✅ PARITY: Levanter base 1M survivor F1=0.697 ≈ torch 0.705 (migration validated, ckpt saved).
✅ DATA: ~10M random survivors → 40 presharded shards (us-central2) + mirrored us-east5. Leakage-guarded (21 val/test shards excluded).
✅ LAUNCHED all 14 runs (random pool, pdp=1, splash, warm-start). wandb project modernbert-useful:
   STATUS @07:27 — all 14 state=running, step=None (spinning up: TPU alloc + warm-start HF load + first compile; normal ~20-25min in). No crashes.
   | run | model | data | ctx | hw | F1 |
   | mb-clf-large-50k-surv | large | 50k | 8192 | v4-8 c2 | pending |
   | mb-clf-large-200k-surv | large | 200k | 8192 | v4-8 c2 | pending |
   | mb-clf-large-1M-surv | large | 1M | 8192 | v4-8 c2 | pending |
   | mb-clf-base-1M-rand | base | 1M | 8192 | v4-8 c2 | pending |
   | mb-clf-large-5M-surv | large | 5M | 8192 | v5p-8 e5 | pending |
   | mb-clf-base-5M-surv | base | 5M | 8192 | v5p-8 e5 | pending |
   | mb-clf-{large,base}-10M-c{1024,2048,4096,8192} | both | 10M | sweep | v5p-8 e5 | pending |
REFERENCE base curve (old pool): 50k 0.564 / 200k 0.645 / 1M 0.697.
WATCH: short-ctx 10M (c1024/c2048/c4096) for data-loader starvation (>30s/it). iris control-plane flaky → monitor via wandb.

## 2026-06-25 08:00 — v5p OOM FIX (autonomous)
- ISSUE: all 10 v5p runs (5M + 10M ctx sweep) CRASHED at startup, Exit 137 host-RAM OOM.
  ROOT CAUSE: TextClassificationDataset.read_fasttext_shards materializes ALL --train-rows texts in
  host RAM (~25KB/doc); with_tpu default ram=128g → 5M(~125GB)/10M(~250GB) exceed it. v4 runs (≤1M) fine.
- FIX: added `--memory-gb` to launch_modernbert_levanter.py (passes ram= to with_tpu). Relaunched all
  10 v5p runs with --memory-gb 400 (v5p host=448GB). 5M/10M now fit. launch_mb_sweep.sh v5plong updated.
- v4 runs (large 50k/200k/1M, base 1M-rand) UNAFFECTED, advancing (steps 16-32 @07:27).
- PROPER long-term fix (morning note): stream the dataset instead of materializing (avoids the 400GB
  host footprint); fine for now via RAM bump. If 10M still OOMs at 400g → streaming required.

## 2026-06-25 08:21 — all 14 RUNNING post-OOM-fix
- v4 (advancing): base-1M-rand s=114, large-1M s=62, large-200k s=74, large-50k s=78.
  THROUGHPUT: large@8192 on v4-8 ~46s/step (v4 slower per-chip than v6e smoke; data-loader contention
  from 4 co-located runs possible) → large-50k ~2.7h, large-1M ~2 days. Progressing, not stuck.
- v5p (400g fix WORKED): all 10 running, step=None = loading 5M/10M rows into RAM (minutes).
  base-5M earlier "crashed" was transient (old 128g attempt); relaunch @14:58 running. WATCH: 10M runs
  load ~250GB at 400g — confirm they STEP (not OOM-during-load) next tick; if OOM → bump 440.
- First F1 (large-50k) ETA ~09:45. iris control plane flaky → wandb is truth.

## 2026-06-25 08:56 — 10M OOM round 2 (autonomous)
- v4 (4): advancing — base-1M-rand s=199, large-{50k s=125,200k s=121,1M s=109}. large-50k ~64%, F1 ~09:45.
- v5p 5M (large+base): running at 400g (fit).
- v5p 10M c1024/c2048 (base+large, 4 runs): running at 400g, loading (may survive — smaller buffers).
- v5p 10M c4096/c8192 (base+large, 4 runs): OOM'd at 400g (text load ~400GB > 400g). Stopped lingering
  coords, RELAUNCHED at --memory-gb 440 (max on 448GB host). ONE shot — 440g is marginal for c8192.
  IF still OOM next tick → 10M high-ctx BLOCKED on streaming-dataset change (morning task; NOT safe
  to write unattended). Won't thrash further.
- NOTE: a zsh word-split bug created 4 junk coords named "mb-clf-...-c8192 large 8192-coord" (spaces);
  they fail argparse harmlessly (empty --model-size) — IGNORE in wandb/job list.
- ROOT FIX (deferred, morning): TextClassificationDataset materializes all train-rows in host RAM;
  stream it (read by index from shards) to remove the ~400GB/host footprint → 10M works at any ctx.

## 2026-06-25 09:22 — 10M high-ctx BLOCKED (streaming needed); core sweep healthy
- v4 (4) advancing: large-50k s=182/195 (~93%, F1 imminent), large-200k s=177, large-1M s=165, base-1M-rand s=302.
- v5p 5M: large-5M STEPPING s=85 (400g holds for 5M ✓); base-5M loading.
- v5p 10M c4096+c8192 (4): OOM'd AGAIN at 440g → CONFIRMED BLOCKED. 10M text ~400GB materialized
  exceeds even 440g (host=448GB). NOT retrying (no thrash). NEEDS streaming-dataset fix (MORNING).
- v5p 10M c1024/c2048: 3 running at 400g (large-c1024, large/base-c2048, loading — may fit w/ small
  buffers), base-10M-c1024 crashed. Leaving as-is; morning streaming fix will cover all 10M cells.
- ⇒ RELIABLE results: base-vs-large DATA curve 50k→5M (+ existing base old-pool). 10M data-endpoint +
  full 10M context sweep = BLOCKED on streaming. 5M is the largest reliable point tonight.

### MORNING ACTION ITEM: stream TextClassificationDataset (read_fasttext_shards materializes all
### train-rows in host RAM → ~400GB at 10M). Make it read lazily/by-index from shards (or shard by
### host rank) so 10M fits. Then relaunch the 8 10M-ctx-sweep cells. Don't attempt unattended.

## 2026-06-25 09:49 — ALL 8 10M runs OOM/BLOCKED; 6 reliable runs stepping
- ALL 8 10M-ctx-sweep runs now CRASHED (low-ctx c1024/c2048 OOM'd too once loaded). ENTIRE 10M sweep
  blocked on streaming-dataset fix (morning). NOT relaunching any. base-vs-large 10M endpoint + context
  curve = deferred to post-fix.
- RELIABLE & stepping (6): large-50k s=193/195 (F1 IMMINENT), large-200k s=223, large-1M s=210,
  base-1M-rand s=384, large-5M s=172, base-5M s=130.
- THROUGHPUT REALITY (~46s/step large@8192 on v4-8): large-50k done ~now; large-200k ~10h (≈17:00);
  large-1M ~50h; base-1M-rand faster. 5M on v5p (rate TBD, many hrs). So tonight: large-50k F1; rest
  trickle in over hrs→days. Curve fills gradually.

## 2026-06-25 10:15 — ★ FIRST F1: large-50k = 0.616 ★
RESULTS TABLE (best_f1 @8192, random pool unless noted):
| data | base | large |
| 50k  | 0.564 (OLD pool ref) | **0.6162** (thr0.34) ✓ |
| 200k | 0.645 (OLD pool ref) | running s=267 |
| 1M   | 0.697 (OLD pool, parity) ; 1M-rand running s=464 | running s=255 |
| 5M   | running s=274 | running s=255 |
| 10M  | BLOCKED (OOM/streaming) | BLOCKED |
- large-50k 0.616 vs base-50k 0.564 (+0.052) → bigger model helps at 50k (CAVEAT: large=random pool,
  0.564=old time-sorted pool; clean base-vs-large lands at 1M/5M both random).
- All 6 reliable runs healthy/stepping. All 8 10M crashed (blocked on morning streaming fix). No action.

## 2026-06-25 12:19 — base-1M-rand v4 node-migration freeze (watching)
- base-1M-rand: step FROZEN at 689 (11:48→12:19, ~34min). Diagnosis: migrated to NEW v4-reserved-8
  node (worker ...20260625-1845...), Preemptions=0 (reserved reschedule, not preempt). Recompiling/
  reloading; checkpoint activity at 12:14 (not dead). WATCHING 1 tick → if still 689 next tick,
  relaunch (auto-resumes from step-689 ckpt, non-lossy).
- Other v4 (large-1M +15, large-200k +14) SLOWED (shared v4 pod contention) but advancing.
- v5p 5M advancing normally (large-5M s=655, base-5M s=879).
- No new F1s. large-50k=0.6162 still the only finished result. 10M parked.

## 2026-06-25 12:41 — base-1M-rand RESUMED (s=712, past 689) — migration recompile done, no relaunch needed.
v4 runs slow but advancing (large-1M +14/tick, large-200k s=477 ~61%); v5p 5M healthy (large-5M s=726, base-5M s=990). No new F1s (large-50k=0.6162 only). 10M parked. Back to 30-min steady-state.

## 2026-06-25 13:14 — v4 pod RECLAIMED (all 3 v4 runs preempted) → RELAUNCHED (resume from ckpt)
- All 3 v4 runs crashed/killed together (Preemptions=1) — v4-reserved-us-central2 pod reclaimed.
  large-200k (was s=499), large-1M (s=464), base-1M-rand (s=752). v5p 5M UNAFFECTED (us-east5).
- Checkpoints in ttl-temp path (gs://marin-us-central2/tmp/ttl=14d/checkpoints-temp/.../<rid>/checkpoints/):
  large-200k step-476, large-1M step-451, base-1M-rand step-727. RELAUNCHED all 3 (v4-8 us-central2);
  auto-resume from those steps (~15-25 steps lost each).
- NOTE: v4 "reserved" got preempted → if it recurs (preempt-loop), v4 capacity is contended; runs will
  resume each time from ckpt (slow but progress). Watching. v5p (large-5M s=826, base-5M s=1141) healthy.

## 2026-06-25 13:58 — v4 NODE INSTABILITY (auto-retrying); throughput = days for 1M/5M
- v4 runs (large-200k/1M, base-1M-rand): Preemptions=2, "v4-preemptible worker reconcile failure
  threshold exceeded" → v4 nodes flaky. Coords RUNNING, auto-retrying; progress persists via ckpt.
  NOT re-relaunching (auto-retry handles it; re-relaunch=thrash). Will step when v4 stabilizes.
- large-5M (v5p): advancing (log step 897 @18.2s/it; wandb lagged). base-5M advancing s=1365.
- THROUGHPUT: large-5M 18s/it → ~4 days/19.5k steps; large-1M v4 ~46s/it → ~2 days. So 1M/5M F1s land
  over DAYS (large@8192 expensive + preemption tax). large-200k finishes in hrs once v4 settles.
- RELIABLE RESULT so far: large-50k=0.6162. Others pending (v4 settling / long runs). 10M parked.

## 2026-06-25 14:32 — STREAMING DATA FIX implemented (10M unblock)
- ROOT CAUSE (user was right): our TextClassificationDataset was a BESPOKE in-memory loader
  (read_fasttext_shards → all rows in a Python list). Levanter already streams TB via the tokenized
  TreeCache (tensorstore, read-by-index) used by LM training + the audio task-head. We just bypassed it.
- FIX (lib/levanter/.../main/train_classifier.py): added ClassificationLineProcessor(BatchProcessor)
  + CachedClassificationDataset (mirrors ProcessedAudioCache); build_train now build_or_load_cache +
  streams. ONE cache (variable-len input_ids ≤8192) is ctx-agnostic → serves all sweep cells; also
  KILLS the short-ctx data-loader starvation (pre-tokenized). Validated end-to-end locally (cache
  build + get_batch correct). py_compile OK. (pre-commit lint TODO.)
- Cache-build job `clf-cache-build-10m` LAUNCHED (us-east5, 32cpu) → cache at
  gs://marin-us-east5/classifiers/useful_fasttext/presharded_survivor_random_par/_clf_token_cache.
- When cache done → run scratchpad/relaunch_10m_streaming.sh → relaunches 8 10M cells STREAMING
  (NO --memory-gb; memory-bounded). 1M/5M untouched (old code, still running).
- New files: experiments/baseline_collection/build_clf_cache.py.

## 2026-06-25 15:00 — v4→us-east5 MIGRATION (make-before-break, user-approved)
- Resume path confirmed from marin training.py: temp = gs://marin-{region}/tmp/ttl=14d/checkpoints-temp/marin-{region}/checkpoints/modernbert-useful/{run-id}/checkpoints/step-N.
- COPYING (bg) 3 v4 latest ckpts → us-east5 temp under -e5 run-ids (~11.3GB, ~$1.4): large-200k step-476,
  large-1M step-464, base-1M-rand step-752 → mb-clf-{...}-e5.
- migrate_v4_to_e5.sh ready: launches mb-clf-large-200k-surv-e5/large-1M-surv-e5/base-1M-rand-e5 on
  v6e-8 us-east5 (streaming cache, resume from copied ckpt). v6e (not v5p) to spread off the 10M sweep.
- SEQUENCE (make-before-break): when token cache ready → launch 8 10M cells + 3 -e5 migrated runs →
  once -e5 ALLOCATED+STEPPING → THEN kill the 3 v4 runs (mb-clf-large-200k-surv/large-1M-surv/base-1M-rand,
  --include-children). v4 keeps limping until then (no gap). 1M/5M-on-v5p untouched.

## 2026-06-25 16:26 — CLUSTER-WIDE TPU reconcile flakiness (v4 AND v5p) + cache 29/40
- 5M pair FROZEN ~1h+ (base-5M s=1729, large-5M s=936): "worker reconcile failure threshold exceeded"
  on v5p slices (base-5M v5p-64 worker-2, large-5M v5p-16 worker-0). SAME failure mode as v4 → it's
  cluster-wide TPU node-health, not pool-specific. Coords running (auto-retry); ckpt-protected. Can't
  fix infra → leave, no thrash. iris should eventually reschedule to healthy slices.
- large-200k RECOVERED (running s=508, advancing) — v4 slice cycled to healthy. 1M pair advancing.
- Cache build 29/40 shards committed (nearly done). When ledger present → launch 8 10M + 3 migrated;
  make-before-break protects v4 if v6e also flaky (-e5 won't step → v4 stays).
- NOTE: launching 10M on v5p during this flakiness → they'll auto-retry too; acceptable (streaming,
  ckpt-protected). Waiting for "cluster stable" is indefinite; proceed when cache ready.

## 2026-06-25 17:15 — cache made OPT-IN; 1M/200k migrated FAITHFULLY (in-memory, no cache wait)
- FIX (user caught it): streaming change had made build_train cache-ONLY, wrongly coupling the small
  runs (which always ran IN-MEMORY, no cache) to the 10M cache build. Restored in-memory path +
  added `use_cache` flag (default FALSE=in-memory, faithful) + launcher --use-cache. 5M/10M opt into
  cache (they OOM in-memory); ≤1M stay in-memory as before.
- MIGRATED (make-before-break): launched mb-clf-large-200k-surv-e5 / large-1M-surv-e5 / base-1M-rand-e5
  on v6e-8 us-east5, IN-MEMORY (no --use-cache), resume from staged ckpts (step-476/516/846). NO cache
  dependency → migrate immediately. v4 originals still running (limping) until -e5 stepping → then kill.
- 10M cells (8) still cache-gated (--use-cache); cache build 33/40, ~20-40min. relaunch_10m_streaming.sh
  now passes --use-cache.

## 2026-06-25 17:44 — cache build STUCK (worker-loop) → relaunched (resumes 33)
- Diagnosis: build stuck at 33/40 ~40min. tmp listing showed shard 00000 with 3 retry attempts
  (528MB/927MB/1058MB orphan partials) → cluster reconcile flakiness killing the build's zephyr
  workers mid-shard on the tail, retry-looping. NOT slow — stuck. (First 33 committed fast ~20min.)
- 33 committed shards are durable (GCS __shards__/ + per-shard ledger). Stopped clf-cache-build-10m,
  relaunched clf-cache-build-10m-r2 (same cache_dir) → build_or_load_cache RESUMES (skips 33, redoes ~7).
  VERIFY next tick: committed stays 33 → 40 (not reset to 0). User OK with reprocess if resume fails.
- Migration: 3 -e5 still spinning up on v6e (none stepping ~30min — v6e load/compile + flakiness).
  v4 twins alive (make-before-break). base-1M-rand(v4) advancing s=1331.

## 2026-06-25 17:56 — cache resume CONFIRMED (33 held) + large-200k migrated
- Cache r2: committed held at 33 (RESUME WORKED — 33 NOT reprocessed, as user wanted). Finishing last 7.
- large-200k-surv-e5: STEPPING on v6e, s=516 (resumed from staged ckpt 476, +40) → make-before-break:
  KILLED v4 twin mb-clf-large-200k-surv-coord. large-200k now on healthy v6e, ~66%, ~1h to finish.
- large-1M-surv-e5, base-1M-rand-e5: still spinning up (no wandb yet, 1M in-memory load slow). v4 twins
  (large-1M-surv, base-1M-rand) ALIVE until their -e5 steps.
- 5M (v5p) still frozen on reconcile-fail (auto-retry, leave).

## 2026-06-25 18:06 — cache rebuild was PENDING (capacity) → relaunched SMALL (cpu8)
- r2 (cpu32) sat PENDING ~21min (no CPU worker — cluster contended). Still 33/40. Stopped r2,
  relaunched clf-cache-build-10m-r3 with --cpu 8 --memory 24GB (easier to schedule; 7 shards is light;
  resumes the 33 committed). 33 shards remain durable regardless.
- Root: cluster-wide contention/flakiness affecting even CPU build scheduling. The first build (cpu32)
  scheduled fine earlier; now contended.

## 2026-06-25 18:18 — cache r3 RUNNING (cpu8 scheduled); v6e capacity-limited migration
- Cache r3 (cpu8): RUNNING (small request scheduled, vs r2 cpu32 pending). Resuming from 33, finishing 7.
- large-200k-e5: 86% (673/781) on v6e, ~27min to F1. GOOD.
- large-1M-e5 + base-1M-rand-e5: child State=PENDING (no v6e slot — only 1 v6e-8 free, large-200k took
  it). NOT stuck — capacity-waiting. v4 twins (large-1M-surv, base-1M-rand) ALIVE (make-before-break,
  no gap). When large-200k-e5 finishes → frees v6e → a pending 1M-e5 starts. Cascade. DON'T thrash.
- 5M (v5p) frozen on reconcile. cluster broadly contended (v4/v5p/v6e).

## 2026-06-25 18:37 — re-staged 1M migration to AVOID regression (user caught it)
- v4 1M both CRASHED (large-1M s=740, base-1M s=1336) but were AHEAD of stale staged ckpts (516/846).
  Migrating from stale would regress ~224/490 steps (~3h redo). Fix: re-staged LATEST v4 ckpts
  (large-1M step-724, base-1M step-1312) → -e5 temp paths; relaunched mb-clf-large-1M-surv-e5 +
  base-1M-rand-e5 (v6e-8, in-memory) → resume from 724/1312 (current progress, no regression).
- large-200k-e5: ~770/781 on v6e, F1 imminent. v4 1M twins ALIVE until -e5 step (make-before-break).
- Cache r3 (cpu8) running, resuming from 33.

## 2026-06-25 19:18 — ★ large-200k = 0.672 ★ + cache char-cap fix
RESULTS so far (best_f1 @8192):
| data | base | large |
| 50k  | 0.564 (old pool) | 0.616 |
| 200k | 0.645 (old pool) | **0.672** (thr0.38) ✓ |
| 1M   | 0.697 (old pool, parity) | running (-e5 v6e, resume 724) |
| 5M   | running (v5p, frozen) | running (v5p, frozen) |
- large > base at 50k (+0.052) and 200k (+0.027). large data curve rising: 0.616→0.672.
- CACHE FIX: 7 tail shards stuck ~2h (pathological ~1MB docs choking tokenizer). Added char-cap
  text[:200000] to ClassificationLineProcessor (IDENTICAL output for ≤8192 tok; metadata unchanged →
  33 committed shards resume, no rebuild). Stopped r3, relaunched clf-cache-build-10m-r4 (cpu8).
- 1M -e5 spinning up on v6e (resume 724/1312); v4 twins alive (make-before-break).

## 2026-06-25 19:47 — ★ v4→v6e MIGRATION COMPLETE (no regression) ★
- 1M -e5 BOTH stepping on v6e, resumed from re-staged ckpts: large-1M-e5 s=747 (from 724),
  base-1M-rand-e5 s=1362 (from 1312). NO regression (user's catch). v4 1M coords already gone
  ("no running jobs matched" — died on flaky v4); -e5 captured their progress via ckpt. All 3 flaky-v4
  runs (200k done 0.672, 1M, base-1M) now on healthy v6e.
- CACHE r4 (char-cap): still 33/40 after ~29min (2nd tick). bug-report log truncated. Give ONE more
  tick → if still 33, FLAG as stubborn blocker (7 tail shards, non-doc-size issue?), STOP relaunching
  (4 attempts = enough, no thrash). 10M sweep stays blocked pending daytime debug; it's days-long
  anyway. base-vs-large curve (50k/200k/1M) + 5M unaffected.

## ========== MORNING SUMMARY (current @20:04) ==========
PARITY ✓ (base 1M = 0.697 ≈ torch 0.705). Levanter ModernBERT migration validated.
RESULTS (best_f1 @8192):
| data | base | large |
| 50k  | 0.564 (old pool) | 0.616 |
| 200k | 0.645 (old pool) | 0.672 |
| 1M   | 0.697 (old pool) | running on v6e (s=860/3906) ; base-1M-random running (s=1503) |
| 5M   | frozen on v5p | frozen on v5p |
- large > base at 50k(+.052) & 200k(+.027). large data curve rising. Same-pool 1M (large vs base-rand)
  will land in a few hours (both advancing healthy on v6e post-migration).
CONTEXT-LENGTH sweep (base): 50k 0.238/0.327/0.451/0.564 ; 200k 0.355/0.441/0.546/0.645 (1024/2048/4096/8192).

INFRA STATE:
- v4→v6e MIGRATION COMPLETE: 200k/1M/base-1M moved off flaky v4 to healthy v6e (resumed from ckpts,
  no regression). v4 originals dead/superseded.
- 5M (large+base) FROZEN on v5p reconcile-failure flakiness (coord running, auto-retry, ckpt-safe).
- 10M CONTEXT SWEEP (8 cells): BLOCKED. Token cache stuck at 33/40 across 4 build attempts (r1 worker-
  loop, r2 pending, r3/r4 stuck incl char-cap fix). 7 tail shards won't commit → NEEDS DAYTIME DEBUG
  (inspect which 7 shards; suspect corrupt shard or tokenizer hang, NOT doc-size since char-cap didn't
  fix). NOT relaunching further (no thrash). 10M is days-long anyway. r4 left running (may self-resolve).
- CODE: streaming cache path added (opt-in use_cache); in-memory default (faithful for <=1M). char-cap
  in processor. Lint pending. Nothing committed to git.

## 2026-06-25 20:45 — 1M -e5 advancing; 5M stuck 5.5h on v5p (flag)
- 1M -e5 (v6e) advancing well: base-1M-rand-e5 s=1791 (~46%), large-1M-surv-e5 s=1133 (~29%). ~140/tick.
  No F1 yet (hours out). Next results = same-pool 1M large-vs-base.
- 5M (base-5M s=1729, large-5M s=936) FROZEN ~5.5h on v5p reconcile flakiness (bouncing crashed/running,
  never advancing). NOT migrating now: v6e full (1M -e5 there) → migration would just pend. PLAN: migrate
  5M → v6e (in-memory 400g, v6e host=720GB) WHEN 1M -e5 finish + free v6e. Leave for now (no thrash).
- Cache 10M still blocked (33/40, no ledger).

## 2026-06-25 22:35 — cluster healing; base-1M-e5 relaunched off stuck node
- large-1M-surv-e5 advancing (s=1850, 47%). 5M BOTH recovered+advancing (base-5M 1962/10%, large-5M 1078/5.5%).
- base-1M-rand-e5: stuck 1.5h frozen 1951 (preempt/reconcile loop on bad v6e node, child Preemptions=2)
  while peers recovered → RELAUNCHED (single intervention; resume from 1951 ckpt, fresh v6e node).
- No new F1 (large-50k 0.616, large-200k 0.672 stand). 10M cache still blocked 33/40.

## 2026-06-25 23:01 — ★ CACHE RESOLVING (char-cap worked) ★ 37/40
- NOT blocked after all: char-cap r4 ground through the stuck shards. 37/40 committed; last 3
  (00006 1.8GB, 00007 917MB, 00024 258MB tmp — the biggest/pathological-doc shards) actively
  processing. Char-cap (text[:200000]) was the fix for the near-1MB-doc tokenizer stall.
- → cache finishes soon → ledger → 8 10M cells launch (relaunch_10m_streaming.sh, --use-cache).
- 1M -e5: large-1M ~52%, base-1M relaunched (resume 1951). 5M advancing. tightening cadence to catch
  cache→10M launch (user wants 10M progress).

## 2026-06-25 23:21 — cache root cause = worker-reconcile-death mid-long-shard; r5 (100k cap)
- ROOT CAUSE (not doc-size): build's worker (on flaky v6e node) hits "reconcile failure threshold
  exceeded" and DIES mid-shard. Shard 00006 had 7 retry attempts (262MB-1.8GB partials). The 3 longest
  shards (00006/00007/00024, ~1.8GB each) take too long to finish between node deaths → never commit.
- FIX: dropped char-cap 200k→100k (still 12x token cap, identical output, resume preserves 37) to HALVE
  per-shard work → shards complete inside a worker lifetime. Stopped r4, relaunched clf-cache-build-10m-r5.
  Cluster is healing (1M/5M recovered) which also helps. When ledger → 8 10M cells launch.

## 2026-06-25 23:35 — ★ CACHE 40/40 (char-cap fix WORKED) ★ ledger finalizing
- 100k char-cap (r5) cleared the last 3 stuck shards → 40/40 committed. shard_ledger.json finalizing
  (r5 running, writing ledger). When ledger present → 8 10M cells launch. Cluster healed (helped).
- All runs healthy: base-1M-rand-e5 resumed s=2080 (relaunch worked), large-1M-surv-e5 s=2274 (58%),
  5M advancing (base-5M 2271, large-5M 1281). No new F1 (1M ~53-58%, may finish ~2-3am).

## 2026-06-25 23:45 — ★ 10M CACHE COMPLETE → 8 10M CELLS LAUNCHING ★
- shard_ledger.json WRITTEN, 40/40. Cache fully built (char-cap 100k fix). 10M sweep UNBLOCKED after
  ~9h of cache struggle (worker-reconcile-death root cause + char-cap fix).
- Launched relaunch_10m_streaming.sh → 8 cells (mb-clf-{base,large}-10M-c{1024,2048,4096,8192}) on
  v5p-8 us-east5, STREAMING from cache (--use-cache, memory-bounded, no OOM). ~39k steps/run (~2 days).
  Some may pend on v5p capacity (queue, fine).
- VERIFY next tick: 8 submitted + they LOAD the cache (TreeCache.load, fast) + start stepping (no OOM,
  no race-build). Short-ctx (c1024/2048) should NOT data-loader-starve now (pre-tokenized cache).

## 2026-06-25 23:59 — 8 10M cells RUNNING (spinning up); USER PRIORITY = c8192
- All 8 10M cells running, s=None (loading cache + compiling; none crashed/pending — v5p had room).
- ★ USER PRIORITY: if v5p capacity forces keeping only a few 10M alive → KEEP base-10M-c8192 +
  large-10M-c8192 (full-ctx data-scaling endpoints); drop/deprioritize the 6 shorter-ctx
  (c1024/c2048/c4096 × base/large). Only act if contention (pending/crash/thrash); not preemptive.
- 1M: base-1M-rand-e5 57%, large-1M-surv-e5 62% (slowed slightly this tick, watch). 5M: base 12%, large 7%.

## 2026-06-26 00:16 — 10M cells LOADING cache ✓ (streaming works) but v5p contended
- large-10M-c8192 log: "Loading cache from .../_clf_token_cache" → STREAMING CONFIRMED (loads, not builds).
  Cache effort paid off. But got PREEMPTED by external job (tonyhlee/eval-starcoder...) → recompiling
  on fresh v5p worker. All 8 cells s=None ~30min (8 v5p-8 + 2 5M + external = v5p-a contended).
- DECISION: 1 more tick. If c8192 cells (base/large-10M-c8192) STILL not stepping → enforce USER PRIORITY:
  stop 6 shorter-ctx 10M cells (c1024/c2048/c4096 × base/large) to free v5p for the 2 c8192 + 5M.
- 1M recovered: base-1M-rand-e5 s=2368 (61%), large-1M-surv-e5 s=2549 (65%) — advancing, may finish ~2-3am.

## 2026-06-26 00:35 — enforced c8192 PRIORITY (10M); pruned 6 shorter-ctx
- 8 10M cells all s=None ~50min (thrash: 8 v5p + 2 5M + external preempt tonyhlee on v5p-a). Enforced
  USER PRIORITY: STOPPED 6 shorter-ctx (base/large × c1024/c2048/c4096). KEPT base-10M-c8192 +
  large-10M-c8192 + 2 5M (4 v5p-8 jobs now). Should let c8192 step. 6 paused @0% (relaunch when v5p frees).
- 1M (v6e, separate): large-1M-surv-e5 ~68% (~3-3.5h to F1), base-1M-rand-e5 ~64%.

## 01:03 — c8192 MIGRATED v5p→v6e (escape external preemption)
- ROOT CAUSE the c8192 weren't stepping: NOT our contention. bug-report on large-10M-c8192 showed
  Preemptions=4, `Preempted by /tonyhlee/eval-starcoder...` (a SERIES of his eval jobs on
  v5p-us-east5-a). Cache-reload logged twice 15 min apart → preempted mid-compile each time before
  first step. tonyhlee's ~15-min eval cadence < c8192 ~15-20min time-to-first-checkpoint → preempt-loop.
  (Pruning our 6 shorter-ctx didn't help — external, not internal, contention. The 5M survive on
  v5p only because they're older + hold their nodes.)
- FIX: stopped both v5p c8192 coords (+/train_classifier children) and relaunched on **v6e-8 us-east5**
  (us-east5-b — different node pool, no tonyhlee; in-region cache, NO cross-region). At 0% (never
  stepped) so clean restart, no loss. v6e-b also hosts the 2 1M-e5 runs (finish ~2-3h) → c8192 may
  PEND behind them, then run cleanly. Pending ≠ preempt-loop. Verify next tick they hold a node + step.
- zsh `set -- $spec` non-split bug recurred (made 2 garbled spaced-name jobs) → stopped them; relaunched
  with EXPLICIT per-arg commands (no loop var splitting).
- 1M pair on v6e advancing: base-1M-rand-e5 s≈2624 (67%), large-1M-surv-e5 s≈2792 (71%). large-1M ETA ~3-4am.

## 01:23 — c8192 migration CONFIRMED healthy on v6e
- base-10M-c8192 bug-report: on v6e-preemptible-8-us-east5-b worker, **Preemptions=0** (escaped tonyhlee),
  cache loaded → now "Reading ModernBERT-base/model.safetensors" (warm-start) + "Progress on:train -/39062"
  (compiling at 8192). step>0 not logged yet but healthy startup, NOT pending/NOT preempted. Both leave.
- 1M: large-1M-surv-e5 s=3001 (77%), base-1M-rand-e5 s=2848 (73%) — advancing, ETA ~2.5-3h. 5M-base s=2798.

## 01:43 — c8192 STEPPING on v6e (migration fully successful)
- base-10M-c8192 s=192, large-10M-c8192 s=89 — both banking steps, escaped tonyhlee (v6e-b, Preemptions=0).
  These are long runs (10M rows → ~39062 steps). c8192 priority now satisfied.
- 1M: large-1M-surv-e5 s=3135 (80%), base-1M-rand-e5 s=2990 (76%) of ~3906 steps — large-1M F1 ETA ~30-45min.
  5M-base s=2895.

## 03:59 — large-1M training DONE (ckpt safe), in final eval/HF-save
- large-1M-surv-e5 reached step 3905, checkpoint saved to gs://marin-us-east5/.../mb-clf-large-1M-surv-e5/checkpoints/step-3905
  at 10:38 UTC. Last log = PjRt compile-start warning 10:38:44 → compiling the final eval forward (inference @8192,
  fresh XLA graph) + scoring 7000-doc test (silent, no per-batch log). ~20min in eval phase — at the edge but worker
  HEALTHY, no error. F1 not posted yet. MODEL SAFE on GCS regardless. Watching one more tick; if still frozen ~32min
  total, dig into eval/HF-save path (would be a real hang then — but ckpt recoverable).
- base-1M-rand-e5 s=3820 (98%) — will hit same eval phase shortly. c8192 climbing (base 1860, large 1022). 5M-base 3554.

## 04:15 — large-1M F1 LANDED = 0.7036 @ thr 0.40 (eval was slow, NOT hung)
- mb-clf-large-1M-surv-e5 FINISHED, eval/best_f1=0.7036, best_threshold=0.40. The ~25min "frozen" was the
  silent 7000-doc@8192 inference eval — completed fine. HF model at gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-large-1M-surv-e5/hf/.
- KEY same-pool large-vs-base @1M: large-1M(random survivor pool)=0.7036 vs base-1M-old(time-sorted)=0.697 → large +0.007.
  PROPER same-pool base is base-1M-rand-e5 (random pool) — in eval now (s=3904), F1 next tick. Regen plot once it lands.
- c8192 climbing (base 2179, large 1199). 5M-base 3679.

### RESULTS TABLE (best-F1 on frozen 7000-doc test)
| data | base | large |
|------|------|-------|
| 50k  | 0.564 | 0.6162@0.34 |
| 200k | 0.645 | 0.6716@0.38 |
| 1M   | 0.697 (time-sorted) / base-1M-rand pending | 0.7036@0.40 |
| 5M   | running | running |
| 10M c8192 | running | running |

## 04:40 — base-1M-rand F1 = 0.6869 @ 0.26; PLOTS REGENERATED with real large-1M
- base-1M-rand-e5 (random survivor pool) FINISHED, eval/best_f1=0.6869 @ thr 0.26.
- **KEY SAME-POOL @1M (random survivor pool): large 0.7036 vs base 0.6869 → large +0.0167.** Clear win for the
  bigger model, as expected (bigger ceiling). (base-1M-old time-sorted=0.6969 is on a different/easier pool; the
  apples-to-apples comparison is large-1M-surv 0.7036 vs base-1M-rand 0.6869.)
- Data-scaling plot REGENERATED (scratch/modernbert_scaling_plots/modernbert_data_scaling.png): large now has 3
  measured pts (50k/200k/1M = 0.616/0.672/0.7036), proper 3-pt saturating fit (no longer borrowed curvature).
  Projections: base ceiling 0.748(opt)/0.722(cons); base@10M 0.714-0.730; **large@10M 0.715-0.722** (tighter band now).
  large curve sits ABOVE base at every measured point. (3-pt exact fit → harmless OptimizeWarning on covariance.)
- Still running: base-5M s=3814, large-5M, base/large-10M-c8192 (all multi-day). 6 shorter-ctx 10M cells PAUSED.

### MORNING SUMMARY (for user wake)
- **Both 1M same-pool points landed.** large(395M)=0.7036 > base(149M)=0.6869 at 1M random-survivor docs (+0.017).
  Large wins at every scale measured: 50k 0.616 vs 0.564, 200k 0.672 vs 0.645, 1M 0.704 vs 0.687.
- **c8192 priority honored + rescued.** The two 10M @8192 cells were preempt-looping on v5p (tonyhlee's eval series);
  migrated to v6e (us-east5-b) where they now step cleanly (escaped, Preemptions=0). All deliverables land at
  gs://marin-us-east5/checkpoints/modernbert-useful/<run-id>/hf/.
- **In flight (multi-day):** base-5M ~20%, large-5M ~11%, base-10M-c8192 ~6%, large-10M-c8192 ~3%. 6 shorter-ctx
  10M cells (1024/2048/4096) PAUSED per your "prioritize 8k if only a few alive" call — relaunch when v5p frees.
- Plots: scratch/modernbert_scaling_plots/{modernbert_data_scaling,modernbert_context_scaling}.png.

## 06:37 — large-5M crashed @ ckpt step-2828, AUTO-RESUMED (no action, no loss)
- mb-clf-large-5M-surv showed wandb state=crashed (stale summary s=2606); coord auto-resubmitted a fresh
  train_classifier child @13:36 UTC on a new healthy v5p-us-east5-a worker (Preemptions=0). It RESTORED from
  step-2828 checkpoint and resumed stepping (s=2895+, loss ~0.18, rate ~18.8s/it → ~86h/3.6d remaining). No
  progress lost — auto-resume-on-preemption working as designed. Left alone (relaunching would have wasted progress).
- Other 3 healthy/climbing: base-5M s=4264(~22%), base-10M-c8192 s=3778(~10%), large-10M-c8192 s=2091(~5%).

## 12:40 — OPERATING CURVE: partial delivered (fastText + large-1M); base-1M/large-200k BLOCKED by tonyhlee
- Operating curve = recall of useful kept (x) vs % useless removed (y=TN/(TN+FP)), frozen-7k test (516 useful).
  Script: scratch/modernbert_scaling_plots/plot_operating_curve.py (reads frozen_preds/<run>.json + fastText overlay).
  New scorer: experiments/baseline_collection/score_frozen_eval.py (loads hf, re-scores frozen-7k, dumps preds JSON;
  has skip-if-done + resumable 800-doc checkpoints + persistent XLA cache + --backend vanilla|splash|auto).
- DELIVERED (partial) modernbert_operating_curve.png — % useless removed @ recall:
    R=0.90 / 0.95 / 0.99 :  fastText 0.867/0.763/0.587 ;  large-1M 0.925/0.880/0.752  (large-1M re-score 0.7040 ≈ recorded ✓)
  => stage-2 BERT removes +12pts useless @95% recall, +16pts @99% recall vs fastText stage-1.
- BLOCKED: large-200k-surv-e5 + base-1M-rand-e5 re-scoring stuck. ROOT CAUSE: tonyhlee eval-starcoder series
  (v77→v81+) saturating v5p-us-east5-a, preempting every ~3-5min — faster than the scorer's startup (jax init +
  read test shards + load 1.5GB ckpt + compile), so it never reaches first 'scored' chunk. v6e capped (c8192).
  Tried skip/resumable/cache/vanilla — config is correct, blocker is external capacity. STOPPED scorer (was also
  stealing v5p windows from base-5M training). RE-RUN when v5p frees (tonyhlee series ends): relaunch score-frozen-e5
  vanilla+cache+resumable for large-200k-surv-e5,base-1M-rand-e5. large-50k (c2) still flaky on v4 — optional.
- Training: large-10M-c8192 4486, base-10M-c8192 8050, large-5M 3462 climbing. base-5M throttled by same tonyhlee v5p-a.

## FULL OPERATING CURVE DELIVERED (all 4 hf checkpoints re-scored on frozen-7k)
modernbert_operating_curve.png — % useless content removed (TN/(TN+FP)) at fixed useful-recall.
best_f1 sanity all OK (rescored≈recorded). Table:
| model      | R=0.90 | R=0.95 | R=0.99 |
|------------|--------|--------|--------|
| fastText   | 0.867  | 0.763  | 0.587  |
| large-50k  | 0.816  | 0.698  | 0.451  |
| large-200k | 0.899  | 0.835  | 0.581  |
| base-1M    | 0.912  | 0.868  | 0.674  |
| large-1M   | 0.925  | 0.880  | 0.752  |
Findings: (1) data scaling monotone for large (50k<200k<1M @ every recall). (2) large-50k UNDERPERFORMS
fastText (too little data) — fastText stage-1 is a strong baseline; 200k+ clearly beats it. (3) large-1M
removes +12pts useless @R0.95, +16.5pts @R0.99 vs fastText. (4) base-vs-large @1M: large edges base
(+1.2pts @R0.95, +7.8pts @R0.99 — large's edge grows at high recall). Plot also has projected 5M/10M band.
Bundle fix: gitignored experiments/baseline_collection/pipeline_planner/web/data/ (was 285MB tracked, broke iris bundle).

## 09:10 (06-27) — revived c1024 pair of the 10M context sweep
- tonyhlee cleared + c8192 well underway → relaunched mb-clf-large-10M-c1024 + mb-clf-base-10M-c1024 (were paused
  at 0% during v5p crunch). v5p-8, --use-cache (10M streaming cache, serves any ctx<=8192), max-seq-len 1024, splash.
- 10M ctx sweep status: c8192 base/large running (56%/31%); c1024 base/large JUST RELAUNCHED; c2048+c4096 (base/large)
  STILL PAUSED at 0% (can revive same way if wanted). bundle fix (gitignore pipeline_planner/web/data) held (10.1MB).

## 09:15 (06-27) — FULL 10M context sweep revived (capacity confirmed free)
- Verified: large-10M-c1024 on v5p worker Preemptions=0 loading cache (healthy), nothing pending, 5M advancing,
  tonyhlee clear → capacity genuinely free. Launched c2048 + c4096 pairs (base+large).
- 10M ctx sweep NOW ALL 8 IN FLIGHT: c8192 (base 56%/large 31%, on v6e), c1024/c2048/c4096 (base+large, on v5p-8,
  --use-cache, splash, pdp=1, just launched/starting). Plus 5M base 49%/large 35%. ~8 v5p-8 + 2 v6e jobs total —
  watch for re-contention if tonyhlee returns (they auto-resume/pend, no loss).

## ~01:13 (06-28) — FIRST 10M context-sweep F1: base-c1024 = 0.4212
- mb-clf-base-10M-c1024 FINISHED, eval/best_f1=0.4212 @ ctx1024. EXPECTED low (ctx1024 truncates docs). Fits
  context-sweep pattern: base@ctx1024 = 0.2376(50k) → 0.3549(200k) → 0.4212(10M data). 10M@8192 (base-c8192, ~86%)
  will be the high end (~0.70+, the data-scaling endpoint vs base-1M 0.697).
- 10M CONTEXT-vs-F1 curve so far (base): ctx1024=0.4212 (10M). Awaiting c2048/c4096/c8192 (base) + all large.
- Progress (~01:13): base-c8192 86%, base-c2048 71%, base-c4096 38%, base-5M 71%, large-c8192 48%, large-c1024 63%,
  large-c2048 40%, large-c4096 20%, large-5M 49%. All advancing.

## ~05:55 (06-28) — base-c8192=0.7219 (KEY 10M data endpoint) + base-c2048=0.4940
- mb-clf-base-10M-c8192 FINISHED eval/best_f1=0.7219 → base DATA-SCALING @8192: 0.564/0.645/0.697/0.7219 @ 50k/200k/1M/10M.
  1M→10M = +0.025 (real gain, not saturated). 10M@8192 (0.722) is the best base result yet.
- mb-clf-base-10M-c2048 FINISHED eval/best_f1=0.4940.
- 10M CONTEXT-vs-F1 curve (base): ctx1024=0.4212, ctx2048=0.4940, ctx8192=0.7219 (ctx4096 pending, expect ~0.60-0.65).
  Steep rise with context — matches 50k/200k ctx-sweep shape; context length matters a lot.
- Still running: base-5M, base-c4096, all 5 large cells (c1024/c2048/c4096/c8192 + large-5M).

## ~15:56 (06-28) — large-c1024=0.4746 (first LARGE 10M ctx point) > base-c1024 0.4212
- mb-clf-large-10M-c1024 FINISHED eval/best_f1=0.4746. Large > base at ctx1024 (+0.053), consistent w/ large's
  edge at 8192 — bigger model helps even at short context.
- 10M CONTEXT-vs-F1 so far: base {1024:0.4212, 2048:0.4940, 8192:0.7219}; large {1024:0.4746}. (base-c4096 ~65%, large c2048/c4096/c8192 running.)
- base-5M ~87% (step16994), base-c4096 ~65% (step25462) — next to finish.

## ~01:50 (06-29) — base-5M=0.7131 → BASE DATA-SCALING CURVE COMPLETE
- mb-clf-base-5M-surv FINISHED eval/best_f1=0.7131. Base DATA-SCALING @8192 (50k/200k/1M/5M/10M) = 0.564/0.645/0.697/0.7131/0.7219.
  Monotonic, diminishing returns; 5M between 1M(.697) and 10M(.7219). base-5M(.7131) > base-1M-rand(.6869): 5x data = +0.026.
- Still running: large-c2048 (~90%), base-c4096 (~84%), large-c8192 (~73%), large-c4096 (~48%), large-5M (~72%).

## ~02:46 (06-29) — base-c4096=0.6288 (BASE 10M CONTEXT CURVE COMPLETE) + large-c2048=0.5579
- mb-clf-base-10M-c4096 FINISHED f1=0.6288. BASE 10M CONTEXT CURVE COMPLETE: ctx 1024/2048/4096/8192 = 0.4212/0.4940/0.6288/0.7219.
  Monotonic steep rise with context — 1024→8192 = +0.30 F1. Context length matters hugely.
- mb-clf-large-10M-c2048 FINISHED f1=0.5579. Large 10M ctx: 1024=0.4746, 2048=0.5579 (large > base at both ctx: +0.053@1024, +0.064@2048).
- Still running: large-c8192 (~75%, = large@10M@8192 endpoint vs large-1M 0.7036), large-c4096 (~50%), large-5M (~74%, vs large-1M 0.7036).

## ~09:30 (06-30) — large-c8192 FINISHED F1=0.7394 (BEST OF SWEEP; finish-line preempt auto-recovered)
- mb-clf-large-10M-c8192 FINISHED eval/best_f1=0.7394 (was preempted at step39060, auto-resumed, completed).
  HEADLINE: large @10M @8192 = 0.7394 = BEST result in the whole sweep.
  - vs base @10M @8192 (.7219): large +0.0175 (large wins at 10M).
  - large data-scaling @8192: 1M→10M = .7036→.7394 (+0.036, strong gain from 10x data).
- Large 10M CONTEXT curve: 1024=.4746, 2048=.5579, 4096=PENDING, 8192=.7394.
- Remaining: large-c4096 (~80%, ~9h, the last ctx pt), large-5M (~84%, large data-scaling 5M). large-c8192 freed a v6e slot;
  large-5M still ADVANCING on v5p so leave it (migrate only if it re-stalls).

## ~13:40 (06-30) — REGENERATED scaling plots with measured data
- scratch/modernbert_scaling_plots/plot_modernbert_scaling.py updated to all measured points (no more wide projection bands).
- modernbert_data_scaling.png: base 5-pt complete (.5635/.6454/.6969/.7131/.7219), large 4-pt (.6162/.6716/.7036/.7394; 5M pending). Saturating fit + dotted extrapolation past 10M.
- modernbert_context_scaling.png: 4 curves — base@50k, base@200k, base@10M (complete .4212/.4940/.6288/.7219), large@10M (.4746/.5579/[4096 pending]/.7394).
- Fit ceilings: base 0.734, large 0.769. 100M extrapolation: base ~0.730, large ~0.754 (modest gain over 10M — supports the diminishing-returns read on the proposed 100M run).
- Will re-regen once large-5M + large-c4096 land (fills the 2 remaining gaps).

## ~14:40 (06-30) — 10M RUN DURATIONS (active compute vs wall-clock) — for writeup
Active = wandb _runtime (true TPU compute); wall = launch→finish (incl preemption downtime). 8 chips each.
| run | hw | F1 | active_h | wall_h |
|-----|----|----|----------|--------|
| large@10M@8192 | v6e-8 | 0.7394 | 100.8 | 121.7 |
| base@10M@8192  | v6e-8 | 0.7219 | 56.8  | 76.1  |
| base@10M@4096  | v5p-8 | 0.6288 | 45.6  | 96.1  |
| large@10M@2048 | v5p-8 | 0.5579 | 43.3  | 93.6  |
| large@10M@1024 | v5p-8 | 0.4746 | 27.8  | 78.1  |
| base@10M@2048  | v5p-8 | 0.4940 | 24.4  | 74.6  |
| base@10M@1024  | v5p-8 | 0.4212 | 16.3  | 66.6  |
Takeaways: headline large@10M@8192 = ~101 TPU-h (~806 v6e chip-h), ~5d wall. Compute sub-quadratic in ctx
(base 16→24→46→57h @ 1024→2048→4096→8192, ~3.5x for 8x ctx via SPLASH). large ~2x base at matched ctx.
v6e clean (~+20h preempt tax); v5p heavily preempted (~+50h each, tonyhlee) → wall 2-4x active.
Pending durations: large@10M@4096, large@5M (still running ~86%/89%).

## ~00:40 (07-02) — large-c4096 FINISHED 0.6579 (LARGE CONTEXT CURVE COMPLETE); large-5M FAILED@98%→relaunched
- mb-clf-large-10M-c4096 FINISHED eval/best_f1=0.6579, active 87.1h (v5p-8). LARGE 10M CONTEXT CURVE COMPLETE:
  ctx 1024/2048/4096/8192 = 0.4746/0.5579/0.6579/0.7394. (base ctx: 0.4212/0.4940/0.6288/0.7219 → large > base at every ctx.)
- mb-clf-large-5M-surv FAILED at step19077/19531 (98%) after ~124h — TRANSIENT HuggingFace connectivity error
  (LocalEntryNotFoundError, couldn't connect to huggingface.co) during a resume, exhausted retries. Ckpt safe @step-19048.
  RELAUNCHED @00:40 matching ORIGINAL config (v5p-8, in-memory --memory-gb 400, no cache, presharded_survivor_random_par)
  → resumes from step-19048, finishes last ~2% + eval. Only remaining cell.

## ~06:07 (07-02) — large-5M=0.7381 → FULL 10-CELL SWEEP COMPLETE
- mb-clf-large-5M-surv FINISHED eval/best_f1=0.7381 (training done days ago @step-19530; eval kept getting v5p-preempted,
  succeeded on the ~4th relaunch, iris State=succeeded). Duration ~128h+ (heavily tonyhlee-throttled on v5p).
- FINAL large data-scaling @8192: 50k/200k/1M/5M/10M = 0.6162/0.6716/0.7036/0.7381/0.7394. NOTE: large SATURATES early —
  5M→10M only +0.0013 (vs base 5M→10M +0.0088). Large's gains came 1M→5M (+0.0345). large > base at every data size.
- SWEEP DONE (10/10 cells). Plots regenerated (both data-scaling curves 5 pts, both 10M ctx curves 4 pts).
