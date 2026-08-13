# Architecture sweep: faster AND more accurate than ModernBERT-base (lpv11 useful-classifier)

Owner: coordinator agent, kicked off 2026-08-08. USER GOAL: research + implement ~10 alternative
architectures for the useful-vs-NO_USEFUL_CONTENT classifier, in Levanter/JAX on TPU (same stack as
the ModernBERT port). Gate each: (1) implement, (2) TPU runtime benchmark — **slower than
ModernBERT-base ⇒ branch dies**, (3) faster ⇒ train on the 1M lpv11 trainset + eval frozen-7k,
(4) still promising ⇒ launch full 10M. Chunked scoring (e.g. 1k from start/middle/end + aggregate,
like the datakit quality classifier's bme scoring) is explicitly in scope for fast archs.

## Baselines to beat (the lpv11 track — current redo)

- **Accuracy bar**: `mb-clf-lpv11-base-1M-c8192` best F1 **0.8424** @ thr 0.38 on the frozen lpv11
  7k test (`gs://marin-us-east5/classifiers/useful_fasttext_lpv11/full_prep_body_strip_test7k/test_sample_7k.txt.gz`).
  (fastText w640 stage-1 = 0.799 for context. 10M ModernBERT run in flight — bar may rise.)
- **Speed bar**: ModernBERT-base @8192 splash ≈ 21.5 docs/s/chip on v6e (modernbert_inference_benchmark).
  RE-MEASURE side-by-side in the new harness for a fair gate; the gate metric is
  **docs/sec/chip at the arch's intended deployment scoring config** on the survivor doc-length
  distribution (chunked/bme configs count TOTAL per-doc cost).
- **Train data (1M)**: first 1M rows of
  `gs://marin-us-east5/classifiers/useful_fasttext_lpv11/presharded_survivor_w640_10M/train_shard_*.txt.gz`
  via `--train-rows 1000000` (TreeCache exists at `_clf_token_cache`, 8192-token cap).
- **10M**: same shards, `--train-rows 10000000`.
- Region: us-east5. v6e for training/benchmarks. NEVER disturb the running
  `mb-clf-lpv11-base-10M-c8192` run. Batch priority for fleets; TRC compute is free but capacity is shared.

## Candidate slate (v1 FINAL — post R1 web research, 2026-08-08)

R1 verified checkpoints/papers. Key finding: **Ettin family (`jhu-clsp/ettin-encoder-{17m,32m,68m}`)
is ModernBERT architecture at small scale, 8K ctx, SAME 50368-vocab tokenizer, MLM-pretrained 2T
tokens, beats ModernBERT at matched size (arXiv 2507.11412)** → three near-free high-probability wins
that reuse modernbert.py + the existing TreeCache unchanged.

| # | branch | warm start | impl | speed est | risk |
|---|--------|-----------|------|-----------|------|
| 1 | Ettin-17m (7L, h256) | jhu-clsp/ettin-encoder-17m | ZERO (config) | ~20x FLOPs (5-15x real) | low |
| 2 | Ettin-32m (10L, h384) | jhu-clsp/ettin-encoder-32m | ZERO | ~8x | low |
| 3 | Ettin-68m (19L, h512) | jhu-clsp/ettin-encoder-68m | ZERO | ~2.5-3x | low |
| 4 | Layer-pruned ModernBERT-base (bottom ~8/22, keep global/local interleave) | surgery on base | LOW | ~2.5x | low ("Poor Man's BERT": −2 GLUE w/o retrain) |
| 5 | Funnel-hybrid super-token: first 4-6 Ettin/MB layers → 8-16x mean-pool → 2-4 new layers + head | partial (embeds+early layers) | LOW-MED | ~3-10x | med |
| 6 | From-scratch tiny ModernBERT (2-4L, ctx 8192 splash) | no (optional embed init) | ZERO (config) | ~10-40x | med (no pretrain; distill-from-champion is the known fix later) |
| 7 | Pooled FastTransformer scaled (in-repo quality-clf arch, bme scoring) | no | LOW (code exists, raw Equinox) | ~100x+ | high (scratch, tiny) |
| 8 | MiniLM-L6 (nreimers/MiniLM-L6-H384-uncased, 512ctx) + begin/middle/end chunk aggregate | YES | MED (classic BERT port: learned pos emb, WordPiece tokenizer) | ~15x/token + chunking | med (tests "does chunking cost F1") |
| 9 | M2-BERT-80M-8k (togethercomputer/m2-bert-80M-8k, Monarch Mixer, subquadratic) | YES | HIGH (long-conv FFT on TPU) — TIMEBOXED spike | ~2-4x | high |
| 10 | Hydra bidirectional Mamba2 (goombalab/hydra) | YES-ish (short-ctx) | HIGH (associative scan) — TIMEBOXED spike | ~2-4x @8192 | high |
| 11 (stretch) | DeBERTa-v3-xsmall chunked (microsoft/deberta-v3-xsmall) | YES | MED-HIGH (disentangled attn port) | ~10-15x/doc chunked | med — only if capacity allows |

CUT after research: FNet (92-97% of BERT GLUE — quality gap fatal; attention isn't our bottleneck
anyway), CNN/DPCNN (pre-2018 evidence, fastText already shows shallow models lose several F1),
LION/GLA linear attention (no pretrained bidirectional weights, trails BERT), RWKV/xLSTM (no
pretrained encoder), NeoBERT (bigger than base), mmBERT-small (dominated by Ettin-68m for English),
Qwen3-0.6B decoder (already tried, F1 0.681).

Portfolio: 6 near-certain (1-6), 2 medium (7-8), 2 timeboxed upside bets (9-10), 1 stretch (11).

## Infra plan (shared scaffolding BEFORE arch fan-out)

1. **Generic classifier main**: generalize `levanter/main/train_classifier.py` so the model config is
   pluggable (registry, like `LmConfig.register_subclass`). Data/eval/F1/chunked-eval code is already
   model-agnostic via `ClassificationExample`. Precedent: `train_decoder_classifier.py`.
2. **Generic TPU benchmark harness**: extend `modernbert_inference_benchmark.py` →
   `arch_inference_benchmark.py`: registered archs × configs, one TPU job, realistic doc-length
   mixture, reports docs/s/chip + per-doc cost at deployment config. ModernBERT-base rebenchmarked in
   the same run = the gate number.
3. **Tests**: CPU smoke for every arch; HF-parity oracle tests for ported pretrained archs
   (pattern: `lib/levanter/tests/test_modernbert_classifier.py`).
4. Launcher: extend `launch_modernbert_levanter.py` or new `launch_arch_classifier.py` with
   `--arch` presets.

## Execution phases

- **R — research** (parallel, background): R1 web research (arch survey, HF checkpoint availability,
  TPU-friendliness); R2 codebase implementation map (what generalizing train_classifier touches,
  splash/mask constraints, registry mechanics, test patterns).
- **S — scaffolding**: coordinator + 1 agent build the generic main/benchmark/launcher.
- **I — implementation fan-out**: 1 background agent per arch, self-contained model file + registration
  + CPU tests. Worktree isolation where files overlap.
- **B — benchmark**: single v6e job sweeping all impls. KILL slower-than-base branches.
- **T — 1M training**: survivors, batch priority, us-east5, wandb `modernbert-useful` project
  (tag `arch-sweep`). Eval frozen-7k in-run (same F1 sweep).
- **X — 10M**: winners that beat 0.8424 (or come close with big speed wins — surface tradeoff to user).

## Ops guardrails (from memory — real incidents)

- Launch coordinators AS cluster jobs, not from laptop (local gcsfs SSL kills submits silently).
- `-e MARIN_PREFIX gs://marin-us-east5` (repo .env pins us-central2, trips region check).
- Survivor shards are class-ordered → trainer `.shuffle()` already handles; never bypass.
- pdp small at 8192 (pdp=-1 OOMs); splash mandatory at 8192; vanilla ≤4096.
- checkpointer keep:[] + preemption storm = data loss → keep the babysitter pattern for long runs.
- Never `&& eval` after train; separate jobs. Never kill running jobs w/o permission. One parent per
  TPU type. Monitor GCS/wandb artifacts, not job state.

## Status log

- 2026-08-08: project kicked off. Plan doc created. R1/R2 research agents launching.
- 2026-08-08 (later): R1+R2 done → slate v1 locked. **Scaffolding LANDED** (all tests green, 23 pass):
  - `lib/levanter/src/levanter/models/classification.py` — NEW: `ClassificationExample` (moved from
    modernbert, re-exported), `register_classifier_arch`/`build_classifier`/`save_classifier`
    registry (MRO dispatch on config type), `save_eqx_classifier`/`load_eqx_classifier` for non-HF
    archs.
  - `train_classifier.py` generalized: `model: LmConfig`, builder/saver dispatch, pad-id fallback
    to tokenizer (explicit None checks — 0 is a valid pad id).
  - `modernbert.py`: registered ModernBERT builder/saver; NEW `PrunedModernBertConfig`
    ("modernbert_pruned") — loads reference at full depth, keeps bottom `num_layers` (prefix keeps
    global/local pattern). Warm-start oracle test vs torch passes.
  - Launcher `launch_modernbert_levanter.py`: presets ettin17/ettin32/ettin68/pruned8/tiny4/tiny2
    (Ettin HF configs verified drop-in: same tokenizer/pad/local-attn; rope thetas 160k/160k,
    classifier_pooling=mean). All presets build via --no-submit. wandb tag `arch-{preset}`.
  - Ettin = same tokenizer ⇒ branches 1-4,6 reuse the existing 10M TreeCache unchanged.
- 2026-08-08 (fan-out): 6 agents launched in parallel — funnelbert (funnel-hybrid super-token),
  pooled_transformer (FastTransformer port), bert (MiniLM-L6 port + chunked), arch_inference_benchmark
  (multi-arch TPU harness w/ gate), M2-BERT spike (port-or-kill), Hydra/BiGDN spike (port-or-kill).
  Constraint: agents create only new files; shared-file changes come back as reports for me to apply.
  NO cluster launches until fan-out settles (half-written files in levanter/models would break the
  auto-importing bundle).
- **Branch 9 (M2-BERT) KILLED at spike** (2026-08-08): attention-free Monarch/Hyena mixer = 24
  bidirectional fp32 FFT long-convs/doc at 2L=16384 — non-MXU work; break-even vs base needs
  ≥~2 TF/s sustained TPU FFT (implausible per FNet's TPU DFT-vs-FFT data); port ≈600-900 lines
  (bidirectional filter MLPs, Monarch blockdiag, no-residual post-LN topology, custom state-dict
  conversion); quality ceiling = BERT-base GLUE parity, below the 0.8424 champion. 3.5x FLOP
  advantage (122 vs 430 MF/token @8192) does not survive the op mix. No repo files touched.
- **FAN-OUT COMPLETE (2026-08-08 ~03:20 PT), all 6 agents landed; 71 tests green across suites:**
  - `levanter/models/funnelbert.py` ("funnelbert"): 4 full MB layers → 8x mask-aware mean-pool →
    4 global layers; 79.4M params, ~5.8x fewer FLOPs; warm-starts embeddings+bottom layers from
    base (torch-oracle-verified); save = eqx. 6 tests.
  - `levanter/models/pooled_transformer.py` ("pooled_transformer"): FastTransformer port with
    2-class CE head + segment-id pads; 25.9M params, ~414k FLOPs/token (~2000x cheaper); from
    scratch only. 25 tests incl. bitwise pad invariance.
  - `levanter/models/bert.py` ("bert"): classic BERT (post-LN, learned positions, pooler) for
    MiniLM-L6 warm start (22.7M, ctx 512, WordPiece 30522 → own token-cache namespace); HF parity
    oracle @1e-4; HF-round-trippable saver. 6 tests. Trains chunked (bme-style at deploy).
  - `levanter/models/bigdn.py` ("bigdn"): bidirectional GatedDeltaNet encoder (fwd + reversed
    passes summed, in-tree TPU-proven kernel reused untouched); h512/L12 86M, 138MF/token
    (~3.1x cheaper, ctx-independent; crossover vs base ~ctx1500); exact pad invariance; from
    scratch. 6 tests. Risk: sequential chunk scan may eat the FLOP margin on TPU — gate decides.
    Hydra port rejected: Triton-only scan, weights on wrong tokenizer/short ctx.
  - `arch_inference_benchmark.py`: 11-arch registry (mb-base gate + ettin17/32/68, pruned8,
    tiny4/2, funnelbert, pooled_transformer(4096x2 windows), bert(512x3 bme), bigdn), CPU smoke
    green for all 11. Fix applied: build under `haliax.partitioning.set_mesh` (bert init needs it).
  - Launcher: `MODEL_FACTORIES` (funnelbert/pooled/bert/bigdn) + per-preset ctx/chunked defaults +
    per-preset data tokenizer; all four validated via --no-submit.
- **TPU benchmark launched**: `/michaelryan/arch-bench-v6e-1` (v6e-4 us-east5, interactive) →
  `gs://marin-us-east5/benchmarks/arch_inference/v6e-4-run1.json`; monitor armed (5-min poll).
- **ettin17 TPU smoke launched** (`arch-smoke-ettin17-coord`): validates warm-start→train→eval→
  HF-save through the new registry end-to-end before the 1M wave.
- 1M wave plan (fires after gate): ettin17/32/68 + pruned8 + funnelbert + bert @ lr 5e-5
  (warm-started); tiny4/tiny2 + pooled + bigdn @ lr 3e-4 (from scratch); batch 256, 1 epoch,
  1M rows, v6e-4 preemptible, frozen-7k in-run eval; bigdn+bert contingent on gate PASS.
- **TPU BENCHMARK GATE (v6e-4, 2026-08-08 ~04:30 PT)** — mb-base measured 22.07 docs/s/chip
  (matches historical 21.5 ⇒ harness validated). Results (eff docs/s/chip @ deployment):
  pooled_transformer 2040.9 (92.5x) · tiny2 518.9 (23.5x) · bert 221.2 (10.0x) · tiny4 200.3
  (9.1x) · ettin17 177.4 (8.0x) · funnelbert 92.5 (4.2x) · ettin32 88.9 (4.0x) · pruned8 59.2
  (2.7x) · ettin68 37.5 (1.7x) · **bigdn 14.8 (0.67x) FAIL → BRANCH DEAD** (scan latency ate the
  3.1x FLOP margin, as the spike predicted). JSON: gs://marin-us-east5/benchmarks/arch_inference/v6e-4-run1.json.
- ettin17 TPU smoke: COMPLETE end-to-end (warm-start→train→eval→HF export, params 16.86M correct).
- **1M WAVE LAUNCHED** for the 9 survivors via scratch/launch_arch_sweep_1m.sh:
  run ids mb-clf-lpv11-{ettin32,ettin68,ettin17,pruned8,funnelbert,bert,tiny4,pooled,tiny2}-1M,
  v6e-4 us-east5 preemptible, batch 256, 1 epoch over 1M rows, frozen-7k in-run eval.
- 1M wave: all 9 coordinators submitted 03:55-03:58 PT (bash-3.2 assoc-array bug fixed first).
  Monitors armed: 45-min startup check (0-step stall detection) + persistent per-run completion
  events (wandb, 20-min poll, fresh Api per poll).
- **10M decision rule (pre-committed):** 1M F1 >= 0.8424 → launch 10M immediately; F1 >= 0.83 AND
  >= 4x speed → also launch (data scaling closed 0.697→0.722 on hq, may close 1pt here); cap ~3-4
  concurrent 10M runs. Chunked winners (pooled/bert) need a cache build first: pooled → lpv11
  `_clf_token_cache_chunk32768` TreeCache (CPU job); bert → MiniLM-tokenizer cache.
- **INCIDENT + fix (wave, ~04:40 PT): bert + pooled (the two CHUNKED runs) OOM-killed at 0 steps**
  — exit 137 in the child during the ProcessPoolExecutor tokenizer fork: 1M docs (~33GB text)
  in-RAM + fork duplication > the default 128g container. Fix: `--memory-gb 360` (v6e host = 720g),
  baked into scratch/launch_arch_sweep_1m.sh; relaunched as `*-coord-r2` (same run-ids → wandb +
  checkpoints resume). Non-chunked runs unaffected (they stream shards without the fork burst
  at that scale... verify at the 45-min check anyway).
- **Chunked-run pivot (~05:20 PT):** 360g r2 retries were PENDING-futile (autoscaler tier_blocked;
  best free worker 148g). Stopped my r2 coords; pivoted bert+pooled to the STREAMING chunked
  TreeCache path (memory-bounded — also pre-stages 10M for chunked winners). Two cache builds
  launched (cpu64/128g/100g us-east5, resumable): `_clf_token_cache_chunk32768` (ModernBERT tok)
  + `_clf_token_cache_chunk32768_minilm` (MiniLM tok — separate dir; cache metadata mismatch only
  WARNS, so tokenizer-blind dir reuse = silent corruption). Launcher grew `--cache-dir`.
  Relaunch on build completion: `--use-cache --cache-dir <dir>` at default RAM. REMEMBER at
  relaunch: clear bert/pooled from the wave monitor seen-file
  (scratchpad/wave1m_reported.txt) or their eventual completions never get reported.
  Also learned: `iris job status` doesn't exist — use `job bug-report`; job-state checks in earlier
  monitors were dead code (GCS/wandb artifact checks were the real signal — keep it that way).
- 45-min check: all 7 non-chunked runs stepping (62-190 steps). 7 training + 2 pivoting.
- **~05:45 PT: MASS PREEMPTION of the us-east5 preemptible v6e pool** — all 7 running wave children
  hit mid-training (healthy losses, steps 735-1022) AND the 10M baseline (step 13171). Children
  `pending`, iris auto-retries + levanter temp-ckpt resume; NO manual relaunches (duplicate-writer
  rule). Wave monitor v1 killed (crashed→FAILED too trigger-happy for preemptible churn); v2 armed:
  DONE-with-F1 / RECOVERED / stuck-crashed>90min escalation only. 1M runs run WITHOUT babysitters
  (accepted risk: keep:[]+15-min temp ckpts, runs are hours not days; 10M winners WILL get
  babysitters per the known keep:[] preemption-storm hazard).
- **RESULT ettin32 1M (first in): best_f1=0.8161 @ thr 0.36** (3905 steps = full epoch). Misses
  both 10M gates (0.8424 / 0.83+4x). Borderline: -2.6pts F1 for 4x speed. HOLD for full table.
- **Capacity crunch (~10:00-20:00 PT):** post-preemption pool shrank + tier_blocked autoscaler;
  5 runs pending at steps 3300-3616 (85-92% done). Diagnosis: my 2 cpu64 cache builds bin-packed
  onto TPU-host workers = the exact "2 workers short 32 cores" in the pending reason → PAUSED both
  builds (resumable, shards committed) to drain the training queue; relaunch builds after the wave.
  Cache-build monitor stopped meanwhile. (`iris cluster status` RPC parse-fails from this client —
  version skew; use job bug-report for ground truth.)
- **1M RESULTS TABLE (2026-08-09, 7 of 9 in; bar = base 0.8424):**
  ettin68 0.8367 (1.7x) · pruned8 0.8249 (2.7x) · funnelbert 0.8172 (4.2x) · ettin32 0.8161 (4.0x)
  · tiny4 0.8110 (9.1x) · ettin17 0.8068 (8.0x) · tiny2 0.7888 (23.5x) · bert/pooled pending caches.
  PARETO FRONTIER: ettin68 → pruned8 → funnelbert → tiny4 → tiny2 (funnelbert dominates ettin32;
  tiny4 dominates ettin17). SURPRISE: from-scratch tiny4 beats warm-started ettin17 on BOTH axes —
  at 1M in-domain labels, MLM pretraining buys little below ~30M params; bodes well for
  pooled_transformer (also from-scratch).
- **10M PROMOTION: ettin68 launched** (mb-clf-lpv11-ettin68-10M, v6e-4, use-cache, splash, pdp2,
  lr 5e-5, 39062 steps) — deviation from the strict pre-committed rule (0.8367 < 0.8424; 1.7x < 4x)
  justified by user's step-5 delegation + hq data-scaling precedent (+2.5pts 1M→10M); the one
  candidate that can plausibly end BOTH faster and more accurate. 2h-babysitter + completion
  monitor armed. Others HELD pending full table.
- Cache builds resumed as `-r2` (batch+preemptible now — polite to TPU jobs); both RUNNING.
- **Cache-build stall diagnosed (2026-08-09):** levanter build_cache spawns a 40-worker zephyr
  actor group; only 1 actor scheduled on the tier-blocked pool → days-long ETA, zero GCS output
  (`__shards__/` never created). finelog down → artifact-level diagnosis only. PIVOT for the 1M
  runs: `build_chunk_token_cache.py` FLAT caches (single job, no zephyr, ProcessPool tokenize),
  2 jobs launched (`build-flat-cache-{mb,minilm}-1M`, cpu20/160g, rows=1M, cap 32768) → on _DONE,
  pooled+bert 1M run in-RAM chunked at default trainer RAM (~20GB pre-tokenized arrays). TreeCache
  r2 builds left running for the 10M phase. Monitor on `_chunk_token_cache/rows1000000_*/_DONE`.
- **Flat builds ALSO OOM'd (exit 137 @160GB)** — root cause in `_tokenize_docs`: it materializes ALL
  1M docs' text (~33 GB at ~33 KB/doc) then pickles 1/16 slices to a ProcessPool → parent + pickle
  buffers + worker copies > 100 GB. (Log also revealed `os.cpu_count()`=180 = the TPU HOST's cores,
  not the `--cpu` request.) FIX: NEW `experiments/baseline_collection/build_flat_token_cache_streaming.py`
  — streams shard-by-shard (read ~25k docs → tokenize → keep only int32 tokens → free text), peak
  RSS = all-tokens (~20 GB) + one shard, writes the IDENTICAL ids/offsets/labels+_DONE format at the
  same `_token_cache_root` so the trainer loads it transparently. Locally tested (ordering, limit
  across shard boundaries, round-trip through `_load_token_cache`); lint clean. Launched as
  `build-flat-cache-{mb,minilm}-1M-s` (cpu20/200g). NOTE: local ProcessPool tests must live in a
  FILE not a heredoc (macOS spawn re-imports __main__ → BrokenProcessPool).
- **Streaming flat builds OOM'd TOO — then MEASURED the corpus (the step I should have taken first):
  lpv11 survivor docs average 54,138 chars (~13.5k tokens), not the ~33 KB I assumed.** So 1M docs
  = ~54 GB text AND ~54 GB int32 tokens; `_save_token_cache`'s concat doubles that to ~108 GB. ANY
  in-RAM flat cache is unviable at 1M on this corpus — memory tuning was never going to fix it.
- **RESOLUTION (2026-08-09) — pooled needs NO new cache:** train it **truncated @ ctx 8192**
  (pool_window 64 → 128 super-tokens) off the ALREADY-FINISHED 8192-cap TreeCache
  (`_clf_token_cache`, 10,002,709 rows, ModernBERT tokenizer, is_finished=true) — memory-bounded
  streaming, and its deployment recipe (2x4096-token windows) covers the same 8192 tokens, so
  nothing is lost AND it becomes directly comparable to the other archs (all truncated@8192).
  Preset updated; `mb-clf-lpv11-pooled-1M` launched (coord `-r3`, lr 3e-4, pdp 8).
  The same cache serves pooled-10M immediately if promoted → the user's 92x branch is unblocked.
- bert still needs its own tokenizer's cache → `build-clfcache-minilm` (8192-cap TreeCache,
  MiniLM WordPiece, separate dir). bert trains chunked@512 from it (16 chunks x 512 = 8192 = the
  cap exactly). Stopped the two starving 32768-cap chunk TreeCache builds (obsolete under this plan).
- LESSON: measure the data before sizing memory; and prefer the streaming TreeCache over any
  in-RAM cache for anything at >=1M lpv11 docs.

## 🏁 10M RESULTS (2026-08-10)

- **`mb-clf-lpv11-base-10M-c8192` FINISHED: F1 = 0.8684** (1M was 0.8424 → **+2.6 pts** from 10x
  data, matching the hq precedent). **This is the true 10M bar.**
- **`mb-clf-lpv11-pooled-10M` FINISHED (full epoch, 39,061 steps): F1 = 0.8229** — **+3.0 pts over
  its own 1M (0.7925), and it now BEATS fastText (0.8110) by +1.2 pts** at 92-172x ModernBERT speed.
  Data scaling helped the from-scratch model MORE than it helped the pretrained baseline, exactly as
  predicted. (NOTE: correct step count for 10M @ batch 256 is 39,062 — my earlier "97,656" was
  wrong, so a "40%" progress reading was actually a COMPLETED epoch.)
- `mb-clf-lpv11-ettin68-10M`: ~60% (step 23,485). wandb showed `crashed` but the CHILD IS `running`
  — stale heartbeat after a preemption; it resumed itself. Verify child state before ever acting on
  a wandb `crashed`.
- **POOLED-10M CASCADE ANALYSIS DONE — it now DOMINATES fastText at every operating point:**

| metric | fastText w640 | pooled 1M | **pooled 10M** |
|---|---|---|---|
| best F1 | 0.8111 | 0.7933 | **0.8237** |
| junk excl @ R>=0.99 | 44.3% | 54.8% | **55.8%** |
| junk excl @ R>=0.975 | 63.5% | 64.5% | **71.4%** |
| junk excl @ R>=0.95 | **79.3%** | 74.8% (lost) | **81.6%** (now wins) |

  At 1M, fastText still beat pooled on F1 and at R>=0.95; **at 10M pooled wins on ALL FOUR.**
  Cascade `fastText -> pooled10M` at fixed combined recall: **0.98 → 60.8% to 72.1% junk excluded
  (+11.3 pp**, up from +9.6 pp at 1M), scoring 70.5% of docs; **0.99 → +14.5 pp**. Junk-only score
  correlation 0.689 pearson / 0.582 spearman = still substantially decorrelated. Tool verdict:
  "Worth a cascade slot (>= 5 pp)? YES".
  ⇒ pooled_transformer is a REAL WIN: it beats the shipped stage-1 outright AND stacks with it, at
  ~1/172 of a ModernBERT forward. It is NOT a ModernBERT replacement (0.8237 vs base-10M 0.8684).

## 🔬 DIAGNOSTICS: POOLED IS DATA-LIMITED, NOT CAPACITY- OR COMPUTE-LIMITED (2026-08-10)

Three jobs, ~15 TPU-hours total, run to decide whether the 6k-WARC data build is worth it for pooled.

**Speed of the scaled-up variant (v6e-4, `v6e-4-pooled-big.json`) — capacity is nearly FREE:**
| arch | params | docs/s/chip @8192 | vs mb-base |
|---|---|---|---|
| pooled | 25.9M | 4020 | **182x** |
| pooled_big | 62.7M | 3441 | **156x** |
| mb-base | 149.6M | 22.1 | 1x |
**2.4x params costs only 14% throughput.** ms/forward is ~FLAT in ctx for pooled (1.72ms@1k →
1.99ms@8k) vs mb-base 36.5→362ms: the 64-token pooling means the transformer always sees 128
super-tokens, so pooled's advantage GROWS with context (21x @1k, 182x @8k).

**Accuracy levers (all on the same 10M cache, frozen-7k eval):**
| lever | run | F1 | Δ vs pooled-10M (0.8237) |
|---|---|---|---|
| +2.4x capacity | mb-clf-lpv11-pooledbig-10M | **0.8284** | **+0.005** |
| +3 epochs (repeat data) | mb-clf-lpv11-pooled-10M-e3 | 0.8220 | **-0.002 (NOTHING)** |
| +10x FRESH data (1M→10M) | (earlier) | 0.8237 vs 0.7925 | **+0.031** |

**VERDICT: fresh data is worth ~6x more than 2.4x capacity, and repeating data is worth ZERO.**
The 3-epoch null is the load-bearing result: it proves the 1M→10M gain came from NEW information,
not extra gradient steps. ⇒ **The 6k-WARC extraction (→ ~167M survivors) IS the right investment
for pooled**; scaling the architecture is a cheap secondary win (take pooled_big as the default
going forward — +0.005 F1 for 14% speed is a good trade, and it is still 156x mb-base).

## 🔥 TEXT vs HTML: THE REPRESENTATION WAS COSTING ~3 F1 POINTS (2026-08-13)

Every classifier in this project trains on `body_strip` HTML. Measured on the real corpus:
**78.3% of characters are markup**, and **67.0% of docs exceeded the 8192-token window** — i.e. most
training documents were truncated mid-page and the model mostly read tags. Extracting text first
(XenonMolecule resiliparse-rs fork, `main_content=True, preserve_formatting="markdown"`) gives:

| | HTML | TEXT |
|---|---|---|
| mean chars/doc | 53,038 | 3,842 (−92.8%) |
| mean tokens | 19,830 | 1,528 |
| median tokens | 13,210 | 622 |
| **docs truncated @8192** | **67.0%** | **2.8%** |
| dataset | 89.0 GB | 13.4 GB |

**FIRST RESULT — `mb-clf-lpv11-text-pooled-1M` = 0.8207 vs 0.7925 on HTML: +2.8 pts**, far above the
0.015 noise floor. pooled-1M-TEXT (0.8207) ≈ pooled-10M-HTML (0.8237): **changing the representation
bought nearly as much as 10x the data**, and it clears the fastText incumbent (0.8110) at 1M docs.
⇒ Re-examine whether the 90M HTML push is the right investment vs text-at-10M.
Data: `…/presharded_survivor_w640_10M_text/` (10,002,709 docs, verified line-for-line, labels
identical, empty extractions kept as placeholders so index alignment holds) + matching
`full_prep_body_strip_test7k_text/`. Remaining 5 retrains in flight.

## ⚠️ THE REAL BASELINE IS FASTTEXT (2026-08-10) — read every result through this

The 1M F1s were being compared against ModernBERT-base 0.8424, but the shipped **fastText w640 is
the incumbent** and its published 0.7990 was measured on the FULL 89-WARC test, not the frozen 7k —
not comparable. Scored w640 on the IDENTICAL frozen-7k docs
(`score_fasttext_on_bert_test.py` → `…/useful_fasttext_lpv11/eval/ft_w640_on_frozen7k.json`;
6477 docs, 1513 useful = 23.4%):

**fastText w640 on frozen-7k: best F1 = 0.8110 @ thr 0.406.**
High-recall junk exclusion (the cascade-relevant metric): **44.1% @ R>=0.99 · 63.4% @ R>=0.975 ·
79.1% @ R>=0.95.**

Re-ranked 1M table vs the REAL baseline:
| model | F1 | speed | vs fastText |
|---|---|---|---|
| ModernBERT-base | 0.8424 | 1x | +0.031 |
| ettin68 | 0.8367 | 1.7x | **+0.026 BEATS** |
| pruned8 | 0.8249 | 2.7x | **+0.014 BEATS** |
| funnelbert | 0.8172 | 4.2x | **+0.006 BEATS** |
| ettin32 | 0.8161 | 4.0x | **+0.005 BEATS** |
| **fastText w640** | **0.8110** | **CPU ~free** | — |
| tiny4 | 0.8110 | 9.1x | TIE → pointless |
| ettin17 | 0.8068 | 8.0x | -0.004 LOSES |
| pooled | 0.7925 | 92-172x | -0.019 LOSES |
| tiny2 | 0.7888 | 23.5x | -0.022 LOSES |

**Implication: the ultra-fast end of the frontier is dominated by the CPU model we already ship.**
Only the 4 mid-size pretrained encoders earn a TPU. Below ~30M params these models appear to learn
what fastText already captures (bag-of-words-ish signal); the pretrained encoders add the real lift.
USER (2026-08-10): keep all 10M runs going regardless — the finding informs, it does not cancel.

### ...BUT PEAK F1 IS THE WRONG METRIC FOR POOLED — CASCADE ANALYSIS FLIPS IT (2026-08-10)

Scored pooled-1M per-doc on the SAME 6477 docs (`score_pooled_frozen_eval.py` → job
`pooled-score-frozen7k`; order verified aligned with the fastText file element-for-element) and ran
`cascade_analysis.py`. **pooled loses on peak F1 but WINS where a cascade actually operates:**

| model | best F1 | excl@R>=0.99 | excl@R>=0.975 | excl@R>=0.95 |
|---|---|---|---|---|
| fastText w640 | 0.8111 | 44.3% | 63.5% | **79.3%** |
| pooled 1M | 0.7933 | **54.8%** | **64.5%** | 74.8% |

**fastText → pooled cascade at a fixed COMBINED recall budget:**
| comb. R | comb. excl | fastText alone | GAIN | independence | gap | stage-2 load | BERT calls |
|---|---|---|---|---|---|---|---|
| 0.99 | 60.6% | 44.3% | **+16.2 pp** | 70.8% | -0.102 | 70.5% | **-18.9%** |
| 0.98 | 70.3% | 60.8% | **+9.6 pp** | 81.7% | -0.114 | 53.6% | **-13.8%** |
| 0.95 | 82.0% | 79.3% | +2.7 pp | 91.4% | -0.095 | 39.1% | -5.4% |

Errors are only PARTIALLY correlated — junk-only pearson 0.664 / spearman 0.597 (all-doc 0.877 /
0.789). **VERDICT: pooled earns a cascade slot as a HIGH-RECALL (>=0.98) mid-stage**, not as a hard
filter and not as a ModernBERT replacement: it cuts ModernBERT's call volume ~14-19% while itself
costing ~1/172 of a ModernBERT forward. Case collapses below R~0.96 (+2.7 pp), where fastText wins
outright.
Tools (new, reusable): `score_pooled_frozen_eval.py` (reads arch from the ckpt's own config.json →
works for any pooled run) + `cascade_analysis.py` (`--selftest` proves the cascade math: identical
models gain 0.00 pp, complementary models reach 1.0; also sanity-checked by cascading fastText
against itself → exactly 0.00 pp gain). Rerun both against `mb-clf-lpv11-pooled-10M` when it lands.

## POOLED TRAINING SIGSEGV (RESOLVED, 2026-08-09 evening)

Symptom: `mb-clf-lpv11-pooled-1M` (r3, pdp8) and `-r4` (pdp2) both **exit 139 SIGSEGV ~5 min in**,
after `parameter_count=25,947,392` is logged (model builds fine) — i.e. it dies compiling the TRAIN
step, and it is NOT activation size (pdp 8 and 2 fail identically).

Evidence it is an XLA **backward-lowering** bug, not our math:
- Forward at ctx 8192 on v6e is FINE: arch_inference_benchmark cell = **2.1 ms/fwd vs mb-base
  362 ms → ~172x faster at MATCHED ctx** (the 92x headline was conservative: it charged pooled for
  2x4096 windows/doc).
- Forward+backward at ctx 8192 on CPU is FINE: loss 1.099, grad-norm 65.2, all finite.
- Precedent in this repo: TPU XLA `SpatialMajorConvolution` lowering SIGSEGV at ctx>=4096 (the
  chunked-eval bug). Pooled's windowed pooling is exactly a reshape `[b,t,e]->[b,s,w,e]` + reduce
  over a MAJOR axis — the same shape of pattern.

Action: `experiments/baseline_collection/pooled_backward_probe.py` (v6e-4, `pooled-bwd-probe`)
runs the gradient step for 4 pooling variants IN ORDER and writes a `<name>_OK` GCS marker after
each (a segfault kills stdout, so markers are the signal; MISSING marker = the culprit):
`meanmaxmin` (current default) → `mean` → `mean_minor` (transpose so the reduce is minor-most) →
`matmul` (mean via einsum vs a block indicator = pure MXU, no windowed reduce at all).
`mean_minor` + `matmul` are NEW pool kinds added to pooled_transformer.py, verified BIT-IDENTICAL
to `mean` on CPU (max|diff| = 0.0). Markers land in
`gs://marin-us-east5/benchmarks/pooled_backward_probe/`.
### RESOLVED (2026-08-09) — ROOT CAUSE: raw-array params vs gradient accumulation

Probe stage 1 EXONERATED pooling: **all four pool_kinds passed the TPU backward** (markers
meanmaxmin/mean/mean_minor/matmul all written). So the pooling hypothesis was wrong, and the failure
had to be elsewhere in the training path.

Probe stage 2 (`--mode trainer`: bf16_cast → optimizer → grad_accum → real Trainer) **reproduced it
on CPU in seconds**, which made iteration trivial:
```
TypeError: cannot reshape array of shape (50368, 256) into shape (4, 1, 256)
  levanter/grad_accum.py:181 _reshape  <- the EMBEDDING TABLE being reshaped as batch data
```
`trainer.py:776` calls the microbatched grad fn as `fn(model, *batch)`, and
`_reshape_for_microbatch` maps over ALL args: NamedArray leaves are skipped when they lack a Batch
axis, but **every plain `jnp.ndarray` leaf is reshaped unconditionally**. Levanter models keep params
as NamedArrays so they are immune; `pooled_transformer` is the raw-Equinox port whose params are raw
arrays → every weight is treated as batch data. (Fixing the reshape then exposed the same issue one
level down: `hax.fold`'s scan tries to scan each raw param over its leading axis.)

**FIX (no shared-code change): `per_device_parallelism=-1`** → `TrainerConfig.microbatch_size` is
None (trainer.py:942) → no accumulation, no reshape, no scan. Baked into the `pooled` preset with a
comment. Verified end-to-end on CPU: all four probe stages incl. the real Trainer now pass.
I deliberately REVERTED an attempted `grad_accum.py` guard — it did not fix the scan half, and
shared training code is used by the in-flight 10M runs on any relaunch; not worth the risk for a
partial fix. Proper long-term fix = give pooled NamedArray params (then FSDP + accumulation work).
`mb-clf-lpv11-pooled-1M-r5` launched (batch 256, pdp -1, lr 3e-4, ctx 8192, streaming TreeCache).

Also: my CPU probe wrote 296 MB of checkpoints into `./checkpoints/pooled-probe-trainer/`, which
blew the 25 MB iris bundle limit — deleted. Watch for local Trainer runs polluting the repo.

**RESULT: `mb-clf-lpv11-pooled-1M-r5` best_f1 = 0.7925 @ 3905 steps (full epoch).** vs base 0.8424
(-0.050) at **172x faster at matched ctx** (92x charged for 2 windows/doc). Sits at tiny2's accuracy
(0.7888) while being 4-8x faster still. Per the user's pre-authorization ("10M if remotely
promising"), **`mb-clf-lpv11-pooled-10M` LAUNCHED** (same finished 8192 TreeCache, 97,656 steps,
batch 256, pdp -1, lr 3e-4). From-scratch models gain most from 10x data, so this is the run that
decides whether the cascade idea has legs.
STILL TODO for the cascade decision: exclusion-at-recall>=0.99/.975/.95 for pooled (its role as a
fastText -> pooled -> BERT mid-stage is high-recall junk removal, NOT peak F1). Needs per-doc scores
on the frozen 7k — `score_frozen_eval.py` is ModernBERT/HF-shaped; pooled saves eqx via
`save_eqx_classifier`, so it needs a small loader branch (`load_pooled_transformer_classifier`).
- **USER DIRECTIVE (2026-08-09): pooled_transformer is priority.** 92x is strategically big:
  envisioned cascade fastText -> pooled -> BERT (pooled as a new mid-stage after the existing
  fastText stage-1). **Pre-authorized: train pooled on 10M if the 1M result is "remotely
  promising"** (interpretation: useful signal well above fastText-tier, not the 0.84 bar — its
  cascade role is high-recall mid-filtering, so also look at exclusion-at-recall≥.99 when scoring,
  not just best-F1). Path: 1M chunked via streaming `_clf_token_cache_chunk32768` when the mb
  cache lands → if promising, same cache serves the 10M run immediately.
- Baseline health: mb-clf-lpv11-base-10M-c8192 step 20831/39073 (53%), loss 0.262, running.
  INCIDENT (resolved): a failed `git stash push` + reflexive pop applied the user's pre-existing
  `michael-distill: all-wip` stash → conflicts; fully restored (stash intact, no losses). Memory
  written: feedback_git_stash_pop_preexisting_stash.
- **ettin68 timing RE-MEASURED (2026-08-12): 36.4 docs/chip/s @8192, 1.69x — supersedes run1's 37.5.**
  Run1's ettin68 cells were noisy (ctx1024 measured SLOWER than ctx2048, which cannot be real);
  the precise sweep (`--archs ettin68,mb-base --warmup 15 --iters 50` ->
  `gs://marin-us-east5/benchmarks/arch_inference/v6e-4-ettin68-precise.json`) is monotonic, with
  mb-base at 21.6 vs the planner registry's independent 21.5 (0.5% cross-check). The 1.7x headline
  is therefore REAL, not a measurement artifact: ettin68 is h512xL19 vs base h768xL22 — narrower but
  nearly as DEEP, and at 8192 depth sets latency. Params (68M vs 149M) do not predict speed here.
- **mb-clf-lpv11-ettin68-10M is now a Cascade Pipeline Planner stage** (`bert_lpv11_ettin68_10M`,
  col `bert_lpv11_prob_ettin68_10M`, family `modernbert` so it variant-swaps against base-10M).
  Scored over the 100k sample: 99,996 rows, 0 nulls, mean P=0.2135, 19.9% >= 0.5 (lpv11 keeps ~21%).
  No new scorer was needed — the checkpoint is ModernBERT-`model_type`, so `score_modernbert_useful`
  reads its geometry and `classifier_pooling="mean"` straight from `config.json`.
