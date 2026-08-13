# w320 fastText runbook — how the high_quality model was built, and how to redo it for llm_pipeline_v1_1

Research write-up (2026-08-05) of exactly how `body_strip_scale_w320_strat_prep_mc500` was produced,
so the same process can be repeated on llm_pipeline_v1_1 outputs. Sources:
`.agents/projects/{fasttext_data_scaling_sweep,fasttext_useful_classifier_findings,hq3000_distill_and_classifier}.md`,
`experiments/baseline_collection/fasttext_useful_classifier.py`, `scratch/launch_fasttext_*.sh`,
GCS artifacts in `gs://marin-{us-central2,us-east5}/classifiers/useful_fasttext/`.

## What the model is

Stage-1 cascade filter: binary fastText on **raw HTML** (body_strip view) predicting whether the LLM
extractor would keep the page vs output `[NO_USEFUL_CONTENT]`. "w320" = trained on 320 stratified
train-WARC shards. Deployed as fastpipe_v1–v3 stage-1:
`gs://marin-us-east5/classifiers/useful_fasttext/body_strip_scale_w320_strat_prep_mc500/model.bin`,
threshold **0.0368** (high-recall operating point; ~97% recall / ~0.28 precision on the frozen test).
Headline: natural-ratio best F1 **0.637** (thr 0.3, P 0.559 / R 0.741) — vs w80 0.619, shipped front-w80 0.594.
(An `mc1000` variant also exists in us-east5; the deployed one is mc500.)

## The locked recipe (do NOT re-derive — these were all empirically settled)

- Representation: `body_strip` (strip `<script>`, keep `<body>`, collapse whitespace, lowercase).
- **NO truncation** — 8k-char head-truncation cost ~0.10–0.12 F1 (length itself is signal: useful pages are long).
- **Train at the deployment ratio** (`--neg-per-pos` = natural no_useful:useful ratio; was 12 for hq's 11.8:1).
  Ratio is the dominant lever; balanced training costs ~0.12 F1. Don't exceed the natural ratio.
- `--train-sample stratified` (diversity across all CC snapshots breaks the front-sampling plateau at scale).
- Fixed DCLM config, never autotune: `--epoch 5 --lr 0.1 --dim 100 --word-ngrams 2 --minn 0 --maxn 0 --loss softmax`.
- `--min-count 500` → ~1–2 GB deployable model at zero quality loss.
- Eval ONLY on the frozen natural-ratio snapshot-stratified test (`full_prep_*/test/*.txt.gz`,
  35 WARC-disjoint shards), best-F1 over threshold sweep; per-doc scores saved to
  `eval_natural.preds.parquet` for offline re-thresholding (that's where 0.0368 came from).

## Pipeline (4 steps, all in `experiments/baseline_collection/fasttext_useful_classifier.py`)

1. **Labeled dataset build** (hq case: `build_hq_distill_dataset.py`):
   `data/` = kept docs WITH `raw_html` (10.4M); `data_no_useful/` = abstentions WITH `raw_html`
   (~115M, recovered by **set-difference**: fed pages ≤172,032 chars not in the kept set, since the
   extractor never persists drops). One parquet pair per WARC, in **us-central2** (data-local).
2. **`prep` subcommand** (Zephyr, `max_workers=256`, cpu=2/ram=8g per worker, us-central2): per-WARC
   gz text shards `full_prep_body_strip/{train,val,test}/data-{i:05d}.txt.gz` — ALL positives then ALL
   negatives per shard (so a later per-shard neg cap is exact), no balancing. Frozen split via
   `held_out_indices()`: 35 val + 35 test WARCs, one+ per snapshot, WARC-disjoint. Atomic tmp→rename +
   skip-existing = preemption-proof. hq: 2,930 train shards, 764 GiB. ~2 min/WARC.
3. **`train-from-prep` subcommand** (single box): select N train shards (stratified), parallel
   decompress+concat into tmpfs `train.txt` keeping neg_per_pos per shard, train fixed config, upload
   `model.bin` then `metrics.json`. Prep is durable → preemption only loses the train step.
4. **`eval` subcommand** (separate job — never `&&` after train; a preemption during eval re-triggers
   training): threshold sweep vs `full_prep_body_strip/test/*.txt.gz` → `eval_natural.json` + preds parquet.
   Then `leaderboard` to refresh `LEADERBOARD.md`.

## Regions + resources (the part that bit us repeatedly)

- Everything runs where the data lives (**us-central2** for hq) — cross-region dataset reads are forbidden.
- No real CPU VMs in us-central2: `--extra cpu --extra dclm --enable-extra-resources` jobs bin-pack onto
  TPU hosts. Use `--priority batch --preemptible` — non-preemptible CPU jobs get demand-routed onto
  `tpu_v4-reserved_8` (the hero-run pool) and get killed by operators ("Terminated by user").
- `/tmp` is a **RAM tmpfs sized to `--memory`** — train.txt lives in RAM. Size `--memory` ≈ train.txt
  (~33 KB/line untruncated) + fastText + overhead. w80 ≈ 200 GB, w160 ≈ 340 GB.
- **w320 does NOT fit us-central2** (train.txt ~475 GB > v4 host 400 GB RAM). The actual w320 procedure:
  1. In us-central2: `train-from-prep --train-warcs 320 --train-sample stratified --dry-run` → writes
     `full_prep_body_strip/_dryrun_selected_320_stratified.txt` (needs in-region parquet listing for
     snapshot stratification).
  2. Copy ONLY the 320 selected train shards + all 35 test shards (~93 GB) us-central2 → **us-east5**
     (same-continent, ≈$2), preserving `data-{i:05d}.txt.gz` names, into
     `gs://marin-us-east5/classifiers/useful_fasttext/full_prep_body_strip/{train,test}/`.
  3. Run train-from-prep in **us-east5** on a v6e host (720 GB RAM) with
     `USEFUL_FT_ROOT=gs://marin-us-east5/classifiers/useful_fasttext` and
     `--shard-indices <the 320 csv>` (skips the us-central2 parquet reads entirely).
  4. Eval in us-east5 against the copied test shards.
- Launcher templates: `scratch/launch_fasttext_from_prep.sh` (cpu 64, mem per size, disk 20GB,
  batch+preemptible, us-central2). ~hours for train at cpu64; w320 ~10–12 h train+eval.

## EXECUTION LOG (2026-08-05, in progress)

- Built `experiments/baseline_collection/build_lpv11_distill_dataset.py` (+ tests) — the hq clone
  with all deltas below, PLUS a label-leak fix the hq build had: kept pages whose extracted text
  duplicates another kept page are no longer mislabeled negative (assemble emits ALL kept ids;
  only `data/` writing dedups by content id).
- **Raw HTML reality check: the 10k pool was DELETED 2026-04-28** (`10k_warc_cleanup.md`); lpv11
  re-downloaded WARCs itself. Surviving coverage: `baseline_3000_random-34884d` (1423 of lpv11's
  3000) + `baseline_3000-265ff5` (477 more) = 1900, minus 7 gap WARCs in that set → **1893 usable**.
  Decode parity is fine (`run_extract_standalone` imports `_download_one_warc` from
  `download_warcs.py`, the same module that built the pools) — smoke join found 0 null raw_html.
- Output: `gs://marin-us-central2/datasets/llm_pipeline_v1_1_3000_distill/` (same 7-col schema as hq).
- `fasttext_useful_classifier.py` grew `--dataset-root` on pilot/prep/train-from-prep; lpv11
  classifier artifacts go under `USEFUL_FT_ROOT=gs://marin-us-central2/classifiers/useful_fasttext_lpv11`.
- Smoke (3 WARCs, `_smoke` namespace): all 4 stages validated — 0 null html, 0 pos∩neg ids,
  undeduped kept-set honored, shards aligned. Preliminary natural ratio ≈ 3.0:1.
- Full assemble: 1893/1893 shards DONE. htmljoin + negatives launched (us-central2).
- Launcher: `scratch/launch_lpv11_distill.sh <stage>`.

### Dataset + prep FINAL (2026-08-06)
- Dataset stats: **19,934,864 positives / 69,151,658 negatives → natural ratio 3.469:1**
  (train at `--neg-per-pos 3.5`), 2.15 TB parquet, `stats.json` at the dataset root.
- Prep: 1893 shards → **1715 train / 89 val / 89 test** (89 snapshots in this pool, k=1 each).
  Train shards ≈ 400 MB gz / **~2.4 GB uncompressed per WARC** (no fed-length cap → long pages in).
- **w320 is INFEASIBLE untruncated for lpv11**: 320 × 2.4 GB ≈ 770 GB train.txt > 720 GB v6e RAM,
  and every iris worker group has `disk: 100GB` (checked marin.yaml) — no disk path either.
  Flagship = **w256** (~585 GB, --memory 660GB on v6e). In LINES (the axis that matters),
  lpv11-w256 ≈ 12M ≈ hq-w320's 14.4M.
- Stratified selections with the same seed are prefix-nested: w160 ⊆ w256 (verified) — one
  256-shard copy to us-east5 serves both. Copy = 256 train + 89 test + selection file (~137 GB, ~$2.7).

### Training runs (launched 2026-08-06 early AM)
- `lpv11-ft-w80-prep` — us-central2, cpu64/280GB, batch+preemptible. Out: `useful_fasttext_lpv11/body_strip_scale_w80_strat_prep_mc500/`.
- `lpv11-ft-w256-prep2` — us-east5 v6e, cpu64/660GB, **interactive** (user-authorized; batch had 0 free CPU) — RUNNING. Out (us-east5): `body_strip_scale_w256_strat_prep_mc500/`.
- `lpv11-ft-w160-prep` — us-east5, cpu64/450GB, batch+preemptible (insurance rung).
- All: `--neg-per-pos 3.5`, stratified, untruncated, DCLM fixed recipe, mc500.
- **w256 attempts 1+2 (`prep2` 660GB, `prep3` 690GB) BOTH FAILED `[Errno 28]` at the ~34-min mark,
  no preemption.** Root cause (env probe `_envprobe_600g.txt` + iris source):
  **iris mounts `/tmp` with bare Docker `--tmpfs` → tmpfs = 50% of host RAM, NOT the memory request.**
  On v6e-4 (700GiB) that is 355GiB = **381 GB** — a hard wall independent of `--memory`.
  (`lib/iris/src/iris/cluster/runtime/docker.py` ~line 712; `/app` is quota'd to `--disk`, scheduler-capped
  at the worker's 100GB. The old "tmpfs sized to request" probe result was a coincidence: 200GB request on a
  400GB v4 host = 50% RAM.) Exact post-cap sizing (gzip ISIZE footers × manifest counts): w256 = **553 GB**
  → cannot fit ANY worker. **UNBLOCK: patch docker.py to `--tmpfs /tmp:size=<mem request>` + redeploy
  workers (cluster op, needs user), then rerun `prep3`'s exact command — shards/selection already in us-east5.**
  Sizing rule once unblocked: post-cap bytes + ~130GB slack ≤ min(tmpfs, request).
- **De-facto flagship = w160 (350.4 GB post-cap, 31 GB under the wall)** — running. w80 (~176 GB
  vs 214 GB v4 wall) — running. Max-fit today ≈ w165; alternative to discuss: "wide-thin" (more WARCs,
  per-WARC line subsample) trades depth for diversity at fixed bytes — recipe deviation, user's call.
- Evals: run as SEPARATE jobs after model.bin lands (never `&& eval` — preemption re-triggers training).
  us-east5 evals need `-e USEFUL_FT_ROOT gs://marin-us-east5/classifiers/useful_fasttext_lpv11`
  so the default test glob hits the copied test shards.

### RESULTS FINAL (frozen 89-WARC natural-ratio test: 880,407 pos / 3,208,921 neg = 3.64:1)
| model | warcs | train lines | best F1 | excl@R.99 | excl@R.975 | excl@R.95 |
|---|---|---|---|---|---|---|
| w80 (depth) | 80 | 3,354,739 | 0.7919 | 50.1% | 68.8% | 79.0% |
| w160 (depth) | 160 | 6,658,901 | **0.7996** | **52.7%** | **70.1%** | 79.8% |
| w256 sub-0.58 | 256 | 6,002,655 | 0.7964 | 50.7% | 68.5% | 78.9% |
| **w640 sub-0.22 (wide) — SHIPPED** | 640 | 5,633,122 | 0.7990 | 52.1% | 67.6% | **80.4%** |

**DIVERSITY VERDICT (w256 completes the near-equal-lines test, 2026-08-10):** at ~matched lines
(w256 6.00M vs w160 6.66M) MORE WARC DIVERSITY DID **NOT** HELP — w256 0.7964 < w160 0.7996, and
w256 is worse at every high-recall operating point. Combined with w640 (4x the WARCs of w160, also
not better), the hq "diversity breaks the plateau" result does **NOT** transfer to lpv11. Plausible
reason: lpv11's frozen pool spans 89 snapshots and every arm samples stratified across all of them,
so even w80 is already snapshot-diverse; hq's gain came from escaping *front*-sampling bias, which
we never had here. **All four fastText arms sit within 0.008 F1 — fastText is saturated on this task
by ~3.4M lines.** Scale the stage-2 model, not stage-1.

- The wide-thin diversity bet did NOT clearly beat depth (unlike hq front-vs-stratified) —
  though w640 fought with ~15% fewer lines (5.63M vs 6.66M; 0.22 vs equal-lines 0.25 subsample).
- **USER DECISION 2026-08-07: ship w640** (ΔF1 −0.0006 acceptable; prefers its R.95 behavior; running
  at R.99 for now → stage-1 threshold **0.0125**). Model:
  `gs://marin-us-east5/classifiers/useful_fasttext_lpv11/body_strip_scale_w640_sub0p22_strat_prep_mc500/model.bin`.
- **w256 REVISITED 2026-08-10 (goal-directed):** full untruncated w256 (553 GB) stays impossible
  without deploying the iris tmpfs patch (a cluster op needing user sign-off), but the
  `--line-keep-frac` knob built for w640 solves it without touching the cluster:
  **w256 @ keep-frac 0.58 ≈ 321 GB** — 60 GB under the 381 GB v6e tmpfs wall. Launched
  `lpv11-ft-w256-sub058` (interactive, `--memory 550GB` deliberately > v5p's 448 GB host so it
  CANNOT land on v5p, whose 224 GB tmpfs would ENOSPC again). Shards/selection already in us-east5.
  Bonus: at ~6.1M lines vs w160's 6.66M this doubles as the **near-equal-lines diversity rematch**
  (256 WARCs vs 160 WARCs at matched scale) that the w640-vs-w160 comparison couldn't answer.
- Open rematch still not run: equal-lines w640@0.25.

### STAGE-2 ModernBERT pipeline (launched 2026-08-07, mirrors mb-clf-base-10M-c8192)
1. `cascade_survivor_filter.py` (patched: `--num-shards`) — w640 @ 0.0125 over the 1893-shard dataset,
   **excluding the 178 held-out shard indices** (val 89 + test 89; leakage guard), target 10M →
   `gs://marin-us-central2/classifiers/useful_fasttext_lpv11/survivor_w640_thr0p0125/parts` — RUNNING.
2. Frozen 7k ModernBERT test: stratified stride-sample, ~79 docs × 89 test WARCs →
   `…/useful_fasttext_lpv11/full_prep_body_strip_test7k/test_sample_7k.txt.gz` (us-east5) — building.
3. Survivor filter DONE: 345 parts, ~10.0M survivors. Preshard: 4-way parallel
   `sample_preshard_survivors.py` slices (serial was ~10h; distinct `train_shard_p{0..3}` prefixes +
   seeds, `--part-start/end`), written DIRECTLY to us-east5 → **10,002,709 docs in 40 shards** at
   `…/useful_fasttext_lpv11/presharded_survivor_w640_10M/`. TreeCache built at
   `…/presharded_survivor_w640_10M/_clf_token_cache` (trainer auto-derives `<shard-dir>/_clf_token_cache`).
4. TRAINING LAUNCHED: run-id `mb-clf-lpv11-base-10M-c8192` (base, 10M rows, 8192, SPLASH, use-cache,
   batch 256/pdp 2, v6e-8 us-east5), coordinator `/michaelryan/mb-clf-lpv11-10M-coord`. Multi-day;
   wandb `marin-community/modernbert-useful/mb-clf-lpv11-base-10M-c8192`.
   **Launch gotchas:** (a) marin's region check reads `MARIN_PREFIX` (repo .env pins us-central2) —
   pass `-e MARIN_PREFIX gs://marin-us-east5`; (b) do NOT run the launcher on the laptop — local
   gcsfs SSL failure killed the submit (wandb run created then `failed`, exit 0, NO child job);
   submit it as a cluster coordinator job like the hq `*-coord` runs.
   Known hazards in `modernbert_levanter_launch_tracker.md`: v5p preempt-loops (use v6e),
   multi-host quota-blocked (v6e-8 is single-host), shuffle bug (fixed in trainer).
5. **Parallel 1M comparison run DONE (2026-08-08): `mb-clf-lpv11-base-1M-c8192` best F1 0.8424
   @ thr 0.38** on the frozen 7k test (vs w640 fastText 0.799 on the full test — +4.3pts,
   hq-like stage-2 gap). HF ckpt: `gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-base-1M-c8192/hf/`.
6. Ops notes for the long runs: checkpointer is `keep:[]` + 15-min TEMP ckpts only (the known
   loss hazard) — a 2-hourly babysitter snapshots the latest temp ckpt to `<run>/backup/`.
   10M survived **11 preemptions** over ~3 days (incl. one at 97.6%) + a v6e `tier_blocked`
   capacity pause; every resume came off a temp ckpt with ≤15 min lost. Babysitter retired at
   completion.

## ✅ CASCADE COMPLETE (2026-08-10) — lpv11 twin of the hq stage-1/stage-2 filter

| stage | model | F1 (frozen test) | threshold | artifact |
|---|---|---|---|---|
| stage-1 | fastText **w640** sub-0.22 | **0.7990** | 0.0125 (R.99 deploy) | `gs://marin-us-east5/classifiers/useful_fasttext_lpv11/body_strip_scale_w640_sub0p22_strat_prep_mc500/model.bin` |
| stage-2 | ModernBERT-base **1M** | **0.8424** | 0.38 | `gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-base-1M-c8192/hf/` |
| stage-2 | ModernBERT-base **10M** | **0.8684** | 0.40 | `gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-base-10M-c8192/hf/` |

- ModernBERT F1s are on the frozen **7k** stratified sample; fastText F1 on the full 4.09M-row test.
  Both drawn from the same 89 held-out WARCs (WARC-disjoint from all training data).
- Data scaling 1M→10M = **+0.026** (0.8424→0.8684), same shape as hq's 200k→1M (+0.023) — still climbing.
- Stage-2 over stage-1 = **+0.069** at 10M, mirroring hq's +0.070 (fastText 0.597 → BERT 1M 0.667).
- NOT yet done: cascade operating-curve analysis (stage1@R.99 → stage2 exclusion at matched recall)
  — needs per-doc preds like `scratch/bert_curves/`; the deployed spec threshold pair is the output.

## Adapting to llm_pipeline_v1_1 — the real deltas

1. **No labeled dataset exists yet — build it first.** lpv11 (`run_extract_standalone.py --pipeline`,
   `_process_batch_pipeline`) persists **KEEP docs only** (text + `warc_record_id`/`warc_file` join keys,
   NO raw_html); drops are counted, not written. So replicate the hq recipe:
   positives = kept docs joined back to raw HTML by normalized `warc_record_id`;
   negatives = fed pages NOT in the kept set (set-difference), both carrying `raw_html`, one parquet
   pair per WARC. Clone `build_hq_distill_dataset.py` (esp. its `negatives` stage).
   - Raw HTML source = the 10k pool `gs://marin-us-central2/raw/commoncrawl/dclm_400m_1x_10k-ee2365/`.
   - **VERIFIED 2026-08-05: the first 3000 random WARCs are already consolidated** (pre-dedup) into
     `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/{region}/…` with
     manifest `resolved/resolved_llm_pipeline_v1_1.jsonl.gz` — 3000/3000 done WARCs, 552,085 unique
     batch shards, the 26,415 cross-region duplicate keys ALREADY resolved, 0 invalid. This kills the
     two worst prep headaches (5-region fragmentation, steal-mode dup keys); read the kept set via the
     resolved manifest, all us-central1-local. This is enough corpus: same size as hq's (which also
     used 3000 with 320 train + 70 held-out), and lpv11 has ~2.5× kept docs/WARC (~8.9k vs ~3.5k), so
     w320@natural-ratio ≈ 15M lines ≈ hq w320's 14.4M.
   - **8 WARCs have batch index gaps** (`missing_batches_llm_pipeline_v1_1.jsonl.gz`): their missing
     batches' true-kept docs would be mislabeled negative by set-difference — EXCLUDE those 8 WARCs.
   - Kept-set join keys must move us-central1 → us-central2 for the set-difference against raw HTML
     (keys only, a few GB — same pattern as the hq build, <$1).
   - Watch DECISION_ERROR docs (in timing sidecars): errors are NOT drops — exclude them from both classes.
2. **Recompute the natural ratio.** lpv11 keeps ~19% (48.3/250 per batch) vs high_quality's ~7.8% →
   natural ratio ≈ **4.3:1**, not 12:1. Set `--neg-per-pos` and build the frozen test at THAT ratio
   (the ratio-matching principle, not the number 12, is what's locked).
3. **Regenerate the frozen split** for the lpv11 WARC set (same `held_out_indices` snapshot-stratified
   mechanism). Keep it WARC-disjoint. If comparing head-to-head with the hq classifier, consider reusing
   overlapping held-out WARCs, but the manifests differ (three 3000-WARC manifests, ∩=1423).
4. **Sizing:** at similar per-WARC density, w320-scale training again needs the us-east5 v6e detour;
   with a 4.3:1 ratio train.txt is ~2.6× smaller per WARC than at 12:1, so w320 may fit a 340–400 GB
   us-central2 box — recompute from actual prep-shard sizes before deciding to copy.
5. **Deployment threshold:** re-derive from the new `eval_natural.preds.parquet` at the recall target
   (hq used ~97% recall → 0.0368), don't reuse 0.0368.

## 🆕 SCALE-UP PLAN: 6,415-WARC lpv11 → ~90M-doc stage-2 dataset (2026-08-10)

**Trigger:** lpv11 extraction reached **6,415 WARCs** (all `_completed` registered in us-central1 —
single-region this time, unlike the old 3,000 which spanned 5). User wants ~100M training rows for a
NEW (non-ModernBERT) classifier arch that is ~172x faster than ModernBERT.

### The binding constraint is STILL raw HTML, not extraction
The classifier trains on `body_strip(raw_html)`; extraction output keeps only extracted `text`.
The 10k pool was deleted 2026-04-28, so coverage comes only from the two surviving us-central2 pools
(`baseline_3000_random-34884d` + `baseline_3000-265ff5`, 5,112 unique hashes). Measured 2026-08-10:

| set | count |
|---|---|
| lpv11 extracted (`_completed`) | 6,415 |
| …with surviving raw HTML | **3,290** |
| NEW beyond the old 3,000 | 3,415 |
| …of those, with raw HTML | **1,390** |

**Path A (CHOSEN, no CC re-download):** 3,290 − 178 held-out − ~15 gap ≈ **3,097 train WARCs**
× **28,993 survivors/WARC** (measured: 10,002,709 survivors / 345 WARCs at w640 thr 0.0125)
= **~90M survivors**. Path B (re-download the 3,125 HTML-less WARCs from CC) would reach ~180M;
not chosen, but the pipeline takes it unmodified.

### 🔒 FROZEN-TEST INVARIANT (user directive)
REUSE the existing 89 val + 89 test WARCs **verbatim**; all new WARCs are **train-only**.
Do NOT re-derive the split — re-deriving reshuffles the held-out set, invalidates every measured
number (4 fastText + 2 ModernBERT), and can leak old test WARCs into new training data.

### 💰 EGRESS: the permanent fix is to stop copying central→east
Measured per-10M artifact sizes: token cache **76.9 GB**, presharded text **89 GB**,
survivor parts **0.248 GB/WARC**, labeled parquet **1.22 GB/WARC**, **47,061 docs/WARC**.
At 90M: text ~801 GB + cache ~692 GB = 1,493 GB → **$14.9 @ $0.01/GB, $29.9 @ $0.02/GB
(BREAKS the $25 cap)**. NOTE: a web search wrongly returns the tiered **internet** egress rate
($0.12/0.11/0.08 per GB) — that is NOT inter-region and would be a ~$179 mis-estimate.
**Fix: site ALL lpv11 classifier compute in us-central2**, where the ~2 TB of immovable raw HTML
already lives (moving the HTML instead would be ~$80-100). us-east5 was only ever used because the
hq ModernBERT stack lived there; the 172x-faster arch removes the need for v6e. Residual egress =
the assemble stage reading us-central1 extraction text ≈ 20 GB ≈ **$0.20-0.40/run**.

### ⚠️ The token cache is TOKENIZER-SPECIFIC
`_clf_token_cache` was built with `answerdotai/ModernBERT-base`. A different arch/tokenizer makes it
worthless. The **presharded TEXT shards (`__label__x <text>`) are the durable, tokenizer-agnostic
artifact** — build those unconditionally; build a cache only once the arch's tokenizer is known
(and possibly not at all, if the arch tokenizes on the fly).

### Pipeline (all us-central2 unless noted)
1. **Consolidate 6,415** (us-central1, intra-region, no egress): `consolidate/launch_inventory.py`
   → `launch_resolve.py` (`--spec llm_pipeline_v1_1`) → fresh `resolved_/duplicates_/
   missing_batches_/integrity_report_` so NEW gap-WARCs get excluded like the original 7.
   NOTE both launchers require `IRIS_CONTROLLER_ADDRESS` → must run AS an iris job, not on the laptop.
2. `build_lpv11_distill_dataset.py` for the +1,390 new HTML-covered WARCs (+1.7 TB parquet).
3. fastText `prep` for the new WARCs (+0.6 TB) — train split only.
4. `cascade_survivor_filter.py --num-shards <N>` @ w640 thr 0.0125, excluding held-out → ~90M (~0.8 TB).
5. `sample_preshard_survivors.py` in **4+ parallel part-slices** (serial is ~10h; use distinct
   `train_shard_p{i}` prefixes + seeds + `--part-start/--part-end`) → ~0.8 TB text shards.
6. Token cache — DEFERRED pending the new arch's tokenizer.

Total new storage ≈ 3.8 TB, all in-region.

### ⚠️ ZEPHYR FAN-OUT FAILS UNDER CPU CONTENTION — use `--standalone-workers` (2026-08-11)
The extension assemble burned ~6h across FOUR failed launches before producing a shard. Sequence,
because each looked like a different bug but two were self-inflicted:
1. **Missing transfer step.** Ran consolidate `inventory` + `resolve` but NOT `transfer`, so
   `remap_to_consolidated()` pointed at us-central1 archive paths that were never mirrored for the
   new WARCs (their data is in marin-us-west4 / marin-eu-west4). 1h of reads against nonexistent
   paths, 0 shards. → added **`--read-origin`** (read the manifest's canonical ORIGIN paths).
   Chose this over running `transfer`: origin-direct moves **34 GB** (only the 1,402 usable WARCs)
   vs **80 GB** for a full rsync — and 38 GB of that rsync is europe-west4→us-central1, a
   CROSS-CONTINENT hop. ~$2.20 vs ~$5.44.
2. **Preempt-loop.** Coordinator preempted 3x in 4h; each eviction killed its Zephyr children
   before ANY shard existed, so `skip_existing` had nothing to resume and every restart went to 0.
3. **`--no-preemptible` + oversized request = unschedulable.** us-central1's only on-demand CPU pool
   is `e2-highmem-2` (2 vCPU / ~14 GiB); a `--cpu 4 --memory 16GB` non-preemptible request can never
   fit. Sat pending forever. (Also: making the COORDINATOR non-preemptible forces its Zephyr WORKERS
   onto the same tiny pool → 1 worker.)
4. **Reserved-capacity squat.** CPU-only iris jobs DEFAULT to non-preemptible → bin-packed onto
   `marin-tpu-v4-reserved-2048-us-central2-b` (hero-run capacity). Passing `--preemptible` on the
   PARENT did **not** move the workers — Zephyr worker actor-groups are scheduled from their own
   `ResourceConfig` and inherit neither `--extra` nor capacity type.

**ROOT CAUSE (all four): Zephyr needs the scheduler to place up to 200 worker actor-groups; a
contended cluster places ONE.** Region and preemptibility flags cannot fix contention.
**FIX: `assemble --standalone-workers 64`** on a single `--cpu 64` box — one scheduling decision
instead of 200, threads (reads are GCS-bound, GIL released), same per-WARC filenames + skip-existing
so it is interchangeable/resumable with the Zephyr path. **0 shards in 6h → 50 shards in 5 min.**
Use this shape for the remaining stages (htmljoin/negatives/prep) rather than re-fighting the scheduler.

**Monitoring lesson:** a blank `state=` from `iris job list` means the CLI was re-establishing its SSH
tunnel and returned log noise, NOT that the job vanished. Also: a Monitor loop's first tick fires
immediately — it is a t=0 sample, not a t=interval one (I misread this once and cried stall).

## ✅ 90M STAGE-2 DATASET BUILT (2026-08-11)

**Result:** `gs://marin-us-central2/classifiers/useful_fasttext_lpv11/presharded_survivor_w640_90M/`
— **88,839,133 docs**, 96 shard files, **790 GB**, label mix **36.8% useful / 63.2% no_useful**
(~32.7M / 56.1M). 8.9x the previous 10M corpus. Token cache (ModernBERT-base, 8192) alongside at
`_clf_token_cache`. Built from 3,117 train WARCs = 1,715 (orig root) + 1,402 (ext root).
Total egress for the whole scale-up ≈ **$2.30** (one-time origin-direct read of extraction batches).

**Frozen test PROVABLY untouched:** ext prep ran `--k-holdout 0` → 1,402 train / **0 val / 0 test**
(monitor asserted this continuously). Ext WARCs are disjoint from the original 1,893 by allowlist
construction. All six previously-measured model numbers remain comparable.

### ⚠️ TRAINERS MUST SHUFFLE
Every shard is internally CLASS-ORDERED (a WARC's positives, then its negatives) — inherited from the
prep shards. Reading sequentially gives class-homogeneous batches → loss collapses to ~0 (this exact
bug bit the 2026-06 ModernBERT runs). Any `--train-rows` prefix IS a valid uniform sample of WARCs;
it is the within-shard order that must be randomized.

### THE BIG PERF LESSON: read the right artifact
`cascade_survivor_filter.py` originally re-read `raw_html` parquet (~2.4 GB/WARC) and re-applied
`body_strip` — 7.5 TB of reads for 3,117 WARCs, ~37 min/shard, ETA 60h. The **prep shards already
hold exactly that output as text (~400 MB/WARC)**. Added **`--from-prep <full_prep_root>`**:
~6x less I/O, no regex. Measured **0.75 → 30 shards/min**. Use it always.
(Caveat: the parquet path caps html at 1 MB before body_strip, prep does not → marginal score
differences on enormous pages. The 90M corpus has 533 parquet-path parts + 2,584 prep-path parts;
far below R.99 threshold sensitivity, but noted for provenance.)

### Scaling shapes that worked (contended cluster, 2026-08-11)
| stage | shape | note |
|---|---|---|
| assemble | 1 box, **64 threads** | GCS-I/O-bound, gzip releases GIL → threads fine. 0→14 WARCs/min |
| htmljoin / negatives | N boxes, **48 PROCESSES** | `json.loads` on ~47k recs/WARC is GIL-bound; threads collapsed to ~1 core. 1→57/min |
| survivor filter | 4 boxes × 24 procs, `--from-prep` | I/O-bound; **14 boxes CAUSED COLLAPSE** (bucket throttling) — fewer, leaner readers win |
| preshard | **24 disjoint slices × 1 core** | single-threaded gzip; slices MUST be disjoint (no skip-existing, own prefix → overlap DUPLICATES docs) |

**Zephyr fan-out is unusable on a contended cluster** — it needs the scheduler to place up to 200
worker groups and gets 1. Every stage now has a `--standalone-workers` escape hatch.

### 🐚 zsh TRAPS HIT (three, all cost real time)
1. `${6:+--flag "$6"}` → flag+value became ONE argv token → argparse "unrecognized arguments".
2. `set -- $spec` and unquoted `$COMMON` **do NOT word-split in zsh** → garbled job names, empty args.
3. **`gsutil cat "$D/pre"*"_progress.txt"` → zsh globs the gs:// path LOCALLY, fails `no matches found`**,
   command dies before gsutil runs → monitor silently reported 0 progress. **Quote gs:// wildcards end-to-end.**
Write multi-job launches into a `.sh` and run with `bash`, not inline in this shell.

### ⛔ TOKEN CACHE BLOCKED (2026-08-11) — levanter fan-out has no escape hatch
`build_clf_cache.py` → `build_or_load_cache` spawns its own
`ZephyrContext(max_workers=min(128, len(shard_jobs)))` **inside levanter**
(`lib/levanter/src/levanter/store/cache.py` ~line 326). On the contended cluster it got **1 worker**
and wrote **0 bytes in 1 hour**; the autoscaler added nothing. Unlike every other stage, there is no
`--standalone-workers` flag to pass — the fan-out is hardcoded. Job stopped (it was squatting on
`tpu_v4-reserved-2048` with no output). **No partial state — a rerun starts clean.**

Options when picking this back up:
1. **Retry when the cluster drains** — same command, nothing lost.
2. **Tokenize on the fly from the text shards** — they are the durable, tokenizer-agnostic artifact
   (790 GB, 88.8M docs). For a fast arch this may beat a 690 GB cache anyway.
3. **Add a process-pool path to levanter's cache builder** (mirror the `--standalone-workers` pattern);
   the clean fix, but it touches shared levanter code used by the LM path too.

Command to resume (unchanged):
```
uv run iris --cluster marin job run --no-wait --cpu 64 --memory 128GB --disk 100GB \
  --priority interactive --preemptible --extra cpu --enable-extra-resources \
  --region us-central2 --job-name lpv11-cache-90m -e HF_TOKEN <tok> -- \
  python experiments/baseline_collection/build_clf_cache.py \
    --train-glob "gs://marin-us-central2/classifiers/useful_fasttext_lpv11/presharded_survivor_w640_90M/train_shard_*_[0-9][0-9].txt.gz" \
    --cache-dir  "gs://marin-us-central2/classifiers/useful_fasttext_lpv11/presharded_survivor_w640_90M/_clf_token_cache" \
    --max-length 8192
```
