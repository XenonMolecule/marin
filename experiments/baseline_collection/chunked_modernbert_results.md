# Chunked-document ModernBERT useful-classifier — RESULTS & PROVENANCE

Design doc (the "why" + locked decisions): `.agents/projects/chunked_modernbert_classifier.md`.
This file is the durable **results ledger + run registry + provenance** so we never lose numbers or
forget how a run was produced. Update the registry table when a run reaches an F1.

## What "chunked" means (one paragraph)
Instead of truncating each doc to the first `ctx` tokens, we split it into `ctx`-sized **chunks** and
train the classifier so EACH chunk carries the document's label (plain multiple-instance learning).
At eval we score every chunk of a doc and **aggregate** (mean/max/min/top-k) into a doc-level
keep/drop, sweeping aggregation × threshold for best F1. Training caps at **16 chunks/doc**
(random-sampled, seeded, when a doc has more); **inference uses the full doc (all chunks, no cap)**.
Tiling is swept: **non-overlap** (stride = ctx) vs **50% overlap** (stride = ctx/2).

## Provenance — code (commit before launch: `e6aa1c2ee7` + uncommitted chunked edits)
- **Dataset / training**: `lib/levanter/src/levanter/main/train_classifier.py`
  - `ChunkedTextClassificationDataset` — pre-tokenizes each doc once (to `max_doc_tokens=32768`),
    builds a flat `(doc_idx, start)` chunk index, seeded random-subset to `max_chunks_per_doc`.
  - `chunk_starts(n_tokens, stride)` — window start offsets.
  - `score_texts_chunked(...)` + `aggregate_sweep(...)` + `CHUNK_AGGREGATORS` (mean/max/min/top2mean/top3mean).
  - `ClassificationDataConfig` new fields: `chunked`, `max_chunks_per_doc=16`, `chunk_overlap`,
    `chunk_sample_seed=0`, `max_doc_tokens=32768`; `chunk_stride(Pos)` helper; `build_train` branch.
  - `main()` recomputes `num_train_steps = n_chunks // batch` worker-side (1 epoch over chunks) and
    runs the chunked eval (`eval/{agg}_f1`, `eval/{agg}_threshold`, `eval/best_f1`, `eval/best_agg`).
- **Launcher**: `experiments/baseline_collection/launch_modernbert_levanter.py`
  - New flags: `--chunked`, `--max-chunks-per-doc 16`, `--chunk-overlap {0.0|0.5}`.
- **Tests** (all green): `lib/levanter/tests/test_modernbert_classifier.py` — `test_chunk_starts*`,
  `test_chunked_index_counts_and_cap`, `test_chunked_sampling_is_deterministic`,
  `test_chunked_overlap_doubles_chunks`, `test_chunked_get_batch_content_and_label`,
  `test_aggregate_sweep_prefers_separating_aggregator`, `test_f1_sweep_perfect_separation`.

## Provenance — data & eval (us-east5, in-region; NO cross-region reads)
- **Train**: first 200k rows of
  `gs://marin-us-east5/classifiers/useful_fasttext/presharded_survivor_1M_w4/train_shard_*.txt.gz`
  (in-memory, FULL body text — the 8192-token cap only ever bit the `--use-cache` 10M path).
- **Eval**: frozen-7k stratified sample of
  `gs://marin-us-east5/classifiers/useful_fasttext/full_prep_body_strip/test/*.txt.gz` (seed 0,
  516 useful). Deterministic → comparable to every prior ModernBERT number.
- **Checkpoints out**: `gs://marin-us-east5/checkpoints/modernbert-useful/<run-id>/` (`hf/` = final).

## Launch command template (Iris coordinator → nested TPU `train_classifier`)
```bash
uv run iris --cluster marin job run --region us-east5 \
  --cpu 4 --memory 8GB --disk 10GB --priority interactive --no-wait \
  --job-name <run-id>-coord \
  -e WANDB_API_KEY <key> -e HF_TOKEN <token> \
  -- python -m experiments.baseline_collection.launch_modernbert_levanter \
       --run-id <run-id> --model-size <base|large> --train-rows 200000 \
       --max-seq-len <ctx> --chunked --chunk-overlap <0.0|0.5> \
       --per-device-parallelism <pdp> --attn-backend <vanilla|splash> \
       --tpu-type v6e-4 --region us-east5
```
`pdp` (v6e-4): base 512→16 1024→8 2048→4 4096→2 8192→2 · large 512→8 1024→4 2048→2 4096→1 8192→1.
`attn-backend`: vanilla for ctx ≤ 4096; **splash** for ctx 8192. Steps are recomputed worker-side
from the chunk count (1 epoch), so the launcher's printed `steps=` is just a placeholder.

## Baselines to beat — TRUNCATED, frozen-7k best-F1 (from prior context-scaling runs)
| size  | data | ctx 512 | ctx 1024 | ctx 2048 | ctx 4096 | ctx 8192 |
|-------|------|--------:|---------:|---------:|---------:|---------:|
| base  | 200k |    —    |  0.3549  |  0.4412  |  0.5456  |  0.6454  |
| base  | 50k  |    —    |  0.2376  |  0.3274  |  0.4509  |  0.5635  |
| large | 200k |    —    |    —     |    —     |    —     |  0.6716  |
Reference (NOT this regime): base data-scaling @8192 = .564/.645/.697/.7131/.7219 @50k/200k/1M/5M/10M.
**Hypothesis**: chunked-512 ≫ truncated-512; chunked small-ctx approaches truncated-8192 (~0.645)
because it sees the WHOLE doc in pieces, at a fraction of the per-step cost.

## Run registry (200k pilot)
Status: `queued`→`compiling`→`training`→`done`/`failed`. Fill F1 when `eval/best_f1` lands on wandb.
wandb project: `modernbert-useful`. best_agg ∈ {mean,max,min,top2mean,top3mean}.

| run-id | size | ctx | tiling | pdp | attn | tpu | status | best_f1 | best_agg | per-agg F1 (mean/max/min/t2/t3) | notes |
|--------|------|----:|--------|----:|------|-----|--------|--------:|----------|--------------------------------|-------|
| mb-clf-base-200k-c512-chunk-no   | base  |  512 | non    | 16 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-base-200k-c512-chunk-ov   | base  |  512 | 50%    | 16 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-base-200k-c1024-chunk-no  | base  | 1024 | non    |  8 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-base-200k-c1024-chunk-ov  | base  | 1024 | 50%    |  8 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-base-200k-c2048-chunk-no  | base  | 2048 | non    |  4 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-base-200k-c2048-chunk-ov  | base  | 2048 | 50%    |  4 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-large-200k-c512-chunk-no  | large |  512 | non    |  8 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-large-200k-c512-chunk-ov  | large |  512 | 50%    |  8 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-large-200k-c1024-chunk-no | large | 1024 | non    |  4 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-large-200k-c1024-chunk-ov | large | 1024 | 50%    |  4 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-large-200k-c2048-chunk-no | large | 2048 | non    |  2 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-large-200k-c2048-chunk-ov | large | 2048 | 50%    |  2 | vanilla | v6e-4 | LAUNCHED | | | | wave1 |
| mb-clf-base-200k-c4096-chunk-no  | base  | 4096 | non    |  2 | vanilla | v6e-4 | wave2 @5% | | | | |
| mb-clf-base-200k-c4096-chunk-ov  | base  | 4096 | 50%    |  2 | vanilla | v6e-4 | wave2 @5% | | | | |
| mb-clf-base-200k-c8192-chunk-no  | base  | 8192 | non    |  2 | splash  | v6e-4 | wave2 @5% | | | | |
| mb-clf-base-200k-c8192-chunk-ov  | base  | 8192 | 50%    |  2 | splash  | v6e-4 | wave2 @5% | | | | |
| mb-clf-large-200k-c4096-chunk-no | large | 4096 | non    |  1 | vanilla | v6e-4 | wave2 @5% | | | | |
| mb-clf-large-200k-c4096-chunk-ov | large | 4096 | 50%    |  1 | vanilla | v6e-4 | wave2 @5% | | | | |
| mb-clf-large-200k-c8192-chunk-no | large | 8192 | non    |  1 | splash  | v6e-4 | wave2 @5% | | | | |
| mb-clf-large-200k-c8192-chunk-ov | large | 8192 | 50%    |  1 | splash  | v6e-4 | wave2 @5% | | | | |

## How to read a finished run
On wandb (`modernbert-useful`, run = run-id): `eval/best_f1`, `eval/best_agg`, and per-aggregator
`eval/{mean,max,min,top2mean,top3mean}_f1` + `_threshold`. The winning aggregator tells us how to
combine a doc's chunk scores at deployment time. To re-score / build operating curves later, extend
`experiments/baseline_collection/score_frozen_eval.py` with the same `aggregate_sweep`.

## Launch log
- **2026-06-29 ~15:01 PT** — wave 1 (12 small-ctx runs) submitted as Iris coordinators
  (`<run-id>-coord`, interactive, us-east5). All 12 coordinators running; children submitted as
  `.../train_classifier`. us-east5 v6e currently saturated → most children **pending** ("Insufficient
  TPUs, available 0"), scheduling as preemptible v6e frees. Canary `base-200k-c2048-chunk-no` got a
  v6e-4 (us-east5-b) and is in startup. (finelog log plane was DOWN at launch — validate via wandb.)
- **Wave 2** (8 runs: base+large × ctx{4096,8192} × {non,50%}) to fire once any checkpoint hits ~5%
  (per user). c4096 → vanilla; c8192 → splash. pdp: base 4096→2 8192→2, large 4096→1 8192→1.

## INCIDENT 2026-06-29 — wave 1 stuck in startup; root cause = eager tokenization
- **Symptom**: all 12 wave-1 runs `running` but **0 steps / 0 checkpoints after ~70 min**; healthy
  workers, 0 preemptions, 0 crashes. (finelog log plane was down cluster-wide → no worker logs.)
- **Root cause**: the chunked dataset tokenized all 200k docs **eagerly in `__init__`**. On a TPU
  worker HF disables tokenizer parallelism after the JAX fork → single-threaded → ~1 hr, repeated
  per-run (×12) and on every preemption. (The truncated path tokenizes lazily → reached step 0 in 77s.)
  Measured tokenize rate: 203 docs/s multi-core (laptop) → ~16 min; far slower single-threaded.
- **Fix (shared token cache)**: `build_or_load_token_cache` tokenizes ONCE to flat GCS arrays
  (`ids/offsets/labels.npy` + `_DONE`); every run/resume loads in seconds. Cache key excludes
  ctx/stride/seed → **one cache serves all 20 runs** (base+large share the tokenizer). Prebuilt by a
  CPU job (`build_chunk_token_cache.py`) where parallelism stays on. Dataset now takes pre-tokenized
  `doc_ids`. Tests: +`test_token_cache_roundtrip` (8 pass total).
- **Action**: killed all 12 (`--include-children`); prebuilding cache; then **canary-first** (one run
  to TPU to confirm it actually steps+checkpoints — that never happened before), then relaunch all 20.

## STREAMING RELAUNCH — pilot live (2026-06-30 03:10 UTC)
After the flat in-RAM cache OOM'd, pivoted to the **streaming `TreeCache`** path (the canonical one now):
- `ChunkedCachedClassificationDataset` reads tokens by index from a `TreeCache`; chunk index built from
  per-doc lengths via the jagged-array offsets (no token-data read); labels read from the scalar field.
  Wired to `--chunked --use-cache`. Integration-tested vs a real TreeStore (`test_chunked_cached_*`).
- **Two 32768-cap caches built** (zephyr, parallel+resumable), both `is_finished=True`:
  `…/presharded_survivor_1M_w4/_clf_token_cache_chunk32768` (1,000,000 rows) and
  `…/presharded_survivor_random_par/_clf_token_cache_chunk32768` (10,000,756 rows, 40 shards).
  Build jobs were operator-reclaimed at the cleanup stage but the caches finished first (intact).
- **Canary** `mb-clf-canary-c2048-stream` (throwaway id) on the 10M cache → **stepped in ~11.6 min**
  (params=149.6M logged → dataset+model init OK), validating the streaming path on TPU.
- **DECISION (timestamped): fleet on 10M(random_par, 200k subset)** — one consistent pool for
  pilot→10M. All **20 runs launched** 03:07–03:10 UTC, job-names `<run-id>-coordF`, `--use-cache`,
  `--train-glob …random_par…`, `--train-rows 200000`. 20 coords running, children queued on v6e.
- Data note: pilot trains on **random_par (200k subset)**, not 1M_w4. Frozen-7k eval is unchanged, so
  F1s are comparable as absolute numbers; truncated baselines were on 1M_w4 — for a perfectly matched
  200k truncated baseline, re-run truncated-200k on random_par (cheap) if needed.
- **Scale plan (user, 2026-06-30): 200k pilot FIRST → pick winning ctx/tiling → scale only the best
  2–3 configs to full 10M** (`--train-rows 10000000`, same cache, no rebuild). Do NOT launch all 20 at 10M.

## RESULTS (2026-06-30, updating as runs finish) — frozen-7k best-F1, chunked eval
Train = random_par 200k subset; eval = frozen-7k (516 useful). best-agg in parens.
| config | best-F1 | best agg |
|--------|--------:|----------|
| base  c2048 non-overlap | **0.5602** | top3mean |
| canary c2048 non-overlap | **0.5602** | top3mean | (= base-c2048-no exactly → reproducible) |
| base  c1024 non-overlap | 0.4932 | max |
| large c512  non-overlap | 0.4917 | top2mean |
| base  c512  non-overlap | 0.4777 | top3mean |
| base  c512  50%-overlap | 0.4753 | top3mean |
| base  c1024 50%-overlap | 0.4682 | top2mean |
(c4096/c8192 + remaining large runs still training.)

Truncated base@200k baselines (1M_w4): c1024=0.355 c2048=0.441 c4096=0.546 c8192=0.645.

### CHUNKED vs TRUNCATED context sweep (base @200k, frozen-7k best-F1)
Quick comparison — truncated trained on 1M_w4, chunked on random_par (longer docs); suggestive, not matched.
| ctx | truncated (1M_w4) | chunked-no (random_par) | Δ |
|----:|------------------:|------------------------:|----:|
| 512 | — | 0.478 | — |
| 1024 | 0.355 | 0.493 | +0.138 |
| 2048 | 0.441 | 0.560 | +0.119 |
| 4096 | 0.546 | **0.6455** | **+0.099** |
| 8192 | 0.645 | **0.657** | **+0.012** |
**COMPLETE base curve (chunked-no): 512=0.478 1024=0.493 2048=0.560 4096=0.6455 8192=0.657** — monotone,
dominates truncated everywhere, gains shrink as ctx grows. Large-no so far: c512=0.492 c1024=0.480 c2048=0.577.

⚠️ **c4096 eval bug + fix:** chunked end-of-run eval at **ctx≥4096 with VANILLA** attention SIGSEGVs the TPU
XLA compiler (`SpatialMajorConvolution` lowering, batch-16 eval forward). **Fix = eval with SPLASH** (proven at
c8192). Affected runs relaunched with `--attn-backend splash` (resume from checkpoint → eval clean). Launcher
now auto-forces splash for chunked ctx≥4096 (protects the 10M scale-up). base-c4096-no via splash = 0.6455.
**Takeaways:** (1) **chunked > truncated at EVERY matched ctx, including beating the best truncated model**
(chunked-8192 0.657 > trunc-8192 0.645); (2) gains are largest at small ctx (+0.14 @1024) and shrink as docs
fit the window (+0.01 @8192) — chunking recovers what truncation discards; (3) **chunked@ctx X ≈ truncated@ctx 2X**
(chunked-1024≈trunc-2048, chunked-2048>trunc-4096) → ~2× effective context at a fraction of per-step cost.
GRID COMPLETE (21/21, 2026-07-01). LARGE chunked-no: c512=0.4917 c1024=0.4797 c2048=0.5774 c4096=0.6410
c8192=**0.7100** — best result; **beats large-truncated-8192 (0.6716) by +0.038**. LARGE overlap: c512=0.4871
c1024=0.5081 c2048=0.5055 c4096=0.5945 c8192=0.6370 — **all on full 7000-doc eval** (c4096-ov/c8192-ov
re-scored via the resumable `score_chunked_eval.py` on preemptible v6e after the in-job eval preempt-looped;
GRID NOW FULLY CONSISTENT at 7000 docs. Resumable scorer = the 10M eval tool, loads HF + checkpoints per block.)
Large ≥ base at matched ctx. **FINAL overlap verdict: underperforms non-overlap everywhere** — large-c8192-ov
0.627 ≪ no 0.710 contradicts the lone base-c8192-ov exception (0.672>0.657), which was noise. Use NON-OVERLAP.

### FULL per-aggregator F1 (regenerate: `uv run --no-sync python -m experiments.baseline_collection.dump_chunk_results`)
All aggregations we sweep at eval (best is just the max of these). Snapshot 2026-06-30 ~15:30 UTC (19 of 21):
```
run                  |     mean      max      min top2mean top3mean | best (agg @ thr)
large-c8192-no       |   0.7074   0.7004   0.6159   0.6980   0.7100 | 0.7100 (top3mean @ 0.26)
base-c8192-ov        |   0.6637   0.6723   0.5482   0.6698   0.6639 | 0.6723 (max @ 0.48)
base-c8192-no        |   0.6560   0.6571   0.5616   0.6543   0.6562 | 0.6571 (max @ 0.38)
base-c4096-no        |   0.5964   0.6455   0.4603   0.6334   0.6216 | 0.6455 (max @ 0.44)
large-c4096-no       |   0.6092   0.6410   0.5436   0.6386   0.6225 | 0.6410 (max @ 0.48)
base-c4096-ov        |   0.5735   0.5903   0.4704   0.5818   0.5835 | 0.5903 (max @ 0.48)
large-c2048-no       |   0.5317   0.5774   0.4463   0.5707   0.5678 | 0.5774 (max @ 0.48)
base-c2048-no/canary |   0.5146   0.5600   0.3656   0.5525   0.5602 | 0.5602 (top3mean @ 0.32)
base-c2048-ov        |   0.4996   0.5297   0.3500   0.5324   0.5270 | 0.5324 (top2mean @ 0.62)
large-c1024-ov       |   0.4461   0.5081   0.3200   0.5008   0.5009 | 0.5081 (max @ 0.74)
large-c2048-ov       |   0.4774   0.5055   0.4205   0.4946   0.4935 | 0.5055 (max @ 0.72)
base-c1024-no        |   0.4381   0.4932   0.3150   0.4833   0.4808 | 0.4932 (max @ 0.64)
large-c512-no        |   0.4299   0.4837   0.2932   0.4917   0.4803 | 0.4917 (top2mean @ 0.62)
large-c512-ov        |   0.4293   0.4681   0.2853   0.4750   0.4871 | 0.4871 (top3mean @ 0.64)
large-c1024-no       |   0.4344   0.4763   0.3403   0.4797   0.4736 | 0.4797 (top2mean @ 0.56)
base-c512-no         |   0.4053   0.4707   0.2653   0.4744   0.4777 | 0.4777 (top3mean @ 0.50)
base-c512-ov         |   0.4207   0.4626   0.2636   0.4746   0.4753 | 0.4753 (top3mean @ 0.58)
base-c1024-ov        |   0.4247   0.4601   0.2920   0.4682   0.4673 | 0.4682 (top2mean @ 0.60)
```
(19 of 21; large-c4096-ov + large-c8192-ov pending. Aggregator pattern: min worst everywhere; max/top-k win at
small/mid ctx; at large ctx (c8192) mean catches up (0.707≈top3 0.710) — few, uniformly-informative chunks.)

**Findings:**
- **Chunked F1 rises with ctx: 512→0.478, 1024→0.493, 2048→0.560.** chunked-c2048 (0.560) > truncated-c2048
  (0.441, +0.12) AND > truncated-c4096 (0.546); tracking toward truncated-c8192 (0.645). c4096/c8192 pending.
- **Overlap mostly UNDERPERFORMS (non-overlap = safe default):** base no-vs-ov — c512 tie (0.478/0.475),
  c1024 0.493>0.468, c2048 0.560>0.532, c4096 0.6455>0.590 (all non-overlap wins); only c8192 flips (ov 0.6723 >
  no 0.6571). The lone c8192-ov edge may be noise or a real very-large-ctx effect — inconclusive. **Use non-overlap.**
  base-c8192-ov 0.6723 is still the best base point and well above trunc-8192 0.645.
- **large > base at matched ctx:** large-c512 0.492 > base-c512 0.478.
- Per-aggregator (base): c1024-no mean=0.438 max=0.493 min=0.315 top2=0.483 top3=0.481;
- **Aggregation: max / top-k-mean win; min worst; mean mediocre.** Signal is in a doc's most-useful
  chunks (classic MIL). Gap best-vs-mean ≈ 0.05–0.08 → aggregator choice matters. Deploy with max or top-3-mean.
- **Chunking beats truncation at matched small ctx (suggestive):** chunked-c1024 **0.493** vs
  truncated-c1024 **0.355** (+0.14); chunked-c1024 also edges truncated-c2048 (0.441). Caveat: truncated
  baselines were trained on **1M_w4**, chunked on **random_par** (longer docs, median 11,848 tok), so not a
  perfectly matched comparison — for a clean head-to-head, run truncated-c1024/c2048 on random_par.
- c512 overlap ≈ non-overlap (0.475 vs 0.478) — 50% overlap gives no clear lift at 512 so far.
- Aggregator note: at eval, NO max-chunks cap (full doc); training caps at 16 chunks/doc.

## Next steps after the pilot
- If chunked small-ctx beats its truncated baseline (esp. ctx 512/1024 ≈ truncated-8192) → scale to
  10M. That needs a **chunk-aware cache rebuild** (drop the 8192-token cap in
  `ClassificationLineProcessor`; store full token seq or pre-chunked), since 10M won't fit in-memory.
- Pick the deployment aggregator from the eval winner; decide overlap on/off from the tiling sweep.
