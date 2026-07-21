# Random-1000 WARC crossover grid

Goal: push the HQ−DCLM crossover hunt to N=1000 random WARCs. At N=500 the flip is
marginal (only Core-v2 d768 top budget; ao3 val loss flips at N=300). 1k ~doubles
tokens again and should make the flip decisive across cells. User chose: **skip
biased-1k, go straight to random-1k, all three methods** (dclm, nemotron_full,
high_quality).

## Status

- [x] Draw `experiments/distill/subsets/baseline_warcs_1000_random.txt` (seed 0).
      Nesting verified: 100 ⊂ 300 ⊂ 500 ⊂ 1000 (all OK).
- [x] Subset + tokenize via `subset_random100_10k.py` with `WARC_N=1000`. Token counts:
      dclm 706,817,668 (`dclm_random_1000warcs-5a69c1`, us-central2);
      nemotron_full 1,028,007,768 (`nemotron_full_random_1000warcs-83bd90`, us-central2);
      high_quality 2,089,503,582 (`high_quality_random_1000warcs-2470d1`, us-central1).
- [x] Registered `{dclm,nemotron_full,high_quality}_random_1000` in curation_plan
      (_D_OBS_DEFAULTS + METHODS). pin_region: dclm→us-east5, nemotron→us-east1,
      high_quality→us-central1 (co-located w/ its 2.1B cache = zero mirror egress).
- [x] Launched training grid: 90 runs (30/method) via coordinator Iris job
      `/michaelryan/warc-scaling-random-1000` (us-central1 CPU parent, --child-priority batch).
      Children pending on TPU capacity (batch). Monitor `bb0026dxu` on DONE markers.
- [ ] Launch Core v2 evals (fires at ~70/90 DONE) + extend ladder plotters to N=1000.

## Gotchas hit
- Coordinator MUST run inside an Iris job (`IRIS_CONTROLLER_ADDRESS` required) — cannot
  run launch_warc_scaling_sweep.py locally. Submit as CPU parent w/ --enable-extra-resources
  (4GB triggers it). Children nest under `/michaelryan/warc-scaling-random-1000/curation-curation-...`.

## Key facts

- Iris user = `michaelryan`. Region→bucket: us-central2→marin-us-central2, etc.
- Tokenized cache path: `gs://marin-{region}/tokenized/{method}_random_1000warcs-{hash}/`,
  token count in `train/.stats.json:total_tokens`.
- Region pinning for subset reads (in-region, no egress): us-central2 = dclm/nemotron_full
  (+metadata keyset), us-central1 = high_quality. Sources = existing 10,364-WARC extractions
  (DCLM_10K, NEMOTRON_FULL_10K in c2; HQ_10K_JOINED in c1).
- Token estimate (≈2× N=500): dclm ~712M, nemotron ~1.04B, hq ~2.1B. Crossover
  data-rich regime for ~1B-param model ≈ 1e19–3e19 FLOPs.
- Biased-1k ALREADY trained (dclm 33 / hq 29 / nemo 32 DONE across 5 regions) but
  0 Core evals — NOT used (user chose random path); available as fallback proxy.

## Training-region strategy (from 300/500 experience)

us-east5 primary (v5p), us-central1/us-east1 great extras, europe-west4 = v5litepod-4
(needs `--force-primary-tpu v5litepod-4`). Spread methods across regions to dodge
saturation. Pin `--allowed-regions` so temp-checkpoint resume stays bucket-local.

## N=2000 rung (started 2026-07-19)

Context: 1k did NOT reproduce a clean crossover (mean HQ-DCLM gap shrank +0.024→+0.010
across the ladder but HQ still ahead 17/22 at N=1000; only d256-top flipped). 3k is a
KNOWN crossover point (user). So 2k is the decisive middle rung.

- [x] Killed 2 >5h N=1000 stragglers (hq + dclm d256-3e19-B512, ~27h/13h).
- [x] Nesting-preserving 2000 sample: `baseline_warcs_2000_random.txt` = nested-1000 ∪
      1000-from-remainder (seed 0). Plain sample(pool,2000) BREAKS nesting — k=2000 exceeds
      CPython selection-sampling setsize (16405) for the 10,364 pool, flipping to pool-tracking.
      Verified 100⊂300⊂500⊂1000⊂2000.
- [x] Aligned `_HIDDEN_SIZES_PER_N[2000]` (256,512,768,1536) to the ladder (was 512,1280,2048).
      Budgets already 1e17..3e20 (data-rich). Biased-2000 already trained on old grid, untouched.
- [x] Subset+tokenize done. Tokens: dclm 1,424,810,618 (`-7ba7b6`); nemotron 2,035,820,453 (`-7e5953`);
      hq 4,181,779,555 (`-06a890`). ~2× the 1k rung, as expected.
- [x] PRE-COPIED caches to train regions BEFORE launch (the 1k lesson): dclm→us-east5, nemo→us-east1,
      both verified; hq stays us-central1. Registered all 3 `*_random_2000` in curation_plan.
- [x] Launched training: 93 runs (31/method) via coordinator Iris job `/michaelryan/warc-scaling-random-2000`
      (--child-priority batch). Monitor `b2fh5ltsx` on DONE markers; eval-trigger at 72/93.
- [ ] Evals via launch_10k_manifest as Iris job (fires at 72 DONE) + extend ladders to N=2000.

Gotcha: iris client version gate bumped to min 2026-07-05; fixed by BUILD_DATE="2026-07-19"
in lib/iris/src/iris/_build_info.py.

## d2432 model-size probe (started 2026-07-19)

Theory (user): the crossover may be MODEL-SIZE driven, not WARC-count driven. The 10k
"HQ worse" result was at d2432 (2.9B), but the crossover ladder capped at d1536 (998M).
So train d2432 at ALL WARC scales (100/300/500/1000/2000) to see if the flip appears at
low N once the model is big enough.

- Added 2432 to `_HIDDEN_SIZES_PER_N` for every N (auto-config: L24, 19 heads, ~2.9B).
- Verified ALL random_N caches present in their pin regions (no cross-region reads):
  dclm 100/300/1000/2000→us-east5, 500→us-east1; hq 100/500→us-east5, 300/1000/2000→us-central1;
  nemo 100/300→us-east5, 1000/2000→us-east1. EXCLUDED nemo_500 (pins eu-west4=v5e, can't host 2.9B).
- Budgets 1e18/3e18/1e19/3e19 (capped at 3e19: ~5-6h/cell on v5p-8; 1e20 would be ~18h).
- Launched: `warc-scaling-d2432-dclmhq` (38 runs, PRIORITY) + `warc-scaling-d2432-nemo` (15, no 500).
  = 53 runs total. Monitor `b27tojawt` on d2432 DONE markers; eval-trigger at 24.
- [ ] Evals on d2432 cells (via launch_10k_manifest) + add d2432 column to the ladder grid.

### d2432 OOM fix (2026-07-20)
First d2432 launch: ALL 53 cells failed exit-137 (host-memory OOM) — `_memory_gb_for_hidden`
floors host RAM at 48 GiB regardless of size, which is too small for a 2.9B model's checkpoint
serialization (holds ~35GB model+AdamH state on host) + loader + eval-harness. Fix: relaunch with
`--force-memory-gb 256`. Coordinators: `warc-scaling-d2432-dclmhq-v2` + `warc-scaling-d2432-nemo-v2`.
Cache locality note: dclm+nemo tokenized in us-central2, hq in us-central1 — ALL N-caches of a method
live in one source region, so floating d2432 jobs still find a local cache.

## OLMES/OLMo/Core eval offline-cache regression (2026-07-20)
Symptom: olmo_bpb (132 failed) + olmes_base (220 failed), 0 results, exit 1, HF offline error
("Check your internet connection... offline mode"). NOT qa_rc, NOT the checkpoints, NOT OOM.
Root cause: the runners call `HFCheckpointConverter.from_hf(checkpoint)` (Levanter), which SCANS
the ENTIRE LmConfig registry — `v().hf_checkpoint_converter()` for every config — and each
converter's __init__ EAGERLY loads that config's default reference tokenizer via `_infer_tokenizer`.
Since the proven 2026-07-04 run, Levanter added Apertus/Gemma/OLMo-3/etc configs whose default
tokenizers (swiss-ai/Apertus-8B-2509, google/gemma-*, etc.) were NOT in `core_tasks_hub_cache`
(which only had NousResearch/Llama-2-7b-hf) → offline lookup fails during the scan, before the
runner can override. The scan runs in registry order until the arch matches (qwen3 = position 13),
so ALL 12 refs up to qwen3 must be cached.
Fix: downloaded all 12 reference tokenizers (config+tokenizer only, no weights) and rsynced them
(dereferenced symlinks) into `eval_datasets/core_tasks_hub_cache/hub/` in all 4 regions
(us-east5/us-east1/us-central1/eu-west4). Verified from_hf passes offline → Qwen3Config, vocab 128000.
Then relaunched olmo_bpb (--tasks all) + olmes_base. This also unblocks d2432 Core v2 + OLMo evals.
Refs cached: Llama-2-7b(had it), Apertus-8B-2509, Qwen3-0.6B, OLMo-2-1124-7B, Olmo-3-1025-7B,
ModernBERT-base, Mixtral-8x7B-v0.1, gemma-2b, gemma-2-2b, gemma-3-1b-pt, gpt2, Mistral-7B-v0.1.
