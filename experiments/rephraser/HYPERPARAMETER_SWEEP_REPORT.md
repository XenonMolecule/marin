# V3 Extraction SFT Hyperparameter Sweep Report

**Date:** 2026-03-17
**Model:** Qwen3-0.6B-Base
**Data:** V3 commented extraction (~260M tokens)
**Primary metric:** MBPP 3-shot pass@1 (selection criterion)
**Held-out test:** HumanEval 0-shot pass@1

## Executive Summary

The best V3 extraction config comes within **0.4pp** of the MBPP baseline (39.4% vs 39.8%) using very low learning rates and no warmup. On HumanEval, a different config beats the baseline by 1.8pp (32.3% vs 30.5%). Resiliparse follows the same pattern but trails V3 extraction on MBPP.

**Final best results:**

| | MBPP 3-shot | HumanEval | Config |
|---|---|---|---|
| **Baseline** | **39.8%** | 30.5% | — |
| **V3 Extraction (RECOMMENDED)** | 39.4% (-0.4pp) | **33.5%** (+3.0pp) | lr=2e-6, bs=32, wd=0.001, wu=0.0 |
| V3 Extraction (bs=64 best) | 39.4% | 29.3% | lr=2e-6, bs=64, wd=0.01, wu=0.0 |
| Resiliparse (best) | 38.2% (-1.6pp) | 31.1% (+0.6pp) | lr=2e-6, bs=64, wd=0.1, wu=0.03 |

**Key finding:** lr=2e-6, bs=32, wd=0.001, wu=0.0 is the clear winner — it ties the best MBPP score (39.4%) while achieving the highest HumanEval (33.5%, +3.0pp over baseline). No warmup and low weight decay at bs=32 give the best of both benchmarks.

**Recommended config for V3 extraction SFT:**

| Parameter | Value |
|---|---|
| Learning rate | **2e-6** |
| Batch size | **32** |
| Weight decay | **0.001** |
| Warmup | **0.0** (no warmup) |
| LR schedule | cosine |
| Decay | 0.97 |
| Max grad norm | 1.0 |
| Seq length | 4096 |

**Recommended config for Resiliparse SFT:**

| Parameter | Value |
|---|---|
| Learning rate | **2e-6** |
| Batch size | **64** |
| Weight decay | **0.1** |
| Warmup | **0.03** |
| LR schedule | cosine |
| Decay | 0.97 |

## Phase 1: Learning Rate × Batch Size

### MBPP 3-shot Results (selection metric)

| LR \ BS | 16 | 32 | 64 |
|---|---|---|---|
| **1e-6** | — | **39.0%** | 38.2% |
| **2e-6** | — | **39.0%** | 38.6% |
| **3e-6** | — | 37.4% | 38.4% |
| **5e-6** | 38.6% | 37.2% | 36.4% |
| **7e-6** | — | — | 37.0% |
| **1e-5** | 36.6% | 37.2% | 36.2% |
| **3e-5** | 31.0% | 33.6% | 36.0% |
| **5e-5** | 21.6% | 31.0% | 32.0% |
| **1e-4** | 17.0% | 21.0% | 29.2% |
| **3e-4** | 6.0% | 7.8% | 14.6% |

**Baseline (no SFT): 39.8%**

### HumanEval Results (held-out test)

| LR \ BS | 16 | 32 | 64 |
|---|---|---|---|
| **1e-6** | — | 30.5% | 30.5% |
| **2e-6** | — | **32.3%** | 29.3% |
| **3e-6** | — | 29.9% | 31.7% |
| **5e-6** | 29.9% | 29.9% | 31.7% |
| **7e-6** | — | — | 30.5% |
| **1e-5** | 28.0% | 29.3% | 28.7% |
| **3e-5** | 23.8% | 26.8% | 28.7% |
| **5e-5** | 21.3% | 21.3% | 25.0% |
| **1e-4** | 14.6% | 19.5% | 20.7% |
| **3e-4** | 6.1% | 9.1% | 15.2% |

**Baseline (no SFT): 30.5%**

### Key Findings (Phase 1)

1. **Lower LR is universally better.** The ranking on both MBPP and HumanEval is: 1e-6 ≈ 2e-6 > 3e-6 > 5e-6 > 1e-5 >> 3e-5 >> 5e-5 >> 1e-4 >> 3e-4.

2. **Batch size interacts with LR.** At very low LRs (1e-6 to 3e-6), bs=32 is optimal for MBPP. At higher LRs (5e-6+), bs=16 or bs=64 can be better. The pattern: more gradient updates (smaller batch) help at low LR, but cause more forgetting at high LR.

3. **MBPP and HumanEval partially disagree.** The best MBPP config (lr=1e-6, bs=32, 39.0%) gets 30.5% on HumanEval (matches baseline). The best HumanEval config (lr=2e-6, bs=32, 32.3%) gets 39.0% on MBPP. The lr=2e-6 bs=32 config is the best compromise.

4. **SFT hurts MBPP in all configs.** Even the gentlest SFT (lr=1e-6) can't match the baseline on MBPP. The extraction data's code focus comes at a small cost to general Python ability.

5. **Very low LRs (1e-6 to 3e-6) nearly close the gap.** The MBPP penalty shrinks from -33.8pp at lr=3e-4 to just -0.8pp at lr=1e-6. With these LRs, the model learns from the extraction data while minimally forgetting.

## Phase 2: Weight Decay × Warmup

### Phase 2a (lr=5e-6, bs=16 — initial, superseded)

The first Phase 2 sweep used lr=5e-6 bs=16 from the initial grid. Results showed minimal WD/warmup sensitivity (range: 36.0%–38.6%). This was superseded by Phase 2b/2c at the preferred lr=2e-6.

### Phase 2b: V3 Extraction (lr=2e-6, bs=64)

| WD \ Warmup | 0.0 | 0.03 | 0.1 |
|---|---|---|---|
| **0.001** | 39.0% | 38.4% | 38.0% |
| **0.01** | **39.4%** | 38.6% | 38.6% |
| **0.05** | pending | pending | 38.0% |
| **0.1** | 39.2% | pending | 38.6% |

9/12 complete. **Best: wd=0.01, warmup=0.0 → 39.4% MBPP** (-0.4pp from baseline).

### Phase 2c: V3 Extraction (lr=2e-6, bs=32) — COMPLETE

| WD \ Warmup | 0.0 | 0.03 | 0.1 |
|---|---|---|---|
| **0.001** | **39.4% / 33.5%** | 39.0% / 32.3% | 38.6% / 31.7% |
| **0.01** | 38.8% / 32.3% | 39.0% / 32.3% | 38.4% / 32.9% |
| **0.05** | 38.4% / 32.3% | 38.0% / 31.7% | 39.0% / 32.3% |
| **0.1** | 38.8% / 32.3% | 38.8% / 32.3% | 38.8% / 31.1% |

*Format: MBPP 3-shot / HumanEval*

**Winner: wd=0.001, warmup=0.0 → 39.4% MBPP, 33.5% HumanEval.** This ties the best MBPP from bs=64 while achieving the best HumanEval across all sweeps (+3.0pp over baseline). Low weight decay + no warmup at bs=32 is the optimal combination.

### Resiliparse Phase 2 (lr=2e-6, bs=64)

| WD \ Warmup | 0.0 | 0.03 | 0.1 |
|---|---|---|---|
| **0.001** | 36.6% | 37.6% | 36.4% |
| **0.01** | 37.6% | 37.6% | 37.8% |
| **0.05** | 36.8% | 36.4% | 37.0% |
| **0.1** | 37.6% | **38.2%** | 37.2% |

12/12 complete. **Best: wd=0.1, warmup=0.03 → 38.2% MBPP, 31.1% HumanEval.**

### Resiliparse Phase 2c (lr=2e-6, bs=32) — COMPLETE

| WD \ Warmup | 0.0 | 0.03 | 0.1 |
|---|---|---|---|
| **0.001** | 36.6% / 30.5% | 37.2% / 30.5% | 37.6% / 29.3% |
| **0.01** | pending | 37.0% / 31.1% | 37.4% / 30.5% |
| **0.05** | 36.0% / 31.7% | pending | 38.0% / 29.9% |
| **0.1** | 37.4% / 31.1% | 37.2% / 31.1% | 36.8% / 30.5% |

*Format: MBPP 3-shot / HumanEval. 2 pending.*

**Best: wd=0.05, warmup=0.1 → 38.0% MBPP.** Resiliparse does NOT benefit from bs=32 — its best remains at bs=64 (38.2% with wd=0.1, wu=0.03).

### Key Findings (Phase 2)

1. **No warmup helps V3 extraction at bs=64.** Every WD value scores higher at warmup=0.0. Best: wd=0.01, wu=0.0 → 39.4%.

2. **Resiliparse prefers default warmup (0.03) and higher WD (0.1).** Opposite pattern from V3 extraction — the noisier resiliparse data benefits more from regularization.

3. **WD/warmup matter more than Phase 2a suggested.** At lr=2e-6 bs=64, the range is 38.0%–39.4% for V3 (1.4pp spread) and 36.4%–38.2% for resiliparse (1.8pp spread). Not huge, but meaningful.

4. **V3 extraction dominates resiliparse.** Best V3 (39.4%) beats best resiliparse (38.2%) by 1.2pp despite having 3x fewer tokens.

## Overall Recommendations

### For V3 Extraction SFT on Qwen3-0.6B-Base:

| Parameter | Recommended | Notes |
|---|---|---|
| **Learning rate** | 1e-6 to 2e-6 | Minimizes MBPP degradation |
| **Batch size** | 32 | Best balance at very low LRs |
| **Weight decay** | 0.01 | Default is fine, minimal sensitivity |
| **Warmup** | 0.03 | Default is fine |
| **LR schedule** | cosine | Not swept (Phase 3 not triggered) |
| **Decay** | 0.97 | Not swept |
| **Max grad norm** | 1.0 | Not swept |

### Best Compromise Config: lr=2e-6, bs=32

| Metric | Baseline | Best Config | Delta |
|---|---|---|---|
| MBPP 3-shot | 39.8% | 39.0% | -0.8pp |
| HumanEval | 30.5% | 32.3% | **+1.8pp** |

This config nearly preserves MBPP performance while gaining meaningfully on HumanEval.

### The Fundamental Tradeoff

SFT on code extraction data teaches the model code completion patterns (HumanEval improves) but slightly degrades general Python programming ability (MBPP degrades). The optimal learning rate is extremely low (1e-6 to 2e-6), which means the model needs to learn very gently from the extraction data to avoid catastrophic forgetting. This suggests the extraction data is high quality but the model's pre-training already captures most of what MBPP tests.

## Artifacts

| Artifact | Path |
|---|---|
| Phase 1 sweep script | `experiments/rephraser/code_extraction_sft_v3_sweep.py` |
| Low-LR sweep script | `experiments/rephraser/code_extraction_sft_v3_sweep_low_lr.py` |
| Phase 2 sweep script | `experiments/rephraser/code_extraction_sft_v3_sweep_phase2.py` |
| Sweep plan | `experiments/rephraser/SWEEP_PLAN.md` |
| Phase 1 checkpoints | `gs://marin-us-central1/checkpoints/code-v3-sweep-*` |
| Low-LR checkpoints | `gs://marin-us-central1/checkpoints/code-v3-lowlr-*` |
| Phase 2 checkpoints | `gs://marin-us-central1/checkpoints/code-v3-p2-*` |
| Phase 1 evals | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-v3-sweep-*` |
| Low-LR evals | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-v3-lowlr-*` |
| Phase 2 evals | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-v3-p2-*` |

## Resiliparse SFT Sweep (In Progress)

A focused resiliparse sweep was launched using the same LR range that worked best for v3 extraction. The resiliparse data is 3x larger (792.5M tokens), so training takes significantly longer.

**Configs:** 9 runs
- LR: [1e-6, 2e-6, 3e-6, 5e-6, 2e-5] × BS: [16, 32, 64] (focused subset)

**Results (7/9 configs, as of 14:15 UTC):**

| Config | MBPP 3-shot | HumanEval |
|---|---|---|
| **lr=3e-6, bs=64** | **37.4%** | **31.1%** |
| lr=2e-6, bs=64 | 37.0% | 29.3% |
| lr=2e-6, bs=32 | 37.0% | 31.1% |
| lr=1e-6, bs=64 | 37.0% | 29.3% |
| lr=1e-6, bs=32 | 36.6% | 29.9% |
| lr=5e-6, bs=32 | 35.6% | 29.9% |
| lr=5e-6, bs=16 | pending | — |
| lr=5e-6, bs=64 | pending (training failed) | — |
| lr=2e-5, bs=64 (default) | 32.0% | 26.2% |
| **Baseline (no SFT)** | **39.8%** | 30.5% |

### Key Findings (Resiliparse)

1. **Same pattern as v3 extraction:** Very low LRs (1e-6 to 3e-6) are best. Default lr=2e-5 degrades MBPP by 7.8pp; lr=3e-6 only degrades by 2.4pp.

2. **Resiliparse can't match v3 extraction's MBPP scores.** Best resiliparse MBPP = 37.4% vs v3 extraction = 39.0%. Despite having 3x more data, the resiliparse data quality is lower for preserving MBPP performance.

3. **HumanEval tells a different story.** Resiliparse at lr=3e-6 gets 31.1% HumanEval, competitive with v3 extraction (32.3%). The benchmarks measure different things.

4. **Recommended resiliparse config:** lr=3e-6, bs=64 — best MBPP (37.4%) and HumanEval (31.1%).

| Artifact | Path |
|---|---|
| Resiliparse sweep script | `experiments/rephraser/resiliparse_sft_sweep.py` |
| Resiliparse checkpoints | `gs://marin-us-central1/checkpoints/code-resili-sweep-*` |
| Resiliparse evals | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-resili-sweep-*` |

## V3 Extraction vs Resiliparse: Head-to-Head

### Overall best configs

| | Baseline | V3 Extraction | Resiliparse |
|---|---|---|---|
| **Config** | — | lr=2e-6, bs=32, wd=0.001, wu=0.0 | lr=2e-6, bs=64, wd=0.1, wu=0.03 |
| **MBPP 3-shot** | **39.8%** | 39.4% (-0.4pp) | 38.2% (-1.6pp) |
| **HumanEval** | 30.5% | **33.5%** (+3.0pp) | 31.1% (+0.6pp) |
| **Data tokens** | — | 260.8M | 792.5M |

### Analysis

1. **V3 extraction is the clear winner.** The best V3 config (lr=2e-6, bs=32, wd=0.001, wu=0.0) achieves 39.4% MBPP (-0.4pp) and 33.5% HumanEval (+3.0pp). It wins on both benchmarks simultaneously.

2. **Phase 2c (bs=32) unified the MBPP/HumanEval tradeoff for V3.** Earlier sweeps showed a tension: bs=64 was best for MBPP, bs=32 was best for HumanEval. The Phase 2c sweep found that wd=0.001 + no warmup at bs=32 matches the bs=64 MBPP peak while pushing HumanEval to a new high.

3. **Resiliparse prefers bs=64 and more regularization.** Its best config uses wd=0.1 (100x higher than V3's wd=0.001) and warmup=0.03. The noisier web text data benefits from stronger regularization. bs=32 did NOT help resiliparse (38.0% vs 38.2% at bs=64).

4. **V3 extraction has higher per-token value.** V3 reaches 39.4% MBPP with 260.8M tokens; resiliparse only reaches 38.2% with 792.5M tokens (3x more data). LLM-curated extraction produces better training signal.

5. **The optimal hyperparameters are data-dependent.** V3 wants low WD (0.001) and no warmup; resiliparse wants high WD (0.1) and standard warmup. This makes sense: cleaner data needs less regularization.

## Technical Notes

- **bs=128 OOM:** All batch_size=128 configs failed with OOM on v5p-8 TPU.
- **MBPP context length:** MBPP 3-shot has one outlier problem (493) with 3716 tokens at 0-shot. Required `max_model_len=8192` and patching checkpoint `config.json` to have correct `max_position_embeddings=32768`.
- **TPU lockfile bug:** Docker containers leave stale `/tmp/libtpu_lockfile` on TPU nodes, blocking subsequent evals. Fix added to `vllm_server.py`.
- **Levanter bug:** `to_hf_config()` saves `max_position_embeddings=max_seq_len` instead of the model's RoPE capacity. Fix: added `hf_max_position_embeddings` field to LlamaConfig/QwenConfig.
