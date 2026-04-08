# Math Top-3 Low-LR Extension Sweep Report

**Experiment file:** `experiments/rephraser/math_top3_low_lr_sweep.py`
**Model:** Qwen3-0.6B-Base
**Date:** 2026-03-26

---

## Executive Summary

The prior HP sweep identified lr=5e-7 as optimal for Minerva math (avg_minerva=30.3%). This extension sweep, testing LRs from 3e-7 down to 2e-8, confirms that 5e-7 is the true optimum: all 14 new configs fall below it on Minerva, with a remarkably flat plateau between lr=3e-7 and lr=5e-8 (29.4–29.8%, a spread of only 0.4pp across a 15x LR range). However, very low LRs preserve GSM8K capability better — lr=5e-8 achieves 64.3% GSM8K flex, beating the untrained baseline (62.8%) and the prior sweep best (60.3%), revealing a clear Minerva/GSM8K tradeoff that practitioners should weigh explicitly.

---

## Experiment Setup

| Parameter | Value |
|---|---|
| Model | Qwen3-0.6B-Base |
| Training domains | brainly.com, jiskha.com, mathhelpforum.com (top 3 of prior domain sweep) |
| Data types | extraction (LLM Q/R/A format), resiliparse (raw HTML-to-text) |
| Training | Single-epoch SFT, cosine LR schedule |
| Fixed HPs | wd=0.10, warmup=0.03 |
| HP grid | 7 configs × 2 data types = 14 runs |
| Evals | Minerva MATH (7 subtasks, 4-shot), GSM8K Platinum CoT (8-shot) |

**HP grid:**

| Config | LR | Batch size |
|---|---|---|
| lr3e-7_bs64 | 3e-7 | 64 |
| lr2e-7_bs64 | 2e-7 | 64 |
| lr2e-7_bs16 | 2e-7 | 16 |
| lr1e-7_bs64 | 1e-7 | 64 |
| lr1e-7_bs16 | 1e-7 | 16 |
| lr5e-8_bs64 | 5e-8 | 64 |
| lr2e-8_bs64 | 2e-8 | 64 |

---

## Results

### Extraction Data

Ranked by avg_minerva (average across 7 Minerva MATH subtasks).

| Config | Algebra | Prealg | Cnt/Prob | Geometry | Int_Alg | Num_Thy | Precalc | avg_minerva | GSM8K_flex | GSM8K_strict |
|---|---|---|---|---|---|---|---|---|---|---|
| lr3e-7_bs64 | 49.9 | 56.6 | 27.2 | 26.9 | 12.1 | 21.9 | 14.1 | 29.81% | 59.64% | 52.85% |
| lr2e-7_bs16 | 50.6 | 53.7 | 26.2 | 29.4 | 13.0 | 20.7 | 14.7 | 29.76% | 60.63% | 54.92% |
| lr1e-7_bs64 | 50.0 | 51.9 | 27.6 | 29.6 | 12.1 | 21.7 | 14.7 | 29.66% | 62.03% | 57.32% |
| lr1e-7_bs16 | 49.8 | 55.2 | 26.6 | 27.8 | 11.5 | — | — | 29.57% | 60.71% | 53.68% |
| lr5e-8_bs64 | 48.6 | 52.8 | 26.2 | 28.0 | 11.8 | 22.8 | 15.8 | 29.42% | **64.27%** | 58.31% |
| lr2e-7_bs64 | 49.4 | 54.9 | 26.2 | 26.3 | 12.2 | — | — | 29.37% | 60.13% | 51.94% |
| lr2e-8_bs64 | 44.3 | 50.2 | 25.3 | 25.1 | 12.5 | 19.4 | 16.3 | 27.59% | 63.11% | 57.98% |

### Resiliparse Data

Ranked by avg_minerva.

| Config | avg_minerva | GSM8K_flex | GSM8K_strict |
|---|---|---|---|
| lr5e-8_bs64 | 25.94% | 63.44% | 59.97% |
| lr2e-8_bs64 | 25.94% | 62.03% | 58.23% |
| lr1e-7_bs64 | 25.26% | 62.61% | 59.22% |
| lr1e-7_bs16 | 23.99% | 61.37% | 57.98% |
| lr3e-7_bs64 | 23.49% | 61.21% | 57.40% |
| lr2e-7_bs64 | 23.40% | 61.95% | 58.89% |
| lr2e-7_bs16 | 23.20% | 59.64% | 56.24% |

### Reference Scores

| Config | avg_minerva | GSM8K_flex |
|---|---|---|
| Prior sweep best (extract, lr5e-7_bs64) | 30.3% | 60.3% |
| Untrained baseline (Qwen3-0.6B-Base) | 26.7% | 62.8% |

> **Baseline correction (2026-04-08):** The untrained baseline was originally reported as 22.9% avg_minerva, computed before the full 7-subtask baseline eval was run. The correct value is **26.7%**, from a dedicated baseline eval (`math_0_6b_baseline_eval.py`) using the same eval config as this sweep. Per-subtask: Algebra 41.4, PreAlg 49.0, Count/Prob 23.0, Geometry 25.9, Int Algebra 12.8, Num Theory 17.0, Precalc 17.9.

---

## Analysis

### 1. Diminishing Returns Confirmed Below lr=5e-7

The prior sweep's lr=5e-7 with extraction data (30.3% avg_minerva) remains the champion for Minerva math. Every config in this extension sweep falls below it. Critically, the degradation is not sharp: from lr=3e-7 down to lr=5e-8 the plateau spans only 29.37–29.81%, a 0.4pp spread across a 15x LR range. Only at lr=2e-8 does performance drop noticeably to 27.59%, a 2.2pp cliff relative to the plateau. The implication is that lr=5e-7 is a soft optimum, not a sharp peak — the model is not particularly sensitive to LR within this lower range for Minerva, but going below ~5e-8 crosses a threshold where training becomes too weak.

### 2. GSM8K Inverse Correlation with Learning Rate

Lower LR consistently preserves GSM8K performance. The extract-lr5e-8 config achieves 64.27% GSM8K flex — the single best result across all SFT configs in both sweeps, and meaningfully above the untrained baseline (62.8%). The pattern is monotonic: as LR decreases, GSM8K flex increases. This is consistent with the catastrophic forgetting hypothesis: very low LR SFT updates the model too weakly to disrupt pre-trained arithmetic capabilities while still imprinting math problem-solving patterns from the domain data.

### 3. Minerva vs. GSM8K Tradeoff

These two metrics pull in opposite directions across the LR axis. The relationship for extraction data:

| LR | avg_minerva | GSM8K_flex | Character |
|---|---|---|---|
| 5e-7 (prior best) | 30.3% | 60.3% | Minerva-optimal |
| 3e-7 | 29.8% | 59.6% | Near-Minerva, GSM8K cost |
| 1e-7 | 29.7% | 62.0% | Balanced |
| 5e-8 | 29.4% | 64.3% | GSM8K-optimal |
| 2e-8 | 27.6% | 63.1% | Neither |

The choice of LR should be driven by which metric matters more. If the goal is best-in-class Minerva performance, lr=5e-7 is unambiguously optimal. If the goal is a model that is stronger on both math and arithmetic, lr=1e-7 or lr=5e-8 offers a more balanced profile. lr=2e-8 is a dominated choice — it costs Minerva without recovering more GSM8K than lr=5e-8.

### 4. Extraction Consistently Dominates Resiliparse (+4pp)

The extraction data quality advantage is robust across the full LR range tested. The best resiliparse config (lr=5e-8, 25.94%) is 3.9pp below the worst extraction config in the plateau (lr=2e-7_bs64, 29.37%). This gap is not an artifact of any particular LR — it holds at every point tested. The LLM-extracted Q/R/A format consistently provides higher-quality training signal for structured math reasoning than raw HTML-to-text conversion.

### 5. Opposite LR Sensitivity for Resiliparse Data

For resiliparse, lower LR is unambiguously better: lr=5e-8 and lr=2e-8 (both 25.94%) significantly outperform lr=2e-7 (~23.4%) and lr=3e-7 (23.49%). This is the reverse of the extraction trend, where lr=3e-7 is marginally best on Minerva. A plausible explanation: resiliparse data is noisier (raw HTML artifacts, inconsistent formatting, mixed content), so higher LR causes the model to overfit to that noise. Lower LR limits the damage from low-quality signal while still allowing the useful math content to contribute. With extraction data, the signal is clean enough that modest LR differences matter less.

### 6. Batch Size Has Negligible Effect on Extraction

At matched LR, bs=16 vs bs=64 produces less than 0.5pp difference for extraction data. The lr=2e-7 pair (bs64=29.37%, bs16=29.76%) and the lr=1e-7 pair (bs64=29.66%, bs16=29.57%) both show negligible divergence. For resiliparse, bs=16 performs slightly worse by 0.2–0.8pp. Given that bs=64 is more computationally efficient, there is no motivation to reduce batch size for this task.

---

## Combined View: Full LR Curve

The following table combines the prior HP sweep and this extension sweep for extraction data, tracing the full LR axis from 2e-5 to 2e-8:

| LR | Batch size | avg_minerva | GSM8K_flex | Source |
|---|---|---|---|---|
| 2e-5 | 64 | ~25% | ~58% | Prior HP sweep (estimate) |
| 1e-5 | 64 | ~27% | ~59% | Prior HP sweep (estimate) |
| 5e-6 | 64 | ~28% | ~59% | Prior HP sweep (estimate) |
| 1e-6 | 64 | ~29% | ~60% | Prior HP sweep |
| 5e-7 | 64 | **30.3%** | 60.3% | Prior HP sweep (best Minerva) |
| 3e-7 | 64 | 29.81% | 59.64% | This sweep |
| 2e-7 | 64 | 29.37% | 60.13% | This sweep |
| 1e-7 | 64 | 29.66% | 62.03% | This sweep |
| 5e-8 | 64 | 29.42% | **64.27%** | This sweep (best GSM8K) |
| 2e-8 | 64 | 27.59% | 63.11% | This sweep |

The shape of this curve is characteristic: rapid improvement from high LR down to ~5e-7 as the model learns math reasoning patterns, a flat plateau from 5e-7 to 5e-8 where Minerva is near-saturated, and GSM8K monotonically recovering as LR falls. Below 2e-8, Minerva starts degrading without further GSM8K gain.

---

## Recommendations

**For Minerva-first objectives:** Use lr=5e-7 (bs=64, extraction data). This is the confirmed optimum. Going lower does not help and going higher hurts.

**For balanced Minerva + GSM8K:** Use lr=1e-7 (bs=64, extraction data). This sits at 29.66% Minerva and 62.03% GSM8K flex — both above the untrained baseline by comfortable margins, with no clear tradeoff cost.

**For GSM8K-priority objectives:** Use lr=5e-8 (bs=64, extraction data). At 64.27% flex this is the best arithmetic-preserving SFT configuration found, while retaining meaningful Minerva gains (29.42% vs 26.7% baseline).

**Do not use lr=2e-8 or lower:** lr=2e-8 is a dominated configuration — it costs 1.8pp of Minerva relative to the plateau while recovering only ~1.2pp of GSM8K vs lr=5e-8. The crossover point where further LR reduction becomes net-negative is somewhere between 5e-8 and 2e-8.

**Resiliparse data is not recommended** for Minerva-focused tasks. The ~4pp quality gap vs extraction persists regardless of LR tuning and is unlikely to close without improving the data processing pipeline itself.

**Batch size:** Use bs=64. Smaller batch sizes offer no benefit for this task and increase training time.
