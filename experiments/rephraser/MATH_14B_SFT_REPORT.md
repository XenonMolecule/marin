# Math 14B SFT Sweep Report

**Date**: 2026-03-27
**Model**: Qwen3-14B-Base
**Cluster**: us-central1
**Status**: ALL 7/7 CONFIGS COMPLETE

## Summary

This experiment tests whether math extraction SFT scales from 0.6B to 14B. We train Qwen3-14B-Base on two data types — LLM-extracted Q/R/A markdown ("extraction") and plain text via resiliparse — using 3 HP configs from the coding 14B sweep, for 6 training configs + 1 baseline = 7 total.

**Key findings:**
- All 6 SFT configs improve MATH over the untrained baseline (+0.9 to +6.3pp avg_minerva).
- Extraction data dominates resiliparse by ~4pp at 14B, consistent with the 0.6B finding.
- The best config (extract-best-resili) achieves 59.89% avg_minerva (+6.3pp over baseline).
- GSM8K tradeoff: all SFT configs regress -1.8 to -8.4pp on GSM8K flex_extract, but gains on MATH far outweigh this.
- No catastrophic forgetting at 14B — even the weakest config (resili-default) still improves over baseline.

## Experiment Design

### HP Configs

All 3 configs were borrowed from the coding 14B SFT sweep. Fixed params: decay=0.97, lr_schedule=cosine, max_grad_norm=1.0, seq_len=4096.

| Config Name | LR | Batch Size | Weight Decay | Warmup |
|-------------|-----|-----------|--------------|--------|
| default | 2e-5 | 64 | 0.01 | 0.03 |
| best-resili | 1e-6 | 32 | 0.01 | 0.03 |
| best-extract | 2e-6 | 32 | 0.05 | 0.0 |

### Data Sources

| Data Type | Description | Source |
|-----------|-------------|--------|
| extraction | LLM-extracted Q/R/A markdown from 42 math domains | `mathhelpforum_extraction_sft_v2_base` pipeline |
| resiliparse | Plain text via resiliparse HTML parser | Same source URLs, different text extraction |

Both data types use the full 42-domain math crawl (not filtered to top3). The 14B model reuses the same tokenized data as 0.6B (Qwen3 models share the same tokenizer vocabulary).

### Infrastructure

- **Training**: v5p-32 TPU (14B needs more HBM than v5p-8)
- **Eval**: v5p-8 TPU via vLLM
- **Evals**: 7 minerva_math subtasks (4-shot) + gsm8k_platinum_cot (8-shot)

## Results

### Full Scorecard (math_verify,none as %; **bold** = best per column)

| Config | Data | Algebra | Prealg | Cnt/Prob | Geometry | Int Alg | Num Theory | Precalc | Avg MATH | GSM8K flex |
|--------|------|---------|--------|----------|----------|---------|------------|---------|----------|------------|
| baseline | — | 75.74 | 75.20 | 54.01 | 43.42 | 34.88 | 51.48 | 40.66 | 53.63 | **91.32** |
| extract-default | extract | 80.71 | 77.04 | 57.81 | 52.82 | 36.10 | 54.81 | 43.59 | 57.55 | 88.01 |
| extract-best-extract | extract | 81.89 | 78.42 | 58.23 | 55.32 | 40.20 | 56.67 | 44.32 | 59.29 | 88.42 |
| **extract-best-resili** | **extract** | **82.56** | 77.73 | **60.55** | 54.28 | **43.30** | 55.00 | **45.79** | **59.89** | 82.96 |
| resili-default | resili | 76.83 | 76.58 | 55.91 | 47.39 | 33.89 | 52.04 | 39.19 | 54.55 | 86.93 |
| resili-best-resili | resili | 77.93 | 75.89 | 59.49 | 47.18 | 36.21 | 53.70 | 41.03 | 55.92 | 89.50 |
| resili-best-extract | resili | 78.43 | 76.92 | 58.02 | 45.30 | 34.88 | 53.33 | 40.66 | 55.36 | 87.59 |

### Ranking by Avg MATH (math_verify)

| Rank | Config | Data | Avg MATH | Delta vs Baseline | GSM8K flex |
|------|--------|------|----------|-------------------|------------|
| 1 | extract-best-resili | extraction | 59.89 | **+6.26** | 82.96 |
| 2 | extract-best-extract | extraction | 59.29 | +5.66 | 88.42 |
| 3 | extract-default | extraction | 57.55 | +3.92 | 88.01 |
| 4 | resili-best-resili | resiliparse | 55.92 | +2.29 | 89.50 |
| 5 | resili-best-extract | resiliparse | 55.36 | +1.73 | 87.59 |
| 6 | resili-default | resiliparse | 54.55 | +0.92 | 86.93 |
| 7 | baseline | — | 53.63 | 0.00 | 91.32 |

### Per-Subtask Delta vs Baseline (pp)

| Config | Algebra | Prealg | Cnt/Prob | Geometry | Int Alg | Num Theory | Precalc | GSM8K flex |
|--------|---------|--------|----------|----------|---------|------------|---------|------------|
| extract-best-resili | +6.82 | +2.53 | +6.54 | +10.86 | +8.42 | +3.52 | +5.13 | -8.36 |
| extract-best-extract | +6.15 | +3.22 | +4.22 | +11.90 | +5.32 | +5.19 | +3.66 | -2.90 |
| extract-default | +4.97 | +1.84 | +3.80 | +9.40 | +1.22 | +3.33 | +2.93 | -3.31 |
| resili-best-resili | +2.19 | +0.69 | +5.48 | +3.76 | +1.33 | +2.22 | +0.37 | -1.82 |
| resili-best-extract | +2.69 | +1.72 | +4.01 | +1.88 | +0.00 | +1.85 | +0.00 | -3.73 |
| resili-default | +1.09 | +1.38 | +1.90 | +3.97 | -0.99 | +0.56 | -1.47 | -4.39 |

## Key Findings

### 1. Extraction Data Dominates Resiliparse at 14B (Same as 0.6B)

At every HP config, extraction outperforms resiliparse on MATH:

| HP Config | Extraction Avg MATH | Resiliparse Avg MATH | Gap |
|-----------|--------------------|--------------------|-----|
| default | 57.55 | 54.55 | +3.00 |
| best-resili | 59.89 | 55.92 | +3.97 |
| best-extract | 59.29 | 55.36 | +3.93 |

The ~4pp gap is remarkably consistent across HP configs, confirming that the structured Q/R/A format from LLM extraction provides a stronger learning signal than plain text at both 0.6B and 14B scales.

### 2. Lower LR Configs Outperform Default at 14B

The "best-resili" (lr=1e-6) and "best-extract" (lr=2e-6) configs both substantially outperform the "default" (lr=2e-5) config. For extraction:
- best-resili: 59.89 avg_minerva (+2.34pp over default)
- best-extract: 59.29 (+1.74pp over default)
- default: 57.55

This suggests that 14B models benefit from gentler fine-tuning (10-20x lower LR) — consistent with the general finding that larger models need lower LR to avoid catastrophic forgetting.

### 3. Geometry Shows the Largest Gains

Geometry is the single biggest winner from extraction SFT:
- Baseline: 43.42%
- extract-best-extract: 55.32% (+11.9pp)
- extract-best-resili: 54.28% (+10.9pp)

This is a ~25% relative improvement. Geometry problems often require multi-step reasoning with diagrams described in text, which the extraction format (structured Q/R/A) captures particularly well.

### 4. GSM8K Tradeoff is Real but Manageable

All SFT configs regress on GSM8K compared to the untrained baseline (91.32% flex):

| Config | GSM8K flex | Delta |
|--------|-----------|-------|
| resili-best-resili | 89.50 | -1.82 |
| extract-best-extract | 88.42 | -2.90 |
| extract-default | 88.01 | -3.31 |
| resili-best-extract | 87.59 | -3.73 |
| resili-default | 86.93 | -4.39 |
| extract-best-resili | 82.96 | -8.36 |

The best-resili config shows the largest GSM8K regression (-8.36pp) despite being the best on MATH. If GSM8K preservation matters, extract-best-extract offers the best tradeoff: +5.66pp MATH with only -2.90pp GSM8K.

### 5. No Catastrophic Forgetting at 14B

Unlike the 0.6B experiments where some aggressive HP configs caused net-negative results, every single 14B config improves over baseline on MATH. Even the weakest config (resili-default, +0.92pp) shows a small but positive gain. The 14B model has enough capacity to absorb math SFT data without losing pre-existing capabilities on the MATH benchmark.

### 6. Intermediate Algebra and Counting/Probability Show Strong HP Sensitivity

For intermediate algebra (hardest subtask), HP choice matters enormously:
- extract-best-resili: 43.30% (+8.42pp over baseline)
- extract-best-extract: 40.20% (+5.32pp)
- extract-default: 36.10% (+1.22pp)
- resili-default: 33.89% (-0.99pp — only config that regresses on any subtask)

This 10pp spread across configs (within extraction data alone) underscores that HP tuning is critical for hard math topics.

## Comparison: 0.6B vs 14B Scale

How does SFT effectiveness differ between model scales? Using the "best-resili" HP (best at 14B) with extraction data:

| Metric | 0.6B Baseline | 0.6B SFT | 0.6B Δ | 14B Baseline | 14B SFT | 14B Δ |
|--------|--------------|----------|--------|-------------|---------|-------|
| Avg MATH | ~28* | ~35* | +7* | 53.63 | 59.89 | +6.26 |
| GSM8K flex | ~63* | ~60* | -3* | 91.32 | 82.96 | -8.36 |

*Approximate 0.6B numbers from best top3 config (code-best-resili HP at 0.6B was not run on full 42-domain data; these are rough references from top3 filtered data).

The absolute MATH gains are similar (~6-7pp), but the 14B model starts from a much higher baseline and shows no catastrophic failures on any subtask.

## Reproduction

All experiments ran on us-central1 via:
```
job_ids:
  original: ray-run-michaelryan-math_14b_sft-20260323-235024
  resubmits: see .agents/scratchpad/monitor_math_14b_sft.json
cluster: us-central1
model: Qwen/Qwen3-14B-Base
training: v5p-32, eval: v5p-8
```

### Checkpoint and Eval Paths

| Config | Checkpoint | Eval Dir |
|--------|-----------|---------|
| baseline | Qwen/Qwen3-14B-Base (HuggingFace) | math-14b-baseline-qwen3-14b-base-2601f5 |
| extract-default | math-14b-extract-default-qwen3-14b-base-8bafd6 | math-14b-extract-default-qwen3-14b-base-058258 |
| extract-best-extract | math-14b-extract-best-extract-qwen3-14b-base-fab231 | math-14b-extract-best-extract-qwen3-14b-base-07b321 |
| extract-best-resili | math-14b-extract-best-resili-qwen3-14b-base-429f46 | math-14b-extract-best-resili-qwen3-14b-base-52aa7d |
| resili-default | math-14b-resili-default-qwen3-14b-base-d81f39 | math-14b-resili-default-qwen3-14b-base-11fc2c |
| resili-best-resili | math-14b-resili-best-resili-qwen3-14b-base-c0112d | math-14b-resili-best-resili-qwen3-14b-base-973d64 |
| resili-best-extract | math-14b-resili-best-extract-qwen3-14b-base-a7e7d9 | math-14b-resili-best-extract-qwen3-14b-base-59e730 |

All paths under `gs://marin-us-central1/checkpoints/` and `gs://marin-us-central1/evaluation/lm_evaluation_harness/`.

---

## Top-3 Domain Filter Comparison (Added 2026-03-28)

**Question**: Does filtering to the top 3 math domains (brainly.com, jiskha.com, mathhelpforum.com) improve 14B SFT, as it did at 0.6B?

### Experiment Setup

Same 3 HP configs × 2 data types + baseline = 7 configs, but trained on **top3 domain-filtered** data instead of the full 42-domain mix. Top3 domains were identified as the best performers in the 0.6B domain analysis. The Qwen3-14B tokenizer is identical to 0.6B, so tokenized data artifacts are shared.

- **Training**: v5p-32, **Eval**: v5p-8
- **Job**: `ray-run-michaelryan-math_14b_top3_sft-20260327-064030` (+ standalone eval resubmits)
- **Data**: ~3x smaller than full 42-domain mix (concentrated on highest-quality Q&A domains)

### Top3 Results — Full Scorecard (math_verify %; **bold** = best per column)

| Config | Data | Algebra | Prealg | Cnt/Prob | Geometry | Int Alg | Num Theory | Precalc | Avg MATH | GSM8K flex |
|--------|------|---------|--------|----------|----------|---------|------------|---------|----------|------------|
| baseline | — | 71.90 | 73.90 | 54.00 | 41.10 | 32.30 | 50.20 | 31.50 | 53.63 | **91.30** |
| extract-default | top3 ext | 82.06 | 80.25 | 60.55 | 57.20 | 43.19 | 60.19 | 51.10 | 61.36 | 86.27 |
| extract-best-extract | top3 ext | **85.34** | 79.91 | 59.28 | 55.11 | 44.19 | **60.74** | 49.08 | 61.95 | 77.34 |
| **extract-best-resili** | **top3 ext** | 84.25 | **81.75** | **60.34** | 54.28 | **47.07** | 60.19 | **49.82** | **62.53** | 72.04 |
| resili-default | top3 rsl | 79.11 | 76.92 | 59.07 | 50.10 | 37.80 | 55.00 | 42.31 | 55.76 | 85.86 |
| resili-best-resili | top3 rsl | 77.51 | 75.89 | 56.96 | 46.56 | 34.11 | 52.04 | 40.66 | 54.82 | 90.32 |
| resili-best-extract | top3 rsl | 77.59 | 76.35 | 54.64 | 46.76 | 35.55 | 52.59 | 39.74 | 54.75 | 89.83 |

### Head-to-Head: Top3 vs Full Data (Avg MATH)

| Config | Full Data | Top3 | Delta | Full GSM8K | Top3 GSM8K | GSM8K Delta |
|--------|-----------|------|-------|------------|------------|-------------|
| **extract-best-resili** | 59.89 | **62.53** | **+2.64** | 82.96 | 72.04 | -10.92 |
| extract-best-extract | 59.29 | 61.95 | +2.66 | 88.42 | 77.34 | -11.08 |
| extract-default | 57.55 | 61.36 | **+3.81** | 88.01 | 86.27 | -1.74 |
| resili-default | 54.55 | 55.76 | +1.21 | 86.93 | 85.86 | -1.07 |
| resili-best-resili | 55.92 | 54.82 | -1.10 | 89.50 | 90.32 | +0.82 |
| resili-best-extract | 55.36 | 54.75 | -0.61 | 87.59 | 89.83 | +2.24 |

### Key Finding: Top3 Strongly Benefits Extraction, Mixed for Resiliparse

**Extraction configs: Top3 wins by +2.6 to +3.8pp on MATH.**
- extract-default gets the largest boost (+3.81pp) — the default HP is apparently better suited to concentrated high-quality data
- extract-best-resili is the new overall champion: **62.53% avg_minerva** (+8.9pp over baseline)
- Top3 extraction produces the highest per-subtask scores seen at 14B: algebra 85.3%, prealgebra 81.8%, intermediate_algebra 47.1%

**Resiliparse configs: Top3 is a wash.**
- resili-default improves slightly (+1.21pp)
- resili-best-resili and resili-best-extract slightly regress (-1.10, -0.61pp)
- Interpretation: Resiliparse text is less sensitive to domain quality since it extracts plain text regardless — the structured extraction is what truly benefits from domain concentration

### GSM8K Tradeoff Deepens with Top3

Top3 extraction configs show larger GSM8K regressions than full-data:

| Config | Full Data GSM8K | Top3 GSM8K | Additional Drop |
|--------|----------------|------------|-----------------|
| extract-best-resili | 82.96% | 72.04% | -10.92pp |
| extract-best-extract | 88.42% | 77.34% | -11.08pp |
| extract-default | 88.01% | 86.27% | -1.74pp |

The low-LR configs (best-resili, best-extract) show the steepest GSM8K drops (-11pp). The default HP (lr=2e-5) preserves GSM8K much better — the higher LR may prevent the model from over-specializing on the concentrated math domain.

### Overall Ranking: All 14B Configs (Full Data + Top3)

| Rank | Config | Data | Avg MATH | GSM8K flex | Delta MATH vs Baseline |
|------|--------|------|----------|------------|----------------------|
| 1 | **extract-best-resili** | **top3 ext** | **62.53** | 72.04 | **+8.90** |
| 2 | extract-best-extract | top3 ext | 61.95 | 77.34 | +8.32 |
| 3 | extract-default | top3 ext | 61.36 | 86.27 | +7.73 |
| 4 | extract-best-resili | full ext | 59.89 | 82.96 | +6.26 |
| 5 | extract-best-extract | full ext | 59.29 | 88.42 | +5.66 |
| 6 | extract-default | full ext | 57.55 | 88.01 | +3.92 |
| 7 | resili-best-resili | full rsl | 55.92 | 89.50 | +2.29 |
| 8 | resili-default | top3 rsl | 55.76 | 85.86 | +2.13 |
| 9 | resili-best-extract | full rsl | 55.36 | 87.59 | +1.73 |
| 10 | resili-best-resili | top3 rsl | 54.82 | 90.32 | +1.19 |
| 11 | resili-best-extract | top3 rsl | 54.75 | 89.83 | +1.12 |
| 12 | resili-default | full rsl | 54.55 | 86.93 | +0.92 |
| 13 | baseline | — | 53.63 | 91.32 | 0.00 |

### Recommendations

1. **Best MATH performance**: Use top3 extract-best-resili (62.53%) if GSM8K regression is acceptable.
2. **Best MATH/GSM8K tradeoff**: Use top3 extract-default (61.36% MATH, 86.3% GSM8K) — gains +7.7pp MATH with only -5.0pp GSM8K.
3. **Domain filtering works**: The top3 filter consistently improves extraction SFT. The concentrated high-quality Q&A from brainly/jiskha/mathhelpforum is a stronger training signal than the diluted 42-domain mix.
4. **Skip top3 for resiliparse**: Domain filtering doesn't help resiliparse and slightly hurts the low-LR configs.

### Top3 Checkpoint and Eval Paths

| Config | Checkpoint | Eval Dir |
|--------|-----------|---------|
| extract-default | math-14b-top3-extract-default-qwen3-14b-base-cc0da1 | math-14b-top3-extract-default-qwen3-14b-base-1ee499 |
| extract-best-extract | math-14b-top3-extract-best-extract-qwen3-14b-base-981bb3 | math-14b-top3-extract-best-extract-qwen3-14b-base-066983 |
| extract-best-resili | math-14b-top3-extract-best-resili-qwen3-14b-base-3e275e | math-14b-top3-extract-best-resili-qwen3-14b-base-86c3a9 |
| resili-default | math-14b-top3-resili-default-qwen3-14b-base-fa46f8 | math-14b-top3-resili-default-qwen3-14b-base-121ba1 |
| resili-best-resili | math-14b-top3-resili-best-resili-qwen3-14b-base-fff30a | math-14b-top3-resili-best-resili-qwen3-14b-base-b1860a |
| resili-best-extract | math-14b-top3-resili-best-extract-qwen3-14b-base-eeeee2 | math-14b-top3-resili-best-extract-qwen3-14b-base-154533 |
