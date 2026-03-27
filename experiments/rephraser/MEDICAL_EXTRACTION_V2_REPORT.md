# Medical Domain Extraction SFT V2 — Complete Results Report

**Date**: 2026-03-28
**Model**: Qwen3-0.6B-Base
**Evals**: MMLU Generative 5-shot (7 medical subtasks)
**V2 Sweep**: 21 HP configs (20 completed, 1 failed training)

## Executive Summary

**V2 extraction improves over V1 but still cannot beat resiliparse.** The revised prompt (broader medical definition, optional Q/R/A format, page 2+ support) produced a modest improvement (+0.42% over V1 extraction best) but extraction remains below baseline. Resiliparse continues to dominate for medical domain with a +2.33% advantage.

| Model | Best AVG | Delta vs Baseline |
|-------|---------|-------------------|
| **Resiliparse best** (lr=7e-6, bs=64) | **53.55%** | **+2.33%** |
| Baseline (Qwen3-0.6B) | 51.22% | — |
| **V2 Extraction best** (lr=5e-6, bs=32) | **51.46%** | **+0.24%** |
| V1 Extraction best (lr=5e-6, bs=32) | 51.04% | -0.18% |

## V2 Prompt Changes (vs V1)

The V2 prompt addressed three systematic failures identified in the V1 deep dive:

1. **Broader medical definition**: Added nursing practice, clinical workflows, caregiver descriptions, scope-of-practice debates
2. **Optional Q/R/A format**: Plain Markdown allowed for non-Q&A content (articles, reference pages, practice discussions)
3. **Page 2+ support**: "For page 2+ of a thread... extract in reading order" instead of forcing Q/R/A
4. **Anti-summarization**: "Do not summarize or compress. Preserve specific details from every reply."
5. **Relaxed filtering**: More specific [NO_USEFUL_CONTENT] criteria to reduce over-filtering

## Dataset Size Comparison

| Dataset | Tokens | Change vs V1 |
|---------|--------|-------------|
| V1 Extraction | 468M | — |
| **V2 Extraction** | **560M** | **+20%** |
| Resiliparse | 2.43B | 4.3x larger than V2 |

The V2 prompt retains 20% more content than V1 (560M vs 468M tokens), confirming the reduced over-filtering. However, the dataset is still 4.3x smaller than resiliparse (down from 5.2x with V1).

## Full V2 Extraction Sweep (20 configs)

### Phase 1: LR x Batch Size (fixed wd=0.01, wu=0.03)

| Rank | Config | AVG | Delta |
|------|--------|-----|-------|
| 1 | lr=5e-6, bs=32 | **51.46%** | +0.24% |
| 2 | lr=5e-6, bs=64 | 51.28% | +0.06% |
| 3 | lr=7e-6, bs=64 | 51.24% | +0.02% |
| 4 | lr=3e-6, bs=32 | 50.95% | -0.27% |
| 5 | lr=3e-6, bs=64 | 50.46% | -0.76% |
| 6 | lr=1e-6, bs=32 | 50.61% | -0.61% |
| 7 | lr=2e-6, bs=32 | 50.50% | -0.72% |
| 8 | lr=2e-6, bs=64 | 50.33% | -0.89% |
| 9 | lr=1e-6, bs=64 | 50.28% | -0.94% |

**Best HP**: lr=5e-6, bs=32 — same as V1 extraction best. The sweet spot is lr=5e-6 to 7e-6.

### Phase 2: Weight Decay x Warmup (fixed lr=2e-6, bs=32)

| Rank | Config | AVG | Delta |
|------|--------|-----|-------|
| 1 | wd=0.01, wu=0.0 | 50.78% | -0.44% |
| 2 | wd=0.001, wu=0.03 | 50.69% | -0.53% |
| 3 | wd=0.1, wu=0.1 | 50.69% | -0.53% |
| 4 | wd=0.01, wu=0.03 | 50.50% | -0.72% |
| 5 | wd=0.05, wu=0.1 | 50.48% | -0.74% |
| 6 | wd=0.01, wu=0.1 | 50.38% | -0.84% |
| 7 | wd=0.05, wu=0.03 | 50.35% | -0.87% |
| 8 | wd=0.05, wu=0.0 | 50.32% | -0.90% |
| 9 | wd=0.001, wu=0.1 | 50.24% | -0.98% |
| 10 | wd=0.001, wu=0.0 | 50.17% | -1.05% |
| 11 | wd=0.1, wu=0.0 | 49.96% | -1.26% |
| — | wd=0.1, wu=0.03 | *FAILED* | — |

No Phase 2 config beats baseline. Weight decay and warmup don't rescue extraction performance.

## Per-Task Breakdown (Key Configs)

| Task | Baseline | Resili Best | V2 Extract Best | V1 Extract Best |
|------|----------|-------------|-----------------|-----------------|
| | | lr=7e-6 bs=64 | lr=5e-6 bs=32 | lr=5e-6 bs=32 |
| anatomy | 48.89% | 49.63% | 46.67% | 45.93% |
| clinical_knowledge | 52.08% | 55.85% | 55.85% | 56.98% |
| college_biology | 56.25% | 54.86% | 52.78% | 54.86% |
| college_medicine | 48.55% | 50.87% | 50.29% | 49.71% |
| high_school_biology | 54.19% | 61.61% | 61.94% | 58.06% |
| medical_genetics | 53.00% | 55.00% | 53.00% | 55.00% |
| professional_medicine | 45.59% | 47.06% | 39.71% | 36.76% |
| **AVERAGE** | **51.22%** | **53.55%** | **51.46%** | **51.04%** |

### Key Observations

1. **professional_medicine still regresses**: V2 extraction drops to 39.71% (vs baseline 45.59%, resiliparse 47.06%). Improved from V1's 36.76% but still the biggest source of regression. This task requires broad clinical reasoning that extraction may strip away.

2. **high_school_biology improves significantly**: V2 extraction hits 61.94% — actually beating resiliparse (61.61%) on this single task. The broader medical definition may help retain biology-adjacent content.

3. **clinical_knowledge strong**: V2 extraction matches resiliparse at 55.85% on clinical knowledge, suggesting the extraction quality is high for factual medical content.

4. **anatomy regresses**: V2 extraction (46.67%) below baseline (48.89%). Anatomical content may be filtered or reformatted by the extraction prompt in ways that lose information.

## V1 vs V2 Extraction Comparison

| Metric | V1 Extraction | V2 Extraction | Change |
|--------|--------------|---------------|--------|
| Tokens | 468M | 560M | +20% |
| Best AVG | 51.04% | 51.46% | **+0.42%** |
| Best HP | lr=5e-6, bs=32 | lr=5e-6, bs=32 | Same |
| Configs above baseline | 0/21 | 3/20 | Improved |
| professional_medicine | 36.76% | 39.71% | +2.95% |
| high_school_biology | 58.06% | 61.94% | +3.88% |

The V2 prompt helped:
- **+20% more tokens** from reduced over-filtering
- **+0.42% average** improvement
- **+2.95% professional_medicine** (still below baseline, but less regression)
- **+3.88% high_school_biology** (now beats resiliparse on this task)

But it wasn't enough to close the gap with resiliparse.

## Why Resiliparse Still Wins

1. **4.3x more tokens**: Resiliparse produces 2.43B tokens vs V2 extraction's 560M. For medical domain, data volume dominates data quality at this scale.

2. **Extraction still loses content**: Even with the V2 prompt's broader definition, the extraction model makes judgment calls about what's "medical" that systematically remove useful medical vocabulary. The extraction model can't perfectly replicate the training signal that raw text provides.

3. **professional_medicine is the key differentiator**: This task alone accounts for most of the gap. It tests broad clinical reasoning across specialties — exactly the kind of diverse medical context that resiliparse preserves but extraction distills away.

4. **Diminishing returns from prompt engineering**: V2 improved V1 by only +0.42% despite significant prompt changes. The extraction approach appears to have a ceiling for medical content — more prompt iteration is unlikely to close the 2% gap with resiliparse.

## Cross-Domain Summary

| Domain | Extraction Best | Resiliparse Best | Winner |
|--------|----------------|-----------------|--------|
| **Code** (HumanEval) | **29.9%** | 26.2% | **Extraction** |
| **Medical** (MMLU avg) | 51.46% | **53.55%** | **Resiliparse** |

The extraction approach excels for **code** (where structured format helps) but loses for **medical** (where volume and diversity matter more). This suggests extraction is not universally better — it depends on domain characteristics.

## Recommendations

1. **Use resiliparse for medical domain** with lr=7e-6, bs=64 (+2.33% over baseline)
2. **Extraction has a ceiling for medical**: Two prompt iterations (V1→V2) gained only +0.42%. Further prompt engineering unlikely to close the 2% gap.
3. **Consider hybrid approaches**: Mix extraction and resiliparse data (e.g., use extraction for forum Q&A where structure helps, resiliparse for reference sites where volume helps)
4. **The "right" extraction approach may differ by domain**: Code benefits from structure, medical benefits from volume. Future domains should be evaluated empirically.
5. **Best extraction HP is consistent**: lr=5e-6, bs=32 was optimal for both V1 and V2, suggesting this HP is robust for extraction-based SFT regardless of prompt.
