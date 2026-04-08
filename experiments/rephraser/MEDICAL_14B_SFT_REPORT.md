# Medical 14B SFT Report

**Date**: 2026-04-01
**Model**: Qwen3-14B-Base
**Cluster**: us-central1
**Status**: ALL 7/7 CONFIGS COMPLETE

## Summary

This experiment tests whether LLM extraction or resiliparse produces better medical SFT data at 14B scale. At 0.6B, resiliparse dominated (+2.3% over baseline) while extraction barely helped (+0.2%). **At 14B, the trend reverses: extraction wins.**

| Scale | Extraction Best | Resiliparse Best | Winner |
|-------|----------------|-----------------|--------|
| **0.6B** | 51.5% (+0.2%) | **53.6% (+2.3%)** | **Resiliparse** |
| **14B** | **86.9% (+0.9%)** | 86.6% (+0.6%) | **Extraction** |

This is consistent with code and math domains, where extraction also wins at 14B.

## What We Did

We compared two approaches for turning Common Crawl medical web pages into SFT training data:

1. **LLM Extraction**: Feed raw HTML to Qwen3-235B-A22B, which extracts medical content into structured Markdown (Q/A format for forum threads, clean prose for reference pages). Produces 560M tokens from 13 medical domains (healthboards.com, allnurses.com, WebMD, Mayo Clinic, etc.).

2. **Resiliparse**: Use the resiliparse HTML parser to strip tags and extract plain text. No LLM involved — just rule-based HTML-to-text conversion. Produces 2.43B tokens (4.3x more than extraction) from the same source URLs.

Both datasets are used for single-epoch supervised fine-tuning of Qwen3-14B-Base. We evaluate on 7 MMLU medical subtasks (generative, 5-shot): anatomy, clinical knowledge, college biology, college medicine, high school biology, medical genetics, and professional medicine.

### HP Configs

We reused 3 HP configs from the coding 14B sweep rather than sweeping again. Fixed params: decay=0.97, lr_schedule=cosine, max_grad_norm=1.0, seq_len=4096.

| Config Name | LR | Batch Size | Weight Decay | Warmup | Tokens/Epoch |
|-------------|-----|-----------|--------------|--------|-------------|
| default | 2e-5 | 64 | 0.01 | 0.03 | — |
| best-resili | 1e-6 | 32 | 0.01 | 0.03 | — |
| best-extract | 2e-6 | 32 | 0.05 | 0.0 | — |

Each HP config is crossed with both data types (extraction and resiliparse), giving 6 training runs. Plus 1 baseline eval of unmodified Qwen3-14B-Base = 7 total configs.

### Infrastructure

- **Training**: v5p-32 TPU (14B needs more HBM than v5p-8)
- **Eval**: v5p-8 TPU via vLLM
- **Training time**: ~18-19 hours per 18.6K-step run, ~7 hours per 9.3K-step run

## Results

### Ranking

| Rank | Config | Data | HP | AVG | Delta vs Baseline |
|------|--------|------|----|-----|-------------------|
| 1 | **extract-default** | **Extraction** | lr=2e-5, bs=64 | **86.9%** | **+0.9%** |
| 2 | resili-best-extract | Resiliparse | lr=2e-6, bs=32 | 86.6% | +0.6% |
| 3 | extract-best-resili | Extraction | lr=1e-6, bs=32 | 86.2% | +0.2% |
| 4 | resili-best-resili | Resiliparse | lr=1e-6, bs=32 | 86.1% | +0.2% |
| 5 | **baseline** | — | — | **86.0%** | — |
| 6 | extract-best-extract | Extraction | lr=2e-6, bs=32 | 86.0% | -0.0% |
| 7 | resili-default | Resiliparse | lr=2e-5, bs=64 | 85.2% | -0.8% |

### Per-Task Breakdown (exact_match %)

| Task | Baseline | Extract default | Resili best-extract | Resili default |
|------|----------|----------------|--------------------:|---------------:|
| anatomy | 73.3 | 74.8 | 74.8 | 71.9 |
| clinical_knowledge | 82.3 | 84.2 | 83.8 | 84.2 |
| college_biology | 90.3 | **94.4** | 92.4 | 92.4 |
| college_medicine | **85.5** | 83.2 | 83.2 | 81.5 |
| high_school_biology | 93.9 | **95.2** | 94.8 | 94.8 |
| medical_genetics | 88.0 | **90.0** | 89.0 | 86.0 |
| professional_medicine | **88.6** | 86.4 | 87.9 | 85.7 |
| **AVERAGE** | **86.0** | **86.9** | **86.6** | **85.2** |

## Key Findings

### 1. Extraction wins at 14B — reversing the 0.6B result

At 0.6B, resiliparse's 4.3x data volume advantage dominated: every resiliparse config beat baseline, while no extraction config did. At 14B, extraction takes the lead. The best extraction config (86.9%) beats the best resiliparse config (86.6%) by 0.3 percentage points.

This is consistent with code (extraction wins at both scales) and math (extraction wins at both scales). Medical was the only domain where resiliparse won at 0.6B, and even that advantage disappears at 14B.

### 2. The default HP (lr=2e-5, bs=64) is best for extraction, worst for resiliparse

The "default" HP config (lr=2e-5, bs=64) was the best extraction config (+0.9%) but the worst resiliparse config (-0.8%, actually below baseline). This high learning rate works well for the smaller, cleaner extraction dataset but likely causes overfitting or catastrophic forgetting on the much larger (4.3x) resiliparse dataset.

Resiliparse prefers lower learning rates: its best config uses lr=2e-6 with bs=32.

### 3. Margins are much smaller at 14B

At 0.6B, the gap between best resiliparse and best extraction was 2.1 percentage points. At 14B, it's only 0.3pp. The 14B model is already strong on medical knowledge (86.0% baseline), so there's less room for improvement from either data source. The ceiling effect compresses the differences.

### 4. professional_medicine no longer regresses

At 0.6B, extraction caused a dramatic regression on professional_medicine (45.6% baseline → 36.8% extraction, -8.8pp). At 14B, the regression is much smaller (88.6% → 86.4%, -2.2pp) and resiliparse also regresses on this task (88.6% → 85.7-87.9%). The 14B model's stronger baseline makes it more robust to SFT-induced forgetting.

### 5. college_biology and high_school_biology are the biggest wins

Extraction's strongest gains are in biology: college_biology jumps from 90.3% to 94.4% (+4.1pp) and high_school_biology from 93.9% to 95.2% (+1.3pp). These tasks test factual biological knowledge that the extraction prompt preserves well in structured format.

## Cross-Domain Summary (14B)

| Domain | Eval | Extraction Best | Resiliparse Best | Winner | Margin |
|--------|------|----------------|-----------------|--------|--------|
| **Code** | HumanEval | 55.5% | 51.8% | **Extraction** | +3.7pp |
| **Math** | avg MATH (minerva) | 59.9% | 55.9% | **Extraction** | +4.0pp |
| **Medical** | avg MMLU medical | 86.9% | 86.6% | **Extraction** | +0.3pp |

**Extraction wins across all three domains at 14B.** The margin is largest for math (+4.0pp), moderate for code (+3.7pp), and slim for medical (+0.3pp). Medical is the domain where the data volume vs. data quality tradeoff is most balanced — the 4.3x resiliparse volume advantage nearly compensates for extraction's structured format advantage.

## Cross-Scale Summary (Extraction vs Resiliparse)

| Domain | 0.6B Winner | 14B Winner | Trend |
|--------|------------|-----------|-------|
| **Code** | Extraction | Extraction | Consistent |
| **Math** | Extraction | Extraction | Consistent |
| **Medical** | Resiliparse | Extraction | **Reversed at scale** |

Medical is the only domain where the winner changes with scale. At 0.6B, the small model benefits more from resiliparse's raw data volume (2.43B tokens). At 14B, the larger model can better leverage extraction's structured, clean format — even though it has 4.3x fewer tokens.

## Data Sources

13 medical domains from Common Crawl, spanning forums, reference sites, and clinical resources:

**Forums** (highest extraction value): healthboards.com, allnurses.com, medhelp.org, healthunlocked.com, patient.info, forums.studentdoctor.net

**Reference** (large, well-structured): webmd.com, mayoclinic.org, drugs.com, clevelandclinic.org, medlineplus.gov, merckmanuals.com, ncbi.nlm.nih.gov

**Crawl indices**: 10 indices from CC-MAIN-2013-48 through CC-MAIN-2025-47.

## Extraction Prompt (V2)

The extraction prompt instructs Qwen3-235B-A22B to convert raw HTML into clean medical Markdown:

- **What to extract**: All content related to medicine, health, clinical practice, or patient care — including nursing practice, clinical workflows, diagnostic reasoning, drug protocols
- **Format**: Q/R/A for forum threads with clear questions; plain Markdown for articles, reference pages, practice discussions
- **Filter**: Output `[NO_USEFUL_CONTENT]` only for genuinely non-medical pages (login forms, directories, non-English content)
- **Critical rules**: No added information, no summarizing, preserve all medical terms/dosages/lab values

This is the V2 prompt, revised from V1 based on data analysis that found V1 was over-filtering nursing content and struggling with page 2+ of forum threads.

## Experiment Script

`experiments/rephraser/medical_14b_sft.py`
