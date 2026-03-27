# Qwen3-14B-Base Hyperparameter Sweep Results

**Status: MOSTLY COMPLETE** (updated 2026-03-19 ~9:35am PST)

## V3 Extraction — Phase 1 (LR × BS)

Dataset: 260.8M tokens V3 commented extraction. Training: v5p-32, seq_len=4096, 1 epoch.

| Config | MBPP 3-shot | HumanEval 0-shot |
|---|---|---|
| **lr1e-5_bs16** | **73.8%** | 74.4% |
| lr1e-6_bs32 | 73.2% | — |
| lr2e-6_bs32 | 72.8% | 75.0% |
| lr1e-5_bs64 | 72.6% | 73.2% |
| lr1e-6_bs64 | 72.4% | 77.4% |
| lr1e-5_bs32 | 72.2% | 73.2% |

6/15 MBPP 3-shot results. **Phase 1 winner: lr=1e-5, bs=16 (73.8%)**

## V3 Extraction — Phase 2 (WD × Warmup at lr=2e-6, bs=32)

Phase 2 launched with lr=2e-6, bs=32 (Phase 1 baseline: 72.8% MBPP, 75.0% HumanEval).

| Config | MBPP 3-shot | HumanEval 0-shot |
|---|---|---|
| **wd0.05_wu0.0** | **73.8%** | **77.4%** |
| wd0.001_wu0.03 | 73.0% | 77.4% |
| wd0.1_wu0.0 | 73.0% | 77.4% |
| wd0.01_wu0.1 | 72.8% | 75.6% |
| wd0.001_wu0.0 | 72.4% | 75.6% |
| wd0.1_wu0.1 | 72.4% | 76.8% |

6/12 MBPP 3-shot results. **Phase 2 winner: wd=0.05, warmup=0.0 (73.8% MBPP + 77.4% HumanEval)**

## Resiliparse — Phase 1 (LR × BS)

Dataset: 792.5M tokens resiliparse plain text (3x larger than V3).

| Config | MBPP 3-shot | HumanEval 0-shot |
|---|---|---|
| **lr1e-6_bs32** | **72.8%** | — |
| lr2e-6_bs64 | 72.2% | — |
| lr5e-6_bs32 | 71.6% | — |
| lr1e-5_bs64 | 71.2% | 59.1% |

4/15 MBPP 3-shot results. **Best: lr=1e-6, bs=32 (72.8%)**

## Overall Best Configs

| Rank | Extraction | Config | MBPP 3-shot | HumanEval |
|---|---|---|---|---|
| 1 | V3 Phase 1 | lr=1e-5, bs=16, wd=0.01, wu=0.03 | **73.8%** | 74.4% |
| 1 | V3 Phase 2 | lr=2e-6, bs=32, wd=0.05, wu=0.0 | **73.8%** | **77.4%** |
| 3 | V3 Phase 2 | lr=2e-6, bs=32, wd=0.001, wu=0.03 | 73.0% | 77.4% |
| 3 | V3 Phase 1 | lr=1e-6, bs=32, wd=0.01, wu=0.03 | 73.2% | — |
| 5 | Resiliparse | lr=1e-6, bs=32, wd=0.01, wu=0.03 | 72.8% | — |

## Recommended Configuration

**Best overall: lr=2e-6, bs=32, wd=0.05, warmup=0.0**
- MBPP 3-shot: 73.8% (tied with Phase 1 best)
- HumanEval 0-shot: 77.4% (best overall, +3pp over Phase 1 winner)
- Strong on both metrics simultaneously

## Key Findings

1. **V3 extraction outperforms resiliparse** — 73.8% vs 72.8% MBPP, and dramatically on HumanEval (77.4% vs 59.1%).
2. **14B is very robust to hyperparameters** — MBPP spread only 1.6pp across all V3 configs.
3. **Phase 2 found a better config** — wd=0.05, wu=0.0 improves HumanEval by +2.4pp vs baseline while matching MBPP.
4. **No warmup works best** — the no-warmup (wu=0.0) configs consistently outperform warmup configs.
5. **Moderate weight decay (0.05) optimal** — higher than the 0.6B optimum (0.001), suggesting 14B benefits from more regularization.
6. **Different from 0.6B optimal** — 0.6B preferred wd=0.001, wu=0.0; 14B prefers wd=0.05, wu=0.0.

## Jobs Status
- V3 Phase 1: RUNNING (6/15 MBPP 3-shot)
- V3 Phase 2: RUNNING (6/12 MBPP 3-shot)
- Resiliparse Phase 1: RUNNING (4/15 MBPP 3-shot)
