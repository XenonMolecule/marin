# 100-WARC Dense Grid Plan

**Goal:** map the (N, C) space at WARCs=100 for all three quality tiers
(low/med/high_quality) so the scaling-law fit can identify cross-method
crossovers without relying on synthetic drags.

**Date:** 2026-05-24

## Grid

- **Methods:** `low_quality`, `med_quality`, `high_quality` (3)
- **N (hidden_dim):** 68M (d=256), 156M (d=512), 272M (d=768), 998M (d=1536)
- **C (FLOPs):** 3e17, 1e18, 3e18, 1e19, 3e19, 1e20
- **WARCs:** 100 (fixed)

Full Cartesian product: 4 N × 6 C × 3 methods = **72 runs total**.

## Cells with token counts and epoch counts

Tokens-per-WARC: low=15.06M, med=9.41M, high=3.69M → at WARCs=100:
U_low=1.51B, U_med=941M, U_high=369M. D = C/(6N).

| N | C | D (tokens) | D/N | ε_low | ε_med | ε_high | notes |
|---:|---:|---:|---:|---:|---:|---:|---|
| 68M | 3e17 | 0.74B | 11 | 0.49 | 0.79 | 2.00 | param-bound (sub-epoch low/med) |
| 68M | 1e18 | 2.45B | 36 | 1.62 | 2.60 | 6.64 | Chinchilla-ish |
| 68M | 3e18 | 7.35B | 109 | 4.87 | 7.81 | 19.9 | mild over-epoch |
| 68M | 1e19 | 24.5B | 360 | 16.2 | 26.0 | 66.4 | over-epoch all methods |
| 68M | 3e19 | 73.5B | 1090 | 48.7 | 78.1 | 199 | deep over-epoch |
| 68M | 1e20 | 245B | 3600 | 162 | 260 | 664 | extreme |
| 156M | 3e17 | 0.32B | 2.1 | 0.21 | 0.34 | 0.87 | very undertrained — low signal |
| 156M | 1e18 | 1.07B | 6.9 | 0.71 | 1.13 | 2.89 | param-bound |
| 156M | 3e18 | 3.21B | 21 | 2.12 | 3.41 | 8.69 | Chinchilla-ish |
| 156M | 1e19 | 10.7B | 69 | 7.08 | 11.4 | 28.9 | mild over-epoch |
| 156M | 3e19 | 32.1B | 207 | 21.3 | 34.1 | 87.0 | deep over-epoch |
| 156M | 1e20 | 107B | 690 | 70.8 | 114 | 290 | extreme |
| 272M | 3e17 | 0.18B | 0.67 | 0.12 | 0.20 | 0.50 | **likely skip** — too undertrained |
| 272M | 1e18 | 0.61B | 2.25 | 0.41 | 0.65 | 1.66 | undertrained — minimal signal |
| 272M | 3e18 | 1.84B | 6.7 | 1.22 | 1.95 | 4.98 | param-bound |
| 272M | 1e19 | 6.12B | 22 | 4.06 | 6.51 | 16.6 | Chinchilla→mild over |
| 272M | 3e19 | 18.4B | 67 | 12.2 | 19.5 | 49.8 | mild→deep over |
| 272M | 1e20 | 61.2B | 224 | 40.6 | 65.1 | 166 | deep over-epoch |
| 998M | 3e17 | 0.05B | 0.05 | 0.03 | 0.05 | 0.14 | **skip** — model barely starts |
| 998M | 1e18 | 0.17B | 0.17 | 0.11 | 0.18 | 0.45 | **skip** — too undertrained |
| 998M | 3e18 | 0.50B | 0.50 | 0.33 | 0.53 | 1.36 | **likely skip** — undertrained |
| 998M | 1e19 | 1.67B | 1.67 | 1.11 | 1.77 | 4.52 | undertrained but usable |
| 998M | 3e19 | 5.01B | 5.0 | 3.32 | 5.32 | 13.6 | param-bound |
| 998M | 1e20 | 16.7B | 16.7 | 11.1 | 17.7 | 45.3 | Chinchilla→over |

## Recommended cell filter (D/N ≥ 1.5)

Drop 4 hopeless cells (998M × {3e17, 1e18, 3e18}, 272M × 3e17) →
**20 cells × 3 methods = 60 runs**

## Compute

- **Total FLOPs (full 72-run grid):** ~1.72e21
- **After D/N ≥ 1.5 filter (60 runs):** ~1.71e21 (the dropped cells are tiny)
- **Per method:** ~5.7e20 FLOPs
- **Per N column** (sum across 6 C values): 1.43e20

### Wall-clock estimate (v5p-32, ~5e14 FLOPs/sec aggregate)
- Serial: ~40 days
- 5-way parallel (one per N column + slack): ~9 days
- Fleet-wide via Iris (10+ ways): ~2–3 days

## Cross-method crossover targets

The cells that most directly test "does low/med ever beat high?":

| cell | low_q L | med_q L | high_q L | direct test |
|---|---|---|---|---|
| 68M @ 1e19 | medium ε | mod-deep ε | very deep ε | does low-q U-advantage offset high's better B coef? |
| 156M @ 1e19 | mild ε | mild-mod ε | deep ε | same, at bigger N |
| 156M @ 3e19 | deep ε | deep ε | very deep ε | who survives deep over-epoch? |
| 272M @ 3e19 | mod ε | mod ε | deep ε | bigger model, all methods over-epoching |
| 998M @ 1e20 | mild ε | mod ε | deep ε | biggest model + Chinchilla→over |

## Priorities for partial execution

If running the full 60-cell grid is too expensive, run in this order:

1. **First wave (highest info, 18 cells × 3 = 54 runs):** the over-epoch
   diagonal — `{156M, 272M, 998M}` × `{1e19, 3e19, 1e20}` × 3 methods
2. **Second wave (29 cells × 3 = 87 runs):** fill in the Chinchilla and
   below-Chinchilla cells for each N
3. **Third wave (anything else):** the borderline-undertrained cells

## Launch notes

- Use Iris for region-agnostic scheduling (`--allowed-regions us-central1
  us-east5-a europe-west4-a`)
- Parent at default priority, children at `--priority batch`
- WANDB_API_KEY, HF_TOKEN env vars per memory
- Each run produces standard eval_uncheatable_macro_loss; that's the metric
  used for the scaling-law fits
- Watch for stragglers — `iris job logs <id>` for any cell that hangs
