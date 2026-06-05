# 32-Variant Scaling-Law Fit Sweep — REAL Gold Data

**Date:** 2026-05-24
**Script:** `scratch/plots/PLANNING_SANDBOX/test_32_variants_REAL.py`
**Data:** `scratch/plots/PLANNING_SANDBOX/gold/warc_scaling_streamlined__FROZEN.csv` (288 rows, 3 quality tiers, fresh GCS pull this date)
**Wall time:** ~13 minutes on 8-way parallel (M-series, 8 physical cores)

## Setup

**Fixed:** R_D shared, τ pegged at 1, damage_form=`log_quad`, n_restarts=100.
**Methods:** `low_quality`, `med_quality`, `high_quality`.

**Three groups (32 variants total):**
- **Group A (16):** damage ON, no dpm, no weights — full 4×2×2 over (β,C)×E×γ
- **Group B (8):** damage OFF, drop_post_min ON, no weights — full 4×2 over (β,C)×E
- **Group C (8):** damage ON + sharedγ + clipped epoch weights (`clip(epochs, 1, 10)`, normalize to mean=1), no dpm — 4×2 over (β,C)×E

Tag format: `<group>_β<S|P>_C<S|P>_E<S|P>_<γ-treatment>`
- S = shared across methods, P = per-method
- γ-treatment: `shared`, `per-m`, `noG_dpm` (no damage + dpm), `γS_wgt` (shared γ + weights)

## Headline result

**Winner overall (cheats by dropping bend-up data):**
```
B_βP_CS_ES_noG_dpm    RMSE=0.061   R²: 0.995 / 0.994 / 0.995
   per-β | shared C | shared E | no damage | drop_post_min ON
```

**Winner that keeps ALL data (honest, models the bend):**
```
A_βP_CP_ES_γper-m    RMSE=0.098   R²: 0.994 / 0.992 / 0.975
   per-β | per-C | shared E | per-method γ damage | no dpm
```

Both share the same backbone: per-method β, shared E. They differ on whether the
over-epoching data is modeled (damage) or excluded (dpm).

## Full sorted results

```
tag                  agg_RMSE  R2_low  R2_med  R2_high   R_D    β       C        γ_shared  E_low  E_med  E_high  flags
B_βP_CS_ES_noG_dpm   0.0605    0.9952  0.9936  0.9946    3.25   per    655.2    -         1.769  1.769  1.769
B_βS_CS_ES_noG_dpm   0.0837    0.9952  0.9793  0.9898    3.96   0.338  867.3    -         2.075  2.075  2.075
A_βP_CP_ES_γper-m    0.0980    0.9938  0.9920  0.9749    1.64   per    per      -         2.391  2.391  2.391
B_βP_CP_ES_noG_dpm   0.1027    0.9933  0.9655  0.9859   55.84   per    per      -         2.487  2.487  2.487
B_βS_CP_EP_noG_dpm   0.1061    0.9820  0.9837  0.9849  195.997  0.446  per      -         2.562  2.740  2.412
A_βP_CS_ES_γper-m    0.1101    0.9899  0.9921  0.9693    1.54   per   19450    -          2.485  2.485  2.485
A_βP_CS_ES_γshared   0.1122    0.9936  0.9693  0.9755    2.10   per    845.2   0.0224     1.918  1.918  1.918
B_βS_CP_ES_noG_dpm   0.1125    0.9859  0.9927  0.9712   34.09   0.445  per      -         2.584  2.584  2.584
C_βP_CS_ES_γS_wgt    0.1259    0.9887  0.9892  0.9582    1.64   per   2697     0.0250     1.835  1.835  1.835
A_βS_CP_ES_γper-m    0.1372    0.9839  0.9622  0.9654    1.54   0.524  per      -         2.530  2.530  2.530
C_βS_CS_ES_γS_wgt    0.1477    0.9882  0.9466  0.9584    0.98   0.495 12031    0.0184     2.256  2.256  2.256
A_βS_CS_ES_γshared   0.1538    0.9898  0.9334  0.9568    1.25   0.604 73330    0.0125     2.716  2.716  2.716
C_βS_CP_EP_γS_wgt    0.1600    0.9820  0.9726  0.9373   78.38   0.539  per     0.0456     2.532  3.021  2.063
A_βS_CS_ES_γper-m    0.1705    0.9566  0.9914  0.9390   73.78   0.391 1823     -          2.499  2.499  2.499
A_βS_CP_ES_γshared   0.1752    0.9741  0.9691  0.9279    0.81   0.630  per     0.0084     2.856  2.856  2.856
C_βS_CP_ES_γS_wgt    0.1950    0.9461  0.9112  0.9564    0.82   0.561  per     0.0170     2.596  2.596  2.596
A_βP_CP_ES_γshared   0.2306    0.8759  0.9808  0.9320    1.24   per    per     0.0124     2.821  2.821  2.821
A_βP_CP_EP_γper-m    0.3319    0.6158  0.9707  0.9731  123.36   per    per      -         3.462  2.860  1.502
A_βS_CS_EP_γper-m    0.4002    0.9884 -0.0869  0.9721    1.48   0.545 23669    -          2.695  3.616  2.285   ⚠ med collapse
C_βP_CS_EP_γS_wgt    0.4043    0.9914 -0.1173  0.9726    1.27   per   7149     0.0152     2.137  4.174  1.373   ⚠ med collapse
C_βP_CP_ES_γS_wgt    0.4517    0.8621 -0.1215  0.9479    0.70   per    per     0.0103     2.218  2.218  2.218   ⚠ med collapse
B_βP_CP_EP_noG_dpm   0.4758    0.9953  0.9909  0.1708   10.65   per    per      -         2.273  2.813  3.148   ⚠ high collapse
B_βP_CS_EP_noG_dpm   0.4763    0.9958  0.9934  0.1671   18.79   per   5678     -          2.304  2.315  3.083   ⚠ high collapse
A_βP_CS_EP_γshared   0.4771    0.9956  0.9920  0.1784    2.07   per   4611    0.0000      2.263  2.471  3.037   ⚠ high collapse
B_βS_CS_EP_noG_dpm   0.4775    0.9946  0.9936  0.1639  148.87   0.381 1696     -          2.290  2.322  2.887   ⚠ high collapse
A_βP_CP_EP_γshared   0.4795    0.9947  0.9796  0.1772  199.89   per    per     0.0000     2.182  3.046  2.571   ⚠ high collapse
A_βS_CP_EP_γper-m    0.4806    0.9906  0.9714  0.1813  176.06   0.498  per      -         2.501  3.130  3.170   ⚠ high collapse
A_βP_CS_EP_γper-m    0.4820    0.9846  0.9824  0.1766   28.41   per    155.2    -         1.908  2.102  2.351   ⚠ high collapse
A_βS_CS_EP_γshared   0.5030    0.0535  0.9898  0.9696    1.08   0.494 10415    0.0095     3.294  2.598  2.254   ⚠ low collapse
A_βS_CP_EP_γshared   0.5309    0.4927  0.0195  0.9396    3.66   0.567  per     0.0000     3.821  4.178  2.240   ⚠ multi-collapse
C_βS_CS_EP_γS_wgt    0.5444    0.9933  0.9901 -0.0688    2.41   0.531 24181    0.0136     2.323  2.434  2.734   ⚠ high collapse
C_βP_CP_EP_γS_wgt    0.6612    0.9938 -0.0957 -0.0378    2.60   per    per     0.0043     1.602  3.882  2.423   ⚠ multi-collapse
```

## Clean fits (R² ≥ 0.95 on all three methods, no parameter bounds hit) — 10 variants

```
B_βP_CS_ES_noG_dpm   0.0605   per-β | shared C | shared E | no damage | dpm    ← OVERALL WINNER
B_βS_CS_ES_noG_dpm   0.0837   shared β | shared C | shared E | no damage | dpm
A_βP_CP_ES_γper-m    0.0980   per-β | per-C | shared E | per-m γ | damage      ← HONEST WINNER (no dpm)
B_βP_CP_ES_noG_dpm   0.1027   per-β | per-C | shared E | no damage | dpm
B_βS_CP_EP_noG_dpm   0.1061   shared β | per-C | per-E | no damage | dpm       ← only per-E survivor
A_βP_CS_ES_γper-m    0.1101   per-β | shared C | shared E | per-m γ | damage
A_βP_CS_ES_γshared   0.1122   per-β | shared C | shared E | shared γ | damage
B_βS_CP_ES_noG_dpm   0.1125   shared β | per-C | shared E | no damage | dpm
C_βP_CS_ES_γS_wgt    0.1259   per-β | shared C | shared E | shared γ + weights ← top weighted
A_βS_CP_ES_γper-m    0.1372   shared β | per-C | shared E | per-m γ | damage
```

## Axis-by-axis trends (lower median = better)

| axis | option | median RMSE | best | worst | count | comment |
|---|---|---:|---:|---:|---:|---|
| **E sharing** | shared | **0.132** | 0.061 | 0.452 | 16 | per-E drove every multi-method collapse |
|  | per-method | 0.477 | 0.106 | 0.661 | 16 |  |
| **dpm + no damage** | yes (Group B) | **0.109** | 0.061 | 0.478 | 8 | most parsimonious; ties for best |
|  | no | 0.281 | 0.098 | 0.661 | 24 |  |
| **β sharing** | shared | 0.173 | 0.084 | 0.544 | 16 | wide variance with per-β |
|  | per-method | 0.368 | **0.061** | 0.661 | 16 | per-β has the absolute best AND many worst |
| **γ treatment** | off (no damage) | **0.109** | 0.061 | 0.478 | 8 |  |
|  | per-method | 0.251 | 0.098 | 0.482 | 8 | better median than shared γ |
|  | shared | 0.354 | 0.112 | 0.531 | 8 |  |
|  | shared + weights | 0.300 | 0.126 | 0.661 | 8 | weighting hurts |
| **C sharing** | shared | 0.285 | 0.061 | 0.544 | 16 | basically equal to per-C |
|  | per-method | 0.213 | 0.098 | 0.661 | 16 |  |
| **epoch weights** | OFF | **0.203** | 0.061 | 0.531 | 24 | weighted is uniformly worse |
|  | ON (Group C) | 0.300 | 0.126 | 0.661 | 8 |  |

## Three takeaways

1. **Shared E is the single most important constraint** (3.6× median RMSE difference).
   Every per-E configuration with damage collapsed at least one tier. The "asymptote"
   is essentially the same across quality tiers in this regime — forcing per-E creates
   degenerate basins between E and the damage/saturation terms.

2. **Per-β + shared E is the right amount of per-tier flexibility.**
   Per-β + shared E gets us both the absolute best (with dpm) and the honest best
   (without dpm). β captures real per-tier scaling-with-N differences without the
   pathology that per-E introduces.

3. **Sedova-style epoch weighting failed cleanly on this data.**
   Weighted variants average 1.5× worse than unweighted. Even with the clipped (1–10)
   normalized weighting, the upweighting of rare bend-up points pulls the fit toward
   them at the expense of the descent. Don't use it on this dataset.

## Don't re-run unless

- More gold data lands (especially over-epoching cells at small N, high C)
- We change the form materially (e.g., drop R_D entirely → Chinchilla base; add new
  damage term shapes; share δ or α across methods)
- We add a fourth quality tier (resiliparse_dedup, dclm, fineweb, llm_curated_bos_fixed)

## Reproducing

```bash
.venv/bin/python -u scratch/plots/PLANNING_SANDBOX/test_32_variants_REAL.py
```

Takes ~13 min on M-series 8-core. Output streams directly; the final summary
table reproduces the tables above given the same gold CSV.
