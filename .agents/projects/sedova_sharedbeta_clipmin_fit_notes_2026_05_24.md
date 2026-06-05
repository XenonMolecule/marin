# Current Sedova Fit — shared β + shared R_D + clip_to_min

**Date:** 2026-05-24 (evening session, after day-long iteration)
**Status:** Live in `scratch/plots/dashboard_quality_tiers__sharedbeta_clipmin.py`
       and `experiments/scaling_law_sweeps/plot_quality_winner_regions__sharedbeta_clipmin.py`
**Data:** `scratch/exports/warc_scaling_streamlined.csv` (gold only, 297 rows for 3 quality tiers)

## TL;DR

The day's iterations converged on the **simplest fit that works**:

```
L(N, D, U) = E + C·N^(-β) + B_m·N^(δ_m) · D_eff^(-α_m)
D_eff      = U · R_D · (1 − exp(−ε/R_D))             ε = D/U
```

with E, C, R_D, β **all shared** across methods, and only (B, δ, α) per-method.
Gold data filtered via `clip_to_min` (rows past per-cell loss minimum get
their L replaced with the running min; the form's curves flatten through
the bend region).

## Current fitted values

```
shared: E = 2.020    R_D = 2.32     C = 719     β = 0.327
per-method:
  low_quality   B = 218   δ = 0.162   α = 0.436   R² = 0.9942
  med_quality   B = 279   δ = 0.163   α = 0.454   R² = 0.9925
  high_quality  B = 500   δ = 0.169   α = 0.498   R² = 0.9880

aggregate RMSE = 0.0751      n_rows = 297
```

Free parameters: 4 shared (E, R_D, C, β) + 3 per-method × 3 tiers = **13** total.

## How we got here — what we ruled out today

| form | RMSE | issue |
|---|---:|---|
| V3 (per-R_D + per-C + per-β + dpm) | 0.089 | Lowest RMSE on a snapshot, but R_D values bounced wildly across data refreshes (114/74/1.6 → 14/82/2 → 42/1/125 → ...). **Basin-unstable**. |
| W1 (per-β + shared R_D/C + dpm) | 0.094 | Stable across refreshes but per-method β values nearly identical (~0.42-0.43), and the tiny β differences created spurious "low > high asymptote" predictions at small N. |
| Chinchilla + damage (no R_D, no N^δ) | 0.19 | Form too simple; high_quality R² ~0.93. Sedova's N^δ + R_D pieces ARE doing real work. |
| Honest Sedova (per-method γ damage, no dpm) | 0.10 (when it lands) | Damage term + per-method γ is fundamentally fragile — one method's γ collapses with most data refreshes. |
| W1 + sqrt-inverse-epoch weights (no dpm) | 0.12 | Without a damage term, downweighting bend-up doesn't help — the form has no machinery to express bend-up. R_D collapses to ~2 either way. |
| **W1 + clip_to_min** (per-β, shared R_D) | 0.074 | Better than dpm; gold data coverage preserved; form flattens through clipped plateau via R_D ~ 2. |
| **shared β + shared R_D + clip_to_min** ← winner | **0.064** (snapshot) → 0.075 (fresh data) | Best fit. Stable across refreshes. Cleanest form. |

## Per-method (B, δ, α) — what varies

After locking E/R_D/C/β shared, the per-method differences live in:
- **B**: data-term coefficient. low=218, med=279, high=500 — high gets the largest B (more "extra penalty per token" at small data; offset by saturation).
- **δ**: cross-coupling exponent. All three at ~0.16-0.17 — barely different.
- **α**: data-scaling exponent. low=0.436, med=0.454, high=0.498 — high has the steepest data-side decay.

## Stability check (data refresh +4 rows)

Same fit configuration, re-run on 297 rows (was 293):

```
                  293 rows         297 rows
shared E          1.764            2.020
shared R_D        2.747            2.316
shared C          658              719
shared β          0.316            0.327     ← only 3% drift
```

All shared params drift <15% with new data. Per-method (B, δ, α) coordinated-shift
together. All R² ≥ 0.988 in both cases. **Massively more stable than V3.**

## Why each piece survived

- **clip_to_min** beat both `dpm` and `no-policy` because it preserves (N, D)
  coverage without forcing the form to fit an upturn it has no machinery for.
- **shared β** beat per-β because the per-method β values landed within 1% of
  each other anyway, and the tiny differences caused spurious asymptote orderings
  (low > high at moderate N) that aren't supported by data.
- **shared R_D** is required: per-method R_D consistently picks a degenerate
  basin where one method's R_D collapses to ~1-2 and another shoots to 100+,
  with predictions flipping between refreshes.
- **per-method (B, δ, α)** is the right amount of per-tier flexibility.
- **no damage**: with bend-up rows clipped, damage term is unnecessary AND
  unstable when added back.

## File map

```
scratch/plots/
  dashboard_quality_tiers.py                       ← prior W1+dpm (kept for history)
  dashboard_quality_tiers__sharedbeta_clipmin.py   ← THE WINNER (this doc)
  fit_sedova_with_damage.py                        ← fit_flex_sharing + predict_with_damage
  outputs/
    dashboard_quality_tiers.html                       ← prior W1+dpm dashboard
    dashboard_quality_tiers__sharedbeta_clipmin.html   ← current dashboard

experiments/scaling_law_sweeps/
  plot_quality_winner_regions.py                              ← prior W1+dpm figure (kept)
  plot_quality_winner_regions__sqrt_weighted.py               ← sqrt-weight sidecar (kept)
  plot_quality_winner_regions__sharedbeta_clipmin.py          ← THE WINNER figure script
scratch/plots/quality_winner_regions/
  quality_winner_regions__W100_W500_W8000000.{png,pdf}                       ← prior fig
  quality_winner_regions__SQRT_WEIGHTED__W100_W500_W8000000.{png,pdf}        ← sidecar
  quality_winner_regions__SHAREDBETA_CLIPMIN__W100_W500_W8000000.{png,pdf}   ← CURRENT

.agents/projects/
  32_variant_fit_sweep_2026_05_24.md             ← morning's sweep (establishes shared-E)
  sedova_v3_fit_notes_2026_05_24.md              ← afternoon's V3 fit (now superseded)
  sedova_sharedbeta_clipmin_fit_notes_2026_05_24.md  ← THIS DOC
```

## How to refresh data + refit

```bash
gcloud storage cp -r \
  'gs://marin-us-central1/metadata/data_curation_warc_scaling_results/*.json' \
  scratch/audit_summaries/

.venv/bin/python -c "
import sys
sys.path.insert(0, 'experiments/scaling_law_sweeps')
import export_csv as e
e.LOCAL_FM_DIR.mkdir(parents=True, exist_ok=True)
e.LOCAL_FM_LIMA_SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
sys.argv = ['export_csv', '--out-dir', 'scratch/exports']
e.main()
"

# Sync to the dashboard's CSV mirror
cp scratch/exports/warc_scaling_streamlined.csv \
   scratch/plots/PLANNING_SANDBOX/gold/warc_scaling_streamlined__FROZEN.csv

# Rebuild figure
uv run --with matplotlib --with numpy --with pandas --with scipy --with plotly \
  python experiments/scaling_law_sweeps/plot_quality_winner_regions__sharedbeta_clipmin.py

# Rebuild dashboard
.venv/bin/python -u scratch/plots/dashboard_quality_tiers__sharedbeta_clipmin.py
```

## Regression-detection checklist

After a refresh, expect (within ~15% drift):
- shared E in **1.7 – 2.5**
- shared R_D in **2 – 4**
- shared C in **500 – 1500**
- shared β in **0.30 – 0.35**
- All R² ≥ **0.98** on every method
- Aggregate RMSE ≤ **0.10**

If aggregate RMSE jumps above 0.13 or any R² drops below 0.95, basin-hopping
likely landed badly — re-run with n_restarts=200 or a different seed.

If shared params drift outside the expected ranges by a lot, that's a real
signal something changed in the data (probably new cells extended N or warcs
coverage in a way that re-anchors the fit).

## Don't re-run sweeps unless

- A genuinely new form variant is proposed (we've exhausted the simple
  axes: sharing of E/R_D/C/β/γ + damage on/off + 3 gold policies + epoch weights)
- A new quality tier is added (e.g., resiliparse_dedup as a 4th tier alongside
  low/med/high)
- The data coverage changes structurally (e.g., a new hidden_dim or new warcs
  subsample size lands)

Otherwise: trust this fit and iterate on figures + the dashboard.

## Closing note

This took a full day of fit-debugging because the form has multiple sloppy
parameters (E ↔ R_D ↔ γ; C ↔ β) and the data has very few over-epoching cells.
Two key tools we used heavily and should reuse:

1. **Side-by-side parallel sweeps** over sharing choices, with summary tables
   sorted by aggregate RMSE. Pattern in
   `scratch/plots/PLANNING_SANDBOX/test_*.py` files.
2. **Stability-across-refresh check**: fit twice on data N rows apart, see if
   the shared params drift sensibly or jump basins. The single-snapshot RMSE
   is misleading without this.
