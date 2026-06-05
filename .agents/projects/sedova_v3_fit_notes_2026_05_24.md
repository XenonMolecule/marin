# [SUPERSEDED] Sedova V3 Fit — historical notes

> **⚠ SUPERSEDED 2026-05-24 (evening).** This V3 fit (per-method R_D/C/β + dpm)
> was the best snapshot RMSE on a single data refresh but proved **basin-unstable**
> across data refreshes — R_D values bounced 114 → 14 → 42 → ... and predicted
> winner regions flipped. Replaced by the simpler **shared β + shared R_D +
> clip_to_min** form documented in `sedova_sharedbeta_clipmin_fit_notes_2026_05_24.md`.
> This file is kept as history of what was tried.

**Date:** 2026-05-24 (afternoon)
**Status:** SUPERSEDED — see `sedova_sharedbeta_clipmin_fit_notes_2026_05_24.md`
**Data:** `scratch/exports/warc_scaling_streamlined.csv` (gold only)

## The functional form

```
L(N, D, U) = E + C_m · N^(-β_m) + B_m · N^(δ_m) · D_eff_m^(-α_m)

D_eff_m   = U · R_D_m · (1 − exp(−ε / R_D_m))     ε = D/U
```

**Variables:**
- `N` = model parameters (non-embedding count from the warc-scaling experiments)
- `D` = total training tokens
- `U` = unique tokens (tokens-per-WARC × WARCs)
- `ε` = epochs (D/U)

**No damage term.** Gold rows past per-cell loss minimum are dropped (drop_post_min);
the fit doesn't model the over-epoching bend-up. The dashboard uses
`WINNER_MODE = best_L_at_or_below_C` (early-stop view) so post-min behavior is
implicitly the per-method minimum anyway.

## Parameter sharing

| param | role | scope |
|---|---|---|
| **E** | irreducible entropy floor | **shared** across methods |
| C_m | parameter-scaling coefficient | per-method |
| β_m | parameter-scaling exponent | per-method |
| B_m | data-scaling coefficient | per-method |
| δ_m | data-scaling N-coupling exponent | per-method |
| α_m | data-scaling D_eff exponent | per-method |
| R_D_m | saturation horizon (in epochs) | per-method |

Total free params: 1 shared (E) + 6 × 3 methods = **19**.

τ_dmg and γ_dmg are not used here (no damage term).

## Current fitted values

(From `scratch/plots/dashboard_quality_tiers.py` run on 289 rows → 275 after dpm.)

```
shared E = 2.507

method           R_D      C         β       B        δ       α
low_quality      114.60   7.81e3    0.471   192      0.186   0.458
med_quality      73.69    7.05e3    0.465   249      0.179   0.470
high_quality     1.57     3.06e4    0.551   2.89e3   0.118   0.538

Tokens-per-WARC (from data):
  low:   1.515e7
  med:   9.456e6
  high:  3.723e6
```

**Fit quality:**
```
R²:        low=0.994   med=0.994   high=0.980
aggregate RMSE: 0.089 (vs 0.098 for the all-shared-β baseline)
```

## Why these sharing choices

Picked from a 32-variant sweep (32_variant_fit_sweep_2026_05_24.md) plus a
follow-up 4-variant sweep on per-R_D / per-C combinations:

1. **Shared E** is decisive — per-method E collapsed at least one tier in
   every config with damage (3.6× median RMSE difference). The asymptote
   really is the same across quality tiers in this regime.
2. **Per-β** captures real per-tier scaling-with-N differences without the
   pathology that per-E introduces.
3. **No damage** — Chinchilla's L = E + C/N^β + B/D^β was insufficient (best
   RMSE 0.19) because it lacks Sedova's N^δ cross-coupling and R_D
   saturation. With Sedova's structure, the damage term is unnecessary if
   we drop_post_min, since the descent + plateau is what we then need to
   model and the form handles that cleanly.
4. **Per-R_D + per-C** improves RMSE 9% over baseline. R_D ordering
   `low > med > high` matches physics: methods with more unique tokens
   benefit from longer saturation horizons.

## Known issues / caveats

### high_quality R_D = 1.57 collapse

In the fit, high_quality's R_D collapsed to ~1.6, which means D_eff
saturates within ~5 epochs. This is *structurally* the form's way of
saying "high_quality is data-limited very quickly," and is partly
compensated by C_high blooming to 30.6k. Watch this as more high_q
data lands — if R_D stays collapsed across data refreshes, it's a real
signal; if it bounces around, it's a degenerate-basin artifact.

### C and β are coupled (sloppy parameters)

C·N^(-β) is a single 2-parameter term with a sloppy direction:
`log C − β·log N` is well-constrained over the observed N range, but
`log C` and `β` separately can wander far. Across our variants:

- Chinchilla A: 406
- V0 baseline (shared β,C): C = 4,818
- V1 (per-R_D only): C_shared = 661,029 ← extreme
- V3 (this fit): C per-method ranges 7,000–30,000

**Don't compare individual C across variants or to literature.** Trust
RMSE / predicted curves / R², not absolute C magnitude.

### Warcs-flatness in the bulk

With high R_D for low/med (~70–115), D_eff ≈ D for most (N, C) cells
in our grid — so changing the WARC slider doesn't move the heatmap
much except in the saturation corner. This is correct: without a damage
term, the only U-sensitivity is through D_eff, and D_eff is U-saturated
only at high ε.

If we want richer WARC-sensitivity in predictions, we'd need to
reintroduce a damage term (the "honest Sedova" W2 fit did this but was
fragile — it could collapse high_quality's fit when a single new gold
row shifted the basin).

### Bend-up region is not modeled

The fit was trained on descent-only data (dpm filter). It cannot
predict what happens past the per-cell minimum. If you ask "what L
would I see if I trained to D=10·D_min?", the model says "the same as
at D=D_min" (plateau) which is wrong — reality bends up.

For over-training predictions, switch to the **honest Sedova** mode
(`SHARING_MODE = "honest_sedova_perBC_sharedE_perG"` in
`PLANNING_SANDBOX/dashboard_synthetic.py`) which has damage. But that
fit is fragile and changes with each data refresh.

## File map

```
scratch/plots/
  dashboard_quality_tiers.py           ← production dashboard script
  fit_sedova_with_damage.py            ← fit_flex_sharing + predict_with_damage
  outputs/dashboard_quality_tiers.html ← rendered output

scratch/exports/
  warc_scaling_streamlined.csv         ← gold data source (refresh via export_csv.py)

scratch/plots/PLANNING_SANDBOX/
  dashboard_synthetic.py               ← sandbox version with synth-drag interface
  fit_sedova_with_damage.py            ← sandbox copy (kept in sync manually)
  outputs/FROZEN_dashboard_*.html      ← snapshots of prior fits

.agents/projects/
  32_variant_fit_sweep_2026_05_24.md   ← the 32-variant sweep that established
                                          shared-E, per-β as the structural priors
  sedova_v3_fit_notes_2026_05_24.md    ← THIS DOC
```

## How to refresh data

```bash
# Pull latest summaries from GCS
gcloud storage cp -r \
  'gs://marin-us-central1/metadata/data_curation_warc_scaling_results/*.json' \
  scratch/audit_summaries/

# Regenerate streamlined CSV (no --refresh needed; we just refreshed manually)
.venv/bin/python -c "
import sys
sys.path.insert(0, 'experiments/scaling_law_sweeps')
import export_csv as e
e.LOCAL_FM_DIR.mkdir(parents=True, exist_ok=True)
e.LOCAL_FM_LIMA_SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
sys.argv = ['export_csv', '--out-dir', 'scratch/exports']
e.main()
"

# Rebuild dashboard with fresh data
.venv/bin/python -u scratch/plots/dashboard_quality_tiers.py
```

(`export_csv.py --refresh` times out on `data_curation_fixed_model_results`
because of size; refresh just the warc-scaling prefix manually above.)

## How to verify the fit hasn't regressed

After a data refresh, expect:
- `shared E` in the 1.7 – 2.8 range (varies a bit with data)
- `R_D_low` and `R_D_med` in the 50–200 range
- `R_D_high` could be 1–100 (the collapsed and uncollapsed basins both occur)
- All R² ≥ 0.97; aggregate RMSE ≤ 0.10

If aggregate RMSE jumps above 0.12 or any R² drops below 0.95, look at the
plot — likely a basin-hopping landed in a degenerate config and a re-run with
n_restarts=200 or a different seed would recover.

## Don't re-run the 32-variant sweep unless

- The data shape changes materially (10+ new cells, new quality tier added)
- A new form variant is proposed (e.g., we want to test back the damage term
  with the larger gold pool)

Otherwise: trust this fit and iterate on figures, the dashboard, or
launching new training cells to fill in coverage gaps.
