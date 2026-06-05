# Quality-tier winner-region blend figure — V1 milestone (2026-05-25)

This is the first "I really like this" version of the Option B trust-blend figure.
Captured here so we can revert if later iterations regress.

## Artifacts (DO NOT OVERWRITE)

- **Code snapshot**:
  `experiments/scaling_law_sweeps/plot_quality_winner_regions__optionB_trust_blend__V1_2026_05_25.py`
- **Figure (PNG/PDF)**:
  `scratch/plots/quality_winner_regions/quality_winner_regions__OPTIONB_TRUST_BLEND__V1_2026_05_25.{png,pdf}`
- **Bootstrap data**:
  `scratch/plots/bootstrap_results/optionB_bootstrap_B100.npz`
  (B=100 bootstrap, N_RESTARTS=75, completed 2026-05-25 ~01:27)
- **Fit cache**:
  `scratch/plots/fit_cache/optionB__d2be9deaf4535c26.pkl` (+ `.json` manifest)

## What's in V1

Visual elements:
- **RGB blend** per cell — bootstrap winner fractions weighted-mix the
  three shade colors (pink LQ / yellow MQ / green HQ). Uncertain regions
  show as visible blends (peach for LQ/MQ, sage for MQ/HQ).
- **σ=2 Gaussian smoothing** on the bootstrap fractions before rendering —
  kills the discrete-vote pixelation from B=100.
- **Winner boundary line**: dotted, linewidth 1.8, color `#222222`.
- **Min-tokens floor (4e7)**: solid, linewidth 1.4, color `#555555`.
- **Boundary suppressed below log10 C = 22 on panel 3** so the unfocused
  low-compute region doesn't distract.
- **Chinchilla-optimal line on panel 3 only**: solid orange `#ff7f0e`,
  linewidth 1.8 — slope-2 line at log10 C = log10(120) + 2·log10 N.
- **No 95% dashed trust contour** (looked too busy after smoothing).
- **No darkening** of low-trust regions (looked muddy).
- **Empirical winner dots** on panels with available data (W=100, W=500).

Panel layout:
- W=100  → log10 N ∈ [7.8, 10.0],  log10 C ∈ [17, 23]
- W=500  → log10 N ∈ [7.8, 10.7],  log10 C ∈ [17, 24]
- W=7.9M → log10 N ∈ [7.8, 13.0],  log10 C ∈ [17, 30]

## Reproduce

```bash
uv run --with matplotlib --with numpy --with pandas --with scipy --with plotly python \
  experiments/scaling_law_sweeps/plot_quality_winner_regions__optionB_trust_blend__V1_2026_05_25.py \
  --warcs 100 500 7925398 \
  --log-n-ranges 7.8,10 7.8,10.699 7.8,13 \
  --log-c-ranges 17,23 17,24 17,30 \
  --min-tokens 4e7 \
  --bootstrap-npz scratch/plots/bootstrap_results/optionB_bootstrap_B100.npz \
  --trust-threshold 0.95 \
  --smooth-sigma 2.0 \
  --boundary-min-logc none none 22 \
  --chinchilla-panels 3 \
  --suffix V1_2026_05_25
```

Cache hit → ~2 s. Cache miss → ~1 min for the fit.

## Why this is the milestone

After iterating through cross-hatch, gradient darkening, uniform darkening,
and bare blend variants, this is the first version where:
- Pure-color regions read as confidently-one-method
- Uncertain regions read as "these two are competing" without ugly overlays
- The 3-panel story is legible without extra contour lines fighting for attention
- The Chinchilla anchor on the extrapolated W=7.9M panel gives readers a
  reference point they can ground their mental model against

## Fit reference

(From cache manifest at `scratch/plots/fit_cache/optionB__d2be9deaf4535c26.json`)

```
shared:        E=1.8104  R_D=1.47  R_decay=~67  C=1.16e+03  β=0.349
low_quality:   B=52.5    δ=0.185   α=0.378   R²=0.9939
med_quality:   B=67.4    δ=0.187   α=0.398   R²=0.9918
high_quality:  B=82.8    δ=0.198   α=0.425   R²=0.9830
aggregate RMSE = 0.0850
```
