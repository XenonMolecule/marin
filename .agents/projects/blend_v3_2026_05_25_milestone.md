# Quality-tier winner-region blend figure — V3 milestone (2026-05-25)

V3 is V2 + paper-ready polish. "Almost perfection" per user before they
attempted a 1000-bootstrap rerun on Zephyr.

## Artifacts (DO NOT OVERWRITE)

- **Code snapshot**:
  `experiments/scaling_law_sweeps/plot_quality_winner_regions__optionB_trust_blend__V3_2026_05_25.py`
- **Figure (PNG/PDF)**:
  `scratch/plots/quality_winner_regions/quality_winner_regions__OPTIONB_TRUST_BLEND__V3_2026_05_25.{png,pdf}`
- **Bootstrap data** (the one V3 was rendered against):
  `scratch/plots/bootstrap_results/optionB_bootstrap_B100.npz` (B=100 local)
- **Fit cache**:
  `scratch/plots/fit_cache/optionB__d2be9deaf4535c26.pkl` (+ `.json` manifest)
- **Data snapshots** (rows used; kept stable as new rows accumulate):
  - `scratch/plots/data_snapshots/v3_central_fit_data__299rows_2aabd6cd.csv` —
    299 rows, sha256 `2aabd6cd…`, drives the **central fit** (boundary lines).
  - `scratch/plots/data_snapshots/v3_bootstrap_data__298rows_3f38041e.csv` —
    298 rows, sha256 `3f38041e…`, drives the **B=100 bootstrap** (trust blend).
  - The two differ by one row (a fresh result landed between the bootstrap
    capture and the central refit). V4 (zephyr) eliminates this skew by
    embedding the central fit in the bootstrap npz.
- **Comprehensive fit params + R²**: `.agents/projects/blend_v3_fit_params.json`

## What's new in V3 vs V2

Paper-polish layer on top of V2's top2_mix:
- **Thicker lines**: winner-boundary (2.6 px dotted), min-tokens floor
  (2.2 px solid), Chinchilla-optimal (2.6 px solid orange).
- **Legend reordering**: `LQ • LQ/MQ • MQ • MQ/HQ • HQ • Transition •
  40M Token Floor • Chinchilla-Optimal` (uncertainty swatches now sit
  between their parent colors for intuitive reading).
- **Stripped "(ours)"** from method labels.
- **Renamed**: `winner boundary` → `Transition`; `min-tokens floor` →
  `40M Token Floor`.
- **Panel titles**: `100 WARCs`, `500 WARCs`, `Common Crawl (~8M)`.
- **Y-axis**: label is just `FLOPs`; ticks formatted as `1e17, 1e18, ...`;
  adaptive tick step (every-decade ≤6-decade range, every-other-decade
  otherwise) so panel 3 (13 decades) doesn't crowd.
- **X-axis label**: `Parameters` (no `(N)`).
- **Per-panel y-axis cutoffs**: `1e17–1e21`, `1e17–1e22`, `1e17–1e30`
  for the three panels respectively, focusing each panel on its
  data-relevant compute range.
- **Boundary mask on panel 3**: hide winner-boundary below log₁₀ C = 22.5
  (the extrapolated low-compute artifact).
- **Chinchilla-optimal line on panel 3** only.
- **Figure dims**: `figsize=(6.5·n_panels, 7.0)` — between V1/V2's 8.5
  (too tall) and a tested 6.0 (too stout).
- **Tight paper margins**: `tight_layout(rect=(0, 0.07, 1, 1), pad=0.2)`
  and `savefig(..., pad_inches=0.02)`.
- **Single-row legend** (`ncol=len(handles)`, bottom-centered).

## Reproduce

```bash
uv run --with matplotlib --with numpy --with pandas --with scipy --with plotly python \
  experiments/scaling_law_sweeps/plot_quality_winner_regions__optionB_trust_blend__V3_2026_05_25.py \
  --warcs 100 500 7925398 \
  --log-n-ranges 7.8,10 7.8,10.699 7.8,13 \
  --log-c-ranges 17,21 17,22 17,30 \
  --min-tokens 4e7 \
  --bootstrap-npz scratch/plots/bootstrap_results/optionB_bootstrap_B100.npz \
  --trust-threshold 0.95 \
  --smooth-sigma 2.0 \
  --boundary-min-logc none none 22.5 \
  --chinchilla-panels 3 \
  --suffix V3_2026_05_25
```

Cache hit → ~2 s.

## Fit reference

Same as V2 (cache-key `d2be9deaf4535c26`):

```
shared:        E=1.8104  R_D=1.47  R_decay=~67  C=1.16e+03  β=0.349
low_quality:   B=52.5    δ=0.185   α=0.378   R²=0.9939
med_quality:   B=67.4    δ=0.187   α=0.398   R²=0.9918
high_quality:  B=82.8    δ=0.198   α=0.425   R²=0.9830
aggregate RMSE = 0.0850
```

## What might change next (V4 candidate)

V4 will likely incorporate the 1000-bootstrap Zephyr run at
`scratch/plots/bootstrap_results/optionB_bootstrap_zephyr.npz`. The zephyr
npz already carries the central fit (`central_params_json`,
`central_tpw_json`) so it can render without a local refit. Different
data → different cache key → fresh figure once we point the plot script at it.
