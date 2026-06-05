# Quality-tier winner-region blend figure — V2 milestone (2026-05-25)

V2 is the canonical version. Replaces V1's continuous weighted-RGB blend
with a discrete **top-1/top-2 mix** at the 95% trust threshold: pure consensus
color where trust ≥ 95%, 50/50 of the top-1 and top-2 colors where trust < 95%.
Result: crisp confident regions + cleanly readable contention bands.

## Artifacts (DO NOT OVERWRITE)

- **Code snapshot**:
  `experiments/scaling_law_sweeps/plot_quality_winner_regions__optionB_trust_blend__V2_2026_05_25.py`
- **Figure (PNG/PDF)**:
  `scratch/plots/quality_winner_regions/quality_winner_regions__OPTIONB_TRUST_BLEND__V2_2026_05_25.{png,pdf}`
- **Bootstrap data**:
  `scratch/plots/bootstrap_results/optionB_bootstrap_B100.npz`
- **Fit cache**:
  `scratch/plots/fit_cache/optionB__d2be9deaf4535c26.pkl` (+ `.json` manifest)

## What's new in V2 vs V1

- `--blend-mode top2_mix` (new) — discrete 2-color mix per cell:
  - trust ≥ threshold → pure top-1 (consensus winner) shade
  - trust < threshold → 50/50 mix of top-1 and top-2 shades
- `--trust-threshold 0.95` — strict; reads as "this region is unambiguous"
- Everything else (smoothing σ=2, dotted winner boundary, solid min-tokens
  floor, panel-3 boundary cap at log10 C ≥ 22, panel-3 Chinchilla line)
  identical to V1.

V1 (weighted RGB blend) is preserved as `..._V1_2026_05_25.{py,png,pdf}`.

## Reproduce

```bash
uv run --with matplotlib --with numpy --with pandas --with scipy --with plotly python \
  experiments/scaling_law_sweeps/plot_quality_winner_regions__optionB_trust_blend__V2_2026_05_25.py \
  --warcs 100 500 7925398 \
  --log-n-ranges 7.8,10 7.8,10.699 7.8,13 \
  --log-c-ranges 17,23 17,24 17,30 \
  --min-tokens 4e7 \
  --bootstrap-npz scratch/plots/bootstrap_results/optionB_bootstrap_B100.npz \
  --trust-threshold 0.95 \
  --smooth-sigma 2.0 \
  --boundary-min-logc none none 22 \
  --chinchilla-panels 3 \
  --blend-mode top2_mix \
  --suffix V2_2026_05_25
```

Cache hit → ~2 s.

## Why V2 over V1

V1 weighted blend showed magnitude of disagreement (more washed-out color
= less trust). V2 top2_mix shows category of disagreement (which two methods
are competing) with binary visual states. After comparing T80/T90/T95, T95
landed because:
- Confident regions look fully saturated, "this is the answer"
- Uncertain regions read as a single deterministic blended color per pair
- No fuzzy gradients arguing for attention

## Fit reference

Same as V1 (cache-key `d2be9deaf4535c26`):

```
shared:        E=1.8104  R_D=1.47  R_decay=~67  C=1.16e+03  β=0.349
low_quality:   B=52.5    δ=0.185   α=0.378   R²=0.9939
med_quality:   B=67.4    δ=0.187   α=0.398   R²=0.9918
high_quality:  B=82.8    δ=0.198   α=0.425   R²=0.9830
aggregate RMSE = 0.0850
```
