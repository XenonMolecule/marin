# Canonical fit state snapshot — 2026-05-25 ~01:05 PDT

> Written right before context compaction. Captures decisions and in-flight
> work that aren't yet in the older "v3" / "sharedbeta_clipmin" markdowns.

## Current canonical form: Option B (Sedova × exp decay)

**Promoted to canonical 2026-05-25.** Replaces the prior SPP/sharedbeta_clipmin
form. NO clip_to_min — bend-up rows kept and modeled by R_decay multiplier.

```
L = E + C·N^(-β) + B_m·N^(δ_m) · D_eff^(-α_m)
D_eff = U · R_D · (1 − exp(−ε/R_D)) · exp(−ε/R_decay)         ε = D/U
                 └──── Sedova saturation ────┘  └── bend-up ──┘
```

Shared: E, C, β, R_D, R_decay (5)
Per-method: B, δ, α (3 × 3 = 9)
Total: **14 free parameters**.

Most-recent fit (298-row gold, ~Mon May 24 22:39 PDT):
```
shared:  E≈1.87  C≈1370  β≈0.360  R_D≈1.52  R_decay≈67
RMSE ≈ 0.084 (NOTE: not directly comparable to SPP's 0.065 since SPP
              filtered with clip_to_min reducing the row count).
```

## Production figure / dashboard files (KEEP — don't overwrite without copy)

```
experiments/scaling_law_sweeps/
  plot_quality_winner_regions.py                                 (oldest)
  plot_quality_winner_regions__sqrt_weighted.py                  (sidecar)
  plot_quality_winner_regions__sharedbeta_clipmin.py             (prior canonical)
  plot_quality_winner_regions__PSS.py                            (sidecar)
  plot_quality_winner_regions__optionB.py                        (THE CANONICAL)
  plot_quality_winner_regions__optionC.py                        (sidecar — polynomial decay)
  plot_quality_winner_regions__optionB_trust.py                  (trust-overlay version)

scratch/plots/
  dashboard_quality_tiers.py                                     (oldest dashboard)
  dashboard_quality_tiers__sharedbeta_clipmin.py                 (SPP dashboard, kept)
  fit_sedova_with_damage.py                                      (fit utilities; HAS share_delta + share_alpha + d_eff_form flags)

scratch/plots/PLANNING_SANDBOX/
  test_W1_optionB_REAL.py                                        (test variant B)
  test_W1_optionC_REAL.py                                        (test variant C)
  bootstrap_trust_regions__optionB.py                            (main B=100 bootstrap)
  bootstrap_trust_regions__optionB_QUICK.py                      (B=5 quick-prototype version)
```

Don't rename / delete any of these — version history matters per user feedback.

## In-flight tasks (as of 01:05 PDT)

1. **Main B=100 bootstrap** (task `be2i1yp7q`):
   - Script: `scratch/plots/PLANNING_SANDBOX/bootstrap_trust_regions__optionB.py`
   - Started ~00:41 PDT, ~50% done at 01:03, expected complete **~01:25 PDT**
   - Output: `scratch/plots/bootstrap_results/optionB_bootstrap_B100.npz`
   - Streaming via Monitor task `bhrbl1mbq`

2. **Trust figure render** (just kicked off, task `btig5n7l2`):
   - Using B=5 quick data for visual prototyping
   - Output: `scratch/plots/quality_winner_regions/quality_winner_regions__OPTIONB_TRUST__W100_W500_W7925398__B5_QUICK.png`
   - Had a matplotlib `linestyles=(0, (5, 3))` bug — fixed to `linestyles="dashed"`

## Visual treatment for trust regions (DECIDED)

- **Cross-hatch overlay** (`////`) on cells where bootstrap consensus winner trust < threshold
- **Dashed black contour line** at trust = threshold (default 0.95)
- Legend entries: "95% bootstrap trust boundary" + "uncertain (<95% trust)"
- Bootstrap grid can have different resolution from figure grid — contourf handles independently

## Open improvements / TODOs

1. **Refactor bootstrap to save central fit params**, so the trust figure doesn't need to refit Option B at render time. ~5 sec render instead of 30 sec.

2. **Once B=100 lands**, regenerate trust figure pointing at `optionB_bootstrap_B100.npz` (not the QUICK version).

3. **Iterate on threshold** (80% / 90% / 95% / 99%) without re-running bootstrap — just swap `--trust-threshold N`.

4. **Update canonical fit notes markdown** to reflect Option B as canonical (the existing `sedova_sharedbeta_clipmin_fit_notes_2026_05_24.md` is now stale).

5. **Iris fan-out for B≥500** would require vendoring fit code out of scratch/ (gitignored) into experiments/scaling_law_sweeps/. Estimated 30 min infra work. Not yet worth it for B=100.

## Key historical decisions (from earlier today, may have been compressed)

- Per-method E is BAD — collapses med_quality consistently
- Per-method R_D is BAD — basin-unstable across data refreshes
- Per-method β is FINE if shared E
- Shared β with per-method (δ, α) (= SPP) was previous winner
- Then Option B (NO clip, with R_decay) became canonical because clip felt like "cheating"
- Option C (polynomial decay) tied with B; user preferred B
- Erlang single-param D_eff was too aggressive (exp blow-up)
- Tie-bias was hacky, removed
- 40M tokens is the current min-tokens floor for figures

## Data state

- Gold CSV: `scratch/exports/warc_scaling_streamlined.csv` (298 rows for 3 quality tiers as of latest refresh)
- Synced to: `scratch/plots/PLANNING_SANDBOX/gold/warc_scaling_streamlined__FROZEN.csv`
- Synthetic data has been retired (was at `synthetic_additions.csv.RETIRED-*`)
