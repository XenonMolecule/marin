# Data Curation IsoFLOP Scaling Law Analysis

## Overview

We train the same model architecture (Qwen3) at multiple compute budgets on
different data curation methods (DCLM, nemotron_org, nemotron_full, fineweb_edu)
and fit scaling laws to compare their compute-efficiency. The analysis produces
three key relationships per method:

1. **D\*(C)**: optimal token count as a function of compute budget
2. **L(D\*)**: loss as a function of optimal tokens (data quality signal)
3. **L(C)**: loss as a function of compute (the bottom line)

## Step 1: IsoFLOP Curves

For each compute budget C (e.g. 3e18, 9e18, 2e19 FLOPs), we train multiple
model sizes (varying hidden_dim, num_layers). Each run produces a final
evaluation loss (paloma macro bpb). Plotting loss vs tokens-trained at fixed
C gives a U-shaped curve: too few tokens (large model, undertrained) or too
many tokens (small model, overtrained) both hurt.

The minimum of each curve is **D\*(C)** -- the compute-optimal token count at
budget C, with associated **L\*(C)** (loss at the optimum) and **N\*(C)**
(optimal parameter count).

We find D\*(C) by fitting a quadratic in log-token space per budget (via
`marin.scaling_laws.isoflop_analysis.fit_scaling_laws`), then taking the
minimum.

## Step 2: Scaling Law Fits

### D\*(C): Token Scaling

The compute-optimal token count follows a power law:

```
D*(C) = A_d * C^alpha
```

where alpha controls how fast optimal tokens grow with compute. Fit via
log-log linear regression through the (C, D\*) points.

| Method       | alpha | A_d    | Interpretation                          |
|-------------|-------|--------|-----------------------------------------|
| dclm         | 0.585 | 0.025  | Tokens grow fast with compute           |
| nemotron_org | 0.466 | 0.338  | Tokens grow slower -- more params-heavy |

Higher alpha = the method benefits more from additional tokens at higher
compute (data-hungry). Lower alpha = the method allocates more compute to
parameters (params-hungry).

### L(D\*): Loss vs Optimal Tokens

The loss at the compute-optimal point follows:

```
L(D*) = A_t * D*^(-beta_t)
```

Fit via log-log linear regression through (D\*, L\*) points.

| Method       | beta_t | Interpretation                           |
|-------------|--------|------------------------------------------|
| dclm         | 0.096  | Each additional token is worth less       |
| nemotron_org | 0.111  | Each additional token is worth more       |

Higher beta_t = steeper loss improvement per token = higher data quality.
This is the **per-token data quality signal**: nemotron_org's curation
produces more valuable tokens.

### L(C): Loss vs Compute (the bottom line)

The loss at compute-optimal follows:

```
L(C) = A_c * C^(-beta_c)
```

**Consistency check**: since L(C) = L(D\*(C)) = A_t * (A_d * C^alpha)^(-beta_t),
we get:

```
beta_c = alpha * beta_t
```

This holds exactly in our fits (gap < 0.0001), confirming internal consistency.

| Method       | beta_c | Interpretation                          |
|-------------|--------|-----------------------------------------|
| dclm         | 0.056  | Loss drops faster per FLOP              |
| nemotron_org | 0.052  | Loss drops slower per FLOP              |

## Step 3: Cross-Method Comparison

### Token-space crossing

nemotron_org has higher beta_t (0.111 > 0.096), so its loss drops faster per
token. Starting from a higher loss, nemotron_org's curve eventually crosses
DCLM's -- at ~10^13 tokens in our projections. This means **nemotron_org
produces higher-quality data per token**.

### Compute-space non-crossing

Despite higher per-token quality, nemotron_org has lower beta_c (0.052 < 0.056).
At any compute budget, DCLM achieves lower loss. The gap widens with compute.
Why? DCLM's higher alpha (0.585 vs 0.466) means it trains on more tokens at
compute-optimal. Quantity (more tokens) beats quality (better tokens):

```
DCLM:    many tokens * modest per-token value = more loss reduction per FLOP
nemotron: fewer tokens * high per-token value = less loss reduction per FLOP
```

### Implication

If your bottleneck is **compute** (fixed FLOP budget, unlimited data): DCLM wins
at every scale.

If your bottleneck is **data** (fixed token budget, unlimited compute): nemotron_org
eventually catches up -- its higher data quality per token matters when you can't
just add more tokens.

## Generating Plots

```bash
# Download summaries
gcloud storage cp "gs://marin-us-central1/metadata/data_curation_isoflop_results/curation-*-v4.json" /tmp/summaries/

# Generate all plots (per-method + comparison with forecasts)
JAX_ENABLE_X64=1 uv run python experiments/scaling_law_sweeps/plot_curation_isoflop.py \
    --methods dclm nemotron_org nemotron_full fineweb_edu \
    --suffix v4 --results-prefix /tmp/summaries/ \
    --output-dir plots/curation_isoflop
```

Output structure:
```
plots/curation_isoflop/
  _comparison/
    compare_dstar_expA_natural.html      # D* vs loss + forecast
    compare_scaling_expA_natural.html    # Compute vs D* + forecast
    compare_frontier_expA_natural.html   # Compute vs loss + forecast (money plot)
    compare_*_expB_T20T.html            # Same for Experiment B
  dclm/
    isoflop_dclm_expA_natural.html      # Per-budget IsoFLOP curves
    scaling_dclm_expA_natural.html      # D*(C) power law fit
    fit_dclm_expA_natural.json          # Fit coefficients
    records_dclm_expA_natural.csv       # Raw data
  nemotron_org/
    ...
```

## Caveats

- **3 compute budgets** (3e18, 9e18, 2e19) so far. Power law fits through 3 points
  are directionally correct but sensitive to noise. Adding 3e19+ budgets (running
  overnight) will tighten the fits significantly.
- **Projection range**: forecasts extend to 1e14 tokens / 1e27 FLOPs. At 3+ orders
  of magnitude beyond observed data, treat as directional, not predictive.
- **Single eval metric** (paloma bpb). Different metrics (uncheatable_eval, downstream
  tasks) may show different scaling behavior.
- **Experiment A vs B**: ExpA (natural epoching) and ExpB (simulated T=20T) may give
  different scaling exponents. Compare within experiment, not across.
