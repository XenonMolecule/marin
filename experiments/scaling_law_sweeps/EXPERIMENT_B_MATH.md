# Experiment B: Simulated Epoching

## Setup

We sample 3,000 of ~7.9M Common Crawl WARCs and run each curation method on them:

- **D_obs**: tokens in our sample (e.g. 2.66B for DCLM)
- **s** = total_warcs / sampled_warcs ≈ 2,642
- **D_proj** = D_obs × s (projected full-CC token count, e.g. ~7.0T)

## Experiment A (baseline)

Train on the full D_obs with no slicing. No projection to full CC.

## Experiment B (simulated full-scale)

Simulate training on D_proj tokens at target compute T_target by matching the **epoch count** of the full-scale regime.

At full scale, epoch count = `T_target / D_proj`. We match this by training T_exp tokens on a slice:

```
slice = T_exp × D_proj / T_target
```

Verify: `T_exp / slice = T_target / D_proj` ✓

**Ceiling**: slice can't exceed D_obs, so:

```
T_exp ≤ T_target / s
```

Candidates above this ceiling are excluded from Experiment B.

## Example (DCLM, T_target = 20T, budget 3e18, d512 model)

```
slice  = 4.3B × 7.0T / 20T ≈ 1.5B unique tokens
epochs = 4.3B / 1.5B ≈ 2.87  (matches 20T / 7.0T ≈ 2.86 ✓)
```
