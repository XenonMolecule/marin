# PAUSED: 10 lambda=0.01 cells in europe-west4 (2026-08-04)

Stopped at user request 2026-08-04 ~19:50Z to free europe-west4 for higher-priority work.
User: "it is OKAY to not finish these right now, we have got most of the data we need from
0p01 runs (although someday we will probably bring these back up)."

NOT lost — each kept its rolling temp checkpoint and RESUMES from the step below.

| cell (method-expFM_natural-budget-dWIDTH-LN-BSIZE) | resume @ step |
|---|---|
| dclm_10k_mix_lambda0p01-expFM_natural-3e+19-d2432-L24-B8   | 22517 |
| high_quality_10k_mix_lambda0p01-expFM_natural-3e+19-d2432-L24-B8 | 23758 |
| dclm_10k_mix_lambda0p01-expFM_natural-3e+19-d1536-L16-B32  | 19254 |
| high_quality_10k_mix_lambda0p01-expFM_natural-3e+19-d1536-L16-B32 | 19275 |
| dclm_10k_mix_lambda0p01-expFM_natural-3e+19-d1024-L11-B64  | 19332 |
| high_quality_10k_mix_lambda0p01-expFM_natural-3e+19-d1024-L11-B64 | 18641 |
| high_quality_10k_mix_lambda0p01-expFM_natural-2e+19-d512-L6-B128 | 17591 |
| dclm_10k_mix_lambda0p01-expFM_natural-2e+19-d512-L6-B128   | 16882 |
| high_quality_10k_mix_lambda0p01-expFM_natural-3e+19-d512-L6-B256 | 10574 |
| dclm_10k_mix_lambda0p01-expFM_natural-3e+19-d512-L6-B256   |  7860 |

KEPT RUNNING (do not confuse with the above): dclm_10k_mix_lambda0p01 2e+19-d1024-L11-B32,
which was at 100% and left to finish.

## HARD DEADLINE for a free restore

Temps live at
`gs://marin-eu-west4/tmp/ttl=14d/checkpoints-temp/marin-eu-west4/checkpoints/isoflop-curation/<run>/checkpoints/step-N`
Bucket lifecycle reaps `ttl=14d`. Last writes were 2026-08-04 ~19:49Z, so **restore before
~2026-08-18** or these become full restarts from step 0.

## How to restore

    python -m experiments.scaling_law_sweeps.launch_10k_natural \
      --methods dclm_10k_mix_lambda0p01 high_quality_10k_mix_lambda0p01 \
      --max-budget 9e20 --allowed-regions europe-west4 \
      --child-priority batch --wave-size 8 --wave-delay 360 \
      --only <method-qualified cell names above>

* **`--allowed-regions europe-west4` is REQUIRED.** Temps are per-region; a child rescheduled
  into us-central1 would find no temp (or an older one) and silently restart from scratch.
* **Never pass `--run-suffix`** — it isolates checkpoints and discards the temp.
* Use METHOD-QUALIFIED `--only` patterns; a bare shape like `3e+19-d512-L6-B256` matches both
  corpora.
* Check first that no child is already live for these cells (`state IN (1,3)`); Iris retries on
  its own and a manual relaunch on top produces duplicate writers to one checkpoint path.


---

# ALSO PAUSED: 13 us-central1 cells with ETA > 8h (2026-08-04 ~23:10Z)

User: "anything over 8 hours we can remove, this has gone on for quite a while." Stopped in
us-central1 after the sweep had run ~24h. NOT lost -- rolling temps in
`gs://marin-us-central1/tmp/ttl=14d/checkpoints-temp/marin-us-central1/checkpoints/isoflop-curation/`
so each RESUMES from the step below.

| cell | resume @ step | of | % | ETA at pause |
|---|---|---|---|---|
| dclm_10k_mix_lambda0p01-expFM_natural-9e+20-d3584-L35-B128 | 6882 | 32936 | 21% | 13.5h |
| high_quality_10k_mix_lambda0p01-expFM_natural-2e+20-d512-L6-B1024 | 34944 | 61582 | 57% | 13.0h |
| dclm_10k_mix_lambda0p01-expFM_natural-2e+20-d512-L6-B1024 | 34911 | 61582 | 57% | 12.8h |
| high_quality_10k_mix_lambda0p01-expFM_natural-3e+20-d512-L6-B2048 | 29188 | 51319 | 57% | 12.7h |
| dclm_10k_mix_lambda0p01-expFM_natural-3e+20-d512-L6-B2048 | 29670 | 51319 | 58% | 12.4h |
| high_quality_10k_mix_lambda0p01-expFM_natural-3e+20-d2432-L24-B64 | 34764 | 62249 | 56% | 11.4h |
| dclm_10k_mix_lambda0p01-expFM_natural-3e+20-d2432-L24-B64 | 34737 | 62249 | 56% | 11.3h |
| high_quality_10k_mix_lambda0p01-expFM_natural-9e+19-d512-L6-B512 | 36754 | 61582 | 60% | 11.0h |
| dclm_10k_mix_lambda0p01-expFM_natural-9e+19-d512-L6-B512 | 37146 | 61582 | 60% | 10.8h |
| high_quality_10k_mix_lambda0p01-expFM_natural-9e+20-d2432-L24-B256 | 20162 | 46686 | 43% | 10.7h |
| high_quality_10k_mix_lambda0p01-expFM_natural-9e+20-d3584-L35-B128 | 15628 | 32936 | 47% | 9.0h |
| high_quality_10k_mix_lambda0p01-expFM_natural-3e+19-d512-L6-B256 | 18555 | 41055 | 45% | 9.0h |
| dclm_10k_mix_lambda0p01-expFM_natural-3e+19-d512-L6-B256 | 20179 | 41055 | 49% | 8.4h |

Restore is the same recipe as the eu-west4 block above, but pin **us-central1**:

    python -m experiments.scaling_law_sweeps.launch_10k_natural \
      --methods dclm_10k_mix_lambda0p01 high_quality_10k_mix_lambda0p01 \
      --max-budget 9e20 --allowed-regions us-central1 \
      --child-priority batch --wave-size 8 --wave-delay 360 \
      --only <method-qualified names above>

**ttl=14d applies here too: restore before ~2026-08-18 or these become restarts from step 0.**

## Resulting lambda=0.01 coverage

38 of 72 cells complete. The pauses are concentrated at the HIGH-compute end
(9e+19..9e+20, and the wide d2432/d3584 cells), so the lambda contrast is well covered at the
low/mid budgets and thin at the top -- the same shape the lambda=0.05 arms had before their
big cells landed. Read the lambda deltas with that in mind.


---

# ALSO PAUSED: 3 more cells, >=4h ETA (2026-08-05 ~07:00Z)

User tightened the cutoff: "Kill any greater than or equal to 4 hours. Just keep the sub 3 hour
ones running."

| cell | resume @ step | of | % |
|---|---|---|---|
| dclm_10k_mix_lambda0p01-expFM_natural-3e+19-d1024-L11-B64 | 28641 | 46669 | 61% |
| dclm_10k_mix_lambda0p01-expFM_natural-2e+19-d512-L6-B128 | 32417 | 49266 | 66% |
| high_quality_10k_mix_lambda0p01-expFM_natural-2e+19-d512-L6-B128 | 31471 | 49266 | 64% |

All three had ALREADY been stopped once in europe-west4 and came back on their own -- Iris
re-queued them into us-central1, where they resumed from ~18h-stale us-central1 temps instead
of their newer eu-west4 ones. **Two temps exist per cell, in different regions, at different
steps.** On restore, pick the region whose temp is furthest along; `--allowed-regions` decides
which one the child can see. Check BOTH:
  gs://marin-us-central1/tmp/ttl=14d/checkpoints-temp/marin-us-central1/checkpoints/isoflop-curation/<run>/checkpoints/
  gs://marin-eu-west4/tmp/ttl=14d/checkpoints-temp/marin-eu-west4/checkpoints/isoflop-curation/<run>/checkpoints/

Stopping a child does NOT prevent Iris from re-queueing it. If a paused cell must stay down,
re-check it a cycle later and stop it again.
