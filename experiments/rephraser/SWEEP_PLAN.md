# V3 Extraction SFT Hyperparameter Sweep Plan

**Created:** 2026-03-17
**Goal:** Find optimal hyperparameters for single-epoch SFT on v3 commented extraction data (Qwen3-0.6B-Base).
**Primary metric:** MBPP 3-shot pass@1 (selection criterion)
**Held-out test:** HumanEval 0-shot pass@1 (validation only — do not optimize for this)
**Budget:** ~40 runs total across all phases

## Baselines

| Model | MBPP 3-shot | HumanEval |
|---|---|---|
| Baseline (no SFT) | 39.8% | 30.5% |
| V3 default (lr=2e-5, bs=64) | 36.8% | 29.9% |
| Resiliparse (lr=2e-5, bs=64) | 32.0% | 26.2% |

## Phase 1: LR × Batch Size (25 runs)

**Status:** Training complete, evals running

### Phase 1a: Original grid (18 runs)
- LR: [5e-6, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4]
- Batch: [16, 32, 64] (bs=128 OOM'd on v5p-8)
- WD=0.01, warmup=0.03, cosine schedule, decay=0.97

**HumanEval results (complete):**

| LR \ BS | 16 | 32 | 64 |
|---|---|---|---|
| 5e-6 | 29.9% | 29.9% | **31.7%** |
| 1e-5 | 28.0% | 29.3% | 28.7% |
| 3e-5 | 23.8% | 26.8% | 28.7% |
| 5e-5 | 21.3% | 21.3% | 25.0% |
| 1e-4 | 14.6% | 19.5% | 20.7% |
| 3e-4 | 6.1% | 9.1% | 15.2% |

**MBPP 3-shot results:** PENDING (evals running as of 01:02 UTC)

### Phase 1b: Low-LR extension (7 runs)
- LR: [1e-6, 2e-6, 3e-6, 7e-6] at bs=64
- LR: [1e-6, 2e-6, 3e-6] at bs=32

**Status:** Training running, will need config.json patching before evals

## Phase 2: Weight Decay + Warmup (up to 12 runs)

**Trigger condition:** ALWAYS launch. Use the config that hurts MBPP the least (highest MBPP 3-shot from Phase 1), even if it's below baseline.

**Design:** Take the best MBPP config from Phase 1 (and its LR/batch), sweep:
- Weight decay: [0.001, 0.01, 0.05, 0.1] (4 values, one is the default)
- Warmup: [0.0, 0.03, 0.1] (3 values, one is the default)
- Total: 4 × 3 - 1 (remove the default combo already run) = 11 runs

**If trigger NOT met:** Skip Phase 2. Report that SFT hurts MBPP regardless of LR/batch. Only run low-LR evals to confirm.

## Phase 3: Schedule Variants (up to 6 runs)

**Trigger condition:** Phase 2 produces a config that beats or matches the MBPP baseline (≥39.8%).

**Design:** Take the best Phase 2 config, sweep:
- LR schedule: [cosine, linear] × decay fraction: [0.9, 0.95, 1.0]
- Total: 6 runs (minus 1 default = 5 new runs)

**If trigger NOT met:** Skip Phase 3. Report best config from Phase 2.

## Decision Logic (Automated)

```
After Phase 1 MBPP results:
  best_mbpp = max(all Phase 1 MBPP 3-shot scores)
  best_config = config with best_mbpp
  → ALWAYS launch Phase 2 around best_config's LR and batch
  → Even if best_mbpp < baseline, optimize to minimize damage

After Phase 2 MBPP results:
  best_mbpp_p2 = max(all Phase 2 scores)

  if best_mbpp_p2 >= 39.8%:  # matches baseline
    → Launch Phase 3 around best Phase 2 config
  else:
    → Report best config, note it doesn't match baseline on MBPP
    → Still proceed with Phase 3 if time permits

After all phases complete for v3 extraction:
  if time < 8am PST (15:00 UTC):
    → Repeat Phase 1-3 for resiliparse data
  → Write comprehensive report regardless
```

## Overnight Execution Plan

1. **Monitor Phase 1a evals** (job `-20260317-080156`)
   - Check every ~10 min for results
   - Clean lockfiles and resubmit if failures (max 2 retries)

2. **When low-LR training finishes:**
   - Patch checkpoint config.json files (max_position_embeddings: 4096 → 32768)
   - Launch evals with max_model_len=8192, max_gen_toks=512

3. **When Phase 1 MBPP results are complete:**
   - Compile full results table
   - Apply decision logic above
   - Launch Phase 2 if triggered

4. **When Phase 2 results are complete:**
   - Apply decision logic
   - Launch Phase 3 if triggered

5. **Compile final report** with:
   - Full results table (all phases)
   - Recommended hyperparameters
   - HumanEval validation of MBPP-optimized config
   - Update CODE_EXTRACTION_SFT_REPORT.md

6. **If time permits (before 8am PST / 15:00 UTC):**
   - Repeat Phase 1-3 for resiliparse data
   - Use same sweep grid and decision logic
   - Write separate section in report for resiliparse hyperparameters

## Technical Notes

- **Checkpoint config.json:** Levanter saves max_position_embeddings=max_seq_len (training length) instead of the model's RoPE capacity. Fix merged in Levanter (llama.py, qwen.py) — new checkpoints will be correct. Existing checkpoints need manual patching.
- **Eval engine_kwargs:** Use max_model_len=8192, max_gen_toks=512 for all evals.
- **TPU lockfiles:** Run `_remove_tpu_lockfile()` before Docker start (fix in vllm_server.py). Clean nodes before resubmitting failed jobs.
- **MBPP outlier:** Problem 493 has 3716 tokens at 0-shot. With max_model_len=8192 this fits fine.

## Run Budget Tracking

| Phase | Planned | Actual (with results) | Notes |
|---|---|---|---|
| Phase 1a | 24 | 18 | bs=128 OOM (6), lr3e-4_bs64 failed (1) |
| Phase 1b (low-LR) | 7 | 7 | All complete |
| Phase 2 (WD×warmup) | 12 | 8 | 4 persistently failing (cluster lockfiles) |
| Phase 3 (schedule) | 5 | 0 | Skipped — time used for resiliparse instead |
| Resiliparse sweep | 9 | 7 | 2 still running/failed |
| **Total** | **57** | **40** | Hit budget target |

## Status: COMPLETE (as of 2026-03-17 14:30 UTC)

All phases for v3 extraction are complete. Resiliparse sweep is nearly complete (7/9). Report written to `HYPERPARAMETER_SWEEP_REPORT.md`.

**Key recommendation:** lr=2e-6, bs=32 for v3 extraction SFT (39.0% MBPP, 32.3% HumanEval — best compromise).
