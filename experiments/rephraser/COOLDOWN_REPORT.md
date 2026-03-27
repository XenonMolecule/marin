# Rephraser Cooldown Experiment Results

**Last updated**: 2026-03-04

**How to update this report**: See `.agents/docs/rephraser-cooldown-report-guide.md`

## Setup

- **Model**: Custom 1.385B model (Qwen3 architecture, trained from scratch in Marin)
  - Architecture: hidden=1792, layers=18, heads=14, kv_heads=14, intermediate=7168
  - Tokenizer: Llama3 (128,256 vocab, `meta-llama/Meta-Llama-3.1-8B`)
  - Source run: `exp2166-scaling-ladder-nemotron-validation-optimal-1e+20`
  - Training schedule: WSD (10% warmup, 70% stable, 20% decay)
- **Pre-cooldown checkpoint**: step 35,000 (right before cooldown begins)
- **Full cooldown budget**: 9,759 steps = ~2.56B tokens (BATCH_SIZE=64, SEQ_LEN=4096)
- **Short cooldown budget**: 5,000 steps = ~1.31B tokens
- **Optimizer**: CautiousConfig, linear LR decay from 0.001473 to 0, no warmup

## Baselines

| Checkpoint | Description | GCS Path |
|---|---|---|
| step-35000 | Pre-cooldown (before any cooldown training) | `gs://marin-us-central1/exp2166-.../checkpoints/step-35000` |
| step-44758 | Post-cooldown (original nemotron-only cooldown) | `gs://marin-us-central1/exp2166-.../hf/step-44758` |

---

## choice_prob_norm Comparison (Primary Metric)

This is the primary metric for comparing all conditions. `choice_prob_norm` is available for all multiple-choice tasks except `agieval_lsat_ar` (uses `acc_norm` instead) and `lambada_openai` (uses `acc` instead).

### Full Cooldown — choice_prob_norm (9,759 steps, ~2.56B tokens)

| Task | Pre-cooldown (35k) | Nemotron-only (44k) | Rephraser v2 (25W) | DCLM 300M | DCLM Filtered | Rephraser 150W |
|---|---|---|---|---|---|---|
| arc_easy | 0.3130 | 0.3270 | 0.3284 | 0.3291 | **0.3295** | 0.3264 |
| arc_challenge | 0.2581 | **0.2646** | 0.2635 | 0.2631 | 0.2639 | 0.2630 |
| boolq | 0.5079 | **0.5149** | 0.4923 | 0.4987 | 0.5021 | 0.5131 |
| commonsense_qa | **0.2046** | 0.2001 | 0.1986 | 0.1996 | 0.2000 | 0.2009 |
| copa | 0.5157 | 0.5219 | 0.5207 | 0.5225 | 0.5221 | **0.5231** |
| hellaswag_0shot | 0.2720 | 0.2789 | 0.2788 | **0.2790** | 0.2789 | 0.2780 |
| hellaswag_10shot | 0.2717 | 0.2787 | 0.2787 | **0.2788** | **0.2788** | 0.2779 |
| openbookqa | 0.2710 | 0.2795 | 0.2790 | **0.2808** | 0.2782 | 0.2780 |
| piqa | 0.5137 | 0.5175 | 0.5178 | 0.5180 | **0.5181** | 0.5172 |
| winogrande | 0.5014 | 0.5018 | **0.5020** | 0.5018 | **0.5020** | 0.5017 |
| wsc273 | 0.5030 | 0.5040 | 0.5041 | 0.5042 | 0.5039 | **0.5043** |

Tasks without choice_prob_norm (shown with their native metric):

| Task | Metric | Pre-cooldown (35k) | Nemotron-only (44k) | Rephraser v2 (25W) | DCLM 300M | DCLM Filtered | Rephraser 150W |
|---|---|---|---|---|---|---|---|
| agieval_lsat_ar | acc_norm | 0.2391 | **0.2478** | 0.2217 | 0.2348 | 0.2348 | 0.2435 |
| lambada_openai | acc | 0.3695 | 0.4687 | 0.4786 | **0.4826** | 0.4788 | 0.4661 |

**Source**: Pre-cooldown and Nemotron-only from GCS `results.json`. Rephraser v2 and DCLM 300M from WandB (in-training eval). DCLM Filtered from standalone eval `lmeval_debug_hf_step-9758-c460fa`. Rephraser 150W from WandB (in-training eval on v5p-8).

### Short Cooldown — choice_prob_norm (5,000 steps, ~1.31B tokens)

| Task | Pre-cooldown (35k) | Short Rephraser 150W | Short DCLM | Short Nemotron-only |
|---|---|---|---|---|
| arc_easy | 0.3130 | 0.3211 | 0.3251 | **0.3273** |
| arc_challenge | 0.2581 | 0.2600 | 0.2626 | **0.2636** |
| boolq | 0.5079 | 0.4936 | 0.4988 | **0.5233** |
| commonsense_qa | **0.2046** | 0.2007 | 0.2008 | 0.2012 |
| copa | 0.5157 | 0.5191 | 0.5228 | **0.5232** |
| hellaswag_0shot | 0.2720 | 0.2757 | 0.2774 | **0.2781** |
| hellaswag_10shot | 0.2717 | 0.2755 | 0.2772 | **0.2781** |
| openbookqa | 0.2710 | 0.2761 | 0.2773 | **0.2784** |
| piqa | 0.5137 | 0.5164 | 0.5170 | **0.5177** |
| winogrande | 0.5014 | 0.5017 | 0.5017 | **0.5018** |
| wsc273 | 0.5030 | 0.5035 | **0.5040** | **0.5040** |

Tasks without choice_prob_norm (shown with their native metric):

| Task | Metric | Pre-cooldown (35k) | Short Rephraser 150W | Short DCLM | Short Nemotron-only |
|---|---|---|---|---|---|
| agieval_lsat_ar | acc_norm | **0.2391** | 0.2130 | 0.1957 | 0.2130 |
| lambada_openai | acc | 0.3695 | 0.4401 | **0.4712** | 0.4590 |

**Source**: Short rephraser v2 from standalone eval `lmeval_debug_hf_step-4999-8b06e2` on us-central1. Short DCLM and short nemotron-only from WandB (in-training eval).

---

## acc / acc_norm Comparison (Secondary)

For reference, here are the acc and acc_norm metrics. Note: `acc_norm` is NOT the same as `choice_prob_norm` — `acc_norm` is length-normalized accuracy while `choice_prob_norm` normalizes by choice probability.

### Full Cooldown — acc / acc_norm (9,759 steps, ~2.56B tokens)

| Task | Metric | Pre-cooldown (35k) | Nemotron-only (44k) | Rephraser v2 (25W) | DCLM 300M | DCLM Filtered | Rephraser 150W |
|---|---|---|---|---|---|---|---|
| agieval_lsat_ar | acc_norm | 0.2391 | 0.2478 | 0.2217 | 0.2348 | 0.2348 | 0.2435 |
| arc_easy | acc_norm | 0.5560 | 0.6199 | 0.6263 | 0.6271 | 0.6271 | 0.6242 |
| arc_challenge | acc_norm | 0.2654 | 0.3208 | 0.3268 | 0.3174 | 0.3276 | 0.3080 |
| boolq | acc | 0.4991 | 0.5150 | 0.4245 | 0.4489 | 0.4593 | 0.4988 |
| commonsense_qa | acc | 0.2039 | 0.2064 | 0.1941 | 0.1925 | 0.2015 | 0.2056 |
| copa | acc | 0.6500 | 0.7100 | 0.6900 | 0.7200 | 0.7300 | 0.7300 |
| hellaswag_0shot | acc_norm | 0.4321 | 0.5121 | 0.5100 | 0.5133 | 0.5139 | 0.5018 |
| hellaswag_10shot | acc_norm | 0.4332 | 0.5088 | 0.5125 | 0.5149 | 0.5132 | 0.5037 |
| lambada_openai | acc | 0.3695 | 0.4687 | 0.4786 | 0.4826 | 0.4788 | 0.4661 |
| openbookqa | acc_norm | 0.3060 | 0.3420 | 0.3380 | 0.3480 | 0.3400 | 0.3460 |
| piqa | acc_norm | 0.6790 | 0.7138 | 0.7220 | 0.7225 | 0.7236 | 0.7122 |
| winogrande | acc | 0.5406 | 0.5635 | 0.5659 | 0.5675 | 0.5762 | 0.5549 |
| wsc273 | acc | 0.6081 | 0.6300 | 0.6520 | 0.6337 | 0.6374 | 0.5971 |

### Short Cooldown — acc / acc_norm (5,000 steps, ~1.31B tokens)

| Task | Metric | Pre-cooldown (35k) | Short Rephraser 150W | Short DCLM | Short Nemotron-only |
|---|---|---|---|---|---|
| agieval_lsat_ar | acc_norm | 0.2391 | 0.2130 | 0.1957 | 0.2130 |
| arc_easy | acc_norm | 0.5560 | 0.5934 | 0.6103 | 0.6111 |
| arc_challenge | acc_norm | 0.2654 | 0.2901 | 0.3148 | 0.3106 |
| boolq | acc | 0.4991 | 0.4440 | 0.4443 | 0.5385 |
| commonsense_qa | acc | 0.2039 | 0.2039 | 0.1900 | 0.1974 |
| copa | acc | 0.6500 | 0.7000 | 0.7100 | 0.7500 |
| hellaswag_0shot | acc_norm | 0.4321 | 0.4727 | 0.4904 | 0.5000 |
| hellaswag_10shot | acc_norm | 0.4332 | 0.4751 | 0.4930 | 0.5018 |
| lambada_openai | acc | 0.3695 | 0.4401 | 0.4712 | 0.4590 |
| openbookqa | acc_norm | 0.3060 | 0.3340 | 0.3520 | 0.3420 |
| piqa | acc_norm | 0.6790 | 0.7057 | 0.7165 | 0.7133 |
| winogrande | acc | 0.5406 | 0.5596 | 0.5691 | 0.5643 |
| wsc273 | acc | 0.6081 | 0.6300 | 0.6227 | 0.6264 |

---

## Token Counts

### Full Cooldown Experiments (9,759 steps, ~2.56B tokens)

| Condition | Nemotron Tokens | Mixin Tokens | Tokenized Hash | Mixin % | Status |
|---|---|---|---|---|---|
| Rephraser v2 (25 WARCs) | 2,558,263,296 | 45,904,757 | `0b5b27` | 1.8% | SUCCESS |
| DCLM 300M baseline | 2,558,263,296 | 307,051,596 | `211afe` | 10.7% | SUCCESS |
| DCLM filtered | 2,558,263,296 | 5,671,922 | `406c6b` | 0.2% | SUCCESS |
| Rephraser 150 WARCs | 2,558,263,296 | 362,242,311 | `02c17e` | 12.4% | SUCCESS |

**Note on DCLM filtered**: The DCLM-Baseline filtering pipeline (RefinedWeb heuristics + FastText quality classifier) is extremely aggressive on raw WARC data, reducing 150 WARCs to only ~5.7M tokens (0.2% of training budget). This means the DCLM-filtered run is essentially a nemotron-only run with a negligible mixin.

**Note on Rephraser v2 (25 WARCs)**: The WandB tag `rephraser-tokens=45904757` confirms this run used only 46M tokens of rephraser data (1.8% mixin), from `0b5b27`. This is much less rephraser data than the 150-WARC run (362M, 12.4% mixin).

### Short Cooldown Experiments (5,000 steps, ~1.31B tokens)

| Condition | Nemotron Tokens | Mixin Tokens | Tokenized Hash | Mixin % | Status |
|---|---|---|---|---|---|
| Short rephraser 150W | 1,000,079,360 | 362,242,311 | `02c17e` | 26.6% | SUCCESS |
| Short DCLM | 1,000,079,360 | 307,051,596 | `211afe` | 23.5% | SUCCESS |
| Short nemotron-only | 1,300,234,240 | 0 | — | 0% | SUCCESS |

**Note**: The "Short Rephraser v2" run (`eebec0`) uses the same 150-WARC tokenized data (`02c17e`, 362M tokens) as the full Rephraser 150W run. There is **no valid short 25-WARC run** (the non-v2 short rephraser `ff008c` was bugged with 70.4% mixin).

---

## Validation Perplexity (eval_metrics.jsonl)

### Full Cooldown

| Condition | Step | Loss | BPB | GCS Run Dir |
|---|---|---|---|---|
| Rephraser v2 (25 WARCs) | 9758 | 2.8165 | 0.9624 | `cooldown-rephraser-d7d976d3-v2-1cdc5a` |
| DCLM 300M v2 | 9758 | 2.8070 | 0.9593 | `cooldown-dclm-100m-v2-32bcbc` |
| DCLM filtered | 9758 | 2.8114 | 0.9615 | `cooldown-dclm-filtered-v1-54cdb7` |
| Rephraser 150 WARCs v2 | 9758 | 2.8289 | 0.9664 | `cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b` |
| Nemotron-only (step-44758) | 44758 | 2.828 | 0.966 | (from original exp2166 training logs) |

### Short Cooldown

| Condition | Step | Loss | BPB | GCS Run Dir |
|---|---|---|---|---|
| Short rephraser v2 | 4999 | 2.8795 | 0.9845 | `short-cooldown-rephraser-d7d976d3-v2-eebec0` |
| Short DCLM | 4999 | 2.8351 | 0.9697 | `short-cooldown-dclm-49e19a` |
| Short nemotron-only | 4999 | 2.8653 | 0.9793 | `short-cooldown-nemotron-only-391eeb` |

---

## Key Findings

### choice_prob_norm tells a different story than acc/acc_norm

The `choice_prob_norm` metric shows much smaller differences between conditions than `acc` or `acc_norm`. Most choice_prob_norm values differ by less than 0.01 across conditions, suggesting that the data mix during cooldown has minimal impact on calibrated probability estimates, even when raw accuracy varies more.

### Full Cooldown

1. **All cooldown conditions improve over pre-cooldown baseline** (step-35000) on choice_prob_norm, as expected since cooldown adds ~2.56B more training tokens.

2. **On choice_prob_norm, no mixin condition clearly beats nemotron-only** (step-44758). Differences are extremely small (typically <0.005). The conditions are effectively indistinguishable on this metric.

3. **Rephraser v2 (25W) used only 46M rephraser tokens (1.8% mixin)**, much less than the 150W run (362M, 12.4%). Despite this difference in mixin fraction, both rephraser conditions perform similarly on choice_prob_norm.

4. **boolq is the most variable task** on choice_prob_norm. Rephraser v2 (25W) scores 0.4923 vs nemotron-only at 0.5149 — the largest gap across any task/condition pair. Rephraser 150W (0.5131) performs much closer to nemotron-only on boolq.

5. **DCLM filtered is effectively nemotron-only** due to extreme filtering reducing the mixin to 0.2% of training tokens. Its choice_prob_norm results are near-identical to the nemotron-only baseline, serving as a useful sanity check.

6. **DCLM 300M has a slight edge** on choice_prob_norm across several tasks (hellaswag, openbookqa, arc_easy), but the margins are tiny.

### Short Cooldown

1. **Short nemotron-only is the strongest short-cooldown condition on choice_prob_norm**, winning or tying on every single task. With only 5k steps, replacing 25% of training data with web text hurts.

2. **Short rephraser 150W is the weakest on choice_prob_norm**, consistent with diluting 26.6% of training tokens with rephraser data during a short cooldown being detrimental. Note: this run uses 150-WARC data (362M tokens, `02c17e`), not 25-WARC data.

3. **boolq again shows the largest spread**: short nemotron-only (0.5233) vs short rephraser 150W (0.4936), a ~3pp gap.

4. **There is no valid short 25-WARC run**. The original short rephraser (`ff008c`) was bugged (70.4% mixin). A short 25-WARC run with correct fractions (~4.4% mixin) has not been done.

---

## Run Status Registry

### Full Cooldown Runs (9,759 steps)

| Run | GCS Dir | Status | Eval Source |
|---|---|---|---|
| Rephraser v2 (25W) | `cooldown-rephraser-d7d976d3-v2-1cdc5a` | SUCCESS | WandB (in-training) |
| DCLM 300M v2 | `cooldown-dclm-100m-v2-32bcbc` | SUCCESS | WandB (in-training) |
| DCLM filtered | `cooldown-dclm-filtered-v1-54cdb7` | SUCCESS | Standalone eval `lmeval_debug_hf_step-9758-c460fa` |
| Rephraser 150W v2 | `cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b` | SUCCESS | WandB (in-training, v5p-8) |

### Short Cooldown Runs (5,000 steps)

| Run | GCS Dir | Status | Eval Source |
|---|---|---|---|
| Short rephraser 150W | `short-cooldown-rephraser-d7d976d3-v2-eebec0` | SUCCESS | Standalone eval `lmeval_debug_hf_step-4999-8b06e2` |
| Short DCLM | `short-cooldown-dclm-49e19a` | SUCCESS | WandB (in-training) |
| Short nemotron-only | `short-cooldown-nemotron-only-391eeb` | SUCCESS | WandB (in-training) |

### Baseline Evals

| Eval | GCS Dir | Status |
|---|---|---|
| Pre-cooldown (step-35000) | `evaluation/.../lmeval_debug_hf/scaling-1e20-step-35000-662640` | SUCCESS |
| Post-cooldown (step-44758) | `evaluation/.../lmeval_debug_hf_step-44758-8a20e0` | SUCCESS |

### Invalid / Bugged Runs (for reference)

| Run | GCS Dir | Issue |
|---|---|---|
| Short rephraser (BUGGED) | `short-cooldown-rephraser-d7d976d3-ff008c` | 70.4% rephraser fraction due to token count bug |
| 150W rephraser (BUGGED) | `cooldown-rephraser-d7d976d3-150warc-56ba41` | 48.2% rephraser fraction due to token count bug |

---

## Known Issues

### Token Count Bug (fixed 2026-03-03)

**Bug**: `_read_token_count` in `rephraser_cooldown.py` fell back to
`shard_ledger.json rows * SEQ_LEN` when `.stats.json` was missing. For
variable-length tokenizations (rephraser data), this gave 2.38B instead of 362M
tokens, causing the model to train with ~48% rephraser data instead of ~12%.

**Affected runs**:
- `cooldown-rephraser-d7d976d3-150warc-56ba41` (150 WARCs, ran with 48.2% rephraser) - INVALID
- `short-cooldown-rephraser-d7d976d3-ff008c` (short cooldown, ran with 70.4% rephraser) - INVALID

**Not affected** (these runs used correct token counts because their `.stats.json`
was in the expected location):
- `cooldown-rephraser-d7d976d3-v2-1cdc5a` (rephraser v2, 25 WARCs) - VALID
- `cooldown-dclm-100m-v2-32bcbc` (DCLM 300M) - VALID
- `cooldown-dclm-filtered-v1-54cdb7` (DCLM filtered) - VALID
- `short-cooldown-dclm-49e19a` (short DCLM) - VALID
- `short-cooldown-nemotron-only-391eeb` (short nemotron) - VALID
- `cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b` (150W v2, fixed) - VALID
- `short-cooldown-rephraser-d7d976d3-v2-eebec0` (short rephraser v2, fixed) - VALID

**Fix**: `_read_token_count` now checks per-shard stats and crashes instead of
falling back. `_validate_mixin_fraction` was added to crash if mixin fraction
exceeds 30%.

### DCLM Filtered Token Count

The DCLM-Baseline filtering pipeline is extremely aggressive on raw WARC HTML.
Only 5,671,922 tokens survived filtering from 150 WARCs, making the mixin fraction
0.22%. The DCLM-filtered condition is effectively identical to nemotron-only cooldown.

### Cluster Name Confusion

`--cluster us-east1-d` resolves to the **vllm inference cluster**, not the training
cluster. Use `--cluster us-east1` for training on us-east1-d.

---

## GCS Path Reference

### Tokenized Data

| Dataset | GCS Path | Tokens |
|---|---|---|
| Nemotron cooldown (2.56B) | `gs://marin-{region}/tokenized/nemotron_cooldown_1e20-666089` | 2,558,263,296 |
| Nemotron short (1B) | `gs://marin-{region}/tokenized/nemotron_cooldown_1e20_short_1b-413400` | 1,000,079,360 |
| Rephraser 150W (d7d976d3) | `gs://marin-{region}/tokenized/rephraser_spec_d7d976d3_cooldown-02c17e` | 362,242,311 |
| Rephraser 25W (d7d976d3) | `gs://marin-{region}/tokenized/rephraser_spec_d7d976d3_cooldown-0b5b27` | 45,904,757 |
| DCLM 300M | `gs://marin-us-central1/tokenized/dclm_baseline_100m-211afe` | 307,051,596 |
| DCLM filtered | `gs://marin-us-east1/tokenized/dclm_filtered_warcs_llama3-406c6b` | 5,671,922 |

Available regions: `us-central1`, `us-east1`, `eu-west4`, `us-east5`
(not all datasets are copied to all regions)
