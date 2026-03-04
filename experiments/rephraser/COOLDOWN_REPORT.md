# Rephraser Cooldown Experiment Results

**Last updated**: 2026-03-03

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

## Full Cooldown Experiments (9,759 steps, ~2.56B tokens)

### Token Counts

| Condition | Nemotron Tokens | Mixin Tokens | Mixin Name | Mixin % | Status |
|---|---|---|---|---|---|
| Rephraser v2 (25 WARCs) | 2,558,263,296 | 362,242,311 | rephraser_d7d976d3 | 12.4% | SUCCESS |
| DCLM 100M baseline | 2,558,263,296 | 307,051,596 | dclm_baseline_100m | 10.7% | SUCCESS |
| DCLM filtered | 2,558,263,296 | TBD | dclm_filtered | TBD | TBD |
| Rephraser 150 WARCs | 2,558,263,296 | 362,242,311 | rephraser_d7d976d3 | 12.4% | RE-RUNNING (prev run bugged) |

### Validation Perplexity (eval_metrics.jsonl)

| Condition | Step | Loss | BPB | GCS Run Dir |
|---|---|---|---|---|
| Rephraser v2 (25 WARCs) | 9758 | 2.8165 | 0.9624 | `cooldown-rephraser-d7d976d3-v2-1cdc5a` |
| DCLM 100M v2 | 9758 | 2.8070 | 0.9593 | `cooldown-dclm-100m-v2-32bcbc` |
| Nemotron-only (step-44758) | 44758 | 2.828 | 0.966 | (from original exp2166 training logs) |

### lm-eval Results (CORE_TASKS)

**Source**: WandB (in-training eval) or GCS `results.json` (standalone eval).

#### Post-cooldown Nemotron Baseline (step-44758)

Source: `gs://marin-us-central1/evaluation/.../lmeval_debug_hf_step-44758-8a20e0/results.json`

| Task | acc | acc_norm |
|---|---|---|
| agieval_lsat_ar | 0.2435 | 0.2478 |
| arc_easy | 0.6292 | 0.6199 |
| arc_challenge | 0.2927 | 0.3208 |
| boolq | 0.5150 | - |
| commonsense_qa | 0.2064 | - |
| copa | 0.7100 | - |
| hellaswag_0shot | 0.3998 | 0.5121 |
| hellaswag_10shot | 0.3957 | 0.5088 |
| lambada_openai | 0.4687 | - |
| openbookqa | 0.2260 | 0.3420 |
| piqa | 0.7144 | 0.7138 |
| winogrande | 0.5635 | - |
| wsc273 | 0.6300 | - |

#### Pre-cooldown Baseline (step-35000)

Source: Not yet available. The standalone eval (`scaling_1e20_baseline_eval.py`)
needs to be run or its results located.

TODO: Run `scaling_1e20_baseline_eval.py` on a cluster with v5p-8 and record results here.

#### Rephraser v2 (25 WARCs) lm-eval

Source: WandB (in-training eval at step 9758)

TODO: Query WandB for run with tags `rephraser-cooldown, spec-d7d976d3` and
extract `lm_eval/*` metrics from run summary. See guide for query instructions.

#### DCLM 100M v2 lm-eval

Source: WandB (in-training eval at step 9758)

TODO: Query WandB for run with tags `rephraser-cooldown` matching `dclm` runs.

---

## Short Cooldown Experiments (5,000 steps, ~1.31B tokens)

### Token Counts

| Condition | Nemotron Tokens | Mixin Tokens | Mixin Name | Mixin % | Status |
|---|---|---|---|---|---|
| Short rephraser | 1,000,079,360 | 362,242,311 | rephraser_d7d976d3 | 26.6% | RE-RUNNING (prev run bugged: 70.4%) |
| Short DCLM | 1,000,079,360 | 307,051,596 | dclm_baseline_100m | 23.5% | lm-eval running |
| Short nemotron-only | 1,300,234,240 | 0 | - | 0% | lm-eval running |

**Note on short nemotron-only**: Uses 1B + 300M nemotron tokens (~1.3B total) with
no mixin data. This is the within-budget baseline.

### Validation Perplexity

| Condition | Step | Loss | BPB | GCS Run Dir |
|---|---|---|---|---|
| Short rephraser (BUGGED) | 4999 | 2.9663 | 1.0144 | `short-cooldown-rephraser-d7d976d3-ff008c` |
| Short DCLM | 4999 | 2.8351 | 0.9697 | `short-cooldown-dclm-49e19a` |
| Short nemotron-only | 4999 | 2.8653 | 0.9793 | `short-cooldown-nemotron-only-391eeb` |

**WARNING**: The short rephraser run at `ff008c` trained with BUGGED mixin
fraction (70.4% instead of 26.6%) due to the `_read_token_count` bug.
Its results are invalid. A corrected re-run was launched on 2026-03-03.

### lm-eval Results

TODO: lm-eval is currently running on us-east5-a for the short DCLM and
nemotron-only conditions. The short rephraser is being re-trained.

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
- `cooldown-dclm-100m-v2-32bcbc` (DCLM 100M) - VALID
- `short-cooldown-dclm-49e19a` (short DCLM) - VALID
- `short-cooldown-nemotron-only-391eeb` (short nemotron) - VALID

**Fix**: `_read_token_count` now checks per-shard stats and crashes instead of
falling back. `_validate_mixin_fraction` was added to crash if mixin fraction
exceeds 30%.

### Cluster Name Confusion

`--cluster us-east1-d` resolves to the **vllm inference cluster**, not the training
cluster. Use `--cluster us-east1` for training on us-east1-d.

---

## GCS Path Reference

### Tokenized Data

| Dataset | GCS Path |
|---|---|
| Nemotron cooldown (2.56B) | `gs://marin-{region}/tokenized/nemotron_cooldown_1e20-666089` |
| Nemotron short (1B) | `gs://marin-{region}/tokenized/nemotron_cooldown_1e20_short_1b-413400` |
| Rephraser (25 WARCs) | `gs://marin-{region}/tokenized/rephraser_spec_d7d976d3_cooldown-02c17e` |
| DCLM 100M | `gs://marin-us-central1/tokenized/dclm_baseline_100m-211afe` |

Available regions: `us-central1`, `us-east1`, `eu-west4`, `us-east5`
(not all datasets are copied to all regions)

### Training Outputs (us-central1)

| Run | GCS Dir | Status |
|---|---|---|
| Rephraser v2 | `cooldown-rephraser-d7d976d3-v2-1cdc5a` | SUCCESS |
| DCLM 100M v2 | `cooldown-dclm-100m-v2-32bcbc` | SUCCESS |
| DCLM filtered | `cooldown-dclm-filtered-v1-54cdb7` | TBD |
| Short rephraser (BUGGED) | `short-cooldown-rephraser-d7d976d3-ff008c` | INVALID |
| Short DCLM | `short-cooldown-dclm-49e19a` | lm-eval running |
| Short nemotron-only | `short-cooldown-nemotron-only-391eeb` | lm-eval running |

### Standalone Evals

| Eval | GCS Dir |
|---|---|
| Post-cooldown (step-44758) | `evaluation/.../lmeval_debug_hf_step-44758-8a20e0` |
| Pre-cooldown (step-35000) | Not yet run |
