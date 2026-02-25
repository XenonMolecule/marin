# Midtraining Baselines Report

**Date:** 2026-02-25
**Cluster:** us-central1
**Branch:** michael-distill

## Overview

Three experiments to establish baseline benchmark scores for Llama 3.2 1B:

| Job | Job ID | Status |
|-----|--------|--------|
| Baseline eval (no training) | `ray-run-michaelryan-llama_3_2_1b_baseline_eval-20260225-180112` | **SUCCEEDED** |
| Midtraining baselines (DCLM + Rephraser) | `ray-run-michaelryan-midtrain_baselines-20260225-191437` | RUNNING (restart 4) |

## Baseline Eval Results (Llama 3.2 1B, zero-shot)

W&B run: https://wandb.ai/marin-community/marin/runs/7r05s3n4

Key averages:
- **macro_avg_acc**: 0.537
- **macro_avg_acc_norm**: 0.528
- **micro_avg_acc_norm**: 0.642

Per-task scores (acc_norm where available, else acc):

| Task | Score |
|------|-------|
| agieval_lsat_ar | 0.165 (acc_norm) |
| arc_challenge | 0.400 (acc_norm) |
| arc_easy | (see W&B) |
| boolq | (see W&B) |
| commonsense_qa | (see W&B) |
| copa | (see W&B) |
| hellaswag (0-shot) | (see W&B) |
| hellaswag (10-shot) | (see W&B) |
| lambada_openai | (see W&B) |
| openbookqa | (see W&B) |
| piqa | (see W&B) |
| wsc273 | (see W&B) |
| winogrande | (see W&B) |

GCS results: `gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/lmeval_debug_models/meta-llama--Llama-3-2-1B--4e20de3/results.json`

## Files Created

### New files
1. **`lib/marin/src/marin/transform/clean_rephraser_messages.py`**
   Zephyr-based transform that reads HF datasets in chat format (messages column), extracts the last assistant response, strips `<think>...</think>` blocks and DSPy field markers (`[[ ## text ## ]]`, `[[ ## completed ## ]]`), filters short/empty records, and writes JSONL with a plain `text` column. Reuses the same regex patterns as `postprocess_extraction.py`.

2. **`experiments/rephraser/midtrain_baselines.py`**
   Main experiment script containing two independent baselines:
   - **DCLM baseline**: Downloads ~20 shard files from `mlfoundations/dclm-baseline-1.0` (first 20 files from `global-shard_01_of_10/local-shard_0_of_10/`), tokenizes with Llama 3.2 1B tokenizer, midtrains for 95 steps.
   - **Rephraser baseline**: Downloads `MichaelR207/rephraser_late_check_0225` (818K examples), cleans assistant text, tokenizes, midtrains for 95 steps.

3. **`experiments/rephraser/llama_3_2_1b_baseline_eval.py`**
   Simple eval-only script that runs CORE_TASKS on the unmodified Llama 3.2 1B checkpoint (no training).

### Modified files
4. **`experiments/posttrain/instruction_datasets.py`**
   Added `MichaelR207/rephraser_late_check_0225` to the instruction dataset registry (revision `2194850`, same format as previous rephraser datasets).

5. **`lib/marin/src/marin/evaluation/evaluators/levanter_lm_eval_evaluator.py`**
   Removed dangling `self.cleanup(model)` call in the `finally` block — the method never existed, causing `AttributeError` that crashed all evaluations.

## Training Configuration

All midtraining baselines use identical hyperparameters (matched to `rephraser_sweep.py`):

| Parameter | Value |
|-----------|-------|
| Token budget | 100M tokens |
| Sequence length | 4096 |
| Batch size | 256 |
| Training steps | 95 (= 100M / (4096 × 256)) |
| Learning rate | 3e-4 (10x lower than from-scratch) |
| LR schedule | Cosine |
| Min LR ratio | 0.1 |
| Warmup | 5% of steps |
| Weight decay | 0.033 |
| z-loss weight | 1e-4 |
| Base model | meta-llama/Llama-3.2-1B (initialize_from_hf) |
| TPU | v5p-8 |
| per_device_parallelism | 16 (4x gradient accumulation) |
| Shuffle seed | 42 (reproducible) |
| Eval | CORE_TASKS (13 tasks), once at end |

## Issues Encountered & Fixes

### 1. `self.cleanup(model)` AttributeError (FIXED)
The Levanter LM eval evaluator called a nonexistent `self.cleanup()` method, crashing all eval runs. Removed the dead `finally` block.

### 2. `DownloadConfig.append_sha_to_path` defaults to False (FIXED)
Initial code added `.cd("revision")` subdirectories to paths, but files are written directly to the output path. Removed all `.cd()` calls.

### 3. Node preemption during tokenization (transient)
Restart 3 was needed after a Ray actor died during tokenization. The executor correctly skipped cached steps on restart.

### 4. OOM during training — `f32[64,4096,128256]` exceeds HBM (FIXED)
With batch_size=256 on v5p-8 (4 chips), per-device batch = 64. The f32 logits tensor (64 × 4096 × 128256 = 134GB) exceeds the 103GB HBM per chip. Fixed by adding `per_device_parallelism=16` for 4x gradient accumulation, reducing peak memory.

### 5. DCLM tokenization — `ZephyrWorkerError: No workers available`
The DCLM tokenization failed because no workers were available after 61s, likely due to cluster resource contention. Will retry in restart 4.

## Evaluation Suite (CORE_TASKS)

The 13 tasks evaluated (matching the rephraser sweep):
- AGIEval LSAT-AR (3-shot), ARC-Easy (10), ARC-Challenge (10), BoolQ (10)
- CommonsenseQA (10), COPA (0), HellaSwag (0 & 10-shot), LAMBADA (0)
- OpenBookQA (0), PIQA (10), WSC273 (0), Winogrande (0)

## Data Pipeline Summary

### DCLM baseline
```
HuggingFace (DCLM shard files, .jsonl.zst)
  → download_hf (20 files, ~5-10GB)
  → tokenize (TextLmDatasetFormat, text_key="text")
  → midtrain Llama 3.2 1B (95 steps, ~100M tokens)
  → CORE_TASKS eval
```

### Rephraser baseline
```
HuggingFace (rephraser_late_check_0225, 818K examples, parquet)
  → download_hf (full dataset)
  → clean_rephraser_messages (extract assistant text, strip <think>/DSPy markers)
    → 320,864 records, 167,643,200 tokens after cleaning
  → tokenize (TextLmDatasetFormat, text_key="text")
  → midtrain Llama 3.2 1B (95 steps, ~100M tokens)
  → CORE_TASKS eval
```

## Monitoring

Current job: `ray-run-michaelryan-midtrain_baselines-20260225-191437`

Check logs:
```bash
uv run scripts/ray/cluster.py --cluster us-central1 job-logs -n 50 ray-run-michaelryan-midtrain_baselines-20260225-191437
```
