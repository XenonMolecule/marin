# How to Produce a Rephraser Cooldown Report

This guide documents the exact process for gathering data, validating token counts,
finding lm-eval results, and checking for bugs in rephraser cooldown experiments.

## Overview

The rephraser cooldown experiments take a Qwen3 1.385B model trained with the
Nemotron data mix (exp2166, WSD schedule), resume from the pre-cooldown checkpoint
(step 35,000), and train for the cooldown phase with a mixture of Nemotron data and
rephraser-extracted data. The goal is to measure whether mixing rephraser data
during cooldown improves downstream lm-eval benchmarks.

**Key constants** (from `experiments/rephraser/rephraser_cooldown.py`):
- Model: Custom 1.385B (Qwen3 architecture, trained from scratch in Marin exp2166;
  uses Llama3 tokenizer, NOT a pretrained Qwen3 model)
- SEQ_LEN = 4096
- BATCH_SIZE = 64
- Full cooldown: 9,759 steps (steps 35,000 -> 44,759), ~2.56B tokens
- Short cooldown: 5,000 steps, ~1.31B tokens
- Optimizer: CautiousConfig, linear LR decay from 0.001473 to 0

---

## 1. Identify All Experiment Runs

### Where experiments live in code

| Experiment | Script | Budget | Rephraser Data |
|---|---|---|---|
| Rephraser v2 (25 WARCs) | `rephraser_cooldown.py` | 2.56B tokens (9,759 steps) | `0b5b27` (46M, 1.8% mixin) |
| Rephraser 150 WARCs | `rephraser_cooldown_150warc_train*.py` | 2.56B tokens (9,759 steps) | `02c17e` (362M, 12.4% mixin) |
| DCLM 300M baseline | `dclm_cooldown.py` | 2.56B tokens (9,759 steps) | N/A (DCLM data) |
| DCLM filtered | `dclm_filtered_cooldown.py` | 2.56B tokens (9,759 steps) | N/A (DCLM data) |
| Short rephraser 150W | `short_cooldown/rephraser_east1d.py` | 1.31B tokens (5,000 steps) | `02c17e` (362M, 26.6% mixin) |
| Short DCLM | `short_cooldown/dclm.py` | 1.31B tokens (5,000 steps) | N/A (DCLM data) |
| Short nemotron-only | `short_cooldown/nemotron_only.py` | 1.31B tokens (5,000 steps) | None |
| Baseline pre-cooldown | `scaling_1e20_baseline_eval.py` | Eval only (step 35,000) | N/A |
| Baseline post-cooldown | `scaling_1e20_final_eval.py` | Eval only (step 44,758) | N/A |

**Note**: There is no valid short 25-WARC run. The original short rephraser (`ff008c`) was bugged (70.4% mixin due to token count bug).

### Where outputs live on GCS

Training outputs use the executor step name as the directory:
```
gs://marin-{region}/{step-name}-{hash}/
```

To list all cooldown-related outputs on a region:
```bash
gcloud storage ls gs://marin-us-central1/ | grep -E "(cooldown|rephraser)"
```

Key output directories (us-central1 examples):
```
gs://marin-us-central1/cooldown-rephraser-d7d976d3-v2-{hash}/       # Rephraser v2
gs://marin-us-central1/cooldown-rephraser-d7d976d3-150warc-{hash}/   # 150 WARCs
gs://marin-us-central1/cooldown-dclm-100m-{hash}/                    # DCLM 300M
gs://marin-us-central1/cooldown-dclm-filtered-v1-{hash}/             # DCLM filtered
gs://marin-us-central1/short-cooldown-rephraser-d7d976d3-{hash}/     # Short rephraser
gs://marin-us-central1/short-cooldown-dclm-{hash}/                   # Short DCLM
gs://marin-us-central1/short-cooldown-nemotron-only-{hash}/          # Short nem-only
```

**Note**: Multiple hashes may exist for the same experiment if it was re-run (e.g. after a bug fix). Check `.executor_status` to find successful runs.

### Completed run registry (verified SUCCESS)

#### 150-WARC Rephraser Cooldown (full, 9,759 steps)

| Cluster | GCS Path | Final Step | HF Export | Executor Status | W&B | Notes |
|---|---|---|---|---|---|---|
| us-central1 | `gs://marin-us-central1/cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b/` | step-9758 | `hf/step-9758` | SUCCESS | `cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b` | Clean run, 0 restarts, v5p-8, inline lm-eval succeeded |
| us-east1 | `gs://marin-us-east1/cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b/` | step-9758 | `hf/step-9758` | SUCCESS | `cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b` | 5 restarts due to v6e-32 preemption, inline lm-eval may have failed on multi-host |

- **Script**: `experiments/rephraser/rephraser_cooldown_150warc_train_central1.py` (central1) / `rephraser_cooldown_150warc_train_east1d.py` (east1)
- **Step name**: `cooldown-rephraser-d7d976d3-150warc-v2` (hash: `f8aa0b`)
- **Tokenized data**: rephraser at `tokenized/rephraser_spec_d7d976d3_cooldown-02c17e`, nemotron at `tokenized/nemotron_cooldown_1e20-666089`
- **Final loss**: ~2.66-2.69
- **Training time**: ~7.5 hours on v5p-8 (central1)

#### Short Rephraser 150W Cooldown (5,000 steps)

| Cluster | GCS Path | Final Step | HF Export | Executor Status | Notes |
|---|---|---|---|---|---|
| us-central1 | `gs://marin-us-central1/short-cooldown-rephraser-d7d976d3-v2-eebec0/` | step-4999 | `hf/step-4999` | SUCCESS (training on east1, HF copied to central1) | HF export is 5.17 GB (2 safetensors shards) |
| us-east1 | `gs://marin-us-east1/short-cooldown-rephraser-d7d976d3-v2-eebec0/` | step-4999 | `hf/step-4999` | FAILED (lm-eval crash on multi-host v6e-32) | Training complete, HF export exists, but executor marked FAILED due to lm-eval |

- **Script**: `experiments/rephraser/short_cooldown/rephraser_east1d.py` (east1) / `short_cooldown/eval_rephraser.py` (eval on central1)
- **Step name**: `short-cooldown-rephraser-d7d976d3-v2` (hash: `eebec0`)
- **Rephraser data**: `02c17e` (362M tokens from ~150 WARCs, 26.6% mixin)
- **HF model size**: 5.17 GB (bf16, 2 safetensors shards)
- **Note**: Despite the step name containing `-v2` (not `-150warc`), this run uses the 150-WARC tokenized data (`02c17e`). There is no valid short 25-WARC run.

#### Standalone Eval Runs

| Eval Target | GCS Results Path | Status |
|---|---|---|
| Pre-cooldown (step-35000) | `gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/lmeval_debug_hf/scaling-1e20-step-35000-662640/` | SUCCESS |
| Post-cooldown (step-44758) | `gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/lmeval_debug_hf_step-44758-8a20e0/` | SUCCESS |
| Short rephraser 150W (step-4999) | `gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/lmeval_debug_hf_step-4999-8b06e2/` | SUCCESS |
| DCLM filtered (step-9758) | `gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/lmeval_debug_hf_step-9758-c460fa/` | SUCCESS |
| 150-WARC (step-9758) | Inline with training run on central1 (WandB only) | SUCCESS |

**Note**: The pre-cooldown eval is in a nested subdirectory (`lmeval_debug_hf/scaling-1e20-step-35000-662640/`) due to the HF export step name containing a slash.

#### Failed / Dead Clusters

| Cluster | Issue | Date |
|---|---|---|
| eu-west4-a | Persistent TPU HAL init error on node 10.164.1.112; all v6e-32 allocations fail | 2026-03-04 |
| us-east5-a | Severe v5p-8 preemption pressure; zero training progress across 4+ restarts | 2026-03-04 |

### Multi-host lm-eval limitation

lm-eval harness **does not work on multi-host TPUs** (e.g. v6e-32 with 8 VMs). Symptoms:
- HuggingFace API rate limiting (429 errors) from 8 nodes hitting the API simultaneously
- `TypeError: 'NoneType' object is not iterable` during task loading
- `ValueError: device_put's first argument must be a fully addressable array`

**Workaround**: Run eval separately on single-host TPUs (v5p-8) using `default_eval()` from `experiments/evals/evals.py`, or rely on inline lm-eval only when training on single-host TPUs.

### Checking run status
```bash
gcloud storage cat "gs://marin-us-central1/{step-name}-{hash}/.executor_status"
```

---

## 2. Get Token Counts

Token counts are the **most critical** piece of the report and the most common
source of bugs. There are two different storage formats depending on the pipeline.

### Format 1: Newer tokenize pipeline (top-level `.stats.json`)
```
{cache_path}/train/.stats.json
```
Contains: `{"total_tokens": <int>, "total_elements": <int>}`

```bash
gcloud storage cat "gs://marin-us-central1/tokenized/{name}/train/.stats.json"
```

### Format 2: Older tokenize pipeline (per-shard `.stats.json`)
```
{cache_path}/train/part-00000/.stats.json
```
Contains: `{"count": <int>, "token_count": <int>}`

```bash
gcloud storage cat "gs://marin-us-central1/tokenized/{name}/train/part-00000/.stats.json"
```

### Format 3: Fixed-length extraction (e.g. NemotronCooldown)

The `extract_cooldown_data` function writes fixed-length sequences (exactly SEQ_LEN
tokens per row). After fixing, it now writes a `.stats.json` at the top level.
For older extractions, a `.stats.json` was uploaded manually:

```bash
gcloud storage cat "gs://marin-us-central1/tokenized/nemotron_cooldown_1e20-666089/train/.stats.json"
# => {"total_tokens": 2558263296, "total_elements": 0}
```

### How to find the right tokenized path

Each experiment script defines its tokenized data path. Grep for the variable:
```bash
grep -n "TOKENIZED_PATH\|tokenized_path\|cache_dir" experiments/rephraser/<script>.py
```

Or look at the executor pipeline — the tokenize step is typically named:
```
tokenized/rephraser_spec_{hash}_cooldown-{hash}
tokenized/dclm_baseline_100m-{hash}
tokenized/dclm_filtered_rephraser_spec_{hash}_cooldown-{hash}
tokenized/nemotron_cooldown_1e20-{hash}
tokenized/nemotron_cooldown_1e20_short_1b-{hash}
```

### Known token counts (reference)

| Dataset | Path suffix | Tokens | Notes |
|---|---|---|---|
| Nemotron cooldown (2.56B) | `nemotron_cooldown_1e20-666089` | 2,558,263,296 | Full cooldown base data |
| Nemotron short (1B) | `nemotron_cooldown_1e20_short_1b-413400` | 1,000,079,360 | Short cooldown base data |
| Rephraser 150W (d7d976d3) | `rephraser_spec_d7d976d3_cooldown-02c17e` | 362,242,311 | From ~150 WARCs |
| Rephraser 25W (d7d976d3) | `rephraser_spec_d7d976d3_cooldown-0b5b27` | 45,904,757 | From ~25 WARCs |
| DCLM 300M baseline | `dclm_baseline_100m-211afe` | 307,051,596 | |
| DCLM filtered | `dclm_filtered_warcs_llama3-406c6b` | 5,671,922 | Extremely aggressive filtering |

**IMPORTANT — 25W vs 150W tokenized data**:
There are TWO different rephraser tokenized datasets with the same spec hash (`d7d976d3`):
- `0b5b27` = 46M tokens from ~25 WARCs (used by Rephraser v2 25W full cooldown)
- `02c17e` = 362M tokens from ~150 WARCs (used by Rephraser 150W full cooldown AND Short Rephraser)

The 150-WARC training scripts (`rephraser_cooldown_150warc_train_*.py`) and the short
rephraser v2 scripts (`short_cooldown/rephraser_east1d.py` etc.) all **hardcode** the
`02c17e` path. The original `rephraser_cooldown.py` pipeline dynamically resolves to
`0b5b27` based on the pipeline config at the time it was run.

Always verify which tokenized hash a run actually used by checking the WandB tags
(`rephraser-tokens=...`) or the executor step config.

### Verifying token counts (IMPORTANT)

The `_read_token_count` function in `rephraser_cooldown.py` checks `.stats.json`
files. It was previously buggy and would silently fall back to
`shard_ledger.json rows * SEQ_LEN`, which is WRONG for variable-length
tokenizations. The fix:

1. Checks `{base}/.stats.json` for `"total_tokens"` key
2. Checks `{base}/part-*/.stats.json` for `"token_count"` key (sums all shards)
3. Crashes with `ValueError` if neither is found

**To manually verify a token count**:
```bash
# Method 1: Top-level stats
gcloud storage cat "{path}/train/.stats.json" 2>/dev/null

# Method 2: Per-shard stats (if top-level doesn't exist)
gcloud storage cat "{path}/train/part-00000/.stats.json" 2>/dev/null

# Method 3: For fixed-length caches, cross-check with shard_ledger
gcloud storage cat "{path}/train/shard_ledger.json" | python3 -c "
import sys, json
ledger = json.load(sys.stdin)
rows = sum(s['num_rows'] for s in ledger['shards'])
print(f'Rows: {rows:,}')
print(f'Rows * SEQ_LEN: {rows * 4096:,}')
"
```

**Red flag**: If the mixin data fraction exceeds 30%, something is likely wrong.
The `_validate_mixin_fraction` function (added after the token count bug) will
crash with a clear error message. Expected fractions:
- Full cooldown: rephraser ~12-14% of total tokens
- Short cooldown: rephraser ~26-27% of total tokens

---

## 3. Get Evaluation Results

### Source 1: In-training lm-eval (WandB only)

When `eval_harness_steps` is set in the training config, lm-eval runs at that step
during training. Results are logged to WandB ONLY (not to `eval_metrics.jsonl`).

**How to query WandB**:
```bash
# Install wandb if needed
pip install wandb

# Query via API
python3 -c "
import wandb
api = wandb.Api()
runs = api.runs('marin', filters={'tags': 'rephraser-cooldown'})
for run in runs:
    print(f'{run.name} ({run.state})')
    # Get lm-eval metrics from summary
    for k, v in run.summary.items():
        if 'lm_eval' in k:
            print(f'  {k}: {v}')
"
```

**WandB metric format**: `lm_eval/{task_alias}/{metric}`

Example metrics (primary — use these first):
```
lm_eval/hellaswag_0shot/choice_prob_norm
lm_eval/arc_challenge/choice_prob_norm
lm_eval/piqa/choice_prob_norm
lm_eval/winogrande/choice_prob_norm
lm_eval/boolq/choice_prob_norm
```

Example metrics (secondary — for reference):
```
lm_eval/hellaswag_0shot/acc_norm
lm_eval/arc_challenge/acc_norm
lm_eval/lambada_openai/acc
```

**Primary metric**: `choice_prob_norm` (NOT `acc_norm` — these are different metrics!).

**WandB tags to filter by**:
- `rephraser-cooldown` — all rephraser cooldown runs
- `short-cooldown` — short (5k step) runs
- `spec-d7d976d3` — specific rephraser spec
- `v6e-32` — runs on v6e-32 TPUs
- `150warc` — 150-WARC runs

### Source 2: Standalone eval results (GCS `results.json`)

For standalone evaluation runs (e.g. baseline evals), results are stored on GCS:
```
gs://marin-{region}/evaluation/lm_evaluation_harness_levanter/lmeval_debug_{model_name}-{hash}/results.json
```

**How to find the eval output path**:

The `default_eval` function constructs the step name as:
```
evaluation/lm_evaluation_harness_levanter/lmeval_debug_{model_name}
```

Where `model_name` comes from:
- For `ExecutorStep` inputs: the step's `.name` attribute
- For string (GCS path) inputs: last two path components joined with `_`

**Example**: For `MODEL_PATH = "gs://.../hf/step-44758"`:
```bash
gcloud storage ls gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/ | grep "step-44758"
# => lmeval_debug_hf_step-44758-8a20e0/
```

**Reading results** (primary metric — choice_prob_norm):
```bash
gcloud storage cat "gs://marin-us-central1/evaluation/.../results.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for task, metrics in d['results'].items():
    cpn = metrics.get('choice_prob_norm,none', 'N/A')
    acc = metrics.get('acc,none', 'N/A')
    acc_norm = metrics.get('acc_norm,none', 'N/A')
    print(f'{task}: choice_prob_norm={cpn}, acc={acc}, acc_norm={acc_norm}')
"
```

**results.json structure**:
```json
{
  "results": {
    "task_name": {
      "acc,none": 0.5,
      "acc_stderr,none": 0.01,
      "acc_norm,none": 0.55,
      "acc_norm_stderr,none": 0.01,
      ...
    }
  },
  "configs": {...},
  "versions": {...},
  "n-shot": {...}
}
```

### Source 3: Validation perplexity (`eval_metrics.jsonl`)

The `eval_metrics.jsonl` file in the checkpoints directory contains validation
loss/perplexity but NOT lm-eval results:

```bash
gcloud storage cat "gs://marin-us-central1/{run-dir}/checkpoints/eval_metrics.jsonl" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    step = d.get('step')
    loss = d.get('eval/loss')
    bpb = d.get('eval/bpb')
    paloma = d.get('eval/paloma/bpb')
    print(f'step={step}, loss={loss:.4f}, bpb={bpb:.4f}, paloma_bpb={paloma:.4f}')
"
```

---

## 4. Check for Bugs

### Bug 1: Wrong token count (CRITICAL)

**Symptoms**: Mixin data fraction is unexpectedly high (>30%). The model trains
with far too much rephraser/DCLM data relative to Nemotron.

**How to check**:
1. Look at WandB run tags — the tag `rephraser-frac=X.XXXX` shows the fraction
2. Check training logs for the "Mixin Sanity Check" output
3. Manually verify token counts (see Section 2)

**Root cause**: `_read_token_count` previously fell back to
`shard_ledger.json rows * SEQ_LEN` when `.stats.json` was missing. For
variable-length tokenizations, this massively overestimates the token count
(e.g. 2.38B instead of 362M for 150 WARCs).

**Prevention**: The `_validate_mixin_fraction` function now crashes if the fraction
exceeds `max_mixin_fraction` (default 0.30). To override intentionally, pass
`max_mixin_fraction=` to the config dataclass.

### Bug 2: Wrong cluster (training on vllm cluster)

**Symptoms**: Job submitted to `us-east1-d` lands on the vllm inference cluster
instead of the training cluster.

**How to check**: The cluster name mapping is confusing:

| To target cluster... | Use `--cluster` | Config file |
|---|---|---|
| us-central1 | `us-central1` | `marin-us-central1.yaml` |
| us-east1-d (training) | **`us-east1`** | `marin-us-east1.yaml` |
| us-east1-d-vllm (inference) | `us-east1-d` | `marin-us-east1-d-vllm.yaml` |
| us-east5-a | `us-east5-a` | `marin-us-east5-a.yaml` |
| eu-west4-a | `eu-west4-a` | `marin-eu-west4-a.yaml` |

**DANGER**: `--cluster us-east1-d` resolves to the **vllm** cluster, NOT training.

### Bug 3: Missing `.stats.json` for extracted data

**Symptoms**: `_read_token_count` raises `ValueError` for NemotronCooldown paths.

**Fix**: The `extract_cooldown_data` function now writes `.stats.json` after
extraction. For older extractions, manually upload:
```bash
echo '{"total_tokens": <N>, "total_elements": 0}' | \
  gcloud storage cp - "gs://marin-{region}/tokenized/{name}/train/.stats.json"
```

---

## 5. Produce the Report

### Recommended report structure

```markdown
# Rephraser Cooldown Results — {date}

## Setup
- Model: Qwen3 1.385B (exp2166, 1e20 token budget)
- Pre-cooldown checkpoint: step 35,000
- Cooldown: {steps} steps, {budget} tokens
- Optimizer: CautiousConfig, linear LR decay

## Baselines
| Checkpoint | Description |
| step-35000 | Pre-cooldown (before any cooldown) |
| step-44758 | Post-cooldown (original nemotron-only) |

## Token Counts
| Condition | Nemotron tokens | Mixin tokens | Mixin % | Total |
| ... | ... | ... | ... | ... |

## Validation Perplexity
(from eval_metrics.jsonl)

## lm-eval Results (CORE_TASKS)
| Task | Metric | Baseline-35k | Nemotron-44k | Rephraser | DCLM | ... |
| hellaswag_0shot | acc_norm | ... | ... | ... | ... | ... |
| arc_challenge | acc_norm | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... |

## Notes / Known Issues
- Any bugged runs, re-runs, etc.
```

### Step-by-step data gathering

1. **List all GCS runs**:
   ```bash
   for region in us-central1 us-east1 eu-west4; do
     echo "=== $region ==="
     gcloud storage ls gs://marin-$region/ | grep -E "(cooldown|rephraser)" | grep -v tokenized
   done
   ```

2. **Get token counts** for each run's tokenized data (see Section 2).

3. **Get validation perplexity** from `eval_metrics.jsonl` (see Source 3).

4. **Get lm-eval from WandB** for in-training evals (see Source 1).

5. **Get lm-eval from GCS** for standalone evals (see Source 2).

6. **Run the validation checks** (see Section 4).

### CORE_TASKS reference

From `experiments/evals/task_configs.py`:

| Task | Shots | choice_prob_norm available? | Fallback metric |
|---|---|---|---|
| agieval_lsat_ar | 3 | NO | acc_norm |
| arc_easy | 10 | YES | acc_norm |
| arc_challenge | 10 | YES | acc_norm |
| boolq | 10 | YES | acc |
| commonsense_qa | 10 | YES | acc |
| copa | 0 | YES | acc |
| hellaswag_0shot | 0 | YES | acc_norm |
| hellaswag_10shot | 10 | YES | acc_norm |
| lambada_openai | 0 | NO | acc |
| openbookqa | 0 | YES | acc_norm |
| piqa | 10 | YES | acc_norm |
| wsc273 | 0 | YES | acc |
| winogrande | 0 | YES | acc |

### Metrics (IMPORTANT — read this)

**Primary metric**: `choice_prob_norm`. This is the metric we care about most.

**`choice_prob_norm` is NOT the same as `acc_norm`!**
- `acc_norm` = length-normalized accuracy (normalizes by number of tokens in each choice)
- `choice_prob_norm` = probability-normalized accuracy (normalizes by the probability of each choice)

In GCS `results.json`, the key is `choice_prob_norm,none`.
In WandB, the key is `lm_eval/{task}/choice_prob_norm`.

`choice_prob_norm` is available for 11 of 13 CORE_TASKS. The two exceptions:
- `agieval_lsat_ar` — use `acc_norm` instead
- `lambada_openai` — use `acc` instead

When producing the report, always create a dedicated `choice_prob_norm` table as the PRIMARY table, with `acc`/`acc_norm` tables as secondary reference.

---

## Exact Data Source Paths (Verified 2026-03-04)

These are the exact paths to retrieve every number in the COOLDOWN_REPORT.

### GCS Standalone Eval Results (results.json)

| Condition | Full GCS Path |
|---|---|
| Pre-cooldown (step-35000) | `gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/lmeval_debug_hf/scaling-1e20-step-35000-662640/results.json` |
| Post-cooldown (step-44758) | `gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/lmeval_debug_hf_step-44758-8a20e0/results.json` |
| DCLM Filtered (step-9758) | `gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/lmeval_debug_hf_step-9758-c460fa/results.json` |
| Short Rephraser v2 (step-4999) | `gs://marin-us-central1/evaluation/lm_evaluation_harness_levanter/lmeval_debug_hf_step-4999-8b06e2/results.json` |

**Note**: The pre-cooldown eval is in a nested subdirectory (`lmeval_debug_hf/scaling-1e20-step-35000-662640/`) because the HF export step name contains a slash.

### WandB In-Training Eval Results

| Condition | WandB Run Name (display_name filter) | Rephraser Data | Metric Key Pattern |
|---|---|---|---|
| Rephraser v2 (25W) | `cooldown-rephraser-d7d976d3-v2-1cdc5a` | `0b5b27` (46M) | `lm_eval/{task}/choice_prob_norm` |
| DCLM 300M v2 | `cooldown-dclm-100m-v2-32bcbc` | N/A | `lm_eval/{task}/choice_prob_norm` |
| Rephraser 150W v2 | `cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b` | `02c17e` (362M) | `lm_eval/{task}/choice_prob_norm` |
| Short DCLM | `short-cooldown-dclm-49e19a` | N/A | `lm_eval/{task}/choice_prob_norm` |
| Short Nemotron-only | `short-cooldown-nemotron-only-391eeb` | None | `lm_eval/{task}/choice_prob_norm` |

**WandB project**: `marin-community/marin`

### Validation Perplexity (eval_metrics.jsonl)

| Condition | Full GCS Path |
|---|---|
| Rephraser v2 (25W) | `gs://marin-us-central1/cooldown-rephraser-d7d976d3-v2-1cdc5a/checkpoints/eval_metrics.jsonl` |
| DCLM 300M v2 | `gs://marin-us-central1/cooldown-dclm-100m-v2-32bcbc/checkpoints/eval_metrics.jsonl` |
| DCLM Filtered | `gs://marin-us-central1/cooldown-dclm-filtered-v1-54cdb7/checkpoints/eval_metrics.jsonl` |
| Rephraser 150W v2 | `gs://marin-us-central1/cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b/checkpoints/eval_metrics.jsonl` |
| Short Rephraser 150W | `gs://marin-us-east1/short-cooldown-rephraser-d7d976d3-v2-eebec0/checkpoints/eval_metrics.jsonl` |
| Short DCLM | `gs://marin-us-central1/short-cooldown-dclm-49e19a/checkpoints/eval_metrics.jsonl` |
| Short Nemotron-only | `gs://marin-us-central1/short-cooldown-nemotron-only-391eeb/checkpoints/eval_metrics.jsonl` |

---

## 6. Updating the Living Report

The living report is at `experiments/rephraser/COOLDOWN_REPORT.md`. When updating:

1. Re-run the data gathering steps above for any new experiments.
2. Add new rows to the token count and lm-eval tables.
3. Note any bugged runs in the "Known Issues" section.
4. Update the "Last updated" date at the top.
