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

| Experiment | Script | Budget |
|---|---|---|
| Rephraser v2 (25 WARCs) | `rephraser_cooldown.py` | 2.56B tokens (9,759 steps) |
| Rephraser 150 WARCs | `rephraser_cooldown_150warc_train*.py` | 2.56B tokens (9,759 steps) |
| DCLM 100M baseline | `dclm_cooldown.py` | 2.56B tokens (9,759 steps) |
| DCLM filtered | `dclm_filtered_cooldown.py` | 2.56B tokens (9,759 steps) |
| Short rephraser | `short_cooldown/rephraser.py` | 1.31B tokens (5,000 steps) |
| Short DCLM | `short_cooldown/dclm.py` | 1.31B tokens (5,000 steps) |
| Short nemotron-only | `short_cooldown/nemotron_only.py` | 1.31B tokens (5,000 steps) |
| Baseline pre-cooldown | `scaling_1e20_baseline_eval.py` | Eval only (step 35,000) |
| Baseline post-cooldown | `scaling_1e20_final_eval.py` | Eval only (step 44,758) |

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
gs://marin-us-central1/cooldown-dclm-100m-{hash}/                    # DCLM 100M
gs://marin-us-central1/cooldown-dclm-filtered-v1-{hash}/             # DCLM filtered
gs://marin-us-central1/short-cooldown-rephraser-d7d976d3-{hash}/     # Short rephraser
gs://marin-us-central1/short-cooldown-dclm-{hash}/                   # Short DCLM
gs://marin-us-central1/short-cooldown-nemotron-only-{hash}/          # Short nem-only
```

**Note**: Multiple hashes may exist for the same experiment if it was re-run (e.g. after a bug fix). Check `.executor_status` to find successful runs.

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

| Dataset | Path suffix | Tokens |
|---|---|---|
| Nemotron cooldown (2.56B) | `nemotron_cooldown_1e20-666089` | 2,558,263,296 |
| Nemotron short (1B) | `nemotron_cooldown_1e20_short_1b-413400` | 1,000,079,360 |
| Rephraser (25 WARCs) | `rephraser_spec_d7d976d3_cooldown-02c17e` | 362,242,311 |
| DCLM 100M baseline | `dclm_baseline_100m-211afe` | 307,051,596 |

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

Example metrics:
```
lm_eval/hellaswag_0shot/acc_norm
lm_eval/arc_challenge/acc_norm
lm_eval/piqa/acc_norm
lm_eval/winogrande/acc
lm_eval/lambada_openai/acc
```

**Primary metric**: `acc_norm` (choice_prob_norm) where available, `acc` otherwise.

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

**Reading results**:
```bash
gcloud storage cat "gs://marin-us-central1/evaluation/.../results.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for task, metrics in d['results'].items():
    acc = metrics.get('acc,none', 'N/A')
    acc_norm = metrics.get('acc_norm,none', 'N/A')
    print(f'{task}: acc={acc}, acc_norm={acc_norm}')
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

| Task | Shots | Metric to report |
|---|---|---|
| agieval_lsat_ar | 3 | acc_norm |
| arc_easy | 10 | acc_norm |
| arc_challenge | 10 | acc_norm |
| boolq | 10 | acc |
| commonsense_qa | 10 | acc |
| copa | 0 | acc |
| hellaswag_0shot | 0 | acc_norm |
| hellaswag_10shot | 10 | acc_norm |
| lambada_openai | 0 | acc |
| openbookqa | 0 | acc_norm |
| piqa | 10 | acc_norm |
| wsc273 | 0 | acc |
| winogrande | 0 | acc |

**Primary metric**: `acc_norm` (choice_prob_norm) where available, `acc` otherwise.
Tasks without `acc_norm`: boolq, commonsense_qa, copa, lambada_openai, winogrande, wsc273.

---

## 6. Updating the Living Report

The living report is at `experiments/rephraser/COOLDOWN_REPORT.md`. When updating:

1. Re-run the data gathering steps above for any new experiments.
2. Add new rows to the token count and lm-eval tables.
3. Note any bugged runs in the "Known Issues" section.
4. Update the "Last updated" date at the top.
