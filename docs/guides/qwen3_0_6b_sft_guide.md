# Guide: SFT Fine-Tuning Qwen3 0.6B on Marin

This guide walks through supervised fine-tuning (SFT) of Qwen3-0.6B on the
`MichaelR207/rephraser_small_check_0211` distillation dataset using Marin's TPU
infrastructure. It covers dataset registration, model loading, training
configuration (with validation), and job submission/monitoring.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Dataset Registration](#2-dataset-registration)
3. [Experiment File](#3-experiment-file)
4. [Hyperparameter Choices](#4-hyperparameter-choices)
5. [Launch the Job](#5-launch-the-job)
6. [Monitor and Debug](#6-monitor-and-debug)
7. [Key Files Reference](#7-key-files-reference)
8. [FAQ / Gotchas](#8-faq--gotchas)

---

## 1. Pipeline Overview

```
HuggingFace Dataset  (MichaelR207/rephraser_small_check_0211, train + validation)
        |
        v
  [Transform]         get_instruction_dataset() → JSONL.GZ (OpenAI messages format)
        |
        v
  [Tokenize]          default_tokenize() + ChatLmDatasetFormat() → Levanter cache
        |
        v
  [Configure]         lm_data_config() combines train + validation sets
        |
        v
  [Train]             default_sft() → SimpleSFTConfig + Qwen3Config → Levanter train_lm
        |
        v
  [Checkpoints]       gs://marin-us-central2/checkpoints/... (Levanter + HF format)
```

Every box is an `ExecutorStep`. The executor resolves the DAG automatically --
you only need to pass the final training step to `executor_main()` and all
upstream steps (transform, tokenize) run as dependencies.

---

## 2. Dataset Registration

The dataset `MichaelR207/rephraser_small_check_0211` has been registered in
`experiments/posttrain/instruction_datasets.py`:

```python
"MichaelR207/rephraser_small_check_0211": InstructionDatasetConfig(
    hf_dataset_id="MichaelR207/rephraser_small_check_0211",
    revision="df78040",
    adapter=multi_turn_adapter(),
    metadata_columns=["warc_file", "doc_id", "spec_id", "spec"],
    name="MichaelR207/rephraser_small_check_0211",
    splits=["train"],
),
```

**Why `multi_turn_adapter()` with defaults?** The dataset uses a `messages`
column with standard `role`/`content` keys and `system`/`user`/`assistant` role
values — exactly what the default adapter expects.

### Adding a different dataset

| Your data looks like | Adapter to use |
|---|---|
| `messages` column with `[{"role":"user","content":"..."},...]` | `multi_turn_adapter()` |
| Two separate columns, e.g. `question` + `answer` | `instruction_response_adapter(instruction_column="question", response_column="answer")` |
| Messages with non-standard keys like `"from":"human"` | `multi_turn_adapter(role_key="from", user_value="human", assistant_value="gpt", content_key="value")` |

---

## 3. Experiment File

The full experiment is in `experiments/exp_qwen3_0_6b_rephraser_sft.py`.

### Key design decisions

**Model config:** Uses `qwen3_0_6b_hd128` from `experiments/qwen3.py`. The
`hd128` suffix is necessary because Qwen3-0.6B uses `head_dim=128` while the
default `hidden_dim // num_heads = 1024/16 = 64` would produce a dimension
mismatch when loading HF weights.

**Validation:** The train and validation splits are transformed and tokenized
separately, then combined via `lm_data_config()`:

```python
data_config = lm_data_config(
    training_set=train_tokenized,
    validation_sets={"rephraser_val": val_tokenized},
)
```

The validation component gets weight 0.0, meaning it is never sampled for
training but is evaluated every `steps_per_eval` steps.

**Checkpoint loading:** `initialize_from_hf="Qwen/Qwen3-0.6B"` streams weights
from HuggingFace Hub via `HFCheckpointConverter`. The
`pad_tokenizer_to_match_model=True` flag handles Qwen's vocab padding (model has
151936 rows, tokenizer has 151664 tokens).

**Loss masking:** `ChatLmDatasetFormat()` applies the Qwen3 chat template during
tokenization and produces an `assistant_mask` so loss is only computed on
assistant tokens, not user prompts or system messages.

---

## 4. Hyperparameter Choices

Based on similar runs in the repo (Qwen3 0.6B/1.7B experiments,
`exp606_sft`, `exp1880_sft_baseline`) and standard SFT practice:

| Parameter | Value | Rationale |
|---|---|---|
| **TPU** | `v4-8` (4 chips) | 0.6B fits easily in memory |
| **Batch size** | 64 | 16 per device; good gradient signal |
| **Learning rate** | `2e-5` | SFT standard for small models (repo uses 5e-6 at 8B) |
| **LR schedule** | `cosine` | Standard for fine-tuning |
| **Warmup** | 3% of steps | Consistent across all repo SFT experiments |
| **Weight decay** | 0.01 | Light regularization |
| **Grad clip** | 1.0 | Used in Qwen3 scaling experiments |
| **Epochs** | 3 | Matches repo convention (exp606, exp808, exp1880) |
| **Train steps** | 1345 | `ceil(3 * 28678 / 64)` |
| **Eval frequency** | 100 steps | ~13 evals; catches divergence early |
| **Checkpoint freq** | 250 steps | ~5 checkpoints over training |
| **pad_tokenizer** | `True` | Required for all Qwen models |

---

## 5. Launch the Job

### Prerequisites (one-time setup)

```bash
# GCP authentication
gcloud auth login
gcloud auth application-default login
gcloud config set project hai-gcp-models

# Fetch Ray auth token
make get_ray_auth_token
```

### Code deployment

**No git push or Docker rebuild is required.** `ray_run.py` uploads your local
working directory to the cluster via Ray's `runtime_env`. Just edit files locally
and submit.

### Local dry run (validate config)

```bash
uv run python experiments/exp_qwen3_0_6b_rephraser_sft.py --prefix local_store --dry_run true
```

### Submit to the TPU cluster

**You need two terminals** (per the internal dev guide):

```bash
# Terminal 1: Keep running — establishes SSH port-forwarding to the dashboard
uv run scripts/ray/cluster.py dashboard
```

```bash
# Terminal 2: Submit the job
uv run lib/marin/src/marin/run/ray_run.py \
    --no_wait \
    --env_vars WANDB_API_KEY=${WANDB_API_KEY} \
    -- python experiments/exp_qwen3_0_6b_rephraser_sft.py
```

The output will show a **Job ID** like `raysubmit_pAJM8vKfHPhiyHBa`.

---

## 6. Monitor and Debug

### Job management

```bash
# List all jobs
uv run scripts/ray/cluster.py --cluster us-central2 list-jobs

# View logs (live tail)
uv run scripts/ray/cluster.py --cluster us-central2 job-logs <job_id> --tail 100

# Wait for completion with live logs
uv run scripts/ray/cluster.py --cluster us-central2 wait-job <job_id> \
    --match --poll 5 --show-logs --tail 200

# Stop a job
uv run scripts/ray/cluster.py --cluster us-central2 stop-job <job_id>
```

### WandB

Training metrics (train loss, validation loss, learning rate, throughput) are
logged to Weights & Biases. The validation loss appears with the tag
`rephraser_val` (the key used in `validation_sets`).

### Ray Dashboard

The dashboard URL printed by Terminal 1 shows:
- Active jobs and their status
- Resource utilization (TPU chips, memory)
- Per-job logs

### Output artifacts

```
gs://marin-us-central2/checkpoints/qwen3-0.6b-rephraser-sft-<hash>/
    checkpoints/            # Levanter-format (for resuming)
        step-0/
        step-250/
        ...
    hf/                     # HuggingFace-format (for inference)
        step-250/
        step-500/
        ...
```

### Rerunning after failure

```bash
uv run lib/marin/src/marin/run/ray_run.py \
    --no_wait \
    --env_vars WANDB_API_KEY=${WANDB_API_KEY} \
    -- python experiments/exp_qwen3_0_6b_rephraser_sft.py --force_run_failed true
```

---

## 7. Key Files Reference

| File | Purpose |
|---|---|
| `experiments/exp_qwen3_0_6b_rephraser_sft.py` | **This experiment** |
| `experiments/posttrain/instruction_datasets.py` | Dataset registration |
| `experiments/qwen3.py` | Qwen3 model configs |
| `experiments/simple_sft_config.py` | `SimpleSFTConfig` dataclass |
| `experiments/defaults.py` | `default_tokenize()`, `default_sft()`, `default_train()` |
| `lib/marin/src/marin/processing/tokenize/data_configs.py` | `lm_data_config()` |
| `lib/levanter/src/levanter/models/qwen.py` | `Qwen3Config`, `Qwen3LMHeadModel` |
| `lib/levanter/src/levanter/data/text/formats.py` | `ChatLmDatasetFormat` |
| `lib/levanter/src/levanter/compat/hf_checkpoints.py` | `HFCheckpointConverter` |
| `experiments/exp1880_sft_baseline.py` | Reference: 8B SFT mixture |
| `experiments/exp1994_32b_sft.py` | Reference: 32B Qwen3 SFT |

---

## 8. FAQ / Gotchas

### Why `qwen3_0_6b_hd128` instead of `qwen3_0_6b`?

Qwen3-0.6B uses `head_dim=128`, but the default computation
`hidden_dim // num_heads = 1024/16 = 64` gives the wrong value. The `hd128`
variant sets `head_dim=128` explicitly to match the HF checkpoint. Using
`qwen3_0_6b` would cause a dimension mismatch when loading weights.

### Why `pad_tokenizer_to_match_model=True`?

Qwen pads embedding tables to sizes divisible by 4 for TPU efficiency.
Without this flag, Levanter errors on the shape mismatch between model
(151936 rows) and tokenizer (151664 tokens).

### Why `ChatLmDatasetFormat()` instead of `TextLmDatasetFormat()`?

`ChatLmDatasetFormat` applies the chat template and produces an `assistant_mask`
so loss is only computed on assistant tokens. Essential for SFT.

### Can I use a mixture of datasets?

Yes. Use `lm_mixture_data_config` instead of `lm_data_config`. See
`experiments/exp1880_sft_baseline.py` for a full example.

### Can I warmstart from a Levanter checkpoint?

Yes. Use `initialize_from_checkpoint_path` instead of `initialize_from_hf`:
```python
sft_config = SimpleSFTConfig(
    initialize_from_checkpoint_path="gs://marin-us-central2/checkpoints/.../step-5000/",
    ...
)
```
See `experiments/exp1994_32b_sft.py` for an example.
