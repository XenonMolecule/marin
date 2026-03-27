# Qwen3 0.6B Mathhelpforum SFT — Experiment Guide

A walkthrough of the files, scripts, and lessons learned from running the Qwen3 0.6B mathhelpforum SFT sweep.

## Overview

This experiment fine-tunes Qwen3 0.6B Base on different representations of math data from mathhelpforum.com, then evaluates on GSM8K CoT and MATH (hendrycks). There are 6 conditions (baseline + 5 SFT variants), all defined in a single DAG file. The data pipelines were already built by earlier experiments — this sweep reuses their cached outputs and only re-tokenizes with the Qwen3 tokenizer.

## File Map

### The experiment itself

| File | Role |
|------|------|
| [`qwen3_mathhelpforum_sft.py`](qwen3_mathhelpforum_sft.py) | **Main experiment.** Defines all 6 conditions as parallel DAG branches. This is the only file you launch. |
| [`QWEN3_MATHHELPFORUM_SFT_REPORT.md`](QWEN3_MATHHELPFORUM_SFT_REPORT.md) | Results report with tables, analysis, and sample model outputs. |

### Data source scripts (imported, not launched directly)

These scripts define `ExecutorStep` objects that the main experiment imports. Because their outputs already exist on GCS, the executor skips them — they just provide the DAG edges so the tokenization and training steps know where to find their input data.

| File | What it provides | Imported as |
|------|-----------------|-------------|
| [`mathhelpforum_qra_sft.py`](mathhelpforum_qra_sft.py) | Q/R/A extraction → chat messages JSONL | `chat_transform_step` |
| [`mathhelpforum_qra_sft_plaintext.py`](mathhelpforum_qra_sft_plaintext.py) | Q/R/A extraction → plain text JSONL | `plaintext_transform_step` (as `qra_plaintext_step`) |
| [`mathhelpforum_resiliparse_sft.py`](mathhelpforum_resiliparse_sft.py) | HTML → resiliparse plain text | `extract_text_step` |
| [`gsm8k_sft_plaintext.py`](gsm8k_sft_plaintext.py) | GSM8K train → plain text Q/A JSONL | `plaintext_transform_step` (as `gsm8k_plaintext_step`) |

### Model and template configuration

| File | Role |
|------|------|
| [`experiments/qwen3.py`](../qwen3.py) | Defines `qwen3_0_6b_hd128` model config (architecture dimensions, head_dim=128) |
| [`experiments/chat_templates/qwen3_chat_template.py`](../chat_templates/qwen3_chat_template.py) | `QWEN_3_CHAT_TEMPLATE` — Jinja template with `<\|im_start\|>`/`<\|im_end\|>` tokens |

### Shared helpers

| File | What's used |
|------|-------------|
| [`experiments/defaults.py`](../defaults.py) | `default_tokenize()` — standard tokenization step builder |
| [`experiments/evals/evals.py`](../evals/evals.py) | `evaluate_lm_evaluation_harness()` — creates vLLM eval steps |
| [`experiments/posttrain/instruction_datasets.py`](../posttrain/instruction_datasets.py) | `get_instruction_dataset()` — downloads HF datasets as chat JSONL |
| [`rephraser_cooldown.py`](rephraser_cooldown.py) | `_read_token_count()` — reads token count from Levanter cache metadata |

### Infrastructure (you don't edit these, but you use them)

| File | Role |
|------|------|
| `lib/marin/src/marin/run/ray_run.py` | Job submission to Ray cluster |
| `scripts/ray/cluster.py` | Cluster management (dashboard, job-logs, list-jobs, stop-job) |
| `lib/marin/src/marin/training/training.py` | `TrainLmOnPodConfig`, `run_levanter_train_lm` — training orchestration |
| `lib/marin/src/marin/execution/executor.py` | `ExecutorStep`, `executor_main`, `output_path_of`, `this_output_path` — DAG execution |
| `lib/marin/src/marin/processing/tokenize/tokenize.py` | `lm_data_config()` — wraps tokenized data into Levanter dataset configs |
| `lib/marin/src/marin/evaluation/evaluation_config.py` | `EvalTaskConfig` dataclass |
| `lib/levanter/src/levanter/eval_harness.py` | Levanter's lm-eval integration (modified to fix `max_gen_toks` default) |

### Previous experiment (for comparison)

| File | Role |
|------|------|
| [`MATHHELPFORUM_SFT_REPORT.md`](MATHHELPFORUM_SFT_REPORT.md) | Original report using ~1.385B model with Llama3 tokenizer (~2% GSM8K across all conditions) |

## How to Run

### Dry run (local, no cluster)

```bash
MARIN_PREFIX=gs://marin-us-central1 uv run python \
    experiments/rephraser/qwen3_mathhelpforum_sft.py --dry_run true
```

This validates the DAG structure, imports, and config without submitting anything. Note: may fail with SSL errors on some local setups; if so, verify syntax with `python -c "import ast; ast.parse(open('experiments/rephraser/qwen3_mathhelpforum_sft.py').read())"` as a fallback.

### Submit to cluster

```bash
uv run lib/marin/src/marin/run/ray_run.py \
    --cluster us-central1 --no_wait \
    -e WANDB_API_KEY $WANDB_API_KEY \
    -e HF_TOKEN $HF_TOKEN \
    -- python experiments/rephraser/qwen3_mathhelpforum_sft.py
```

### Monitor

```bash
# Check job status
uv run scripts/ray/cluster.py --cluster us-central1 list-jobs

# Tail job logs (most reliable way to check if a job is alive)
uv run scripts/ray/cluster.py --cluster us-central1 job-logs <job-id>

# Check executor status for a specific step
gcloud storage cat gs://marin-us-central1/<step-path>/.executor_status

# Check eval results
gcloud storage cat gs://marin-us-central1/evaluation/lm_evaluation_harness/<eval-hash>/<task>/*/results.json
```

## How the DAG Works

```
                    ┌─ baseline_eval (vLLM, no training)
                    │
gsm8k_train ───────┼─ gsm8k_chat_tokenized ── gsm8k_chat_train ── gsm8k_chat_eval
                    │
gsm8k_plaintext ───┼─ gsm8k_plain_tokenized ── gsm8k_plain_train ── gsm8k_plain_eval
                    │
qra_chat_data ─────┼─ qra_chat_tokenized ── qra_chat_train ── qra_chat_eval
                    │
qra_plain_data ────┼─ qra_plain_tokenized ── qra_plain_train ── qra_plain_eval
                    │
resiliparse_data ──┴─ resiliparse_tokenized ── resiliparse_train ── resiliparse_eval
```

All 6 branches are independent. The executor discovers upstream dependencies automatically and runs branches in parallel. Steps whose outputs already exist on GCS (all data source steps) are skipped.

## Gotchas and Lessons Learned

### 1. Use vLLM evaluator for generation tasks, NOT Levanter

The Levanter evaluator (`evaluate_levanter_lm_evaluation_harness` / `default_eval`) is designed for **loglikelihood-based tasks** (MMLU, perplexity). For **generation tasks** like GSM8K CoT and MATH, use `evaluate_lm_evaluation_harness` (vLLM). The Levanter evaluator also has persistent TPU scheduling issues ("No accelerator found") because it requires JAX to detect the TPU at `TrainerConfig` init time.

SFT checkpoints are accessible to vLLM via the HF export: `output_path_of(train_step, "hf")`. This works because the training config includes `hf_save_steps=num_train_steps`, which saves a HuggingFace-format checkpoint at the end of training.

There's a comment in the experiment file (lines 98-104) explaining this.

### 2. Executor hash caching can serve stale results

The executor uses a content hash of each step's config to determine its output path. If you change eval parameters (e.g., 0-shot → 4-shot, or change `max_gen_toks`) but not the step name/config, the hash stays the same and the executor skips the step because it sees `SUCCESS` on GCS.

**Fix:** Change the `model_name` parameter (e.g., add a `-4shot` or `-vllm` suffix) to force a new hash. Or manually reset the `.executor_status` file on GCS:
```bash
echo "PENDING" | gcloud storage cp - gs://marin-us-central1/<path>/.executor_status
```

### 3. `max_gen_toks` default was 256 in Levanter eval harness

The Levanter eval harness (`eval_harness.py`) had a hardcoded default of `max_gen_toks=256`, truncating chain-of-thought reasoning before the model could produce the `#### <number>` answer. This was fixed to 1024 across 5 locations in `eval_harness.py`. The vLLM evaluator accepts `max_gen_toks` via `engine_kwargs`:
```python
GSM8K_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 1024}
```

### 4. `list-jobs` only returns a subset of jobs

The Ray `list-jobs` API is paginated/limited to ~9-10 jobs. A job not appearing in `list-jobs` does NOT mean it's dead. Always verify via `job-logs` — if logs show recent activity, the job is alive.

### 5. TPU device contention with vLLM Docker

vLLM evals launch Docker containers that need exclusive TPU access (`/dev/vfio/1`). If another process (another eval, a Levanter training job, or even a stale container) is using the TPU on that node, the vLLM container crashes with `Device or resource busy`. These evals may retry automatically, but can get stuck. Check the Ray dashboard for node-level resource usage if evals seem stuck.

### 6. Chat template eval destroys MATH scores

The Q/R/A chat SFT condition trained with chat template and evaluated with `apply_chat_template=True`. This wraps the few-shot prompt in `<|im_start|>`/`<|im_end|>` tokens, which confuses the answer extraction. MATH scores dropped to ~0% even though the model was producing correct reasoning (visible in the flexible-match gap: 6.90% strict vs 34.04% flexible on GSM8K). The model also learned to generate `<think>` tags and forum-style prose from the training data, making extraction even harder.

### 7. Resiliparse SFT is surprisingly competitive on MATH

Unstructured continued pretraining on resiliparse-extracted text (no Q&A formatting at all) achieved **10.07% avg MATH** — better than Q/R/A plaintext SFT (9.06%) on 6 of 7 MATH subtopics. Q/R/A still wins on GSM8K by ~7pp. This suggests that for MATH-style problems, exposure to diverse mathematical text may matter more than structured Q&A formatting.

### 8. `pad_tokenizer_to_match_model=True` is needed for Qwen3

Qwen3 pads its vocab to multiples of 4 for TPU efficiency. Without `pad_tokenizer_to_match_model=True` in the training config, there's a mismatch between the tokenizer vocab size and the model's embedding table. This flag is set in `_run_single_epoch_sft()`.

## How the Report Was Written and Updated

The report ([`QWEN3_MATHHELPFORUM_SFT_REPORT.md`](QWEN3_MATHHELPFORUM_SFT_REPORT.md)) was written iteratively as results came in:

1. **Initial draft** — Created when the experiment was planned, with experiment descriptions, model details, and empty results tables.

2. **First results (Levanter evaluator)** — Filled in GSM8K plaintext SFT, Q/R/A chat SFT, and Q/R/A plaintext SFT results from the Levanter evaluator. Discovered the `max_gen_toks=256` bug during this phase.

3. **4-shot MATH addition** — Originally only evaluated GSM8K. Added MATH subtopics (4-shot) after realizing 0-shot MATH gave near-zero scores. Had to force new executor hashes to get fresh results.

4. **Switched to vLLM evaluator** — After persistent Levanter eval failures ("No accelerator found" on TPU), switched all SFT evals to use `evaluate_lm_evaluation_harness` (vLLM). Added a separate "vLLM evaluator results (canonical)" table, keeping Levanter results as "preliminary" for reference.

5. **Baseline results** — The baseline eval (no SFT) was the first vLLM eval to complete. Added baseline numbers to all tables.

6. **Sample outputs** — Added sample model outputs from multiple conditions (baseline, GSM8K plaintext SFT, Q/R/A plaintext SFT, Q/R/A chat SFT) showing both correct and incorrect examples on GSM8K and MATH. Included a cross-condition comparison table on the same problem.

7. **vLLM results filling in** — Q/R/A plaintext and Resiliparse vLLM results completed and were added to the canonical table. GSM8K plaintext and Q/R/A chat still running at time of writing.

The report uses two parallel results tables (vLLM canonical + Levanter preliminary) so readers can see both. The analysis section references whichever numbers are available, with notes about which evaluator produced them.

## GCS Paths

Eval results live at:
```
gs://marin-us-central1/evaluation/lm_evaluation_harness/<model-name>-<hash>/
    <task_alias>/
        <model_path_escaped>/
            results.json        # aggregate metrics
            samples_*.jsonl     # per-example predictions
```

Training checkpoints:
```
gs://marin-us-central1/checkpoints/<name>-<hash>/
    hf/step-<N>/    # HF-format export (used by vLLM evaluator)
```

Tokenized data:
```
gs://marin-us-central1/tokenized/<name>-<hash>/
    train/          # Levanter cache format
```
