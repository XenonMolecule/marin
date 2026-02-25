# Rephraser Spec Sweep

Compare M extraction specs by midtraining Llama-3.2-1B on each spec's extracted text and evaluating on CORE_TASKS.

## Pipeline

```
warc_paths.txt          SPECS list
     |                     |
     v                     v
 [1] download_warcs    (per-spec loop)
     |                     |
     v                     v
 [2] inference_spec_{hash}     (TPU v5p-8, vLLM)
     |
     v
 [3] postprocess_spec_{hash}   (CPU: strip <think>, filter)
     |
     v
 [4] tokenize_spec_{hash}      (CPU)
     |
     v
 [5] midtrain_spec_{hash}      (TPU v5p-8, CORE_TASKS eval)
```

Each spec flows independently through steps 2-5. Adding/removing a spec never invalidates others.

## Configuration

Edit `rephraser_sweep.py`:
- `REPHRASER_MODEL`: GCS path to your trained rephraser HF export
- `SPECS`: list of extraction prompt strings
- `warc_paths.txt`: one S3-style Common Crawl path per line

## Running

### Dry run (verify DAG locally)

```bash
MARIN_PREFIX="gs://marin-us-central1/experiments" \
    .venv/bin/python experiments/rephraser/rephraser_sweep.py --dry_run true
```

### Smoke test (1 spec, 2 WARCs)

```bash
uv run lib/marin/src/marin/run/ray_run.py \
    --cluster us-central1 --no_wait \
    --env_vars WANDB_API_KEY=$WANDB_API_KEY \
    -- python experiments/rephraser/rephraser_sweep.py
```

### Full run (M specs)

```bash
uv run lib/marin/src/marin/run/ray_run.py \
    --cluster us-central1 --no_wait \
    --env_vars WANDB_API_KEY=$WANDB_API_KEY \
    -- python experiments/rephraser/rephraser_sweep.py --max_concurrent 5
```

## Monitoring

- **Ray dashboard**: `uv run scripts/ray/cluster.py dashboard`
- **W&B**: filter by tag `rephraser-sweep`, compare `lm_eval/hellaswag_0shot/acc_norm`

## Debugging

### WARC download failures

Check `raw/commoncrawl/rephraser_sweep_batch0/` for output JSONL files. Each record should have `id`, `html`, `url`, `metadata` fields. Common issues:
- HTTP 503 from Common Crawl: retry or use different WARC paths
- Empty HTML: the WARC record had no HTML content-type response

### Inference OOM

If vLLM runs out of memory on v5p-8:
- Reduce `batch_size` in the inference step (currently 64)
- Reduce `max_model_len` if the model supports shorter contexts
- Check that `max_doc_tokens` (28672) doesn't exceed model capacity

### Post-processing stats

Check `processed/rephraser_spec_{hash}/postprocess_stats.json` for filter counts. If most records are filtered:
- `[NO_USEFUL_CONTENT]` filtering: the spec may not be producing useful extractions
- `min_output_chars` (50): lower this if the spec produces terse but valid output

### Tokenization issues

Check `tokenized/rephraser_spec_{hash}/` for Levanter cache files. Verify the tokenizer matches `meta-llama/Llama-3.2-1B`.

### Training divergence

If midtraining loss doesn't decrease or diverges:
- Lower `MIDTRAIN_LR` from 3e-4 to 3e-5
- Check that the tokenized data actually has content (not all filtered out)
- Verify the token estimate and `num_train_steps` are reasonable

## Adding more WARCs

Add more paths to `warc_paths.txt`. The executor will re-run inference/tokenize/train for all specs with the updated data. For batch extensibility (keeping old downloads cached), split into multiple download steps — see the plan file for the pattern.

## Scaling to M > 10 specs

The current per-spec design runs independent vLLM instances per spec. At M > 10, consider switching to a shared vLLM server for prefix caching (see `TODO(shared-vllm)` comments in the code). This avoids redundant prefill computation across specs.

## Hyperparameters

Adapted from DCLM 1B/1x (`experiments/tutorials/exp1077_reproduce_dclm_1b1x.py`):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Llama-3.2-1B | Matches DCLM 1B experiment family |
| Seq len | 4096 | Llama-3.2-1B native context |
| Batch size | 256 | Matches DCLM 1B/1x |
| Learning rate | 3e-4 | ~10x lower than from-scratch 3e-3 |
| Weight decay | 0.033 | Matches DCLM 1B/1x |
| z_loss | 1e-4 | Matches DCLM 1B/1x |
