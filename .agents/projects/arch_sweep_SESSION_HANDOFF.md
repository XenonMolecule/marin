# Session handoff — arch sweep / new_classifiers (2026-08-11)

Monitors are LOCAL to a Claude session and die with it. **Cluster jobs keep running.** This file
lists what is live on the cluster and what to re-arm after restarting.

Full project log: `.agents/projects/arch_sweep_faster_than_modernbert.md`
Published report: https://claude.ai/code/artifact/e7f27787-065e-409e-ac83-6b08bf8ab719

## 1. LIVE ON THE CLUSTER (survives the restart — do NOT relaunch these)

| job | what | state at handoff |
|---|---|---|
| `/michaelryan/mb-clf-lpv11-ettin68-10M-coord` | ettin68 @10M, the last sweep result | **running, step 37,718/39,062 (97%)**, loss 0.230 — ~1,300 steps left |
| `/michaelryan/build-clfcache-90m` | 90M TreeCache, us-central2, ModernBERT tok @8192 | building (no ledger yet — early); target `…/presharded_survivor_w640_90M/_clf_token_cache`, expect ~680 GB |

Nothing else of mine is running. Everything else finished.

## 2. MONITORS TO RE-ARM (all were local; none survive)

**(a) ettin68-10M babysitter + completion** — REQUIRED while it finishes. Checkpointer is `keep:[]`
so ONLY temp checkpoints exist, under a ttl=14d path; the babysitter copies the newest to a durable
`backup/`. NOTE the temp path is NOT under the output path — resolve it with
`marin.training.training.temporary_checkpoint_base_path(output_path)`. For this run it is:
`gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-ettin68-10M/checkpoints`
→ backs up to `gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-ettin68-10M/backup/`.
Poll ~90 min; report when wandb state is `finished` AND `eval/best_f1` is set.
Last backup taken: **step-37569**.

**(b) 90M cache build progress + completion** — poll the ledger every ~30 min:
`gs://marin-us-central2/classifiers/useful_fasttext_lpv11/presharded_survivor_w640_90M/_clf_token_cache/shard_ledger.json`
→ report `is_finished`, `len(finished_shards)`/96, `total_num_rows`; also watch for the job going
`failed`/`killed` via `iris job list --prefix /michaelryan/build-clfcache-90m`.

## 3. NEXT ACTION when the 90M cache finishes

Launch pooled at 90M **in us-central2 on v4 = zero egress** (pre-flighted, config builds clean):

```bash
uv run iris --cluster marin job run --region us-central2 \
  --cpu 4 --memory 8GB --disk 10GB --enable-extra-resources \
  --priority interactive --no-wait --job-name mb-clf-lpv11-pooled-90M-coord \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.baseline_collection.launch_modernbert_levanter \
     --run-id mb-clf-lpv11-pooled-90M --model-size pooled \
     --train-glob 'gs://marin-us-central2/classifiers/useful_fasttext_lpv11/presharded_survivor_w640_90M/train_shard_*.txt.gz' \
     --test-glob  'gs://marin-us-central2/classifiers/useful_fasttext_lpv11/full_prep_body_strip_test7k/*.txt.gz' \
     --cache-dir  'gs://marin-us-central2/classifiers/useful_fasttext_lpv11/presharded_survivor_w640_90M/_clf_token_cache' \
     --train-rows 88839133 --batch-size 256 --epochs 1 --lr 3e-4 \
     --tpu-type v4-8 --region us-central2
```
- Do **NOT** pass `-e MARIN_PREFIX` for us-central2 (the repo `.env` already pins it there; the
  override is only needed for us-east5 runs).
- 347,027 steps ≈ **45 h** at the measured 471 ms/step → **arm a babysitter from the start**.
- `per_device_parallelism=-1` is auto-applied by the preset and is MANDATORY (raw-array params +
  gradient accumulation = opaque TPU SIGSEGV).
- Optional: also run `--model-size pooled_big` (costs only 14% throughput; the +0.005 at 10M was
  inside the noise floor, so 90M may finally separate them).

## 4. Deferred / open

- **bert (MiniLM-L6)** — implemented + passed the speed gate, never trained; needs its own
  MiniLM-tokenizer cache (`build-clfcache-minilm` was stopped). Expected to land inside the
  statistical tie band, so low priority.
- **After pooled-90M**: re-run `score_pooled_frozen_eval.py` + `cascade_analysis.py` against the new
  checkpoint to refresh the cascade numbers (commands in the project log).
