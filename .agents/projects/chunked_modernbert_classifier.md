# Chunked-document ModernBERT useful-classifier

## The idea
Today the classifier sees only the first `ctx` tokens of a doc (ctx 512→8192) and labels the whole
doc from that prefix. The context-scaling curve shows F1 rises steeply with ctx (base @10M:
0.42/0.49/0.63/0.72 @ 1024/2048/4096/8192) → **more of the document = more signal**. So instead of
truncating: **split each doc into ctx-sized chunks, train the classifier to label each chunk with
the document's label (MIL), and at inference score every chunk and aggregate** (min/max/avg/…
decided at eval) into a doc-level keep/drop. Goal: see the WHOLE doc even at small ctx → potentially
beat the 8k-first-tokens model, cheaper. Pilot at ~200k docs (we have ctx baselines there); scale to
10M if promising.

## Motivating data fact (measured 2026-06-29, 3000-doc sample of presharded_survivor_random_par)
Body-text char length: median **32,921** (~7.3k tok), p90 81,306 (~18k tok), p99 124,950, max 151,425.
% of docs with content BEYOND the window: >512 **96.7%** · >1024 95.2% · >2048 90.6% · >4096 74.7% ·
**>8192 44.6%**. → ~45% of docs have material the 8k model NEVER saw; median doc barely fits 8192.

## Truncation status (RESOLVED — was the user's worry)
- **Raw fastText shards: effectively FULL body text.** Only cap is `html[:1_000_000]` BEFORE body_strip
  (`experiments/baseline_collection/cascade_survivor_filter.py:87`) — bites only pathological >1MB-HTML
  docs; sample shows no clustering at any cap (smooth to 151k body chars). `to_fasttext_text` default
  `max_chars=None` (full). So in-memory training reads full docs.
- **The 8192-token CACHE is the real truncation.** `ClassificationLineProcessor` caps `text[:100_000]`
  then tokenizes `truncation=True, max_length=8192` (`train_classifier.py:135-136`); cache
  `max_cache_token_len=8192`. The 10M runs (`--use-cache`) thus NEVER saw past 8192 tok.
- **Implication:** 200k pilot uses the IN-MEMORY path (`TextClassificationDataset` reads raw shards) →
  full text, chunk immediately, NO data rebuild. 10M scale-up needs a chunk-aware cache rebuild
  (store full token seq / pre-chunked, drop the 8192 cap) — the shards already have the text, so no
  re-source from high_quality_3000_distill parquet needed (that's only for the negligible >1MB tail).
- Full-text source IF ever needed: gs://marin-us-central2/datasets/high_quality_3000_distill/data/*.parquet (`raw_html`).

## DESIGN (decisions locked with user 2026-06-29)
### Training = "chunk-as-example" MIL (plain)
- Tokenize full doc → split into ctx-sized chunks → **each chunk = one training example carrying the
  doc label**. Model learns P(this chunk looks like it's from a useful doc). Reuses the ENTIRE existing
  training loop; only the dataset changes. Keeps aggregation OUT of training (sweep it at eval).
- **Cap: 16 chunks/doc for TRAINING. If a doc has >16 chunks → RANDOM-SAMPLE 16** (deterministic
  seed per doc, 1 epoch). Inference uses the FULL doc (all chunks, no cap).
- **Tiling: SWEEP BOTH** non-overlapping (stride=ctx) AND 50%-overlap (stride=ctx/2). Hyperparameter.
- **Labels: plain MIL — accept the noise** (a boilerplate chunk of a useful doc is labeled useful).
  No loss weighting for the pilot.
- (vs true bag-level MIL = forward all chunks, pool, loss on doc → bakes aggregation into training +
  heavy code → DEFERRED.)

### Eval = full doc, aggregate, sweep
- Per frozen-7k doc: tokenize full → ALL chunks (no 16 cap) → score each → aggregate → doc score.
- **Sweep aggregation × threshold:** mean, max, min, top-k mean (k=2,3), frac-of-chunks≥t. Report
  best-F1 per aggregation on DOC labels. Compare vs existing truncated-ctx baselines.

## Implementation plan
1. `ChunkedTextClassificationDataset(AsyncDataset)` in `lib/levanter/src/levanter/main/train_classifier.py`:
   - init(texts, labels, tokenizer, Pos, pad_id, max_chunks=16, stride, seed). Pre-tokenize each doc
     once (bound long docs ~ max_chunks*stride+ctx), build flat chunk index [(doc_i, start)]; if a doc
     has >max_chunks → seeded random subset. async_len = #chunks. get_batch pads each chunk to Pos +
     segment mask (reuse `_encode_one` shape). ~6GB RAM @200k — fine on TPU host; won't fit 10M → cache later.
   - stride param: `Pos.size` (non-overlap) or `Pos.size//2` (50% overlap).
2. `ClassificationDataConfig`: add `chunked: bool`, `max_chunks_per_doc: int = 16`, `chunk_overlap: float = 0.0`.
   `build_train` branches to ChunkedTextClassificationDataset when `chunked and not use_cache`.
3. Launcher `experiments/baseline_collection/launch_modernbert_levanter.py`: `--chunked`,
   `--max-chunks-per-doc 16`, `--chunk-overlap {0|0.5}`.
4. Eval: `score_texts_chunked(model, texts, tokenizer, Pos, pad)` → per-doc list of chunk probs;
   `aggregate_sweep(chunk_probs_per_doc, labels)` → best (agg, thr, F1) over the agg set. Either inline
   in train_classifier end-of-run eval (add chunked branch) OR a standalone scorer like
   `experiments/baseline_collection/score_frozen_eval.py` (which already loads hf + scores frozen-7k;
   extend with a `--chunked` agg sweep). NOTE: standalone re-score is how the operating-curve was built.

## Launch plan (200k pilot)
- Data: first 200k rows of `gs://marin-us-east5/classifiers/useful_fasttext/presharded_survivor_random_par/`
  (in-memory, full text). Eval: frozen-7k (`full_prep_body_strip/test`, 516 useful). Region us-east5.
- Matrix: base+large × ctx{512,1024,2048,4096,8192} × tiling{non-overlap, 50%-overlap} = up to 20 runs.
  **Prioritize SMALL ctx** (512/1024/2048) — that's where chunking adds most (many chunks/doc, whole-doc
  coverage). At 8192 most docs are ≤1 chunk so chunked≈truncated. Suggested FIRST batch (cheap, biggest
  signal): base+large × {512,1024,2048} × non-overlap = 6 runs; add overlap + 4096/8192 if promising.
- **Baselines to beat (existing, TRUNCATED, frozen-7k):**
  base @200k ctx 1024/2048/4096/8192 = 0.3549/0.4412/0.5456/0.6454 (no 512 baseline exists).
  base @50k ctx = 0.2376/0.3274/0.4509/0.5635. (large 200k truncated @8192 = 0.6716.)
  Full base 10M data-scaling @8192: .564/.645/.697/.7131/.7219 @50k/200k/1M/5M/10M.
  Full base 10M ctx curve @8192-data: .4212/.4940/.6288/.7219 @1024/2048/4096/8192.
- **Hypothesis:** chunked-512 (sees whole doc in pieces) >> truncated-512; chunked small-ctx approaches/
  beats truncated-8192 (~0.645 @200k) at a fraction of per-step cost.

## Open knobs (defaults chosen, revisit after pilot)
- min chunk size (drop tail <64 tok? or keep+pad) — default keep+pad.
- class balance: non-useful docs may be shorter → fewer chunks → chunk-level positive rate ≠ doc-level.
  Accept for pilot (plain MIL); measure; consider 1/n_chunks weighting later.
- eval aggregation winner → informs the deployment aggregator (min/max/avg over a doc's chunks).

## Status
- Design locked, data verified (full text in-memory). NOT YET IMPLEMENTED. Next: write
  ChunkedTextClassificationDataset + launcher flags + chunked eval, smoke-test, launch 200k pilot.
