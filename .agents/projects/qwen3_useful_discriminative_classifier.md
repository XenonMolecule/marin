# Qwen3-0.6B discriminative useful-vs-NO_USEFUL classifier

**Goal:** Reframe the generative high_quality web-extraction distillation as a *discriminative*
binary classifier. Take stock **`Qwen/Qwen3-0.6B`** (NOT the distilled ckpt — base, per user),
add a 2-way classification head, train it to predict keep (`useful`) vs discard
(`[NO_USEFUL_CONTENT]`). Hypothesis (user's): beats the ModernBERT classifier (F1≈0.70).

Branch: `qwen3-useful-classifier`. Owner left it to the agent overnight (2026-06-25). Deliver a
**W&B link** showing training/trained model by morning. No shortcuts.

## Design decisions (locked with user)
- **Input format: FULL teacher chat incl. spec** (user chose). Per example feed exactly what the
  teacher saw, ending at the generative decision point:
  `<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n[[ ## html ## ]]\n{HTML}\n\n[[ ## extraction_spec ## ]]\n{SPEC}\n\nRespond...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n[[ ## text ## ]]\n[`
  Pool the **last token** = the trailing `[` (id 58) — under a causal mask it has attended to the
  whole doc, and its next-token prediction *is* keep/discard.
- **Head warm-start:** classifier row for label 0 (=no_useful/discard) initialized to the model's
  unembedding row for token **`NO` = 8996** (the *distinctive* token; first token `[`=58 is
  generic — verified). So at init, discard-logit == the model's own logit for emitting "NO" at the
  decision position. label 1 (=useful) row = 0. Ablation: `--no-warm-head` (random head).
- **Backbone init:** base `Qwen/Qwen3-0.6B` (default). Ablation arm later: distilled ckpt.
- **Label convention (reuse ModernBERT's):** `__label__useful`→1 (positive, KEEP), else→0.
  F1 reported on the `useful` class, directly comparable to ModernBERT's 0.70.
- **Data (reuse ModernBERT's exact shards, us-east5, in-region):**
  - train `gs://marin-us-east5/classifiers/useful_fasttext/presharded_survivor_1M_w4/train_shard_*.txt.gz`
  - test  `gs://marin-us-east5/classifiers/useful_fasttext/full_prep_body_strip/test/*.txt.gz` (frozen ~7000, natural ratio)
  - text is `body_strip(raw_html)`, newline-flattened by the fastText prep — minor fidelity cost vs
    teacher (acceptable; gains direct comparability).

## Tokenizer facts (verified)
- `[NO_USEFUL_CONTENT]` = `[`(58) `NO`(8996) `_USE`(22295) `FUL`(49636) `_CONTENT`(25560) `]`(60).
- markers: `<|im_start|>`=151644 `<|im_end|>`=151645 `<think>`=151667 `</think>`=151668; pad=`<|endoftext|>`=151643.
- Fixed prompt overhead with spec = **1804 tok** (head 122 + tail 1682). Pos must be ≥ ~1810; smoke uses 4096, real uses 8192.

## Files
- `lib/levanter/src/levanter/models/qwen.py` — NEW `Qwen3ForSequenceClassification` (causal, last-token pool, warm-start head).
- `lib/levanter/src/levanter/main/train_decoder_classifier.py` — NEW entrypoint (prompt-wrapping dataset, causal score, build+train+F1 sweep+save). Reuses `read_fasttext_shards`/`read_frozen_eval`/`f1_sweep`/`_expand_globs`/`ClassificationExample` from `train_classifier.py`/`modernbert.py`.
- `lib/marin/src/marin/training/training.py` — NEW `run_levanter_train_decoder_classifier` (reuses generic `TrainClassifierOnPodConfig`; only the entrypoint module differs).
- `experiments/baseline_collection/exp_qwen3_0_6b_useful_classifier.py` — launcher (mirrors `launch_modernbert_levanter.py`).
- `lib/levanter/tests/test_qwen3_classifier.py` — local CPU unit test (tiny config) for pool/loss/warm-start before TPU.

## Plan / status
1. [ ] Write model + entrypoint + marin fn + launcher.
2. [ ] Local CPU unit test (tiny Qwen3): forward shape, pool index == trailing `[`, warm-start row == emb[8996], loss runs.
3. [ ] TPU smoke (v6e-4 us-east5, Pos=4096, 1024 rows, warm, base) end-to-end → F1 logged.
4. [ ] Launch real run(s): Pos=8192, ~200k rows, focused LR set. W&B project `qwen3-useful`.
5. [ ] Babysit (wakeups), fix issues autonomously, expand sweep, report W&B link.

## Launch recipe (CPU coordinator submits the TPU job, blocks to keep children alive)
```
uv run iris --cluster marin job run --region us-east5 --cpu 2 --memory 8GB --disk 10GB \
  --priority interactive --no-wait --extra cpu --enable-extra-resources \
  -e WANDB_API_KEY ***REMOVED-WANDB-KEY*** \
  -e HF_TOKEN ***REMOVED-HF-TOKEN*** \
  --job-name qwen3clf-<id> \
  -- python -m experiments.baseline_collection.exp_qwen3_0_6b_useful_classifier --run-id <id> [--smoke]
```
