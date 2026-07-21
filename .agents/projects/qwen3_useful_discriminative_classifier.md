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
1. [x] Write model + entrypoint + marin fn + launcher. Committed e6aa1c2ee7 on branch `qwen3-useful-classifier`.
2. [x] Local CPU unit test — 4/4 pass (pool index, warm-start row==emb[8996], finite forward/loss, dataset assembly).
3. [x] TPU smoke SUCCEEDED 2026-06-25 14:12-14:25Z (13m15s, v6e-4 us-east5, exit 0). Validated end-to-end:
       compile, 64 steps, checkpoint, F1 eval (best_f1=0.1525 — smoke-only, meaningless at 1024 rows),
       HF save to gs://marin-us-east5/checkpoints/qwen3-useful/qwen3clf-smoke-r0/hf. NOTE: initial loss=6.64
       (hot warm-start head; emb[NO] unscaled -> large init logits). Recoverable; hedged with a random-head arm.
4. [~] Real sweep SUBMITTED 2026-06-25 ~07:28 PT. 4 coordinators (Pos=8192, 100k rows, batch128, pdp2, 1 epoch,
       ~780 steps each), W&B project `qwen3-useful`:
         - /michaelryan/qwen3clf-lr5e6      (lr 5e-6, warm head)
         - /michaelryan/qwen3clf-lr1e5      (lr 1e-5, warm head)
         - /michaelryan/qwen3clf-lr2e5      (lr 2e-5, warm head)
         - /michaelryan/qwen3clf-lr1e5-rh   (lr 1e-5, RANDOM head — warm-start ablation)
       Children nested as `<coordinator>/train_decoder_classifier`. Used 100k (not 200k) to finish by morning;
       extend the winner to 200k after.
5. [~] Babysit. CHECK 1 (14:42Z, step ~14/781): ALL 4 running healthy, no OOM/NaN. pdp2 fits at 8192.
       Steady ~15 s/it (improving from compile) -> ETA ~3h (~11:00 PT / ~18:00Z). Loss @ step14:
       lr5e6=2.29, lr1e5=2.01, lr2e5=1.37 (best), lr1e5-rh=0.71 (random head ~ln2). Warm-start pulled
       6.64 -> ~2 via warmup; it's hotter early than the random-head control -> the ablation will decide
       by final F1. No action needed; let them run.
       CHECK 2 (15:20Z, step ~165/781): all 4 healthy, ~14.8 s/it, no OOM/NaN/preempt. Loss collapsed —
       warm runs ~0.25-0.45, random-head ~0.37-0.53 (warm now BELOW random in train loss; F1 decides).
       ETA ~2.5h (~17:51Z / ~10:51 PT).
       CHECK 3 (16:22Z): all 4 healthy, loss stable ~0.25-0.31. lr5e6 & lr2e5 were PREEMPTED and
       auto-resumed cleanly (step preserved ~341, elapsed reset — recovery works, no data loss); lr1e5 &
       lr1e5-rh uninterrupted at step ~420. All finish ~17:52-18:11Z (~10:50-11:10 PT).
       CHECK 4 (17:25Z): all still running, none done. Steps: lr1e5=667, lr1e5-rh=671 (finish ~17:52Z),
       lr5e6=592, lr2e5=597 (preempted earlier, finish ~18:10Z). Loss ~0.24-0.35. F1 posts only at step 781.
       CHECK 5 (17:58Z): lr1e5 (final loss 0.20) & lr1e5-rh (0.18) DONE training, running end-of-run F1 eval
       (scoring 7000 test docs, ~6-10 min). lr5e6 step725, lr2e5 step730 (~13 min to finish). No F1 logged yet.
       CHECK 6 (18:11Z): lr1e5 & lr1e5-rh still `running` the F1 eval (silent — score_texts has no per-batch
       log; ~18 min in, eval is SLOW at batch_size=8 / 8192 ctx / 7000 docs = 875 jit dispatches + per-doc
       tokenize). Not stuck (state=running, no errors). lr5e6 & lr2e5 finished training ~18:11, evals starting.
       OPTIMIZATION NOTE for later: bump score_texts batch_size (8 -> 32) to speed end-of-run eval.

## RESULTS — 100k-row sweep COMPLETE (F1 on frozen 7000-doc test; ModernBERT baseline = 0.70)
| run | head | lr | best_f1 | thresh | state |
|---|---|---|---|---|---|
| qwen3clf-lr2e5    | warm   | 2e-5 | **0.6681** | 0.50 | succeeded, HF saved ⭐ WINNER |
| qwen3clf-lr1e5    | warm   | 1e-5 | 0.6607 | 0.38 | succeeded, HF saved |
| qwen3clf-lr1e5-rh | random | 1e-5 | 0.6536 | 0.32 | succeeded, HF saved |
| qwen3clf-lr5e6    | warm   | 5e-6 | 0.6506 | 0.28 | succeeded, HF saved |

VERDICTS:
- **Warm-start head HELPS**: emb[NO] warm vs random at lr1e-5 = 0.6607 vs 0.6536 (+0.7pt). Validates the hunch.
- **Higher LR better** (warm): 5e-6=0.6506 < 1e-5=0.6607 < 2e-5=0.6681 (monotonic). 2e-5 best of swept range;
  an even higher LR (3e-5/5e-5) might help further — future probe.
- All ~0.65-0.67 at 100k rows (HALF ModernBERT's 200k), within ~0.03 of the 0.70 baseline. 200k should close it.
HF artifacts: gs://marin-us-east5/checkpoints/qwen3-useful/<run>/hf (backbone + classifier_head.npz).

## 200k HEADLINE RUN — COMPLETE (succeeded 02:12Z / 19:12 PT 2026-06-25)
- `/michaelryan/qwen3clf-lr2e5-200k` (warm head, lr 2e-5, 200k rows, 1562 steps, ~7h incl. one preempt+resume).
- **FINAL best_f1 = 0.6809 @ t=0.44** on the frozen 7000-doc test. HF saved to .../qwen3clf-lr2e5-200k/hf.
- vs 100k same config (0.6681): **doubling data = +1.3pt**. vs ModernBERT 0.70: **-0.019** (within ~2pt).

## FINAL SUMMARY (experiment complete)
- Qwen3-0.6B decoder reframed as a discriminative useful-vs-[NO_USEFUL_CONTENT] classifier WORKS: 0.681 F1
  at 200k rows, ~2pt under the ModernBERT encoder baseline (0.70), trained on identical data + frozen test.
- Warm-start head (emb[NO]=8996 unembed row) HELPS: +0.7pt vs random head (0.6607 vs 0.6536 @ lr1e-5, 100k).
- Trends both positive: higher LR better (0.651->0.661->0.668 over 5e-6/1e-5/2e-5) and more data better (+1.3pt 100k->200k),
  so neither has saturated — a higher-LR (3e-5/5e-5) and/or larger-data run would likely push past 0.70. NOT a regression.
- All code committed on branch `qwen3-useful-classifier` (Qwen3ForSequenceClassification + train_decoder_classifier.py
  + marin run fn + launcher + passing CPU unit tests). 5 W&B runs in project qwen3-useful.

## W&B (DELIVERABLE)
- Project: **https://wandb.ai/marin-community/qwen3-useful** (entity `marin-community`, user michaeljryan)
- Runs: .../runs/qwen3clf-lr5e6 | qwen3clf-lr1e5 | qwen3clf-lr2e5 | qwen3clf-lr1e5-rh
- Watch `train/loss` curve + final `eval/best_f1` (summary) vs ModernBERT **0.70**.
- HF artifacts per run: gs://marin-us-east5/checkpoints/qwen3-useful/<run-id>/hf (backbone + classifier_head.npz).

## Monitoring commands
- Coordinator/child status: `uv run iris --cluster marin job bug-report /michaelryan/qwen3clf-smoke-r0` (or `job list --json | grep qwen3clf`).
- W&B: project `qwen3-useful`, run `qwen3clf-smoke-r0`. Look for loss curve + `eval/best_f1` summary at the end.
- Child TPU job name is `train_decoder_classifier` (nested under the coordinator).

## Known risks to watch on the smoke
- HBM at Pos=4096 batch16 pdp2 on v6e-4 (should be fine; 0.6B is small). If OOM on real Pos=8192, drop pdp to 1.
- Causal+segment mask under the TPU splash kernel (default attn backend) — if splash rejects the combo, fall back attn_backend=vanilla (slower) or jax.
- HF save path (`save_classifier`): backbone via Qwen3 converter + head npz. If it errors at the very end, the F1 is already logged to W&B; fix save and it's non-fatal to the result.
- Token 8996 ("NO") warm-start assumes the Qwen3 tokenizer; launcher hardcodes it (verified).

## Launch recipe (CPU coordinator submits the TPU job, blocks to keep children alive)
```
uv run iris --cluster marin job run --region us-east5 --cpu 2 --memory 8GB --disk 10GB \
  --priority interactive --no-wait --extra cpu --enable-extra-resources \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -e HF_TOKEN "$HF_TOKEN" \
  --job-name qwen3clf-<id> \
  -- python -m experiments.baseline_collection.exp_qwen3_0_6b_useful_classifier --run-id <id> [--smoke]
```
