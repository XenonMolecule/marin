# fastText "useful vs [NO_USEFUL_CONTENT]" Classifier — Findings

**Task:** Stage-1 quality filter for the HTML→LLM extraction pipeline. Binary classifier on **raw HTML** (body_strip representation: strip `<script>`, keep `<body>`) predicting whether the LLM extractor produced useful content vs `[NO_USEFUL_CONTENT]`. Runs on raw HTML — the LLM does the actual extraction downstream.

**Code:** `experiments/baseline_collection/fasttext_useful_classifier.py` (+ tests). **Data:** `gs://marin-us-central2/datasets/high_quality_3000_distill/` (3000 per-WARC parquets, 10.2M useful + 129M `[NO_USEFUL_CONTENT]`). **Leaderboard:** `gs://marin-us-central2/classifiers/useful_fasttext/LEADERBOARD.md`.

## Evaluation protocol (the only number that matters)

- **Frozen split:** snapshot-stratified held-out — 35 val + 35 test WARCs, one+ per CC-MAIN snapshot (2013–2017, 35 snapshots). Materialized at `…/full_prep_body_strip/{val,test}/` (9.2 / 9.5 GB, 35 shards each).
- **Deployment ratio ≈ 11.8 : 1** (non-useful : useful). All headline F1s are **best-F1 over a threshold sweep on this frozen ~12:1 test** — NOT each run's own balanced pilot test (those are inflated ~2× and not comparable).
- Per-doc scores saved as `<run>/eval_natural.preds.parquet` so any threshold/curve recomputes offline.

## Headline results (natural-ratio frozen-test best F1)

| model | ratio | sampling | natural F1 |
|---|---|---|---|
| DCLM off-the-shelf (oh-eli5, neg-label cc) | — | — | 0.210 |
| balanced 1M (front) | 1:1 | front | 0.425 |
| balanced 200k (front, = DCLM recipe) | 1:1 | front | 0.472 |
| balanced 200k (random / stratified) | 1:1 | rand/strat | 0.474 / 0.477 |
| stratified 4:1 | 4:1 | stratified | 0.479 |
| stratified 8:1 | 8:1 | stratified | 0.556 |
| stratified 12:1 | 12:1 | stratified | 0.576 |
| stratified 16:1 | 16:1 | stratified | 0.578 |
| **front 12:1 (`natratio200kpos`) — WINNER** | **12:1** | **front** | **0.597** |

## Five findings (each with evidence)

1. **Training ratio is the dominant lever — monotonic, peaks at the deployment ratio.** As neg/pos climbs 1→4→8→12, natural F1 rises 0.477→0.479→0.556→0.597 (+26% over the 0.472 balanced baseline). Pushing *past* deployment (16:1) doesn't help (0.578, recall drops) — train at the real ratio, not beyond. **Why:** balanced-trained scores are miscalibrated for a 12:1 world; matching the ratio fixes precision.
2. **More balanced data HURTS deployment.** 200k→1M balanced went 0.472→**0.425**. fastText saturates; the problem is calibration, not capacity. Don't scale data under the wrong ratio.
3. **Sampling is a small, noisy lever (~±0.02) — it does NOT compound with ratio.** At 1:1 stratified (0.477) ≈ random (0.474) ≈ front (0.472); at 12:1, stratified (0.576) was actually *below* front (0.597). The "stratified + 12:1 compounds" hypothesis was wrong. (Front-first IS biased — shard index correlates with snapshot — but it didn't matter empirically. Knob exists: `--train-sample {front,random,stratified}`.)
4. **Autotuning HURTS — the hand-set DCLM recipe wins.** Three independent 12 h fastText autotunes (200k bal, 1M bal, 200k@12:1) ALL converged on the same config (lr 0.1, dim 100, epoch 5, **unigrams**, no buckets) and ALL underperformed the fixed **bigram** DCLM recipe (autotune balanced → ~0.42–0.47, no gain; autotune@12:1 pilot 0.536 < fixed 0.568). Don't autotune; use the recipe.
5. **minCount pruning → 10× smaller model, ZERO quality loss (deployability solved).**
   | minCount | model size | natural F1 |
   |---|---|---|
   | 1 (DCLM recipe) | 9.85 GB | 0.597 |
   | 10 | 2.89 GB | 0.595 |
   | 50 | 2.42 GB | 0.597 |
   | 100 | 1.75 GB | 0.594 |
   | **500** | **0.96 GB** | **0.594** |
   Singleton markup tokens (unique URLs/hashes) are pure bloat and carry no signal. `--min-count 500` gives a **sub-GB deployable model at full quality**. (Model size tracks vocabulary, not #docs — and the neg-heavy 12:1 ratio is what made the unpruned winner the *biggest* model at 9.85 GB.)

## Deployment numbers (winner, on the frozen ~12:1 test)

At a chosen recall, fraction of non-useful **excluded** (specificity = 1 − FPR):
- **95% useful retained → ~78% non-useful excluded** (precision 0.27 at 12:1)
- 90% retained → 86% excluded · 99% retained → 54% excluded
- vs DCLM off-the-shelf: 95%→27%, 90%→39% (our model ≈ **3× the filtering power** at matched recall)

## Recommended production config

**front 12:1, DCLM bigram recipe (epoch 5, lr 0.1, dim 100, wordNgrams 2, softmax), `--min-count 500`** → 0.96 GB, F1 0.594, excludes ~78% of non-useful at 95% recall. This is the **stage-1** of the cascade (fastText ≪ ModernBERT ≪ LLM); stage-2 ModernBERT work is in [[project_modernbert_tpu_torch_xla]].

## Plots (regenerable; currently in /tmp/pr_curves/, not yet durable)
`plot_pr.py` (PR curves), `plot_filter.py` (retained-vs-excluded), `plot_compare.py` (DCLM vs best, from raw preds). Built from `eval_natural.preds.parquet`.

## Loose ends
- 1M @ 12:1 longshot (`natratio1Mpos`) was the last fastText run in flight (does scale help at the right ratio?) — check `…/body_strip_natratio1Mpos/` for a result.
- Plots live in /tmp (ephemeral) — move into `experiments/baseline_collection/plots/` if we want them durable.

See also: [[project_useful_classifier_warc_sampling_bias]], [[project_hq3000_distill_hf_dataset]].
