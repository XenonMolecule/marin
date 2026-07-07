# ModernBERT "useful vs [NO_USEFUL_CONTENT]" Classifier — Findings

**Task:** Stage-2 of the cascaded quality filter for the HTML→LLM extraction pipeline. Same binary target as the fastText stage-1 ([fastText findings](fasttext_useful_classifier_findings.md)): given a web document, predict whether the LLM extractor will produce useful content vs `[NO_USEFUL_CONTENT]`. The cascade is **fastText (cheap, runs on everything) → ModernBERT (accurate, runs only on fastText survivors) → LLM (does the actual extraction)**. ModernBERT is the high-accuracy second opinion that recovers precision the cheap stage-1 leaves on the table.

**Model:** `answerdotai/ModernBERT-base` (149M params, native 8192 context), fine-tuned as a sequence classifier on **body_strip** HTML (strip `<script>`, keep `<body>`), max_length 8192.

**Code:** `experiments/baseline_collection/modernbert_tpu_smoke.py` (training + eval + result JSON). **Analysis:** `scratch/bert_curves/` (`cascade_1M.py`, `plot_operating_1M.py`, `analyze.py`, `cascade.py`).
**Final model:** `gs://marin-us-east5/classifiers/useful_fasttext/modernbert_ckpts/mb-1M-e5w4/latest.pt` (+ local `~/models/modernbert_1M_useful/`).
**Result JSON (F1 + threshold sweep + 7000 per-doc preds):** `gs://marin-us-east5/classifiers/useful_fasttext/modernbert_results/mb-1M-final-e5w4.json`.

## Why ModernBERT on TPU/PyTorch-XLA (not Levanter)

Levanter (the lab's JAX trainer) only does decoder LMs — no encoder/classifier head. So ModernBERT runs on **PyTorch + torch_xla 2.7** on TPU (v6e-4, single host, 4 chips, bf16 autocast). bf16 autocast is load-bearing: fp32 NaN'd, bf16 is ModernBERT's native precision and is XLA-friendly. Single-host (world=4) avoids multi-host gang-coscheduling and bad-node broadcast hangs.

## Evaluation protocol (the only number that matters)

- **Same frozen test as fastText:** snapshot-stratified held-out, materialized at `…/full_prep_body_strip/test/` — **7,000 docs, 516 useful (7.4% base rate ≈ deployment ~12:1)**.
- Headline F1 = **best-F1 over a threshold sweep** on this frozen test (NOT the run's balanced pilot).
- Per-doc `[P(useful), label]` saved in the result JSON so any threshold/curve/cascade recomputes offline. fastText scores aligned to the identical 7000-doc test order: `scratch/bert_curves/fasttext_probs_on_bert_test.json` (verified 0/7000 label mismatches).
- Config held constant across the data-scaling sweep: **effective batch 256** (bs 2 × grad-accum 32 × world 4), lr 5e-5, warmup 200, 1 epoch over the per-scale data, sdpa attention, 8192 ctx.

## Headline results (natural-ratio frozen-test best F1)

| model | train docs | natural F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| DCLM off-the-shelf | — | 0.210 | — | — |
| fastText (stage-1 winner, 1 GB pruned) | 189k×~2.3M | 0.594–0.597 | 0.944 | 0.557 |
| ModernBERT 50k | 50k | 0.611 | — | — |
| ModernBERT 100k | 100k | 0.618 | — | — |
| ModernBERT 200k | 200k | 0.644 | — | — |
| **ModernBERT 1M — FINAL** | **1M** | **0.667** (t=0.26, acc 0.947) | **0.959** | **0.570** |

## Findings (each with evidence)

1. **Peak F1 scales smoothly with data and is still climbing at 1M.** 0.597 (fastText) → 0.611 (50k) → 0.618 (100k) → 0.644 (200k) → **0.667 (1M)**. The 200k→1M step (5× data) still adds +0.023, so the curve has not saturated — more data likely helps further. ModernBERT clears fastText by **+0.070 absolute** at 1M.

2. **Scale is what unlocks the high-recall tail — this is the operationally important result.** For an LLM pre-filter you fix a high recall (keep ~all useful docs) and ask how much junk you can drop. **Non-useful excluded at fixed recall (7000-doc test):**

   | recall ≥ | DCLM (ref) | fastText | 200k BERT | **1M BERT** | cascade |
   |---|---|---|---|---|---|
   | 0.99 | — | 58.6% | 50.0% | **68.5%** | 75.4% |
   | 0.975 | — | 67.5% | 64.6% | **78.7%** | 81.5% |
   | 0.95 | 27% | 76.3% | 68.7% | **82.9%** | 84.5% |
   | 0.90 | 39% | 86.7% | — | **88.9%** | 89.6% |

   **Key:** at **200k**, ModernBERT had *higher peak F1 than fastText (0.644 vs 0.597) but a WORSE high-recall tail* — it could not beat fastText in the regime that actually matters for filtering. **Only at 1M does the tail cross above fastText** (+10–11 pts more exclusion at R≥0.975–0.99). Peak F1 is a misleading headline for a pre-filter; the high-recall operating point is the real metric, and it needed ~1M docs of scale to win. (See `scratch/bert_curves/operating_curve_1M_cascade.png`.)

3. **The cascade beats either stage alone, at a fraction of the BERT compute.** fastText@R.99 → 1M-BERT@R.99-on-survivors excludes **75.4% of non-useful at recall 0.981**, vs 68.5% (BERT alone) and 58.6% (fastText alone) — and **ModernBERT only scores the 45.6% of docs that survive fastText** (fastText cheaply discards the obvious junk first). The two models make correlated errors (an independence model predicts 86.9% exclusion → an 11.6-pt correlation gap), but the cascade is still the best accuracy/cost trade-off.

4. **DCLM off-the-shelf is far below everything** (F1 0.210; 27% / 39% exclusion at R0.95 / R0.90) — both our fastText and ModernBERT are large improvements over the standard quality classifier for this task.

## Infrastructure / training notes (for methods + reproducibility)

The 1M run was a ~2-day single-host v6e-4 job on us-east5 preemptible. Hard-won lessons (all fixed in `modernbert_tpu_smoke.py`):

- **Multi-host checkpointing must be rank-0-only.** Save AND load: 4 ranks touching the ~600 MB checkpoint simultaneously deadlocks. Rank-0 saves (no rendezvous), then on resume rank 0 loads → `broadcast_master_param` distributes weights → `all_reduce` of the resume position → scheduler restored by deterministic re-stepping. Checkpoint frequently but not every step (`CKPT_EVERY_OPT_STEPS≈10`).
- **Preemption-native resume.** us-east5 spot preempts every ~15–40 min; each restart costs ~14 min (env install + reload 250k-row shard per rank + XLA compile). The run survives via iris auto-restart + the rank-0 resume; the rate-limiter is that the restart cost must fit inside the spot window. Immutable backups copied in-region every 15% of progress.
- **Deterministic "poison batch" hangs.** The fixed data shuffle (`torch.manual_seed(1000+epoch)`) means specific micro-batches deterministically hang the forward pass (identical losses on every replay confirm it's data, not preemption). Fixed with a `--resume-skip-micro N` flag that fast-forwards the data iterator a few hundred steps past a hanging batch (weights unchanged, ≪0.5% of data skipped). Two such batches were hit at ~51% (steps ~63,750 and ~64,050); skip=1000 cleared both.
- **NEVER run an `rm`-on-stall watcher.** An earlier wipe-and-fresh watcher daemon (`gcloud storage rm` the ckpt dir on stall) was left running and deleted a checkpoint at 45% (buckets have no soft-delete/versioning → unrecoverable). Recovery is resume-from-checkpoint, never wipe. (Detail: [[feedback_rogue_watcher_wiped_checkpoints]].)

## Artifacts

- Model: `~/models/modernbert_1M_useful/modernbert_1M_useful.pt` (state_dict; load into ModernBERT-base seq-classifier), `result_f1_0.667.json`.
- fastText stage-1 (shippable): `~/models/fasttext_useful_best/fasttext_useful_mc500_f1_0.594.bin` (986 MB).
- Operating-curve plot: `scratch/bert_curves/operating_curve_1M_cascade.png` (+ `~/models/`).
- Scaling-curve checkpoints (`mb-{100k,200k}-8192-resv`) and result JSONs in `gs://marin-us-central2/classifiers/useful_fasttext/`.

See also: [[project_modernbert_1M_final_result]], [[project_modernbert_tpu_torch_xla]], [[feedback_modernbert_tpu_checkpoint_resume]].
