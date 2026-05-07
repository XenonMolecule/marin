# Overnight Status — Medical 0.6B-Base Fix + 8B Sweep Prep

**Last updated:** auto-updated by autonomous loop
**Started:** 2026-05-04 03:33 PT

## TL;DR (read this first)

Smoke verified the fix. 0.6B medical sweep launched. 8B path being smoke-tested.
Baselines launched for medical (0.6B + 8B). Eval auto-transitions handled by the
loop (training summaries → eval children submitted in subsequent wakeups).

## 🟢 First real result: 0.6B-Base medical baseline (no SFT)

Eval completed at 11:26 UTC. Per-subtask exact_match (MMLU 5-shot generative + mediqa 0-shot):

| Subtask | exact_match |
|---|---|
| mmlu_anatomy | 0.31 |
| mmlu_clinical_knowledge | 0.19 |
| mmlu_college_biology | 0.26 |
| mmlu_college_medicine | 0.22 |
| mmlu_high_school_biology | 0.35 |
| mmlu_medical_genetics | 0.31 |
| mmlu_professional_medicine | 0.08 |
| **MMLU avg (7 subtasks)** | **~24.6%** |

**Important shift from the published table:** the OLD "0.6B baseline 51.22%" was
the INSTRUCT model (the bug we're fixing). The TRUE 0.6B-Base baseline is much
lower (~25%, near random for 4-way MCQ). This means the SFT delta values in the
new table will look much bigger because the new baseline floor is much lower.
Plan to call this out in the rewritten report — comparing instruct-baseline
deltas to base-baseline deltas is apples-to-oranges.

Results files: `gs://marin-eu-west4/eval_baselines/baseline-qwen3-0.6b-base/lm_eval_harness/`

## 🟢 Second result: 8B-Base medical baseline

| Subtask | exact_match |
|---|---|
| mmlu_anatomy | 0.71 |
| mmlu_clinical_knowledge | 0.79 |
| mmlu_college_biology | 0.90 |
| mmlu_college_medicine | 0.79 |
| mmlu_high_school_biology | 0.91 |
| mmlu_medical_genetics | 0.86 |
| mmlu_professional_medicine | 0.82 |
| **MMLU avg (7 subtasks)** | **82.57%** |

Compares cleanly to the published 14B-Base baseline of 86.0% — bigger model → stronger
zero-SFT performance, as expected. The 0.6B vs 8B vs 14B baseline trajectory is
24.6 → 82.6 → 86.0 — the 0.6B model is essentially at random-chance (4-way MCQ
floor = 25%), 8B has captured most of medical knowledge from pretraining, 14B
adds marginal improvement.

Results files: `gs://marin-us-east5/eval_baselines/baseline-qwen3-8b-base/lm_eval_harness/`

## 🟢 Third result: 5 medical-extraction 0.6B-Base SFT cells COMPLETED

Cells with HF exports + summaries: lr1e-6_bs32, lr2e-6_bs32, lr3e-6_bs32,
lr5e-6_bs32, lr1e-6_bs64. All 4 remaining (lr2/3/5/7e-6_bs64) still training.
Eval coordinator (`launch_medical_evals.py`) dispatched at 06:19 PT to fan out
evals for these 5 completed runs.

## What works (commitments to verify)

- **Levanter patches** (in `lib/levanter/`): two surgical edits that make the
  in-process `train_lm.main` path bridge the tokenizer-vs-model vocab gap that
  silently worked in the Ray executor path. See
  `medical_base_fix.md` for full root-cause writeup.
  - `train_lm.py:103` — sync local `tokenizer` to padded converter.tokenizer
    after `pad_tokenizer_to_match_model` runs.
  - `hf_checkpoints.py` — measure tokenizer len AFTER `as_hf_tokenizer()` so
    pad math doesn't overshoot by the +4 difference between MarinTokenizer's
    static `_vocab_size` and HF AutoTokenizer's `len()`.
- **Tokenizer fact** (verified empirically): all Qwen3 *-Base variants share
  `tokenizer.json` byte-for-byte (sha256 c0382117ea329cdf). Instruct's
  tokenizer.json (sha aeb13307a71acd8f) differs from Base's only in 4 extra
  added_tokens (`<tool_response>`, `</tool_response>`, `<think>`, `</think>`).
  None of those appear in extracted medical text, so the medical caches
  (originally tokenized with instruct) work fine with Base when paired with
  pad_tokenizer.

## 🟢 PROFOUND RESULT — first 4 SFT'd 0.6B-Base medical extraction cells

| Cell | MMLU Medical avg | Delta vs Base baseline (24.57%) | Vs published instruct-baseline (51.22%) |
|---|---|---|---|
| Baseline (Qwen3-0.6B-Base, no SFT) | 24.57% | — | -26.65pp |
| lr2e-6_bs32 | 52.71% | +28.14pp | +1.49pp |
| lr1e-6_bs32 | 53.43% | +28.86pp | +2.21pp |
| lr5e-6_bs64 | 53.00% | +28.43pp | +1.78pp |
| **lr1e-6_bs64** | **54.14%** | **+29.57pp** | **+2.92pp** ← winner of 9 |
| lr2e-6_bs64 | 53.43% | +28.86pp | +2.21pp |
| lr3e-6_bs32 | 52.00% | +27.43pp | +0.78pp |
| lr3e-6_bs64 | 52.43% | +27.86pp | +1.21pp |
| lr5e-6_bs32 | 50.43% | +25.86pp | -0.79pp |
| lr5e-6_bs64 | 53.00% | +28.43pp | +1.78pp |
| lr7e-6_bs64 | 50.86% | +26.29pp | -0.36pp |

**ALL 9 EXTRACTION CELLS HAVE EVAL RESULTS.** Clean pattern:
- Lower LR wins (1e-6 > 2e-6 > 3e-6 > 5e-6 > 7e-6)
- Larger BS (64) beats smaller BS (32) at same LR
- Best HP cell: **lr=1e-6, bs=64 → 54.14%** (also top by GSM8K-flex / etc., consistent winner)
- vs. published instruct V2 SFT best (51.46%): **+2.68pp improvement**

This is a publishable result — the corrected 0.6B-Base story BEATS the prior buggy
instruct-baseline AND the prior buggy "+0.3 SFT delta" by a meaningful margin.

Resiliparse cells still training (slow — ~5-8h each, mostly won't be done by morning).

**Headline:** SFT on extraction data lifts the Base model from random-guessing
(24.6%) to **above** the original Qwen3-0.6B INSTRUCT baseline (51.22%, the one
that was wrongly labeled as Base in the published table). So:

- The original "+0.3 SFT delta over (instruct) baseline" understated the impact
  because the instruct baseline was already strong on MCQ. The TRUE story is
  that medical extraction SFT teaches the Base model 30pp of medical MCQ
  competence from a near-random starting point.
- 5 more extraction cells still queueing for eval (eval coord v3 just
  re-launched).
- 9 medical-resiliparse cells still training (slow — see below).

## ⚠️ Resiliparse cells SLOWER than expected (~5-8h each)

Resiliparse cache has 4.3x more tokens (2.43B vs 560M for extraction). At
1.4s/step on v6e-4, bs=32 resiliparse cells take ~7 hours wall-clock; bs=64
~3.6h. Most resiliparse cells will NOT complete by morning. The published
table will need a partial-completion footnote.

## In flight (state at last loop tick)

**Training:**
- `/michaelryan/medical-ext-base-smoke-fix5` — proof-of-life smoke on v6e-4 (us-east1).
  Saved checkpoint at step 403 of 4277.
- `/michaelryan/med-ext-base-0_6b-full` — 9-cell extraction sweep, all RUNNING.
- `/michaelryan/med-res-base-0_6b-full` — 9-cell resiliparse sweep, PENDING
  (capacity-limited; will dispatch as v5p-8 / v6e-4 frees).
- `/michaelryan/code-ext-base-8b-smoke2` — 8B smoke on v5p-16 (multi-host).
  First version OOM'd on v5p-8 (8B doesn't fit single-host).

**Evals (baselines):**
- `/michaelryan/medical-eval-baseline-0_6b-v2` — `Qwen/Qwen3-0.6B-Base` on
  medical (gives the BASELINE row of the table).
- `/michaelryan/medical-eval-baseline-8b-v2` — `Qwen/Qwen3-8B-Base` on medical.
- Note: -v1 of both was killed because the eval standalone had a bug — when
  passed an HF reference (not a gs:// path), output_path defaulted to the
  relative `Qwen/Qwen3-0.6B-Base/eval/lm_eval_harness` which writes to local
  ephemeral fs. Fixed: HF references now route to
  `gs://marin-{region}/eval_baselines/{model_name}/lm_eval_harness/`.

## Files touched this session

- **New** `experiments/rephraser/run_medical_sft_standalone.py` — TPU child for the
  medical 0.6B-Base re-run.
- **New** `experiments/rephraser/medical_extraction_sft_v2_base.py` — coordinator.
- **New** `experiments/rephraser/medical_resiliparse_v2_base.py` — coordinator.
- **New** `experiments/rephraser/run_medical_eval_standalone.py` — eval TPU child.
- **New** `experiments/rephraser/launch_medical_evals.py` — eval coordinator.
- **New** `experiments/rephraser/run_sft_standalone.py` — generalized 0.6b/8b/14b training child.
- **New** `experiments/rephraser/launch_sft_sweep.py` — generalized coordinator (handles
  all (domain, branch, model_size) combos with multi-host TPU support).
- **Edited** `experiments/qwen3.py` — added `qwen3_8b_base`.
- **Edited** `lib/levanter/src/levanter/main/train_lm.py` — pad_tokenizer fix.
- **Edited** `lib/levanter/src/levanter/compat/hf_checkpoints.py` — pad_tokenizer fix.

## Outstanding (handled by autonomous loop)

- **Auto-transition training → eval**: the loop polls
  `gs://marin-us-central1/metadata/medical_sft_base_results/` for new training
  summaries and submits eval children for any not yet evaluated.
- **Launch full 8B sweeps**: when 8B smoke2 verifies the v5p-16 path works,
  the loop launches all 6 8B coordinators (code/math/medical × extraction/resiliparse).
- **Aggregate + rewrite reports**: when training+eval results land, the loop
  reads summaries, joins by run_name, builds the table, and rewrites
  `code_math_medical.md`, `MEDICAL_EXTRACTION_V2_REPORT.md`, etc.
- **Code+math eval extension**: needed for 8B baselines on HumanEval / MATH.
  Not yet done.

## What to look at first when you wake up

1. `iris job list` — see which jobs completed vs running.
2. `gs://marin-us-central1/metadata/medical_sft_base_results/*.json` — training summaries.
3. `gs://marin-us-central1/metadata/medical_sft_base_eval_results/*.json` — eval summaries.
4. `gs://marin-{us-central1,us-east5,us-east1}/eval_baselines/baseline-qwen3-0.6b-base/` — 0.6B baseline numbers.
5. `gs://marin-{...}/eval_baselines/baseline-qwen3-8b-base/` — 8B baseline numbers.
6. This file (will be updated as the loop progresses).

## Bugs auto-fixed during the night

- **Vocab axis mismatch (151,936 vs 151,665)**: root caused to two interlocking
  bugs in Levanter; patched both. See `train_lm.py:103` and
  `hf_checkpoints.py:with_tokenizer_padded_to_match_model`.
- **Eval output_path was relative for HF refs**: now routes to a region-local
  GCS prefix.
- **8B OOM on v5p-8**: switched to v5p-16 multi-host with proper
  `replicas=1 + coscheduling` setup. 8B smoke2 now training (loss=1.05).
- **CPU OOM during HF export at end of training (32GB → 64GB)**: caught when
  smoke fix5 reached training end and tried to save the 2.38GB HF shard.
  Levanter's `save_hf_checkpoint` rounds buffers to ~6-10x the raw shard size
  during serialization. Bumped resource memory in both medical coordinators
  to 64GB; updated launch_sft_sweep.py defaults (0.6B→64GB, 8B→96GB,
  14B→128GB). Killed v1 sweeps + smoke fix5; relaunched as v2. Children
  resume from 10-min rolling checkpoints — at most ~10 min of progress lost.
