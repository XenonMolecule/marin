# Medical 0.6B-Base Re-run — Status & Report-Rewrite Plan

**Date opened:** 2026-05-04
**Status:** training scripts written, smokes in flight, eval pipeline ready, report rewrite pending

---

## What's wrong with the published numbers

The `code_math_medical.md` table claims **all** medical 0.6B numbers used `Qwen3-0.6B-Base`. They didn't.

Confirmed by direct script inspection + git history (every branch, every commit):

| Cell | Script | Actual model |
|---|---|---|
| Code 0.6B (baseline + SFT) | `code_extraction_sft_v3_base.py` + `code_extraction_sft_v3_sweep*.py` | `Qwen3-0.6B-Base` ✓ |
| Code 14B | `code_extraction_sft_v3_14b_sweep*.py` | `Qwen3-14B-Base` ✓ |
| Math 0.6B (baseline + SFT) | `math_top3_hp_sweep.py`, `math_0_6b_baseline_eval.py` | `Qwen3-0.6B-Base` ✓ |
| Math 14B | `math_14b_top3_sft.py` | `Qwen3-14B-Base` ✓ |
| **Medical 0.6B baseline** (51.2) | `medical_extraction_sft_v2.py` recipe path | `Qwen/Qwen3-0.6B` ❌ instruct |
| **Medical 0.6B SFT (E + R)** | `medical_extraction_sft_v2.py:178`, `medical_resiliparse_sweep.py:122,166`, `medical_extraction_v2_sweep.py:124,168`, `medical_extraction_sft_hpsweep.py:101`, `medical_resiliparse_sft_hpsweep.py:99` | `Qwen/Qwen3-0.6B` ❌ instruct |
| Medical 14B | `medical_14b_sft.py:164`, `medical_14b_baseline_eval.py:29` | `Qwen3-14B-Base` ✓ |

The wrong claim propagated from the docstring at `medical_extraction_sft.py:13` into 5+ reports.

---

## What we did to fix it (this branch)

### New files (all Iris-native, no Ray, no Marin executor)

- `experiments/rephraser/run_medical_sft_standalone.py` — TPU SFT child. 1:1 with `extraction_sft_recipe._run_single_epoch_sft` for the TrainLmConfig build; the only intentional divergence is `initialize_from_hf="Qwen/Qwen3-0.6B-Base"`.
- `experiments/rephraser/medical_extraction_sft_v2_base.py` — extraction coordinator. 9-cell Phase-1 LR×BS grid (matches `medical_extraction_v2_sweep.py:68-83`).
- `experiments/rephraser/medical_resiliparse_v2_base.py` — resiliparse coordinator (same 9-cell grid).
- `experiments/rephraser/run_medical_eval_standalone.py` — TPU eval child. Calls `LMEvaluationHarnessEvaluator.evaluate(...)` IN-PROCESS (the in-process method, NOT the `launch_evaluate` Ray-submit wrapper).
- `experiments/rephraser/launch_medical_evals.py` — eval coordinator. Auto-discovers completed training runs from the training summary prefix; submits one vLLM-TPU eval child per run.

### Data mirrors (region pre-staging, not new extraction)

| Region | extraction cache (e74e0d) | resiliparse cache (2061a9) |
|---|---|---|
| us-central1 | original | original |
| us-east5 | mirrored 2026-05-04 | mirrored 2026-05-04 |
| us-east1 | mirrored 2026-05-04 | mirrored 2026-05-04 |

Total cross-region egress: ~$1 (well under the $10 cap).

### Coordinator constraints

- `DEFAULT_ALLOWED_REGIONS = ("us-central1", "us-east5", "us-east1")` — HARD constraint.
- `DEFAULT_TPU_VARIANT = "v5p-8"` + `DEFAULT_TPU_ALTERNATIVES = ("v6e-4",)` — both vm_count=1, mixed via `device_variant_constraint`.
- `--child-priority interactive` for the smoke + the full launch (per "ASAP" directive).

---

## Smokes in flight (2026-05-04)

| Job | TPU | HP cell | Status (last seen) |
|---|---|---|---|
| `/michaelryan/medical-ext-base-smoke` | v5p-8 | lr5e-6_bs32 | pending (capacity exhausted in central1+east5) |
| `/michaelryan/medical-ext-base-smoke-v6e` | v6e-4 | lr2e-6_bs32 | pending scheduler feedback |

Two different HP cells so both runs are useful regardless of which lands first.

---

## Eval suite (1:1 with the original instruct runs)

`run_medical_eval_standalone.py:MEDICAL_EVALS` mirrors `medical_extraction_sft_v2.py:130-155` exactly:

- `mmlu_anatomy_generative` 5-shot
- `mmlu_clinical_knowledge_generative` 5-shot
- `mmlu_college_medicine_generative` 5-shot
- `mmlu_medical_genetics_generative` 5-shot
- `mmlu_professional_medicine_generative` 5-shot
- `mmlu_college_biology_generative` 5-shot
- `mmlu_high_school_biology_generative` 5-shot
- `mediqa_qa2019_lite` 0-shot (for completeness, table only uses the 7 MMLU)

`engine_kwargs={"max_model_len": 8192, "max_gen_toks": 512}`, `apply_chat_template=False`. Identical to the original.

---

## Report-rewrite plan (do AFTER results land)

User directive: "Just rewrite at the end." Below are the EXACT lines/sections that need updates. Mechanical search-and-replace once we have the new numbers.

### `experiments/rephraser/code_math_medical.md`

- **Line 8**: `"All experiments fine-tune **Qwen3-0.6B-Base** and **Qwen3-14B-Base**..."` — was wrong for medical 0.6B, but with the re-run it WILL be true. Leave the line as-is once new numbers land.
- **Lines 282-296** (Medical results table): replace the 4 medical 0.6B rows with new Base numbers. Keep the 14B rows as-is.
- **Line 295** (commentary on "winner flips with scale"): re-evaluate; the 0.6B-Base winner may differ from the 0.6B-instruct winner.

### `experiments/rephraser/MEDICAL_EXTRACTION_V2_REPORT.md`

- **Line 4**: header says "Model: Qwen3-0.6B-Base" — already true once re-run lands.
- **Line 15**: table label "Baseline (Qwen3-0.6B)" — change to "Baseline (Qwen3-0.6B-Base)" + new number.
- All baseline / extraction / resiliparse numbers in the per-subtask breakdown (lines 78-98 area) — replace with new numbers.

### `experiments/rephraser/MEDICAL_EXTRACTION_REPORT.md`

- Same kind of fixes as V2 report. Numbers throughout need replacement.

### `experiments/rephraser/medical_extraction_sft.py`

- **Line 13**: docstring `"Model: Qwen3-0.6B-Base (same as math/code experiments)."` — was a lie when written. After we fix the script's actual `hf_model_name` (or after we just deprecate this old script in favor of `medical_extraction_sft_v2_base.py`), update or delete the docstring.
- The buggy script ITSELF stays as an audit trail of the original error. Mark it as deprecated at the top.

### `experiments/rephraser/MATH_REGRESSION_ANALYSIS.md`

- Search for "Qwen3-0.6B-Base" in medical context — verify accurate after re-run.

### `experiments/rephraser/PROGRESS_REPORT.md`

- Same.

### `experiments/rephraser/optimal_hyperparameters.json`

- Either add a new section for `medical_0_6b_base` keyed by branch (extract / resiliparse), or replace the old entries in place. New best HP configs come from the re-run sweeps.

---

## What the rewrite is NOT for

- **Code 0.6B + 14B numbers**: untouched by this fix. Those were always Base.
- **Math 0.6B + 14B numbers**: untouched. Those were always Base.
- **Medical 14B numbers**: untouched. Those were always Base.

Only the 4 medical 0.6B cells (baseline + 3 SFT comparisons in the published table) need new numbers. Everything else stays.

---

## Open questions / followups

1. **8B-Base extension**: deferred until 0.6B-Base fix lands (per user directive). When we do this, the same scripts apply with `--hf-model-name Qwen/Qwen3-8B-Base` and a different RoPE config.
2. **Phase-2 (WD × warmup) sweep**: omitted for time. If results suggest WD/warmup is the bottleneck, re-run.
3. **Verify the cache tokenizer string assumption**: my standalone passes `cache_tokenizer="Qwen/Qwen3-0.6B"` (matches the cache's `.executor_info`). If Levanter ever cares that the tokenizer matches `hf_model_name`, this will surface. Both tokenizer.jsons are byte-identical so it should be fine.
