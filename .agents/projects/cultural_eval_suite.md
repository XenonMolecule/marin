# Cultural eval suite (`cultural_sl_verb`) and MMLU category evals

Rules of record for the human-centered domain evals used to test the "data mixing wins
every domain" claim. Decided 2026-09-07; research trail in `domain_evals_research.md`.

## Scope

Two eval families, both rendered as the CRFM lm-eval fork's `mmlu_sl_verb` shape
(verbalized multiple choice, lettered choices in the context and in the scored
continuation, 5-shot, seven soft metrics) and run through the Levanter harness exactly
like `experiments/scaling_law_sweeps/mmlu/`:

1. **MMLU by category** — the existing `mmlu_sl_verb` 5-shot results, consolidated per
   category. No new eval runs: `consolidate_mmlu_categories.py` groups the 57 subject rows
   already in every results.json.
2. **Cultural suite** — four benchmarks staged as custom lm-eval tasks under
   `experiments/scaling_law_sweeps/cultural/`.

## Benchmarks (citation counts from Semantic Scholar, 2026-09-07)

| Benchmark | Paper | Cites | Format | Items used | Regions | Source pin |
|---|---|---|---|---|---|---|
| BLEnD MCQ | arXiv 2406.09948 (NeurIPS 2024 D&B) | 238 | MC-4, distractors from other countries | 5,081 unique before filters | 16 | `nayeon212/BLEnD` @ `488f141c`, `data/mc_questions_hf/mc_questions_file_v1.1.json` |
| CulturalBench-Easy | arXiv 2410.02677 (ACL 2025) | 79 | MC-4 | 1,227 | 45 | `kellycyy/CulturalBench` @ `996d62a5`, `CulturalBench-Easy.csv` |
| FORK | ACL 2023 (Palta & Rudinger) | 69 | 2-way | 153 | 11 | github `shramay-palta/FORK_ACL2023` @ `13e6f7c1`, `fork.jsonl` |
| NormAd | arXiv 2404.12464 (NAACL 2025) | 48 | 2-way yes/no | 1,818 | 75 | `akhilayerukola/NormAd` @ `bb1a2997`, `normad_etiquette_final_data.csv` |

Rejected: EtiCor (negatives are BART-generated negations of crawled sentences; form-gated),
CulturalBench-Hard (same questions as Easy), NormAd neutral class (depends on a hidden rule),
TriviaRoomQA (pop trivia), Global-MMLU CS subset (MMLU-verbatim), SemEval-2026 Task 7
(questions only on CodaBench; no few-shot, no redistribution). Full reasoning in
`domain_evals_research.md`.

## Cleaning rules (implemented in `cultural/cultural_sources.py`, pinned by tests)

- **BLEnD**
  - One document per (country, question ID): the MC file holds ~60 distractor draws per
    question; keep the first draw in file order (the `dedup_mc_rows` convention already
    used by the bpb builder).
  - Cut the generation instruction at " Without any explanation" (existing rule).
  - Strip a trailing parenthetical only when it starts with "(Provide": 737 questions
    carry short-answer format instructions ("(Provide Arabic numerals (e.g., 12) only.)",
    "(Provide in HH:MM format ...)", "(Provide the official name.)") that are meaningless
    for a lettered choice. Keep the 45 answer-type hints such as
    "(e.g., breakfast, lunch, dinner)"; they disambiguate the question.
  - Drop documents whose gold answer is `not-applicable`.
  - One task per country: `cultural_sl_verb_blend_<country>`, static description
    "The following are multiple choice questions (with answers) about everyday life in
    {country}." (sl_verb layout).
- **CulturalBench-Easy**: use every row unchanged, including the 86 composite
  "Read the following statements and select the option ..." items (tagged
  `category=composite`). One task, country line rendered per document.
- **FORK**: drop the 31 `Type == "Underspecified"` items (US-default gold with no country
  named). One task, static description about dining customs; `category` = Dimension.
- **NormAd**: keep only `Gold Label in {yes, no}` (drop 797 neutral). Question = the story
  (which ends in "Is what X did acceptable?"), choices `yes` / `no`, country line rendered
  per document; `category` = Subaxis. No country-swap or other controls (user decision).

## Prompt format

Identical to `mmlu_sl_verb`: `doc_to_text` = "<question>\nA. <c0>\nB. <c1>[\nC. ...]\nAnswer:",
`doc_to_choice` = ["A. <c0>", "B. <c1>", ...], metric list `acc, acc_norm, bpb, logprob,
choice_logprob, choice_prob_norm, choice_logprob_norm`. 2-way tasks (FORK, NormAd) use
letters A and B only. Tasks whose documents span countries (CulturalBench, NormAd) put the
country line at the start of `doc_to_text` so every few-shot example names its own country;
single-country tasks (BLEnD per country) and FORK use a static `description` like MMLU.

## Shot count

5-shot everywhere, matching `mmlu_sl_verb`. Each task's `dev` split holds five documents
removed from `test`, chosen round-robin over the gold index in source order (the first
gold-A document, then the first gold-B, ...; `cultural_sources.select_balanced_shots`).
A plain "first five" was rejected on the real data: NormAd's CSV is sorted by country
and label, so its first five rows are all Egypt / yes, and BLEnD UK's first five are four
gold-A. Round-robin gives 2-way tasks a 3/2 label split and MC-4 tasks A, B, C, D, A, so
the shot block carries no one-sided answer prior. Shots are per country for BLEnD and
global for the other three; the dev split keeps the alternating order for lm-eval's
`fewshot_config.sampler: first_n`. Built sizes (2026-09-07): BLEnD 4,873 test docs over
16 tasks (128 `not-applicable` golds dropped), CulturalBench 1,222, FORK 148, NormAd 1,813;
8,056 test documents, ~1 MB of parquet.

## Metrics and aggregation

- Report all seven soft metrics per task. Raw accuracy is at chance at sweep scale.
- Chance rates differ (25% for MC-4, 50% for 2-way), so cross-benchmark comparison uses
  chance-centered accuracy `(acc - 1/k) / (1 - 1/k)` and the zero-knowledge null gap from
  `analyze_cultural_signal.py` (the `analyze_mmlu_signal.py` method).
- Benchmark level: BLEnD = unweighted mean over its 16 country tasks; the other three are
  single tasks. Suite macro average = unweighted mean over the four benchmarks.
- Unit of analysis is the benchmark or a (benchmark, country/category) cell, never an item.

## Layout on GCS

- Staged data: `gs://marin-<region>/eval_datasets/cultural_sl_verb/<task>/{dev,test}.parquet`
  plus `_manifest.json`, mirrored to all five regional buckets (a few MB total). Sync
  with `gcloud storage rsync -r` without `-d`.
- Results: `gs://<bucket>/metadata/cultural_sl_verb_results/<shots>shot/<run_name>/results.json`,
  in the checkpoint's own bucket. Rows are keyed `cultural_sl_verb_<task>_5shot`; the
  group row is `cultural_sl_verb_5shot`.
- MMLU category CSV: `gs://marin-us-central1/metadata/mmlu_sl_verb_5shot_by_category.csv`.

## Commands

```bash
# 1. Build + upload the staged data (CPU job; HF + GitHub reads only)
iris --cluster marin job run --region us-east5 --cpu 2 --memory 8GB --enable-extra-resources \
  --extra eval -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.scaling_law_sweeps.cultural.build_cultural_tasks \
       --output-dir /tmp/cultural_sl_verb \
       --upload-to gs://marin-us-east5/eval_datasets/cultural_sl_verb/ \
                   gs://marin-us-central1/eval_datasets/cultural_sl_verb/ \
                   gs://marin-eu-west4/eval_datasets/cultural_sl_verb/ \
                   gs://marin-us-central2/eval_datasets/cultural_sl_verb/ \
                   gs://marin-us-east1/eval_datasets/cultural_sl_verb/

# 2. Smoke one checkpoint (--limit), then the sweep
iris --cluster marin job run -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.scaling_law_sweeps.cultural.launch_cultural_manifest \
       --manifest experiments/core_eval_manifests/checkpoint_manifest_10k_dedup245.txt \
       --launch --max-count 1 --limit 8 --name-suffix=-smoke1
iris --cluster marin job run -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.scaling_law_sweeps.cultural.launch_cultural_manifest \
       --manifest experiments/core_eval_manifests/checkpoint_manifest_10k_dedup245.txt --launch

# 3. Consolidate (CPU job, in-cloud reads)
iris --cluster marin job run --region us-central1 --cpu 2 --memory 16GB --enable-extra-resources \
  -- python -m experiments.scaling_law_sweeps.cultural.consolidate_cultural_results --shots 5
iris --cluster marin job run --region us-central1 --cpu 2 --memory 16GB --enable-extra-resources \
  -- python -m experiments.scaling_law_sweeps.mmlu.consolidate_mmlu_categories --shots 5
```

## Reported domains (user decision 2026-09-07)

The headline per-domain table has four columns: **STEM**, **Humanities**, **Social
sciences** (the official MMLU categories) and **Cultural** (the four-benchmark suite
macro average above). MMLU's "Other" category (8 health + 3 business + 3 misc subjects,
with `miscellaneous` alone 24% of its weight) is not a reported domain; the consolidator
still emits its row for reference. Health knowledge is covered separately by the
MedMCQA / MedQA bpb tasks already in the OLMIX suite.

## MMLU categories (the official MMLU release grouping)

The four categories are MMLU's own, from Hendrycks et al.'s `categories.py` in the
original `hendrycks/test` release (subcategories such as physics or law roll up into
STEM / humanities / social sciences / other). `olmo_bpb/mmlu_olmes.py` mirrors that table
verbatim and is the copy the consolidator reads.

| Category | Subjects | Members |
|---|---|---|
| stem | 18 | abstract_algebra, astronomy, college_biology, college_chemistry, college_computer_science, college_mathematics, college_physics, computer_security, conceptual_physics, electrical_engineering, elementary_mathematics, high_school_biology, high_school_chemistry, high_school_computer_science, high_school_mathematics, high_school_physics, high_school_statistics, machine_learning |
| humanities | 13 | formal_logic, high_school_european_history, high_school_us_history, high_school_world_history, international_law, jurisprudence, logical_fallacies, moral_disputes, moral_scenarios, philosophy, prehistory, professional_law, world_religions |
| social_sciences | 12 | econometrics, high_school_geography, high_school_government_and_politics, high_school_macroeconomics, high_school_microeconomics, high_school_psychology, human_sexuality, professional_psychology, public_relations, security_studies, sociology, us_foreign_policy |
| other | 14 | anatomy, business_ethics, clinical_knowledge, college_medicine, global_facts, human_aging, management, marketing, medical_genetics, miscellaneous, nutrition, professional_accounting, professional_medicine, virology |

Category aggregation follows the sl_verb group YAML: `acc` and `acc_norm` weighted by
subject example count (the `OLMIX_MMLU_WEIGHTS` fractions), soft metrics an unweighted
mean over subjects. The sl_verb `tag` subgroups produce no rows in results.json, which is
why the category numbers come from the consolidator rather than the harness.

## Harness entry point after the 2026-08-13 upstream sync

`marin.evaluation.run.evaluate` and `EvaluationConfig` no longer exist (upstream replaced
marin's lm-eval evaluators with Evalchemy over a served model, #7281), and the leftover
`marin.evaluation.evaluators.levanter_lm_eval_evaluator` module fails to import. Sweep eval
children therefore call Levanter's harness directly through
`experiments/scaling_law_sweeps/levanter_lm_eval.run_levanter_lm_eval`, which builds
`EvalHarnessMainConfig` exactly as the old evaluator did and writes `results.json` in the
same shape. `cultural/run_cultural_eval.py` and `mmlu/run_mmlu_eval.py` use it;
`core_tasks/run_core_tasks_eval.py` and `olmes_base/run_olmes_eval.py` still import the
removed API and will fail at the same line until they are switched over.

## Smoke (2026-09-08)

`/michaelryan/cultural-sl-verb-smoke3` ran the 1e17 d512 dclm checkpoint with `--limit 8`
through the fixed runner: results.json (1.3 MB) has the group row + 19 task rows with all
seven metrics; `consolidate_cultural_results.task_metrics/aggregate` and
`analyze_cultural_signal` parse it (blend 128x4, culturalbench 8x4, fork 8x2, normad 8x2
samples). NormAd acc=1.0 under `--limit 8` is an artifact: the test split keeps source
order (Egypt / yes first), so the first eight docs share a label. Full runs are balanced.
The suite is ready to launch over the 245-checkpoint manifest.
