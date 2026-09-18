# Background Research Brief: human-centered, per-domain evals for the mixing claim

- Effort: high (three parallel tracks: repo survey, cultural-eval literature, domain-eval + small-scale-signal literature).
- Stop rule: stop when new sources stop changing the ranked experiments. Reached after the small-scale-signal track converged on the same answer as the repo's own MMLU and BLEnD analyses.
- Date: 2026-09-07
- Echo: `infra/echo/cli.py search` returned 403 for this account, so internal search used the repo, `.agents/projects/`, and session memory only.

## Question

Which domain-specific, human-centered evals can a base model at the sweep's scale (d512 to d3584, 3e16 to 3e20 FLOPs; OLMIX proxies at 157M / 4.3B tokens) actually be scored on, so that the claim "data mixing makes this corpus best in every domain" is measurable rather than a restatement of "the mixed model has lower loss everywhere"?

## Current Marin Context

The two evals the user proposed already exist in the repo, both in the oe-eval `requests.jsonl.gz` format consumed by `experiments/scaling_law_sweeps/olmo_bpb/run_olmo_bpb_eval.py`:

- MMLU by category: `mmlu_{stem,humanities,social_sciences,other}/rc_5shot`, defined in `experiments/data_mixing/olmix_tasks.py` (`MMLU_BPB`) and inside the 42-task OLMIX objective.
- BLEnD: 16 per-country tasks `blend_<country>/rc_5shot`, built by `olmo_bpb/build_blend_bpb_requests.py`, wired as `--metric blend` in `run_olmix_fit.py` and marked DIAGNOSTIC ONLY.
- Also staged: `medmcqa`, `medqa_en`, `sciq`, `lab_bench_*`, `jeopardy`, `naturalqs_open`, `squad`, code and math gold-bpb (60-task full suite, complete on all 726 swarm proxies).
- Raw-text domain bpb logged in-loop: Paloma 16 subsets (`experiments/datasets/paloma.py`) and uncheatable 7 subsets (`experiments/datasets/uncheatable.py`), harvested by `experiments/spec_explorer/eval_aggregate.py`.
- Per-domain targeting hook: `run_olmix_fit.py --metric {bpb,core_v2,blend}` swaps the task basis wholesale; results roots in `collect_olmix_swarm.py`. Backfilling new tasks onto evaluated checkpoints uses `--merge --done-marker` (see `olmix_olmixexact_mixture_handoff.md` section 4).

The blocking finding, found twice independently in this repo:

1. `run_olmix_fit.py` help text: across-swarm BLEnD gold-bpb variance is about 90% one general-quality axis, and gold bpb never sees BLEnD's cross-country distractors.
2. Memory `project_mmlu_eval_wiring`: on the 180-run 10k sweep, the MMLU `choice_logprob` curve is 92% language modeling and 8% knowledge once a zero-knowledge null is subtracted; the normalized-choice metrics sit at chance; only the null-corrected gap (`acc_gap`, `clp_gap`) measures knowledge, and it saturates near 1e19.

Root cause is mechanical: `run_olmo_bpb_eval.py:89-93` keeps only the record with `label == idx`, so every rank-classification task is scored as the gold continuation's bpb alone. That metric is dominated by fluency, which a better mixture improves everywhere at once. A "best in every domain" table built on gold-bpb would be true and uninformative.

## Internal Prior Work

- `.agents/projects/olmix_swarm_core_v2_and_full60_evals.md`: the 42-task devset explains only ~48% of Core v2 variance; per-task SNR screening at n=313 keeps 9 of 22 Core v2 tasks (reliability 0.975 vs 0.820). This screening recipe (se_noise vs sd_swarm) is the template for admitting any new domain task.
- `.agents/projects/devset_domain_priorities_WIP.md`: per-benchmark bpb decomposition already ties registers to evals (fiction to lambada, fact recall to jeopardy/naturalqs, science explainers to arc/sciq, how-to blogs to cooking/health subjects). This is a ready-made register-to-eval map.
- `.agents/projects/eval_prompt_formats_blend_and_mmlu_sl_verb.md`: BLEnD record schema; all four choices and `blend_choice_countries` are already on disk, so scoring distractors needs no rebuild.
- Memory `feedback_mmlu_noisy_at_scale`: raw MMLU accuracy pinned at 25% across the ladder; user chose to skip it. That decision still holds for accuracy; it does not hold for null-corrected margins.
- OLMIX target grid is 24 WebOrganizer topics x quality buckets (`olmix_domains.py`, `NUM_TOPICS = 24`). A domain eval can only be "won by mixing" if some topic cell moves it.

## External Prior Art

Cultural / everyday-knowledge benchmarks scorable on base models (English, permissive license):

| Name | Size | Format | License | Fit here |
|---|---|---|---|---|
| BLEnD MCQ (`nayeon212/BLEnD`, arXiv 2406.09948) | 5,081 unique (question, country) after dedup, 16 regions, 6 topic categories | MC-4, distractors are other countries' answers | CC-BY-SA-4.0 | already staged; needs contrastive scoring and a category split |
| CulturalBench (`kellycyy/CulturalBench`, arXiv 2410.02677) | Easy 1,227 MC-4 (HF v1); Hard 4,908 binary | MC-4 / T-F | CC-BY-4.0 | best second cultural set; 45 regions, 17 topics; use Easy only (binary is noise per OLMo 3) |
| Global-MMLU culturally-sensitive `en` subset (`CohereLabs/Global-MMLU`) | 792 CS items | MC-4 | Apache-2.0 | small; a tag over MMLU rather than a new set |
| NormAd (`akhilayerukola/NormAd`) | 2,615 | yes/no/neither | CC-BY-4.0 | etiquette; ternary, LLM-written stories; optional |
| Hendrycks ETHICS, Moral Stories | ~4k per config / 12k | binary | MIT | not cultural; binary; skip |
| GeoMLAMA, SeaEval cultural, ArabCulture, INCLUDE, WorldValuesBench, GlobalOpinionQA, CIVICS | various | tiny, NC, non-English, or distributional | mixed | skip |

Domain-knowledge benchmarks scorable on base models, mapped to WebOrganizer topics:

| Topic cell(s) | Eval | Notes |
|---|---|---|
| Health | MedMCQA (4,183), MedQA (1,270), HeadQA en (2,742, MIT, MC-5) | first two staged; PubMedQA is 3-way, skip |
| Science & Tech | SciQ, MMLU STEM, LAB-Bench dbqa/protocolqa (small n) | staged |
| Software Dev / Software | HumanEval, MBPP, mt_mbpp x17 gold-bpb | staged |
| Crime & Law | MMLU {international_law, jurisprudence, professional_law}; CaseHOLD (5.3k, MC-5, long holdings); LegalBench yes/no subtasks (binary, noisy) | CaseHOLD is the only base-scorable law MC; license on card unverified |
| Finance & Business | MMLU {econometrics, macro, micro, accounting, management, marketing, business_ethics}; Financial PhraseBank (NC license, sentiment only) | MMLU subjects are the usable part |
| History, Literature | Jeopardy gauntlet splits (Literature, US History, World History, Word Origins, Science; 2,117 total) | staged as one task; split by category |
| Politics | MMLU {gov_and_politics, us_foreign_policy, security_studies, public_relations} | |
| Religion, Social Life, Food & Dining, Sports & Fitness, Education & Jobs | BLEnD by category (food, sport, family, education, holidays, work-life); CulturalBench by topic | the human-centered arm |
| Any topic | Paloma `dolma_100_subreddits`, `c4_100_domains`, `m2d2_wikipedia`, `m2d2_s2orc`; uncheatable (ao3 fiction, arxiv physics/cs, github, bbc, wikipedia) | raw-text bpb, highest SNR, matches the DoReMi / Data Mixing Laws per-domain-loss convention |

Long-tail entity recall (TriviaQA, NQ-open, PopQA with popularity bins) is base-scorable and has the highest SNR of any task family at 1B (Heineman et al. 2025), but is not a "domain".

## Negative / Failed Leads

- No cultural benchmark reports any base-model or sub-7B result (BLEnD, CulturalBench, NormAd all chat-only, >=7B). ArabCulture shows 7-9B base models at chance in completion format. Small-scale signal is unproven and must be measured before spending a fleet.
- OLMIX (arXiv 2602.12237) claims only average bpb (+11.6%); it publishes no per-task or per-family win table. RegMix, DoReMi, Data Mixing Laws, Aioli make per-domain claims only via held-out loss on training domains. A per-domain downstream win table is a new claim, not a replication.
- Binary and 3-way tasks (BoolQ, PubMedQA, LegalBench yes/no, CulturalBench-Hard) oscillate between majority and minority class at small scale (OLMo 3 paper); DataDecide finds SocialIQA and BoolQ unpredictable at all scales.
- Heineman et al.: MMLU SNR is higher with 16 of 57 subjects. Per-subject MMLU is noisy and carries label noise (MMLU-Redux: up to 57% label error in virology). Category-level, subject-filtered aggregates are the honest unit.
- Contamination: MMLU is the most contaminated of nine benchmarks in web corpora (FLUX); SemEval-2026 organizers suspect BLEnD has been trained on by frontier labs. Decon in this repo covers CORE v2 only. BLEnD, CulturalBench, MedMCQA would need adding to the decon source before any final-scale claim.
- BLEnD gold answers have 63% mean annotator agreement; CulturalBench keeps 8.4% multi-answer items. Per-item noise is intrinsic; use category aggregates, never per-item.
- Echo unavailable (403), so prior Slack/GitHub discussion of eval choice was not searched.

## Evidence Map

#### Claim: gold-bpb on a rank-classification task measures fluency, not domain knowledge
- Support:
  - `run_olmix_fit.py` help: BLEnD gold-bpb ~90% single axis across the swarm.
  - Memory `project_mmlu_eval_wiring`: MMLU choice_logprob rise 92% explained by a zero-knowledge null.
  - OLMES (Gu et al. 2024): PMI normalization needed on ARC-C/CSQA/OBQA because answer priors dominate raw likelihood.
- Contradictions:
  - Heineman et al. 2025: bpb on the correct continuation raises SNR for model ranking. Not a contradiction: it ranks general quality well, which is exactly the confound here.
- Directness to Marin: measured on this repo's own runs. Direct.
- Confidence: stable.
- Action: score all candidates and report a contrastive, null-corrected metric per domain.

#### Claim: a contrastive metric on BLEnD isolates culture-specific knowledge
- Support:
  - BLEnD distractors are the same question's answers from other countries, so gold-vs-distractor margin conditions on question and register and leaves only the country-specific association.
  - All four continuations plus `blend_choice_countries` are already in the staged files.
- Contradictions:
  - At d512 / 1e17 the MMLU null-gap was negative (tiny models prefer generic strings). Small models may not separate "soccer" from "stick game" given a Korea prompt either.
  - No published base-model BLEnD numbers at any scale.
- Directness to Marin: untested; the pilot below settles it.
- Confidence: exploratory.
- Action: pilot on existing checkpoints before any rebuild.

#### Claim: per-country BLEnD is the wrong split for a topic mixture; per-category is the right one
- Support:
  - Mixture weights are over 24 WebOrganizer topics; no topic cell is "South Korea". BLEnD categories (food, sport, family, education, holidays, work-life) map onto Food & Dining, Sports & Fitness, Social Life, Education & Jobs, Religion.
  - Pooling 16 countries per category gives ~800 docs per task vs 120-360 per country, cutting doc-level noise.
- Contradictions:
  - Per-country tasks remain the human-centered headline for a paper figure; keep both, fit on categories.
- Directness to Marin: structural, follows from the grid definition.
- Confidence: replicated reasoning, not yet measured.
- Action: rebuild BLEnD with a category join (SAQ file carries topic per question ID; MC IDs are `<Country>-en-<ID>_<perm>`).

#### Claim: raw-text per-domain bpb is the highest-SNR "every domain" evidence and matches prior mixing papers
- Support:
  - DoReMi, Data Mixing Laws, Aioli, RegMix all report per-domain held-out loss.
  - Paloma and uncheatable already log in-loop on every run in the sweep; zero new compute.
  - Signal-and-noise: bpb metrics have 5-20x the SNR of accuracy at 1B.
- Contradictions:
  - KR-Test (arXiv 2601.03505): perplexity confounds stylistic mimicry with factual internalization. So raw-text bpb is necessary but not sufficient for a "knows the domain" claim.
- Directness to Marin: direct, already collected.
- Confidence: stable.
- Action: make this Tier 1 of the claim; downstream contrastive tasks are Tier 2.

#### Claim: MMLU category subsets carry usable knowledge signal only as null-corrected gaps with subject filtering
- Support:
  - Memory: `acc_gap` grows with scale (+0.005 to +0.012, best cell +0.031 at +9 sigma); `clp_gap` plateaus at +0.03-0.05 from 1e19.
  - Heineman: 16-subject MMLU subset has higher SNR than all 57.
  - Bhagia et al.: MMLU accuracy predictable within 2 pts from a 190M-1.3B ladder via task-loss.
- Contradictions:
  - E2LM and Madaan et al.: MMLU is noise below ~1B in MC format. Consistent: use cloze/rc, not verbalized MC, and never raw accuracy.
- Directness to Marin: direct.
- Confidence: replicated.
- Action: keep the 4 categories as gold-bpb in the objective (already there); add the null-gap metric for reporting; screen subjects by swarm SNR.

## Recommended Next Experiments

#### 1. Pilot: does a contrastive metric on BLEnD and MMLU have swarm signal at 157M?
- Minimum experiment: add a `--score-all-choices` path to `run_olmo_bpb_eval.py` that keeps every record, emits per-doc per-choice `logprob` and byte length, and computes three metrics per task: `gold_bpb` (current), `margin_bpb` = mean(distractor bpb) - gold bpb, and `null_gap` = observed choice_logprob minus the gold-marginal null (port `experiments/scaling_law_sweeps/mmlu/analyze_mmlu_signal.py`). Run over ~80 already-evaluated OLMIX proxies (40 dclm + 40 hq, spread across the swarm) with `--merge --done-marker` on `blend_*` and `mmlu_*`; plus the 8 corpus runs at 3e18 d512 for between-corpus spread.
- Baseline/control: `gold_bpb` on the same runs (known ~90% single-axis).
- Expected signal: `margin_bpb` and `null_gap` per task should have reliability = 1 - se_noise^2 / sd_swarm^2 above ~0.7 (bootstrap se over docs), and the first principal component of the 16 country tasks should drop well below 90%.
- Falsifier: reliability under 0.5 or PC1 still >80% means 157M proxies cannot see culture-specific knowledge; fall back to Tier 1 raw-text domain bpb plus the 42-task devset.
- Cost/risk: ~80 x 16 min of v5p-8 / v6e-4 (TRC-free); GCS in-region only. Risk: HF rate limit; use the wave throttle.
- Sources: `run_olmo_bpb_eval.py:89-126`; memory `project_mmlu_eval_wiring`; Gu et al. 2024.

#### 2. Rebuild BLEnD as category tasks and add CulturalBench-Easy
- Minimum experiment: extend `build_blend_bpb_requests.py` to join SAQ topic by question ID and emit `blend_cat_{food,sport,family,education,holidays,worklife}/rc_5shot` (pooled countries, 5 shots per category drawn from held-out IDs); write `build_culturalbench_bpb_requests.py` from the same template for `kellycyy/CulturalBench` Easy (MC-4, 17 topics, add a `cb_topic` field). Stage under `eval_datasets/olmo_in_loop_evals/oe_eval_tasks/` in all 5 regions with `rsync -r` and never `-d`.
- Baseline/control: per-country BLEnD tasks.
- Expected signal: category tasks show higher reliability than country tasks and respond to Food & Dining / Sports & Fitness weight in a mixture.
- Falsifier: category tasks no more reliable than country tasks.
- Cost/risk: one CPU job to build; one backfill wave. CulturalBench HF v1 has 1,227 Easy items, so expect wider CIs than BLEnD.
- Sources: `nayeon212/BLEnD` card; `kellycyy/CulturalBench` card; arXiv 2410.02677.

#### 3. Assemble the per-domain suite aligned to the 24 topics
- Minimum experiment: define `DOMAIN_TASKS: dict[topic, tuple[task,...]]` in `olmix_tasks.py` with: Health = medmcqa, medqa_en, headqa_en; Science = sciq, mmlu_stem, lab_bench; Software = humaneval, mbpp, mt_mbpp; Law = mmlu law subjects (+ CaseHOLD if a pilot shows signal); Finance/Business = mmlu econ/business subjects; History/Literature = jeopardy split by its 5 categories; Politics = mmlu politics subjects; Everyday-culture = BLEnD categories + CulturalBench topics; plus raw-text bpb from Paloma/uncheatable mapped to the same topics. Screen every task by swarm reliability as in `olmix_swarm_core_v2_and_full60_evals.md` section 4.
- Baseline/control: the existing 42-task flat objective.
- Expected signal: at least 8 topic cells with a reliable, topic-matched task family.
- Falsifier: fewer than ~5 reliable topic-matched families, in which case the claim reduces to raw-text bpb per domain.
- Cost/risk: builders for jeopardy split, headqa, mmlu subject regroup, CaseHOLD; one backfill wave per new family.
- Sources: hendrycks/test `categories.py`; olmes `task_suites.py`; `soldni/jeopardy`.

#### 4. The "best in every domain" experiment
- Minimum experiment: for each topic-matched domain d, `run_olmix_fit.py --metric domain:<d>` (new branch at `run_olmix_fit.py:212`) fits a specialist mixture; also fit the all-domain mixture. Train specialists, all-domain, and natural at one target scale (d1024 or d1536 at 3e19, matching the 10k ladder) and report, per domain, the contrastive metric for natural, all-domain, specialist-on-d, and best-other-specialist.
- Baseline/control: natural mixture and the 42-task-devset mixture.
- Expected signal: specialist-on-d beats every other arm on d (validates the eval is domain-sensitive); all-domain beats natural on every d and is within the noise band of specialist-on-d on most (the headline).
- Falsifier: specialists do not beat all-domain on their own domain, meaning the evals are not domain-discriminative even at target scale.
- Cost/risk: ~(D + 2) target-scale runs per corpus; largest item in the plan. Add BLEnD, CulturalBench, MedMCQA, MedQA to the decon source before these runs.
- Sources: OLMIX arXiv 2602.12237 Table 9; `olmix_swarm_core_v2_and_full60_evals.md`.

## Hypothesis Queue Update
- Add: H1 contrastive BLEnD/MMLU metrics have swarm signal at 157M (exploratory). H2 category-split BLEnD is more identifiable than country-split under a topic mixture (exploratory). H3 raw-text per-domain bpb is the reliable Tier-1 "every domain" evidence (stable).
- Revise: "BLEnD is diagnostic only" becomes "BLEnD gold-bpb is diagnostic only"; the contrastive form is untested.
- Falsify / stop: raw MMLU accuracy at sweep scale (already stopped); binary tasks (PubMedQA, LegalBench yes/no, CulturalBench-Hard, ETHICS) as domain targets.
- Promote: none until Experiment 1 reports.

## Source Ledger
| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| run_olmo_bpb_eval.py | Marin code | `experiments/scaling_law_sweeps/olmo_bpb/run_olmo_bpb_eval.py:89-126` | gold-only scoring; bpb formula | high | |
| build_blend_bpb_requests.py | Marin code | `experiments/scaling_law_sweeps/olmo_bpb/` | BLEnD record schema, distractors on disk | high | |
| run_olmix_fit.py | Marin code | `experiments/data_mixing/run_olmix_fit.py:212-240` | --metric hook; BLEnD 90% single-axis | high | |
| olmix_tasks.py | Marin code | `experiments/data_mixing/olmix_tasks.py` | 42-task basis, MMLU_BPB | high | |
| olmix_swarm_core_v2_and_full60_evals.md | logbook | `.agents/projects/` | SNR screening recipe; devset explains 48% Core v2 | high | |
| devset_domain_priorities_WIP.md | logbook | `.agents/projects/` | register-to-eval map | medium | WIP |
| project_mmlu_eval_wiring | memory | session memory | 92/8 fluency/knowledge; null-gap method | high | measured on 180 runs |
| BLEnD | paper + HF | arxiv.org/abs/2406.09948; hf nayeon212/BLEnD | formats, categories, agreement 63% | high | CC-BY-SA-4.0 |
| CulturalBench | paper + HF | arxiv.org/abs/2410.02677; hf kellycyy/CulturalBench | sizes, formats | medium | HF v1 counts differ from paper v2 |
| Global-MMLU | HF | hf CohereLabs/Global-MMLU | CS subset 792 | high | |
| NormAd | paper + HF | arxiv.org/abs/2404.12464 | ternary etiquette | medium | |
| OLMES | paper | arxiv.org/abs/2406.08446 | CF vs MCF; PMI normalization | high | |
| DataDecide | paper | arxiv.org/abs/2504.11393 | correct-prob best at small scale; SIQA/BoolQ unpredictable | high | |
| Task scaling ladders | paper | arxiv.org/abs/2412.04403 | task-loss predictability | high | |
| Signal and Noise | paper | arxiv.org/abs/2508.13144 | bpb SNR; MMLU 16-subject subset | high | |
| Madaan et al. 2024 | paper | arxiv.org/abs/2406.10229 | MMLU MC at chance below 1B | high | |
| OLMo 3 | paper | arxiv.org/abs/2512.13961 | Base Easy composition; binary tasks dropped | high | |
| OLMIX | paper | arxiv.org/abs/2602.12237 | average-bpb-only claim; Table 9 task list | high | |
| RegMix / DoReMi / Data Mixing Laws / Aioli | papers | 2407.01492 / 2305.10429 / 2403.16952 / 2411.05735 | per-domain = held-out loss convention | high | |
| FLUX contamination | paper | arxiv.org/pdf/2603.13972 | MMLU most contaminated | medium | |
| hendrycks/test categories.py | external code | github.com/hendrycks/test | category to subject mapping | high | |
| olmes task_suites.py | external code | github.com/allenai/olmes | oe-eval families | high | |
| Jeopardy gauntlet | HF | hf soldni/jeopardy | 5 category splits | high | j-archive TOS |
| CaseHOLD | HF | hf casehold/casehold | law MC-5 | medium | license unverified |
| HeadQA | HF | hf dvilares/head_qa | health MC-5, MIT | high | |
| Anthropology critique of culture benchmarks | paper | arxiv.org/abs/2510.05931 | consensus assumption caveat | medium | |

## Handoff
- Suggested issue `Prior work` block: "BLEnD and MMLU-by-category are already staged as gold-bpb tasks; repo measurements show gold-bpb is ~90% general quality (BLEnD) and 92% LM / 8% knowledge (MMLU). External literature (OLMES, DataDecide, Signal-and-Noise) agrees that small-scale signal needs correct-answer likelihood but that answer-prior fluency must be normalized out. No cultural benchmark has published base-model or sub-7B results. Prior mixing papers claim per-domain wins only via held-out domain loss; OLMIX claims average bpb only."
- Suggested logbook entry: link this file; record Experiment 1 launch and its reliability table when run.
- Open questions: does the SAQ topic join cover every MC question ID; CaseHOLD license; whether Paloma per-subdomain (100 subreddits, 100 C4 domains) is logged per subdomain or only as a subset mean.
- Stop reason: ranked experiments stopped changing once the small-scale-signal literature converged with the repo's own null-gap and BLEnD single-axis findings.

## Addendum 2026-09-07: cultural composite and the `mmlu_sl_verb` rendering decision

User decisions: MMLU category subsets accepted as-is; every cultural candidate will be rendered
as an lm-eval `multiple_choice` task in the CRFM fork's `mmlu_sl_verb` shape (all choices scored;
seven soft metrics `acc, acc_norm, bpb, logprob, choice_logprob, choice_prob_norm,
choice_logprob_norm`; logged samples carry every per-choice log-likelihood for the null-gap
analysis in `experiments/scaling_law_sweeps/mmlu/analyze_mmlu_signal.py`). Route: one YAML per
dataset under a `custom_tasks/` dir installed by the `install_custom_task_path()` mechanism from
`experiments/scaling_law_sweeps/olmes_base/olmes_custom_tasks.py`, run through the existing
mmlu launcher pattern (`experiments/scaling_law_sweeps/mmlu/`). The bpb-runner change in
Experiment 1 is no longer needed. Render each dataset twice from the same rows: a `_verb`
variant (lettered choices in text and continuation, as sl_verb) and a `_cloze` variant (raw
answer continuations, no letters), because OLMES, ArabCulture, and Global PIQA all show
verbalized MC at chance for small base models while cloze scoring is not; keep whichever has
the higher swarm reliability.

### Cultural candidates, screened (English, fixed candidates, per-region metadata)

Tier A, human-sourced, permissive or evaluation-permitted:

| Dataset | Items | Regions / topics | Format | License / policy | Role |
|---|---|---|---|---|---|
| BLEnD MCQ (`nayeon212/BLEnD`) | 5,081 unique | 16 regions, 6 topics | MC-4, distractors from other countries | CC-BY-SA-4.0 | anchor |
| SemEval-2026 Task 7 (BLEnD extension, github `BLEnD-SemEval2026/SemEval-2026-Task-7`) | counts unverified | ~31 regions; adds en-AU, en-SG, ar-EG/MA/SA, bg, es-EC, eu, fr, ga, ja, sv, ta-LK/SG, tl, zh-SG/TW, ms-SG; English MCQ for all | MC-4 | no formal license; "not for fine-tuning or few-shot learning", no redistribution | superset of BLEnD; dedupe the original 16; use 0-shot or BLEnD-sourced shots; keep staged copies private |
| CulturalBench-Easy (`kellycyy/CulturalBench`) | 1,227 (HF v1 only) | 45 regions, 17 topics | MC-4 | CC-BY-4.0 | breadth; Hard shares the same questions, do not use both |
| TriviaRoomQA (arXiv 2607.21445) | 3,300 en | 24 categories, region + period metadata, Europe-centric | MC-4 | CC-BY-SA-4.0; hosting unverified | trivia-flavored; quiz-DB origin means contamination risk |
| FORK (github `shramay-palta/FORK_ACL2023`) | 184 | food customs, country field | MC-2 | CC-BY-4.0 | too small alone; fold into a food topic |
| Global PIQA `eng_latn` (`mrlbenchmarks/global-piqa-nonparallel`) | 100 | topic + cultural-score metadata | 2-way cloze | CC-BY-SA-4.0, no-training clause | sanity item only |

Tier B, larger but web-derived or LLM-generated (secondary, contamination-prone):

| Dataset | Items | Format | License | Why secondary |
|---|---|---|---|---|
| NormAd (`akhilayerukola/NormAd`) | 2,615, 75 countries | yes/no/neither | CC-BY-4.0 | LLM-written stories from Cultural Atlas; ternary |
| CultureForest (arXiv 2606.01879) | 5,378, 53 regions, 8 domains | MC-4 | unstated; release unverified | GPT-4.1-generated; possible fixed gold position |
| EtiCor (`Exploration-Lab/EtiCor`) | 36,347, 5 regions | binary | CC-BY-NC-SA | crawled web; binary is noise at small scale |
| SANSKRITI (Google Drive) | 21,853, India only | MC-4 | CC-BY-4.0 | Wikipedia-derived; single country |
| CultureAtlas | 10k | binary | CC-BY-SA-4.0 | Wikipedia-derived; binary |

Skip: DOSA (open-ended, 10-49 per state), CultureBank (descriptors, no candidates), MultiNativQA
(NC, paragraph answers), CDEval and WorldValuesBench (values, no gold), CaLMQA (long-form),
COPAL-ID / IndoCulture / mCSQA (non-English), CulturalTeaming v0.1 (likely a CulturalBench
subset), Shen et al. 2024 (repo 404).

### Robustness design for the composite

- Unit of analysis is (region, topic), never the item; report per-region and per-topic means
  with bootstrap CIs over items and admit cells by swarm reliability.
- Triangulate: a region covered by both BLEnD/SemEval and CulturalBench gets a pooled
  estimate; disagreement between sources on the same region flags an artifact.
- Two distractor regimes need separate nulls: BLEnD/SemEval distractors are other regions'
  answers (cross-cultural contrast), CulturalBench and TriviaRoomQA distractors are
  same-region plausible wrong answers.
- Add every Tier A set to the decon source before target-scale runs; SemEval-2026 forbids
  training and redistribution, so staged copies stay in the private GCS eval cache.
- Expected item counts: Tier A pooled is roughly 10k English MC items across ~50 regions,
  enough for ~200 per region on the BLEnD family and ~27 per region on CulturalBench.

## Addendum 2 (2026-09-07): deep dive on the cultural composite candidates

Citation counts are Semantic Scholar on 2026-09-07.

| Benchmark | Paper | Cites | Format | Chance | English items available now | Regions | Source | License / policy |
|---|---|---|---|---|---|---|---|---|
| BLEnD | arXiv 2406.09948, NeurIPS 2024 D&B | 238 | MC-4 | 25% | 5,081 unique (question, country); 305,939 distractor draws (~60 per item) | 16 | human annotators | CC-BY-SA-4.0 |
| CulturalBench | arXiv 2410.02677; ACL 2025 | 79 (+19 arXiv entry) | Easy MC-4; Hard T/F on the same questions | 25% | 1,227 (HF v1) | 45 | human red-teaming | CC-BY-4.0 |
| SemEval-2026 Task 7 | arXiv 2605.02601 | 1 | MC-4 (+SAQ track) | 25% | gold zips hold answer letters only; questions live on CodaBench comp 10281 (login, auto-approve registration). Original 16 regions = BLEnD verbatim (37,557 draws); 14 new regions add 9,457 draws (en-AU 513, ar-EG 368, ar-MA 543, ar-SA 444, bg 648, es-EC 977, eu 1,075, fr 307, ga 856, ja 410, sv 447, ta-LK 1,114, tl 1,327, zh-SG 428) | 30 | human annotators | no formal license; research only, no fine-tuning or few-shot use, no redistribution |
| FORK | ACL 2023 (Palta & Rudinger) | 69 | question + 2 options | 50% | 184 (153 excluding 31 "Underspecified" with US-default gold); China 66, USA 58, Japan 28, India 17, rest <=4 | 11 | human-written | CC-BY-4.0 |
| Global PIQA | arXiv 2510.24081 | 1 | 2-way PIQA cloze | 50% | eng_latn 100; plus 13,500 non-English items whose `eng_translated0/1` render prompt+solution as one English sentence each; 8,153 rows flagged `approx_cultural_score==1`, 555 flagged `llm_used` | 136 varieties | human contributors | CC-BY-SA-4.0, no training on it or on synthetic data seeded from it |
| NormAd | arXiv 2404.12464, NAACL 2025 | 48 | 3-way yes/no/neutral judgment of a story vs a country's norm | 33% | 2,615 (yes 943 / no 875 / neutral 797); 15-53 per country | 75 | LLM-written stories from Cultural Atlas, human-checked | CC-BY-4.0 |
| CultureForest | arXiv 2606.01879 | 0 | MC-4 (Easy) + open | 25% | 5,378; domains Etiquette 1,262 / Dates 1,223 / Business 959 / Communication 685 / Greetings 640 / Naming 412 / Religion 131 / Family 66; HF `TommyYe/CultureForest` | 53 | GPT-4.1 from Cultural Atlas, human-verified; use `Options_Shuffle` (pre-shuffle D is always gold) | MIT |
| EtiCor | arXiv 2310.18974, EMNLP 2023 | 43 | binary acceptable / not | 50% | 36,347 over 5 regions | 5 | crawled web (gov sites, tour guides, tweets) | CC-BY-NC-SA-4.0 |
| TriviaRoomQA | arXiv 2607.21445 | 0 | MC-4 | 25% | 3,300 en; People 660, Entertainment 360, Sport 270, Food 120; Europe 1,652 | continent tag | human quiz DB | CC-BY-SA-4.0 per paper; no LICENSE file in repo |
| Global-MMLU en CS | arXiv 2412.03304 | - | MC-4 | 25% | 792, MMLU-verbatim rows | Western-dominated | - | Apache-2.0 |

Not all are multiple choice, but all have a fixed candidate set, so all render as lm-eval
`multiple_choice` tasks: MC-4 (BLEnD, CulturalBench-Easy, SemEval, CultureForest,
TriviaRoomQA), 2-way (FORK, Global PIQA), 3-way (NormAd), binary (EtiCor). Chance rates
differ (25/33/50%), so a macro average must use chance-centered accuracy or the per-benchmark
null-gap rather than raw metrics.

Global PIQA English rendering has no prompt/continuation split; score each merged sentence
as the continuation after a fixed per-variety description line. Translation provenance of
`eng_translated*` is contributor-supplied per the paper (UNVERIFIED per row).

Recommended six for the macro average: BLEnD, CulturalBench-Easy, CultureForest-Easy, NormAd,
Global PIQA cultural-flagged English renderings, and the 14 new SemEval regions if the
CodaBench questions can be obtained (0-shot only). Fallback sixth: EtiCor (NC, binary,
crawled) or fold FORK into a food/etiquette slice. Exclude TriviaRoomQA (pop trivia,
Europe-heavy) and Global-MMLU CS (duplicates MMLU).

Downloaded working copies (session scratchpad, not durable):
`.../scratchpad/bench/` holds SemEval gold TSVs, `fork.jsonl`, Global PIQA TSVs, CultureForest,
NormAd, Global-MMLU en, BLEnD MCQ parquet.

## Addendum 3 (2026-09-07): >40-citation cut and EtiCor finding

User selected the >40-citation set: BLEnD (238), CulturalBench (79), FORK (69), NormAd (48),
EtiCor (43). EtiCor inspection: the GitHub repo holds no data (Google-form access; HF copy
`Exploration-Lab/EtiCor` is gated and refused the repo token), and
`Dataset/Negative_Sentence_Generator.py` shows the "not appropriate" class is produced by
`minwhoo/bart-base-negative-claim-generation` applied to crawled positive sentences. The task
therefore largely measures negation detection, and paper Table 11 shows label noise
("English is widely spoken in business and government contexts..." gold = not appropriate).
Recommendation: drop EtiCor from the macro average; the working set is four benchmarks.
BLEnD example rows show degenerate golds ("not-applicable" for Thanksgiving in Ethiopia) and
contested distractors (UK Christmas: "family gathering" is a West Java answer but also true of
the UK); filter `not-applicable` golds and expect the 63% annotator-agreement ceiling.

User dropped NormAd (2026-09-07): the "neutral" label depends on a rule of thumb the model
never sees, so items are ambiguous by construction. Working cultural set is now BLEnD,
CulturalBench-Easy, FORK. Candidates to reach 5-6 if the citation bar is relaxed: Global PIQA
cultural-flagged English renderings (8,153 items, community benchmark, 2025) and
CultureForest-Easy (5,378, MIT, synthetic from Cultural Atlas, 2026); SemEval-2026 new regions
need CodaBench access.

NormAd reinstated as a binary set (2026-09-07): drop `neutral`, keep yes 943 / no 875 = 1,818
items over 75 countries (12-31 each, median 24); subaxes basic_etiquette 446, eating 416,
visiting 415, gift_giving 356. FORK: drop the 31 Underspecified items (153 remain).
User decision: no country-swap or other extra controls for NormAd; run it as a plain 2-way
task like the others. Aggregate by subaxis or region, not per country.

### Prompt-cleaning rules and shot count (2026-09-07)

- BLEnD: 782/5,081 questions end in a parenthetical. 737 are SAQ-era format instructions
  starting "(Provide ..." (Arabic numerals, HH:MM, MM/DD, official name) that are meaningless
  once the answer is a lettered choice: strip any trailing "(Provide ...)" tail. The other 45
  are answer-type hints "(e.g., breakfast, lunch, dinner)", "(e.g. Monday, Tuesday)",
  "(e.g., ___ vs ___)" that disambiguate the question: keep them. The builder already splits
  the MC prompt at " Without any explanation".
- CulturalBench-Easy: no instruction tails; 86/1,227 are composite "Read the following
  statements and select the option that include all the appropriate statements" items whose
  options are roman-numeral sets. Keep them unchanged (the sentence is part of the question
  and the choices only make sense with it); optionally tag `composite=True` for analysis.
- FORK: nothing to strip. NormAd: 1,804/1,818 stories end in "Is what X did (socially)
  acceptable?"; nothing to strip.
- Shot count: 5-shot everywhere, matching `mmlu_sl_verb` (dev split, `first_n`). Shots are
  chosen round-robin over the gold index in source order (see `cultural_eval_suite.md`):
  a naive first-five gave NormAd five Egypt / yes shots because its CSV is sorted by
  country and label.

## Addendum 4 (2026-09-07): code landed

`experiments/scaling_law_sweeps/cultural/` (sources + builder + task set + runner +
launcher + consolidator + null-gap analysis), `mmlu/consolidate_mmlu_categories.py`,
`analyze_mmlu_signal.py --by-category`, shared `lm_eval_include_path.py`, and
`tests/scaling_law_sweeps/test_cultural_sl_verb.py`. Rules of record moved to
`cultural_eval_suite.md`. The generated YAMLs were loaded in the pinned lm-eval fork and
rendered the exact sl_verb 5-shot layout for all four benchmarks.
