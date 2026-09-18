# Domain-eval benchmarks: MMLU subdivisions and the Cultural suite

The two eval families behind the domain-targeted mixture fits. Both render as the
`mmlu_sl_verb` shape (verbalized multiple choice, lettered choices, 5-shot) and report the
seven soft metrics (`acc`, `acc_norm`, `bpb`, `logprob`, `choice_logprob`,
`choice_prob_norm`, `choice_logprob_norm`); the mixture objective maximizes raw
`choice_logprob` (user decision 2026-09-11; not the length-normalised `_norm` variant). Task registries: `mmlu/mmlu_tasks_set.py` and
`cultural/cultural_tasks_set.py`; category tables: `olmo_bpb/mmlu_olmes.py` (verbatim from
Hendrycks et al.'s `categories.py`) and `.agents/projects/cultural_eval_suite.md` (design
decisions and cleaning rules).

## MMLU subdivisions (the official Hendrycks categories)

57 subjects, evaluated per subject and aggregated per category (soft metrics: unweighted
mean over subjects). Paper: [MMLU (Hendrycks et al., 2021)](https://arxiv.org/abs/2009.03300).

### STEM (18 subjects)

abstract_algebra · astronomy · college_biology · college_chemistry ·
college_computer_science · college_mathematics · college_physics · computer_security ·
conceptual_physics · electrical_engineering · elementary_mathematics · high_school_biology ·
high_school_chemistry · high_school_computer_science · high_school_mathematics ·
high_school_physics · high_school_statistics · machine_learning

### Humanities (13 subjects)

formal_logic · high_school_european_history · high_school_us_history ·
high_school_world_history · international_law · jurisprudence · logical_fallacies ·
moral_disputes · moral_scenarios · philosophy · prehistory · professional_law ·
world_religions

### Social sciences (12 subjects)

econometrics · high_school_geography · high_school_government_and_politics ·
high_school_macroeconomics · high_school_microeconomics · high_school_psychology ·
human_sexuality · professional_psychology · public_relations · security_studies ·
sociology · us_foreign_policy

### Other (14 subjects — measured but NOT a reported domain)

anatomy · business_ethics · clinical_knowledge · college_medicine · global_facts ·
human_aging · management · marketing · medical_genetics · miscellaneous · nutrition ·
professional_accounting · professional_medicine · virology

## Cultural suite (`cultural_sl_verb`, 19 tasks / 8,056 test documents)

Four benchmarks; suite macro = unweighted mean over the four (BLEnD itself is the
unweighted mean over its 16 country tasks). Chance rates differ (25% for the MC-4
benchmarks, 50% for the 2-way ones), so cross-benchmark accuracy comparisons use
chance-centered accuracy.

| Benchmark | Format | Test docs | Paper | Data |
|---|---|---|---|---|
| **BLEnD MCQ** | MC-4, distractors drawn from other countries | 4,873 over 16 country tasks | [arXiv 2406.09948](https://arxiv.org/abs/2406.09948) (NeurIPS 2024 D&B) | [nayeon212/BLEnD](https://huggingface.co/datasets/nayeon212/BLEnD) |
| **CulturalBench-Easy** | MC-4 | 1,222 | [arXiv 2410.02677](https://arxiv.org/abs/2410.02677) (ACL 2025) | [kellycyy/CulturalBench](https://huggingface.co/datasets/kellycyy/CulturalBench) |
| **FORK** | 2-way, dining customs | 148 | [Palta & Rudinger, ACL 2023](https://aclanthology.org/2023.findings-acl.631/) | [shramay-palta/FORK_ACL2023](https://github.com/shramay-palta/FORK_ACL2023) |
| **NormAd** | 2-way yes/no etiquette judgments | 1,813 | [arXiv 2404.12464](https://arxiv.org/abs/2404.12464) (NAACL 2025) | [akhilayerukola/NormAd](https://huggingface.co/datasets/akhilayerukola/NormAd) |

BLEnD's 16 country tasks: Algeria, Assam, Azerbaijan, China, Ethiopia, Greece, Indonesia,
Iran, Mexico, North Korea, Northern Nigeria, South Korea, Spain, UK, US, West Java.

Selection notes (full reasoning in `.agents/projects/domain_evals_research.md`): rejected
EtiCor (BART-generated negations; form-gated), CulturalBench-Hard (same questions as
Easy), NormAd's neutral class (depends on a hidden rule), TriviaRoomQA (pop trivia),
Global-MMLU CS subset (MMLU-verbatim), and SemEval-2026 Task 7 (no few-shot, no
redistribution). Cleaning rules (dedup per (country, question), dropped `not-applicable`
golds, dropped FORK's 31 underspecified items, balanced 5-shot dev splits) live in
`cultural/cultural_sources.py` and are pinned by tests.
