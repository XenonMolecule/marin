# Eval prompt formats: BLEnD (rc:bpb) and `mmlu_sl_verb`

Two evals in this repo use opposite few-shot conventions. This doc shows a real
rendered prompt for each and the code that produces it.

| | BLEnD | `mmlu_sl_verb` |
|---|---|---|
| Formulation | OLMES **rc** (cloze) — choices are **not** in the context | **verbalized MC** — lettered choices in the context *and* in the continuation ("verb" = verbalized) |
| Shots | 5, from the **first 5 questions per country** (held out of test) | 5, from MMLU's **`dev` split** (`fewshot_split: dev`, `sampler: first_n`) |
| Built where | staged offline into `oe_eval_tasks/` by `build_blend_bpb_requests.py` | rendered in-harness by lm-eval (CRFM fork YAML) |
| Scored by | `olmo_bpb/run_olmo_bpb_eval.py` — keeps only the gold continuation, reports `bits_per_byte` | Levanter lm-eval harness — `acc`, `acc_norm`, `bpb`, `logprob`, `choice_logprob`, `choice_prob_norm`, `choice_logprob_norm` |
| Role | diagnostics only, **not** in the olmix mixture objective | scaling-law sweeps (soft metrics stay informative below the 25% accuracy floor) |

---

## 1. BLEnD — everyday cultural knowledge, rc:bpb

Source: `experiments/scaling_law_sweeps/olmo_bpb/build_blend_bpb_requests.py`
(prompt affixes come from `mmlu_olmes.cloze_query`, i.e. ai2's
`oe_eval.tasks.utils.make_cloze_prompt` with OLMES defaults).

### Structure

```
<description>\n\n
Question: <fewshot q 1>\nAnswer: <gold 1>\n\n
...  (5 shots)
Question: <test q>\nAnswer:
```

* Description: `"The following are questions about everyday life in {country}.\n\n"`
  (display names come from `BLEND_COUNTRIES`, e.g. `US` → "the United States").
* Each shot is `cloze_query(q) + " " + gold`, joined by `\n\n`; the block ends with
  a trailing `\n\n` before the test query.
* The **choices are never shown**. The 4 candidate continuations are the raw answer
  strings, each prefixed with a leading space.
* Few-shot examples are the first `BLEND_NUM_SHOTS = 5` questions per country in ID
  order, and they are **removed from the test set** (OLMES `first_n` convention).
* Dedup: the 305,939-row `mc_questions_file_v1.1.json` collapses to 5,081 documents
  (one per `(country, ID)`; the extra rows are distractor permutations with identical
  gold). Per-country doc counts after dropping the 5 shots, e.g. South Korea 366 → 361.

### Real example (`blend_south_korea`, `rc_5shot`)

Context sent to the model:

```
The following are questions about everyday life in South Korea.

Question: What is a common snack for preschool kids in South Korea?
Answer: cookie

Question: What is a popular food to go with beer in South Korea?
Answer: fried chicken

Question: What is the most popular fruit in South Korea?
Answer: apple

Question: What is a common school cafeteria food in South Korea?
Answer: kimchi

Question: What is a popular snack at an amusement park in South Korea?
Answer: churros

Question: What is a popular afterschool sport for elementary schools in South Korea?
Answer:
```

Continuations scored for this document (one record each; gold is `idx == label`):

```
" American Football"     idx=0   (choice country: US)
" hide and seek"         idx=1   (Assam)
" soccer"                idx=2   (South_Korea)   <-- gold, label=2
" stick game"            idx=3   (Northern_Nigeria)
```

### On-disk record (one line of `requests.jsonl.gz`)

```json
{
  "request_type": "loglikelihood",
  "doc": {
    "index": 0,
    "query": "Question: What is a popular afterschool sport for elementary schools in South Korea?\nAnswer:",
    "choices": ["American Football", "hide and seek", "soccer", "stick game"],
    "gold": 2
  },
  "request": {
    "context": "The following are questions about everyday life in South Korea.\n\nQuestion: What is a common snack for preschool kids in South Korea?\nAnswer: cookie\n\n... (5 shots) ...\n\nQuestion: What is a popular afterschool sport for elementary schools in South Korea?\nAnswer:",
    "continuation": " soccer"
  },
  "idx": 2,
  "task_name": "blend_south_korea",
  "doc_id": 0,
  "native_id": "Al-en-17_14",
  "label": 2,
  "blend_country": "South_Korea",
  "blend_choice_countries": {"A": "US", "B": "Assam", "C": "South_Korea", "D": "Northern_Nigeria"}
}
```

Layout on disk (16 country tasks):

```
<root>/oe_eval_tasks/blend_<country>/rc_5shot/{config.json,requests.jsonl.gz}
```

The bpb runner keeps records where `label == idx`, so all 4 records are written but
only the gold continuation is scored — that keeps a future full-MC/rc variant able to
reuse the same files unchanged.

Build:

```bash
python -m experiments.scaling_law_sweeps.olmo_bpb.build_blend_bpb_requests \
    --output-dir /tmp/blend_bpb
# then: gcloud storage cp -r ...   (never rsync -d)
```

---

## 2. `mmlu_sl_verb` — verbalized MC, 5-shot

Source: the CRFM lm-eval fork, `lm_eval/tasks/mmlu/sl_verb/`
(`_default_template_yaml` + one YAML per subject + the group YAML).
Wired up in `experiments/scaling_law_sweeps/mmlu/mmlu_tasks_set.py`.

### Template (`_default_template_yaml`)

```yaml
dataset_path: hails/mmlu_no_train   # cais/mmlu without the auxiliary_train split, pure parquet
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_choice: "{{['A. ' + choices[0], 'B. ' + choices[1], 'C. ' + choices[2], 'D. ' + choices[3]]}}"
doc_to_target: answer
```

Per-subject YAML supplies only the description, dataset config, tag, and alias:

```yaml
"dataset_name": "anatomy"
"description": "The following are multiple choice questions (with answers) about anatomy.\n\n"
"include": "_default_template_yaml"
"tag": "mmlu_sl_verb_stem"
"task": "mmlu_sl_verb_anatomy"
"task_alias": "anatomy"
```

### Few-shot structure

```
<description>\n\n
<doc_to_text(shot 1)> <doc_to_choice(shot 1)[answer]>\n\n
...  (5 shots = the dev split, in order, first_n — MMLU's dev split is exactly 5/subject)
<doc_to_text(test doc)>
```

Key points:

* **The letter is part of the continuation**, not just the context: the target is
  `"D. The second and third pharyngeal arches"`, not `"D"` and not the bare answer
  text. That is what makes this the verbalized variant — the model's choice
  log-probs cover letter *and* answer string.
* Shot separator is `\n\n`; the target delimiter after `Answer:` is a single space.
* `num_fewshot` is **not** set in the YAML — the caller always passes it.
  `EvalTaskConfig.num_fewshot` is required in marin and
  `convert_to_levanter_task_config` always forwards it, so there is no
  "inherit the YAML default" path.
* 0-shot and 5-shot must be **separate runs** — they share the lm-eval task name
  `mmlu_sl_verb` and would collide in the harness's task dict. Disambiguate by alias.

### Task configs used in this repo

```python
MMLU_SL_VERB_BY_SHOTS = {
    0: EvalTaskConfig("mmlu_sl_verb", 0, task_alias="mmlu_sl_verb_0shot"),
    5: EvalTaskConfig("mmlu_sl_verb", 5, task_alias="mmlu_sl_verb_5shot"),
}
```

5-shot is what the sweeps run; 0-shot is defined but not run by default (all seven
metrics are emitted at every shot count, so 0-shot buys no extra soft signal).

### Real example (`mmlu_sl_verb_anatomy`, 5-shot)

Context sent to the model:

```
The following are multiple choice questions (with answers) about anatomy.

What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D. The second and third pharyngeal arches

Which of these branches of the trigeminal nerve contain somatic motor processes?
A. The supraorbital nerve
B. The infraorbital nerve
C. The mental nerve
D. None of the above
Answer: D. None of the above

The pleura
A. have no sensory innervation.
B. are separated by a 2 mm space.
C. extend into the neck.
D. are composed of respiratory epithelium.
Answer: C. extend into the neck.

In Angle's Class II Div 2 occlusion there is
A. excess overbite of the upper lateral incisors.
B. negative overjet of the upper central incisors.
C. excess overjet of the upper lateral incisors.
D. excess overjet of the upper central incisors.
Answer: C. excess overjet of the upper lateral incisors.

Which of the following is the body cavity that contains the pituitary gland?
A. Abdominal
B. Cranial
C. Pleural
D. Spinal
Answer: B. Cranial

A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral
A. paralysis of the facial muscles.
B. paralysis of the facial muscles and loss of taste.
C. paralysis of the facial muscles, loss of taste and lacrimation.
D. paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.
Answer:
```

Continuations scored (one loglikelihood request each, leading space):

```
" A. paralysis of the facial muscles."                                                        <-- gold
" B. paralysis of the facial muscles and loss of taste."
" C. paralysis of the facial muscles, loss of taste and lacrimation."
" D. paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
```

(The 5 shots above are `dev[0:5]` of `anatomy`, verbatim and in order; the test doc is
`test[0]`. Rendered locally from the cached `cais/mmlu` parquet, which carries the same
`dev`/`test` rows as `hails/mmlu_no_train`.)

### Grouping and results parsing (see also §3 for a hypothetical BLEnD-as-sl_verb)

`mmlu_sl_verb` is a **group**: 56 subject tasks → 4 subgroups (`stem`, `other`,
`social_sciences`, `humanities`) → one group row. Aggregation
(`mmlu_sl_verb.yaml`): `acc`/`acc_norm` are size-weighted; `bpb`, `logprob`,
`choice_logprob`, `choice_prob_norm`, `choice_logprob_norm` are unweighted means.

When reading `results.json` (verified 2026-07-15):

* 58 rows; the group row is keyed by the **alias** (`mmlu_sl_verb_5shot`), not
  `mmlu_sl_verb`. Subject rows are aliased too (`mmlu_sl_verb_anatomy_5shot`).
  Read the aliased group row; do not flat-average every key.
* Metrics are suffixed `,none` (`acc,none`, `choice_prob_norm,none`); the
  `*_stderr,none` companions come back as the string `"N/A"`.
* `outputs` (logged samples) is a **global** list duplicated into every row — it is
  not per-subject. It is also why `results.json` runs ~8 MB.

---

## 3. Hypothetical: BLEnD rendered in the `sl_verb` (verbalized MC) format

Not currently built — this is what the same documents would look like if they went
through the `mmlu_sl_verb` template instead of the OLMES rc/cloze one. It is a
plausible variant because BLEnD's source file is already 4-way MC with letter keys, so
the choices and gold letter are available without any extra work; only the rendering
changes.

### What changes

| | BLEnD today (`rc_5shot`) | BLEnD as sl_verb (`mc_5shot`) |
|---|---|---|
| Choices in context | no | yes, `A.`–`D.` under the question |
| Continuation | `" soccer"` | `" C. soccer"` (letter **and** text) |
| Few-shot target | bare gold answer | letter + gold answer |
| Usable metrics | gold `bits_per_byte` only | `acc`, `acc_norm`, plus the soft `choice_logprob` / `choice_prob_norm` family |
| Records scored | 1 of 4 (`label == idx`) | all 4 (rank classification) |

Everything else is unchanged: same dedup to one doc per `(country, ID)`, same first-5
questions-per-country held out as shots, same 16 country tasks.

### Real example (`blend_south_korea`, hypothetical `mc_5shot`)

Same documents as §1 — same 5 shots, same test doc — re-rendered with
`doc_to_text` / `doc_to_choice` from `_default_template_yaml`:

```
The following are questions about everyday life in South Korea.

What is a common snack for preschool kids in South Korea?
A. cilok
B. cookie
C. egg
D. jam sandwiches
Answer: B. cookie

What is a popular food to go with beer in South Korea?
A. chickpea
B. fried chicken
C. tacos
D. tapas
Answer: B. fried chicken

What is the most popular fruit in South Korea?
A. apple
B. durian
C. malbhog banana
D. orange
Answer: A. apple

What is a common school cafeteria food in South Korea?
A. fritter
B. kimchi
C. pizza
D. tea
Answer: B. kimchi

What is a popular snack at an amusement park in South Korea?
A. churros
B. cotton candy
C. crisps
D. shawarma
Answer: A. churros

What is a popular afterschool sport for elementary schools in South Korea?
A. American Football
B. hide and seek
C. soccer
D. stick game
Answer:
```

Continuations scored (all four, one loglikelihood request each):

```
" A. American Football"
" B. hide and seek"
" C. soccer"            <-- gold, label=2
" D. stick game"
```

Note the shot questions now expose their own distractors, which the rc rendering hides
— e.g. the first shot's `A. cilok` (West Java) and `D. jam sandwiches` (UK). Choices
in the MC file are in letter order, which is alphabetical by answer string, so gold
position is not uniform across documents (`B, B, A, B, A` in these five shots).

### Description wording

The description above is BLEnD's own
(`"The following are questions about everyday life in South Korea.\n\n"`). A stricter
sl_verb transcription would use MMLU's phrasing:

```
The following are multiple choice questions (with answers) about everyday life in South Korea.
```

Pick one and keep it fixed — the description is part of every context, so changing it
invalidates comparisons against previously scored checkpoints.

### Wiring notes if this were actually built

* **Staged-request route** (cheapest): the existing builder already writes all four
  choices per document with `idx`/`label`, so only `rc_context` → an MC context builder
  and the `continuation` string change, into a sibling `mc_5shot/` variant dir. But
  `run_olmo_bpb_eval.py` filters `label == idx`, so it would score gold-only bpb over
  `" C. soccer"` — a *letter-prefixed* bpb, not comparable to the rc numbers and not
  comparable across documents whose gold letter differs. Getting `acc` out requires
  the runner to keep all four records and argmax over them.
* **lm-eval route**: BLEnD is not an lm-eval task, so this means a real YAML per
  country (`dataset_path: nayeon212/BLEnD`) plus a preprocessing step for the dedup
  and the shot holdout — the MC file's 305,939 rows are not usable as a `test_split`
  directly, and `fewshot_split: dev` has no counterpart in the source dataset.
* **Metric caveat**: raw `acc` on BLEnD is only meaningful once a model is off the 25%
  floor. The reason the current staging is rc:bpb is that these runs span 1e17–2e21
  FLOPs, where the soft/bpb signal is the only thing moving — the same argument that
  motivates `sl_verb` over stock MMLU. If built, the soft metrics
  (`choice_logprob`, `choice_prob_norm`) are the ones to read, not `acc`.
* BLEnD stays diagnostics-only either way — it is **not** part of the olmix mixture
  objective.
