# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OLMES ``mmlu_<subject>:rc::olmes`` prompt construction and the 4 MMLU categories.

MMLU is absent from the staged ``allenai/OLMo-in-loop-evals`` bundle (116 variants,
none of them MMLU), so the bpb requests have to be materialised here. Everything in
this module is transcribed from ai2's ``allenai/olmes`` so the prompts match
``mmlu_<subject>:rc::olmes`` exactly:

* ``oe_eval/data/mmlu_tasks.py`` — the 57 subjects and their 4 categories.
* ``oe_eval/tasks/oe_eval_tasks/mmlu.py`` — ``GenericMMLU`` (the ``rc`` variant):
  cloze query, ``choices`` = the raw answer strings, per-subject description, few-shot
  examples = the ``dev`` split's first *k* docs in unchanged order.
* ``oe_eval/tasks/utils.py`` — ``make_cloze_prompt``.
* ``oe_eval/tasks/base_task.py`` — ``Task.fewshot_context`` (non-chat branch) and
  ``MultipleChoiceTask.construct_requests`` / ``doc_to_target``.
* ``oe_eval/configs/tasks.py`` — ``mmlu_{sub}:rc::olmes`` = ``split=test``,
  ``num_shots=5``, ``primary_metric=acc_per_char``, no ``limit``.

Category aggregation
--------------------
olmix reports MMLU as **4 category tasks**, not 57 subtasks ("We treat subtasks as
standalone tasks, except for MMLU, where we use averages over the four MMLU
categories"). Its ``aggregate_mmlu`` (``olmix/fit/utils.py``) takes an
example-count-weighted mean of per-subject metrics; :data:`OLMIX_MMLU_WEIGHTS` is that
weight table copied verbatim.

An example-count-weighted mean of per-subject means is *identically* the micro-mean
over the category's documents::

    sum_i (n_i/N) * (1/n_i) sum_{d in i} x_d  ==  (1/N) sum_{d in category} x_d

So we stage one request file per category holding every document of every subject in
it, and the bpb runner's plain mean over documents already **is** olmix's number — no
aggregation code, no chance of a weighting bug. ``tests/data_mixing/`` asserts the
staged per-category document counts reproduce :data:`OLMIX_MMLU_WEIGHTS` exactly.
"""

from __future__ import annotations

MMLU_STEM: tuple[str, ...] = (
    "abstract_algebra",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
)

MMLU_HUMANITIES: tuple[str, ...] = (
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
)

MMLU_SOCIAL_SCIENCES: tuple[str, ...] = (
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
)

MMLU_OTHER: tuple[str, ...] = (
    "anatomy",
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology",
)

MMLU_CATEGORIES: dict[str, tuple[str, ...]] = {
    "stem": MMLU_STEM,
    "humanities": MMLU_HUMANITIES,
    "social_sciences": MMLU_SOCIAL_SCIENCES,
    "other": MMLU_OTHER,
}

MMLU_SUBJECTS: tuple[str, ...] = MMLU_STEM + MMLU_HUMANITIES + MMLU_SOCIAL_SCIENCES + MMLU_OTHER

# Verbatim from olmix `olmix/fit/utils.py::aggregate_mmlu` (stem/humanities/
# social_sciences/other weight dicts). Keys keep olmix's metric names so the
# provenance is unambiguous; they are MMLU-test example counts / category total.
OLMIX_MMLU_WEIGHTS: dict[str, dict[str, float]] = {
    "stem": {
        "mmlu_abstract_algebra:rc::olmes": 0.03313452617627568,
        "mmlu_astronomy:rc::olmes": 0.05036447978793903,
        "mmlu_college_biology:rc::olmes": 0.04771371769383698,
        "mmlu_college_chemistry:rc::olmes": 0.03313452617627568,
        "mmlu_college_computer_science:rc::olmes": 0.03313452617627568,
        "mmlu_college_mathematics:rc::olmes": 0.03313452617627568,
        "mmlu_college_physics:rc::olmes": 0.033797216699801194,
        "mmlu_computer_security:rc::olmes": 0.03313452617627568,
        "mmlu_conceptual_physics:rc::olmes": 0.07786613651424784,
        "mmlu_electrical_engineering:rc::olmes": 0.04804506295559974,
        "mmlu_elementary_mathematics:rc::olmes": 0.12524850894632206,
        "mmlu_high_school_biology:rc::olmes": 0.10271703114645461,
        "mmlu_high_school_chemistry:rc::olmes": 0.06726308813783963,
        "mmlu_high_school_computer_science:rc::olmes": 0.03313452617627568,
        "mmlu_high_school_mathematics:rc::olmes": 0.08946322067594434,
        "mmlu_high_school_physics:rc::olmes": 0.050033134526176276,
        "mmlu_high_school_statistics:rc::olmes": 0.07157057654075547,
        "mmlu_machine_learning:rc::olmes": 0.03711066931742876,
    },
    "other": {
        "mmlu_anatomy:rc::olmes": 0.04164096236890808,
        "mmlu_business_ethics:rc::olmes": 0.030845157310302282,
        "mmlu_clinical_knowledge:rc::olmes": 0.08173966687230105,
        "mmlu_college_medicine:rc::olmes": 0.05336212214682295,
        "mmlu_global_facts:rc::olmes": 0.030845157310302282,
        "mmlu_human_aging:rc::olmes": 0.06878470080197409,
        "mmlu_management:rc::olmes": 0.03177051202961135,
        "mmlu_marketing:rc::olmes": 0.07217766810610735,
        "mmlu_medical_genetics:rc::olmes": 0.030845157310302282,
        "mmlu_miscellaneous:rc::olmes": 0.24151758173966686,
        "mmlu_nutrition:rc::olmes": 0.09438618136952498,
        "mmlu_professional_accounting:rc::olmes": 0.08698334361505243,
        "mmlu_professional_medicine:rc::olmes": 0.08389882788402221,
        "mmlu_virology:rc::olmes": 0.05120296113510179,
    },
    "social_sciences": {
        "mmlu_econometrics:rc::olmes": 0.03704907377315567,
        "mmlu_high_school_geography:rc::olmes": 0.06434839129021774,
        "mmlu_high_school_government_and_politics:rc::olmes": 0.06272343191420214,
        "mmlu_high_school_macroeconomics:rc::olmes": 0.12674683132921677,
        "mmlu_high_school_microeconomics:rc::olmes": 0.07734806629834254,
        "mmlu_high_school_psychology:rc::olmes": 0.17712057198570036,
        "mmlu_human_sexuality:rc::olmes": 0.04257393565160871,
        "mmlu_professional_psychology:rc::olmes": 0.19889502762430938,
        "mmlu_public_relations:rc::olmes": 0.03574910627234319,
        "mmlu_security_studies:rc::olmes": 0.07962300942476438,
        "mmlu_sociology:rc::olmes": 0.0653233669158271,
        "mmlu_us_foreign_policy:rc::olmes": 0.032499187520311994,
    },
    "humanities": {
        "mmlu_formal_logic:rc::olmes": 0.026780021253985122,
        "mmlu_high_school_european_history:rc::olmes": 0.03506907545164718,
        "mmlu_high_school_us_history:rc::olmes": 0.04335812964930925,
        "mmlu_high_school_world_history:rc::olmes": 0.050371944739638685,
        "mmlu_international_law:rc::olmes": 0.0257173219978746,
        "mmlu_jurisprudence:rc::olmes": 0.022954303931987247,
        "mmlu_logical_fallacies:rc::olmes": 0.034643995749202974,
        "mmlu_moral_disputes:rc::olmes": 0.07353878852284804,
        "mmlu_moral_scenarios:rc::olmes": 0.1902231668437832,
        "mmlu_philosophy:rc::olmes": 0.06609989373007438,
        "mmlu_prehistory:rc::olmes": 0.06886291179596174,
        "mmlu_professional_law:rc::olmes": 0.32603613177470775,
        "mmlu_world_religions:rc::olmes": 0.03634431455897981,
    },
}

# oe-eval task settings for `mmlu_<subject>:rc::olmes`.
MMLU_RC_VARIANT = "rc_5shot"
MMLU_NUM_SHOTS = 5
MMLU_SPLIT = "test"
MMLU_FEWSHOT_SPLIT = "dev"
MMLU_DATASET_PATH = "cais/mmlu"
MMLU_PRIMARY_METRIC = "acc_per_char"


def olmix_metric_name(subject: str) -> str:
    """``"astronomy"`` -> ``"mmlu_astronomy:rc::olmes"`` (olmix's metric key)."""
    return f"mmlu_{subject}:rc::olmes"


def category_of(subject: str) -> str:
    for category, subjects in MMLU_CATEGORIES.items():
        if subject in subjects:
            return category
    raise KeyError(f"{subject!r} is not an MMLU subject")


def subject_description(subject: str) -> str:
    """``GenericMMLU_MC.fewshot_context``'s ``default_description``."""
    return f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"


def cloze_query(question: str) -> str:
    """``oe_eval.tasks.utils.make_cloze_prompt`` with OLMES' default affixes."""
    return f"Question: {question}\nAnswer:"


def rc_context(question: str, fewshot: list[tuple[str, str]], subject: str) -> str:
    """The 5-shot rc context: description + labelled dev examples + the doc's query.

    ``fewshot`` is ``[(question, gold_choice_text), ...]`` in dev-split order.
    Mirrors ``Task.fewshot_context``'s non-chat, ``num_fewshot > 0`` branch.
    """
    labeled = "\n\n".join(f"{cloze_query(q)} {gold}" for q, gold in fewshot) + "\n\n"
    return subject_description(subject) + labeled + cloze_query(question)
