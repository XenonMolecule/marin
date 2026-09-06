# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The optimization target must never touch DCLM Core v2.

Core v2 is the held-out test set for the whole mixture comparison, so overlap here
would invalidate the headline result rather than merely degrade it. These tests
compute the overlap from ``dclm_core/additional_aggregation.json`` instead of
trusting a hard-coded list, so that adding a task to Core v2 upstream turns into a
test failure here.
"""

from __future__ import annotations

import pytest

from experiments.data_mixing.olmix_tasks import (
    CORE_V2_ALIASES,
    GSM8K_VARIANT,
    MMLU_BPB,
    OLMIX_DROPPED_TASKS,
    build_olmix_exact_tasks,
    build_target_tasks,
    core_v2_task_names,
    family_counts,
    overlaps_core_v2,
    task_family,
    task_name,
)
from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import (
    OLMO_BASE_EASY_BPB,
    QA_RC_MC_BPB,
)


def test_core_v2_is_the_22_low_variance_datasets():
    core_v2 = core_v2_task_names()
    assert len(core_v2) == 22
    # Spot-check the aggregation key really is the DCLM CORE set.
    assert {"hellaswag", "arc_easy", "squad", "coqa", "boolq"} <= core_v2


def test_target_tasks_are_disjoint_from_core_v2():
    core_v2 = core_v2_task_names()
    for task_variant in build_target_tasks():
        assert not overlaps_core_v2(task_variant, core_v2), f"{task_variant} is in DCLM Core v2"


def test_every_core_v2_overlap_is_actually_excluded():
    """The disjointness test above passes trivially if the alias map is empty, so pin
    that the 10 known overlaps really were present and really were removed."""
    core_v2 = core_v2_task_names()
    suite = (*OLMO_BASE_EASY_BPB, *QA_RC_MC_BPB)
    overlapping = {task_name(tv) for tv in suite if overlaps_core_v2(tv, core_v2)}
    assert overlapping == {
        "coqa",
        "jeopardy",
        "lambada",
        "squad",
        "arc_challenge",
        "arc_easy",
        "csqa",
        "hellaswag",
        "winogrande",
        "piqa",
    }
    kept = {task_name(tv) for tv in build_target_tasks()}
    assert not (overlapping & kept)


def test_aliased_names_resolve_to_core_v2():
    """The alias map is the only thing standing between `csqa`/`lambada` and silent
    contamination -- their dir names differ from DCLM's task names."""
    core_v2 = core_v2_task_names()
    for our_name, dclm_names in CORE_V2_ALIASES.items():
        assert any(n in core_v2 for n in dclm_names), f"alias {our_name} -> {dclm_names} matches nothing in Core v2"


def test_olmix_dropped_tasks_are_excluded():
    kept = {task_name(tv) for tv in build_target_tasks()}
    assert not (kept & OLMIX_DROPPED_TASKS)


def test_gsm8k_is_kept_because_it_is_not_in_core_v2():
    """gsm8k appears in DCLM's 95%/99% CI aggregations but NOT in low_variance_datasets,
    so it is legal and should survive -- an over-broad filter would drop it."""
    core_v2 = core_v2_task_names()
    assert "gsm8k" not in core_v2
    assert "gsm8k/gold_bpb_5shot" in build_target_tasks()


def test_mmlu_is_legal_and_included():
    core_v2 = core_v2_task_names()
    assert not any(n.startswith("mmlu") for n in core_v2)
    tasks = build_target_tasks(include_mmlu=True)
    assert set(MMLU_BPB) <= set(tasks)
    assert set(MMLU_BPB).isdisjoint(build_target_tasks(include_mmlu=False))


def test_target_task_composition():
    tasks = build_target_tasks(include_mmlu=True)
    assert len(tasks) == 42
    assert len(set(tasks)) == 42, "duplicate task variants would double-weight a capability"
    assert family_counts(tasks) == {"math": 8, "code": 19, "qa": 15}

    without_mmlu = build_target_tasks(include_mmlu=False)
    assert len(without_mmlu) == 38
    assert family_counts(without_mmlu) == {"math": 8, "code": 19, "qa": 11}


def test_non_olmix_variants_are_excluded():
    """olmix's suite uses 3-shot code and no minerva_math_500; keeping the extra marin
    variants would silently reweight math and code in the flat mean."""
    tasks = set(build_target_tasks())
    assert "codex_humaneval/gold_bpb_0shot" not in tasks
    assert "codex_mbpp/gold_bpb_0shot" not in tasks
    assert "minerva_math_500/gold_bpb_0shot" not in tasks
    assert "codex_humaneval/gold_bpb_3shot" in tasks
    assert "codex_mbpp/gold_bpb_3shot" in tasks


# The underlying benchmark behind each of the 42 target tasks, and behind each of DCLM
# Core v2's 22. Name equality is not enough to prove disjointness: the same dataset shows
# up as `csqa`/`commonsense_qa`, `lambada`/`lambada_openai`, `socialiqa`/`siqa`. Pinning
# the *dataset* forces a new task to be classified before it can enter the objective.
TARGET_TASK_DATASETS: dict[str, str] = {
    "codex_humaneval": "openai/HumanEval",
    "codex_mbpp": "google-research/MBPP",
    "gsm8k": "openai/GSM8K",
    "drop": "allenai/DROP",
    "naturalqs_open": "google/NaturalQuestions-open",
    "socialiqa": "allenai/SocialIQA",  # DCLM calls this `siqa`; NOT one of the 22
    "sciq": "allenai/SciQ",
    "medmcqa": "MedMCQA",
    **{
        f"mt_mbpp_{lang}": "google-research/MBPP (translated)"
        for lang in (
            "python",
            "java",
            "javascript",
            "typescript",
            "cpp",
            "c",
            "csharp",
            "go",
            "rust",
            "ruby",
            "php",
            "r",
            "bash",
            "scala",
            "swift",
            "haskell",
            "matlab",
        )
    },
    **{
        f"minerva_math_{subject}": "hendrycks/MATH"
        for subject in (
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        )
    },
    **{
        f"basic_skills_{skill}": "allenai/basic_skills"
        for skill in ("arithmetic", "coding", "common_knowledge", "logical_reasoning", "pattern", "string_operations")
    },
    **{f"mmlu_{category}": "cais/MMLU" for category in ("stem", "humanities", "social_sciences", "other")},
}

CORE_V2_DATASETS: dict[str, str] = {
    "hellaswag": "Rowan/HellaSwag",
    "hellaswag_zeroshot": "Rowan/HellaSwag",
    "jeopardy": "Jeopardy!",
    "bigbench_qa_wikidata": "BIG-bench/qa_wikidata",
    "arc_easy": "allenai/ARC-Easy",
    "arc_challenge": "allenai/ARC-Challenge",
    "copa": "COPA",
    "commonsense_qa": "CommonsenseQA",
    "piqa": "PIQA",
    "openbook_qa": "allenai/OpenBookQA",
    "lambada_openai": "LAMBADA (OpenAI)",
    "winograd": "Winograd Schema Challenge",
    "winogrande": "WinoGrande",
    "bigbench_dyck_languages": "BIG-bench/dyck_languages",
    "agi_eval_lsat_ar": "AGIEval/LSAT-AR",
    "bigbench_cs_algorithms": "BIG-bench/cs_algorithms",
    "bigbench_operators": "BIG-bench/operators",
    "bigbench_repeat_copy_logic": "BIG-bench/repeat_copy_logic",
    "squad": "SQuAD",
    "coqa": "CoQA",
    "boolq": "BoolQ",
    "bigbench_language_identification": "BIG-bench/language_identification",
}


def test_dataset_table_covers_every_target_task():
    """A task with no classified dataset cannot be cleared of Core v2 overlap."""
    unclassified = sorted({task_name(tv) for tv in build_target_tasks()} - set(TARGET_TASK_DATASETS))
    assert not unclassified, f"classify these before they enter the objective: {unclassified}"


def test_dataset_table_covers_every_core_v2_task():
    assert set(CORE_V2_DATASETS) == core_v2_task_names()


def test_no_target_task_shares_a_dataset_with_core_v2():
    """The real disjointness check: by *dataset*, not by task name.

    Catches an aliased dataset the name-based filter would let through -- e.g. if
    `siqa`/SocialIQA were ever added to Core v2's 22, this fails even though the strings
    `socialiqa` and `siqa` never match.
    """
    core_v2_datasets = set(CORE_V2_DATASETS.values())
    collisions = {
        name: dataset
        for name, dataset in TARGET_TASK_DATASETS.items()
        if dataset in core_v2_datasets and name in {task_name(tv) for tv in build_target_tasks()}
    }
    assert not collisions, f"target tasks share a dataset with DCLM Core v2: {collisions}"


@pytest.mark.parametrize(
    ("task_variant", "expected"),
    [
        ("gsm8k/gold_bpb_5shot", "math"),
        ("minerva_math_algebra/gold_bpb_0shot", "math"),
        ("codex_mbpp/gold_bpb_3shot", "code"),
        ("mt_mbpp_rust/gold_bpb_3shot", "code"),
        ("drop/bpb_5shot", "qa"),
        ("mmlu_stem/rc_5shot", "qa"),
        ("basic_skills_arithmetic/rc_5shot", "qa"),
    ],
)
def test_task_family_classification(task_variant, expected):
    assert task_family(task_variant) == expected


# --- olmix's own suite, reproduced exactly (paper Table 9) -------------------------------
#
# Transcribed from the paper, NOT from our implementation, so a drift in either direction
# fails. Table 9 lists per-family counts and marks subtasks; MMLU is averaged into its 4
# categories ("We treat subtasks as standalone tasks, except for MMLU").
TABLE_9_FAMILY_COUNTS = {"math": 7, "code": 19, "qa": 25}

# The 10 tasks the marin objective holds out for Core v2 but Table 9 fits on.
TABLE_9_CORE_V2_OVERLAP = {
    "arc_easy",
    "arc_challenge",
    "csqa",
    "hellaswag",
    "winogrande",
    "piqa",
    "coqa",
    "jeopardy",
    "squad",
    "lambada",
}


def test_olmix_exact_matches_table_9_size_and_families():
    tasks = build_olmix_exact_tasks()
    assert len(tasks) == 51
    assert family_counts(tasks) == TABLE_9_FAMILY_COUNTS


def test_olmix_exact_fits_on_core_v2_unlike_marin_objective():
    """The whole point of the exact suite: it does NOT hold Core v2 out."""
    exact = {task_name(tv) for tv in build_olmix_exact_tasks()}
    marin = {task_name(tv) for tv in build_target_tasks()}
    assert TABLE_9_CORE_V2_OVERLAP <= exact
    assert TABLE_9_CORE_V2_OVERLAP.isdisjoint(marin)


def test_olmix_exact_drops_gsm8k():
    """Table 9's Math family is Minerva's 7 splits; "GSM" never appears in the paper."""
    assert GSM8K_VARIANT in build_target_tasks()
    assert GSM8K_VARIANT not in build_olmix_exact_tasks()


def test_olmix_exact_keeps_the_five_metrics_olmix_drops():
    kept = {task_name(tv) for tv in build_olmix_exact_tasks()}
    assert kept.isdisjoint(OLMIX_DROPPED_TASKS)


def test_olmix_exact_uses_the_seven_minerva_splits_at_bpb():
    """Minerva contributes exactly 7 subject splits -- not minerva_math_500, not 8 with gsm8k."""
    minerva = sorted(tv for tv in build_olmix_exact_tasks() if task_name(tv).startswith("minerva_math"))
    assert len(minerva) == 7
    assert not any("minerva_math_500" in tv for tv in minerva)


def test_olmix_exact_differs_from_marin_objective_only_as_documented():
    exact, marin = set(build_olmix_exact_tasks()), set(build_target_tasks())
    assert {task_name(tv) for tv in exact - marin} == TABLE_9_CORE_V2_OVERLAP
    assert exact - marin != set()
    assert marin - exact == {GSM8K_VARIANT}


def test_olmix_exact_mmlu_is_four_categories():
    tasks = build_olmix_exact_tasks(include_mmlu=True)
    assert set(MMLU_BPB) <= set(tasks)
    assert len([tv for tv in tasks if task_name(tv).startswith("mmlu")]) == 4
    assert set(MMLU_BPB).isdisjoint(build_olmix_exact_tasks(include_mmlu=False))
