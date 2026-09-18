# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The cultural suite must apply the decided cleaning rules and render the sl_verb prompt.

Rules of record: `.agents/projects/cultural_eval_suite.md`. What can silently go wrong:

1. BLEnD's distractor draws duplicate documents, `not-applicable` golds survive, or the
   "(Provide ...)" format tails leak into prompts while the "(e.g., ...)" hints get cut.
2. FORK's Underspecified items or NormAd's neutral items survive the filters.
3. The five few-shot documents also get scored.
4. The generated lm-eval YAML renders something other than the `mmlu_sl_verb` layout.
5. Category / benchmark aggregation weights or centers accuracy wrongly.

Nothing here touches the network: sources are synthetic rows in the real file schemas.
"""

from __future__ import annotations

import json
import sys
import types

import jinja2
import pandas as pd
import pytest
import yaml

from experiments.scaling_law_sweeps.cultural.analyze_cultural_signal import benchmark_of, task_name_of
from experiments.scaling_law_sweeps.cultural.build_cultural_tasks import SourceRows, build_all, write_all
from experiments.scaling_law_sweeps.cultural.consolidate_cultural_results import (
    ACC_CENTERED,
    aggregate,
    task_metrics,
)
from experiments.scaling_law_sweeps.cultural.cultural_sources import (
    blend_tasks,
    culturalbench_task,
    fork_task,
    normad_task,
    strip_blend_instruction_tail,
)
from experiments.scaling_law_sweeps.cultural.cultural_tasks_set import (
    GROUP_TASK_NAME,
    METRIC_NAMES,
    NUM_SHOTS,
    TASK_SPECS,
    blend_task_name,
    result_key,
    write_task_yamls,
)
from experiments.scaling_law_sweeps.lm_eval_include_path import install_include_paths
from experiments.scaling_law_sweeps.mmlu.analyze_mmlu_signal import extract_samples, signal
from experiments.scaling_law_sweeps.mmlu.consolidate_mmlu_categories import category_rows, subject_key
from experiments.scaling_law_sweeps.olmo_bpb.build_blend_bpb_requests import BLEND_COUNTRIES
from experiments.scaling_law_sweeps.olmo_bpb.mmlu_olmes import MMLU_CATEGORIES, OLMIX_MMLU_WEIGHTS, olmix_metric_name

_BLEND_INSTRUCTION = (
    " Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C). "
    'Provide as JSON format: {"answer_choice":""}\n\nA. w\nB. x\nC. y\nD. z\n\nAnswer:'
)


def _blend_row(country: str, qid: str, draw: int, question: str, choices: list[str], gold: int) -> dict:
    letters = "ABCD"
    return {
        "MCQID": f"{qid}_{draw}",
        "ID": qid,
        "country": country,
        "prompt": question + _BLEND_INSTRUCTION,
        "choices": json.dumps({letters[i]: c for i, c in enumerate(choices)}),
        "choice_countries": json.dumps({letters[i]: "X" for i in range(4)}),
        "answer_idx": letters[gold],
    }


def _blend_rows(country: str, n_questions: int) -> list[dict]:
    rows = []
    for q in range(n_questions):
        for draw in range(2):  # two distractor draws of the same question
            rows.append(
                _blend_row(country, f"{country}-en-{q:02d}", draw, f"Q{q} in {country}?", ["a", "b", "c", "d"], q % 4)
            )
    return rows


def _cb_row(idx: int, country: str, question: str, answer: str) -> dict:
    return {
        "data_idx": idx,
        "question_idx": idx,
        "prompt_question": question,
        "prompt_option_a": "opt a",
        "prompt_option_b": "opt b",
        "prompt_option_c": "opt c",
        "prompt_option_d": "opt d",
        "answer": answer,
        "country": country,
    }


def _fork_row(i: int, type_: str, question: str = "Eat with what?") -> dict:
    return {
        "ID": f"fork-{i}",
        "Question": question,
        "Option A": "fork",
        "Option B": "hands",
        "Answer Key": "B",
        "Country": "India",
        "Type": type_,
        "Dimension": "Eating Utensils",
    }


def _normad_row(i: int, label: str, country: str = "japan", story: str = "Story. Is what X did acceptable?") -> dict:
    return {
        "ID": str(i),
        "Country": country,
        "Background": "",
        "Axis": "etiquette",
        "Subaxis": "visiting",
        "Value": "",
        "Rule-of-Thumb": "hidden",
        "Story": story,
        "Explanation": "",
        "Gold Label": label,
    }


def _sources() -> SourceRows:
    blend = [r for c in BLEND_COUNTRIES for r in _blend_rows(c, 8)]
    cb = [_cb_row(i, "Poland", f"Question {i}?", "ABCD"[i % 4]) for i in range(8)]
    fork = [_fork_row(i, "Explicit") for i in range(8)]
    normad = [_normad_row(i, "yes" if i % 2 else "no") for i in range(8)]
    return SourceRows(blend=blend, culturalbench=cb, fork=fork, normad=normad)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "How many hours do students spend at school? "
            "(Provide Arabic numerals in integers (0~24), without any decimal points.)",
            "How many hours do students spend at school?",
        ),
        ("What time is dinner? (Provide in HH:MM format (e.g., 18:00, 09:00).)", "What time is dinner?"),
        ("What is the capital? (Provide the official name.)", "What is the capital?"),
        (
            "What is the biggest meal? (e.g., breakfast, lunch, dinner)",
            "What is the biggest meal? (e.g., breakfast, lunch, dinner)",
        ),
        ("Which day is a rest day? (e.g. Monday, Tuesday)", "Which day is a rest day? (e.g. Monday, Tuesday)"),
        ("A plain question?", "A plain question?"),
    ],
)
def test_blend_tail_stripping_only_removes_provide_instructions(raw, expected):
    assert strip_blend_instruction_tail(raw) == expected


def test_blend_dedupes_draws_and_drops_not_applicable_gold():
    rows = _blend_rows("Greece", 7)
    rows += [
        _blend_row("Greece", "Greece-en-99", 0, "Thanksgiving dish?", ["not-applicable", "turkey", "paella", "x"], 0)
    ]
    [task] = blend_tasks(rows)
    assert task.spec.name == blend_task_name("Greece")
    ids = [d.native_id for d in task.docs]
    assert len(ids) == 7 and all(i.endswith("_0") for i in ids)
    assert not any("Thanksgiving" in d.question for d in task.docs)
    assert {d.country for d in task.docs} == {"Greece"}


def test_culturalbench_maps_gold_letter_and_tags_composite():
    composite = (
        "How do people welcome each other?\nRead the following statements and select the option "
        "that include all the appropriate statements for this question.\n(i) A\n(ii) B"
    )
    task = culturalbench_task([_cb_row(0, "Lebanon", composite, "D"), _cb_row(1, "France", "Hello?", "b")])
    assert [(d.gold, d.category) for d in task.docs] == [(3, "composite"), (1, "single")]
    assert task.docs[0].choices == ("opt a", "opt b", "opt c", "opt d")


def test_fork_drops_underspecified_items():
    task = fork_task([_fork_row(0, "Explicit"), _fork_row(1, "Underspecified"), _fork_row(2, "Implicit")])
    assert [d.native_id for d in task.docs] == ["fork-0", "fork-2"]
    assert task.docs[0].choices == ("fork", "hands") and task.docs[0].gold == 1


def test_normad_drops_neutral_and_keeps_yes_no_as_two_way():
    task = normad_task([_normad_row(0, "yes"), _normad_row(1, "neutral"), _normad_row(2, "no", country="south korea")])
    assert [(d.native_id, d.gold, d.country) for d in task.docs] == [("0", 0, "Japan"), ("2", 1, "South Korea")]
    assert all(d.choices == ("yes", "no") for d in task.docs)


def test_five_shots_are_held_out_of_test(tmp_path):
    manifest = write_all(str(tmp_path), build_all(_sources()))
    for name, entry in manifest["tasks"].items():
        assert entry["n_dev"] == NUM_SHOTS
        dev = pd.read_parquet(tmp_path / name / "dev.parquet")
        test = pd.read_parquet(tmp_path / name / "test.parquet")
        assert set(dev.native_id) == set(entry["fewshot_ids"])
        assert set(dev.native_id).isdisjoint(test.native_id)
        assert len(dev) + len(test) == entry["n_dev"] + entry["n_test"]
    assert set(manifest["tasks"]) == {s.name for s in TASK_SPECS}


def test_shots_alternate_gold_labels_even_when_the_source_is_sorted_by_label():
    # NormAd's CSV is sorted by country then label: a naive first-five is all "yes".
    rows = [_normad_row(i, "yes", country="egypt") for i in range(6)] + [_normad_row(i, "no") for i in range(6, 12)]
    dev, test = normad_task(rows).split()
    assert [d.gold for d in dev] == [0, 1, 0, 1, 0]
    assert [d.native_id for d in dev] == ["0", "6", "1", "7", "2"]
    assert [d.native_id for d in test] == ["3", "4", "5", "8", "9", "10", "11"]

    # A 4-way task cycles A, B, C, D, A; a label that runs out is skipped.
    cb = [_cb_row(i, "Poland", f"Q{i}?", "ABCD"[[0, 0, 0, 1, 2, 3, 0, 2][i]]) for i in range(8)]
    dev, _ = culturalbench_task(cb).split()
    assert [d.gold for d in dev] == [0, 1, 2, 3, 0]
    fork = [_fork_row(i, "Explicit") for i in range(7)]  # every gold is B
    dev, _ = fork_task(fork).split()
    assert [d.native_id for d in dev] == [f"fork-{i}" for i in range(5)]


def _render(cfg: dict, key: str, doc: dict):
    return jinja2.Template(cfg[key]).render(**doc)


def test_generated_yaml_renders_the_sl_verb_prompt(tmp_path):
    data_dir = tmp_path / "data"
    write_all(str(data_dir), build_all(_sources()))
    tasks_dir = tmp_path / "tasks"
    written = write_task_yamls(str(data_dir), str(tasks_dir))
    assert len(written) == len(TASK_SPECS) + 1

    with open(tasks_dir / f"{GROUP_TASK_NAME}.yaml") as f:
        group = yaml.safe_load(f)
    assert group["group"] == GROUP_TASK_NAME and set(group["task"]) == {s.name for s in TASK_SPECS}

    with open(tasks_dir / "cultural_sl_verb_normad.yaml") as f:
        normad = yaml.safe_load(f)
    assert [m["metric"] for m in normad["metric_list"]] == list(METRIC_NAMES)
    assert normad["fewshot_split"] == "dev" and normad["fewshot_config"] == {"sampler": "first_n"}
    test_doc = pd.read_parquet(normad["dataset_kwargs"]["data_files"]["test"]).iloc[0].to_dict()
    assert _render(normad, "doc_to_text", test_doc) == (
        "The following are questions (with answers) about social etiquette in Japan.\n\n"
        "Story. Is what X did acceptable?\nA. yes\nB. no\nAnswer:"
    )
    assert _render(normad, "doc_to_choice", test_doc) == "['A. yes', 'B. no']"
    assert normad["doc_to_target"] == "gold" and "description" not in normad

    with open(tasks_dir / f"{blend_task_name('UK')}.yaml") as f:
        uk = yaml.safe_load(f)
    uk_doc = pd.read_parquet(uk["dataset_kwargs"]["data_files"]["test"]).iloc[0].to_dict()
    assert uk["description"] == (
        "The following are multiple choice questions (with answers) about everyday life in the United Kingdom.\n\n"
    )
    assert _render(uk, "doc_to_text", uk_doc) == "Q5 in UK?\nA. a\nB. b\nC. c\nD. d\nAnswer:"
    assert _render(uk, "doc_to_choice", uk_doc) == "['A. a', 'B. b', 'C. c', 'D. d']"


def test_include_path_patch_accumulates_directories(monkeypatch, tmp_path):
    seen: list = []
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"

    class FakeTaskManager:
        def __init__(self, verbosity=None, include_path=None):
            seen.append(include_path)

    fake = types.ModuleType("lm_eval.tasks")
    fake.TaskManager = FakeTaskManager
    monkeypatch.setitem(sys.modules, "lm_eval", types.ModuleType("lm_eval"))
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", fake)
    monkeypatch.setattr("experiments.scaling_law_sweeps.lm_eval_include_path._REGISTERED_DIRS", [])

    install_include_paths(dir_a)
    install_include_paths(dir_b, str(dir_a))
    fake.TaskManager()
    fake.TaskManager(include_path="/explicit")
    assert seen == [[str(dir_a.resolve()), str(dir_b.resolve())], "/explicit"]


def _cultural_results_doc(acc_by_task: dict[str, float], shots: int = 5) -> dict:
    return {
        "results": {
            result_key(name, shots): {"acc,none": acc, "bpb,none": 1.0, "acc_stderr,none": "N/A"}
            for name, acc in acc_by_task.items()
        }
    }


def test_cultural_aggregation_centers_by_chance_and_averages_benchmarks():
    accs = {s.name: 0.25 for s in TASK_SPECS if s.benchmark == "blend"}
    accs[blend_task_name("UK")] = 1.0  # one country perfect -> blend mean over 16 tasks
    accs["cultural_sl_verb_culturalbench"] = 0.25
    accs["cultural_sl_verb_fork"] = 0.75
    accs["cultural_sl_verb_normad"] = 0.5
    rows = {r["name"]: r for r in aggregate(task_metrics(_cultural_results_doc(accs), 5))}

    assert rows["cultural_sl_verb_fork"][ACC_CENTERED] == pytest.approx(0.5)
    assert rows["cultural_sl_verb_normad"][ACC_CENTERED] == pytest.approx(0.0)
    assert rows["cultural_sl_verb_culturalbench"][ACC_CENTERED] == pytest.approx(0.0)
    assert rows["blend"]["acc"] == pytest.approx((15 * 0.25 + 1.0) / 16)
    assert rows["blend"][ACC_CENTERED] == pytest.approx(1.0 / 16)
    assert rows["blend"]["n_tasks"] == 16
    assert rows["cultural_macro"][ACC_CENTERED] == pytest.approx((1 / 16 + 0.0 + 0.5 + 0.0) / 4)
    assert rows["cultural_macro"]["n_tasks"] == 4 and rows["cultural_macro"]["bpb"] == pytest.approx(1.0)


def test_cultural_aggregation_flags_missing_tasks():
    rows = {r["name"]: r for r in aggregate(task_metrics(_cultural_results_doc({"cultural_sl_verb_fork": 0.5}), 5))}
    assert set(rows) == {"cultural_sl_verb_fork", "fork", "cultural_macro"}
    assert rows["cultural_macro"]["n_tasks"] == 1 and rows["cultural_macro"]["n_tasks_expected"] == 4


def test_mmlu_category_rows_weight_accuracy_by_size_and_average_soft_metrics():
    shots = 5
    results = {}
    for subjects in MMLU_CATEGORIES.values():
        for i, s in enumerate(subjects):
            results[subject_key(s, shots)] = {"acc,none": float(i % 2), "bpb,none": float(i), "acc_stderr,none": "N/A"}
    doc = {"results": results}
    rows = category_rows(doc, shots)
    assert set(rows) == set(MMLU_CATEGORIES)  # no group row in this doc -> no "all"
    for category, subjects in MMLU_CATEGORIES.items():
        metrics, n = rows[category]
        assert n == len(subjects)
        weights = [OLMIX_MMLU_WEIGHTS[category][olmix_metric_name(s)] for s in subjects]
        expected_acc = sum(w * (i % 2) for i, w in enumerate(weights)) / sum(weights)
        assert metrics["acc"] == pytest.approx(expected_acc)
        assert metrics["bpb"] == pytest.approx(sum(range(len(subjects))) / len(subjects))
    assert sum(n for _, n in rows.values()) == 57


def test_mmlu_category_rows_renormalize_when_a_subject_is_missing():
    shots = 5
    subjects = MMLU_CATEGORIES["stem"]
    results = {subject_key(s, shots): {"acc,none": 1.0} for s in subjects[1:]}
    (metrics, n) = category_rows({"results": results}, shots)["stem"]
    assert n == len(subjects) - 1
    assert metrics["acc"] == pytest.approx(1.0)


def _sample(choices: list[str], lls: list[float], target: int) -> dict:
    return {"doc": {"choices": choices}, "filtered_resps": [[ll, False] for ll in lls], "target": target}


def test_cultural_signal_groups_by_task_and_benchmark_without_mixing_choice_counts():
    doc = {
        "samples": {
            "cultural_sl_verb_normad_5shot": [
                _sample(["yes", "no"], [-1.0, -2.0], 0),
                _sample(["yes", "no"], [-2.0, -1.0], 1),
            ],
            "cultural_sl_verb_fork_5shot": [_sample(["a", "b"], [-1.0, -3.0], 1)],
            "cultural_sl_verb_blend_uk_5shot": [_sample(["a", "b", "c", "d"], [-1.0, -2.0, -3.0, -4.0], 0)],
            "cultural_sl_verb_blend_us_5shot": [_sample(["a", "b", "c", "d"], [-4.0, -1.0, -3.0, -2.0], 1)],
            "cultural_sl_verb_5shot": [_sample(["x", "y"], [-1.0, -1.0], 0)],  # the group row is ignored
        }
    }
    by_task = extract_samples(doc, task_name_of, all_group=None)
    by_benchmark = extract_samples(doc, benchmark_of, all_group=None)
    assert set(by_task) == {
        "cultural_sl_verb_normad",
        "cultural_sl_verb_fork",
        "cultural_sl_verb_blend_uk",
        "cultural_sl_verb_blend_us",
    }
    assert set(by_benchmark) == {"normad", "fork", "blend"}
    logps, _, golds = by_benchmark["blend"]
    assert logps.shape == (2, 4) and golds.tolist() == [0, 1]

    two_way = signal(*by_benchmark["normad"], seed=0)
    assert two_way["obs_acc"] == 1.0 and two_way["null_acc"] == pytest.approx(0.5)
    assert two_way["pick_C"] == 0.0 and two_way["pick_D"] == 0.0
    # A pooled `all` group would stack (n, 2) with (n, 4) rows; MMLU keeps it only because
    # every subject is 4-way, the cultural analysis must opt out.
    with pytest.raises(ValueError):
        extract_samples(doc, task_name_of)
