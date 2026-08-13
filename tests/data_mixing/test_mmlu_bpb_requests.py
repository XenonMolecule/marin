# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MMLU must enter the mixture objective as exactly 4 category tasks, correctly weighted.

Two things can silently break the objective here:

1. **Cardinality.** 57 standalone MMLU subtasks would be 57 of 95 tasks in the flat
   1/n mean and would drown every other capability. The objective must see 4.
2. **Weighting.** olmix aggregates MMLU with an example-count-weighted mean of
   per-subject metrics. We instead concatenate a category's documents into one request
   file and let the bpb runner take a plain mean over documents. Those agree *only*
   because olmix's weights are exactly example-count fractions -- which is what
   :func:`test_olmix_weights_are_exact_example_count_fractions` pins down, and
   :func:`test_category_micro_mean_equals_olmix_weighted_mean` exercises end to end.

Nothing here touches the network: the format tests run on synthetic MMLU rows.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from fractions import Fraction

import pytest

from experiments.scaling_law_sweeps.olmo_bpb.build_mmlu_bpb_requests import (
    _concat_for_category,
    _write_task,
    build_subject_requests,
)
from experiments.scaling_law_sweeps.olmo_bpb.mmlu_olmes import (
    MMLU_CATEGORIES,
    MMLU_NUM_SHOTS,
    MMLU_SUBJECTS,
    OLMIX_MMLU_WEIGHTS,
    category_of,
    olmix_metric_name,
    rc_context,
)
from experiments.scaling_law_sweeps.olmo_bpb.run_olmo_bpb_eval import _read_bpb_requests

# MMLU's test split: 14,042 questions. DCLM's own eval_meta_data.csv agrees (n=14042).
MMLU_TEST_SIZE = 14042
# Per-category totals, cross-checked against the built request files.
EXPECTED_CATEGORY_TOTALS = {"stem": 3018, "humanities": 4705, "social_sciences": 3077, "other": 3242}


def _fake_docs(n: int, *, offset: int = 0) -> list[dict]:
    return [
        {
            "question": f"q{offset + i}?",
            "choices": [f"c{offset + i}a", f"c{offset + i}b", f"c{offset + i}c", f"c{offset + i}d"],
            "answer": i % 4,
        }
        for i in range(n)
    ]


def test_objective_sees_four_mmlu_tasks_not_fifty_seven():
    from experiments.data_mixing.olmix_tasks import MMLU_BPB, build_target_tasks

    mmlu_in_objective = [tv for tv in build_target_tasks() if tv.split("/", 1)[0].startswith("mmlu")]
    assert len(mmlu_in_objective) == 4, "MMLU must be 4 category tasks; 57 subtasks would dominate the flat mean"
    assert set(mmlu_in_objective) == set(MMLU_BPB)
    assert {tv.split("/", 1)[0] for tv in mmlu_in_objective} == {f"mmlu_{c}" for c in MMLU_CATEGORIES}


def test_categories_partition_the_57_subjects():
    assert len(MMLU_SUBJECTS) == 57
    assert len(set(MMLU_SUBJECTS)) == 57
    flat = [s for subjects in MMLU_CATEGORIES.values() for s in subjects]
    assert sorted(flat) == sorted(MMLU_SUBJECTS), "a subject in two categories would be double-counted"
    assert {len(v) for v in MMLU_CATEGORIES.values()} == {18, 13, 12, 14}


def test_olmix_weight_tables_cover_exactly_our_categories():
    """A subject present in one source and not the other is a silent mis-aggregation."""
    assert set(OLMIX_MMLU_WEIGHTS) == set(MMLU_CATEGORIES)
    for category, subjects in MMLU_CATEGORIES.items():
        assert {olmix_metric_name(s) for s in subjects} == set(OLMIX_MMLU_WEIGHTS[category])


@pytest.mark.parametrize("category", sorted(MMLU_CATEGORIES))
def test_olmix_weights_sum_to_one(category):
    assert sum(OLMIX_MMLU_WEIGHTS[category].values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("category", sorted(MMLU_CATEGORIES))
def test_olmix_weights_are_exact_example_count_fractions(category):
    """Recover integer example counts from the float weights.

    This is the load-bearing claim: if every weight is ``n_i / N`` then olmix's
    weighted mean of per-subject means equals the micro-mean over the category's
    documents, so staging one concatenated request file per category needs no
    aggregation code at all. Reconstructing the integers from the floats and checking
    they sum to ``N`` would fail loudly if olmix used anything else (e.g. a macro-mean,
    or counts from a different split).
    """
    weights = OLMIX_MMLU_WEIGHTS[category]
    total = EXPECTED_CATEGORY_TOTALS[category]
    counts = {}
    for metric, weight in weights.items():
        scaled = weight * total
        counts[metric] = round(scaled)
        assert abs(scaled - counts[metric]) < 1e-6, f"{metric}: weight*{total}={scaled} is not an integer"
    assert sum(counts.values()) == total
    assert min(counts.values()) >= 100  # MMLU's smallest subjects have exactly 100 test questions


def test_category_totals_cover_the_mmlu_test_split():
    assert sum(EXPECTED_CATEGORY_TOTALS.values()) == MMLU_TEST_SIZE


def test_weight_denominators_divide_the_category_total():
    """Recover each weight as an exact rational and check its denominator divides ``N``.

    Independent of the ``weight * N`` check above: it derives the denominators from the
    float bit patterns instead of assuming ``N``, so a weight that was e.g. a macro-mean
    ``1/12`` rather than an example-count fraction would not divide 3077.
    """
    for category, weights in OLMIX_MMLU_WEIGHTS.items():
        denominator = 1
        for weight in weights.values():
            frac = Fraction(weight).limit_denominator(100000)
            denominator = denominator * frac.denominator // math.gcd(denominator, frac.denominator)
        assert EXPECTED_CATEGORY_TOTALS[category] % denominator == 0


def test_category_of_is_consistent_with_the_tables():
    for category, subjects in MMLU_CATEGORIES.items():
        for subject in subjects:
            assert category_of(subject) == category
    with pytest.raises(KeyError):
        category_of("not_a_subject")


def test_rc_context_matches_olmes_prompt_layout():
    """Transcribed from ``allenai/olmes``: per-subject description, then the dev
    examples as ``Question: ...\\nAnswer: <gold>`` joined by blank lines, then the
    doc's own query ending in a bare ``Answer:``."""
    context = rc_context("What is 2+2?", [("What is 1+1?", "2"), ("What is 3+3?", "6")], "elementary_mathematics")
    assert context == (
        "The following are multiple choice questions (with answers) about elementary mathematics.\n\n"
        "Question: What is 1+1?\nAnswer: 2\n\n"
        "Question: What is 3+3?\nAnswer: 6\n\n"
        "Question: What is 2+2?\nAnswer:"
    )


def test_build_subject_requests_emits_one_record_per_choice():
    dev, test = _fake_docs(MMLU_NUM_SHOTS), _fake_docs(3, offset=100)
    built = build_subject_requests("astronomy", dev, test)

    assert built.n_docs == 3
    assert len(built.records) == 12
    first = built.records[0]
    assert first["request"]["continuation"] == " c100a"
    assert first["label"] == 0 and first["idx"] == 0
    assert first["doc"] == {"index": 0, "query": "Question: q100?\nAnswer:", "choices": test[0]["choices"], "gold": 0}
    assert first["mmlu_subject"] == "astronomy"
    # Every choice of a doc shares one context; only the continuation varies.
    assert len({r["request"]["context"] for r in built.records[:4]}) == 1
    assert [r["request"]["continuation"] for r in built.records[:4]] == [f" c100{c}" for c in "abcd"]
    # The gold record the bpb runner keeps is the one whose idx == label.
    assert sum(1 for r in built.records if r["idx"] == r["label"]) == 3


def test_build_subject_requests_rejects_too_few_fewshot_docs():
    with pytest.raises(ValueError, match="dev split"):
        build_subject_requests("astronomy", _fake_docs(2), _fake_docs(1))


def test_concat_for_category_keeps_doc_ids_unique():
    parts = [
        build_subject_requests("astronomy", _fake_docs(MMLU_NUM_SHOTS), _fake_docs(3, offset=10)),
        build_subject_requests("college_biology", _fake_docs(MMLU_NUM_SHOTS), _fake_docs(2, offset=20)),
    ]
    merged = _concat_for_category(parts)

    assert len({r["doc_id"] for r in merged}) == 5, "a repeated doc_id would merge two documents' choices"
    assert {r["mmlu_subject"] for r in merged} == {"astronomy", "college_biology"}
    assert len(merged) == 20
    # A document's choices stay contiguous and in order, so `_read_bpb_requests` still
    # picks exactly one gold per doc.
    assert [r["idx"] for r in merged[:8]] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert {r["mmlu_subject_doc_id"] for r in merged} == {0, 1, 2}


def test_concat_for_category_interleaves_subjects_so_limit_is_representative():
    """``--limit N`` keeps the first N documents; a subject-major layout would make a
    smoke test report a whole category from its first subject alone."""
    parts = [
        build_subject_requests("astronomy", _fake_docs(MMLU_NUM_SHOTS), _fake_docs(4, offset=10)),
        build_subject_requests("college_biology", _fake_docs(MMLU_NUM_SHOTS), _fake_docs(4, offset=20)),
        build_subject_requests("machine_learning", _fake_docs(MMLU_NUM_SHOTS), _fake_docs(2, offset=30)),
    ]
    merged = _concat_for_category(parts)

    subject_per_doc = {}
    for rec in merged:
        subject_per_doc[rec["doc_id"]] = rec["mmlu_subject"]
    first_three = [subject_per_doc[i] for i in range(3)]
    assert first_three == ["astronomy", "college_biology", "machine_learning"]
    # Subjects that run out early must not stall the rotation.
    assert [subject_per_doc[i] for i in range(6, 8)] == ["astronomy", "college_biology"]
    assert len(subject_per_doc) == 10


def test_written_task_is_readable_by_the_bpb_runner(tmp_path):
    """The runner is the real consumer: it must find the file and keep exactly the gold
    continuation per document."""
    built = build_subject_requests("astronomy", _fake_docs(MMLU_NUM_SHOTS), _fake_docs(4, offset=30))
    _write_task(str(tmp_path), "mmlu_astronomy", built.records, ["astronomy"], built.n_docs)

    requests = _read_bpb_requests(str(tmp_path), "mmlu_astronomy/rc_5shot", None)
    assert len(requests) == 4
    assert [r["continuation"] for r in requests] == [" c30a", " c31b", " c32c", " c33d"]
    assert all(r["context"].startswith("The following are multiple choice questions") for r in requests)

    config = json.loads((tmp_path / "oe_eval_tasks" / "mmlu_astronomy" / "rc_5shot" / "config.json").read_text())
    assert config["task_config"]["num_shots"] == MMLU_NUM_SHOTS
    assert config["task_config"]["metadata"]["alias"] == "mmlu_astronomy:rc::olmes"
    assert config["marin_provenance"]["n_docs"] == 4


def test_category_micro_mean_equals_olmix_weighted_mean(tmp_path):
    """End-to-end: a distinct bpb per subject, aggregated both ways, must agree.

    Uses real per-subject example counts (recovered from olmix's weights) but a tiny
    scale factor so the test stays fast. The two aggregations are computed by genuinely
    different code paths -- ours is the runner's mean over the concatenated file's gold
    records, olmix's is ``sum(weight_i * mean_i)``.
    """
    category = "social_sciences"
    subjects = MMLU_CATEGORIES[category]
    total = EXPECTED_CATEGORY_TOTALS[category]
    # Real counts, scaled down by 50 (all are multiples of 50 except a few; round-trip
    # through the same integers used for the weighted mean so the comparison is exact).
    counts = {s: max(round(OLMIX_MMLU_WEIGHTS[category][olmix_metric_name(s)] * total) // 50, 1) for s in subjects}
    subject_bpb = {s: 0.5 + 0.1 * i for i, s in enumerate(subjects)}

    parts = [
        build_subject_requests(s, _fake_docs(MMLU_NUM_SHOTS), _fake_docs(counts[s], offset=1000 * i))
        for i, s in enumerate(subjects)
    ]
    merged = _concat_for_category(parts)
    _write_task(str(tmp_path), f"mmlu_{category}", merged, list(subjects), sum(counts.values()))

    requests = _read_bpb_requests(str(tmp_path), f"mmlu_{category}/rc_5shot", None)
    subject_of_doc = {}
    for rec in merged:
        subject_of_doc[rec["doc_id"]] = rec["mmlu_subject"]
    assert len(requests) == sum(counts.values())

    # Our path: plain mean over documents (exactly what `_bpb_for_task` computes).
    micro = sum(subject_bpb[subject_of_doc[r["doc_id"]]] for r in requests) / len(requests)
    # olmix's path: example-count-weighted mean of per-subject means, with the same counts.
    n_total = sum(counts.values())
    weighted = sum(counts[s] / n_total * subject_bpb[s] for s in subjects)
    assert micro == pytest.approx(weighted, rel=1e-12)


def test_written_layout_matches_the_staged_oe_eval_convention(tmp_path):
    built = build_subject_requests("virology", _fake_docs(MMLU_NUM_SHOTS), _fake_docs(2))
    task_dir = _write_task(str(tmp_path), "mmlu_virology", built.records, ["virology"], built.n_docs)

    assert task_dir.endswith(os.path.join("oe_eval_tasks", "mmlu_virology", "rc_5shot"))
    assert sorted(os.listdir(task_dir)) == ["config.json", "requests.jsonl.gz"]
    with gzip.open(os.path.join(task_dir, "requests.jsonl.gz"), "rt") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 8
    assert all(r["request_type"] == "loglikelihood" for r in records)
