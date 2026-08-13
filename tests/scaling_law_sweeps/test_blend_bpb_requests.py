# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BLEnD rc:bpb staging must produce files the olmo_bpb runner scores correctly.

The load-bearing properties:

1. **Gold filter compatibility.** The runner keeps records where ``label == idx``;
   every document must survive as exactly its gold continuation.
2. **Few-shot holdout.** The 5 dev questions prime every context and must never
   appear as scored documents.
3. **Dedup.** The 305k-row MC file collapses to one document per (country, ID);
   variant rows (same question, different distractors) must not duplicate documents.

Nothing here touches the network: tests run on synthetic MC-file rows, written and
read back through the real builder and the real runner reader.
"""

from __future__ import annotations

import gzip
import json
import os

import pytest

from experiments.scaling_law_sweeps.olmo_bpb.build_blend_bpb_requests import (
    BLEND_NUM_SHOTS,
    BLEND_RC_VARIANT,
    OE_EVAL_TASKS_SUBDIR,
    REQUESTS_FILENAME,
    blend_task_name,
    build_all,
    build_country_records,
    dedup_mc_rows,
    parse_mc_row,
)
from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import BLEND_BPB, resolve_tasks
from experiments.scaling_law_sweeps.olmo_bpb.run_olmo_bpb_eval import _read_bpb_requests

_INSTRUCTION = (
    " Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C). "
    'Provide as JSON format: {"answer_choice":""}\n\nA. w\nB. x\nC. y\nD. z\n\nAnswer:'
)


def _mc_row(country: str, qid: str, variant: int, question: str, choices: list[str], gold: int) -> dict:
    letters = "ABCD"
    return {
        "MCQID": f"{qid}_{variant}",
        "ID": qid,
        "country": country,
        "prompt": question + _INSTRUCTION,
        "choices": json.dumps({letters[i]: c for i, c in enumerate(choices)}),
        "choice_countries": json.dumps({letters[i]: "X" for i in range(len(choices))}),
        "answer_idx": letters[gold],
    }


def _country_rows(country: str, n_questions: int) -> list[dict]:
    rows = []
    for q in range(n_questions):
        qid = f"Al-en-{q:02d}"
        # Two variant rows per question: dedup must keep only the first.
        for variant in range(2):
            rows.append(
                _mc_row(
                    country,
                    qid,
                    variant,
                    f"What is thing {q} in {country}?",
                    [f"gold answer {q}", f"distractor {variant}a", "distractor b", "distractor c"],
                    0,
                )
            )
    return rows


def test_parse_mc_row_strips_instruction():
    row = _mc_row("US", "Al-en-01", 0, "What is a common snack in the US?", ["fruit", "candy", "egg", "squid"], 0)
    question, choices, gold = parse_mc_row(row)
    assert question == "What is a common snack in the US?"
    assert choices == ["fruit", "candy", "egg", "squid"]
    assert gold == 0


def test_parse_mc_row_rejects_unmarked_prompt():
    row = _mc_row("US", "Al-en-01", 0, "q", ["a", "b", "c", "d"], 1)
    row["prompt"] = "A prompt without the marker"
    with pytest.raises(ValueError, match="instruction marker"):
        parse_mc_row(row)


def test_dedup_keeps_first_variant_per_question():
    rows = _country_rows("US", 8)
    by_country = dedup_mc_rows(rows)
    assert [r["MCQID"] for r in by_country["US"]] == [f"Al-en-{q:02d}_0" for q in range(8)]
    assert all(not by_country[c] for c in by_country if c != "US")


def test_records_hold_out_fewshot_and_key_docs():
    rows = dedup_mc_rows(_country_rows("US", 9))["US"]
    records, fewshot_ids = build_country_records("US", rows)

    assert len(fewshot_ids) == BLEND_NUM_SHOTS
    n_docs = 9 - BLEND_NUM_SHOTS
    assert len(records) == n_docs * 4
    scored_ids = {r["native_id"] for r in records}
    assert scored_ids.isdisjoint(fewshot_ids)

    # Every context carries the description and all few-shot golds; never its own answer.
    for rec in records:
        ctx = rec["request"]["context"]
        assert ctx.startswith("The following are questions about everyday life in the United States.")
        assert ctx.count("Question:") == BLEND_NUM_SHOTS + 1
        assert ctx.rstrip().endswith("Answer:")


def _all_country_rows(n_questions: int) -> list[dict]:
    from experiments.scaling_law_sweeps.olmo_bpb.build_blend_bpb_requests import BLEND_COUNTRIES

    rows: list[dict] = []
    for country in BLEND_COUNTRIES:
        rows.extend(_country_rows(country, n_questions))
    return rows


def test_runner_reader_keeps_exactly_gold_continuations(tmp_path):
    report = build_all(str(tmp_path), _all_country_rows(12))
    n_docs = 12 - BLEND_NUM_SHOTS
    assert report["doc_counts"][blend_task_name("US")] == n_docs

    kept = _read_bpb_requests(str(tmp_path), f"blend_us/{BLEND_RC_VARIANT}", None)
    assert len(kept) == n_docs
    golds = {f" gold answer {q}" for q in range(BLEND_NUM_SHOTS, 12)}
    assert {r["continuation"] for r in kept} == golds


def test_build_all_rejects_missing_country(tmp_path):
    rows = [r for r in _all_country_rows(12) if r["country"] != "Greece"]
    with pytest.raises(ValueError, match="Greece"):
        build_all(str(tmp_path), rows)


def test_requests_file_is_one_json_record_per_line(tmp_path):
    build_all(str(tmp_path), _all_country_rows(7))
    path = os.path.join(str(tmp_path), OE_EVAL_TASKS_SUBDIR, "blend_uk", BLEND_RC_VARIANT, REQUESTS_FILENAME)
    with gzip.open(path, "rt") as f:
        recs = [json.loads(line) for line in f]
    assert all(rec["request_type"] == "loglikelihood" for rec in recs)
    assert all(rec["request"]["continuation"].startswith(" ") for rec in recs)


def test_resolve_tasks_blend_keyword_matches_tuple():
    assert resolve_tasks("blend") == list(BLEND_BPB)
    assert all(tv.endswith("/rc_5shot") and tv.startswith("blend_") for tv in BLEND_BPB)
    assert len(BLEND_BPB) == 16
