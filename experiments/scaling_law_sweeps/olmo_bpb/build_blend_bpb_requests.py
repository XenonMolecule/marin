# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialise BLEnD everyday-cultural-knowledge bpb requests in the ``oe_eval_tasks`` layout.

BLEnD (https://github.com/nlee0212/BLEnD) asks the same everyday-life questions
(food, celebrations, school, sports, ...) across 16 countries/regions; its
multiple-choice file poses each question for a target country with the country's
top-voted answer as gold and other countries' answers as distractors. We render the
English MC set in the OLMES ``rc`` (cloze) formulation — question only, choices NOT
shown — so the bpb runner (``olmo_bpb/run_olmo_bpb_eval.py``) scores the gold answer
exactly like every other rc task in the suite:

    <root>/oe_eval_tasks/blend_<country>/rc_5shot/{config.json,requests.jsonl.gz}

One task dir per country (16 total), diagnostics only — these are NOT part of the
olmix mixture objective. Records carry all 4 choices (``idx``/``label``), so the
runner's gold-continuation filter applies unchanged and a future MC/full-rc variant
can reuse the same files.

Source: the 305,939-row ``mc_questions_file_v1.1.json`` collapses to 5,081 distinct
(country, question) documents — the remaining rows are distractor/permutation
variants with the same gold, indistinguishable under gold-bpb. We keep the first
occurrence per (country, ID), which is deterministic (the file order is fixed).
Few-shot context is the first ``BLEND_NUM_SHOTS`` questions per country in ID order,
held out of the test set (OLMES ``first_n`` convention).

Usage (local build, then upload with ``gcloud storage cp -r``; never ``rsync -d``):

    python -m experiments.scaling_law_sweeps.olmo_bpb.build_blend_bpb_requests \\
        --output-dir /tmp/blend_bpb
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os

from experiments.scaling_law_sweeps.olmo_bpb.mmlu_olmes import cloze_query

logger = logging.getLogger(__name__)

BLEND_HF_REPO = "nayeon212/BLEnD"
BLEND_MC_FILE = "data/mc_questions_hf/mc_questions_file_v1.1.json"

OE_EVAL_TASKS_SUBDIR = "oe_eval_tasks"
REQUESTS_FILENAME = "requests.jsonl.gz"
CONFIG_FILENAME = "config.json"

BLEND_RC_VARIANT = "rc_5shot"
BLEND_NUM_SHOTS = 5

# The generation-style instruction appended to every BLEnD MC prompt; the bare
# question is everything before it.
_PROMPT_MARKER = " Without any explanation"

# Country key (as in the MC file) -> display name for the task description.
BLEND_COUNTRIES: dict[str, str] = {
    "Algeria": "Algeria",
    "Assam": "Assam",
    "Azerbaijan": "Azerbaijan",
    "China": "China",
    "Ethiopia": "Ethiopia",
    "Greece": "Greece",
    "Indonesia": "Indonesia",
    "Iran": "Iran",
    "Mexico": "Mexico",
    "North_Korea": "North Korea",
    "Northern_Nigeria": "Northern Nigeria",
    "South_Korea": "South Korea",
    "Spain": "Spain",
    "UK": "the United Kingdom",
    "US": "the United States",
    "West_Java": "West Java",
}


def blend_task_name(country: str) -> str:
    """``"South_Korea"`` -> ``"blend_south_korea"``."""
    return f"blend_{country.lower()}"


def country_description(country: str) -> str:
    return f"The following are questions about everyday life in {BLEND_COUNTRIES[country]}.\n\n"


def parse_mc_row(row: dict) -> tuple[str, list[str], int]:
    """(question, choices in letter order, gold index) from one MC-file row."""
    prompt = row["prompt"]
    if _PROMPT_MARKER not in prompt:
        raise ValueError(f"{row['MCQID']}: prompt lacks the instruction marker")
    question = prompt.split(_PROMPT_MARKER, 1)[0]
    by_letter = json.loads(row["choices"])
    letters = sorted(by_letter)
    return question, [by_letter[letter] for letter in letters], letters.index(row["answer_idx"])


def dedup_mc_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """First MC row per (country, base question ID), grouped by country, ID order."""
    first: dict[tuple[str, str], dict] = {}
    for row in rows:
        first.setdefault((row["country"], row["ID"]), row)
    by_country: dict[str, list[dict]] = {country: [] for country in BLEND_COUNTRIES}
    for (country, _), row in sorted(first.items()):
        by_country[country].append(row)
    return by_country


def rc_context(question: str, fewshot: list[tuple[str, str]], country: str) -> str:
    """Description + labelled few-shot examples + the doc's cloze query."""
    labeled = "\n\n".join(f"{cloze_query(q)} {gold}" for q, gold in fewshot) + "\n\n"
    return country_description(country) + labeled + cloze_query(question)


def build_country_records(country: str, rows: list[dict]) -> tuple[list[dict], list[str]]:
    """(records for the country's test docs, held-out few-shot MCQIDs).

    One loglikelihood record per (document, answer choice), oe-eval schema: ``label``
    is the gold choice index and ``idx`` the record's choice index — exactly what the
    bpb runner filters on to keep the gold continuation.
    """
    if len(rows) <= BLEND_NUM_SHOTS:
        raise ValueError(f"{country}: only {len(rows)} questions, need more than {BLEND_NUM_SHOTS}")
    dev, test = rows[:BLEND_NUM_SHOTS], rows[BLEND_NUM_SHOTS:]
    fewshot = []
    for row in dev:
        question, choices, gold = parse_mc_row(row)
        fewshot.append((question, choices[gold]))

    records: list[dict] = []
    for doc_id, row in enumerate(test):
        question, choices, gold = parse_mc_row(row)
        context = rc_context(question, fewshot, country)
        oe_doc = {"index": doc_id, "query": cloze_query(question), "choices": choices, "gold": gold}
        for idx, choice in enumerate(choices):
            records.append(
                {
                    "request_type": "loglikelihood",
                    "doc": oe_doc,
                    "request": {"context": context, "continuation": f" {choice}"},
                    "idx": idx,
                    "task_name": blend_task_name(country),
                    "doc_id": doc_id,
                    "native_id": row["MCQID"],
                    "label": gold,
                    "blend_country": country,
                    "blend_choice_countries": json.loads(row["choice_countries"]),
                }
            )
    return records, [row["MCQID"] for row in dev]


def _task_config(country: str, n_docs: int, n_records: int, fewshot_ids: list[str]) -> dict:
    """A config.json shaped like the staged oe-eval ones, plus honest provenance."""
    task_name = blend_task_name(country)
    return {
        "task_name": task_name,
        "task_config": {
            "task_name": task_name,
            "task_core": task_name,
            "limit": None,
            "split": "test",
            "num_shots": BLEND_NUM_SHOTS,
            "primary_metric": "bits_per_byte",
            "context_kwargs": {},
            "generation_kwargs": {},
            "metric_kwargs": {},
            "native_id_field": "index",
            "fewshot_source": None,
            "dataset_path": BLEND_HF_REPO,
            "dataset_name": country,
            "use_chat_format": None,
            "version": 0,
            "compute_gold_bpb": True,
            "metadata": {"alias": task_name, "regimes": []},
        },
        "num_instances": n_records,
        "marin_provenance": {
            "generated_by": "experiments/scaling_law_sweeps/olmo_bpb/build_blend_bpb_requests.py",
            "source": f"hf://datasets/{BLEND_HF_REPO}/{BLEND_MC_FILE}",
            "n_docs": n_docs,
            "fewshot_mcqids": fewshot_ids,
            "note": (
                "OLMES rc (cloze) rendering of BLEnD's English MC set: choices are NOT shown "
                "in the context; gold = the target country's top-voted answer. Deduped to one "
                "document per (country, question ID); the MC file's remaining rows are "
                "distractor/permutation variants with identical gold continuations."
            ),
        },
    }


def _write_task(output_dir: str, country: str, records: list[dict], n_docs: int, fewshot_ids: list[str]) -> str:
    task_dir = os.path.join(output_dir, OE_EVAL_TASKS_SUBDIR, blend_task_name(country), BLEND_RC_VARIANT)
    os.makedirs(task_dir, exist_ok=True)
    with gzip.open(os.path.join(task_dir, REQUESTS_FILENAME), "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(task_dir, CONFIG_FILENAME), "w") as f:
        json.dump(_task_config(country, n_docs, len(records), fewshot_ids), f)
    return task_dir


def build_all(output_dir: str, mc_rows: list[dict]) -> dict:
    """Build every country task. Returns a report dict."""
    by_country = dedup_mc_rows(mc_rows)
    unknown = sorted(set(r["country"] for r in mc_rows) - set(BLEND_COUNTRIES))
    if unknown:
        raise ValueError(f"MC file has countries missing from BLEND_COUNTRIES: {unknown}")

    doc_counts: dict[str, int] = {}
    written: list[str] = []
    for country, rows in by_country.items():
        records, fewshot_ids = build_country_records(country, rows)
        n_docs = len(rows) - BLEND_NUM_SHOTS
        assert len({r["doc_id"] for r in records}) == n_docs
        doc_counts[blend_task_name(country)] = n_docs
        written.append(_write_task(output_dir, country, records, n_docs, fewshot_ids))
        logger.info("built %s: %d docs, %d records", blend_task_name(country), n_docs, len(records))
    return {"doc_counts": doc_counts, "written": written}


def load_mc_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(BLEND_HF_REPO, BLEND_MC_FILE, repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="Local dir to write oe_eval_tasks/ into.")
    args = ap.parse_args()

    rows = load_mc_rows()
    report = build_all(args.output_dir, rows)
    logger.info("Total: %d tasks, %d docs", len(report["doc_counts"]), sum(report["doc_counts"].values()))
    with open(os.path.join(args.output_dir, "_blend_build_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
