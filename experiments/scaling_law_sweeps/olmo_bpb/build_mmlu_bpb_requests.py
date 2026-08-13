# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialise MMLU ``rc::olmes`` bpb requests in the ``oe_eval_tasks/`` layout.

MMLU is the one gap in the staged ``allenai/OLMo-in-loop-evals`` bundle: 116 variants
across 57 task dirs, none of them MMLU. The bpb runner
(``olmo_bpb/run_olmo_bpb_eval.py``) reads pre-materialised oe-eval requests, so we
generate MMLU's here to the identical on-disk shape:

    <root>/oe_eval_tasks/<task>/rc_5shot/{config.json,requests.jsonl.gz}

Two layouts are written from one pass over the data:

* **4 category tasks** — ``mmlu_stem`` / ``mmlu_humanities`` / ``mmlu_social_sciences``
  / ``mmlu_other``. These are what the mixture objective consumes. Each holds every
  document of every subject in the category, so the runner's plain mean-over-documents
  equals olmix's example-count-weighted category average (see ``mmlu_olmes``).
* **57 subject tasks** — ``mmlu_<subject>``. Diagnostics only; never put them in the
  objective (57 of 95 tasks would drown every other capability in the flat 1/n mean).

Prompt construction is transcribed from ``allenai/olmes``; see ``mmlu_olmes``.

Usage (local, then upload in-region):

    python -m experiments.scaling_law_sweeps.olmo_bpb.build_mmlu_bpb_requests \\
        --output-dir /tmp/mmlu_bpb \\
        --upload-to gs://marin-us-central1/eval_datasets/olmo_in_loop_evals/

``--upload-to`` takes the ``olmo_in_loop_evals/`` root; ``oe_eval_tasks/`` is appended,
so this merges into the existing staged bundle without touching the other 57 task dirs.
It writes through ``rigging.filesystem``, which needs working GCS credentials in the
process -- run it as an iris job, or upload the produced tree with ``gsutil cp -r``
(never ``rsync -d``, which would delete the other task dirs).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass

from experiments.scaling_law_sweeps.olmo_bpb.mmlu_olmes import (
    MMLU_CATEGORIES,
    MMLU_DATASET_PATH,
    MMLU_FEWSHOT_SPLIT,
    MMLU_NUM_SHOTS,
    MMLU_PRIMARY_METRIC,
    MMLU_RC_VARIANT,
    MMLU_SPLIT,
    MMLU_SUBJECTS,
    cloze_query,
    olmix_metric_name,
    rc_context,
)

logger = logging.getLogger(__name__)

OE_EVAL_TASKS_SUBDIR = "oe_eval_tasks"
REQUESTS_FILENAME = "requests.jsonl.gz"
CONFIG_FILENAME = "config.json"
# `allenai/olmes` commit the prompt construction was transcribed from.
OLMES_SOURCE = "https://github.com/allenai/olmes (oe_eval/tasks/oe_eval_tasks/mmlu.py, rc::olmes)"


@dataclass(frozen=True)
class SubjectRequests:
    subject: str
    records: list[dict]
    n_docs: int


def _load_subject(subject: str, dataset_path: str) -> tuple[list[dict], list[dict]]:
    """(dev docs, test docs) for one MMLU subject, as raw HF rows."""
    from datasets import load_dataset

    ds = load_dataset(dataset_path, subject)
    return list(ds[MMLU_FEWSHOT_SPLIT]), list(ds[MMLU_SPLIT])


def build_subject_requests(subject: str, dev_docs: list[dict], test_docs: list[dict]) -> SubjectRequests:
    """One loglikelihood record per (document, answer choice), oe-eval schema.

    ``label`` is the gold choice index and ``idx`` the record's choice index, which is
    exactly what the bpb runner filters on to keep the gold continuation.
    """
    if len(dev_docs) < MMLU_NUM_SHOTS:
        raise ValueError(f"{subject}: dev split has {len(dev_docs)} docs, need {MMLU_NUM_SHOTS} for few-shot")
    fewshot = [(d["question"], d["choices"][d["answer"]]) for d in dev_docs[:MMLU_NUM_SHOTS]]

    records: list[dict] = []
    for doc_id, doc in enumerate(test_docs):
        context = rc_context(doc["question"], fewshot, subject)
        oe_doc = {
            "index": doc_id,
            "query": cloze_query(doc["question"]),
            "choices": list(doc["choices"]),
            "gold": int(doc["answer"]),
        }
        for idx, choice in enumerate(doc["choices"]):
            records.append(
                {
                    "request_type": "loglikelihood",
                    "doc": oe_doc,
                    "request": {"context": context, "continuation": f" {choice}"},
                    "idx": idx,
                    "task_name": f"mmlu_{subject}",
                    "doc_id": doc_id,
                    "native_id": doc_id,
                    "label": int(doc["answer"]),
                    "mmlu_subject": subject,
                }
            )
    return SubjectRequests(subject=subject, records=records, n_docs=len(test_docs))


def _concat_for_category(parts: list[SubjectRequests]) -> list[dict]:
    """Merge subject records into one category task, re-keying ``doc_id`` globally.

    Documents are **interleaved round-robin across subjects** rather than concatenated
    subject-by-subject. The full-file bpb is a mean over documents and so is unaffected
    by order, but ``run_olmo_bpb_eval --limit N`` keeps the first N documents: under
    plain concatenation a smoke test would silently score only the category's first
    subject (e.g. all of ``mmlu_humanities`` reported from ``formal_logic`` alone).
    Interleaving makes any prefix a spread-out sample of the whole category.

    ``doc_id`` must stay unique within a request file (oe-eval groups a document's
    choices by it). ``mmlu_subject`` / ``mmlu_subject_doc_id`` preserve per-subject
    identity so the category can still be split apart afterwards.
    """
    by_subject = [(part.subject, _group_by_doc(part.records)) for part in parts]
    out: list[dict] = []
    doc_id = 0
    for position in range(max(len(docs) for _, docs in by_subject)):
        for subject, docs in by_subject:
            if position >= len(docs):
                continue
            for rec in docs[position]:
                merged = dict(rec)
                merged["mmlu_subject_doc_id"] = rec["doc_id"]
                merged["doc_id"] = doc_id
                merged["native_id"] = f"{subject}:{rec['doc_id']}"
                out.append(merged)
            doc_id += 1
    return out


def _group_by_doc(records: list[dict]) -> list[list[dict]]:
    """Split a subject's flat record list into per-document choice groups, in order."""
    groups: list[list[dict]] = []
    for rec in records:
        if rec["idx"] == 0:
            groups.append([])
        groups[-1].append(rec)
    return groups


def _task_config(task_name: str, subjects: list[str], n_docs: int, n_records: int) -> dict:
    """A config.json shaped like the staged oe-eval ones, plus honest provenance.

    These files are generated by marin, not by ai2's oe-eval, so they carry no
    ``task_hash``; ``marin_provenance`` records what they were built from instead.
    """
    return {
        "task_name": task_name,
        "task_config": {
            "task_name": task_name,
            "task_core": task_name,
            "limit": None,
            "split": MMLU_SPLIT,
            "num_shots": MMLU_NUM_SHOTS,
            "fewshot_seed": 1234,
            "primary_metric": MMLU_PRIMARY_METRIC,
            "random_subsample_seed": 1234,
            "context_kwargs": {},
            "generation_kwargs": {},
            "metric_kwargs": {"uncond_docid_offset": 1000000},
            "native_id_field": "index",
            "fewshot_source": None,
            "dataset_path": MMLU_DATASET_PATH,
            "dataset_name": subjects[0] if len(subjects) == 1 else None,
            "use_chat_format": None,
            "version": 0,
            "compute_gold_bpb": True,
            "metadata": {
                "alias": olmix_metric_name(subjects[0]) if len(subjects) == 1 else task_name,
                "regimes": ["OLMES-v0.1"],
            },
        },
        "num_instances": n_records,
        "marin_provenance": {
            "generated_by": "experiments/scaling_law_sweeps/olmo_bpb/build_mmlu_bpb_requests.py",
            "olmes_source": OLMES_SOURCE,
            "mmlu_subjects": subjects,
            "n_docs": n_docs,
            "note": (
                "Category tasks concatenate every document of their subjects, so a plain "
                "mean-over-documents equals olmix's example-count-weighted category average."
            ),
        },
    }


def _write_task(output_dir: str, task_name: str, records: list[dict], subjects: list[str], n_docs: int) -> str:
    task_dir = os.path.join(output_dir, OE_EVAL_TASKS_SUBDIR, task_name, MMLU_RC_VARIANT)
    os.makedirs(task_dir, exist_ok=True)
    with gzip.open(os.path.join(task_dir, REQUESTS_FILENAME), "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(task_dir, CONFIG_FILENAME), "w") as f:
        json.dump(_task_config(task_name, subjects, n_docs, len(records)), f)
    return task_dir


def build_all(output_dir: str, *, dataset_path: str, subjects: tuple[str, ...], write_subject_tasks: bool) -> dict:
    """Build every requested subject, then the 4 category tasks. Returns a report dict."""
    per_subject: dict[str, SubjectRequests] = {}
    for subject in subjects:
        dev_docs, test_docs = _load_subject(subject, dataset_path)
        per_subject[subject] = build_subject_requests(subject, dev_docs, test_docs)
        logger.info("built mmlu_%s: %d docs, %d records", subject, len(test_docs), len(per_subject[subject].records))

    doc_counts = {s: r.n_docs for s, r in per_subject.items()}
    written: list[str] = []

    if write_subject_tasks:
        for subject, part in per_subject.items():
            written.append(_write_task(output_dir, f"mmlu_{subject}", part.records, [subject], part.n_docs))

    category_docs: dict[str, int] = {}
    for category, cat_subjects in MMLU_CATEGORIES.items():
        parts = [per_subject[s] for s in cat_subjects if s in per_subject]
        if len(parts) != len(cat_subjects):
            logger.warning("skipping category %s: only %d/%d subjects built", category, len(parts), len(cat_subjects))
            continue
        records = _concat_for_category(parts)
        n_docs = sum(p.n_docs for p in parts)
        category_docs[category] = n_docs
        written.append(_write_task(output_dir, f"mmlu_{category}", records, list(cat_subjects), n_docs))
        logger.info("built mmlu_%s: %d docs, %d records", category, n_docs, len(records))

        # A repeated doc_id across subjects would silently merge two documents' choices.
        distinct_doc_ids = len(Counter(r["doc_id"] for r in records))
        assert distinct_doc_ids == n_docs, f"mmlu_{category}: {distinct_doc_ids} distinct doc_ids for {n_docs} docs"

    return {"doc_counts": doc_counts, "category_docs": category_docs, "written": written}


def upload(local_dir: str, gcs_root: str) -> int:
    """Upload only ``oe_eval_tasks/`` under ``local_dir`` into ``gcs_root``."""
    from rigging.filesystem import filesystem

    fs = filesystem("gcs")
    dest = gcs_root.rstrip("/")
    src = os.path.join(local_dir, OE_EVAL_TASKS_SUBDIR)
    n = 0
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, local_dir)
        for name in files:
            fs.put(os.path.join(root, name), f"{dest}/{rel}/{name}")
            n += 1
    logger.info("Uploaded %d files to %s/%s/", n, dest, OE_EVAL_TASKS_SUBDIR)
    return n


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="Local dir to write oe_eval_tasks/ into.")
    ap.add_argument("--upload-to", default=None, help="GCS olmo_in_loop_evals/ root to merge the new task dirs into.")
    ap.add_argument("--dataset-path", default=MMLU_DATASET_PATH, help="HF dataset id for MMLU.")
    ap.add_argument(
        "--subjects",
        default="all",
        help="'all' or a comma-separated subset (subset builds no category tasks unless complete).",
    )
    ap.add_argument(
        "--no-subject-tasks",
        dest="write_subject_tasks",
        action="store_false",
        help="Write only the 4 category tasks, skipping the 57 per-subject diagnostic dirs.",
    )
    args = ap.parse_args()

    subjects = MMLU_SUBJECTS if args.subjects == "all" else tuple(s.strip() for s in args.subjects.split(",") if s)
    unknown = sorted(set(subjects) - set(MMLU_SUBJECTS))
    if unknown:
        raise ValueError(f"Unknown MMLU subjects: {unknown}")

    report = build_all(
        args.output_dir,
        dataset_path=args.dataset_path,
        subjects=subjects,
        write_subject_tasks=args.write_subject_tasks,
    )
    total_docs = sum(report["doc_counts"].values())
    logger.info("Total: %d subjects, %d docs, categories=%s", len(subjects), total_docs, report["category_docs"])
    with open(os.path.join(args.output_dir, "_mmlu_build_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if args.upload_to:
        upload(args.output_dir, args.upload_to)


if __name__ == "__main__":
    main()
