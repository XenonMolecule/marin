# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fill the lm-eval tasks the hero ``aa-index-v4.1.1-v3`` decon corpus failed to stage.

The hero corpus (``gs://marin-us-central2/datakit/decontam/evals/aa-index-v4.1.1-v3``) has all 9
AA benchmarks but 118 of its 339 lm-eval tasks failed to load, including gsm8k, truthfulqa,
winogrande, boolq, ifeval, triviaqa and squadv2. The causes, and the fix applied here:

- legacy no-namespace dataset names (``gsm8k``, ``super_glue``, ``truthful_qa``, ``anli``, ...):
  ``lmh_loader`` now resolves them to their current Hub repos;
- dataset loading scripts, which ``datasets`` 4.x refuses: stage under ``datasets`` 3.6 (what the
  pinned lm-eval resolves to) with ``trust_remote_code``;
- ifeval's ``langdetect``/``immutabledict`` imports: install them in the job;
- ``include_base_44_<lang>_few_shot_og`` and ``mmmlu_*`` names that the pinned lm-eval does not
  register: stage ``include_base_44_<lang>`` (same questions), and MMMLU directly from
  ``openai/MMMLU``.

The hero corpus is copied unchanged to ``aa-index-v4.1.1-v3-fill``; only the failed tasks are
staged anew, with the same text policy (``_lmh_doc_text``), ids and writer. Run in-region::

    uv run --with "lm-eval[math,api] @ git+https://github.com/stanford-crfm/lm-evaluation-harness@<LM_EVAL_COMMIT>" \\
        --with langdetect --with immutabledict \\
        python -m experiments.fast_curation.decon_eval_corpus_fill

where ``<LM_EVAL_COMMIT>`` is the commit pinned in ``lib/marin/pyproject.toml``'s ``eval`` extra
(``d5e3391f22cde186c827674d5c3ec7c5f4fe0cab``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterator

from datasets import load_dataset
from google.cloud import storage
from rigging.filesystem.storage_path import StoragePath

from experiments.datakit.decontam.lmh_loader import (
    flatten_task_dict,
    materialize_first_nonempty_split,
    trust_remote_code_for_hf,
)
from experiments.datakit.decontam.prepare_eval_corpus import (
    EVAL_CORPUS_VERSION,
    _lmh_doc_text,
    _write_parquet,
)

logger = logging.getLogger(__name__)

BUCKET = "marin-us-central2"
EVALS = "datakit/decontam/evals"
SOURCE_VERSION = EVAL_CORPUS_VERSION
FILL_VERSION = f"{EVAL_CORPUS_VERSION}-fill"
MMMLU_REPO = "openai/MMMLU"
_INCLUDE_RE = re.compile(r"^(include_base_44_.+)_few_shot_og$")
_MMMLU_RE = re.compile(r"^mmmlu_([a-z]{2})_([a-z]{2})$")


def _copy_tree(src_prefix: str, dst_prefix: str) -> int:
    bucket = storage.Client().bucket(BUCKET)
    n = 0
    for blob in storage.Client().list_blobs(BUCKET, prefix=src_prefix):
        if blob.name.endswith("_manifest.json"):
            continue
        bucket.copy_blob(blob, bucket, dst_prefix + blob.name[len(src_prefix) :])
        n += 1
    return n


def _read_manifest(version: str, suite: str) -> dict:
    with StoragePath(f"gs://{BUCKET}/{EVALS}/{version}/{suite}/_manifest.json").open("r") as f:
        return json.load(f)


def _write_manifest(version: str, suite: str, manifest: dict) -> None:
    with StoragePath(f"gs://{BUCKET}/{EVALS}/{version}/{suite}/_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=1)


def lmh_load_name(task: str) -> str:
    """lm-eval task name to load for a configured task the pinned lm-eval does not register."""
    m = _INCLUDE_RE.match(task)
    return m.group(1) if m else task


def mmmlu_config(task: str) -> str | None:
    """``openai/MMMLU`` config for an ``mmmlu_<lang>_<region>`` task (``mmmlu_ar_xy`` -> ``AR_XY``)."""
    m = _MMMLU_RE.match(task)
    return f"{m.group(1)}_{m.group(2)}".upper() if m else None


def mmmlu_text(row: dict) -> str:
    choices = "\n".join(f"{k}. {row[k]}" for k in ("A", "B", "C", "D"))
    return f"{row['Question']}\n{choices}\nAnswer: {row[row['Answer']]}"


def _stage_mmmlu(task: str, config: str, root: str) -> list[dict]:
    docs = load_dataset(MMMLU_REPO, config, split="test")

    def rows() -> Iterator[dict]:
        for i, row in enumerate(docs):
            yield {"id": f"{task}-test-{i}", "text": mmmlu_text(row)}

    n = _write_parquet(f"{root}/{task}/eval.parquet", rows())
    return [{"task": task, "artifact": f"{task}/eval.parquet", "records": n}]


def _stage_lmh(task: str, root: str) -> list[dict]:
    from lm_eval.tasks import get_task_dict  # noqa: PLC0415  # optional dep: lm_eval

    entries = []
    for child, lm_task in flatten_task_dict(get_task_dict([lmh_load_name(task)])):
        chosen = materialize_first_nonempty_split(lm_task)
        if chosen is None:
            raise ValueError(f"{child}: no docs in any split")
        split, docs = chosen

        def rows(lm_task=lm_task, docs=docs, split=split, child=child) -> Iterator[dict]:
            for i, doc in enumerate(docs):
                text = _lmh_doc_text(doc, lm_task.doc_to_text, lm_task.doc_to_target)
                if text:
                    yield {"id": f"{child}-{split}-{i}", "text": text}

        n = _write_parquet(f"{root}/{child}/eval.parquet", rows())
        entries.append({"task": child, "artifact": f"{child}/eval.parquet", "records": n})
    return entries


def fill(only: list[str] | None) -> None:
    src = f"{EVALS}/{SOURCE_VERSION}/"
    dst = f"{EVALS}/{FILL_VERSION}/"
    logger.info("copied %d hero corpus objects to %s", _copy_tree(src, dst), dst)

    aa = _read_manifest(SOURCE_VERSION, "aa")
    _write_manifest(FILL_VERSION, "aa", {**aa, "corpus_version": FILL_VERSION})

    lmh = _read_manifest(SOURCE_VERSION, "lmh")
    root = f"gs://{BUCKET}/{dst}lmh"
    trust_remote_code_for_hf()
    artifacts = list(lmh["artifacts"])
    still_failed = []
    for failure in lmh["failed"]:
        task = failure["task"]
        if only and task not in only:
            still_failed.append(failure)
            continue
        try:
            config = mmmlu_config(task)
            entries = _stage_mmmlu(task, config, root) if config else _stage_lmh(task, root)
        except Exception as exc:
            logger.warning("%s: still fails: %s", task, exc)
            still_failed.append({"task": task, "reason": f"fill: {exc}"})
            continue
        logger.info("%s: staged %s", task, [(e["task"], e["records"]) for e in entries])
        artifacts.extend(entries)

    manifest = {
        **lmh,
        "corpus_version": FILL_VERSION,
        "artifact_root": f"{EVALS}/{FILL_VERSION}/lmh",
        "artifacts": artifacts,
        "included_leaf_tasks": sorted({a["task"] for a in artifacts}),
        "failed": still_failed,
        "status": "complete_with_failures" if still_failed else "complete",
        "filled_from": SOURCE_VERSION,
    }
    _write_manifest(FILL_VERSION, "lmh", manifest)
    logger.info(
        "fill: %d artifacts (hero had %d); %d tasks still failing: %s",
        len(artifacts),
        len(lmh["artifacts"]),
        len(still_failed),
        [f["task"] for f in still_failed],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="Stage only these failed tasks (others stay failed).")
    args = ap.parse_args()
    fill(args.only)


if __name__ == "__main__":
    main()
