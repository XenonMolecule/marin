# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach arm A's grid labels to arm B's extracted shards and write B's co-partitioned tables.

Runs after the swap fleet (``swap_extractor.py``) and after ``build_swap_allowlist.py --stage
labels``. For each WARC it reads B's record shard and the WARC's label parquet, joins them by
``doc_id``, and writes B's topic and quality tables plus ``_done`` markers next to the shard's stem
(``swap_extractor.write_attribute_tables``), so ``grid_label merge``, ``grid_tokenize`` and
``grid_store`` treat B like any labelled corpus without either model ever running on B's text.

Guards, all fatal for the WARC: every B record has a label; the label rows B did not use are
exactly the documents the catalog recorded as dropped (short or missing). A WARC whose two
``_done`` markers already exist is skipped, so the job is resumable.

    python -m experiments.fast_curation.attach_swap_labels
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import logging
from collections.abc import Iterator

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.baseline_collection import grid_corpora
from experiments.baseline_collection.grid_corpora import Stage, done_marker
from experiments.fast_curation.swap_extractor import (
    DATASET,
    LABEL_BASE,
    LABEL_COLUMNS,
    OUTPUT_ROOT,
    shard_path,
    write_attribute_tables,
)
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)


def _read_records(shard: str) -> Iterator[dict]:
    with fsspec.open(shard, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def attach_one(warc_hash: str, root: str, dataset: str, label_base: str, label_names: list[str]) -> dict:
    """Join one WARC's B records with A's labels and write B's attribute tables."""
    grid_corpora.OUTPUT_BASE = label_base
    shard = shard_path(root, warc_hash)
    with fsspec.open(f"{root}/catalog/data-{warc_hash}.json", "r") as fh:
        catalog = json.load(fh)
    with fsspec.open(f"{root}/labels/data-{warc_hash}.parquet", "rb") as fh:
        labels = {row["doc_id"]: row for row in pq.read_table(fh).to_pylist()}
    rows = []
    unlabeled = []
    for record in _read_records(shard):
        label = labels.pop(record["doc_id"], None)
        if label is None:
            unlabeled.append(record["doc_id"])
            continue
        rows.append({**record, **{name: label[name] for name in LABEL_COLUMNS}})
    if unlabeled:
        raise RuntimeError(f"{warc_hash}: {len(unlabeled)} B records have no label in A's tables (e.g. {unlabeled[0]})")
    n_dropped = catalog["n_short"] + len(catalog["missing"])
    if len(labels) != n_dropped:
        raise RuntimeError(
            f"{warc_hash}: {len(labels)} label rows unused but the catalog dropped {n_dropped} documents "
            "-- B's shard and the label set disagree"
        )
    write_attribute_tables(dataset, shard, rows, label_names)
    return {"warc_hash": warc_hash, "n_rows": len(rows), "n_unused_labels": len(labels)}


def _pending_warcs(root: str, dataset: str, label_base: str) -> list[str]:
    """WARCs with a written record shard whose attribute tables are not both done yet."""
    grid_corpora.OUTPUT_BASE = label_base
    with_output = []
    for path in fsspec_glob(f"{root}/catalog/data-*.json"):
        with fsspec.open(path, "r") as fh:
            catalog = json.load(fh)
        if not catalog.get("skipped"):
            with_output.append(catalog["warc_hash"])
    done = set()
    for stage in Stage:
        marker_dir = done_marker(dataset, stage, shard_path(root, "x")).rsplit("/", 1)[0]
        stems = {p.rsplit("/", 1)[-1] for p in fsspec_glob(f"{marker_dir}/*")}
        done = stems if stage is Stage.TOPIC else done & stems
    pending = [h for h in with_output if f"data-{h}" not in done]
    logger.info(
        "%d WARCs with output, %d already attached, %d pending",
        len(with_output),
        len(with_output) - len(pending),
        len(pending),
    )
    return pending


def run(root: str, dataset: str, label_base: str, max_workers: int) -> dict:
    with fsspec.open(f"{root}/labels/_meta.json", "r") as fh:
        label_names = json.load(fh)["label_names"]
    pending = _pending_warcs(root, dataset, label_base)
    if not pending:
        return {"attached": 0}
    template = f"{root}/attach_summary/data-{{shard:05d}}-of-{{total:05d}}.jsonl"
    pipeline = (
        Dataset.from_iterable(pending)
        .map(functools.partial(attach_one, root=root, dataset=dataset, label_base=label_base, label_names=label_names))
        .write_jsonl(template)
    )
    ctx = ZephyrContext(
        name="swap-attach-labels", max_workers=max_workers, resources=ResourceConfig(cpu=1, ram="8g", disk="10g")
    )
    ctx.execute(pipeline)
    return {"attached": len(pending)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=OUTPUT_ROOT)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--label-base", default=LABEL_BASE)
    ap.add_argument("--max-workers", type=int, default=200)
    args = ap.parse_args()
    print(json.dumps(run(args.root, args.dataset, args.label_base, args.max_workers)))


if __name__ == "__main__":
    main()
