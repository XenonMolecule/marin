# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach lpv11's grid labels to the fork-extracted shards and write co-partitioned tables.

Runs after ``lpv11_fork_extractor`` and after ``build_lpv11_swap_allowlist --stage labels``. For
each WARC it joins the new record shard with that WARC's label parquet on ``warc_record_id`` and
writes the new arm's topic and quality tables plus ``_done`` markers next to the shard's stem, so
``grid_label merge``, ``grid_tokenize`` and ``grid_store`` treat it like any labelled corpus
without either model running on the new text.

Guards, fatal per WARC: every record has a label; and the unused label rows are exactly the
documents the catalog recorded as dropped (short, crashed, or missing). Resumable -- a WARC whose
two ``_done`` markers exist is skipped.

    python -m experiments.fast_curation.attach_lpv11_fork_labels
"""

from __future__ import annotations

import argparse
import functools
import json
import logging

import fsspec
import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import generate_id
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.baseline_collection import grid_corpora
from experiments.baseline_collection.grid_corpora import ShardDocs, Stage, done_marker, write_done
from experiments.baseline_collection.grid_label import write_quality_shard, write_topic_shard
from experiments.fast_curation.lpv11_fork_extractor import DATASET, LABEL_BASE, OUTPUT_ROOT, read_jsonl_gz, shard_path
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

LABEL_COLUMNS = ("cluster_24", "topic_prob", "token_length", "quality_score", "quality_bucket")


def write_attribute_tables(dataset: str, shard: str, rows: list[dict], label_names: list[str]) -> None:
    """Write the arm's topic + quality tables and ``_done`` markers for record shard ``shard``.

    Every output comes from the one ``rows`` list in one order, which is the co-partitioning
    contract ``datakit_store`` enforces. ``grid_corpora.OUTPUT_BASE`` must already point at the
    label region.
    """
    docs = ShardDocs(
        ids=[generate_id(row["text"]) for row in rows],
        native_ids=[row["warc_record_id"] for row in rows],
        urls=[row["url"] for row in rows],
        texts=[row["text"] for row in rows],
    )
    write_topic_shard(
        dataset,
        shard,
        docs,
        np.asarray([row["cluster_24"] for row in rows], dtype=np.int32),
        np.asarray([row["topic_prob"] for row in rows], dtype=np.float32),
        np.asarray([row["token_length"] for row in rows], dtype=np.int64),
        label_names,
    )
    write_quality_shard(
        dataset,
        shard,
        docs,
        np.asarray([row["quality_score"] for row in rows], dtype=np.float32),
        np.asarray([row["quality_bucket"] for row in rows], dtype=np.int8),
    )
    write_done(dataset, Stage.TOPIC, shard, len(rows))
    write_done(dataset, Stage.QUALITY, shard, len(rows))


def attach_one(warc_hash: str, root: str, dataset: str, label_base: str, label_names: list[str]) -> dict:
    grid_corpora.OUTPUT_BASE = label_base
    shard = shard_path(root, warc_hash)
    with fsspec.open(f"{root}/catalog/data-{warc_hash}.json", "r") as fh:
        catalog = json.load(fh)
    with fsspec.open(f"{root}/labels/data-{warc_hash}.parquet", "rb") as fh:
        labels = {row["warc_record_id"]: row for row in pq.read_table(fh).to_pylist()}
    rows = []
    unlabeled = []
    for record in read_jsonl_gz(shard):
        label = labels.pop(record["warc_record_id"], None)
        if label is None:
            unlabeled.append(record["warc_record_id"])
            continue
        rows.append({**record, **{name: label[name] for name in LABEL_COLUMNS}})
    if unlabeled:
        raise RuntimeError(f"{warc_hash}: {len(unlabeled)} records have no lpv11 label (e.g. {unlabeled[0]})")
    # NOT + n_crashed: a crashed page's text is "", which already tripped the MIN_TEXT_CHARS floor
    # and so is counted inside n_short. Adding it double-counts and fails this check on exactly
    # the WARCs that hit a frameset segfault.
    n_dropped = catalog["n_short"] + len(catalog["missing"])
    if len(labels) != n_dropped:
        raise RuntimeError(
            f"{warc_hash}: {len(labels)} label rows unused but the catalog dropped {n_dropped} documents "
            "-- the shard and the label set disagree"
        )
    write_attribute_tables(dataset, shard, rows, label_names)
    return {"warc_hash": warc_hash, "n_rows": len(rows), "n_unused_labels": len(labels)}


def _pending_warcs(root: str, dataset: str, label_base: str) -> list[str]:
    grid_corpora.OUTPUT_BASE = label_base
    with_output = []
    for path in fsspec_glob(f"{root}/catalog/data-*.json"):
        with fsspec.open(path, "r") as fh:
            catalog = json.load(fh)
        if not catalog.get("skipped"):
            with_output.append(catalog["warc_hash"])
    done: set[str] = set()
    for stage in Stage:
        marker_dir = done_marker(dataset, stage, shard_path(root, "x")).rsplit("/", 1)[0]
        stems = {p.rsplit("/", 1)[-1] for p in fsspec_glob(f"{marker_dir}/*")}
        done = stems if stage is Stage.TOPIC else done & stems
    pending = [h for h in with_output if f"data-{h}" not in done]
    logger.info(
        "%d WARCs with output, %d attached, %d pending", len(with_output), len(with_output) - len(pending), len(pending)
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
        name="lpv11-fork-attach", max_workers=max_workers, resources=ResourceConfig(cpu=1, ram="8g", disk="10g")
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
