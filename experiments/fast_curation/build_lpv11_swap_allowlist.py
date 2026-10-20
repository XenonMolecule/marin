# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regroup the lpv11 LLM-pipeline corpus by WARC for the fork-extraction swap.

The source tree is 3,193 **index-named** shards (``data-NNNNN-of-03193.jsonl.gz``), while the swap
worker processes one WARC at a time, so the documents have to be regrouped by ``warc_hash`` first.
Two independent stages:

* ``--stage allowlist`` -- provenance only, no text and no labels, into
  ``{root}/allowlist/data-{warc_hash}.parquet``. Depends on nothing but the source tree, so the
  extraction fleet can start as soon as it lands.
* ``--stage labels`` -- lpv11's 24x5 grid labels per document, into
  ``{root}/labels/data-{warc_hash}.parquet``, for ``attach_lpv11_fork_labels``.

lpv11 names its provenance differently from the fastpipe line, and the mapping is exact:

===================  ==========================================================
fastpipe column      lpv11 equivalent
===================  ==========================================================
``doc_id``           ``warc_record_id`` (both are the urn:uuid-stripped Record-ID)
``warc_hash``        ``sha256(warc_file)[:12]`` -- derived, not stored
===================  ==========================================================

Guards on the label stage, all fatal: equal row counts across the shard and its two label
parquets; ``topic.native_id == warc_record_id`` for every row; ``topic.id == quality.id``; and
``generate_id(text) == topic.id``, which proves the label was computed on exactly this text.

Run as an Iris CPU job in us-east5 -- the only region with a complete topic table (us-central1 has
17 of 3,193)::

    python -m experiments.fast_curation.build_lpv11_swap_allowlist --stage allowlist
    python -m experiments.fast_curation.build_lpv11_swap_allowlist --stage labels
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
from collections.abc import Callable, Iterator

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import generate_id
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.baseline_collection.grid_corpora import (
    GRID_SUFFIX,
    QUALITY_ROOT,
    TOPIC_ROOT,
    Format,
    iter_records,
    output_stem,
)
from experiments.fast_curation.build_swap_allowlist import Stage, output_exists, write_parquet_bounded
from experiments.fast_curation.lpv11_fork_extractor import LABEL_BASE, OUTPUT_ROOT, SOURCE_DATASET, SOURCE_TREE
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

ALLOWLIST_SCHEMA = pa.schema(
    [
        ("warc_record_id", pa.string()),
        ("url", pa.string()),
        ("warc_file", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("src_shard", pa.string()),
        ("src_row", pa.int32()),
    ]
)
LABELS_SCHEMA = pa.schema(
    [
        ("warc_record_id", pa.string()),
        ("warc_hash", pa.string()),
        ("cluster_24", pa.int32()),
        ("topic_prob", pa.float32()),
        ("token_length", pa.int32()),
        ("quality_score", pa.float32()),
        ("quality_bucket", pa.int8()),
        ("src_shard", pa.string()),
        ("src_row", pa.int32()),
    ]
)


def warc_hash_of(warc_file: str) -> str:
    """``decode_warcs_clean._warc_path_hash`` applied to the stored WARC path."""
    return hashlib.sha256(warc_file.encode()).hexdigest()[:12]


def topic_table_path(label_base: str, dataset: str, stem: str) -> str:
    return f"{label_base}/{TOPIC_ROOT}/{dataset}_{GRID_SUFFIX}/{stem}.parquet"


def quality_table_path(label_base: str, dataset: str, stem: str) -> str:
    return f"{label_base}/{QUALITY_ROOT}/{dataset}_{GRID_SUFFIX}/outputs/main/{stem}.parquet"


def _read_parquet(path: str, columns: list[str]) -> pa.Table:
    with fsspec.open(path, "rb") as fh:
        return pq.read_table(fh, columns=columns)


def provenance_rows(shard_path: str) -> Iterator[dict]:
    """One provenance row per document of ``shard_path``, in file order, no text."""
    stem = output_stem(shard_path)
    for i, record in enumerate(iter_records(shard_path, Format.JSONL_GZ)):
        yield {
            "warc_record_id": record["warc_record_id"],
            "url": record.get("url") or "",
            "warc_file": record["warc_file"],
            "warc_hash": warc_hash_of(record["warc_file"]),
            "snapshot": record.get("snapshot") or "",
            "src_shard": stem,
            "src_row": i,
        }


def labeled_rows(shard_path: str, label_base: str, dataset: str) -> Iterator[dict]:
    """One label row per document of ``shard_path``, verified against lpv11's own label tables."""
    stem = output_stem(shard_path)
    topic = _read_parquet(
        topic_table_path(label_base, dataset, stem), ["id", "native_id", "cluster_24", "topic_prob", "token_length"]
    )
    quality = _read_parquet(quality_table_path(label_base, dataset, stem), ["id", "score", "quality_bucket"])
    n = topic.num_rows
    if quality.num_rows != n:
        raise RuntimeError(f"{stem}: topic has {n} rows, quality has {quality.num_rows} -- co-partitioning broken")
    if n and not pc.all(pc.equal(topic.column("id"), quality.column("id"))).as_py():
        raise RuntimeError(f"{stem}: topic/quality id mismatch -- co-partitioning broken")
    topic_ids = topic.column("id").to_pylist()
    native_ids = topic.column("native_id").to_pylist()
    clusters = topic.column("cluster_24").to_pylist()
    topic_probs = topic.column("topic_prob").to_pylist()
    lengths = topic.column("token_length").to_pylist()
    scores = quality.column("score").to_pylist()
    buckets = quality.column("quality_bucket").to_pylist()

    i = -1
    for i, record in enumerate(iter_records(shard_path, Format.JSONL_GZ)):
        if i >= n:
            raise RuntimeError(f"{stem}: more documents than label rows ({n}) -- co-partitioning broken")
        if native_ids[i] != record["warc_record_id"]:
            raise RuntimeError(
                f"{stem} row {i}: topic native_id {native_ids[i]!r} != warc_record_id {record['warc_record_id']!r}"
            )
        if generate_id(record.get("text") or "") != topic_ids[i]:
            raise RuntimeError(f"{stem} row {i}: label id is not the content hash of this document's text")
        yield {
            "warc_record_id": record["warc_record_id"],
            "warc_hash": warc_hash_of(record["warc_file"]),
            "cluster_24": clusters[i],
            "topic_prob": topic_probs[i],
            "token_length": lengths[i],
            "quality_score": scores[i],
            "quality_bucket": buckets[i],
            "src_shard": stem,
            "src_row": i,
        }
    if i + 1 != n:
        raise RuntimeError(f"{stem}: {i + 1} documents but {n} label rows -- co-partitioning broken")


def stage_dir(root: str, stage: Stage) -> str:
    return f"{root}/{stage.value}"


def shard_output(root: str, stage: Stage, warc_hash: str) -> str:
    return f"{stage_dir(root, stage)}/data-{warc_hash}.parquet"


def write_warc_shard(warc_hash: str, rows: Iterator[dict], root: str, stage: Stage, schema: pa.Schema) -> dict:
    """group_by reducer: write one WARC's parquet (skipping an existing file); return its row count."""
    materialized = list(rows)
    path = shard_output(root, stage, warc_hash)
    if not output_exists(path):
        write_parquet_bounded(path, pa.Table.from_pylist(materialized, schema=schema))
    return {"warc_hash": warc_hash, "n_rows": len(materialized)}


def run(stage: Stage, root: str, src_tree: str, label_base: str, dataset: str, max_workers: int) -> dict:
    shards = sorted(fsspec_glob(f"{src_tree}/data-*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no source shards under {src_tree}")
    rows_fn: Callable[[str], Iterator[dict]]
    if stage is Stage.ALLOWLIST:
        rows_fn, schema = provenance_rows, ALLOWLIST_SCHEMA
    else:
        rows_fn = functools.partial(labeled_rows, label_base=label_base, dataset=dataset)
        schema = LABELS_SCHEMA
    out_dir = stage_dir(root, stage)
    logger.info("%s: %d source shards -> %s", stage.value, len(shards), out_dir)
    summary_template = f"{out_dir}_summary/data-{{shard:05d}}-of-{{total:05d}}.jsonl"
    pipeline = (
        Dataset.from_iterable(shards)
        .flat_map(rows_fn)
        .group_by(
            key=lambda row: row["warc_hash"],
            reducer=functools.partial(write_warc_shard, root=root, stage=stage, schema=schema),
            num_output_shards=256,
        )
        .write_jsonl(summary_template)
    )
    ctx = ZephyrContext(
        name=f"lpv11-fork-{stage.value}", max_workers=max_workers, resources=ResourceConfig(cpu=1, ram="8g", disk="10g")
    )
    ctx.execute(pipeline)

    per_warc: dict[str, int] = {}
    for path in fsspec_glob(f"{out_dir}_summary/*.jsonl"):
        with fsspec.open(path, "r") as fh:
            for line in fh:
                if line.strip():
                    item = json.loads(line)
                    per_warc[item["warc_hash"]] = item["n_rows"]
    n_files = len(fsspec_glob(f"{out_dir}/data-*.parquet"))
    if n_files != len(per_warc):
        raise RuntimeError(f"{n_files} {stage.value} files but {len(per_warc)} WARCs summarized")
    meta: dict[str, object] = {
        "stage": stage.value,
        "source_dataset": dataset,
        "src_tree": src_tree,
        "n_src_shards": len(shards),
        "n_rows": sum(per_warc.values()),
        "n_warcs": len(per_warc),
        "n_files": n_files,
    }
    if stage is Stage.LABELS:
        with fsspec.open(
            f"{label_base}/{TOPIC_ROOT}/{dataset}_{GRID_SUFFIX}/counts/{output_stem(shards[0])}.json", "r"
        ) as fh:
            meta["label_names"] = list(json.load(fh)["counts"].keys())
    with fsspec.open(f"{out_dir}/_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("%s done: %s", stage.value, json.dumps({k: v for k, v in meta.items() if k != "label_names"}))
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=Stage, choices=list(Stage), required=True)
    ap.add_argument("--root", default=OUTPUT_ROOT)
    ap.add_argument("--src-tree", default=SOURCE_TREE)
    ap.add_argument("--label-base", default=LABEL_BASE)
    ap.add_argument("--source-dataset", default=SOURCE_DATASET)
    ap.add_argument("--max-workers", type=int, default=200)
    args = ap.parse_args()
    meta = run(args.stage, args.root, args.src_tree, args.label_base, args.source_dataset, args.max_workers)
    print(json.dumps({k: v for k, v in meta.items() if k != "label_names"}, indent=2))


if __name__ == "__main__":
    main()
