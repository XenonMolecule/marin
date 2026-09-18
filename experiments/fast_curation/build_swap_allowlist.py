# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regroup arm A's post-decon corpus by WARC for the extractor ablation (arm B).

Arm A is ``lpv11_fastpipe_v2`` after fuzzy dedup and CORE-v2 decon, 3,042 hash-partitioned shards.
The swap worker (``swap_extractor.py``) works one WARC at a time, so it needs A's documents grouped
by ``warc_hash``. Two independent passes over A's tree, selected by ``--stage``:

* ``allowlist`` -- provenance only (no text, no labels): ``{root}/allowlist/data-{warc_hash}.parquet``.
  Needs nothing but A's tree, so the extraction fleet can start before A's grid labels exist.
* ``labels`` -- A's 24x5 grid labels per document: ``{root}/labels/data-{warc_hash}.parquet``.
  Reads each A shard together with its co-partitioned topic and quality tables and verifies row for
  row that the label belongs to that document (equal row counts; topic ``native_id`` equals the
  record's ``doc_id``; topic and quality ``id`` agree; ``generate_id(text)`` equals the topic
  ``id``). Consumed by ``attach_swap_labels.py`` after the fleet.

Each stage also writes ``_meta.json`` next to its output with totals (and, for labels, the topic
label names in index order, read off one of A's per-shard tallies).

Run as an Iris CPU job in us-east5 (the only region holding A's tree and tables)::

    python -m experiments.fast_curation.build_swap_allowlist --stage allowlist
    python -m experiments.fast_curation.build_swap_allowlist --stage labels
"""

from __future__ import annotations

import argparse
import enum
import functools
import io
import json
import logging
import os
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
from experiments.baseline_collection.run_extract_standalone import _gcs_client
from experiments.fast_curation.swap_extractor import LABEL_BASE, OUTPUT_ROOT, RECORD_COLUMNS, SOURCE_DATASET
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

A_TREE = "gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_decon_deduped/10364warcs/deduped"

ALLOWLIST_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("fasttext_score", pa.float32()),
        ("pooled_prob", pa.float32()),
        ("modernbert_prob", pa.float32()),  # null = pooled hi-accept (never terminal-scored)
        ("a_shard", pa.string()),  # A shard stem the row came from
        ("a_row", pa.int32()),  # its row index there
    ]
)
LABELS_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("warc_hash", pa.string()),
        ("cluster_24", pa.int32()),
        ("topic_prob", pa.float32()),
        ("token_length", pa.int32()),
        ("quality_score", pa.float32()),
        ("quality_bucket", pa.int8()),
        ("a_shard", pa.string()),
        ("a_row", pa.int32()),
    ]
)


class Stage(enum.StrEnum):
    ALLOWLIST = "allowlist"
    LABELS = "labels"


# Per-file upload bound. The first allow-list run wedged forever inside gcsfs's ``_initiate_upload``
# in one Zephyr reduce subprocess (2026-09-07, 255/256 shards done, thread dump showed the event
# loop wait never returning), so reducer writes bypass fsspec and go through the storage client
# with an explicit timeout and a bounded retry.
UPLOAD_TIMEOUT_SECONDS = 300
UPLOAD_ATTEMPTS = 3


def _blob(path: str):
    bucket, name = path[len("gs://") :].split("/", 1)
    return _gcs_client().bucket(bucket).blob(name)


def output_exists(path: str) -> bool:
    if path.startswith("gs://"):
        return _blob(path).exists(timeout=UPLOAD_TIMEOUT_SECONDS)
    return os.path.exists(path)


def write_parquet_bounded(path: str, table: pa.Table) -> None:
    """Write ``table`` as zstd parquet to ``path`` with a bounded, retried upload."""
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    data = buf.getvalue()
    if not path.startswith("gs://"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return
    blob = _blob(path)
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        try:
            blob.upload_from_string(data, content_type="application/octet-stream", timeout=UPLOAD_TIMEOUT_SECONDS)
            return
        except Exception as e:
            if attempt == UPLOAD_ATTEMPTS:
                raise
            logger.warning("upload of %s failed (attempt %d): %s; retrying", path, attempt, e)


def topic_table_path(label_base: str, dataset: str, stem: str) -> str:
    return f"{label_base}/{TOPIC_ROOT}/{dataset}_{GRID_SUFFIX}/{stem}.parquet"


def quality_table_path(label_base: str, dataset: str, stem: str) -> str:
    return f"{label_base}/{QUALITY_ROOT}/{dataset}_{GRID_SUFFIX}/outputs/main/{stem}.parquet"


def topic_counts_path(label_base: str, dataset: str, stem: str) -> str:
    return f"{label_base}/{TOPIC_ROOT}/{dataset}_{GRID_SUFFIX}/counts/{stem}.json"


def _read_parquet(path: str, columns: list[str]) -> pa.Table:
    with fsspec.open(path, "rb") as fh:
        return pq.read_table(fh, columns=columns)


def provenance_rows(shard_path: str) -> Iterator[dict]:
    """Yield one provenance row (no text) per document of A shard ``shard_path``, in file order."""
    stem = output_stem(shard_path)
    for i, record in enumerate(iter_records(shard_path, Format.JSONL_GZ)):
        row = {name: record.get(name) for name in RECORD_COLUMNS}
        row.update(a_shard=stem, a_row=i)
        yield row


def labeled_rows(shard_path: str, label_base: str, dataset: str) -> Iterator[dict]:
    """Yield one label row per document of A shard ``shard_path``, verified against its tables."""
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
        if native_ids[i] != record["doc_id"]:
            raise RuntimeError(f"{stem} row {i}: topic native_id {native_ids[i]!r} != doc_id {record['doc_id']!r}")
        if generate_id(record.get("text") or "") != topic_ids[i]:
            raise RuntimeError(f"{stem} row {i}: label id is not the content hash of this document's text")
        yield {
            "doc_id": record["doc_id"],
            "warc_hash": record["warc_hash"],
            "cluster_24": clusters[i],
            "topic_prob": topic_probs[i],
            "token_length": lengths[i],
            "quality_score": scores[i],
            "quality_bucket": buckets[i],
            "a_shard": stem,
            "a_row": i,
        }
    if i + 1 != n:
        raise RuntimeError(f"{stem}: {i + 1} documents but {n} label rows -- co-partitioning broken")


def stage_dir(root: str, stage: Stage) -> str:
    return f"{root}/{stage.value}"


def shard_output(root: str, stage: Stage, warc_hash: str) -> str:
    return f"{stage_dir(root, stage)}/data-{warc_hash}.parquet"


def write_warc_shard(warc_hash: str, rows: Iterator[dict], root: str, stage: Stage, schema: pa.Schema) -> dict:
    """group_by reducer: write one WARC's parquet for ``stage`` (skipping an existing file); return its row count."""
    materialized = list(rows)
    path = shard_output(root, stage, warc_hash)
    if not output_exists(path):
        write_parquet_bounded(path, pa.Table.from_pylist(materialized, schema=schema))
    return {"warc_hash": warc_hash, "n_rows": len(materialized)}


def load_label_names(label_base: str, dataset: str, stem: str) -> list[str]:
    """Topic label names in index order, read off one of A's per-shard tallies (dicts keep insertion order)."""
    with fsspec.open(topic_counts_path(label_base, dataset, stem), "r") as fh:
        return list(json.load(fh)["counts"].keys())


def run(stage: Stage, root: str, a_tree: str, label_base: str, dataset: str, max_workers: int) -> dict:
    shards = sorted(fsspec_glob(f"{a_tree}/data-*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no A shards under {a_tree}")
    rows_fn: Callable[[str], Iterator[dict]]
    if stage is Stage.ALLOWLIST:
        rows_fn, schema = provenance_rows, ALLOWLIST_SCHEMA
    else:
        rows_fn = functools.partial(labeled_rows, label_base=label_base, dataset=dataset)
        schema = LABELS_SCHEMA
    out_dir = stage_dir(root, stage)
    logger.info("%s: %d A shards -> %s", stage.value, len(shards), out_dir)
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
        name=f"swap-{stage.value}", max_workers=max_workers, resources=ResourceConfig(cpu=1, ram="8g", disk="10g")
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
        "a_tree": a_tree,
        "n_a_shards": len(shards),
        "n_rows": sum(per_warc.values()),
        "n_warcs": len(per_warc),
        "n_files": n_files,
    }
    if stage is Stage.LABELS:
        meta["label_names"] = load_label_names(label_base, dataset, output_stem(shards[0]))
    with fsspec.open(f"{out_dir}/_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("%s done: %s", stage.value, json.dumps(meta))
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=Stage, choices=list(Stage), required=True)
    ap.add_argument("--root", default=OUTPUT_ROOT)
    ap.add_argument("--a-tree", default=A_TREE)
    ap.add_argument("--label-base", default=LABEL_BASE)
    ap.add_argument("--source-dataset", default=SOURCE_DATASET)
    ap.add_argument("--max-workers", type=int, default=200)
    args = ap.parse_args()
    meta = run(args.stage, args.root, args.a_tree, args.label_base, args.source_dataset, args.max_workers)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
