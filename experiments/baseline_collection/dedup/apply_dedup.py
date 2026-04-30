# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply a marin dedupe sidecar to filter out duplicate documents.

Marin's `dedup_*_document` functions don't delete anything — they emit a
sidecar parquet of duplicate ids at ``{dedup_output}/data/<rebased>.parquet``
with shape ``{id, attributes: {dup_doc: True}}``. This step joins each prepped
input shard with its sidecar parquet and writes a deduped jsonl.gz tree.

Cluster-size stats are also collected by reading the connected-components
final-iteration parquet at ``{dedup_output}/metadata/cc/it_<N>/*.parquet``
when a fuzzy sidecar is supplied.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import fsspec
import pyarrow.parquet as pq

from fray.v2.types import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters
from zephyr.readers import load_file as zephyr_load_file

from marin.utils import fsspec_glob, rebase_file_path

CLUSTER_SIZE_BIN_EDGES = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 100, "11-100")]
"""Edges for cluster_size_histogram. Sizes >100 fall into the ``>100`` bin."""

logger = logging.getLogger(__name__)


@dataclass
class ApplyDedupConfig:
    prepped_path: str
    """Root of `prep_with_id` output (jsonl.gz tree with `id` field)."""

    sidecar_paths: list[str]
    """One or more dedupe output roots to *union* (typical: [exact, fuzzy]).

    Each root must contain ``data/`` aligned by shard name with `prepped_path`.
    """

    output_path: str
    """Output root for deduped jsonl.gz tree."""

    max_parallelism: int = 200


def _load_dup_ids(sidecar_file: str) -> set[str]:
    """Read a sidecar parquet of dup ids. Empty set if file missing."""
    fs, path = fsspec.core.url_to_fs(sidecar_file)
    if not fs.exists(path):
        return set()
    table = pq.read_table(sidecar_file, columns=["id"])
    return set(table.column("id").to_pylist())


def _process_one_shard(item: dict[str, Any]) -> Iterator[dict]:
    prep_file = item["prep_file"]
    sidecar_files = item["sidecar_files"]

    dup_ids: set[str] = set()
    for s in sidecar_files:
        dup_ids |= _load_dup_ids(s)

    total = 0
    dropped = 0
    for record in zephyr_load_file(prep_file):
        total += 1
        if record["id"] in dup_ids:
            dropped += 1
            counters.increment("apply_dedup/dropped")
            continue
        counters.increment("apply_dedup/kept")
        yield record

    counters.increment("apply_dedup/total", total)
    logger.info("  %s: kept=%d / total=%d (dropped %d)", prep_file.split("/")[-1], total - dropped, total, dropped)


def apply_dedup(config: ApplyDedupConfig) -> dict:
    """Filter prepped jsonl.gz against one or more dedupe sidecars.

    Returns counts of total / dropped / kept records across all shards.
    """
    prep_files = sorted(fsspec_glob(f"{config.prepped_path.rstrip('/')}/*.jsonl.gz"))
    if not prep_files:
        raise FileNotFoundError(f"No prepped files under {config.prepped_path}")

    # For each prep file, locate the matching sidecar parquet under each sidecar root.
    tasks = []
    for prep_file in prep_files:
        sidecars = []
        for sc_root in config.sidecar_paths:
            sidecar_file = rebase_file_path(
                config.prepped_path,
                prep_file,
                f"{sc_root.rstrip('/')}/data/",
                old_extension=".jsonl.gz",
                new_extension=".parquet",
            )
            sidecars.append(sidecar_file)
        tasks.append({"prep_file": prep_file, "sidecar_files": sidecars})

    logger.info("apply_dedup: %d shards, %d sidecar root(s)", len(tasks), len(config.sidecar_paths))

    ctx = ZephyrContext(
        name="apply-dedup",
        max_workers=min(config.max_parallelism, len(tasks)),
        resources=ResourceConfig(cpu=1, ram="8g", disk="5g"),
    )

    pipeline = (
        Dataset.from_list(tasks)
        .flat_map(_process_one_shard)
        .write_jsonl(
            f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
            skip_existing=True,
        )
    )
    ctx.execute(pipeline)
    return {"success": True, "input_shards": len(prep_files), "sidecar_roots": len(config.sidecar_paths)}


def _bin_cluster_size(size: int) -> str:
    for lo, hi, label in CLUSTER_SIZE_BIN_EDGES:
        if lo <= size <= hi:
            return label
    return ">100"


def cluster_size_histogram(fuzzy_output_path: str, max_parallelism: int = 200) -> dict[str, Any]:
    """Read the final CC iteration and report cluster sizes via a parallel Zephyr job.

    Returns counts of clusters by size bin: {1, 2, 3-5, 6-10, 11-100, >100}.
    Singletons (size=1) are non-duplicates.

    Two shuffle stages: count members per component_id, then reduce to histogram.
    Avoids holding all component_ids on a single coordinator (would OOM at >100M
    docs).
    """
    cc_root = f"{fuzzy_output_path.rstrip('/')}/metadata/cc"
    fs, root_path = fsspec.core.url_to_fs(cc_root)
    if not fs.exists(root_path):
        logger.warning("No CC metadata at %s", cc_root)
        return {}

    iter_dirs = sorted([d for d in fs.ls(root_path, detail=False) if "it_" in d])
    if not iter_dirs:
        return {}
    last = f"gs://{iter_dirs[-1]}" if cc_root.startswith("gs://") else iter_dirs[-1]
    parts = sorted(fs.ls(last, detail=False))
    parts = [(f"gs://{p}" if cc_root.startswith("gs://") else p) for p in parts if p.endswith(".parquet")]
    if not parts:
        return {}

    ctx = ZephyrContext(
        name="cluster-size-hist",
        max_workers=min(max_parallelism, len(parts)),
        resources=ResourceConfig(cpu=2, ram="32g", disk="5g"),
    )

    def _count_members(_key: str, items: Iterator[dict]) -> Iterator[dict]:
        size = sum(1 for _ in items)
        yield {"bin": _bin_cluster_size(size)}

    def _bin_to_count(_key: str, items: Iterator[dict]) -> Iterator[dict]:
        bins: dict[str, int] = {label: 0 for _, _, label in CLUSTER_SIZE_BIN_EDGES}
        bins[">100"] = 0
        total = 0
        for item in items:
            bins[item["bin"]] += 1
            total += 1
        yield {"bins": bins, "total_clusters": total}

    pipeline = (
        Dataset.from_list(parts)
        .load_parquet()
        .map(lambda r: {"component_id": r["component_id"]})
        .group_by(lambda r: r["component_id"], reducer=_count_members)
        .group_by(lambda _: "global", reducer=_bin_to_count)
    )
    results = ctx.execute(pipeline).results
    final = results[0] if results else {"bins": {}, "total_clusters": 0}

    iteration = os.path.basename(last.rstrip("/")).removeprefix("it_")
    return {
        "cc_final_iteration": int(iteration) if iteration.isdigit() else iteration,
        "total_clusters": final["total_clusters"],
        "cluster_size_bins": final["bins"],
    }
