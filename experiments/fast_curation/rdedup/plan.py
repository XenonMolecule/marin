# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The frozen plan of one dedup run: inputs, units, partition counts, output locations.

A run is identified by its scratch root. ``plan.json`` is created there exactly once (atomic
create); every worker loads it and derives every task, path and partition from it, never from
a listing. Nothing a failed or differently-configured attempt left under another root can leak
into this run, and a second ``init`` with different settings for the same root is refused.

Units are contiguous runs of the sorted kept file list, packed to a byte budget so each
pass-1 and apply task reads about the same amount. Node partitions (connected components) are
contiguous runs of units, so a component step's output for a partition maps straight back to
the units whose documents it holds.
"""

from __future__ import annotations

import dataclasses
import io
import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fast_curation.rdedup.docs import key_file
from experiments.fast_curation.rdedup.store import ObjectStore, PreconditionFailed, dumps, read_json

PLAN_VERSION = 1
# Bytes of compressed kept parquet per document, measured on the 2M corpus (26.5 TB / 15.1B docs).
_COMPRESSED_BYTES_PER_DOC = 1750
_EXACT_RECORD_BYTES = 24
_LSH_RECORD_BYTES = 16 * 26
# Compressed member text per input document (members ~40% of docs, 5.4 KB text, zstd ~2.5x).
_C075_BYTES_PER_DOC = 900


@dataclasses.dataclass(frozen=True)
class Plan:
    """Everything a worker needs to know about a run. Serialized as ``plan.json``."""

    run_root: str
    output_root: str
    spec_id: str
    n_files: int
    unit_starts: list[int]  # first file index of each unit, plus n_files at the end
    exact_partitions: int
    lsh_partitions: int
    units_per_node_partition: int
    c075_partitions: int
    files_per_output: int
    rules: list[str]
    version: int = PLAN_VERSION

    @property
    def n_units(self) -> int:
        return len(self.unit_starts) - 1

    @property
    def node_partitions(self) -> int:
        return math.ceil(self.n_units / self.units_per_node_partition)

    def unit_files(self, unit: int) -> range:
        return range(self.unit_starts[unit], self.unit_starts[unit + 1])

    def node_partition_units(self, partition: int) -> range:
        start = partition * self.units_per_node_partition
        return range(start, min(start + self.units_per_node_partition, self.n_units))

    def unit_of_keys(self, keys: np.ndarray) -> np.ndarray:
        starts = np.asarray(self.unit_starts, dtype=np.uint64)
        return np.searchsorted(starts, key_file(keys), side="right").astype(np.int64) - 1

    def node_partition_of_keys(self, keys: np.ndarray) -> np.ndarray:
        return self.unit_of_keys(keys) // self.units_per_node_partition

    def to_json(self) -> bytes:
        return dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(payload: dict) -> Plan:
        if payload.get("version") != PLAN_VERSION:
            raise ValueError(f"plan version {payload.get('version')} != {PLAN_VERSION}")
        return Plan(**payload)


def plan_path(run_root: str) -> str:
    return f"{run_root.rstrip('/')}/plan.json"


def inputs_path(run_root: str) -> str:
    return f"{run_root.rstrip('/')}/inputs.parquet"


def pack_units(sizes: list[int], target_bytes: int) -> list[int]:
    """Unit start offsets over files in order, each unit holding about ``target_bytes``."""
    starts = [0]
    running = 0
    for i, size in enumerate(sizes):
        if running and running + size > target_bytes:
            starts.append(i)
            running = 0
        running += size
    starts.append(len(sizes))
    return starts


def build_plan(
    *,
    run_root: str,
    output_root: str,
    spec_id: str,
    sizes: list[int],
    unit_bytes: int,
    rules: list[str],
    exact_partition_bytes: int = 1 << 30,
    lsh_partition_bytes: int = 2 << 30,
    node_partitions: int | None = None,
    c075_partition_bytes: int = 3 << 29,
    files_per_output: int = 32,
) -> Plan:
    """Size units and partitions from the total input bytes."""
    total = sum(sizes)
    docs = total / _COMPRESSED_BYTES_PER_DOC
    unit_starts = pack_units(sizes, unit_bytes)
    n_units = len(unit_starts) - 1
    if node_partitions is None:
        node_partitions = max(1, n_units // 4)
    return Plan(
        run_root=run_root.rstrip("/"),
        output_root=output_root.rstrip("/"),
        spec_id=spec_id,
        n_files=len(sizes),
        unit_starts=unit_starts,
        exact_partitions=max(1, math.ceil(docs * _EXACT_RECORD_BYTES / exact_partition_bytes)),
        lsh_partitions=max(1, math.ceil(docs * _LSH_RECORD_BYTES / lsh_partition_bytes)),
        units_per_node_partition=max(1, math.ceil(n_units / node_partitions)),
        c075_partitions=max(1, math.ceil(docs * _C075_BYTES_PER_DOC / c075_partition_bytes)),
        files_per_output=files_per_output,
        rules=rules,
    )


def write_plan(store: ObjectStore, plan: Plan, paths: list[str], sizes: list[int]) -> None:
    """Create ``plan.json`` and the input list once. Re-running with the same plan is a no-op."""
    table = pa.table({"path": paths, "size": pa.array(sizes, type=pa.int64())})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    existing = store.stat(plan_path(plan.run_root))
    if existing is not None:
        current = read_json(store, plan_path(plan.run_root))
        if current != json.loads(plan.to_json()):
            raise ValueError(f"{plan_path(plan.run_root)} exists with a different plan; use a new run root")
        if load_inputs(store, plan.run_root) != paths:
            raise ValueError(f"{inputs_path(plan.run_root)} holds a different input list; use a new run root")
        return
    store.write(inputs_path(plan.run_root), buffer.getvalue())
    try:
        store.create(plan_path(plan.run_root), plan.to_json())
    except PreconditionFailed as e:
        raise ValueError(f"another init created {plan_path(plan.run_root)} concurrently") from e


def load_plan(store: ObjectStore, run_root: str) -> Plan:
    return Plan.from_json(read_json(store, plan_path(run_root)))


def load_inputs(store: ObjectStore, run_root: str) -> list[str]:
    table = pq.read_table(io.BytesIO(store.read(inputs_path(run_root))), columns=["path"])
    return table.column("path").to_pylist()
