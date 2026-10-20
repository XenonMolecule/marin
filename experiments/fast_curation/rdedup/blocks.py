# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Partitioned shuffle objects: one object per map task, one byte range per reduce partition.

A map task writes a single object holding ``n_partitions`` contiguous blocks, block ``p``
carrying the records destined for reduce partition ``p``. A block is a sequence of frames,
``u64 length`` then a zstd frame; a frame decompresses to either raw bytes of a numpy
structured array or one Arrow IPC stream. The map task's done mark records the block offsets
and the object's generation.

When a map stage seals, its offsets are transposed into one index object of shape
``(n_partitions + 1, n_maps)``: reduce partition ``p`` then needs one small ranged read (rows
``p`` and ``p + 1``) to locate its block in every map object, and one ranged read per
non-empty block, pinned to the recorded generation so a rewritten object fails loudly instead
of returning mixed bytes.

The number of reads a shuffle costs is ``n_maps x n_partitions``; stage sizing in
``plan.py`` keeps that product in the low millions.
"""

from __future__ import annotations

import io
import struct
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import zstandard as zstd

from experiments.fast_curation.rdedup.store import ObjectStore

_FRAME_HEADER = struct.Struct("<Q")
_LEVEL = 3
READ_THREADS = 48


def _frame(payload: bytes) -> bytes:
    compressed = zstd.ZstdCompressor(level=_LEVEL).compress(payload)
    return _FRAME_HEADER.pack(len(compressed)) + compressed


def iter_frames(block: bytes) -> Iterator[bytes]:
    """Decompressed payloads of every frame in ``block``."""
    view = memoryview(block)
    position = 0
    decompressor = zstd.ZstdDecompressor()
    while position < len(view):
        (length,) = _FRAME_HEADER.unpack_from(view, position)
        position += _FRAME_HEADER.size
        yield decompressor.decompress(bytes(view[position : position + length]))
        position += length


def table_to_ipc(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def ipc_to_table(payload: bytes) -> pa.Table:
    return pa.ipc.open_stream(payload).read_all()


def split_by_partition(partitions: np.ndarray, n_partitions: int) -> tuple[np.ndarray, np.ndarray]:
    """``(order, bounds)``: ``order`` groups rows by partition (stable); partition ``p`` is
    ``order[bounds[p]:bounds[p+1]]``."""
    order = np.argsort(partitions, kind="stable")
    counts = np.bincount(partitions, minlength=n_partitions)
    bounds = np.zeros(n_partitions + 1, dtype=np.int64)
    np.cumsum(counts, out=bounds[1:])
    return order, bounds


class BlockObjectWriter:
    """Builds one map task's partitioned object from frames added per partition.

    Frames are compressed as they are added, so the writer holds compressed bytes only. Call
    :meth:`add_array` / :meth:`add_table` any number of times; each call adds one frame per
    non-empty partition.
    """

    def __init__(self, n_partitions: int) -> None:
        self.n_partitions = n_partitions
        self._frames: list[list[bytes]] = [[] for _ in range(n_partitions)]
        self._rows = np.zeros(n_partitions, dtype=np.int64)

    def add_array(self, records: np.ndarray, partitions: np.ndarray) -> None:
        if len(records) != len(partitions):
            raise ValueError("records and partitions differ in length")
        if len(records) == 0:
            return
        order, bounds = split_by_partition(partitions, self.n_partitions)
        grouped = records[order]
        for p in np.flatnonzero(np.diff(bounds)):
            piece = np.ascontiguousarray(grouped[bounds[p] : bounds[p + 1]])
            self._frames[p].append(_frame(piece.tobytes()))
            self._rows[p] += len(piece)

    def add_table(self, table: pa.Table, partitions: np.ndarray) -> None:
        if table.num_rows != len(partitions):
            raise ValueError("table and partitions differ in length")
        if table.num_rows == 0:
            return
        order, bounds = split_by_partition(partitions, self.n_partitions)
        grouped = table.take(pa.array(order))
        for p in np.flatnonzero(np.diff(bounds)):
            piece = grouped.slice(bounds[p], bounds[p + 1] - bounds[p])
            self._frames[p].append(_frame(table_to_ipc(piece)))
            self._rows[p] += piece.num_rows

    def finish(self, store: ObjectStore, path: str) -> dict:
        """Write the object and return the metadata its done mark must carry."""
        offsets = [0]
        for frames in self._frames:
            offsets.append(offsets[-1] + sum(len(f) for f in frames))

        def pieces() -> Iterator[bytes]:
            for frames in self._frames:
                yield from frames

        generation = store.write(path, pieces())
        return {
            "path": path,
            "generation": generation,
            "offsets": offsets,
            "rows": int(self._rows.sum()),
            "bytes": offsets[-1],
        }


def write_single_block(store: ObjectStore, path: str, records: np.ndarray | pa.Table) -> dict:
    """Write an unpartitioned object (one block) and return its metadata."""
    writer = BlockObjectWriter(1)
    if isinstance(records, pa.Table):
        writer.add_table(records, np.zeros(records.num_rows, dtype=np.int64))
    else:
        writer.add_array(records, np.zeros(len(records), dtype=np.int64))
    return writer.finish(store, path)


def read_single_block_array(store: ObjectStore, meta: dict, dtype: np.dtype) -> np.ndarray:
    block = store.read(meta["path"], 0, meta["offsets"][-1], generation=meta["generation"])
    return concat_arrays([np.frombuffer(f, dtype=dtype) for f in iter_frames(block)], dtype)


def concat_arrays(pieces: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=dtype)


def index_path(data_dir: str) -> str:
    return f"{data_dir}/_index.bin"


def build_index(store: ObjectStore, data_dir: str, map_results: list[dict], n_partitions: int) -> dict:
    """Seal-time index over a map stage's objects (see module docstring)."""
    n_maps = len(map_results)
    offsets = np.zeros((n_partitions + 1, n_maps), dtype=np.uint64)
    generations = np.zeros(n_maps, dtype=np.uint64)
    for m, result in enumerate(map_results):
        if len(result["offsets"]) != n_partitions + 1:
            raise ValueError(f"map {m} wrote {len(result['offsets']) - 1} blocks, expected {n_partitions}")
        offsets[:, m] = result["offsets"]
        generations[m] = result["generation"]
    payload = generations.tobytes() + offsets.tobytes()
    store.write(index_path(data_dir), payload)
    return {"n_maps": n_maps, "n_partitions": n_partitions, "bytes": int(offsets[-1].sum())}


def _map_path(data_dir: str, m: int) -> str:
    return f"{data_dir}/{m:06d}.bin"


def map_object_path(data_dir: str, m: int) -> str:
    return _map_path(data_dir, m)


def iter_partition_blocks(
    store: ObjectStore, data_dir: str, partition: int, n_maps: int, n_partitions: int, batch: int = 4 * READ_THREADS
) -> Iterator[tuple[int, bytes]]:
    """``(map index, raw block)`` for ``partition`` from every map object of a sealed stage.

    Blocks are fetched ``batch`` maps at a time, so a caller that consumes them one by one holds
    at most a batch of compressed input, however skewed the partition.
    """
    if not 0 <= partition < n_partitions:
        raise ValueError(f"partition {partition} out of range [0, {n_partitions})")
    header = n_maps * 8
    generations = np.frombuffer(store.read(index_path(data_dir), 0, header), dtype=np.uint64)
    start = header + partition * n_maps * 8
    rows = np.frombuffer(store.read(index_path(data_dir), start, start + 2 * n_maps * 8), dtype=np.uint64)
    lo, hi = rows[:n_maps], rows[n_maps:]

    def fetch(m: int) -> bytes:
        if hi[m] == lo[m]:
            return b""
        return store.read(_map_path(data_dir, m), int(lo[m]), int(hi[m]), generation=int(generations[m]))

    with ThreadPoolExecutor(READ_THREADS) as pool:
        for first in range(0, n_maps, batch):
            maps = range(first, min(first + batch, n_maps))
            yield from zip(maps, pool.map(fetch, maps), strict=True)


def partition_bytes(store: ObjectStore, data_dir: str, partition: int, n_maps: int, n_partitions: int) -> int:
    """Total compressed bytes of ``partition`` across a sealed stage's map objects."""
    start = n_maps * 8 + partition * n_maps * 8
    rows = np.frombuffer(store.read(index_path(data_dir), start, start + 2 * n_maps * 8), dtype=np.uint64)
    return int((rows[n_maps:] - rows[:n_maps]).sum())


def read_partition_blocks(
    store: ObjectStore, data_dir: str, partition: int, n_maps: int, n_partitions: int
) -> list[bytes]:
    """Raw blocks for ``partition`` from every map object of a sealed stage, in map order."""
    return [block for _m, block in iter_partition_blocks(store, data_dir, partition, n_maps, n_partitions)]


def read_partition_array(
    store: ObjectStore, data_dir: str, partition: int, n_maps: int, n_partitions: int, dtype: np.dtype
) -> np.ndarray:
    blocks = read_partition_blocks(store, data_dir, partition, n_maps, n_partitions)
    pieces = [np.frombuffer(frame, dtype=dtype) for block in blocks for frame in iter_frames(block)]
    return concat_arrays(pieces, dtype)


def read_partition_tables(
    store: ObjectStore, data_dir: str, partition: int, n_maps: int, n_partitions: int
) -> list[pa.Table]:
    blocks = read_partition_blocks(store, data_dir, partition, n_maps, n_partitions)
    return [ipc_to_table(frame) for block in blocks for frame in iter_frames(block)]
