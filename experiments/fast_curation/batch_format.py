# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""On-disk contract for the CPU -> TPU hand-off and the Phase-2 outputs.

Single source of truth for three artifacts:

* **survivor parquet** (`cpu_survivors/data-{warc_hash}.parquet`) — one row per
  fastText survivor, written by the CPU phase and claimed/scored by the TPU phase.
  Token ids are stored **ragged** (a `list<int32>`, truncated per spec, NOT padded to a
  static `[max_length]`): padding on disk would waste ~5-20x the bytes, and the TPU pads
  to a length-bucket at load time anyway (see :func:`pad_batch`).

  > Truncation of ``input_ids`` is **spec-governed**. ``fastpipe_v1`` is single-window
  > (truncate to ``max_length``). A future chunked spec must store the FULL untruncated
  > token sequence so the chunked ModernBERT can see past ``max_length``; ``n_tokens``
  > records the true length so a chunked TPU phase can detect docs that were capped, and
  > the raw ``text`` is retained so re-tokenization never re-runs the CPU phase.

* **kept parquet** (`kept/data-{warc_hash}.parquet`) — survivors with
  ``modernbert_prob >= threshold`` plus all carried fields.
* **tombstone jsonl.gz** (`tombstones/data-{warc_hash}.jsonl.gz`) — ``{doc_id,
  modernbert_prob}`` for every *dropped* survivor, so re-thresholding ModernBERT is a
  cheap offline re-filter (no TPU re-run).
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Sequence

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_COMPRESSION = "zstd"

# One row per fastText survivor handed from the CPU phase to the TPU phase.
SURVIVOR_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("fasttext_score", pa.float32()),
        ("text", pa.large_string()),  # JustText extraction — the training content.
        ("input_ids", pa.list_(pa.int32())),  # ragged, truncated per spec.
        ("n_tokens", pa.int32()),  # true token count (pre-truncation cap aware).
    ]
)

# Kept docs = survivors that also passed ModernBERT. Survivor schema + the stored prob.
KEPT_SCHEMA = pa.schema([*SURVIVOR_SCHEMA, pa.field("modernbert_prob", pa.float32())])

# --- v2 (ModernBERT-before-JustText) schemas ---
# Phase A output: fastText-survivors carrying RAW html (for later JustText) instead of extracted text.
PRESURVIVOR_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("fasttext_score", pa.float32()),
        ("html", pa.large_string()),  # raw decoded page; Phase C runs JustText on this.
        ("input_ids", pa.list_(pa.int32())),
        ("n_tokens", pa.int32()),
    ]
)
# Phase B output: per-WARC ModernBERT prob for every pre-survivor (Phase C filters by threshold).
KEEPLIST_SCHEMA = pa.schema([("doc_id", pa.string()), ("modernbert_prob", pa.float32())])

# Length buckets for static-shape TPU batching: pad each micro-batch up to the smallest
# bucket >= its longest member, so XLA compiles ~one program per bucket (not per length).
BUCKETS: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192)


def bucket_for(n_tokens: int, max_length: int = 8192) -> int:
    """Smallest bucket >= n_tokens, capped at ``max_length``."""
    for b in BUCKETS:
        if b >= n_tokens and b <= max_length:
            return b
    return max_length


def survivors_to_table(rows: Sequence[dict]) -> pa.Table:
    """Build a survivor :class:`pyarrow.Table` from row dicts (validated against the schema)."""
    cols = {
        "doc_id": [r["doc_id"] for r in rows],
        "url": [r["url"] for r in rows],
        "warc_hash": [r["warc_hash"] for r in rows],
        "snapshot": [r["snapshot"] for r in rows],
        "fasttext_score": [r["fasttext_score"] for r in rows],
        "text": [r["text"] for r in rows],
        "input_ids": [r["input_ids"] for r in rows],
        "n_tokens": [r["n_tokens"] for r in rows],
    }
    return pa.table(cols, schema=SURVIVOR_SCHEMA)


def write_table(path: str, table: pa.Table) -> None:
    """Write a parquet table to a gs:// (or local) path with zstd compression."""
    with fsspec.open(path, "wb") as fh:
        pq.write_table(table, fh, compression=PARQUET_COMPRESSION)


def write_survivors(path: str, rows: Sequence[dict]) -> None:
    """Write survivor rows (or an empty, schema-typed parquet for a 0-survivor WARC, so the
    file's existence still marks the WARC as CPU-processed for resume)."""
    table = SURVIVOR_SCHEMA.empty_table() if not rows else survivors_to_table(rows)
    write_table(path, table)


def write_presurvivors(path: str, rows: Sequence[dict]) -> None:
    """v2 Phase A: write fastText-survivors carrying raw html (no JustText yet)."""
    if not rows:
        write_table(path, PRESURVIVOR_SCHEMA.empty_table())
        return
    cols = {name: [r[name] for r in rows] for name in PRESURVIVOR_SCHEMA.names}
    write_table(path, pa.table(cols, schema=PRESURVIVOR_SCHEMA))


def write_keeplist(path: str, doc_ids: Sequence[str], probs: Sequence[float]) -> None:
    """v2 Phase B: write per-WARC {doc_id, modernbert_prob} for every pre-survivor."""
    table = pa.table({"doc_id": list(doc_ids), "modernbert_prob": list(probs)}, schema=KEEPLIST_SCHEMA)
    write_table(path, table)


def read_table(path: str, columns: list[str] | None = None) -> pa.Table:
    """Read a parquet file (optionally a subset of columns) from gs:// or local."""
    with fsspec.open(path, "rb") as fh:
        return pq.read_table(fh, columns=columns)


def write_kept(path: str, table: pa.Table) -> None:
    """Write the kept-doc parquet (survivor columns + ``modernbert_prob``)."""
    with fsspec.open(path, "wb") as fh:
        pq.write_table(table.cast(KEPT_SCHEMA), fh, compression=PARQUET_COMPRESSION)


def write_tombstones(path: str, rows: Iterable[tuple[str, float]]) -> None:
    """Write ``{doc_id, modernbert_prob}`` for dropped survivors as gzipped JSONL."""
    with fsspec.open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb") as gz:
        for doc_id, prob in rows:
            line = json.dumps({"doc_id": doc_id, "modernbert_prob": round(float(prob), 6)})
            gz.write((line + "\n").encode("utf-8"))


def pad_batch(
    id_lists: Sequence[Sequence[int]],
    target_len: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad ragged token-id lists to a static ``[B, target_len]`` batch + segment ids.

    Returns ``(input_ids, seg_ids)`` both ``int32[B, target_len]``:
      * ``input_ids`` — real tokens then ``pad_token_id`` fill (each list truncated to
        ``target_len`` defensively, though survivors are already <= max_length).
      * ``seg_ids`` — ``0`` at real token positions, ``-1`` at pads. The ModernBERT
        attention mask is ``AttentionMask(is_causal=False).with_segment_ids(seg, seg)``,
        which excludes pads from attention.
    """
    b = len(id_lists)
    input_ids = np.full((b, target_len), pad_token_id, dtype=np.int32)
    seg_ids = np.full((b, target_len), -1, dtype=np.int32)
    for i, ids in enumerate(id_lists):
        n = min(len(ids), target_len)
        if n:
            input_ids[i, :n] = np.asarray(ids[:n], dtype=np.int32)
            seg_ids[i, :n] = 0
    return input_ids, seg_ids
