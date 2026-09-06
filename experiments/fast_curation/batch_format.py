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
import math
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
# With a pooled pre-filter the corpus carries that stage's score too, so every filter's score travels
# with the document (fastText / pooled / ModernBERT) and any of them can be studied or re-thresholded
# offline without a join back to b_keeplist. Conditional on the spec, NOT unconditional: KEPT_SCHEMA is
# shared with the live v1-v3 line (~3,602 WARCs written) and Phase C `pa.concat_tables`, so widening it
# for every spec would mix schemas within an existing corpus.
KEPT_SCHEMA_POOLED = pa.schema([*KEPT_SCHEMA, pa.field("pooled_prob", pa.float32())])

# TEXT-line final corpus (written by Phase B — there is no Phase C): the text-presurvivor columns
# plus BOTH stage scores. ``modernbert_prob`` is NaN for docs the pooled band accepted outright
# (the terminal model never scored them — that skip is the early exit's whole saving).
KEPT_SCHEMA_TEXT = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("fasttext_score", pa.float32()),
        ("text", pa.large_string()),  # resiliparse-rs extraction — the training content.
        ("input_ids", pa.list_(pa.int32())),
        ("n_tokens", pa.int32()),
        ("pooled_prob", pa.float32()),
        ("modernbert_prob", pa.float32()),  # NaN = hi-accepted, never scored by the terminal model.
    ]
)


def kept_schema_for(spec) -> pa.Schema:
    """The final-corpus schema for ``spec`` — widened with ``pooled_prob`` iff it has a pooled stage."""
    if spec.is_text_line:
        # storage_version 4 (fused single-phase) shares the v3 output contract exactly.
        return KEPT_V3_SCHEMA if spec.storage_version in (3, 4) else KEPT_SCHEMA_TEXT
    return KEPT_SCHEMA_POOLED if spec.pooled_ckpt else KEPT_SCHEMA


# --- V3 (storage_version=3, the 8M-scale contract) ---
# No input_ids anywhere: they were 42% of every stored byte, and Phase B re-tokenizes from ``text``
# via the gigatoken arrow path for ~4% of its wall. Text-bearing parquets are written at zstd
# level 12 (measured -21% on text for ~5s/WARC of CPU).
V3_TEXT_COMPRESSION_LEVEL = 12
PRESURVIVOR_V3_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("fasttext_score", pa.float32()),
        ("text", pa.large_string()),
    ]
)
# Phase B appends its scores plus ``n_tokens`` (computed at scoring time — A no longer tokenizes).
KEPT_V3_SCHEMA = pa.schema(
    [
        *PRESURVIVOR_V3_SCHEMA,
        pa.field("n_tokens", pa.int32()),
        pa.field("pooled_prob", pa.float32()),
        pa.field("modernbert_prob", pa.float32()),  # NaN = pooled hi-accept, never terminal-scored.
    ]
)


def write_table_v3(path: str, table: pa.Table) -> None:
    """Parquet write at the V3 text compression level."""
    with fsspec.open(path, "wb") as fh:
        pq.write_table(table, fh, compression="zstd", compression_level=V3_TEXT_COMPRESSION_LEVEL)


def write_presurvivors_v3(path: str, rows: Sequence[dict]) -> None:
    """V3 Phase A: fastText-survivors carrying the extracted ``text`` only (no token columns)."""
    if not rows:
        write_table_v3(path, PRESURVIVOR_V3_SCHEMA.empty_table())
        return
    cols = {name: [r[name] for r in rows] for name in PRESURVIVOR_V3_SCHEMA.names}
    write_table_v3(path, pa.table(cols, schema=PRESURVIVOR_V3_SCHEMA))


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
# TEXT-line Phase A output: extraction already ran, so the survivor carries the final ``text``
# (not raw html) and the classifier-input tokens. Phase B reads this and writes ``kept/`` directly.
PRESURVIVOR_TEXT_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("warc_hash", pa.string()),
        ("snapshot", pa.string()),
        ("fasttext_score", pa.float32()),
        ("text", pa.large_string()),  # resiliparse-rs extraction — the training content.
        ("input_ids", pa.list_(pa.int32())),
        ("n_tokens", pa.int32()),
    ]
)

# Phase B output: per-WARC ModernBERT prob for every pre-survivor (Phase C filters by threshold).
KEEPLIST_SCHEMA = pa.schema([("doc_id", pa.string()), ("modernbert_prob", pa.float32())])
# Same, for a cascade with a pooled pre-filter. A doc pooled dropped is never scored by ModernBERT,
# so its ``modernbert_prob`` is NaN — which Phase C's ``prob >= threshold`` test excludes for free
# (NaN compares False), with no special-casing. ``pooled_prob`` is retained for diagnostics.
KEEPLIST_POOLED_SCHEMA = pa.schema(
    [("doc_id", pa.string()), ("modernbert_prob", pa.float32()), ("pooled_prob", pa.float32())]
)

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


def write_presurvivors_text(path: str, rows: Sequence[dict]) -> None:
    """TEXT-line Phase A: write fastText-survivors carrying the extracted ``text``."""
    if not rows:
        write_table(path, PRESURVIVOR_TEXT_SCHEMA.empty_table())
        return
    cols = {name: [r[name] for r in rows] for name in PRESURVIVOR_TEXT_SCHEMA.names}
    write_table(path, pa.table(cols, schema=PRESURVIVOR_TEXT_SCHEMA))


def write_presurvivors_text_columns(path: str, rows: Sequence[dict], input_ids: pa.Array, n_tokens) -> None:
    """TEXT-line Phase A, columnar: scalar fields from ``rows``; token columns passed as arrays.

    The tokenize seam produces ``input_ids`` as a ready ``list<int32>`` arrow column (the gigatoken
    path never materializes per-doc Python lists), so the write takes it as-is instead of routing
    ids through row dicts.
    """
    if not rows:
        write_table(path, PRESURVIVOR_TEXT_SCHEMA.empty_table())
        return
    scalar_names = [n for n in PRESURVIVOR_TEXT_SCHEMA.names if n not in ("input_ids", "n_tokens")]
    cols: dict = {name: [r[name] for r in rows] for name in scalar_names}
    cols["input_ids"] = input_ids
    cols["n_tokens"] = pa.array(n_tokens, type=pa.int32())
    write_table(path, pa.table(cols, schema=PRESURVIVOR_TEXT_SCHEMA))


def truncate_ids(ids: Sequence[int], target_len: int, sep_token_id: int) -> list[int]:
    """Ids tokenized at a longer max_length -> the EXACT ids ``tokenize(..., max_length=target_len)``
    would yield, without re-tokenizing.

    Right truncation keeps ``[CLS]`` + the first ``target_len - 2`` body tokens + ``[SEP]``, so
    slicing the longer sequence to ``target_len - 1`` and re-appending ``[SEP]`` reproduces the
    shorter tokenization exactly (asserted against the real tokenizer in ``test_preprocess``).
    A sequence already within ``target_len`` is returned unchanged.
    """
    if len(ids) <= target_len:
        return list(ids)
    return [*ids[: target_len - 1], sep_token_id]


def write_keeplist(
    path: str,
    doc_ids: Sequence[str],
    probs: Sequence[float],
    pooled_probs: Sequence[float] | None = None,
) -> None:
    """v2 Phase B: write per-WARC {doc_id, modernbert_prob} for every pre-survivor.

    ``pooled_probs`` (lpv11 line) adds the pooled pre-filter's score; entries pooled dropped carry a
    NaN ``modernbert_prob`` because ModernBERT never ran on them.
    """
    cols: dict[str, list] = {"doc_id": list(doc_ids), "modernbert_prob": list(probs)}
    schema = KEEPLIST_SCHEMA
    if pooled_probs is not None:
        cols["pooled_prob"] = list(pooled_probs)
        schema = KEEPLIST_POOLED_SCHEMA
    write_table(path, pa.table(cols, schema=schema))


def drop_chunk_dir(chunk_dir: str) -> int:
    """Delete a WARC's sub-WARC checkpoint dir once its merged output is written. Returns files removed.

    Chunks exist only so a preempted worker resumes mid-WARC; once the merged parquet is durable they
    are pure waste — at 10k WARCs they project to ~4 TiB, roughly 6x the final corpus. Deleting here
    (rather than in a reaper) is race-free: the caller holds the WARC's claim and has just written the
    file the chunks were building, so nothing else can be reading or resuming them.

    Best-effort: a failure to clean up must never fail a WARC that is otherwise complete.
    """
    fs = fsspec.filesystem("gcs")
    path = chunk_dir.replace("gs://", "")
    try:
        if not fs.exists(path):
            return 0
        files = [f for f in fs.find(path)]
        fs.rm(path, recursive=True)
        return len(files)
    except Exception:
        return 0


def read_table(path: str, columns: list[str] | None = None) -> pa.Table:
    """Read a parquet file (optionally a subset of columns) from gs:// or local."""
    with fsspec.open(path, "rb") as fh:
        return pq.read_table(fh, columns=columns)


def write_kept(path: str, table: pa.Table, schema: pa.Schema | None = None) -> None:
    """Write the kept-doc parquet (survivor columns + ``modernbert_prob``, and ``pooled_prob`` when
    the spec has a pooled stage — pass ``kept_schema_for(spec)``).

    The cast is what enforces the on-disk contract, so it must target the table's OWN schema: casting
    a pooled table to the narrow ``KEPT_SCHEMA`` would silently drop ``pooled_prob``.
    """
    with fsspec.open(path, "wb") as fh:
        pq.write_table(table.cast(schema or KEPT_SCHEMA), fh, compression=PARQUET_COMPRESSION)


def write_tombstones(path: str, rows: Iterable[tuple[str, float]]) -> None:
    """Write ``{doc_id, modernbert_prob}`` for dropped survivors as gzipped JSONL."""
    with fsspec.open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb") as gz:
        for doc_id, prob in rows:
            line = json.dumps({"doc_id": doc_id, "modernbert_prob": round(float(prob), 6)})
            gz.write((line + "\n").encode("utf-8"))


def write_tombstones_band(path: str, rows: Iterable[tuple[str, float, float]]) -> None:
    """TEXT-line tombstones: ``{doc_id, pooled_prob, modernbert_prob}`` per dropped survivor.

    ``modernbert_prob`` is None for docs the band dropped below ``lo`` (the terminal model never
    scored them); band docs the terminal model rejected carry both probs, so re-thresholding the
    terminal model downward stays a free offline re-filter.
    """
    with fsspec.open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb") as gz:
        for doc_id, pooled_prob, mb_prob in rows:
            line = json.dumps(
                {
                    "doc_id": doc_id,
                    "pooled_prob": round(float(pooled_prob), 6),
                    "modernbert_prob": None if math.isnan(mb_prob) else round(float(mb_prob), 6),
                }
            )
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
