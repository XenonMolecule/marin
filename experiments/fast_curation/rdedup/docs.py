# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-document normalization and MinHash, bit-identical to the Zephyr dedup path.

``experiments/fast_curation/dedup.py`` reshapes kept_text to JSONL and hands it to
``marin.datakit.normalize`` and ``fuzzy_minhash``. Along that path a document:

1. is dropped if its text is missing or empty (reshape), or blank after ``str.strip`` (normalize);
2. has every whitespace run longer than 128 characters cut to its first 128 characters;
3. gets ``id = xxh3_128(text)`` as 32 hex digits -- the exact-dedup key;
4. gets 26 LSH buckets from dupekit ``CleanText -> MinHash(286, ngram 5, seed 42) ->
   MinHashLSH(26)`` over the full compacted text (``text_cap_chars=None``);
5. enters connected components as the node ``"source_000|" + id``, ordered by
   ``str(xxh3_128(node))`` -- the canonical of a component is its minimum under that string
   order.

This module reproduces each step with the same library calls so signatures, ids and canonical
choices match the existing trees.
"""

from __future__ import annotations

import dataclasses
import io
import math
import re

import dupekit
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fast_curation.project_text_only import DROP_COLUMNS
from experiments.fast_curation.rdedup.store import ObjectStore

NUM_PERMS = 286
NUM_BANDS = 26
NGRAM_SIZE = 5
SEED = 42
MAX_WHITESPACE_RUN_CHARS = 128
# ``fuzzy_dups`` tags every node with its co-partitioned source; a single-source run is source_000.
CC_SOURCE_TAG = "source_000"
ROW_BITS = 24
MAX_ROWS_PER_FILE = 1 << ROW_BITS

_LONG_WHITESPACE = re.compile(r"\s{" + str(MAX_WHITESPACE_RUN_CHARS + 1) + r",}")
_MINHASH_PIPELINE = [
    dupekit.Transformation.CleanText(input_col="text", output_col="clean_text"),
    dupekit.Transformation.MinHash(
        input_col="clean_text", output_col="signature", num_perms=NUM_PERMS, ngram_size=NGRAM_SIZE, seed=SEED
    ),
    dupekit.Transformation.MinHashLSH(input_col="signature", output_col="buckets", num_bands=NUM_BANDS),
    dupekit.Transformation.SelectColumns(columns=["buckets"]),
]
_JSON = msgspec.json.Encoder()


def compact_whitespace(text: str) -> str:
    """Cut whitespace runs longer than 128 characters to their first 128 characters."""
    if _LONG_WHITESPACE.search(text) is None:
        return text
    return _LONG_WHITESPACE.sub(lambda m: m.group(0)[:MAX_WHITESPACE_RUN_CHARS], text)


def is_kept_text(text: str | None) -> bool:
    """Reshape skips falsy text; normalize then skips text that is blank after ``str.strip``."""
    return bool(text) and bool(text.strip())


def content_id(text: str) -> int:
    """The 128-bit integer behind normalize's 32-hex-digit ``id``."""
    return dupekit.hash_xxh3_128(text.encode("utf-8"))


def id_hex(hi: int, lo: int) -> str:
    return format((int(hi) << 64) | int(lo), "032x")


def doc_keys(file_index: int, rows: np.ndarray) -> np.ndarray:
    """Globally unique, file-ordered document keys: ``file_index << 24 | row``."""
    return (np.uint64(file_index) << np.uint64(ROW_BITS)) | rows.astype(np.uint64)


def key_file(keys: np.ndarray) -> np.ndarray:
    return keys >> np.uint64(ROW_BITS)


def key_row(keys: np.ndarray) -> np.ndarray:
    return keys & np.uint64(MAX_ROWS_PER_FILE - 1)


# The canonical order is the lexicographic order of decimal strings of 128-bit hashes. Padding a
# string on the right with a character below '0' and reading it in base 11 preserves that order;
# 39 digits split into three 13-digit words keep each word below 11**13 < 2**64.
_ORDER_DIGITS = 39
_ORDER_WORD_DIGITS = 13
_ORDER_POWERS = np.array([11**p for p in range(_ORDER_WORD_DIGITS - 1, -1, -1)], dtype=np.uint64)


def order_words(digit_strings: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three uint64 words per decimal string whose tuple order is the strings' lexicographic order."""
    n = len(digit_strings)
    if n == 0:
        empty = np.zeros(0, dtype=np.uint64)
        return empty, empty, empty
    # '/' is the byte just below '0': it maps to digit 0 and the digits '0'-'9' map to 1-10.
    padded = "".join(s.ljust(_ORDER_DIGITS, "/") for s in digit_strings).encode()
    digits = (np.frombuffer(padded, dtype=np.uint8).reshape(n, _ORDER_DIGITS) - ord("/")).astype(np.uint64)
    w = _ORDER_WORD_DIGITS
    return digits[:, 0:w] @ _ORDER_POWERS, digits[:, w : 2 * w] @ _ORDER_POWERS, digits[:, 2 * w : 3 * w] @ _ORDER_POWERS


def id_norm(hi: int, lo: int) -> str:
    """``connected_components``' ordering string for this document's single-source node."""
    return str(dupekit.hash_xxh3_128(f"{CC_SOURCE_TAG}|{id_hex(hi, lo)}".encode()))


def canonical_order_keys(hi: np.ndarray, lo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order words for many docs: the canonical of a component has the smallest."""
    return order_words([id_norm(h, l) for h, l in zip(hi.tolist(), lo.tolist(), strict=True)])


@dataclasses.dataclass
class FileDocs:
    """What pass 1 needs from one kept file, for its kept rows only."""

    file_index: int
    n_rows: int
    rows: np.ndarray  # int64 row index of each kept document
    id_hi: np.ndarray  # uint64
    id_lo: np.ndarray  # uint64
    chars: np.ndarray  # int64 length of the compacted text
    has_signature: np.ndarray  # bool: dupekit produced buckets
    buckets: np.ndarray  # (n_kept, 26) uint64; zeros where has_signature is False
    compacted: int


def read_kept(store: ObjectStore, path: str, columns: list[str] | None = None) -> pa.Table:
    """One ``kept/`` parquet file without the ModernBERT token columns (``DROP_COLUMNS``), so the
    output records hold the same fields the old ``kept_text/`` projection carried."""
    table = pq.read_table(io.BytesIO(store.read(path)), columns=columns)
    return table.drop_columns([name for name in DROP_COLUMNS if name in table.column_names])


def minhash_buckets(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(has_signature, buckets)`` for ``texts`` from the reference dupekit pipeline."""
    n = len(texts)
    buckets = np.zeros((n, NUM_BANDS), dtype=np.uint64)
    if n == 0:
        return np.zeros(0, dtype=bool), buckets
    result = dupekit.transform(pa.record_batch({"text": pa.array(texts, type=pa.large_string())}), _MINHASH_PIPELINE)
    column = result.column("buckets")
    valid = np.asarray(column.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    if valid.any():
        flat = column.filter(pa.array(valid)).flatten().to_numpy(zero_copy_only=False)
        if len(flat) != int(valid.sum()) * NUM_BANDS:
            raise ValueError(f"dupekit returned {len(flat)} buckets for {int(valid.sum())} signatures")
        buckets[valid] = flat.astype(np.uint64).reshape(-1, NUM_BANDS)
    return valid, buckets


def normalize_file(texts: list[str | None]) -> tuple[np.ndarray, list[str], int]:
    """Kept row indices, their compacted texts, and how many texts compaction changed."""
    rows = []
    compacted_texts = []
    compacted = 0
    for i, text in enumerate(texts):
        if not is_kept_text(text):
            continue
        normalized = compact_whitespace(text)
        if len(normalized) != len(text):
            compacted += 1
        rows.append(i)
        compacted_texts.append(normalized)
    return np.asarray(rows, dtype=np.int64), compacted_texts, compacted


def pass1_file(store: ObjectStore, path: str, file_index: int) -> FileDocs:
    """Normalize, hash and MinHash one kept file."""
    texts = read_kept(store, path, ["text"]).column("text").to_pylist()
    if len(texts) >= MAX_ROWS_PER_FILE:
        raise ValueError(f"{path} has {len(texts)} rows; doc keys hold at most {MAX_ROWS_PER_FILE}")
    rows, normalized, compacted = normalize_file(texts)
    ids = [content_id(t) for t in normalized]
    id_hi = np.fromiter((i >> 64 for i in ids), dtype=np.uint64, count=len(ids))
    id_lo = np.fromiter((i & 0xFFFFFFFFFFFFFFFF for i in ids), dtype=np.uint64, count=len(ids))
    chars = np.fromiter((len(t) for t in normalized), dtype=np.int64, count=len(normalized))
    has_signature, buckets = minhash_buckets(normalized)
    return FileDocs(file_index, len(texts), rows, id_hi, id_lo, chars, has_signature, buckets, compacted)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def output_records(table: pa.Table, rows: np.ndarray) -> tuple[list[dict], list[str]]:
    """Output dicts for ``rows`` of a kept table, in the reference's column order and types.

    The reference tree's rows are the kept columns (minus ``DROP_COLUMNS``) in file order with the compacted text in
    place of the original, ``modernbert_prob`` normalized to ``float | None``, and every other
    NaN float written as JSON ``null`` by the encoder. Returns the records and their texts.
    """
    taken = table.take(pa.array(rows, type=pa.int64()))
    records = taken.to_pylist()
    texts = []
    for record in records:
        record["text"] = compact_whitespace(record["text"])
        if "modernbert_prob" in record:
            record["modernbert_prob"] = _finite_or_none(record["modernbert_prob"])
        texts.append(record["text"])
    return records, texts


def encode_jsonl(records: list[dict]) -> bytes:
    """JSONL bytes as the Zephyr writer produces them (msgspec; NaN encodes as null)."""
    return b"".join(_JSON.encode(r) + b"\n" for r in records)
