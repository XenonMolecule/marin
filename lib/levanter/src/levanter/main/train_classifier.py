# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a ModernBERT sequence classifier in Levanter (JAX/TPU).

Mirrors ``train_dpo.py``: a draccus ``main(config)`` entrypoint that flows through the standard
``Trainer`` (native multi-host/FSDP, Tensorstore checkpointing + preemption auto-resume, and the
TPU splash kernel). The data is the fastText line format -- one ``__label__<name> <text>`` per
line, gzipped and sharded -- which is generic enough to live here; ``label = 1`` iff the label
token equals ``useful_label`` (default ``__label__useful``), else 0.

Marin launches this via ``marin.training.training.run_levanter_train_classifier``.
"""

import gzip
import hashlib
import json
import logging
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import fsspec
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from transformers import AutoTokenizer

import haliax as hax
import levanter
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning
from levanter.data._preprocessor import BatchProcessor
from levanter.data.dataset import AsyncDataset
from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.checkpoint import discover_latest_checkpoint
from levanter.layers.attention import AttentionMask
from levanter.store.cache import CacheOptions, TreeCache, build_or_load_cache
from levanter.models.classification import ClassificationExample, build_classifier, save_classifier
from levanter.models.lm_model import LmConfig
from levanter.models.modernbert import ModernBertConfig
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def _parse_line(line: str, useful_label: str) -> tuple[int, str] | None:
    line = line.rstrip("\n")
    if not line or " " not in line:
        return None
    label_tok, text = line.split(" ", 1)
    if not label_tok.startswith("__label__") or not text:
        return None
    return (1 if label_tok == useful_label else 0, text)


def _expand_globs(globs: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for g in globs:
        fs = fsspec.core.url_to_fs(g)[0]
        proto = g.split("://", 1)[0] if "://" in g else ""
        for p in sorted(fs.glob(g)):
            paths.append(f"{proto}://{p}" if proto else p)
    return paths


def read_fasttext_shards(
    paths: Sequence[str], useful_label: str, limit: int | None = None
) -> tuple[list[str], list[int]]:
    """Read (texts, labels) from gzipped fastText-format shards in order, up to ``limit`` rows."""
    texts: list[str] = []
    labels: list[int] = []
    for path in paths:
        with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = _parse_line(line, useful_label)
                if parsed is None:
                    continue
                labels.append(parsed[0])
                texts.append(parsed[1])
                if limit is not None and len(texts) >= limit:
                    return texts, labels
    return texts, labels


def read_frozen_eval(paths: Sequence[str], useful_label: str, rows: int, seed: int = 0) -> tuple[list[str], list[int]]:
    """Snapshot-stratified sample: ``rows // n_shards`` per shard after a deterministic shuffle."""
    per = max(1, rows // max(1, len(paths)))
    texts: list[str] = []
    labels: list[int] = []
    for path in paths:
        s_texts, s_labels = read_fasttext_shards([path], useful_label)
        idx = list(range(len(s_texts)))
        random.Random(seed).shuffle(idx)
        for j in idx[:per]:
            texts.append(s_texts[j])
            labels.append(s_labels[j])
    return texts, labels


class ClassificationLineProcessor(BatchProcessor[str, dict]):
    """Parse + tokenize fastText lines into a streaming TreeCache (the same on-disk, index-addressable
    cache the LM path uses for TB-scale corpora).

    Input: raw ``__label__<name> <text>`` lines (from ``TextUrlDataSource``).
    Output (one dict per non-blank line): ``{"input_ids": int32[≤max_length], "label": int32}``.
    ``input_ids`` are variable-length (truncated, NOT padded), so a SINGLE cache is context-length
    agnostic: every ctx in the sweep (1024..8192) reads the same cache and truncates/pads at load.
    """

    def __init__(self, tokenizer, useful_label: str, max_length: int = 8192):
        self.tokenizer = tokenizer
        self.useful_label = useful_label
        self.max_length = max_length

    def __call__(self, batch: Sequence[str]) -> list[dict]:
        out: list[dict] = []
        for line in batch:
            parsed = _parse_line(line, self.useful_label)
            if parsed is None:
                continue
            label, text = parsed
            # Cap chars before tokenizing: 100k chars >> 8192 tokens (≈4-6 chars/token → ~12x margin), so the first
            # max_length tokens — and thus the stored output — are IDENTICAL to no-cap, but we skip
            # tokenizing pathological near-1MB doc tails that otherwise stall the cache build. Metadata
            # is deliberately unchanged so existing committed shards still resume (no full rebuild).
            text = text[:100_000]
            ids = self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"]
            out.append({"input_ids": np.asarray(ids, dtype=np.int32), "label": np.int32(label)})
        return out

    @property
    def output_exemplar(self) -> dict:
        return {"input_ids": np.zeros((0,), dtype=np.int32), "label": np.zeros((), dtype=np.int32)}

    @property
    def num_cpus(self) -> int:
        return 1

    @property
    def metadata(self) -> dict:
        return {
            "tokenizer": str(getattr(self.tokenizer, "name_or_path", "")),
            "vocab_size": len(self.tokenizer),
            "useful_label": self.useful_label,
            "max_length": self.max_length,
        }


class CachedClassificationDataset(AsyncDataset[ClassificationExample]):
    """Streams (input_ids, label) from a tokenized ``TreeCache`` — memory-bounded, reading by index
    from disk (no in-RAM corpus), exactly like LM training. Pads/truncates each doc to ``Pos`` at
    load and builds the pad-exclusion segment mask (real = segment 0, pad = -1). ``limit`` caps the
    usable length (``--train-rows``) without materializing anything.
    """

    def __init__(self, cache: TreeCache, Pos: Axis, pad_token_id: int, limit: Optional[int] = None):
        super().__init__()
        self.cache = cache
        self.Pos = Pos
        self.pad_token_id = pad_token_id
        self.limit = limit

    async def async_len(self) -> int:
        n = await self.cache.async_len()
        return min(n, self.limit) if self.limit is not None else n

    def is_finite(self) -> bool:
        return True

    def _encode_one(self, ids: np.ndarray, label: int) -> ClassificationExample:
        seq_len = self.Pos.size
        ids = np.asarray(ids, dtype=np.int32)[:seq_len]
        n = len(ids)
        padded = np.full((seq_len,), self.pad_token_id, dtype=np.int32)
        padded[:n] = ids
        seg = np.full((seq_len,), -1, dtype=np.int32)
        seg[:n] = 0
        seg_named = hax.named(seg, self.Pos)
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        return ClassificationExample.init(
            tokens=hax.named(padded, self.Pos),
            label=hax.named(np.int32(label), ()),
            attn_mask=mask,
        )

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ClassificationExample]:
        rows = await self.cache.get_batch(indices)
        return [self._encode_one(r["input_ids"], int(r["label"])) for r in rows]


class TextClassificationDataset(AsyncDataset[ClassificationExample]):
    """In-memory (text, label) corpus tokenized lazily per batch to fixed-length ``Pos``. Simple and
    fast-to-start; fine for small data (<=~1M rows). For larger sets (5M/10M) it OOMs host RAM — use
    the cache-backed path (``CachedClassificationDataset``) instead. Pad positions excluded from
    attention via a segment mask (real = segment 0, pad = -1).
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, Pos: Axis, pad_token_id: int):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.Pos = Pos
        self.pad_token_id = pad_token_id

    async def async_len(self) -> int:
        return len(self.texts)

    def is_finite(self) -> bool:
        return True

    def _encode_one(self, text: str, label: int) -> ClassificationExample:
        seq_len = self.Pos.size
        ids = self.tokenizer(text, truncation=True, max_length=seq_len)["input_ids"]
        n = len(ids)
        padded = np.full((seq_len,), self.pad_token_id, dtype=np.int32)
        padded[:n] = np.asarray(ids, dtype=np.int32)
        seg = np.full((seq_len,), -1, dtype=np.int32)
        seg[:n] = 0
        seg_named = hax.named(seg, self.Pos)
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        return ClassificationExample.init(
            tokens=hax.named(padded, self.Pos),
            label=hax.named(np.int32(label), ()),
            attn_mask=mask,
        )

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ClassificationExample]:
        return [self._encode_one(self.texts[i], self.labels[i]) for i in indices]


def chunk_starts(n_tokens: int, stride: int) -> list[int]:
    """Start offsets of the ``Pos``-sized windows tiling a doc of ``n_tokens`` tokens at ``stride``.

    ``stride == Pos.size`` → non-overlapping tiles; ``Pos.size // 2`` → 50% overlap. Always yields at
    least one start (``[0]``) so an empty/short doc still produces a single (padded) chunk.
    """
    return list(range(0, max(1, n_tokens), stride))


def _token_cache_root(
    cache_dir: Optional[str],
    paths: Sequence[str],
    useful_label: str,
    limit: Optional[int],
    tokenizer_name: str,
    max_doc_tokens: int,
) -> str:
    """Deterministic cache location for the tokenized docs. Independent of ctx/stride/seed, so ONE
    cache serves every chunk size, overlap, and model (base/large share the tokenizer)."""
    key = json.dumps(
        {
            "urls": list(paths),
            "label": useful_label,
            "limit": limit,
            "tokenizer": tokenizer_name,
            "max_doc_tokens": max_doc_tokens,
        },
        sort_keys=True,
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    base = cache_dir.rstrip("/") if cache_dir else paths[0].rsplit("/", 1)[0]
    return f"{base}/_chunk_token_cache/rows{limit}_max{max_doc_tokens}_{h}"


def _save_token_cache(root: str, doc_ids: list[np.ndarray], labels: list[int]) -> None:
    """Persist the tokenized docs as flat arrays (concatenated ids + offsets + labels) + a _DONE marker
    written last, so a partially-written cache is never mistaken for complete."""
    ids = np.concatenate(doc_ids) if doc_ids else np.zeros((0,), dtype=np.int32)
    offsets = np.zeros((len(doc_ids) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum([len(d) for d in doc_ids], dtype=np.int64)
    for name, arr in (("ids", ids.astype(np.int32)), ("offsets", offsets), ("labels", np.asarray(labels, np.int32))):
        with fsspec.open(f"{root}/{name}.npy", "wb") as f:
            np.save(f, arr)
    with fsspec.open(f"{root}/_DONE", "w") as f:
        f.write("ok")


def _load_token_cache(root: str) -> tuple[list[np.ndarray], list[int]]:
    def _load(name: str) -> np.ndarray:
        with fsspec.open(f"{root}/{name}.npy", "rb") as f:
            return np.load(f)

    ids, offsets, labels = _load("ids"), _load("offsets"), _load("labels")
    doc_ids = [ids[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    return doc_ids, [int(x) for x in labels]


def _tokenize_slice(payload: tuple[str, list[str], int]) -> list[np.ndarray]:
    """Worker entrypoint: tokenize a contiguous slice of docs in a fresh process. Top-level (picklable)
    so ProcessPoolExecutor can fan it out — N processes each tokenize single-threaded → ~N× throughput,
    sidestepping the post-fork HF parallelism disable that makes a single process ~1 doc-batch/core."""
    tokenizer_name, texts, max_doc_tokens = payload
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    enc = tok(texts, truncation=True, max_length=max_doc_tokens)["input_ids"]
    return [np.asarray(e, dtype=np.int32) for e in enc]


def _tokenize_docs(texts: list[str], tokenizer_name: str, max_doc_tokens: int) -> list[np.ndarray]:
    """Tokenize all docs, fanning out across CPU cores (order preserved)."""
    n_workers = min(os.cpu_count() or 1, 16)
    if n_workers <= 1 or len(texts) < 4000:
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        return [
            np.asarray(e, dtype=np.int32) for e in tok(texts, truncation=True, max_length=max_doc_tokens)["input_ids"]
        ]
    size = math.ceil(len(texts) / n_workers)
    payloads = [(tokenizer_name, texts[i : i + size], max_doc_tokens) for i in range(0, len(texts), size)]
    doc_ids: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for chunk in ex.map(_tokenize_slice, payloads):  # map preserves submission order
            doc_ids.extend(chunk)
    return doc_ids


def build_or_load_token_cache(
    paths: Sequence[str],
    useful_label: str,
    limit: Optional[int],
    tokenizer_name: str,
    max_doc_tokens: int,
    cache_dir: Optional[str],
) -> tuple[list[np.ndarray], list[int]]:
    """Load the tokenized docs from the shared GCS cache, building it once if absent.

    Tokenizing 200k+ docs single-threaded on a worker (HF disables tokenizer parallelism after the
    iris/JAX fork) takes ~15-60 min and repeats on every preemption AND for every run — so we tokenize
    ONCE (multi-process, see ``_tokenize_docs``) and every run/resume loads the flat arrays in seconds.
    The cache key excludes ctx/stride/seed, so all chunk sizes + base/large share one cache.

    Scale note: the flat in-RAM arrays suit the ≤~1M pilot. 10M needs the streaming TreeCache variant
    (``CachedClassificationDataset`` with the cap raised to ``max_doc_tokens``) — a memory-bounded
    follow-up; the tokenize-once / ctx-agnostic design carries over, only the storage changes.
    """
    root = _token_cache_root(cache_dir, paths, useful_label, limit, tokenizer_name, max_doc_tokens)
    fs = fsspec.core.url_to_fs(f"{root}/_DONE")[0]
    if fs.exists(f"{root}/_DONE"):
        logger.info(f"loading tokenized doc cache: {root}")
        doc_ids, labels = _load_token_cache(root)
        logger.info(f"loaded {len(doc_ids)} tokenized docs from cache")
        return doc_ids, labels
    logger.info(f"tokenized doc cache MISS at {root}; reading + tokenizing once ({os.cpu_count()} cores)")
    texts, labels = read_fasttext_shards(paths, useful_label, limit=limit)
    doc_ids = _tokenize_docs(texts, tokenizer_name, max_doc_tokens)
    _save_token_cache(root, doc_ids, labels)
    logger.info(f"tokenized {len(doc_ids)} docs -> cache {root}")
    return doc_ids, labels


class ChunkedTextClassificationDataset(AsyncDataset[ClassificationExample]):
    """In-memory multiple-instance dataset: each document is split into ``Pos``-sized token chunks and
    EACH chunk becomes one training example carrying the document's label (plain MIL — a boilerplate
    chunk of a useful doc is still labeled useful). The model learns ``P(this chunk looks like it's
    from a useful doc)``; aggregation over a doc's chunks is deferred to eval.

    Takes already-tokenized ``doc_ids`` (full int32 token sequence per doc, from
    ``build_or_load_token_cache``). Chunk starts are ``stride`` apart (``stride == Pos.size`` →
    non-overlap, ``Pos.size // 2`` → 50% overlap). A document with more than ``max_chunks`` chunks
    contributes a deterministic per-doc random subset of ``max_chunks`` of them (seeded), so long docs
    don't dominate. (Inference uses the FULL doc — all chunks, no cap — via ``score_texts_chunked``.)
    """

    def __init__(
        self,
        doc_ids: list[np.ndarray],
        labels: list[int],
        Pos: Axis,
        pad_token_id: int,
        *,
        max_chunks: int,
        stride: int,
        seed: int,
    ):
        super().__init__()
        self.Pos = Pos
        self.pad_token_id = pad_token_id
        self.labels = labels
        self.doc_ids = doc_ids
        # Flat chunk index: one (doc_idx, start_token) per training example.
        self.chunks: list[tuple[int, int]] = []
        for doc_idx, ids in enumerate(self.doc_ids):
            starts = chunk_starts(len(ids), stride)
            if len(starts) > max_chunks:
                starts = sorted(random.Random(seed + doc_idx).sample(starts, max_chunks))
            self.chunks.extend((doc_idx, st) for st in starts)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    async def async_len(self) -> int:
        return len(self.chunks)

    def is_finite(self) -> bool:
        return True

    def _encode_chunk(self, doc_idx: int, start: int) -> ClassificationExample:
        seq_len = self.Pos.size
        ids = self.doc_ids[doc_idx][start : start + seq_len]
        n = len(ids)
        padded = np.full((seq_len,), self.pad_token_id, dtype=np.int32)
        padded[:n] = ids
        seg = np.full((seq_len,), -1, dtype=np.int32)
        seg[:n] = 0
        seg_named = hax.named(seg, self.Pos)
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        return ClassificationExample.init(
            tokens=hax.named(padded, self.Pos),
            label=hax.named(np.int32(self.labels[doc_idx]), ()),
            attn_mask=mask,
        )

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ClassificationExample]:
        return [self._encode_chunk(*self.chunks[i]) for i in indices]


class ChunkedCachedClassificationDataset(AsyncDataset[ClassificationExample]):
    """Streaming chunked (MIL) dataset for scale (10M+). Same chunk semantics as
    ``ChunkedTextClassificationDataset`` (stride, ``max_chunks`` seeded random-subset), but tokens live
    in a Levanter ``TreeCache`` and are read BY INDEX at ``get_batch`` time, so host RAM stays bounded
    regardless of corpus size. The chunk index is built from per-doc token LENGTHS read cheaply from
    the cache's jagged-array offsets (no token data touched); only the small scalar ``label`` field is
    materialized up front.
    """

    def __init__(
        self,
        cache: TreeCache,
        Pos: Axis,
        pad_token_id: int,
        *,
        max_chunks: int,
        stride: int,
        seed: int,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.cache = cache
        self.Pos = Pos
        self.pad_token_id = pad_token_id
        ids_store = cache.store.tree["input_ids"]
        self._ids_store = ids_store
        n_total = ids_store.num_rows
        n = min(n_total, limit) if limit is not None else n_total
        # Per-doc token lengths from the jagged offsets (N+1 int64, cheap). offsets[0] holds the row
        # COUNT (not a real offset), so row 0's start is 0; lengths are the successive differences.
        raw = np.asarray(ids_store.offsets[0 : n_total + 1].read().result(), dtype=np.int64)
        raw[0] = 0
        lengths = np.diff(raw)[:n]
        # Labels: only the scalar label field (not the tokens) — small even at 10M.
        label_rows = cache.store.tree["label"].get_batch_sync(list(range(n)))
        self.labels = [int(np.asarray(r).reshape(-1)[0]) for r in label_rows]
        self.chunks: list[tuple[int, int]] = []
        for doc_idx in range(n):
            starts = chunk_starts(int(lengths[doc_idx]), stride)
            if len(starts) > max_chunks:
                starts = sorted(random.Random(seed + doc_idx).sample(starts, max_chunks))
            self.chunks.extend((doc_idx, st) for st in starts)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    async def async_len(self) -> int:
        return len(self.chunks)

    def is_finite(self) -> bool:
        return True

    def _encode_chunk(self, ids: np.ndarray, start: int, label: int) -> ClassificationExample:
        seq_len = self.Pos.size
        ids = np.asarray(ids, dtype=np.int32)[start : start + seq_len]
        n = len(ids)
        padded = np.full((seq_len,), self.pad_token_id, dtype=np.int32)
        padded[:n] = ids
        seg = np.full((seq_len,), -1, dtype=np.int32)
        seg[:n] = 0
        seg_named = hax.named(seg, self.Pos)
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        return ClassificationExample.init(
            tokens=hax.named(padded, self.Pos),
            label=hax.named(np.int32(label), ()),
            attn_mask=mask,
        )

    async def get_batch(self, indices: Sequence[int]) -> Sequence[ClassificationExample]:
        specs = [self.chunks[i] for i in indices]
        unique_docs = sorted({doc_idx for doc_idx, _ in specs})  # read each doc's tokens once per batch
        rows = await self._ids_store.get_batch(unique_docs)
        doc_to_ids = {doc_idx: np.asarray(r, dtype=np.int32) for doc_idx, r in zip(unique_docs, rows, strict=True)}
        return [self._encode_chunk(doc_to_ids[doc_idx], st, self.labels[doc_idx]) for doc_idx, st in specs]


@dataclass
class ClassificationDataConfig:
    """Data source for classification training (fastText line format)."""

    train_urls: list[str] = field(default_factory=list)
    validation_urls: list[str] = field(default_factory=list)
    tokenizer: str = "answerdotai/ModernBERT-base"
    useful_label: str = "__label__useful"
    max_train_rows: Optional[int] = None
    eval_rows: int = 7000
    # Tokenized TreeCache: built once from train_urls, streamed by index at train time (memory-bounded
    # — supports 10M+ rows where the old in-memory path OOM'd). Defaults next to the shards (region-local);
    # max_cache_token_len caps tokenization (8192 = largest ctx), so ONE cache serves every ctx in a sweep.
    cache_dir: Optional[str] = None
    max_cache_token_len: int = 8192
    # Data path: in-memory (default) reads all rows into host RAM — simple, fast-start, fine for
    # <=~1M rows (how the small runs have always trained). For 5M/10M it OOMs → set use_cache=True to
    # stream from a tokenized TreeCache (memory-bounded). Cache must be prebuilt to avoid build races.
    use_cache: bool = False
    # Marin's _maybe_override_auto_build_caches reads this; we build the cache explicitly here.
    auto_build_caches: bool = False
    # Chunked (MIL) training: split each doc into Pos-sized chunks, each chunk = one example with the
    # doc label (see ChunkedTextClassificationDataset). In-memory path only (the 200k pilot); a
    # chunk-aware cache is needed before this scales to 5M/10M. chunk_overlap 0.0 = non-overlapping
    # tiles, 0.5 = 50% overlap (stride = Pos.size // 2). max_doc_tokens bounds per-doc tokenization.
    chunked: bool = False
    max_chunks_per_doc: int = 16
    chunk_overlap: float = 0.0
    chunk_sample_seed: int = 0
    max_doc_tokens: int = 32768

    @property
    def the_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer)

    def chunk_stride(self, Pos: Axis) -> int:
        """Token stride between consecutive chunks: ``Pos.size`` (non-overlap) or less when overlapping."""
        if not 0.0 <= self.chunk_overlap < 1.0:
            raise ValueError(f"chunk_overlap must be in [0, 1); got {self.chunk_overlap}")
        return max(1, round(Pos.size * (1.0 - self.chunk_overlap)))

    def _train_cache_dir(self) -> str:
        if self.cache_dir:
            return self.cache_dir
        # the directory holding the shards (strip the glob/filename), + a cache subdir
        base = self.train_urls[0].rsplit("/", 1)[0]
        return f"{base}/_clf_token_cache"

    def _chunk_cache_dir(self) -> str:
        """Streaming chunk cache dir — distinct from the 8192 ``_clf_token_cache`` (keyed by the larger
        ``max_doc_tokens`` cap) so the two never collide. Must match ``build_chunk_treecache.py``."""
        if self.cache_dir:
            return self.cache_dir
        base = self.train_urls[0].rsplit("/", 1)[0]
        return f"{base}/_clf_token_cache_chunk{self.max_doc_tokens}"

    def build_train(self, tokenizer, Pos: Axis, pad_token_id: int) -> AsyncDataset[ClassificationExample]:
        paths = _expand_globs(self.train_urls)
        if self.chunked:
            stride = self.chunk_stride(Pos)
            if self.use_cache:
                cache_dir = self._chunk_cache_dir()
                logger.info(
                    f"train shards: {len(paths)}; CHUNKED STREAMING cache {cache_dir} "
                    f"(ctx={Pos.size} stride={stride} max_chunks={self.max_chunks_per_doc})"
                )
                source = TextUrlDataSource(paths)
                processor = ClassificationLineProcessor(tokenizer, self.useful_label, max_length=self.max_doc_tokens)
                cache = build_or_load_cache(cache_dir, source, processor, options=CacheOptions.default())
                return ChunkedCachedClassificationDataset(
                    cache,
                    Pos,
                    pad_token_id,
                    max_chunks=self.max_chunks_per_doc,
                    stride=stride,
                    seed=self.chunk_sample_seed,
                    limit=self.max_train_rows,
                )
            logger.info(
                f"train shards: {len(paths)}; CHUNKED in-RAM (ctx={Pos.size} stride={stride} "
                f"max_chunks={self.max_chunks_per_doc} overlap={self.chunk_overlap})"
            )
            doc_ids, labels = build_or_load_token_cache(
                paths,
                self.useful_label,
                self.max_train_rows,
                self.tokenizer,
                self.max_doc_tokens,
                self.cache_dir,
            )
            return ChunkedTextClassificationDataset(
                doc_ids,
                labels,
                Pos,
                pad_token_id,
                max_chunks=self.max_chunks_per_doc,
                stride=stride,
                seed=self.chunk_sample_seed,
            )
        if not self.use_cache:
            logger.info(f"train shards: {len(paths)}; in-memory load (use_cache=False)")
            texts, labels = read_fasttext_shards(paths, self.useful_label, limit=self.max_train_rows)
            logger.info(f"train docs: {len(texts)}")
            return TextClassificationDataset(texts, labels, tokenizer, Pos, pad_token_id)
        cache_dir = self._train_cache_dir()
        logger.info(f"train shards: {len(paths)}; build/load token cache at {cache_dir}")
        source = TextUrlDataSource(paths)
        processor = ClassificationLineProcessor(tokenizer, self.useful_label, max_length=self.max_cache_token_len)
        cache = build_or_load_cache(cache_dir, source, processor, options=CacheOptions.default())
        return CachedClassificationDataset(cache, Pos, pad_token_id, limit=self.max_train_rows)

    def build_eval(self) -> tuple[list[str], list[int]]:
        if not self.validation_urls:
            return [], []
        paths = _expand_globs(self.validation_urls)
        return read_frozen_eval(paths, self.useful_label, rows=self.eval_rows)


# --------------------------------------------------------------------------------------
# Scoring + F1 sweep (host-side)
# --------------------------------------------------------------------------------------


def score_texts(
    model,
    texts: list[str],
    tokenizer,
    Pos: Axis,
    pad_token_id: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Score texts -> P(useful). MUST be called inside the Trainer's mesh context: the model is
    sharded on the Trainer's device mesh, so we run under that same mesh (a fresh mesh has an
    incompatible device order). Each chunk is padded to a fixed batch_size for static shapes.
    """
    model = inference_mode(model, True)
    Batch = Axis("batch", batch_size)

    @hax.named_jit
    def _probs(m, tokens, mask):
        logits = m(tokens, mask).astype(jnp.float32)
        return hax.nn.softmax(logits, axis="label")["label", 1]

    out = np.zeros((len(texts),), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        n = len(chunk)
        ids = np.full((batch_size, Pos.size), pad_token_id, dtype=np.int32)
        seg = np.full((batch_size, Pos.size), -1, dtype=np.int32)
        for r, text in enumerate(chunk):
            enc = tokenizer(text, truncation=True, max_length=Pos.size)["input_ids"]
            ids[r, : len(enc)] = enc
            seg[r, : len(enc)] = 0
        tokens = hax.named(ids, (Batch, Pos))
        seg_named = hax.named(seg, (Batch, Pos))
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        out[start : start + n] = np.asarray(_probs(model, tokens, mask).array)[:n]
    return out


def score_texts_chunked(
    model,
    texts: list[str],
    tokenizer,
    Pos: Axis,
    pad_token_id: int,
    stride: int,
    max_doc_tokens: int,
    batch_size: int = 16,
) -> list[np.ndarray]:
    """Chunked inference: split each doc into ALL its ``Pos``-sized chunks (no per-doc cap), score
    every chunk, and return per-doc arrays of chunk P(useful). Chunks are flattened across docs so
    the forward pass batches efficiently, then regrouped by document. Must run inside the Trainer mesh.
    """
    model = inference_mode(model, True)
    Batch = Axis("batch", batch_size)

    @hax.named_jit
    def _probs(m, tokens, mask):
        logits = m(tokens, mask).astype(jnp.float32)
        return hax.nn.softmax(logits, axis="label")["label", 1]

    seq_len = Pos.size
    flat_ids: list[np.ndarray] = []
    owner: list[int] = []
    for start in range(0, len(texts), 1000):
        enc = tokenizer(texts[start : start + 1000], truncation=True, max_length=max_doc_tokens)["input_ids"]
        for k, e in enumerate(enc):
            ids = np.asarray(e, dtype=np.int32)
            for st in chunk_starts(len(ids), stride):
                flat_ids.append(ids[st : st + seq_len])
                owner.append(start + k)

    probs = np.zeros((len(flat_ids),), dtype=np.float32)
    for start in range(0, len(flat_ids), batch_size):
        chunk = flat_ids[start : start + batch_size]
        n = len(chunk)
        ids = np.full((batch_size, seq_len), pad_token_id, dtype=np.int32)
        seg = np.full((batch_size, seq_len), -1, dtype=np.int32)
        for r, c in enumerate(chunk):
            ids[r, : len(c)] = c
            seg[r, : len(c)] = 0
        tokens = hax.named(ids, (Batch, Pos))
        seg_named = hax.named(seg, (Batch, Pos))
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        probs[start : start + n] = np.asarray(_probs(model, tokens, mask).array)[:n]

    per_doc: list[list[float]] = [[] for _ in texts]
    for p, o in zip(probs, owner, strict=True):
        per_doc[o].append(float(p))
    return [np.asarray(x, dtype=np.float32) for x in per_doc]


# Aggregators map a doc's chunk-prob array -> a single doc score. The eval sweep picks the best
# aggregator × threshold; the winner informs the deployment-time aggregator over a doc's chunks.
CHUNK_AGGREGATORS = {
    "mean": lambda p: float(p.mean()),
    "max": lambda p: float(p.max()),
    "min": lambda p: float(p.min()),
    "top2mean": lambda p: float(np.sort(p)[-2:].mean()),
    "top3mean": lambda p: float(np.sort(p)[-3:].mean()),
}


def aggregate_sweep(
    chunk_probs_per_doc: list[np.ndarray], labels: Sequence[int]
) -> tuple[dict[str, tuple[float, float]], str]:
    """For each aggregator, collapse per-doc chunk probs to a doc score and run the F1 threshold sweep.
    Returns ``({agg_name: (best_f1, best_threshold)}, best_agg_name)``.
    """
    labels_arr = np.asarray(labels)
    results: dict[str, tuple[float, float]] = {}
    for name, fn in CHUNK_AGGREGATORS.items():
        doc_scores = np.asarray([fn(p) if len(p) else 0.0 for p in chunk_probs_per_doc], dtype=np.float32)
        results[name] = f1_sweep(doc_scores, labels_arr)
    best_agg = max(results, key=lambda k: results[k][0])
    return results, best_agg


def f1_sweep(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Best F1 over thresholds i/50, i in 1..49."""
    best_f1, best_t = 0.0, 0.5
    y = labels.astype(bool)
    for i in range(1, 50):
        t = i / 50
        pred = probs >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


# --------------------------------------------------------------------------------------
# Model construction + loss
# --------------------------------------------------------------------------------------


def build_model(config: LmConfig, Vocab: Axis, *, key, warm_start: bool, axis_mapping=None, compute_dtype=None):
    """Build the classifier for any registered architecture (see levanter.models.classification)."""
    return build_classifier(
        config, Vocab, key=key, warm_start=warm_start, axis_mapping=axis_mapping, compute_dtype=compute_dtype
    )


def classification_loss(model, example: ClassificationExample, *, key=None):
    return model.compute_loss(example, key=key)


# --------------------------------------------------------------------------------------
# Low-level training (also used by the offline integration test)
# --------------------------------------------------------------------------------------


def train_classifier(
    *,
    trainer_config: TrainerConfig,
    model_config: LmConfig,
    optimizer_config: OptimizerConfig,
    train_dataset: AsyncDataset[ClassificationExample],
    tokenizer,
    warm_start: bool,
    eval_texts: Optional[list[str]] = None,
    eval_labels: Optional[list[int]] = None,
    hf_save_path: Optional[str] = None,
    eval_chunked: bool = False,
    eval_stride: Optional[int] = None,
    eval_max_doc_tokens: int = 32768,
):
    optimizer = optimizer_config.build(trainer_config.num_train_steps)
    with Trainer(trainer_config, optimizer, classification_loss) as trainer:
        model_key, training_key = jrandom.split(jrandom.PRNGKey(trainer_config.seed), 2)
        parameter_axis_mapping = trainer.parameter_axis_mapping
        Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), parameter_axis_mapping)

        initial_model = build_model(
            model_config,
            Vocab,
            key=model_key,
            warm_start=warm_start,
            axis_mapping=parameter_axis_mapping,
            compute_dtype=trainer.mp.compute_dtype,
        )
        initial_model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(initial_model)
        state = trainer.initial_state(training_key, model=initial_model)
        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        # Shuffle so microbatches are class-mixed. The fastText survivor shards are class-ordered
        # (useful block then no_useful block); reading in order gives class-homogeneous microbatches
        # and degenerate gradients (loss collapses to ~0). A permutation fixes this regardless of the
        # on-disk order.
        train_dataset = train_dataset.shuffle(jrandom.PRNGKey(trainer_config.seed + 1))
        train_loader = trainer.data_loader(train_dataset).iter_from_step(state.step)
        info = trainer.train(state, train_loader)
        final_model = inference_mode(info.state.model, True)

        # Eval INSIDE the Trainer's mesh context: final_model is sharded on the Trainer's device
        # mesh, so scoring must run under that same mesh (a fresh mesh has an incompatible device
        # order -> jit device mismatch). SPLASH also needs this non-empty mesh.
        if eval_texts and eval_chunked:
            if eval_stride is None:
                raise ValueError("eval_chunked=True requires eval_stride")
            per_doc = score_texts_chunked(
                final_model,
                eval_texts,
                tokenizer,
                model_config.max_Pos,
                model_config.pad_token_id,
                stride=eval_stride,
                max_doc_tokens=eval_max_doc_tokens,
            )
            results, best_agg = aggregate_sweep(per_doc, eval_labels)
            n_chunks = [len(p) for p in per_doc]
            logger.info(
                "[eval] chunked: %d docs, %d total chunks (max %d/doc); per-aggregator best-F1: %s",
                len(eval_labels),
                sum(n_chunks),
                max(n_chunks),
                {k: round(v[0], 4) for k, v in results.items()},
            )
            for name, (f1, t) in results.items():
                levanter.tracker.log_summary({f"eval/{name}_f1": f1, f"eval/{name}_threshold": t})
            best_f1, best_t = results[best_agg]
            logger.info("[eval] best chunked agg=%s best_f1=%.4f @ t=%.2f", best_agg, best_f1, best_t)
            levanter.tracker.log_summary(
                {"eval/best_f1": best_f1, "eval/best_threshold": best_t, "eval/best_agg": best_agg}
            )
        elif eval_texts:
            probs = score_texts(final_model, eval_texts, tokenizer, model_config.max_Pos, model_config.pad_token_id)
            best_f1, best_t = f1_sweep(probs, np.asarray(eval_labels))
            logger.info(f"[eval] best_f1={best_f1:.4f} @ t={best_t:.2f} on {len(eval_labels)} docs")
            levanter.tracker.log_summary({"eval/best_f1": best_f1, "eval/best_threshold": best_t})

        # Save MUST also run inside the Trainer mesh: HF save_pretrained deshards via
        # with_sharding_constraint, which needs the non-empty mesh the model is sharded on.
        if hf_save_path:
            save_classifier(model_config, final_model, hf_save_path)
            logger.info(f"saved classifier to {hf_save_path}")
    return final_model


# --------------------------------------------------------------------------------------
# Config-driven entrypoint (the Fray/Iris entrypoint, submitted by marin)
# --------------------------------------------------------------------------------------


@dataclass
class TrainClassifierConfig:
    data: ClassificationDataConfig = field(default_factory=ClassificationDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=ModernBertConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    warm_start: bool = True
    hf_save_path: Optional[str] = None


def main(config: TrainClassifierConfig):
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer
    # pad id: model config's own (ModernBERT-family) wins; otherwise the tokenizer's. Explicit None
    # checks — 0 is a legitimate pad id for some archs.
    pad_id = getattr(config.model, "pad_token_id", None)
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("no pad_token_id on the model config or the tokenizer")

    train_dataset = config.data.build_train(tokenizer, config.model.max_Pos, pad_id)
    eval_texts, eval_labels = config.data.build_eval()

    trainer_config = config.trainer
    eval_stride = None
    if config.data.chunked:
        # The number of training examples is the number of CHUNKS, not docs — and it depends on the
        # doc length distribution + ctx + stride, knowable only after tokenizing. Recompute 1-epoch
        # steps here (deterministic given data+seed, so resume is stable). Eval uses the same stride.
        n_chunks = train_dataset.num_chunks
        new_steps = max(1, n_chunks // trainer_config.train_batch_size)
        logger.info(
            "chunked: %d chunks across %d docs -> %d steps (1 epoch); launcher had %d",
            n_chunks,
            len(train_dataset.labels),
            new_steps,
            trainer_config.num_train_steps,
        )
        trainer_config = replace(trainer_config, num_train_steps=new_steps, steps_per_eval=new_steps)
        eval_stride = config.data.chunk_stride(config.model.max_Pos)

    # Skip warm-start when a checkpoint already exists to resume from: warm-start re-fetches the HF
    # reference checkpoint (answerdotai/ModernBERT-*) on EVERY start, so on a resume it's both wasted
    # (the checkpoint's weights overwrite it) and a fragility — an HF hiccup/rate-limit turns a routine
    # restart into a crash loop. The checkpoint has the real weights, so resume never needs HF.
    warm_start = config.warm_start
    if warm_start:
        ckpt = trainer_config.checkpointer
        ckpt_paths = [p for p in (ckpt.base_path, ckpt.temporary_base_path) if p]
        if ckpt_paths and discover_latest_checkpoint(*ckpt_paths) is not None:
            logger.info("existing checkpoint found -> skipping HF warm-start (resume from checkpoint)")
            warm_start = False

    # hf_save_path handled inside train_classifier (must run within the Trainer mesh).
    train_classifier(
        trainer_config=trainer_config,
        model_config=config.model,
        optimizer_config=config.optimizer,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        warm_start=warm_start,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        hf_save_path=config.hf_save_path,
        eval_chunked=config.data.chunked,
        eval_stride=eval_stride,
        eval_max_doc_tokens=config.data.max_doc_tokens,
    )


if __name__ == "__main__":
    levanter.config.main(main)()
