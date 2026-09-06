# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a classifier TreeCache as N INDEPENDENT per-shard jobs, then consolidate.

``build_clf_cache.py`` hands the whole shard list to one ``build_or_load_cache`` call, which spawns
a Zephyr actor group sized to the shard count (96 for the 90M corpus). On a busy cluster that group
schedules ONE actor and the build crawls: the 90M build ran 10 h at batch priority and wrote zero
bytes, with no other actor even queued. The fan-out is the point of failure — a coordinated
N-worker group is all-or-nothing, while N ordinary jobs schedule independently as capacity frees.

So: each ``build`` job tokenizes its assigned shards ONE AT A TIME into a single-shard cache under
``{cache_dir}_parts/part_NNN`` (one shard ⇒ one actor ⇒ always schedulable), and a final
``consolidate`` pass merges the parts into the real ``cache_dir`` with a proper ledger. The result
is byte-for-byte what the monolithic builder would have produced, so training loads it unchanged.

Launch the builders (one job per shard group; they are independent and can start in any order)::

    for i in $(seq 0 95); do
      uv run iris --cluster marin job run --region us-central2 \\
        --cpu 8 --memory 32GB --disk 50GB --extra cpu --extra dclm --enable-extra-resources \\
        --priority interactive --no-wait --job-name clfcache-90m-p$i -e HF_TOKEN "$HF_TOKEN" -- \\
        python -m experiments.baseline_collection.build_clf_cache_sharded build \\
          --train-glob '<glob>' --cache-dir '<dir>' --shard-start $i --shard-end $((i+1))
    done

Then, once every part exists::

    python -m experiments.baseline_collection.build_clf_cache_sharded consolidate \\
      --train-glob '<glob>' --cache-dir '<dir>'
"""

import argparse
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.main.train_classifier import ClassificationLineProcessor, _expand_globs, _parse_line
from levanter.store.cache import (
    CacheLedger,
    CacheMetadata,
    CacheOptions,
    SerialCacheWriter,
    build_or_load_cache,
    consolidate_shard_caches,
)
from levanter.store.tree_store import TreeStore
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Chars per doc, matching ClassificationLineProcessor: 100k chars is ~12x what 8192 tokens needs, so
# the stored ids are identical to no-cap while pathological multi-MB tails are skipped.
MAX_TEXT_CHARS = 100_000


class BatchedClassificationLineProcessor(ClassificationLineProcessor):
    """Same output as ``ClassificationLineProcessor``, tokenized in ONE call per batch.

    The stock processor loops ``for line in batch`` and calls the tokenizer per document, so the
    Rust tokenizer's thread pool never engages, and it declares ``num_cpus = 1`` so the actor gets a
    single core regardless. Measured on this corpus: a batched call does ~126 docs/s while the
    per-doc loop is several times slower — enough that a 925k-doc shard ran 6.4 h on an
    un-preempted worker without committing anything.

    Batching the call (and asking for real CPUs) is a pure speedup: ``metadata`` is inherited
    unchanged, so caches built with this processor are interchangeable with the stock builder's.
    """

    def __init__(self, tokenizer, useful_label: str, max_length: int = 8192, num_cpus: int = 4):
        super().__init__(tokenizer, useful_label, max_length)
        self._num_cpus = num_cpus

    @property
    def num_cpus(self) -> int:
        return self._num_cpus

    def __call__(self, batch: Sequence[str]) -> list[dict]:
        pairs = [p for p in (_parse_line(line, self.useful_label) for line in batch) if p is not None]
        if not pairs:
            return []
        texts = [text[:MAX_TEXT_CHARS] for _, text in pairs]
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length)["input_ids"]
        return [
            {"input_ids": np.asarray(ids, dtype=np.int32), "label": np.int32(label)}
            for (label, _), ids in zip(pairs, encoded, strict=True)
        ]


def _part_path(cache_dir: str, index: int) -> str:
    return f"{cache_dir.rstrip('/')}_parts/part_{index:03d}"


def _processor(tokenizer_name: str, useful_label: str, max_length: int) -> ClassificationLineProcessor:
    return BatchedClassificationLineProcessor(AutoTokenizer.from_pretrained(tokenizer_name), useful_label, max_length)


def _build_one_inline(path: str, part: str, processor, batch_lines: int) -> int:
    """Tokenize ONE shard straight into a TreeCache in-process — no Zephyr, no actors.

    `build_or_load_cache` spawns a Zephyr actor group even for a single shard. On this cluster those
    actors report `running` while producing zero bytes for hours (observed at 16 GB and again at
    48 GB, so it is not the actor's 32 GB request), and with the log plane down the actor is
    unobservable. Since each call here handles exactly one shard, the distributed layer buys nothing
    — `SerialCacheWriter` writes the identical TreeCache format (and the same ledger that
    `consolidate_shard_caches` consumes) with a plain Python loop we can actually reason about.
    """
    rows = 0
    batch: list[str] = []
    metadata = CacheMetadata(preprocessor_metadata=processor.metadata)
    with SerialCacheWriter(part, processor.output_exemplar, metadata=metadata, shard_name=path) as writer:
        with fsspec.open(path, "rt", compression="gzip", encoding="utf-8", errors="replace") as f:
            for line in f:
                batch.append(line)
                if len(batch) >= batch_lines:
                    out = processor(batch)
                    if out:
                        writer.write_batch(out)
                        rows += len(out)
                    batch = []
        if batch:
            out = processor(batch)
            if out:
                writer.write_batch(out)
                rows += len(out)
    return rows


def run_build(args) -> None:
    paths = _expand_globs([args.train_glob])
    end = args.shard_end if args.shard_end is not None else len(paths)
    if not 0 <= args.shard_start < end <= len(paths):
        raise ValueError(f"bad shard range [{args.shard_start},{end}) over {len(paths)} shards")
    logger.info("building shards [%d,%d) of %d (inline=%s)", args.shard_start, end, len(paths), args.inline)

    processor = _processor(args.tokenizer, args.useful_label, args.max_length)
    for index in range(args.shard_start, end):
        part = _part_path(args.cache_dir, index)
        fs, rpart = fsspec.core.url_to_fs(part)
        if fs.exists(f"{part}/shard_ledger.json"):
            logger.info("[%d] already built -> %s", index, part)
            continue
        if fs.exists(rpart):
            # A part with no ledger is the debris of a preempted builder. SerialCacheWriter APPENDS
            # to existing zarr arrays, so rebuilding on top of it silently doubles the rows while the
            # new ledger records only this run's count — which shifts every row offset downstream and
            # corrupts the consolidated cache (6 of 96 parts, ~900k phantom rows each, on the 90M
            # HTML build). Incomplete means discard.
            logger.warning("[%d] discarding ledger-less partial part -> %s", index, part)
            fs.rm(rpart, recursive=True)
        started = time.time()
        if args.inline:
            rows = _build_one_inline(paths[index], part, processor, args.batch_lines)
            logger.info("[%d] done rows=%d in %.1f min -> %s", index, rows, (time.time() - started) / 60, part)
        else:
            cache = build_or_load_cache(
                part, TextUrlDataSource([paths[index]]), processor, options=CacheOptions.default()
            )
            logger.info("[%d] done finished=%s rows=%d -> %s", index, cache.is_finished, len(cache), part)
    logger.info("BUILD RANGE DONE [%d,%d)", args.shard_start, end)


def _corrupt_parts(parts: list[str], processor) -> list[tuple[str, int, int]]:
    """Parts whose data holds more rows than their ledger claims — one label per row is the invariant.

    A preempted builder leaves partial zarr arrays; the relaunch appends to them, so the data grows
    while the fresh ledger counts only the second run. Consolidation trusts the ledgers for row
    offsets, so a single bad part silently misaligns every part after it.
    """
    metadata = CacheMetadata(preprocessor_metadata=processor.metadata)

    def check(part: str) -> tuple[str, int, int] | None:
        # A part can also be STRUCTURALLY broken — e.g. `input_ids/offsets/zarr.json` missing because
        # a delete raced a live writer — and then opening it raises instead of returning a count. That
        # is still a bad part, so report it as one (-1 rows) rather than aborting the whole scan.
        try:
            ledger = CacheLedger.load(part, metadata)
            store = TreeStore.open(processor.output_exemplar, part, mode="r", cache_metadata=True)
            data_rows = int(store.tree["label"].data_size)
        except Exception as e:
            logger.warning("part unreadable: %s (%s)", part, type(e).__name__)
            return (part, -1, -1)
        return None if data_rows == ledger.total_num_rows else (part, ledger.total_num_rows, data_rows)

    with ThreadPoolExecutor(16) as pool:
        return [bad for bad in pool.map(check, parts) if bad is not None]


def _consolidate_inline(parts: list[str], output_path: str, exemplar, metadata) -> int:
    """Merge part-caches in-process — the same steps as ``consolidate_shard_caches`` minus Zephyr.

    The stock consolidator runs its probe and copy stages through Zephyr actor groups. On this
    cluster the *probe* actor — which only reads 40 ledgers, seconds of work — sat `running` for 45
    minutes writing nothing, the same failure that stalled the builds. Serial here is fine: probing
    is metadata-only and the copy is GCS-to-GCS within one region.
    """
    import asyncio
    import copy as _copy
    import operator

    import jax
    from levanter.store.cache import (
        CacheLedger,
        _consolidate_metadata,
        _expose_cache_rows,
        _extend_cache_with_other_cache,
        _field_counts_from_store,
        _merge_materialized_ledgers,
    )
    from levanter.store.tree_store import TreeStore

    first = TreeStore.open(exemplar, parts[0], mode="r", cache_metadata=True)
    data_offset_tree = jax.tree.map(lambda x: 0, first.tree)
    shard_info: list[dict] = []
    field_counts: list[dict] = []
    total_rows = 0
    for path in parts:
        ledger = CacheLedger.load(path, metadata)
        store = TreeStore.open(exemplar, path, mode="r", cache_metadata=True)
        # A part written by SerialCacheWriter may leave field_counts empty; recover them from the
        # store, since _merge_materialized_ledgers sums them into the final ledger.
        field_counts.append(ledger.field_counts or _field_counts_from_store(store))
        sizes = jax.tree.map(lambda x: x.data_size, store.tree)
        shard_info.append(
            {
                "path": path,
                "shard_name": os.path.basename(path),
                "row_offset": total_rows,
                "data_offset_tree": _copy.deepcopy(data_offset_tree),
                "ledger": ledger,
            }
        )
        total_rows += ledger.total_num_rows
        data_offset_tree = jax.tree.map(operator.add, data_offset_tree, sizes)
    logger.info("probed %d parts, %d rows total", len(parts), total_rows)

    TreeStore.open(exemplar, output_path, mode="w", cache_metadata=True)
    for i, info in enumerate(shard_info):
        asyncio.run(
            _extend_cache_with_other_cache(
                output_path, info["path"], exemplar, info["data_offset_tree"], info["row_offset"]
            )
        )
        logger.info("copied %d/%d (%s)", i + 1, len(shard_info), info["shard_name"])
    asyncio.run(_consolidate_metadata(output_path, exemplar, shard_info))
    ledger = _merge_materialized_ledgers(output_path, parts, [s["ledger"] for s in shard_info], field_counts, metadata)
    _expose_cache_rows(output_path, exemplar, ledger.total_num_rows)
    return ledger.total_num_rows


def run_consolidate(args) -> None:
    paths = _expand_globs([args.train_glob])
    parts = [_part_path(args.cache_dir, i) for i in range(len(paths))]
    missing = [p for p in parts if not fsspec.core.url_to_fs(p)[0].exists(f"{p}/shard_ledger.json")]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(parts)} parts missing; first: {missing[0]}")

    processor_for_check = _processor(args.tokenizer, args.useful_label, args.max_length)
    corrupt = _corrupt_parts(parts, processor_for_check)
    if corrupt:
        detail = "\n".join(f"  {p}: ledger_rows={r} label_rows={d} (+{d - r})" for p, r, d in corrupt)
        raise SystemExit(
            f"{len(corrupt)} of {len(parts)} parts have more data rows than their ledger claims — "
            f"consolidating them would shift every downstream row offset and produce a cache whose "
            f"rows read back EMPTY. Delete and rebuild these parts first:\n{detail}"
        )

    processor = _processor(args.tokenizer, args.useful_label, args.max_length)
    logger.info("consolidating %d parts -> %s (inline=%s)", len(parts), args.cache_dir, args.inline)
    if args.inline:
        rows = _consolidate_inline(
            parts, args.cache_dir, processor.output_exemplar, CacheMetadata(preprocessor_metadata=processor.metadata)
        )
        logger.info("CACHE DONE rows=%d -> %s", rows, args.cache_dir)
        return
    ledger = consolidate_shard_caches(
        shard_cache_paths=parts,
        output_path=args.cache_dir,
        exemplar=processor.output_exemplar,
        metadata=None,
    )
    logger.info("CACHE DONE finished=%s rows=%d -> %s", ledger.is_finished, ledger.total_num_rows, args.cache_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "consolidate"):
        s = sub.add_parser(name)
        s.add_argument("--train-glob", required=True)
        s.add_argument("--cache-dir", required=True)
        s.add_argument("--tokenizer", default="answerdotai/ModernBERT-base")
        s.add_argument("--max-length", type=int, default=8192)
        s.add_argument("--useful-label", default="__label__useful")
        s.add_argument("--inline", action="store_true", default=True, help="run in-process (no Zephyr actors).")
        s.add_argument("--no-inline", dest="inline", action="store_false", help="use the Zephyr path instead.")
        if name == "build":
            s.add_argument("--shard-start", type=int, default=0)
            s.add_argument("--shard-end", type=int, default=None, help="exclusive; default = all shards")
            s.add_argument("--batch-lines", type=int, default=1000, help="lines per tokenizer batch.")
    args = p.parse_args()
    (run_build if args.cmd == "build" else run_consolidate)(args)


if __name__ == "__main__":
    main()
