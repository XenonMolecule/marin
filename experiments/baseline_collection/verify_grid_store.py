# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a materialized store against the attribute tables it was built from.

The store's own artifact reports what it *wrote*. This recomputes what it *should
have* written, straight from the topic and quality parquets, and compares cell by
cell. That distinction matters: every interesting failure here — a shard silently
skipped, a positional join off by a row, a cell whose tokens went to the wrong
reducer — produces a self-consistent artifact.

Three checks:

1. **Cell membership.** Recount the ``(cluster_24, quality_bucket)`` grid over
   exactly the shards the store consumed, and require an exact match with each
   cell cache's row count.
2. **Loadability.** Open every cell through ``TreeCache.load``. A cache that
   cannot be opened is worse than a missing one, because a mixture will accept the
   path and train on nothing.
3. **Token totals.** Compare the sum of cell tokens against the tokenized parquets.

    python -m experiments.baseline_collection.verify_grid_store \\
        --dataset dclm_10k --store gs://.../store/dclm_10k --tokenize-dir gs://.../tokenize/dclm_10k
"""

from __future__ import annotations

import argparse
import logging
import posixpath
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from levanter.store.cache import TreeCache

from experiments.baseline_collection import grid_store
from experiments.baseline_collection.grid_corpora import GRID_CORPORA, NUM_TOPICS, Stage, stage_output_dir
from experiments.baseline_collection.grid_store import store_output
from experiments.baseline_collection.grid_tokenize import tokenize_dir
from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.store.datakit_store import ClusteredStoreData
from experiments.datakit.store.store_compat import read_artifact, sp_open
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

READ_PARALLELISM = 32
N_QUALITY = len(BUCKET_EDGES) + 1
EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}


def _expected_cells(dataset: str, basenames: list[str]) -> tuple[Counter, int]:
    """Recount the grid over the given shards, reading the attribute tables directly."""
    topic_dir = stage_output_dir(dataset, Stage.TOPIC)
    quality_dir = f"{stage_output_dir(dataset, Stage.QUALITY)}/outputs/main"

    def one(basename: str) -> tuple[Counter, int]:
        with sp_open(f"{topic_dir}/{basename}", "rb") as fh:
            topics = pq.read_table(fh, columns=[f"cluster_{NUM_TOPICS}"]).column(0).to_numpy(zero_copy_only=False)
        with sp_open(f"{quality_dir}/{basename}", "rb") as fh:
            buckets = pq.read_table(fh, columns=["quality_bucket"]).column(0).to_numpy(zero_copy_only=False)
        if len(topics) != len(buckets):
            raise RuntimeError(f"{basename}: topic rows {len(topics)} != quality rows {len(buckets)}")
        return Counter(zip(topics.tolist(), buckets.tolist(), strict=True)), len(topics)

    with ThreadPoolExecutor(max_workers=READ_PARALLELISM) as pool:
        results = list(pool.map(one, basenames))

    cells: Counter = Counter()
    total = 0
    for counts, n in results:
        cells.update(counts)
        total += n
    return cells, total


def _tokenized_totals(paths: list[str]) -> tuple[int, int]:
    """``(docs, tokens)`` across the tokenized parquets the store consumed."""

    def one(path: str) -> tuple[int, int]:
        with sp_open(path, "rb") as fh:
            table = pq.read_table(fh, columns=["input_ids"])
        if table.num_rows == 0:
            return 0, 0
        lengths = table.column("input_ids").combine_chunks().value_lengths()
        return table.num_rows, int(np.asarray(lengths).sum())

    with ThreadPoolExecutor(max_workers=READ_PARALLELISM) as pool:
        results = list(pool.map(one, paths))
    return sum(r[0] for r in results), sum(r[1] for r in results)


def verify(dataset: str, store_path: str, tokenize_path: str) -> None:
    """Compare a store against its inputs; raise on the first inconsistency."""
    tok_shards = fsspec_glob(f"{tokenize_path.rstrip('/')}/*.parquet")
    if not tok_shards:
        raise FileNotFoundError(f"no tokenized shards under {tokenize_path}")
    basenames = sorted(posixpath.basename(p) for p in tok_shards)
    logger.info("%s: verifying %d shards -> %s", dataset, len(basenames), store_path)

    artifact = read_artifact(store_path, ClusteredStoreData)
    expected, expected_docs = _expected_cells(dataset, basenames)
    got = {(b.cluster_id, b.quality_bucket): b for b in artifact.buckets}

    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    if missing or extra:
        raise RuntimeError(f"cell set mismatch: missing={missing!r} extra={extra!r}")

    bad = [
        (cell, expected[cell], got[cell].total_elements)
        for cell in expected
        if expected[cell] != got[cell].total_elements
    ]
    if bad:
        raise RuntimeError(f"row-count mismatch in {len(bad)} cells, first 5: {bad[:5]!r}")

    store_docs = sum(b.total_elements for b in artifact.buckets)
    if store_docs != expected_docs:
        raise RuntimeError(f"store holds {store_docs} docs, attribute tables have {expected_docs}")

    tok_docs, tok_tokens = _tokenized_totals(tok_shards)
    if tok_docs != expected_docs:
        raise RuntimeError(f"tokenized shards hold {tok_docs} docs, attribute tables have {expected_docs}")
    store_tokens = sum(b.total_tokens for b in artifact.buckets)
    if store_tokens != tok_tokens:
        raise RuntimeError(f"store holds {store_tokens} tokens, tokenized shards have {tok_tokens}")

    for bucket in sorted(artifact.buckets, key=lambda b: -b.total_elements):
        cache = TreeCache.load(bucket.path, EXEMPLAR)
        if len(cache) != bucket.total_elements:
            raise RuntimeError(f"{bucket.path}: cache has {len(cache)} rows, artifact claims {bucket.total_elements}")

    logger.info(
        "OK: %d/%d cells, %d docs, %d tokens, every cell loadable",
        len(artifact.buckets),
        NUM_TOPICS * N_QUALITY,
        store_docs,
        store_tokens,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(GRID_CORPORA))
    parser.add_argument("--store", default=None, help="Store path; defaults to the canonical one.")
    parser.add_argument("--tokenize-dir", default=None, help="Tokenization the store consumed.")
    parser.add_argument(
        "--output-base",
        default=None,
        help="Bucket root for this corpus's artifacts, e.g. gs://marin-us-east5. Required for a "
        "corpus outside the default region: --store and --tokenize-dir do NOT cover the topic and "
        "quality tables, which are resolved from this.",
    )
    args = parser.parse_args()
    # Must precede the store_output/tokenize_dir defaults below, and rebinds
    # grid_corpora too so stage_output_dir finds the attribute tables.
    if args.output_base:
        grid_store.rebind_output_base(args.output_base)
    verify(
        args.dataset,
        args.store or store_output(args.dataset),
        args.tokenize_dir or tokenize_dir(args.dataset),
    )


if __name__ == "__main__":
    main()
