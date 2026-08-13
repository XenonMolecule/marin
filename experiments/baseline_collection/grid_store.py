# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize a corpus's 24x5 grid into one Levanter cache per cell.

The grid is currently an *attribute table*: every document carries a topic and a
quality bucket, but the tokens still live in shard order. Training a mixture needs
the opposite layout — each ``(topic, quality)`` cell as its own contiguous cache,
because Levanter's ``DatasetComponent`` has no notion of a row range or filter, so
a cell it can weight has to be a cache of its own.

Regrouping is what :mod:`experiments.datakit.store.datakit_store` does, and this
module is only the wiring: it points the store at this project's three
co-partitioned tables and at the per-cell token counts already measured during the
merge. Nothing here reorders tokens or splits documents — a document moves from
one file to another intact, and Levanter tracks its boundaries with the same
JaggedArray offsets it always has.

One store per corpus. Cells are never pooled across corpora, since comparing and
mixing corpora against each other is the entire point of the sweep.

Decontamination and dedup attributes are deliberately not passed. high_quality was
labelled on its post-dedup, post-decon export and the rest were labelled where we
wanted them; handing the store an all-false decon table would falsely record that
decontamination ran at store time.

    python -m experiments.baseline_collection.grid_store --dataset dclm_10k
"""

from __future__ import annotations

import argparse
import json
import logging

import fsspec
from fray.cluster import ResourceConfig

from experiments.baseline_collection import grid_corpora, grid_tokenize
from experiments.baseline_collection.grid_corpora import (
    GRID_CORPORA,
    GRID_SUFFIX,
    METADATA_ROOT,
    NUM_TOPICS,
    OUTPUT_BASE,
    Stage,
    resolve,
    stage_output_dir,
)
from experiments.baseline_collection.grid_tokenize import DEFAULT_TOKENIZER, TOKENIZE_ROOT, tokenize_dir
from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.store.datakit_store import build_clustered_store
from experiments.datakit.store.store_compat import (
    AssignmentAttrData,
    QualityScores,
    TokenizedAttrData,
)

logger = logging.getLogger(__name__)

STORE_ROOT = "datakit/store"
QUALITY_MODEL_DIR = "gs://marin-us-central1/resources/datakit/quality/pooled_junkgate2"
QUALITY_CALIB_FILE = "calib_bme.json"

# Our hottest cell is ~2.3B tokens, against the ~651B that forced upstream's 32-way
# split. Targeting 8B keeps every cell at k=1 -- one reducer, one cache, and no
# consolidation copy -- while still splitting a future corpus that outgrows it.
# Upstream's 8B leaves nearly every cell at k=1, which is right for corpora whose
# hottest cell is ~2.3B. resiliparse is an order of magnitude fatter (up to 18B a
# cell), and a reducer writes its whole subshard in one uninterrupted stretch, so
# on preemptible workers an 8B cell is repeatedly killed before it can commit.
# 4B halves that write. The plan is pinned on the first build and never
# recomputed, so changing this only affects a store built from scratch.
TARGET_TOKENS_PER_SUBSHARD = 4_000_000_000
MAX_SUBSHARDS = 32


# See grid_tokenize: region follows the corpus, not COMPUTE_REGION.
def worker_resources(region: str) -> ResourceConfig:
    return ResourceConfig(cpu=2, ram="16g", disk="16g", preemptible=True, regions=[region])


DEFAULT_MAX_WORKERS = 256
# Reduce tasks. Only ~120 groups exist per corpus, so more than that idles.
DEFAULT_REDUCE_SHARDS = 128


def store_output(dataset: str) -> str:
    return f"{OUTPUT_BASE}/{STORE_ROOT}/{dataset}_{GRID_SUFFIX}"


def bucket_token_hint(dataset: str) -> dict[tuple[int, int], int] | None:
    """Per-cell token counts from the merged grid, for subshard sizing.

    These come from the topic table's ``token_length`` (the gte tokenizer), not the
    training tokenizer, so they are approximate — which is all a sizing hint needs
    to be. Returns None if the corpus has no merged grid yet, in which case the
    store falls back to a flat subshard count.
    """
    path = f"{OUTPUT_BASE}/{METADATA_ROOT}/{dataset}/distribution.json"
    with fsspec.open(path) as fh:
        summary = json.load(fh)
    grid = summary.get("grid")
    if grid is None:
        logger.warning("%s: distribution.json has no grid; subshard sizing will be flat", dataset)
        return None
    tokens = grid["tokens"]
    hint = {
        (topic, quality): int(count) for topic, row in enumerate(tokens) for quality, count in enumerate(row) if count
    }
    logger.info(
        "%s: %d populated cells, largest %.2fB tokens",
        dataset,
        len(hint),
        max(hint.values()) / 1e9,
    )
    return hint


def run(
    dataset: str,
    source: str,
    max_workers: int,
    reduce_shards: int,
    tokenize_override: str | None = None,
    output_override: str | None = None,
    compute_region: str = "us-central1",
    resume: bool = False,
) -> None:
    """Build the per-cell caches for one corpus.

    ``tokenize_override`` points the join at a partial tokenization; the topic and
    quality tables are still the real ones, since the store reads only the
    attribute shards whose basenames the tokenize directory contains. That makes a
    genuine small-scale smoke possible against real labels.
    """
    corpus = resolve(dataset, source)
    tokenize = TokenizedAttrData(
        output_dirs={"train": tokenize_override or tokenize_dir(dataset)},
        source_main_dirs={"train": corpus.path},
        tokenizer=DEFAULT_TOKENIZER,
    )
    cluster_assign = AssignmentAttrData(
        output_dir=stage_output_dir(dataset, Stage.TOPIC),
        source_main_dir=corpus.path,
        k_train=NUM_TOPICS,
    )
    quality = QualityScores(
        main_output_dir=f"{stage_output_dir(dataset, Stage.QUALITY)}/outputs/main",
        model_dir=QUALITY_MODEL_DIR,
        calib_file=QUALITY_CALIB_FILE,
        bucket_edges=list(BUCKET_EDGES),
    )
    output_path = output_override or store_output(dataset)
    logger.info("%s: %s/%s -> %s", dataset, TOKENIZE_ROOT, dataset, output_path)

    artifact = build_clustered_store(
        tokenize={dataset: tokenize},
        cluster_assign={dataset: cluster_assign},
        quality={dataset: quality},
        output_path=output_path,
        cluster_view=NUM_TOPICS,
        worker_resources=worker_resources(compute_region),
        max_workers=max_workers,
        reduce_shards=reduce_shards,
        bucket_token_hint=bucket_token_hint(dataset),
        target_tokens_per_subshard=TARGET_TOKENS_PER_SUBSHARD,
        max_subshards=MAX_SUBSHARDS,
        default_subshards=1,
        resume=resume,
    )
    logger.info(
        "%s: %d cells, %d docs, %d tokens -> %s",
        dataset,
        len(artifact.buckets),
        sum(b.total_elements for b in artifact.buckets),
        sum(b.total_tokens for b in artifact.buckets),
        output_path,
    )


def rebind_output_base(output_base: str) -> None:
    """Point every module whose path helpers this one calls at ``output_base``.

    ``from grid_corpora import OUTPUT_BASE`` binds the value at import time, so a
    module keeps its own copy. Rebinding only this module and grid_corpora leaves
    ``grid_tokenize.tokenize_dir`` resolving to the default region, which surfaces
    as a confusing "no tokenize shards" rather than as a region error.
    """
    globals()["OUTPUT_BASE"] = output_base
    grid_corpora.OUTPUT_BASE = output_base
    grid_tokenize.OUTPUT_BASE = output_base


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(GRID_CORPORA))
    parser.add_argument("--source", choices=["native", "mirror"], default="mirror")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--reduce-shards", type=int, default=DEFAULT_REDUCE_SHARDS)
    parser.add_argument("--tokenize-dir", default=None, help="Join against this tokenization. For smoke runs.")
    parser.add_argument("--output-path", default=None, help="Write the store here instead of the canonical tree.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cells already materialized by a previous run. The shuffle still repeats; "
        "only finished reducers are skipped.",
    )
    parser.add_argument(
        "--compute-region",
        default="us-central1",
        help="Region to schedule workers in. Must match where the corpus and tokenization live, "
        "or the store reads them cross-region.",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Bucket root for this run's outputs, e.g. gs://marin-us-east5. For a corpus outside "
        "COMPUTE_REGION; do NOT flip that constant, which relocates the other corpora too.",
    )
    args = parser.parse_args()
    if args.output_base:
        rebind_output_base(args.output_base)

    run(
        args.dataset,
        args.source,
        args.max_workers,
        args.reduce_shards,
        tokenize_override=args.tokenize_dir,
        output_override=args.output_path,
        compute_region=args.compute_region,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
