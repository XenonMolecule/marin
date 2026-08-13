# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality-only scoring as a Zephyr pipeline, so it rides on spare CPU.

**Why Zephyr rather than one Iris job per chunk.** The quality model is 3M params
over 512 tokens and its cost is tokenization on the host, not matmul — a whole
TPU task measured only ~2.3x a single laptop core. Submitting Iris jobs for it
takes a worker *slot*, and Iris will happily place a CPU-only task on a TPU host
that has spare cores, so a CPU-bound job ends up evicting accelerator work that
genuinely needs the chips. Zephyr instead spreads small worker actors across
whatever CPU is going spare — including the idle cores alongside a running TPU
job — so this scores on top of the extraction fleet rather than in place of it.

Runs at ``preemptible=True`` and low per-worker CPU for the same reason: be
filler, not a competitor.

**Expect the coordinator to die before a large corpus finishes, and just re-run.**
Being preemptible filler at ``batch`` priority means workers get evicted often,
those evictions count as task failures, and zephyr eventually trips
``max_task_failures`` and gives up — nemotron died this way at 20,841/24,390.
That is not a bug to engineer around: every shard carries a done marker, so a
relaunch resumes with zero recomputation and costs one command. Raising the
failure tolerance instead would only trade a cheap restart for a worker pool that
thrashes silently.

Correctness is shared with the Iris path rather than reimplemented. Both call
:func:`grid_label.score_quality` and :func:`grid_label.write_quality_shard`, so
the two routes emit byte-identical outputs, and the merge cannot tell which
produced a given shard.

Two things this must get right, both of which upstream's ``score.py`` handles
differently because its inputs are already normalized:

* **Empty shards still produce output.** 63% of fineweb_cc's 21,531 shards are
  empty gzips. The co-partitioning contract is one output file per *input* shard,
  so a zero-row file must be written; skipping would shift every later shard's
  positional alignment. Upstream returns early on an empty shard because it
  derives the output name from the first record — which is also why this module
  drives the pipeline off the file list, so the name never depends on content.
* **The ``id`` is computed here.** Our corpora are raw document layers, not
  datakit ``NormalizedData``, so there is no ``id`` column to carry through;
  :func:`grid_corpora.read_shard` hashes the text exactly as the topic stage did.

    python -m experiments.baseline_collection.grid_quality_zephyr \\
        --dataset dclm_10k --quality-model gs://.../pooled_junkgate2 --max-workers 64
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator

import numpy as np
from fray.cluster import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

from experiments.baseline_collection import grid_label
from experiments.baseline_collection.grid_corpora import (
    GRID_CORPORA,
    LABEL_TEXT_CHARS,
    Stage,
    list_shards,
    output_stem,
    pending_shards,
    read_shard,
    resolve,
    write_done,
)

logger = logging.getLogger(__name__)

# Small and preemptible on purpose: these are meant to pack into leftover cores
# beside accelerator jobs, not to claim hosts of their own.
#
# RAM is 16g, matching upstream's *production* override rather than its module
# default of 8g. 8g OOM-killed every worker on nemotron and fineweb_edu here while
# high_quality survived, and the asymmetry is the tell: a worker carries the model
# plus per-sequence-length compiled JAX caches, so memory grows with the number of
# distinct shard shapes a worker sees. high_quality has 512 shards; nemotron has
# 24,390, so its workers churn through far more of them and accumulate more.
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", preemptible=True, regions=["us-central1"])
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="3g", preemptible=False, regions=["us-central1"])
DEFAULT_MAX_WORKERS = 64


def _score_shards(dataset: str, source: str, model_dir: str):
    """Build the ``map_shard`` fn that scores every input file it is handed.

    Each item in the stream is an input *path*, so the output name never depends
    on shard contents and an empty file still yields its zero-row output.
    """

    def score_shard(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        corpus = resolve(dataset, source)
        # Cached per worker process by load_quality's own module-level cache; the
        # InlineRunner keeps that process alive across shards so the 53 MB model
        # is fetched once per worker rather than once per shard.
        scorer = _cached_scorer(model_dir)
        for path in paths:
            docs = read_shard(path, corpus, LABEL_TEXT_CHARS)
            if len(docs):
                scores, buckets = grid_label.score_quality(scorer, docs.texts)
            else:
                scores = np.array([], dtype=np.float32)
                buckets = np.array([], dtype=np.int8)
            grid_label.write_quality_shard(dataset, path, docs, scores, buckets)
            write_done(dataset, Stage.QUALITY, path, len(docs))
            # Local zephyr exposes counters.increment(name, value); upstream's
            # counters.pipeline.update_counter does not exist here. Getting this
            # wrong raised AttributeError *after* each shard's outputs and done
            # marker were already written, so the data still completed while every
            # worker died and retried — correct results, wasted compute, and a
            # coordinator that reported failure on a finished corpus.
            counters.increment("grid_quality/shards", 1)
            counters.increment("grid_quality/docs", len(docs))
            yield {"shard": output_stem(path), "n_docs": len(docs)}

    return score_shard


_SCORER_CACHE: dict[str, object] = {}


def _cached_scorer(model_dir: str):
    """One scorer + calibration per worker process."""
    if model_dir not in _SCORER_CACHE:
        _SCORER_CACHE[model_dir] = grid_label.load_quality(model_dir)
    return _SCORER_CACHE[model_dir]


def run(dataset: str, source: str, model_dir: str, max_workers: int) -> None:
    """Score every pending shard of ``dataset`` across a Zephyr worker pool."""
    corpus = resolve(dataset, source)
    shards = list_shards(corpus)
    todo = pending_shards(dataset, Stage.QUALITY, shards)
    if not todo:
        logger.info("%s: quality already complete (%d shards)", dataset, len(shards))
        return

    logger.info("%s: scoring %d/%d shards on <=%d zephyr workers", dataset, len(todo), len(shards), max_workers)
    pipeline = Dataset.from_list(todo).map_shard(_score_shards(dataset, source, model_dir))
    ctx = ZephyrContext(
        name=f"grid-quality-{dataset.replace('_', '-')}",
        resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=max_workers,
        # Keeps the cached model alive across shards within a worker process.
        stage_runner_factory=InlineRunner,
    )
    result = ctx.execute(pipeline)
    logger.info("%s: done, counters=%s", dataset, dict(result.counters))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(GRID_CORPORA))
    parser.add_argument("--source", choices=["native", "mirror"], default="mirror")
    parser.add_argument("--quality-model", required=True)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()
    run(args.dataset, args.source, args.quality_model, args.max_workers)


if __name__ == "__main__":
    main()
