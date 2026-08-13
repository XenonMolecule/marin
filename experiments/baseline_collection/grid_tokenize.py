# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize a corpus into ``{id, input_ids}`` parquet, co-partitioned with the grid tables.

This is the missing input to ``datakit_store``. The store joins its inputs
**positionally** — it streams the tokenized parquet in lockstep with the topic and
quality tables and hard-fails on a row-count or id mismatch — so the output here
must be one file per attribute shard, same basename, same row order.

That is why this exists instead of :func:`marin.processing.tokenize.default_tokenize`,
which bundles inputs into size-balanced groups and emits a single Levanter cache.
Both are correct; only this shape can be joined.

**Special tokens come from Levanter, not from us.** Tokenizing is delegated to
:class:`levanter.data.text.BatchTokenizer` with ``enforce_bos``/``enforce_eos``, the
same path training uses. Calling an HF tokenizer directly here would silently
produce sequences that differ from every other cache in the project by a BOS token
— the registry still carries a ``nemotron_full_bos_fixed`` cache from the last time
that went wrong.

Text is **not** truncated. The classifiers capped text because a model only reads
its context window, but training consumes whole documents, so a cap here would
silently shorten the corpus.

    python -m experiments.baseline_collection.grid_tokenize \\
        --dataset dclm_10k --source mirror --max-workers 64
"""

from __future__ import annotations

import argparse
import logging
import os
import posixpath
import tempfile

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from marin.datakit.normalize import generate_id
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

from experiments.baseline_collection import grid_corpora
from experiments.baseline_collection.grid_corpora import (
    GRID_CORPORA,
    GRID_SUFFIX,
    OUTPUT_BASE,
    Stage,
    iter_records,
    list_shards,
    output_stem,
    resolve,
    stage_output_dir,
)

logger = logging.getLogger(__name__)

TOKENIZE_ROOT = "datakit/tokenize"
# Upstream's reference pipeline pins this, and marin declares it equivalent to
# meta-llama/Meta-Llama-3.1-8B, so caches built with either can share a mixture.
DEFAULT_TOKENIZER = "marin-community/marin-tokenizer"
# Docs per BatchTokenizer call. Kept modest because a batch holds full documents,
# untruncated, and resiliparse-style pages can run to megabytes.
TOKENIZE_BATCH = 512


# Workers must run where the CORPUS lives, not where COMPUTE_REGION points: this
# stage streams every document, so scheduling it in the wrong region silently
# drags the whole corpus across regions (422 GiB for resiliparse). Built per-run
# from --compute-region rather than pinned, since the corpora are not co-located.
def worker_resources(region: str, ram: str = "16g") -> ResourceConfig:
    return ResourceConfig(cpu=8, ram=ram, preemptible=True, regions=[region])


def coordinator_resources(region: str) -> ResourceConfig:
    return ResourceConfig(cpu=1, ram="3g", preemptible=False, regions=[region])


DEFAULT_MAX_WORKERS = 64

# Mirror of the tokenizer's files, so a wave of workers reads GCS instead of the
# HF Hub. The Hub caps a single actor at 1000 requests/5min; a wave of a few
# hundred workers each calling AutoTokenizer.from_pretrained at job start blows
# through that in minutes and every remaining worker fails identically (either a
# clean 429 or, worse, a partial response surfacing deep inside huggingface_hub
# as a bare AttributeError). Same "mirror once, read many" pattern as the
# corpora themselves -- see grid_mirror.py's docstring.
TOKENIZER_MIRROR_ROOT = "resources/tokenizers"

_SCHEMA = pa.schema([pa.field("id", pa.string()), pa.field("input_ids", pa.list_(pa.int32()))])


# Set by --output-dir to send a smoke run somewhere harmless. The real tree must
# never hold a partial tokenization: the store globs whatever shards it finds, so
# a half-written directory silently produces a partial store.
_OUTPUT_DIR_OVERRIDE: str | None = None

# Docs per BatchTokenizer call, overridable per run. A batch holds full untruncated
# documents, so a corpus of megabyte-scale pages needs a smaller batch than one of
# ordinary web text or the worker runs out of memory mid-shard.
_BATCH_OVERRIDE: int | None = None


def tokenize_dir(dataset: str) -> str:
    """Directory holding one corpus's tokenized shards."""
    if _OUTPUT_DIR_OVERRIDE:
        return _OUTPUT_DIR_OVERRIDE
    return f"{OUTPUT_BASE}/{TOKENIZE_ROOT}/{dataset}_{GRID_SUFFIX}"


def tokenize_output(dataset: str, shard: str) -> str:
    """Output parquet for one input shard — same basename, as the join requires."""
    return f"{tokenize_dir(dataset)}/{output_stem(shard)}.parquet"


def tokenizer_mirror_dir(name: str) -> str:
    return f"{OUTPUT_BASE}/{TOKENIZER_MIRROR_ROOT}/{name}"


def _isolate_hf_cache() -> None:
    """Point the HF cache at a fresh, process-private directory.

    Every Iris job gets ``HF_HOME=~/.cache/huggingface`` by default
    (``add_standard_env_vars`` in ``iris/cli/job.py``), and workers are long-lived
    VMs that run more than one task attempt over their lifetime, sometimes
    concurrently on spare cores. That means every process on a given worker
    shares one HF cache by default, so one process's interrupted download can
    leave a half-written cache entry that every later process on that worker then
    trips over. That reproduced exactly: two independent job submissions landed
    on the same worker VM and failed with two different symptoms (a connection
    error, then the same ``AttributeError: 'NoneType' object has no attribute
    'endswith'`` seen earlier) with zero concurrency either time — a corrupted
    on-disk cache, not a network or rate-limit problem. This must unconditionally
    override ``HF_HOME`` rather than only set it when absent, since Iris's
    default is exactly the shared path that gets poisoned.
    """
    os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="hf_home_")


def _local_tokenizer_files(name: str) -> str:
    """A local directory holding ``name``'s tokenizer files, populated from the
    GCS mirror when present, or from the Hub (with retry) when it is not.

    Falling back to the Hub here — rather than requiring the mirror to exist —
    keeps a lone smoke run working without a separate prewarm step. It is a
    per-worker race the first time a corpus tokenizes (every worker sees a
    missing mirror and downloads independently), which is exactly the failure
    this mirror exists to avoid at wave scale, so :func:`prewarm_tokenizer`
    should be run once before launching a wave of workers.
    """
    local_dir = os.path.join(tempfile.gettempdir(), "grid_tokenizer_" + name.replace("/", "_"))
    if os.path.exists(os.path.join(local_dir, "tokenizer_config.json")):
        return local_dir

    mirror = tokenizer_mirror_dir(name)
    fs = fsspec.core.url_to_fs(mirror)[0]
    if fs.exists(mirror):
        fs.get(mirror.split("://", 1)[-1] + "/", local_dir + "/", recursive=True)
        logger.info("tokenizer %s loaded from mirror %s", name, mirror)
        return local_dir

    _isolate_hf_cache()

    from rigging.timing import retry_with_backoff
    from transformers import AutoTokenizer

    hf = retry_with_backoff(
        lambda: AutoTokenizer.from_pretrained(name),
        max_attempts=8,
        operation=f"AutoTokenizer.from_pretrained({name})",
    )
    hf.save_pretrained(local_dir)
    try:
        fs.put(local_dir + "/", mirror.split("://", 1)[-1] + "/", recursive=True)
        logger.info("tokenizer %s: populated mirror %s", name, mirror)
    except Exception:
        logger.warning("tokenizer %s: failed to populate mirror %s; later workers will hit the Hub too", name, mirror)
    return local_dir


def prewarm_tokenizer(name: str) -> None:
    """Populate the GCS mirror once, before launching a wave of workers.

    Run this as its own single-task job ahead of a tokenize wave; every worker
    the wave then spawns hits GCS, not the Hub.
    """
    local_dir = _local_tokenizer_files(name)
    logger.info("tokenizer %s: mirror ready at %s (local: %s)", name, tokenizer_mirror_dir(name), local_dir)


def _cached_tokenizer(name: str):
    """One BatchTokenizer per worker process, built the way training builds it."""
    if name not in _TOKENIZER_CACHE:
        from levanter.data.text import BatchTokenizer
        from transformers import AutoTokenizer

        local_dir = _local_tokenizer_files(name)
        hf = AutoTokenizer.from_pretrained(local_dir)
        _TOKENIZER_CACHE[name] = BatchTokenizer(hf, text_field="text", enforce_bos=True, enforce_eos=True)
        logger.info("tokenizer ready: %s", name)
    return _TOKENIZER_CACHE[name]


_TOKENIZER_CACHE: dict[str, object] = {}


def _tokenize_one(path: str, corpus, batcher) -> tuple[list[str], list[list[int]]]:
    """Tokenize one shard in file order, returning ``(ids, token_arrays)``.

    A standalone function rather than a closure inside the shard loop: closing
    over per-iteration batch buffers is the classic late-binding footgun, and
    ruff rightly rejects it.
    """
    ids: list[str] = []
    token_arrays: list[list[int]] = []
    batch: list[dict] = []
    batch_ids: list[str] = []

    def flush(batch: list[dict], batch_ids: list[str]) -> None:
        if not batch:
            return
        for row, doc_id in zip(batcher(batch), batch_ids, strict=True):
            ids.append(doc_id)
            token_arrays.append(list(row["input_ids"]))

    for record in iter_records(path, corpus.format):
        text = record.get("text") or ""
        batch_ids.append(generate_id(text))
        batch.append({"text": text})
        if len(batch) >= (_BATCH_OVERRIDE or TOKENIZE_BATCH):
            flush(batch, batch_ids)
            batch, batch_ids = [], []
    flush(batch, batch_ids)
    return ids, token_arrays


def tokenize_shard(dataset: str, path: str, corpus, batcher) -> dict:
    """Tokenize one input shard and write its output. Returns a small summary.

    Deliberately a plain function taking everything it needs: the interesting
    behaviour (1:1 naming, row order, the empty-shard case) is then testable
    without standing up a Zephyr context or reaching into its internals.
    """
    ids, token_arrays = _tokenize_one(path, corpus, batcher)
    # An empty shard still gets its file: the positional join needs one output
    # per INPUT shard, and a missing file breaks alignment for every later shard
    # rather than just this one.
    table = pa.Table.from_pydict({"id": ids, "input_ids": token_arrays}, schema=_SCHEMA)
    with fsspec.open(tokenize_output(dataset, path), "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    counters.increment("grid_tokenize/shards", 1)
    counters.increment("grid_tokenize/docs", len(ids))
    counters.increment("grid_tokenize/tokens", sum(len(a) for a in token_arrays))
    return {"shard": output_stem(path), "n_docs": len(ids)}


def _tokenize_shards(dataset: str, source: str, tokenizer_name: str):
    """Build the ``map_shard`` fn: tokenize each assigned input shard 1:1."""

    def run_shard(paths, shard: ShardInfo):
        corpus = resolve(dataset, source)
        batcher = _cached_tokenizer(tokenizer_name)
        for path in paths:
            yield tokenize_shard(dataset, path, corpus, batcher)

    return run_shard


def pending(dataset: str, shards: list[str]) -> list[str]:
    """Shards without a tokenized output yet, by bulk-listing what exists."""
    from marin.utils import fsspec_glob

    directory = posixpath.dirname(tokenize_output(dataset, shards[0]))
    done = {output_stem(p) for p in fsspec_glob(f"{directory}/*.parquet")}
    todo = [s for s in shards if output_stem(s) not in done]
    logger.info("%s: %d shards, %d done, %d pending", dataset, len(shards), len(shards) - len(todo), len(todo))
    return todo


def run(
    dataset: str,
    source: str,
    tokenizer_name: str,
    max_workers: int,
    max_shards: int | None = None,
    compute_region: str = "us-central1",
    worker_ram: str = "16g",
    batch_size: int = TOKENIZE_BATCH,
) -> None:
    global _BATCH_OVERRIDE
    _BATCH_OVERRIDE = batch_size
    """Tokenize every pending shard of ``dataset`` across a Zephyr worker pool."""
    corpus = resolve(dataset, source)
    shards = list_shards(corpus)
    if max_shards is not None:
        shards = shards[:max_shards]
        logger.info("%s: limited to the first %d shards (smoke run)", dataset, len(shards))
    todo = pending(dataset, shards)
    if not todo:
        logger.info("%s: tokenization already complete", dataset)
        return

    # Sanity: the grid tables must already exist, or there is nothing to join to.
    topic_dir = stage_output_dir(dataset, Stage.TOPIC)
    logger.info("%s: tokenizing %d shards; will co-partition with %s", dataset, len(todo), topic_dir)

    pipeline = Dataset.from_list(todo).map_shard(_tokenize_shards(dataset, source, tokenizer_name))
    ctx = ZephyrContext(
        name=f"grid-tokenize-{dataset.replace('_', '-')}",
        resources=worker_resources(compute_region, worker_ram),
        coordinator_resources=coordinator_resources(compute_region),
        max_workers=max_workers,
        stage_runner_factory=InlineRunner,
    )
    result = ctx.execute(pipeline)
    logger.info("%s: done, counters=%s", dataset, dict(result.counters))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(GRID_CORPORA))
    parser.add_argument("--source", choices=["native", "mirror"], default="mirror")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write tokenized shards here instead of the canonical tree. For smoke runs.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Tokenize only the first N shards. Smoke runs only — pair with --output-dir.",
    )
    parser.add_argument(
        "--prewarm-tokenizer",
        action="store_true",
        help="Only populate the tokenizer's GCS mirror, then exit. Run once before a wave of workers.",
    )
    parser.add_argument(
        "--worker-ram",
        default="16g",
        help="Worker RAM. A batch holds full untruncated documents, so a corpus with megabyte-scale "
        "pages (resiliparse) needs more than the default.",
    )
    parser.add_argument(
        "--tokenize-batch",
        type=int,
        default=TOKENIZE_BATCH,
        help="Documents per BatchTokenizer call. Lower it for corpora with very large documents.",
    )
    parser.add_argument(
        "--compute-region",
        default="us-central1",
        help="Region to schedule workers in. Must match where the corpus lives — this stage streams "
        "every document, so a mismatch drags the whole corpus across regions.",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Bucket root for this run's outputs, e.g. gs://marin-us-east5. For a corpus outside "
        "COMPUTE_REGION; do NOT flip that constant, which relocates the other corpora too.",
    )
    args = parser.parse_args()
    # `from grid_corpora import OUTPUT_BASE` binds the value at import time, so
    # both this module's global AND grid_corpora's must be rebound or paths
    # silently resolve to the default region. The tests monkeypatch both too.
    if args.output_base:
        globals()["OUTPUT_BASE"] = args.output_base
        grid_corpora.OUTPUT_BASE = args.output_base

    if args.prewarm_tokenizer:
        prewarm_tokenizer(args.tokenizer)
        return
    if args.output_dir:
        global _OUTPUT_DIR_OVERRIDE
        _OUTPUT_DIR_OVERRIDE = args.output_dir
        logger.info("tokenize output overridden -> %s", args.output_dir)
    elif args.max_shards is not None:
        raise ValueError("--max-shards without --output-dir would leave a partial tokenization in the real tree")
    run(
        args.dataset,
        args.source,
        args.tokenizer,
        args.max_workers,
        args.max_shards,
        args.compute_region,
        args.worker_ram,
        args.tokenize_batch,
    )


if __name__ == "__main__":
    main()
