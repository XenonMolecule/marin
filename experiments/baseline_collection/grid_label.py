# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Label a corpus with WebOrganizer topic (24-way) and datakit quality (5 buckets).

The two stages are **independently selectable and independently resumable**, so
the same code covers three situations without a fork:

  * ``--stages topic``            — run tonight; the quality model is not needed
  * ``--stages quality``          — catch up later, on CPU, once the model lands
  * ``--stages topic,quality``    — the fused run: read each shard ONCE

Fused is the cheaper path when both models are available, because reading the
text dominates: the topic model is 137M params at 8192 tokens while the quality
model is 3M params at 512, so the second stage is nearly free *if* it rides along
on a read the first stage already paid for. Running them separately costs a
second full pass over the corpus but is otherwise identical — same outputs, same
basenames, same row order — because each stage owns its own output tree and its
own per-shard done markers. Nothing about a topic-only run has to be redone when
quality arrives.

**The co-partitioning contract.** One input shard produces exactly one output file
per stage, same basename, same row order, no shard splitting or merging. The
datakit store joins tokenize/decon/cluster/quality *positionally* and hard-fails
on a row-count or id mismatch, so this is load-bearing rather than tidy.

Rows carry the datakit content-hash ``id``, which is what lets a label set
computed on one layer attach to any pre/post dedup/decon variant of the same
corpus: both stages drop whole documents without mutating text.

    # topic, on TPU, in the compute region against the mirror
    python -m experiments.baseline_collection.grid_label label \\
        --dataset dclm_10k --stages topic --num-chunks 16 --chunk-idx 0

    # quality catch-up, CPU is fine
    python -m experiments.baseline_collection.grid_label label \\
        --dataset dclm_10k --stages quality --quality-model gs://.../pooled_junkgate2

    # merge per-shard tallies into the 24 x 5 grid.
    # Run this IN-REGION as an Iris job, not from a laptop: gcsfs cannot complete
    # the TLS handshake to storage.googleapis.com from the dev machine (the
    # `gcloud` CLI can, gcsfs cannot), so a local merge dies on
    # SSLCertVerificationError. In-region is the right place for it regardless —
    # it reads every shard's tally and, once quality has run, both attribute
    # tables in full.
    python -m experiments.baseline_collection.grid_label merge --dataset dclm_10k
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.baseline_collection import grid_corpora
from experiments.baseline_collection.grid_corpora import (
    DEFAULT_MAX_LENGTH,
    GRID_CORPORA,
    LABEL_TEXT_CHARS,
    METADATA_ROOT,
    NUM_TOPICS,
    ShardDocs,
    Stage,
    assign_shards,
    counts_output,
    list_shards,
    output_stem,
    pending_shards,
    quality_samples_output,
    read_shard,
    resolve,
    stage_output_dir,
    stage_shard_output,
    write_done,
)
from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

# Fraction of docs whose text is kept alongside its scores, for the stage report
# and for eyeballing cells. Upstream uses the same rate.
SAMPLE_PCT = 0.02
SAMPLE_TEXT_CHARS = 4_000
# Docs per score_bme call. Each doc becomes up to 3 windows of 512 tokens, so
# this is ~1536 sequences — comfortably inside the quality model's batch budget.
QUALITY_BATCH = 512
# Calibration artifact name inside the quality model dir. Separate from the
# model and NOT loaded by load_pooled_scorer.
QUALITY_CALIB_FILE = "calib_bme.json"
# Merge-time fan-out for object-storage reads. The merge is pure I/O latency, and
# nemotron's 24,390 shards make serial reads the whole runtime.
READ_PARALLELISM = 32


def parse_stages(raw: str) -> list[Stage]:
    """Parse a comma-separated stage list, preserving a deterministic order.

    Topic is ordered first so a fused run does the expensive stage while the
    shard is freshly read, and a crash mid-shard leaves the cheap stage to redo.
    """
    requested = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = requested - {stage.value for stage in Stage}
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}; expected any of {[s.value for s in Stage]}")
    if not requested:
        raise ValueError("--stages must name at least one stage")
    return [stage for stage in (Stage.TOPIC, Stage.QUALITY) if stage.value in requested]


def build_mesh(scope: str) -> Mesh:
    """Data-parallel mesh over this process's devices.

    ``local`` builds the mesh from ``jax.local_devices()`` and pairs with
    process-disjoint shard assignment, so each host of a multi-host slice runs as
    an independent worker and no cross-host collective is ever issued. ``global``
    reproduces the single-host behaviour of the original topic runs.

    The proven WebOrganizer runs were single-host v6e-4, where the two are the
    same thing. On a v5p slice (4 chips/host, so every slice type is multi-host)
    they are not, which is why this is a flag and why it gets validated on one
    small slice before any fan-out.
    """
    devices = jax.local_devices() if scope == "local" else jax.devices()
    return Mesh(np.array(devices), ("data",))


def process_partition(shards: list[str], scope: str) -> list[str]:
    """This process's share of ``shards`` under a ``local`` mesh.

    Under a ``global`` mesh every process cooperates on one batch stream, so the
    full list is returned unchanged.
    """
    if scope != "local" or jax.process_count() == 1:
        return shards
    mine = shards[jax.process_index() :: jax.process_count()]
    logger.info(
        "process %d/%d owns %d of %d shards",
        jax.process_index(),
        jax.process_count(),
        len(mine),
        len(shards),
    )
    return mine


def systematic_take(index: int, pct: float) -> bool:
    """Deterministic every-1/pct-th selector, matching upstream's sampler.

    Position-based rather than hash-based so the sample is reproducible and
    evenly spread through the shard rather than clumped.
    """
    return int((index + 1) * pct) > int(index * pct)


def load_topic(max_length: int, mesh: Mesh):
    """Load WebOrganizer's topic classifier onto the mesh.

    The heavy imports are deferred to here, not hoisted to module scope, because
    each stage's dependency stack is optional to the other: this pulls torch and
    transformers, which a quality-only catch-up run on a lean CPU worker has no
    reason to require. The reverse holds for the quality scorer's equinox.

    Returns:
        ``(params, tokenizer, label_names)``. Params are replicated across the
        mesh once (the model is 137M, it fits anywhere) so each sharded batch
        finds them locally instead of being broadcast per call.
    """
    from experiments.baseline_collection.weborganizer_gte_jax import extract_params, to_device
    from experiments.baseline_collection.weborganizer_topic_smoke import URL_MODEL, load_model

    config, tokenizer, hf_model = load_model(URL_MODEL)
    label_names = [config.id2label[i] for i in range(config.num_labels)]
    if config.num_labels != NUM_TOPICS:
        raise ValueError(f"expected {NUM_TOPICS} topics, checkpoint has {config.num_labels}")
    params = to_device(extract_params(hf_model), jnp.bfloat16)  # the reference also runs bf16
    del hf_model
    params = jax.device_put(params, NamedSharding(mesh, P()))
    logger.info("topic model ready: %s, max_length=%d, %d labels", URL_MODEL, max_length, len(label_names))
    return params, tokenizer, label_names


def score_topic(docs: ShardDocs, params, tokenizer, max_length: int, mesh: Mesh):
    """Argmax topic, its probability, and token length for every doc in a shard.

    Returns:
        ``(topics, probs, token_lengths)`` as numpy arrays in input row order.
    """
    from experiments.baseline_collection.weborganizer_topic_label import score_bucketed
    from experiments.baseline_collection.weborganizer_topic_smoke import Doc, _render, _softmax

    pages = _render([Doc(url=url, text=text) for url, text in zip(docs.urls, docs.texts, strict=True)], True)
    logits, token_lengths = score_bucketed(params, tokenizer, pages, max_length, mesh, None)
    probs = _softmax(logits)
    return logits.argmax(-1), probs.max(-1), token_lengths


def load_quality(model_dir: str):
    """Load the pooled scorer AND its calibration. Deferred import; see :func:`load_topic`.

    The calibration is a separate artifact from the model and ``load_pooled_scorer``
    does not read it — upstream loads ``calib_bme.json`` alongside. It is not
    optional: ``BUCKET_EDGES`` are cutpoints in *calibrated* space, and the whole
    reason this model was chosen over fastText is that calibration is what makes
    those fixed edges mean the same quality level across content types. Digitizing
    a raw score against them silently mis-buckets, worst at the tails.

    Returns:
        ``(scorer, xk, yk)`` — the piecewise-linear knots map raw to calibrated.
    """
    from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer

    scorer = load_pooled_scorer(model_dir)
    with fsspec.open(f"{model_dir.rstrip('/')}/{QUALITY_CALIB_FILE}") as fh:
        calib = json.load(fh)
    logger.info("quality calibration loaded: xk=%s", calib["xk"])
    return scorer, np.asarray(calib["xk"], dtype=np.float64), np.asarray(calib["yk"], dtype=np.float64)


def score_quality(scorer, texts: list[str]):
    """Calibrated quality score and disjoint bucket for every doc in a shard.

    Returns:
        ``(scores, buckets)`` — ``scores`` calibrated into [0, 1] via the monotonic
        remap, ``buckets`` its digitization against ``BUCKET_EDGES`` giving 0..4.
    """
    from experiments.datakit.cluster.quality.fast_transformer.scorer import score_bme

    model, xk, yk = scorer
    raw = np.concatenate(
        [score_bme(model, texts[start : start + QUALITY_BATCH]) for start in range(0, len(texts), QUALITY_BATCH)]
    )
    calibrated = np.interp(raw, xk, yk)
    return calibrated, np.digitize(calibrated, BUCKET_EDGES).astype(np.int8)


def write_topic_shard(dataset: str, shard: str, docs: ShardDocs, topics, probs, lengths, label_names) -> None:
    """Write the topic attribute table and per-shard tally for one shard."""
    table = pa.table(
        {
            "id": docs.ids,
            "native_id": docs.native_ids,
            "url": docs.urls,
            f"cluster_{NUM_TOPICS}": pa.array(topics, type=pa.int32()),
            "topic_prob": pa.array(probs, type=pa.float32()),
            "token_length": pa.array(lengths, type=pa.int32()),
        }
    )
    with fsspec.open(stage_shard_output(dataset, Stage.TOPIC, shard), "wb") as fh:
        pq.write_table(table, fh, compression="zstd")

    counts = np.bincount(topics, minlength=NUM_TOPICS)
    # Token mass, not just doc counts: training consumes tokens and doc length
    # varies a lot BY topic, so doc share is not a stand-in for the training mix.
    token_counts = np.bincount(topics, weights=lengths, minlength=NUM_TOPICS)
    payload = {
        "dataset": dataset,
        "stem": output_stem(shard),
        "n_docs": len(docs),
        "n_tokens": int(lengths.sum()),
        "counts": {label_names[i]: int(counts[i]) for i in range(NUM_TOPICS)},
        "token_counts": {label_names[i]: int(token_counts[i]) for i in range(NUM_TOPICS)},
    }
    with fsspec.open(counts_output(dataset, Stage.TOPIC, shard), "w") as fh:
        json.dump(payload, fh)


def write_quality_shard(dataset: str, shard: str, docs: ShardDocs, scores, buckets) -> None:
    """Write the quality attribute table, text sample, and tally for one shard.

    The main table's schema is exactly upstream's ``(source, id, score,
    quality_bucket)`` so a ``QualityScores`` artifact can point straight at this
    tree without translation.
    """
    table = pa.table(
        {
            "source": pa.array([dataset] * len(docs), type=pa.string()),
            "id": docs.ids,
            "score": pa.array(scores, type=pa.float32()),
            "quality_bucket": pa.array(buckets, type=pa.int8()),
        }
    )
    with fsspec.open(stage_shard_output(dataset, Stage.QUALITY, shard), "wb") as fh:
        pq.write_table(table, fh, compression="zstd")

    picked = [i for i in range(len(docs)) if systematic_take(i, SAMPLE_PCT)]
    sample = pa.table(
        {
            "source": pa.array([dataset] * len(picked), type=pa.string()),
            "id": [docs.ids[i] for i in picked],
            "score": pa.array([scores[i] for i in picked], type=pa.float32()),
            "quality_bucket": pa.array([buckets[i] for i in picked], type=pa.int8()),
            "text": [docs.texts[i][:SAMPLE_TEXT_CHARS] for i in picked],
        }
    )
    with fsspec.open(quality_samples_output(dataset, shard), "wb") as fh:
        pq.write_table(sample, fh, compression="zstd")

    bucket_counts = np.bincount(buckets, minlength=len(BUCKET_EDGES) + 1)
    payload = {
        "dataset": dataset,
        "stem": output_stem(shard),
        "n_docs": len(docs),
        "n_sampled": len(picked),
        "score_sum": float(scores.sum()),
        "bucket_counts": [int(n) for n in bucket_counts],
    }
    with fsspec.open(counts_output(dataset, Stage.QUALITY, shard), "w") as fh:
        json.dump(payload, fh)


def run_label(
    dataset: str,
    stages: list[Stage],
    source: str,
    num_chunks: int,
    chunk_idx: int,
    max_length: int,
    quality_model: str | None,
    mesh_scope: str,
    max_shards: int | None = None,
) -> None:
    """Score this chunk's pending shards for every requested stage."""
    corpus = resolve(dataset, source)
    logger.info(
        "%s (%s, %s, post_decon=%s): stages=%s",
        dataset,
        corpus.region,
        corpus.path,
        corpus.post_decon,
        [s.value for s in stages],
    )

    all_shards = assign_shards(list_shards(corpus), num_chunks, chunk_idx)
    todo = {stage: set(pending_shards(dataset, stage, all_shards)) for stage in stages}
    shards = process_partition([s for s in all_shards if any(s in todo[stage] for stage in stages)], mesh_scope)
    if max_shards is not None:
        shards = shards[:max_shards]
        logger.warning("SMOKE RUN: capped at %d shards — this output is partial, do not merge it", len(shards))
    if not shards:
        logger.info("nothing pending for chunk %d — all requested stages already done", chunk_idx)
        return

    mesh = build_mesh(mesh_scope)
    logger.info("mesh: %d device(s) %s", mesh.devices.size, mesh.devices.flatten().tolist())

    params = tokenizer = label_names = None
    if Stage.TOPIC in stages:
        params, tokenizer, label_names = load_topic(max_length, mesh)

    scorer = None
    if Stage.QUALITY in stages:
        if not quality_model:
            raise ValueError("--quality-model is required when the quality stage is requested")
        scorer = load_quality(quality_model)
        logger.info("quality model ready: %s, edges=%s", quality_model, BUCKET_EDGES)

    started = time.monotonic()
    n_docs = 0
    for done, shard in enumerate(shards, start=1):
        docs = read_shard(shard, corpus, LABEL_TEXT_CHARS)

        if not len(docs):
            # An empty shard is legitimate for a filtered corpus — it is a WARC
            # from which nothing survived the upstream filter (7 of fineweb_edu's
            # 1,469 shards are 20-byte empty gzips). Emit the empty tables rather
            # than skipping or failing: the co-partitioning contract is one output
            # file per INPUT shard, so a downstream positional join needs a
            # zero-row file here, and a missing one would break the alignment.
            logger.info("shard has 0 records, writing empty outputs: %s", shard)
            empty_topics = np.array([], dtype=np.int64)
            empty_probs = np.array([], dtype=np.float32)
            empty_lengths = np.array([], dtype=np.int64)
            if Stage.TOPIC in stages and shard in todo[Stage.TOPIC]:
                write_topic_shard(dataset, shard, docs, empty_topics, empty_probs, empty_lengths, label_names)
                write_done(dataset, Stage.TOPIC, shard, 0)
            if Stage.QUALITY in stages and shard in todo[Stage.QUALITY]:
                write_quality_shard(dataset, shard, docs, empty_probs, np.array([], dtype=np.int8))
                write_done(dataset, Stage.QUALITY, shard, 0)
            continue

        if Stage.TOPIC in stages and shard in todo[Stage.TOPIC]:
            topics, probs, lengths = score_topic(docs, params, tokenizer, max_length, mesh)
            write_topic_shard(dataset, shard, docs, topics, probs, lengths, label_names)
            write_done(dataset, Stage.TOPIC, shard, len(docs))

        if Stage.QUALITY in stages and shard in todo[Stage.QUALITY]:
            scores, buckets = score_quality(scorer, docs.texts)
            write_quality_shard(dataset, shard, docs, scores, buckets)
            write_done(dataset, Stage.QUALITY, shard, len(docs))

        n_docs += len(docs)
        elapsed = time.monotonic() - started
        if done % 10 == 0 or done == len(shards):
            logger.info(
                "  %d/%d shards, %d docs, %.1f docs/s (%.2f docs/s/chip)",
                done,
                len(shards),
                n_docs,
                n_docs / elapsed,
                n_docs / elapsed / mesh.devices.size,
            )

    logger.info("chunk %d DONE: %d shards, %d docs in %.0fs", chunk_idx, len(shards), n_docs, time.monotonic() - started)


def _read_json(path: str) -> dict:
    with fsspec.open(path) as fh:
        return json.load(fh)


def _load_counts(dataset: str, stage: Stage) -> list[dict]:
    """Every per-shard tally written for ``stage``.

    Read in parallel because shard counts vary by three orders of magnitude
    across corpora: dclm has 100 tallies but nemotron has 24,390, and serial
    round-trips to object storage make the latter dominate the merge entirely.
    """
    paths = sorted(fsspec_glob(f"{stage_output_dir(dataset, stage)}/counts/*.json"))
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=READ_PARALLELISM) as pool:
        return list(pool.map(_read_json, paths))


def _grid_from_tables(dataset: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Cross-tab topic x quality by joining the two attribute tables on ``id``.

    Returns ``None`` when the quality stage has not run, so a topic-only corpus
    still merges into a usable 24-way distribution instead of failing.
    """
    topic_dir = stage_output_dir(dataset, Stage.TOPIC)
    quality_dir = f"{stage_output_dir(dataset, Stage.QUALITY)}/outputs/main"
    quality_paths = {output_stem(p): p for p in fsspec_glob(f"{quality_dir}/*.parquet")}
    if not quality_paths:
        return None

    n_buckets = len(BUCKET_EDGES) + 1
    docs_grid = np.zeros((NUM_TOPICS, n_buckets), dtype=np.int64)
    token_grid = np.zeros((NUM_TOPICS, n_buckets), dtype=np.int64)
    pairs = [
        (p, quality_paths[output_stem(p)])
        for p in sorted(fsspec_glob(f"{topic_dir}/*.parquet"))
        if output_stem(p) in quality_paths
    ]

    # Two object-storage reads per shard, and nemotron has 24,390 shards — read
    # them concurrently and accumulate serially. Accumulation is trivial next to
    # the I/O, so keeping it single-threaded avoids needing a lock on the grids.
    with ThreadPoolExecutor(max_workers=READ_PARALLELISM) as pool:
        for stem, topics, buckets, lengths in pool.map(_read_shard_pair, pairs):
            del stem
            np.add.at(docs_grid, (topics, buckets), 1)
            np.add.at(token_grid, (topics, buckets), lengths)
    return docs_grid, token_grid


def _read_shard_pair(paths: tuple[str, str]):
    """Read one shard's topic + quality tables, asserting they are co-partitioned.

    The equal-row-count and identical-id checks are the same invariant
    ``datakit_store`` enforces on its positional 5-way join; catching a violation
    here names the offending shard instead of surfacing as a silent mis-alignment.
    """
    topic_path, quality_path = paths
    stem = output_stem(topic_path)
    with fsspec.open(topic_path, "rb") as fh:
        topic_table = pq.read_table(fh, columns=["id", f"cluster_{NUM_TOPICS}", "token_length"])
    with fsspec.open(quality_path, "rb") as fh:
        quality_table = pq.read_table(fh, columns=["id", "quality_bucket"])
    if topic_table.num_rows != quality_table.num_rows:
        raise ValueError(
            f"co-partitioning broken for {stem}: topic has {topic_table.num_rows} rows, "
            f"quality has {quality_table.num_rows}"
        )
    topic_ids = topic_table.column("id").to_numpy(zero_copy_only=False)
    quality_ids = quality_table.column("id").to_numpy(zero_copy_only=False)
    if not np.array_equal(topic_ids, quality_ids):
        raise ValueError(f"co-partitioning broken for {stem}: ids differ between topic and quality tables")
    return (
        stem,
        topic_table.column(f"cluster_{NUM_TOPICS}").to_numpy(),
        quality_table.column("quality_bucket").to_numpy(),
        topic_table.column("token_length").to_numpy(),
    )


def run_merge(dataset: str) -> None:
    """Merge per-shard tallies into the distribution, plus the grid if available."""
    topic_counts = _load_counts(dataset, Stage.TOPIC)
    quality_counts = _load_counts(dataset, Stage.QUALITY)
    if not topic_counts and not quality_counts:
        raise RuntimeError(f"no per-shard counts found for {dataset} — has anything run?")

    summary: dict = {"dataset": dataset, "post_decon": GRID_CORPORA[dataset].post_decon}

    if topic_counts:
        totals: dict[str, int] = {}
        token_totals: dict[str, int] = {}
        for payload in topic_counts:
            for label, count in payload["counts"].items():
                totals[label] = totals.get(label, 0) + count
            for label, count in payload["token_counts"].items():
                token_totals[label] = token_totals.get(label, 0) + count
        n_docs = sum(p["n_docs"] for p in topic_counts)
        n_tokens = sum(p["n_tokens"] for p in topic_counts)
        summary["topic"] = {
            "n_shards": len(topic_counts),
            "n_docs": n_docs,
            "n_tokens": n_tokens,
            "counts": totals,
            "shares": {label: count / n_docs for label, count in totals.items()},
            "token_counts": token_totals,
            "token_shares": {label: count / n_tokens for label, count in token_totals.items()} if n_tokens else {},
        }
        for label in sorted(totals, key=lambda x: -totals[x]):
            logger.info("  %-22s %10d  (%5.2f%%)", label, totals[label], 100 * totals[label] / n_docs)

    if quality_counts:
        n_docs = sum(p["n_docs"] for p in quality_counts)
        bucket_totals = np.zeros(len(BUCKET_EDGES) + 1, dtype=np.int64)
        for payload in quality_counts:
            bucket_totals += np.array(payload["bucket_counts"], dtype=np.int64)
        summary["quality"] = {
            "n_shards": len(quality_counts),
            "n_docs": n_docs,
            "bucket_edges": list(BUCKET_EDGES),
            "bucket_counts": bucket_totals.tolist(),
            "bucket_shares": (bucket_totals / n_docs).tolist(),
            "score_mean": sum(p["score_sum"] for p in quality_counts) / n_docs,
        }
        logger.info("  quality buckets: %s", bucket_totals.tolist())

    # The grid can only cover shards present in BOTH tables, while the per-stage
    # totals above cover each stage's shards independently. Merging while quality
    # is still running therefore yields one file whose topic totals and grid
    # disagree — which reads as a real finding rather than an artefact. Warn loudly
    # rather than silently emitting it.
    if topic_counts and quality_counts and len(topic_counts) != len(quality_counts):
        logger.warning(
            "STAGE SKEW: %d topic shards vs %d quality shards. The grid below covers only "
            "the %d shards both stages finished, so it is NOT comparable to the per-stage "
            "totals. Re-merge once both stages are complete.",
            len(topic_counts),
            len(quality_counts),
            min(len(topic_counts), len(quality_counts)),
        )
        summary["stage_skew"] = {"topic_shards": len(topic_counts), "quality_shards": len(quality_counts)}

    grid = _grid_from_tables(dataset)
    if grid is not None:
        docs_grid, token_grid = grid
        summary["grid"] = {
            "axes": {"topic": f"cluster_{NUM_TOPICS}", "quality": "quality_bucket"},
            "docs": docs_grid.tolist(),
            "tokens": token_grid.tolist(),
        }
        logger.info("  grid: %d x %d cells, %d docs", NUM_TOPICS, len(BUCKET_EDGES) + 1, int(docs_grid.sum()))

    out_path = f"{grid_corpora.OUTPUT_BASE}/{METADATA_ROOT}/{dataset}/distribution.json"
    with fsspec.open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("merged -> %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["label", "merge"])
    parser.add_argument("--dataset", required=True, choices=list(GRID_CORPORA))
    parser.add_argument(
        "--stages",
        default="topic,quality",
        help="Comma-separated: topic, quality, or both. Each is independently resumable.",
    )
    parser.add_argument("--source", choices=["native", "mirror"], default="mirror")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--quality-model", default=None, help="Directory holding the pooled scorer artifact.")
    parser.add_argument(
        "--mesh",
        choices=["local", "global"],
        default="local",
        help="Devices in the data-parallel mesh: this process's, or the whole slice's.",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Override the output root. Use a scratch path to validate a run without touching the real tree.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Stop after this many shards. For smoke runs only; a partial corpus must never be merged.",
    )
    args = parser.parse_args()

    if args.output_base:
        grid_corpora.OUTPUT_BASE = args.output_base
        logger.info("output base overridden -> %s", args.output_base)

    if args.command == "merge":
        run_merge(args.dataset)
        return
    if not 0 <= args.chunk_idx < args.num_chunks:
        raise ValueError(f"--chunk-idx {args.chunk_idx} out of range for --num-chunks {args.num_chunks}")
    run_label(
        args.dataset,
        parse_stages(args.stages),
        args.source,
        args.num_chunks,
        args.chunk_idx,
        args.max_length,
        args.quality_model,
        args.mesh,
        args.max_shards,
    )


if __name__ == "__main__":
    main()
