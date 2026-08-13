# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project a partial extraction run's yield onto the 24 x 5 topic x quality grid.

This is a **progress check**, not a data-mixing artifact. It answers one
question: given a random sample of WARCs from a run that is still going, how
many tokens will each (topic, quality) cell hold once the run reaches the full
10,364-WARC pool?

Why this is not :mod:`experiments.baseline_collection.grid_label`. That module
labels a *consolidated* corpus and writes per-document attribute tables under a
strict co-partitioning contract, because its output feeds a positional join in
the datakit store. Here the corpus is not consolidated: it is ~90 batch files
per WARC scattered across five regional buckets, with duplicate batches from
steal mode, and nothing downstream ever joins against it. Emitting per-document
parquet would mean shipping gigabytes across regions to answer a question that
only needs 120 integers. So this driver reuses grid_label's *scoring* — the same
WebOrganizer topic model, the same pooled quality scorer, the same calibration
and the same ``BUCKET_EDGES`` — and writes only tallies. Cells are therefore
directly comparable to ``metadata/grid_v1/{corpus}/distribution.json``.

**Work unit is one (WARC, region) group**, not one shard. Steal mode splits a
WARC's batches across regions, so a group is "the batches of WARC h that live in
region r". Each group is a few hundred thousand documents' worth of work and
writes exactly one tally, which keeps resumability cheap without producing the
~275,000 tiny output files a per-batch unit would.

**Region discipline.** A job reads only shards in its own region (the manifest is
partitioned that way) and writes only tallies, which are kilobytes. No batch data
ever crosses a region boundary.

**Token accounting.** Three masses are tallied per cell, because none alone
answers the question:

* ``docs`` — document count.
* ``chars`` — full ``len(text)``, counted *before* the model input cap, so it is
  the true size of the document rather than the size of what the model read.
* ``gte_tokens`` — the topic tokenizer's length, which is what grid_v1 reported.
  It is capped at ``max_length``, so it undercounts the long tail and must not be
  used as a training-token estimate.
* ``sampled_chars`` / ``sampled_llama_tokens`` — a systematic 1-in-N subsample
  run through the **llama3** tokenizer, the one the curation baselines actually
  tokenize with. Their ratio converts ``chars`` into training tokens per cell,
  which is the number the projection reports.

    python -m experiments.baseline_collection.grid_projection \\
        --manifest gs://marin-us-east1/metadata/lpv1_1_projection/manifest_us-east1.jsonl \\
        --num-chunks 8 --chunk-idx 0 \\
        --quality-model gs://marin-us-central1/resources/datakit/quality/pooled_junkgate2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np

from experiments.baseline_collection.grid_corpora import (
    DEFAULT_MAX_LENGTH,
    LABEL_TEXT_CHARS,
    NUM_TOPICS,
    Format,
    ShardDocs,
    iter_records,
)
from experiments.baseline_collection.grid_label import build_mesh, load_quality, load_topic, score_quality, score_topic
from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.fsspec_paths import fsspec_glob
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

NUM_BUCKETS = len(BUCKET_EDGES) + 1
# Documents scored per model call. Batches on disk hold ~250 records, which is
# far too small to keep a TPU busy, so records are buffered across batch files
# until this many have accumulated.
SCORE_CHUNK = 4096
# Systematic subsample sent through the llama3 tokenizer for the chars -> tokens
# ratio. At ~14M documents this still yields ~70,000 sampled documents, enough
# for a stable per-topic ratio, while costing well under a percent of runtime.
LLAMA_SAMPLE_EVERY = 200
# Concurrent object-storage fetches, and how many decoded shards may sit ahead of
# the consumer. Reading is the bottleneck by two orders of magnitude, so this is
# the single number that sets throughput.
READ_PARALLELISM = 32
READ_PREFETCH = 48
# Tallies are kilobytes, so they all land in one region regardless of where the
# job ran; the merge is then a single local read instead of a five-region fan-out.
TALLY_BASE = "gs://marin-us-central1/metadata/lpv1_1_projection"


def tally_path(out_base: str, warc: str, region: str) -> str:
    return f"{out_base}/tallies/{warc}.{region}.json"


def load_manifest(path: str) -> list[dict]:
    """Read the JSONL group manifest: one ``{warc, region, shards}`` per line."""
    with fsspec.open(path, "rt") as fh:
        groups = [json.loads(line) for line in fh if line.strip()]
    if not groups:
        raise ValueError(f"manifest {path} is empty")
    return groups


def pending_groups(out_base: str, groups: list[dict]) -> list[dict]:
    """Groups with no tally yet, found with one bulk list rather than N exists()."""
    done = {path.rsplit("/", 1)[-1] for path in fsspec_glob(f"{out_base}/tallies/*.json")}
    pending = [g for g in groups if f"{g['warc']}.{g['region']}.json" not in done]
    logger.info("%d groups, %d already tallied, %d pending", len(groups), len(groups) - len(pending), len(pending))
    return pending


class GridTally:
    """Running 24 x 5 accumulators for one group.

    Kept as a small class rather than loose arrays because the four masses must
    stay index-aligned and are always updated together.
    """

    def __init__(self) -> None:
        shape = (NUM_TOPICS, NUM_BUCKETS)
        self.docs = np.zeros(shape, dtype=np.int64)
        self.chars = np.zeros(shape, dtype=np.int64)
        self.gte_tokens = np.zeros(shape, dtype=np.int64)
        self.sampled_docs = np.zeros(shape, dtype=np.int64)
        self.sampled_chars = np.zeros(shape, dtype=np.int64)
        self.sampled_llama_tokens = np.zeros(shape, dtype=np.int64)

    def add(self, topics, buckets, chars, gte_lengths) -> None:
        flat = topics.astype(np.int64) * NUM_BUCKETS + buckets.astype(np.int64)
        size = NUM_TOPICS * NUM_BUCKETS
        self.docs += np.bincount(flat, minlength=size).reshape(self.docs.shape)
        self.chars += np.bincount(flat, weights=chars, minlength=size).astype(np.int64).reshape(self.chars.shape)
        self.gte_tokens += (
            np.bincount(flat, weights=gte_lengths, minlength=size).astype(np.int64).reshape(self.gte_tokens.shape)
        )

    def add_sampled(self, topics, buckets, chars, llama_lengths) -> None:
        flat = topics.astype(np.int64) * NUM_BUCKETS + buckets.astype(np.int64)
        size = NUM_TOPICS * NUM_BUCKETS
        self.sampled_docs += np.bincount(flat, minlength=size).reshape(self.docs.shape)
        self.sampled_chars += np.bincount(flat, weights=chars, minlength=size).astype(np.int64).reshape(self.docs.shape)
        self.sampled_llama_tokens += (
            np.bincount(flat, weights=llama_lengths, minlength=size).astype(np.int64).reshape(self.docs.shape)
        )

    def payload(self, warc: str, region: str, num_shards: int, label_names: list[str]) -> dict:
        return {
            "warc": warc,
            "region": region,
            "num_shards": num_shards,
            "topics": label_names,
            "n_docs": int(self.docs.sum()),
            "docs": self.docs.tolist(),
            "chars": self.chars.tolist(),
            "gte_tokens": self.gte_tokens.tolist(),
            "sampled_docs": self.sampled_docs.tolist(),
            "sampled_chars": self.sampled_chars.tolist(),
            "sampled_llama_tokens": self.sampled_llama_tokens.tolist(),
        }


def iter_buffered(shards: list[str], size: int):
    """Yield ``(urls, texts, full_char_lengths, sample_flags)`` buffers of ``size`` docs.

    Buffering across batch files is what makes the TPU worth using: a single
    on-disk batch holds only ~48 surviving records, so scoring per file would
    leave the chips idle waiting on object storage.

    Shards are fetched **concurrently and consumed in order**. This job is
    overwhelmingly I/O-bound, not compute-bound — a group is ~76 objects of
    ~150 KB each, and reading them one at a time held a v5p-8 at roughly 1% of
    its FLOPs. Prefetching is therefore the whole ballgame; the window is capped
    so a group's resident set stays a few tens of megabytes.

    ``full_char_lengths`` is measured before the model-input cap, so a document
    longer than ``LABEL_TEXT_CHARS`` still contributes its true size to the token
    projection even though the classifiers only read the first 64k characters.
    """
    urls: list[str] = []
    texts: list[str] = []
    lengths: list[int] = []
    flags: list[bool] = []
    seen = 0
    with ThreadPoolExecutor(max_workers=READ_PARALLELISM) as pool:
        pending: deque = deque()
        remaining = list(shards)
        while remaining or pending:
            while remaining and len(pending) < READ_PREFETCH:
                pending.append(pool.submit(lambda s=remaining.pop(0): list(iter_records(s, Format.JSONL_GZ))))
            for record in pending.popleft().result():
                text = record.get("text") or ""
                urls.append(record.get("url") or "")
                texts.append(text[:LABEL_TEXT_CHARS])
                lengths.append(len(text))
                flags.append(seen % LLAMA_SAMPLE_EVERY == 0)
                seen += 1
                if len(urls) >= size:
                    yield urls, texts, lengths, flags
                    urls, texts, lengths, flags = [], [], [], []
    if urls:
        yield urls, texts, lengths, flags


def score_group(
    shards: list[str],
    topic_model,
    scorer,
    llama_tok,
    max_length: int,
    mesh,
) -> GridTally:
    """Score every document of one group and fold it into a tally."""
    params, tokenizer, _ = topic_model
    tally = GridTally()
    for urls, texts, char_lengths, flags in iter_buffered(shards, SCORE_CHUNK):
        # No content hashes: nothing downstream joins against these tallies, and
        # hashing every document would add a full pass over ~40 GB of text for a
        # column that is never read. The id slots are still filled so ``len()``
        # on the container stays truthful.
        docs = ShardDocs(ids=[""] * len(urls), native_ids=[None] * len(urls), urls=urls, texts=texts)
        topics, _, gte_lengths = score_topic(docs, params, tokenizer, max_length, mesh)
        _, buckets = score_quality(scorer, texts)
        chars = np.asarray(char_lengths, dtype=np.float64)
        tally.add(topics, buckets, chars, np.asarray(gte_lengths, dtype=np.float64))

        picked = np.flatnonzero(np.asarray(flags))
        if len(picked):
            # `add_special_tokens=False` because the projection is about content
            # mass; the BOS a packer adds is per-document overhead accounted for
            # elsewhere, and including it would inflate short documents most.
            encoded = llama_tok([texts[i] for i in picked], add_special_tokens=False)["input_ids"]
            tally.add_sampled(
                topics[picked],
                buckets[picked],
                chars[picked],
                np.asarray([len(ids) for ids in encoded], dtype=np.float64),
            )
    return tally


def run(
    manifest: str,
    num_chunks: int,
    chunk_idx: int,
    quality_model: str,
    out_base: str,
    max_length: int,
    mesh_scope: str,
) -> None:
    groups = load_manifest(manifest)
    # Partition FIRST, then drop what is already tallied. Doing it the other way
    # round makes ownership depend on how much was finished at startup, so a job
    # restarting after preemption would claim a different slice than it held
    # before and groups could fall between the two views. Partitioning the full
    # manifest gives every group exactly one permanent owner, and a preempted
    # job resumes precisely where it stopped.
    mine = pending_groups(out_base, groups[chunk_idx::num_chunks])
    if not mine:
        logger.info("chunk %d/%d has nothing pending", chunk_idx, num_chunks)
        return
    logger.info("chunk %d/%d owns %d groups", chunk_idx, num_chunks, len(mine))

    mesh = build_mesh(mesh_scope)
    topic_model = load_topic(max_length, mesh)
    label_names = topic_model[2]
    scorer = load_quality(quality_model)

    from transformers import AutoTokenizer

    llama_tok = AutoTokenizer.from_pretrained(llama3_tokenizer)
    logger.info("llama3 tokenizer ready: %s", llama3_tokenizer)

    started = time.monotonic()
    total_docs = 0
    for done, group in enumerate(mine, start=1):
        tally = score_group(group["shards"], topic_model, scorer, llama_tok, max_length, mesh)
        with fsspec.open(tally_path(out_base, group["warc"], group["region"]), "w") as fh:
            json.dump(tally.payload(group["warc"], group["region"], len(group["shards"]), label_names), fh)
        total_docs += int(tally.docs.sum())
        elapsed = time.monotonic() - started
        logger.info("%d/%d groups, %d docs, %.0f docs/s", done, len(mine), total_docs, total_docs / max(elapsed, 1e-9))
    logger.info(
        "chunk %d DONE: %d groups, %d docs in %.0fs", chunk_idx, len(mine), total_docs, time.monotonic() - started
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL group manifest for THIS region")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--quality-model", required=True)
    parser.add_argument("--out-base", default=TALLY_BASE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--mesh-scope", choices=["local", "global"], default="local")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(
        args.manifest,
        args.num_chunks,
        args.chunk_idx,
        args.quality_model,
        args.out_base.rstrip("/"),
        args.max_length,
        args.mesh_scope,
    )


if __name__ == "__main__":
    main()
