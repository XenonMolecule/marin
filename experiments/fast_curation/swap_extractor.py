# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Arm B of the extractor ablation: arm A's documents, labels and mix with upstream resiliparse text.

Arm A is ``lpv11_fastpipe_v2`` (fork extractor, resiliparse-rs markdown) after dedup, decon and
24x5 grid labelling. Arm B keeps A's document set, A's topic/quality labels and A's fitted mixture
verbatim and regenerates only ``text``: each WARC is refetched from Common Crawl, decoded with the
cascade's own decoder (so the html is byte-identical to what the fork saw), and the documents in the
WARC's allow-list (``build_swap_allowlist.py``) are extracted with the upstream Cython resiliparse
call the ``resiliparse_10k`` baseline used. Nothing else runs: no dedup, no decon, no classifiers.

Per WARC the worker writes

* ``{root}/deduped/data-{warc_hash}.jsonl.gz`` -- arm A's record shape, B's text
* ``{root}/catalog/data-{warc_hash}.json`` -- requested / written / short / missing counts

A's topic and quality labels are attached afterwards by ``attach_swap_labels.py``, which writes B's
own co-partitioned tables (``write_attribute_tables`` below) so the extraction fleet never waits on
A's label wave.

A page whose upstream extraction is shorter than ``MIN_TEXT_CHARS`` is dropped (arm C's filter) and
counted. An allow-listed id the refetched WARC does not contain is listed under ``missing``; the
fleet is not done until that sum is zero.

Standalone GCS-claim worker (same contract as ``cpu_phase_a``); launch a fleet with
``launch_swap.py``.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import logging
import multiprocessing as mp
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import fsspec
import numpy as np
import pyarrow.parquet as pq
from marin.datakit.normalize import generate_id

from experiments.baseline_collection.decode_warcs_clean import _decode_one_warc_stream, _load_manifest, _warc_path_hash
from experiments.baseline_collection.grid_corpora import ShardDocs, Stage, write_done
from experiments.baseline_collection.grid_label import write_quality_shard, write_topic_shard
from experiments.baseline_collection.run_extract_standalone import (
    _claim_warc_atomic,
    _load_completed_registry,
    _refresh_claim,
    _register_completed_warc,
)
from experiments.fast_curation.upstream_resiliparse import MIN_TEXT_CHARS, extract_batch

logger = logging.getLogger(__name__)

DATASET = "lpv11_fastpipe_v2_upstreamrs_10k"
SOURCE_DATASET = "lpv11_fastpipe_v2_10k"
# Named like a post-decon tree on purpose: this IS the document set that came out of arm A's
# dedup + decon, so the grid registry and the natural-arm tokenizer read it as one.
OUTPUT_ROOT = "gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_upstreamrs_decon_deduped/10364warcs"
# Region holding A's label tables and receiving B's.
LABEL_BASE = "gs://marin-us-east5"

# Arm A's record keys other than ``text``, carried onto every B record unchanged.
RECORD_COLUMNS = ("doc_id", "url", "warc_hash", "snapshot", "fasttext_score", "pooled_prob", "modernbert_prob")
# Attached by attach_swap_labels.py from arm A's tables; required by ``write_attribute_tables``.
LABEL_COLUMNS = ("cluster_24", "topic_prob", "token_length", "quality_score", "quality_bucket")
EXTRACT_BATCH_DOCS = 32
# Wall cap on one WARC's extraction; a hung parser must not wedge the worker forever.
EXTRACT_TIMEOUT_SECONDS = 1800


@dataclass(frozen=True)
class SwapResult:
    n_requested: int
    n_written: int
    n_short: int
    missing: list[str]


def swap_records(
    allowlist: list[dict],
    decoded_chunks: Iterable[list[dict]],
    extract: Callable[[list[str]], list[str]],
    refresh: Callable[[], None] = lambda: None,
) -> tuple[list[dict], SwapResult]:
    """Select the allow-listed records from a decoded WARC and attach their upstream-extracted text.

    Pure with respect to storage: ``decoded_chunks`` is the decoder's output and ``extract`` maps a
    list of html strings to a list of texts (the pool in production, a plain function in tests).
    Rows come back in WARC order. Ids the WARC never produced are reported, not invented.
    """
    wanted = {row["doc_id"]: row for row in allowlist}
    n_requested = len(wanted)
    rows: list[dict] = []
    n_short = 0
    for chunk in decoded_chunks:
        hits = [rec for rec in chunk if rec["doc_id"] in wanted]
        if not hits:
            continue
        texts = extract([rec["html"] for rec in hits])
        for rec, text in zip(hits, texts, strict=True):
            row = wanted.pop(rec["doc_id"])
            if len(text) < MIN_TEXT_CHARS:
                n_short += 1
                continue
            rows.append({**row, "text": text})
        refresh()
    return rows, SwapResult(n_requested=n_requested, n_written=len(rows), n_short=n_short, missing=sorted(wanted))


def _pool_extract(pool: mp.pool.Pool, htmls: list[str]) -> list[str]:
    batches = [htmls[i : i + EXTRACT_BATCH_DOCS] for i in range(0, len(htmls), EXTRACT_BATCH_DOCS)]
    out = pool.map_async(extract_batch, batches).get(timeout=EXTRACT_TIMEOUT_SECONDS)
    return [text for batch in out for text in batch]


def shard_path(root: str, warc_hash: str) -> str:
    return f"{root}/deduped/data-{warc_hash}.jsonl.gz"


def write_record_shard(shard: str, rows: list[dict]) -> None:
    """Write B's record shard: arm A's keys plus B's ``text``, one JSON line per row, in ``rows`` order."""
    with fsspec.open(shard, "wb") as raw, gzip.open(raw, "wt", encoding="utf-8") as fh:
        for row in rows:
            record = {name: row[name] for name in RECORD_COLUMNS}
            record["text"] = row["text"]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_attribute_tables(dataset: str, shard: str, rows: list[dict], label_names: list[str]) -> None:
    """Write B's co-partitioned topic + quality tables and ``_done`` markers for record shard ``shard``.

    ``rows`` must be the shard's records in file order with the label columns attached; all outputs
    come from that one list in that one order, which is the co-partitioning contract
    ``datakit_store`` enforces. ``grid_corpora.OUTPUT_BASE`` must already point at the label region.
    """
    docs = ShardDocs(
        ids=[generate_id(row["text"]) for row in rows],
        native_ids=[row["doc_id"] for row in rows],
        urls=[row["url"] for row in rows],
        texts=[row["text"] for row in rows],
    )
    write_topic_shard(
        dataset,
        shard,
        docs,
        np.asarray([row["cluster_24"] for row in rows], dtype=np.int32),
        np.asarray([row["topic_prob"] for row in rows], dtype=np.float32),
        np.asarray([row["token_length"] for row in rows], dtype=np.int64),
        label_names,
    )
    write_quality_shard(
        dataset,
        shard,
        docs,
        np.asarray([row["quality_score"] for row in rows], dtype=np.float32),
        np.asarray([row["quality_bucket"] for row in rows], dtype=np.int8),
    )
    write_done(dataset, Stage.TOPIC, shard, len(rows))
    write_done(dataset, Stage.QUALITY, shard, len(rows))


def _read_allowlist(root: str, warc_hash: str) -> list[dict] | None:
    path = f"{root}/allowlist/data-{warc_hash}.parquet"
    fs = fsspec.filesystem("gcs") if path.startswith("gs://") else fsspec.filesystem("file")
    if not fs.exists(path):
        return None
    with fsspec.open(path, "rb") as fh:
        return pq.read_table(fh).to_pylist()


def swap_one_warc(
    warc_path: str,
    warc_hash: str,
    root: str,
    extract: Callable[[list[str]], list[str]],
    refresh: Callable[[], None],
) -> dict:
    t0 = time.monotonic()
    allowlist = _read_allowlist(root, warc_hash)
    catalog: dict[str, object]
    if allowlist is None:
        # No surviving A document in this WARC: nothing to extract and no shard to co-partition.
        catalog = {
            "warc_hash": warc_hash,
            "n_requested": 0,
            "n_written": 0,
            "n_short": 0,
            "missing": [],
            "skipped": True,
        }
    else:
        rows, result = swap_records(allowlist, _decode_one_warc_stream(warc_path, chunk_records=2000), extract, refresh)
        write_record_shard(shard_path(root, warc_hash), rows)
        catalog = {"warc_hash": warc_hash, **asdict(result), "skipped": False}
        if result.missing:
            logger.error("%s: %d allow-listed ids absent from the refetched WARC", warc_hash, len(result.missing))
    catalog["wall_seconds"] = round(time.monotonic() - t0, 1)
    with fsspec.open(f"{root}/catalog/data-{warc_hash}.json", "w") as fh:
        json.dump(catalog, fh)
    logger.info("B %s: %s", warc_hash, json.dumps({k: v for k, v in catalog.items() if k != "missing"}))
    return catalog


def run_worker(
    manifest: str,
    root: str,
    *,
    shuffle_seed: int,
    start: int,
    limit: int | None,
    poll_seconds: float,
    max_idle_passes: int,
    extract_procs: int,
    claim_stale_hours: float,
) -> None:
    pairs = [(w, _warc_path_hash(w)) for w in _load_manifest(manifest)][start:]
    if limit is not None:
        pairs = pairs[:limit]
    random.Random(shuffle_seed).shuffle(pairs)
    registry = f"{root}/_completed"
    logger.info("swap worker (seed=%d): %d WARCs -> %s", shuffle_seed, len(pairs), root)

    idle = 0
    with mp.get_context("spawn").Pool(extract_procs) as pool:
        extract = functools.partial(_pool_extract, pool)
        while True:
            done = _load_completed_registry(registry)
            remaining = [(w, h) for w, h in pairs if h not in done]
            if not remaining:
                logger.info("all %d WARCs registered done; exiting.", len(pairs))
                return
            progressed = 0
            for warc_path, h in remaining:
                # A pass over 10k WARCs outlives the stale-claim window, so re-check the registry
                # (cached for a minute) before every claim: otherwise a finished WARC's claim goes
                # stale mid-pass, gets reclaimed, and is re-extracted forever (2026-09-07 fleet).
                if h in _load_completed_registry(registry):
                    continue
                claim = f"{root}/_claims/data-{h}"
                if not _claim_warc_atomic(claim, stale_hours=claim_stale_hours):
                    continue
                swap_one_warc(warc_path, h, root, extract, functools.partial(_refresh_claim, claim))
                _register_completed_warc(h, registry)
                progressed += 1
            if progressed:
                idle = 0
                continue
            idle += 1
            if idle >= max_idle_passes:
                logger.warning("no claimable WARCs for %d passes (%d unfinished); exiting.", idle, len(remaining))
                return
            time.sleep(poll_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--root", default=OUTPUT_ROOT)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-idle-passes", type=int, default=5)
    ap.add_argument("--extract-procs", type=int, default=4)
    ap.add_argument("--claim-stale-hours", type=float, default=0.5)
    args = ap.parse_args()
    run_worker(
        args.manifest,
        args.root,
        shuffle_seed=args.shuffle_seed,
        start=args.start,
        limit=args.limit,
        poll_seconds=args.poll_seconds,
        max_idle_passes=args.max_idle_passes,
        extract_procs=args.extract_procs,
        claim_stale_hours=args.claim_stale_hours,
    )


if __name__ == "__main__":
    main()
