# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-extract the lpv11 LLM pipeline's documents with the resiliparse-rs fork.

Separates what the LLM pipeline's **selection** buys from what its **extraction** buys. The
document set is ``llm_pipeline_v1_1``'s post-dedup survivors, verbatim; only ``text`` is
regenerated, by the same Rust fork (and the same spec settings) that produced
``lpv11_fastpipe_v2``'s text. That places the new arm at the open corner of a 2x2:

================  ==========================  =========================
                  fork extract                LLM extract
cascade select    lpv11_fastpipe_v2 (trained)  --
LLM select        **this arm**                 lpv11_10k (trained)
================  ==========================  =========================

vs ``lpv11_10k`` isolates LLM extraction; vs ``lpv11_fastpipe_v2_10k`` isolates LLM selection.

Each WARC is refetched from Common Crawl (free ingress) and decoded with
``decode_warcs_clean._decode_one_warc_stream`` -- the clean WHATWG decoder the fastpipe line uses,
NOT the ``content.decode("utf-8", errors="replace")`` path lpv11 itself used. Both decoders select
records with identical filters, so no records are lost by the choice; what changes is the HTML
*content* on pages that are not valid UTF-8, where the clean decoder is simply correct and the old
one emitted replacement characters. See ``.agents/projects/lpv11_forkextract_swap.md``.

Two things this worker needs that the upstream-resiliparse swap did not:

* the Rust engine SIGSEGVs on ``<frameset>`` pages (0.108% measured), so extraction goes through
  ``cpu_phase_c.ResiliparseRsPool``, which isolates a crashed batch per document and emits ``""``
  for the offender instead of losing the WARC;
* the extractor artifact must be installed per worker (and unpacked outside ``/tmp``, which is
  ``noexec`` on Iris CPU workers).

Per WARC it writes ``{root}/deduped/data-{warc_hash}.jsonl.gz`` (lpv11's record shape with the new
text) and ``{root}/catalog/data-{warc_hash}.json`` (requested / written / short / crashed /
missing). Labels are attached afterwards by ``attach_lpv11_fork_labels``.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import logging
import os
import random
import time
from collections.abc import Callable, Iterable, Iterator

import fsspec
import pyarrow.parquet as pq

from experiments.baseline_collection.decode_warcs_clean import (
    _decode_one_warc_stream,
    _load_manifest,
    _warc_path_hash,
)
from experiments.baseline_collection.run_extract_standalone import (
    _claim_warc_atomic,
    _load_completed_registry,
    _refresh_claim,
    _register_completed_warc,
)
from experiments.fast_curation.cpu_phase_c import _install_rust_extractor
from experiments.fast_curation.spec import RESILIPARSE_RS_ARTIFACT, get_spec
from experiments.fast_curation.upstream_resiliparse import MIN_TEXT_CHARS

logger = logging.getLogger(__name__)

DATASET = "lpv11_forkrs_10k"
SOURCE_DATASET = "lpv11_10k"
SOURCE_TREE = "gs://marin-us-east5/documents/baseline_llm_pipeline_v1_1_deduped/10364warcs/deduped"
OUTPUT_ROOT = "gs://marin-us-east5/documents/baseline_lpv11_forkrs_decon_deduped/10364warcs"
LABEL_BASE = "gs://marin-us-east5"

# The spec whose extractor settings produced arm A's text: resiliparse-rs at the pinned commit,
# main_content=True, preserve_formatting="markdown", 50 MB html cap. Reusing it verbatim is what
# makes "fork extraction" mean the same thing in both arms.
EXTRACTOR_SPEC = "lpv11_fastpipe_v2"

# lpv11's own record columns, carried through unchanged except ``text``. ``warc_hash`` is derived
# (sha256(warc_file)[:12]) rather than stored, and travels so downstream joins stay per-WARC.
RECORD_COLUMNS = ("warc_record_id", "url", "warc_file", "warc_hash", "snapshot")
EXTRACT_BATCH_DOCS = 256


def read_jsonl_gz(path: str) -> Iterator[dict]:
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def select_and_extract(
    allowlist: list[dict],
    decoded_chunks: Iterable[list[dict]],
    extract: Callable[[list[str], list[str]], tuple[list[str], list[str]]],
    refresh: Callable[[], None] = lambda: None,
) -> tuple[list[dict], dict]:
    """Pick the allow-listed records out of a decoded WARC and attach their fork-extracted text.

    ``extract`` maps ``(htmls, ids)`` to ``(texts, crashed_ids)`` -- the pool's contract, so a page
    that kills its worker yields ``""`` and is reported rather than losing the WARC. Pure with
    respect to storage so tests can drive it with a plain function.
    """
    wanted = {row["warc_record_id"]: row for row in allowlist}
    n_requested = len(wanted)
    rows: list[dict] = []
    n_short = 0
    crashed: list[str] = []
    for chunk in decoded_chunks:
        hits = [rec for rec in chunk if rec["doc_id"] in wanted]
        if not hits:
            continue
        for start in range(0, len(hits), EXTRACT_BATCH_DOCS):
            part = hits[start : start + EXTRACT_BATCH_DOCS]
            texts, part_crashed = extract([r["html"] for r in part], [r["doc_id"] for r in part])
            crashed.extend(part_crashed)
            for rec, text in zip(part, texts, strict=True):
                row = wanted.pop(rec["doc_id"])
                if len(text) < MIN_TEXT_CHARS:
                    n_short += 1
                    continue
                rows.append({**row, "text": text})
            refresh()
    stats = {
        "n_requested": n_requested,
        "n_written": len(rows),
        "n_short": n_short,
        "n_crashed": len(crashed),
        "crashed": sorted(crashed),
        "missing": sorted(wanted),
    }
    return rows, stats


def shard_path(root: str, warc_hash: str) -> str:
    return f"{root}/deduped/data-{warc_hash}.jsonl.gz"


def write_record_shard(shard: str, rows: list[dict]) -> None:
    """Write lpv11's record shape with the new text, in ``rows`` order."""
    with fsspec.open(shard, "wb") as raw, gzip.open(raw, "wt", encoding="utf-8") as fh:
        for row in rows:
            record = {name: row[name] for name in RECORD_COLUMNS}
            record["text"] = row["text"]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_allowlist(root: str, warc_hash: str) -> list[dict] | None:
    path = f"{root}/allowlist/data-{warc_hash}.parquet"
    fs = fsspec.filesystem("gcs") if path.startswith("gs://") else fsspec.filesystem("file")
    if not fs.exists(path):
        return None
    with fsspec.open(path, "rb") as fh:
        return pq.read_table(fh).to_pylist()


def extract_one_warc(
    warc_path: str,
    warc_hash: str,
    root: str,
    extract: Callable[[list[str], list[str]], tuple[list[str], list[str]]],
    refresh: Callable[[], None],
) -> dict:
    t0 = time.monotonic()
    allowlist = _read_allowlist(root, warc_hash)
    catalog: dict[str, object]
    if allowlist is None:
        catalog = {
            "warc_hash": warc_hash,
            "n_requested": 0,
            "n_written": 0,
            "n_short": 0,
            "n_crashed": 0,
            "crashed": [],
            "missing": [],
            "skipped": True,
        }
    else:
        rows, stats = select_and_extract(
            allowlist, _decode_one_warc_stream(warc_path, chunk_records=2000), extract, refresh
        )
        write_record_shard(shard_path(root, warc_hash), rows)
        catalog = {"warc_hash": warc_hash, **stats, "skipped": False}
        if stats["missing"]:
            logger.error("%s: %d allow-listed ids absent from the refetched WARC", warc_hash, len(stats["missing"]))
    catalog["wall_seconds"] = round(time.monotonic() - t0, 1)
    with fsspec.open(f"{root}/catalog/data-{warc_hash}.json", "w") as fh:
        json.dump(catalog, fh)
    logger.info(
        "fork %s: %s", warc_hash, json.dumps({k: v for k, v in catalog.items() if k not in ("missing", "crashed")})
    )
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
    artifact: str,
) -> None:
    spec = get_spec(EXTRACTOR_SPEC)
    # Per-PID unpack dir: concurrent installs into one dir race and truncate the .so.
    pool = _install_rust_extractor(artifact, spec, extract_procs, dest=f"/root/resiliparse_rs_{os.getpid()}")
    cap = spec.justext_max_html_chars

    def extract(htmls: list[str], ids: list[str]) -> tuple[list[str], list[str]]:
        return pool.run(htmls, ids, cap)

    pairs = [(w, _warc_path_hash(w)) for w in _load_manifest(manifest)][start:]
    if limit is not None:
        pairs = pairs[:limit]
    random.Random(shuffle_seed).shuffle(pairs)
    registry = f"{root}/_completed"
    logger.info("lpv11 fork worker (seed=%d): %d WARCs -> %s", shuffle_seed, len(pairs), root)

    idle = 0
    try:
        while True:
            done = _load_completed_registry(registry)
            remaining = [(w, h) for w, h in pairs if h not in done]
            if not remaining:
                logger.info("all %d WARCs registered done; exiting.", len(pairs))
                return
            progressed = 0
            for warc_path, h in remaining:
                # Re-check within the pass: a long pass outlives the stale-claim window, so without
                # this a finished WARC's claim goes stale and gets re-extracted forever.
                if h in _load_completed_registry(registry):
                    continue
                claim = f"{root}/_claims/data-{h}"
                if not _claim_warc_atomic(claim, stale_hours=claim_stale_hours):
                    continue
                extract_one_warc(warc_path, h, root, extract, functools.partial(_refresh_claim, claim))
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
    finally:
        pool.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--root", default=OUTPUT_ROOT)
    ap.add_argument("--artifact", default=RESILIPARSE_RS_ARTIFACT)
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
        artifact=args.artifact,
    )


if __name__ == "__main__":
    main()
