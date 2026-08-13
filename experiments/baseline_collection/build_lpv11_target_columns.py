# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach ``llm_pipeline_v1_1`` (lpv11) verdicts + text to the 200-WARC comparison sample.

The cascade planner scores every pipeline against a *gold* extractor. Historically that gold was
the 8B ``high_quality`` run (``label_8b`` / ``text_8b``); this job produces the columns needed to
retarget it at lpv11::

    warc_record_id  — join key, row-aligned with ``sample_100k_scored``
    text_lpv11      — lpv11's extraction (null where it abstained or never saw the page)
    label_lpv11     — "useful" | "not_useful" | null (page outside lpv11's coverage)

**Coverage.** Any WARC lpv11 has not finished yields null labels, and those rows are excluded from
agreement metrics downstream. The 5 WARCs that lagged the rest are listed in
``lpv11_missing_5_warcs.txt`` (all finished 2026-08-11).

**Labels.** lpv11 persists KEPT docs only, so negatives are recovered by set difference exactly as
``build_extractor_comparison_dataset`` does for the 8B: within a covered WARC, a page lpv11 did not
keep is "not_useful". That includes pages over the length cap — ``_filter_by_length`` is shared by
both extraction runs, so such a page yields no output under either and is a genuine negative, which
also keeps the two targets measuring the SAME doc universe.

Batches are read from the ORIGINAL regional buckets via the consolidated *resolved* manifest — the
``by_region/`` archive is the April consolidation of the original spec and does NOT contain lpv11.
That manifest is a point-in-time SNAPSHOT, not a live view: a WARC extracted after it was written is
absent from it (and from ``done_warcs_*.txt``), so those hashes fall back to globbing the regional
buckets directly. Never treat either file as ground truth for "has this WARC finished" — the
authoritative signal is the ``_done`` marker in ``.../{SPEC}/data-<hash>/``.
Egress is ~11 GB (dominated by eu-west4), ≈$0.70 into us-east5.

Run (CPU, us-east5 where the sample lives)::

    uv run iris --cluster marin job run --region us-east5 --cpu 16 --memory 32GB --disk 20GB \\
      --enable-extra-resources --extra cpu --priority interactive --no-wait \\
      --job-name lpv11-target -- \\
      python -m experiments.baseline_collection.build_lpv11_target_columns
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from marin.utils import fsspec_glob

from experiments.baseline_collection.decode_warcs_clean import _normalize_record_id, _warc_path_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPEC = "llm_pipeline_v1_1"
ABSTAIN_MARKER = "[NO_USEFUL_CONTENT]"

OUT_ROOT = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc"
SAMPLE_DIR = f"{OUT_ROOT}/sample_100k_scored"
TARGET_DIR = f"{OUT_ROOT}/target_lpv11"

CONSOLIDATED_ROOT = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated"
RESOLVED_MANIFEST_URI = f"{CONSOLIDATED_ROOT}/resolved/resolved_{SPEC}.jsonl.gz"

READ_THREADS = 64

# Every bucket a WARC's batches may be in — cross-region resume scatters one WARC across regions.
REGIONAL_BUCKETS = (
    "marin-us-central1",
    "marin-us-east5",
    "marin-us-east1",
    "marin-us-west4",
    "marin-eu-west4",
)

_SCHEMA = pa.schema([("warc_record_id", pa.string()), ("text_lpv11", pa.string()), ("label_lpv11", pa.string())])


def _live_paths_for_warc(warc_hash: str) -> list[str]:
    """Batch paths for one WARC straight from the regional buckets (union across all regions).

    Used for WARCs the resolved manifest predates. The union may include duplicate batches from a
    cross-region resume, which is harmless here: ``_kept_text_for_warc`` keys by record id, so a
    repeated record simply overwrites itself.
    """
    paths: list[str] = []
    for bucket in REGIONAL_BUCKETS:
        paths.extend(
            fsspec_glob(f"gs://{bucket}/documents/baseline_llm_extraction/{SPEC}/data-{warc_hash}/batch_*.jsonl.gz")
        )
    return sorted(paths)


def _winner_paths_by_warc(wanted_hashes: set[str]) -> dict[str, list[str]]:
    """warc_hash -> its batch paths: deduplicated winners from the resolved manifest where possible.

    The manifest is a point-in-time snapshot of the consolidation ``resolve`` step, NOT a live view —
    a WARC extracted after it was written is absent entirely. Those fall back to globbing the
    regional buckets so newly finished WARCs are picked up without re-running consolidation.
    """
    paths: dict[str, list[str]] = defaultdict(list)
    with fsspec.open(RESOLVED_MANIFEST_URI, "rb") as fh:
        for line in gzip.GzipFile(fileobj=fh):
            rec = json.loads(line)
            if rec["warc_hash"] in wanted_hashes and rec.get("is_valid"):
                paths[rec["warc_hash"]].append(rec["path"])

    stale = sorted(wanted_hashes - set(paths))
    if stale:
        logger.info("%d WARCs absent from the resolved manifest; globbing buckets: %s", len(stale), stale)
        with ThreadPoolExecutor(max_workers=min(READ_THREADS, len(stale))) as ex:
            for warc_hash, live in zip(stale, ex.map(_live_paths_for_warc, stale), strict=True):
                if live:
                    paths[warc_hash] = live
                    logger.info("  %s: %d batches found live", warc_hash, len(live))
    return {h: sorted(p) for h, p in paths.items()}


def _read_sample() -> tuple[list[str], dict[str, str], list[list[str]]]:
    """Load the sample's join keys in shard+row order.

    Returns ``(shard_files, {record_id: warc_hash}, per_shard_record_ids)``.
    """
    files = sorted(fsspec_glob(f"{SAMPLE_DIR}/*.parquet"))
    if not files:
        raise RuntimeError(f"no sample parquet under {SAMPLE_DIR}")
    warc_by_id: dict[str, str] = {}
    per_shard: list[list[str]] = []
    for path in files:
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["warc_record_id", "warc_file"])
        ids = t.column("warc_record_id").to_pylist()
        for rid, wf in zip(ids, t.column("warc_file").to_pylist(), strict=True):
            warc_by_id[rid] = _warc_path_hash(wf)
        per_shard.append(ids)
    logger.info("sample: %d shards, %d docs, %d WARCs", len(files), len(warc_by_id), len(set(warc_by_id.values())))
    return files, warc_by_id, per_shard


def _kept_text_for_warc(paths: list[str], wanted_ids: set[str]) -> dict[str, str]:
    """{record_id: lpv11 text} for the sampled ids of one WARC (abstains/empties dropped)."""
    kept: dict[str, str] = {}
    for path in paths:
        with fsspec.open(path, "rb") as fh:
            for line in gzip.GzipFile(fileobj=fh):
                rec = json.loads(line)
                rid = _normalize_record_id(rec.get("warc_record_id") or "")
                if rid not in wanted_ids:
                    continue
                text = rec.get("text") or ""
                if not text.strip() or text.strip() == ABSTAIN_MARKER:
                    continue
                kept[rid] = text
    return kept


def run() -> None:
    shard_files, warc_by_id, per_shard = _read_sample()

    ids_by_warc: dict[str, set[str]] = defaultdict(set)
    for rid, warc_hash in warc_by_id.items():
        ids_by_warc[warc_hash].add(rid)

    winners = _winner_paths_by_warc(set(ids_by_warc))
    covered = set(winners)
    uncovered = sorted(set(ids_by_warc) - covered)
    logger.info(
        "lpv11 coverage: %d/%d WARCs (%d batches); uncovered: %s",
        len(covered),
        len(ids_by_warc),
        sum(len(p) for p in winners.values()),
        uncovered,
    )

    kept_text: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=READ_THREADS) as ex:
        futures = {ex.submit(_kept_text_for_warc, winners[h], ids_by_warc[h]): h for h in sorted(covered)}
        for i, fut in enumerate(futures, 1):
            kept_text.update(fut.result())
            if i % 25 == 0:
                logger.info("read %d/%d WARCs, %d keeps so far", i, len(covered), len(kept_text))

    n_useful = n_not = n_null = 0
    for shard_i, (src, ids) in enumerate(zip(shard_files, per_shard, strict=True)):
        texts: list[str | None] = []
        labels: list[str | None] = []
        for rid in ids:
            text = kept_text.get(rid)
            if text is not None:
                labels.append("useful")
                texts.append(text)
                n_useful += 1
            elif warc_by_id[rid] in covered:
                # Includes pages over the length cap: `_filter_by_length` is shared by both pipelines, so
                # such a page yields no output under either and is a genuine negative — the same
                # convention build_extractor_comparison_dataset uses for label_8b. Treating it as
                # "unjudged" instead would give the two targets different doc universes and make their
                # F1 numbers incomparable.
                labels.append("not_useful")
                texts.append(None)
                n_not += 1
            else:
                labels.append(None)
                texts.append(None)
                n_null += 1
        table = pa.Table.from_pydict({"warc_record_id": ids, "text_lpv11": texts, "label_lpv11": labels}, schema=_SCHEMA)
        out = f"{TARGET_DIR}/{src.rsplit('/', 1)[-1]}"
        with fsspec.open(out, "wb") as fh:
            pq.write_table(table, fh, compression="zstd")
        if shard_i % 50 == 0:
            logger.info("wrote shard %d/%d", shard_i, len(shard_files))

    total = n_useful + n_not + n_null
    logger.info(
        "DONE -> %s | useful=%d (%.1f%%) not_useful=%d (%.1f%%) null=%d (%.1f%%) of %d",
        TARGET_DIR,
        n_useful,
        100 * n_useful / total,
        n_not,
        100 * n_not / total,
        n_null,
        100 * n_null / total,
        total,
    )


def main() -> None:
    global TARGET_DIR
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dir", default=TARGET_DIR, help="Output prefix for the lpv11 columns.")
    args = p.parse_args()
    TARGET_DIR = args.target_dir
    run()


if __name__ == "__main__":
    main()
