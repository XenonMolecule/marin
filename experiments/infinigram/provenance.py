# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reattach document provenance (url / warc ids) to text-only tiers.

The LLM-extraction deduped tiers store only ``text`` -- the ids live upstream in
the raw extraction batches (all mirrored into the us-central1 bucket, so reading
them is in-region). Dedup keeps the surviving text verbatim, so we recover
provenance by joining on an exact text hash, byte-for-byte matching
``experiments/baseline_collection/build_high_quality_hf_export.py``.
"""

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

from zephyr.readers import load_file

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

# Provenance fields carried from the raw batch onto each indexed document.
PROVENANCE_FIELDS = ("url", "warc_record_id", "warc_file", "snapshot")

# Raw provenance is spread over many small per-WARC batch shards; reading them
# serially dominates wall-clock, so fan the reads out across threads (I/O-bound).
_PROVENANCE_READ_WORKERS = 32


def content_hash(text: str) -> str:
    """16-byte blake2b digest of the join text (matches the HF-export join key)."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _join_text(record: dict) -> str | None:
    return record.get("text") or record.get("generated_text")


def build_provenance_map(globs: tuple[str, ...], wanted: set[str]) -> dict[str, dict]:
    """Map content-hash -> provenance for the hashes in ``wanted``.

    Streams every raw batch shard once and keeps only records whose text hash is
    wanted, bounding memory to the size of the deduped corpus. On duplicate
    hashes the lexicographically smallest ``(warc_file, warc_record_id)`` wins so
    the attached provenance is deterministic (mirrors the HF-export ``attach``).
    """
    shards = sorted({s for g in globs for s in fsspec_glob(g)})
    if not shards:
        raise ValueError(f"provenance source resolved to no shards: {globs}")
    logger.info("Building provenance map from %d raw shards for %d wanted docs", len(shards), len(wanted))

    def _scan(shard: str) -> list[tuple[str, dict]]:
        hits: list[tuple[str, dict]] = []
        for rec in load_file(shard):
            text = _join_text(rec)
            if not text:
                continue
            h = content_hash(text)
            if h in wanted:
                hits.append((h, {f: rec.get(f) or "" for f in PROVENANCE_FIELDS}))
        return hits

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_PROVENANCE_READ_WORKERS) as ex:
        for hits in ex.map(_scan, shards):
            for h, prov in hits:
                best = out.get(h)
                # Deterministic tie-break: smallest (warc_file, warc_record_id) wins.
                if best is None or (prov["warc_file"], prov["warc_record_id"]) < (
                    best["warc_file"],
                    best["warc_record_id"],
                ):
                    out[h] = prov
    logger.info("Provenance map covers %d/%d wanted docs", len(out), len(wanted))
    return out
