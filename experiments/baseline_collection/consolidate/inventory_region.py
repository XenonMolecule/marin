# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory one region's extraction output.

Runs inside a region-pinned Iris CPU job. Lists every batch file + sidecar
under ``documents/baseline_llm_extraction/`` in the region's local bucket,
decompresses each batch to count records and sum text characters (the
tiebreaker metric for duplicate resolution), and writes a JSONL manifest to
us-central1.

Output layout (one file per region, all in us-central1)::

    gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/
        inventories/inventory_{region}.jsonl.gz

Each file contains two row shapes distinguished by ``record_type``:

- ``batch``: one row per ``data-{hash}/batch_NNNN.jsonl.gz``, with size,
  mtime, decompressed record count, sum of ``text`` characters, and sidecar
  presence + contents (``.count`` integer, ``.tokens.gz`` per-status summary).
- ``warc``: one row per WARC hash observed in the region, with ``num_batches``,
  ``max_batch_idx``, and ``has_done`` / ``has_claimed`` flags.

The batch-row schema is deliberately wide so the resolver (and future analyses
for FLOPs, filter rates, etc.) can operate on the manifest alone without
re-reading any batch file.

Usage (typically invoked by ``launch_inventory.py``)::

    python experiments/baseline_collection/consolidate/inventory_region.py \\
        --region us-east5 --max-workers 64
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

logger = logging.getLogger(__name__)

# Iris region label → GCS bucket. ``europe-west4`` and ``eu-west4`` both accepted
# because iris uses the full label while the bucket uses the short name.
REGION_TO_BUCKET: dict[str, str] = {
    "us-central1": "marin-us-central1",
    "us-east1": "marin-us-east1",
    "us-east5": "marin-us-east5",
    "us-west4": "marin-us-west4",
    "europe-west4": "marin-eu-west4",
    "eu-west4": "marin-eu-west4",
}

SOURCE_SUBDIR = "documents/baseline_llm_extraction"
CONSOLIDATED_SUBDIR = "documents/baseline_llm_extraction_consolidated"
DEST_BUCKET = "marin-us-central1"

_BATCH_RE = re.compile(r"data-([0-9a-f]+)/batch_(\d+)\.jsonl\.gz$")
_COUNT_RE = re.compile(r"data-([0-9a-f]+)/batch_(\d+)\.count$")
_TOKENS_RE = re.compile(r"data-([0-9a-f]+)/batch_(\d+)\.tokens\.gz$")
_DONE_RE = re.compile(r"data-([0-9a-f]+)/_done$")
_CLAIMED_RE = re.compile(r"data-([0-9a-f]+)/_claimed$")


def _list_all_paths_with_meta(bucket: str) -> dict[str, dict]:
    """Recursive list of every file under ``SOURCE_SUBDIR`` with size + mtime.

    Uses ``fs.find(..., detail=True)`` for a single paginated listing call —
    vastly cheaper than N per-file ``fs.info()`` round-trips (which would be
    126K sequential hops in eu-west4).

    Returns a dict keyed by ``gs://`` URI with ``{"size", "mtime"}`` values.
    """
    fs = fsspec.filesystem("gcs")
    prefix = f"{bucket}/{SOURCE_SUBDIR}"
    raw = fs.find(prefix, detail=True)  # dict: bare_path -> info
    out: dict[str, dict] = {}
    for bare_path, info in raw.items():
        gs_uri = bare_path if bare_path.startswith("gs://") else f"gs://{bare_path}"
        size = int(info.get("size", 0))
        mtime = info.get("mtime") or info.get("updated") or info.get("timeCreated")
        out[gs_uri] = {"size": size, "mtime": str(mtime) if mtime is not None else None}
    return out


def _read_count_sidecar(gs_path: str) -> int | None:
    try:
        with fsspec.open(gs_path, "r") as f:
            return int(f.read().strip())
    except Exception as e:
        logger.warning("count sidecar read failed for %s: %s", gs_path, e)
        return None


def _read_tokens_sidecar(gs_path: str) -> dict | None:
    """Summarize .tokens.gz: per-status counts + kept response tokens."""
    try:
        with fsspec.open(gs_path, "rb") as f, gzip.open(f, "rt") as gz:
            by_status: dict[str, int] = {}
            kept_response = 0
            kept_thinking = 0
            for line in gz:
                if not line.strip():
                    continue
                r = json.loads(line)
                s = r.get("status", "unknown")
                by_status[s] = by_status.get(s, 0) + 1
                if s == "kept":
                    kept_response += r.get("response_tokens", 0)
                    kept_thinking += r.get("thinking_tokens", 0)
            return {
                "by_status": by_status,
                "kept_response_tokens": kept_response,
                "kept_thinking_tokens": kept_thinking,
            }
    except Exception as e:
        logger.warning("tokens sidecar read failed for %s: %s", gs_path, e)
        return None


def _inspect_batch(gs_path: str) -> dict:
    """Decompress a batch jsonl.gz, count rows, sum text chars. Never raises."""
    result: dict = {
        "decompress_ok": False,
        "num_records": None,
        "sum_text_chars": None,
        "error": None,
    }
    try:
        with fsspec.open(gs_path, "rb") as f, gzip.open(f, "rt") as gz:
            n = 0
            chars = 0
            for line in gz:
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                chars += len(r.get("text", ""))
            result["num_records"] = n
            result["sum_text_chars"] = chars
            result["decompress_ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def _build_batch_row(
    key: tuple[str, int],
    batch_path: str,
    count_path: str | None,
    tokens_path: str | None,
    size: int,
    mtime: str | None,
    region: str,
) -> dict:
    h, idx = key
    inspection = _inspect_batch(batch_path)
    return {
        "record_type": "batch",
        "region": region,
        "warc_hash": h,
        "batch_idx": idx,
        "path": batch_path,
        "size_bytes": size,
        "mtime": mtime,
        **inspection,
        "has_count_sidecar": count_path is not None,
        "count_sidecar": _read_count_sidecar(count_path) if count_path else None,
        "has_tokens_sidecar": tokens_path is not None,
        "tokens_sidecar": _read_tokens_sidecar(tokens_path) if tokens_path else None,
    }


def inventory_region(region: str, max_workers: int, output_path: str | None = None) -> str:
    """Inventory a region. Returns the output gs:// path."""
    if region not in REGION_TO_BUCKET:
        raise ValueError(f"Unknown region {region!r}; expected one of {sorted(REGION_TO_BUCKET)}")
    bucket = REGION_TO_BUCKET[region]
    logger.info("Inventorying region=%s bucket=%s", region, bucket)

    t0 = time.monotonic()
    meta = _list_all_paths_with_meta(bucket)
    logger.info("Listed %d files (with size+mtime) in %.1fs", len(meta), time.monotonic() - t0)

    batches: dict[tuple[str, int], str] = {}
    counts: dict[tuple[str, int], str] = {}
    tokens: dict[tuple[str, int], str] = {}
    done_hashes: set[str] = set()
    claimed_hashes: set[str] = set()

    for p in meta:
        m = _BATCH_RE.search(p)
        if m:
            batches[(m.group(1), int(m.group(2)))] = p
            continue
        m = _COUNT_RE.search(p)
        if m:
            counts[(m.group(1), int(m.group(2)))] = p
            continue
        m = _TOKENS_RE.search(p)
        if m:
            tokens[(m.group(1), int(m.group(2)))] = p
            continue
        m = _DONE_RE.search(p)
        if m:
            done_hashes.add(m.group(1))
            continue
        m = _CLAIMED_RE.search(p)
        if m:
            claimed_hashes.add(m.group(1))
            continue

    logger.info(
        "Parsed: %d batches, %d count sidecars, %d tokens sidecars, %d _done, %d _claimed",
        len(batches),
        len(counts),
        len(tokens),
        len(done_hashes),
        len(claimed_hashes),
    )

    # Decompress every batch in parallel. Size + mtime come from the bulk listing;
    # the per-file cost here is one GCS read + gzip decompress + JSON parse.
    t0 = time.monotonic()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for key, bpath in batches.items():
            m = meta.get(bpath, {})
            fut = pool.submit(
                _build_batch_row,
                key,
                bpath,
                counts.get(key),
                tokens.get(key),
                int(m.get("size", 0)),
                m.get("mtime"),
                region,
            )
            futures[fut] = key
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if i % 5000 == 0:
                elapsed = time.monotonic() - t0
                rate = i / elapsed
                eta = (len(futures) - i) / max(rate, 1e-6)
                logger.info("Inspected %d/%d (%.0f/s, ETA %.0fs)", i, len(futures), rate, eta)
    logger.info("Inspected %d batches in %.1fs", len(rows), time.monotonic() - t0)

    # Per-WARC summary rows.
    all_hashes: set[str] = set(h for (h, _) in batches.keys()) | done_hashes | claimed_hashes
    by_hash_indices: dict[str, list[int]] = {}
    for h, idx in batches.keys():
        by_hash_indices.setdefault(h, []).append(idx)
    for h in all_hashes:
        idxs = by_hash_indices.get(h, [])
        rows.append(
            {
                "record_type": "warc",
                "region": region,
                "warc_hash": h,
                "num_batches": len(idxs),
                "max_batch_idx": max(idxs) if idxs else None,
                "min_batch_idx": min(idxs) if idxs else None,
                "has_done": h in done_hashes,
                "has_claimed": h in claimed_hashes,
            }
        )

    out_path = output_path or (f"gs://{DEST_BUCKET}/{CONSOLIDATED_SUBDIR}/inventories/inventory_{region}.jsonl.gz")
    logger.info("Writing %d rows to %s", len(rows), out_path)
    with fsspec.open(out_path, "wb") as f, gzip.open(f, "wt", encoding="utf-8") as gz:
        for r in rows:
            gz.write(json.dumps(r, sort_keys=True) + "\n")
    logger.info("DONE region=%s", region)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, choices=sorted(REGION_TO_BUCKET.keys()))
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Override destination path. Default: us-central1 consolidated/inventories/.",
    )
    args = parser.parse_args()
    inventory_region(args.region, max_workers=args.max_workers, output_path=args.output_path)


if __name__ == "__main__":
    main()
