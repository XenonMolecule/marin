#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract FineWeb-CC — full-FineWeb documents for a fixed CommonCrawl WARC set.

Full FineWeb lives only on HuggingFace (`HuggingFaceFW/fineweb`): snapshot-
segmented parquet with a `file_path` column = the source CommonCrawl WARC
(`s3://commoncrawl/...warc.gz`, identical format to our manifest). We fan out
over every parquet shard of the snapshots our WARCs touch, download each shard
to worker-local disk, keep the rows whose `file_path` is one of our WARCs, and
write the survivors — discarding the shard.

Why this shape (the "bulk" approach, done right):
  * **No GCS staging.** The ~45 TB of FineWeb transits worker-local disk (~2 GB
    at a time, free HF ingress) and is never persisted — no `tmp/ttl=Nd/`
    staging, no multi-TB storage bill, no cleanup step.
  * **100% coverage is trivial.** Every shard is read in full (no footer/row-group
    skipping), so no matching row can be missed.
  * **Per-shard idempotent.** Each FineWeb shard writes to a deterministic output
    name and is skipped if already present — so a smoke run over the first few
    shards is *real work the full run reuses*, and the job is freely resumable.

Output: ``{output_path}/data-<snapshot>-<shard>.jsonl.gz`` with FineWeb's
``{text, id, url, dump, file_path, language, token_count}`` for our WARCs.

Usage (Iris, CPU, us-central2)::

    # Smoke: first 12 shards, written to the REAL output path (reused by the full run).
    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB --priority interactive \\
        --extra eval --enable-extra-resources --region us-central2 \\
        --job-name fineweb-cc-smoke -e HF_TOKEN <token> \\
        -- python experiments/baseline_collection/filter_fineweb_cc.py \\
           --warc-manifest experiments/distill/dclm_400m_1x.txt \\
           --output-path gs://marin-us-central2/documents/baseline_fineweb_cc/10364warcs/ \\
           --limit-shards 12

    # Full run: drop --limit-shards (skips the 12 already done).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import tempfile
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from huggingface_hub import HfFileSystem
from rigging.filesystem import filesystem as marin_filesystem
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

_HF_FINEWEB_ROOT = "datasets/HuggingFaceFW/fineweb/data"
_SNAPSHOT_RE = re.compile(r"CC-MAIN-\d{4}-\d{2}")
_OUTPUT_COLUMNS = ["text", "id", "url", "dump", "file_path", "language", "token_count"]

# Per-worker cache of our WARC set (loaded once from the bundled manifest).
_WARC_CACHE: dict = {}


def _load_warcs(manifest_path: str) -> frozenset[str]:
    if "set" not in _WARC_CACHE:
        with open(manifest_path) as f:
            warcs = frozenset(line.strip() for line in f if line.strip())
        _WARC_CACHE["set"] = warcs
        _WARC_CACHE["arr"] = pa.array(sorted(warcs), type=pa.string())  # value_set for pc.is_in
    return _WARC_CACHE["set"]


def _hf_retry(fn, *args, attempts: int = 7, **kwargs):
    """Call an HfFileSystem op with exponential backoff on HF rate-limit (429).

    HF free-tier caps at 1000 API requests / 5 min; with many workers we can brush
    it, so we back off and retry rather than fail the shard.
    """
    for k in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            retriable = "rate limit" in msg or "429" in msg or "quota" in msg or "too many requests" in msg
            if k < attempts - 1 and retriable:
                time.sleep(min(150, 10 * (2**k)))
                continue
            raise


def _output_name(url: str) -> str:
    """Deterministic output filename for a FineWeb shard URL — keyed by the shard's
    identity so processing is idempotent (skip-existing) and a smoke prefix is reused."""
    parts = url.rstrip("/").split("/")
    snapshot = next((p for p in parts if p.startswith("CC-MAIN-")), "unknown")
    base = parts[-1].removesuffix(".parquet")
    return f"data-{snapshot}-{base}.jsonl.gz"


def _list_shards(snapshots: set[str]) -> list[str]:
    """All FineWeb parquet shard paths for the given snapshots, sorted (deterministic)."""
    fs = HfFileSystem()
    urls: list[str] = []
    for snap in sorted(snapshots):
        base = f"{_HF_FINEWEB_ROOT}/{snap}"
        try:
            listing = _hf_retry(fs.ls, base, detail=False)
        except FileNotFoundError:
            logger.warning("FineWeb has no snapshot %s — our WARCs there will have no FineWeb docs", snap)
            continue
        urls.extend(p for p in listing if p.endswith(".parquet"))
    return sorted(urls)


def run_filter(
    manifest_path: str,
    output_path: str,
    max_workers: int,
    snapshots_filter: set[str] | None = None,
    limit_shards: int | None = None,
) -> dict:
    warcs = _load_warcs(manifest_path)
    snapshots = {m.group(0) for w in warcs if (m := _SNAPSHOT_RE.search(w))}
    if snapshots_filter is not None:
        snapshots &= snapshots_filter
        logger.info("restricted to %d snapshot(s): %s", len(snapshots), sorted(snapshots))

    shard_urls = _list_shards(snapshots)
    if limit_shards is not None:
        shard_urls = shard_urls[:limit_shards]
    logger.info("WARCs=%d snapshots=%d FineWeb shards to scan=%d", len(warcs), len(snapshots), len(shard_urls))
    out_root = output_path.rstrip("/")

    def process_shards(urls, _):
        _load_warcs(manifest_path)
        warc_arr = _WARC_CACHE["arr"]
        hf = HfFileSystem()
        gcs = marin_filesystem("gcs")
        for url in urls:
            out_path = f"{out_root}/{_output_name(url)}"
            if gcs.exists(out_path):  # per-shard idempotency
                yield {"shard": _output_name(url), "status": "skip"}
                continue
            fd, local = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            try:
                _hf_retry(hf.get, url, local)  # whole shard -> worker-local disk (free HF ingress)
                pf = pq.ParquetFile(local)
                cols = [c for c in _OUTPUT_COLUMNS if c in pf.schema_arrow.names]
                kept_rows: list[dict] = []
                for i in range(pf.num_row_groups):  # row-group-wise bounds memory
                    rg = pf.read_row_group(i, columns=cols)
                    kept_rows.extend(rg.filter(pc.is_in(rg.column("file_path"), value_set=warc_arr)).to_pylist())
            finally:
                os.remove(local)
            with gcs.open(out_path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb") as gz:
                for row in kept_rows:
                    gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            yield {"shard": _output_name(url), "status": "done", "survivors": len(kept_rows)}

    ctx = ZephyrContext(
        name="fineweb-cc-filter",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=2, ram="16g", disk="20g"),
    )
    ctx.execute(
        Dataset.from_iterable(shard_urls)
        .map_shard(process_shards)
        .write_jsonl(f"{out_root}/_status/shard-{{shard:05d}}-of-{{total:05d}}.jsonl", skip_existing=False)
    )
    return {"warcs": len(warcs), "snapshots": len(snapshots), "shards_scanned": len(shard_urls)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warc-manifest", required=True, help="WARC list (s3://commoncrawl/...warc.gz per line).")
    parser.add_argument("--output-path", required=True, help="gs:// dir for survivor data-*.jsonl.gz.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Concurrent shard downloads. Keep modest: HF free tier rate-limits at 1000 req/5min "
        "(64 workers tripped it). 16 + backoff stays under.",
    )
    parser.add_argument(
        "--snapshots",
        default=None,
        help="Comma-separated CC-MAIN-YYYY-WW to restrict to. Default: all snapshots our WARCs touch.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Process only the first N shards (a real prefix of the full run; reused via skip-existing).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    snapshots_filter = set(args.snapshots.split(",")) if args.snapshots else None
    result = run_filter(args.warc_manifest, args.output_path, args.max_workers, snapshots_filter, args.limit_shards)
    logger.info("FineWeb-CC filter complete: %s -> %s", result, args.output_path)


if __name__ == "__main__":
    main()
