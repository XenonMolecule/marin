# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the URL-index artifacts for one (dataset, collection), in-region.

Streams the dataset's final document tier once (reusing the infinigram registry
and text-hash provenance recovery), reduces each doc to its comparison keys, and
writes ``keys.parquet`` (coverage), ``meta.parquet`` (lookup routing) and
``text.parquet`` (in-region text store, sorted ``(domain, url_key)``) plus a
``stats.json``.

Must run in the dataset's own region (an Iris job pinned there) so every read is
in-region -- ``resolve_target`` refuses otherwise. Run one target::

    python -m experiments.url_index.build --dataset llm_pipeline_v1 --collection small

Memory-bounded: docs stream to a local unsorted parquet in batches, then DuckDB
does the (possibly larger-than-memory, spill-to-disk) sort + projection into the
final artifacts, which are uploaded to GCS.
"""

import argparse
import gc
import json
import logging
import os
import shutil
import tempfile

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from marin.utils import fsspec_glob
from zephyr.readers import load_file

from experiments.infinigram.provenance import build_provenance_map, content_hash
from experiments.infinigram.resolve import ResolvedTarget, resolve_target
from experiments.infinigram.targets import Collection, IndexTarget, get_target
from experiments.url_index import layout
from experiments.url_index.keys import (
    _normalize_record_id,
    domain_of,
    text_hash_u64,
    u64,
    url_key,
)

logger = logging.getLogger(__name__)

# Rows buffered before flushing a row group to the local unsorted parquet.
_BATCH_ROWS = 50_000

# Arrow schema of the per-doc row (unsorted staging + drives every downstream artifact).
_ROW_SCHEMA = pa.schema(
    [
        ("url_key", pa.string()),
        ("domain", pa.string()),
        ("warc_record_id", pa.string()),
        ("snapshot", pa.string()),
        ("warc_file", pa.string()),
        ("text_len", pa.int64()),
        ("rid_h", pa.uint64()),
        ("text_h", pa.uint64()),
        ("dom_h", pa.uint64()),
        ("text", pa.string()),
    ]
)


def _upload(local_path: str, dest: str) -> None:
    """Copy a local file to a gs:// (or local) destination via fsspec."""
    with open(local_path, "rb") as src, fsspec.open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)


def _doc_text(rec: dict) -> str | None:
    return rec.get("text") or rec.get("generated_text")


def _collect_wanted_hashes(shard_urls: tuple[str, ...]) -> set[str]:
    """Text hashes of a text-only tier that need provenance (first of two passes)."""
    wanted: set[str] = set()
    for url in shard_urls:
        for rec in load_file(url):
            text = _doc_text(rec)
            if text:
                wanted.add(content_hash(text))
    return wanted


def _iter_rows(resolved: ResolvedTarget, prov_map: dict[str, dict], *, keys_only: bool = False) -> "list[dict]":
    """Yield one key-row per document, recovering url/id from provenance when text-only.

    In ``keys_only`` mode the ``text`` string is dropped after its hash/length are
    computed, so the staging parquet stays small even for ~500M-doc raw tiers.
    """
    for shard in resolved.shard_urls:
        for rec in load_file(shard):
            text = _doc_text(rec)
            if not text:
                continue
            if prov_map:
                prov = prov_map.get(content_hash(text))
                if prov:
                    rec = {**rec, **prov}
            url = rec.get("url") or ""
            rid = _normalize_record_id(rec.get("warc_record_id") or "")
            uk = url_key(url)
            dom = domain_of(url)
            yield {
                "url_key": uk,
                "domain": dom,
                "warc_record_id": rid,
                "snapshot": rec.get("snapshot") or "",
                "warc_file": rec.get("warc_file") or "",
                "text_len": len(text),
                "rid_h": u64(rid) if rid else None,
                "text_h": text_hash_u64(text),
                "dom_h": u64(dom) if dom else None,
                "text": "" if keys_only else text,
            }


def _write_unsorted(rows_iter, staging_path: str) -> tuple[int, int]:
    """Stream rows to a local unsorted parquet in batches. Returns (doc_count, url_present)."""
    writer = pq.ParquetWriter(staging_path, _ROW_SCHEMA, compression="zstd")
    buf: list[dict] = []
    doc_count = 0
    url_present = 0

    def flush() -> None:
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, schema=_ROW_SCHEMA))
            buf.clear()

    try:
        for row in rows_iter:
            buf.append(row)
            doc_count += 1
            if row["url_key"]:
                url_present += 1
            if len(buf) >= _BATCH_ROWS:
                flush()
        flush()
    finally:
        writer.close()
    return doc_count, url_present


_KEYS_COLS = ["rid_h", "text_h", "dom_h"]
_META_COLS = ["url_key", "domain", "warc_record_id", "snapshot", "warc_file", "text_len"]


def _projected(table: pa.Table, dataset: str, cols: list[str]) -> pa.Table:
    """A ``dataset``-prefixed projection of ``table`` onto ``cols``."""
    dcol = pa.array([dataset] * table.num_rows, type=pa.string())
    arrays = [dcol] + [table.column(c) for c in cols]
    return pa.table(arrays, names=["dataset", *cols])


def _emit_artifacts(staging_path: str, dataset: str, local_out: str, *, keys_only: bool) -> tuple[str, ...]:
    """Stream the staging parquet into the artifacts with pyarrow (bounded memory).

    No sorting and no DuckDB: we read the staging file one row-group batch at a
    time and fan each batch out to ``keys.parquet`` (coverage) and -- unless
    ``keys_only`` -- ``meta.parquet`` (lookup routing) and ``text.parquet``. This
    is O(batch) memory regardless of doc count, which a DuckDB ``COPY`` is not on
    a shared node. The fast lookup path is the ``url_key`` index consolidate.py
    builds on the merged meta, so physical order is irrelevant. ``keys_only`` is
    for huge raw tiers (e.g. resiliparse ~500M docs). Returns the files written.
    """
    pf = pq.ParquetFile(staging_path)
    keys_w = meta_w = text_w = None
    try:
        for batch in pf.iter_batches(batch_size=_BATCH_ROWS):
            table = pa.Table.from_batches([batch])
            keys_tbl = _projected(table, dataset, _KEYS_COLS)
            if keys_w is None:
                keys_w = pq.ParquetWriter(os.path.join(local_out, layout.KEYS_NAME), keys_tbl.schema, compression="zstd")
            keys_w.write_table(keys_tbl)
            if keys_only:
                continue
            meta_tbl = _projected(table, dataset, _META_COLS)
            if meta_w is None:
                meta_w = pq.ParquetWriter(os.path.join(local_out, layout.META_NAME), meta_tbl.schema, compression="zstd")
            meta_w.write_table(meta_tbl)
            text_tbl = _projected(table, dataset, [*_META_COLS, "text"])
            if text_w is None:
                text_w = pq.ParquetWriter(os.path.join(local_out, layout.TEXT_NAME), text_tbl.schema, compression="zstd")
            text_w.write_table(text_tbl)
    finally:
        for w in (keys_w, meta_w, text_w):
            if w is not None:
                w.close()
    return (layout.KEYS_NAME,) if keys_only else (layout.KEYS_NAME, layout.META_NAME, layout.TEXT_NAME)


def build_target(target: IndexTarget, *, local_root: str, overwrite: bool = False, keys_only: bool = False) -> dict:
    """Build and upload the URL-index artifacts for ``target``. Returns the stats dict."""
    out_dir = layout.index_dir(target.region, target.collection, target.dataset)
    sentinel = layout.KEYS_NAME if keys_only else layout.TEXT_NAME
    if not overwrite and fsspec_glob(f"{out_dir}/{sentinel}"):
        logger.info("%s already built at %s (use --overwrite to rebuild)", target.name, out_dir)
        return {"skipped": True, "out_dir": out_dir}

    resolved = resolve_target(target)  # region-checked shard URLs

    prov_map: dict[str, dict] = {}
    if target.provenance_globs:
        wanted = _collect_wanted_hashes(resolved.shard_urls)
        prov_map = build_provenance_map(target.provenance_globs, wanted)
        del wanted

    os.makedirs(local_root, exist_ok=True)
    local_out = tempfile.mkdtemp(dir=local_root, prefix="out-")
    with tempfile.NamedTemporaryFile(dir=local_root, suffix=".parquet", delete=False) as tmp:
        staging_path = tmp.name
    try:
        doc_count, url_present = _write_unsorted(_iter_rows(resolved, prov_map, keys_only=keys_only), staging_path)
        # Free the provenance map (can be GiBs for one-call tiers) before the emit pass.
        prov_map.clear()
        gc.collect()
        names = _emit_artifacts(staging_path, target.dataset, local_out, keys_only=keys_only)

        stats = {
            "dataset": target.dataset,
            "collection": target.collection.value,
            "region": target.region,
            "doc_count": doc_count,
            "url_present": url_present,
            "url_match_rate": round(url_present / doc_count, 6) if doc_count else 0.0,
            "text_only_tier": bool(target.provenance_globs),
            "keys_only": keys_only,
            "shard_count": resolved.shard_count,
            "out_dir": out_dir,
        }
        for name in names:
            _upload(os.path.join(local_out, name), f"{out_dir}/{name}")
        with fsspec.open(f"{out_dir}/{layout.STATS_NAME}", "wt") as f:
            json.dump(stats, f, indent=2)
    finally:
        os.unlink(staging_path)
        shutil.rmtree(local_out, ignore_errors=True)

    logger.info(
        "Built %s: %d docs (url_match_rate=%.4f) -> %s", target.name, doc_count, stats["url_match_rate"], out_dir
    )
    return stats


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build URL-index artifacts for one (dataset, collection).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--collection", required=True, choices=[c.value for c in Collection])
    p.add_argument("--local-root", default="/tmp/url_index", help="Writable local scratch dir.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--keys-only",
        action="store_true",
        help="Only write keys.parquet (coverage), skip the text store. For huge raw tiers.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    target = get_target(args.dataset, Collection(args.collection))
    build_target(target, local_root=args.local_root, overwrite=args.overwrite, keys_only=args.keys_only)


if __name__ == "__main__":
    main()
