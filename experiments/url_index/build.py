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
import dataclasses
import gc
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from marin.utils import fsspec_exists, fsspec_glob

from experiments.infinigram.provenance import build_provenance_map, content_hash
from experiments.infinigram.resolve import ResolvedTarget, resolve_target
from experiments.infinigram.targets import DATASETS, REGION_BUCKET, Collection, IndexSource, IndexTarget, get_target
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
_BATCH_ROWS = 20_000

# Reclaim Python + fsspec caches every this many shards. Datasets can have tens of
# thousands of shards (nemotron ~24k); without this, per-shard residue accumulates.
_GC_EVERY_SHARDS = 100

# Arrow schema of the per-doc row (unsorted staging + drives every downstream artifact).
_ROW_SCHEMA = pa.schema(
    [
        ("url_key", pa.string()),
        ("domain", pa.string()),
        ("warc_record_id", pa.string()),
        ("snapshot", pa.string()),
        ("warc_file", pa.string()),
        ("text_len", pa.int64()),
        ("url_h", pa.uint64()),
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


def _has_inline_url(shard: str) -> bool:
    """Whether the first record of ``shard`` already carries a ``url`` (no join needed)."""
    for rec in _read_records(shard):
        return bool(rec.get("url"))
    return False


# --- WARC-subset filtering (build a 300-WARC index from a 10k extraction) -----
# Mirrors experiments/baseline_collection/subset_random100_10k.py: restrict a 10k
# tier to a random-N WARC manifest by a per-record join field.

_CC_PREFIXES = ("s3://commoncrawl/", "gs://commoncrawl/", "https://data.commoncrawl.org/")


def _normalize_warc_path(s: str) -> str:
    s = s.strip()
    for p in _CC_PREFIXES:
        if s.startswith(p):
            return s[len(p) :]
    return s


def _load_manifest_warcs(manifest_path: str) -> set[str]:
    """WARC paths from a manifest, dropping blanks and ``#`` provenance headers."""
    with open(manifest_path) as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


@dataclasses.dataclass(frozen=True)
class SubsetFilter:
    """Keep only docs whose ``field`` value is in ``keys`` (the random-N WARC subset)."""

    field: str
    keys: frozenset[str]


def build_subset_filter(field: str, manifest_path: str, metadata_glob: str | None) -> SubsetFilter:
    """A :class:`SubsetFilter` restricting a 10k tier to ``manifest_path``'s WARCs.

    ``file_path``/``warc_file`` tiers carry the WARC path inline -> match the
    manifest directly. ``url``/``warc_record_id`` tiers need the 10k WARC metadata
    (``metadata_glob``) to map subset WARCs -> the field values to keep.
    """
    warc_set = _load_manifest_warcs(manifest_path)
    if not warc_set:
        raise ValueError(f"empty subset manifest {manifest_path}")
    if field in ("file_path", "warc_file"):
        return SubsetFilter(field=field, keys=frozenset(warc_set))
    if metadata_glob is None:
        raise ValueError(f"subset field {field!r} needs --subset-metadata")
    shards = fsspec_glob(metadata_glob)
    if not shards:
        raise ValueError(f"no metadata shards at {metadata_glob}")
    keys: set[str] = set()
    logger.info("Building subset key set (field=%s) from %d metadata shards", field, len(shards))
    for shard in shards:
        for rec in _read_records(shard):
            if rec.get("warc_file") in warc_set:
                v = rec.get(field)
                if v:
                    keys.add(v)
    logger.info("Subset key set: %d %s values for %d WARCs", len(keys), field, len(warc_set))
    return SubsetFilter(field=field, keys=frozenset(keys))


def _read_records(shard: str) -> Iterator[dict]:
    """Stream records from a ``.jsonl(.gz|.zst)`` or ``.parquet`` shard, thread-free.

    ``zephyr.load_file`` opens with ``cache_type='background'``, which spawns a
    gcsfs prefetch thread (+16 MiB buffers) per file. Across tens of thousands of
    shards (nemotron ~24k) those live threads/buffers accumulate and OOM-kill the
    container -- gc cannot reclaim them. We read with ``cache_type='none'`` for
    bounded, prefetch-free sequential streaming.
    """
    fs, path = url_to_fs(shard)
    if shard.endswith(".parquet"):
        with fs.open(path, "rb", cache_type="none") as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches():
                yield from batch.to_pylist()
        return
    compression = "gzip" if shard.endswith(".gz") else "zstd" if shard.endswith(".zst") else None
    with fs.open(path, "rb", cache_type="none", compression=compression) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _warc_path_hash(warc_path: str) -> str:
    """``sha256(warc_path)[:12]`` -- the per-WARC shard key (matches decode_warcs_clean)."""
    return hashlib.sha256(warc_path.encode()).hexdigest()[:12]


def _doc_text(rec: dict) -> str | None:
    return rec.get("text") or rec.get("generated_text")


def _collect_wanted_hashes(shard_urls: tuple[str, ...]) -> set[str]:
    """Text hashes of a text-only tier that need provenance (first of two passes)."""
    wanted: set[str] = set()
    for url in shard_urls:
        for rec in _read_records(url):
            text = _doc_text(rec)
            if text:
                wanted.add(content_hash(text))
    return wanted


def _iter_rows(
    resolved: ResolvedTarget,
    prov_map: dict[str, dict],
    *,
    keys_only: bool = False,
    subset: SubsetFilter | None = None,
    min_field: str | None = None,
    min_value: float = 0.0,
) -> "list[dict]":
    """Yield one key-row per document, recovering url/id from provenance when text-only.

    In ``keys_only`` mode the ``text`` string is dropped after its hash/length are
    computed, so the staging parquet stays small even for ~500M-doc raw tiers.
    When ``subset`` is set, only docs whose ``subset.field`` value is in the subset
    key set are emitted (restricting a 10k tier to a random-N WARC sample). When
    ``min_field`` is set, only docs with ``rec[min_field] >= min_value`` are kept
    (e.g. a fastpipe ModernBERT-prob quality band).
    """
    gcs = fsspec.filesystem("gcs") if resolved.shard_urls and resolved.shard_urls[0].startswith("gs://") else None
    for i, shard in enumerate(resolved.shard_urls):
        for rec in _read_records(shard):
            if subset is not None and rec.get(subset.field) not in subset.keys:
                continue
            if min_field is not None and float(rec.get(min_field, float("-inf"))) < min_value:
                continue
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
                "url_h": u64(uk) if uk else None,
                "rid_h": u64(rid) if rid else None,
                "text_h": text_hash_u64(text),
                "dom_h": u64(dom) if dom else None,
                "text": "" if keys_only else text,
            }
        if gcs is not None and (i + 1) % _GC_EVERY_SHARDS == 0:
            gcs.invalidate_cache()
            gc.collect()
            pa.default_memory_pool().release_unused()


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


_KEYS_COLS = ["url_h", "rid_h", "text_h", "dom_h"]
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


def _resolve(target: IndexTarget, source_glob: str | None, source_warc_manifest: str | None) -> ResolvedTarget:
    """Resolve the target's registry source, or a ``source_glob`` override.

    A ``source_glob`` containing ``{warc_hash}`` selects one per-WARC shard per
    entry of ``source_warc_manifest`` (e.g. fastpipe's kept_text tree) -- reading
    only the manifest's WARCs, not the whole 10k pool.
    """
    if source_glob is None:
        return resolve_target(target)
    if "{warc_hash}" in source_glob:
        if not source_warc_manifest:
            raise ValueError("source-glob with {warc_hash} needs --source-warc-manifest")
        warcs = _load_manifest_warcs(source_warc_manifest)
        candidates = [source_glob.format(warc_hash=_warc_path_hash(w)) for w in warcs]
        shards = tuple(sorted(s for s in candidates if fsspec_exists(s)))
        logger.info("Selected %d/%d per-WARC shards from manifest", len(shards), len(warcs))
    else:
        shards = tuple(sorted(fsspec_glob(source_glob)))
    if not shards:
        raise ValueError(f"--source-glob matched no shards: {source_glob}")
    bucket = REGION_BUCKET[target.region]
    bad = [s for s in shards if not s.startswith(f"{bucket}/")]
    if bad:
        raise ValueError(f"--source-glob shards outside region {target.region}: {bad[:2]}")
    return ResolvedTarget(target=target, shard_urls=shards, shard_bytes=tuple(0 for _ in shards))


def build_target(
    target: IndexTarget,
    *,
    local_root: str,
    overwrite: bool = False,
    keys_only: bool = False,
    source_glob: str | None = None,
    source_warc_manifest: str | None = None,
    subset: SubsetFilter | None = None,
    min_field: str | None = None,
    min_value: float = 0.0,
) -> dict:
    """Build and upload the URL-index artifacts for ``target``. Returns the stats dict."""
    out_dir = layout.index_dir(target.region, target.collection, target.dataset)
    sentinel = layout.KEYS_NAME if keys_only else layout.TEXT_NAME
    if not overwrite and fsspec_glob(f"{out_dir}/{sentinel}"):
        logger.info("%s already built at %s (use --overwrite to rebuild)", target.name, out_dir)
        return {"skipped": True, "out_dir": out_dir}

    resolved = _resolve(target, source_glob, source_warc_manifest)  # region-checked shard URLs

    # Recover url/ids by text-hash join only for text-only tiers. Some tiers of a
    # provenance-carrying dataset (e.g. high_quality's trained 300-WARC subset)
    # keep url inline -- detect that and skip the (expensive) raw-batch join.
    prov_map: dict[str, dict] = {}
    if target.provenance_globs and not _has_inline_url(resolved.shard_urls[0]):
        wanted = _collect_wanted_hashes(resolved.shard_urls)
        prov_map = build_provenance_map(target.provenance_globs, wanted)
        del wanted

    os.makedirs(local_root, exist_ok=True)
    local_out = tempfile.mkdtemp(dir=local_root, prefix="out-")
    with tempfile.NamedTemporaryFile(dir=local_root, suffix=".parquet", delete=False) as tmp:
        staging_path = tmp.name
    try:
        doc_count, url_present = _write_unsorted(
            _iter_rows(resolved, prov_map, keys_only=keys_only, subset=subset, min_field=min_field, min_value=min_value),
            staging_path,
        )
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
            "subset_field": subset.field if subset else None,
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
    p.add_argument("--source-glob", default=None, help="Override the registry source with this gs:// glob.")
    p.add_argument(
        "--source-warc-manifest",
        default=None,
        help="With a {warc_hash} source-glob: select one per-WARC shard per manifest WARC.",
    )
    p.add_argument(
        "--region",
        default=None,
        choices=sorted(REGION_BUCKET),
        help="Region for a dataset not in the registry (requires --source-glob).",
    )
    p.add_argument(
        "--subset-manifest",
        default=None,
        help="Restrict to the WARCs in this manifest (build a random-N subset of a 10k tier).",
    )
    p.add_argument("--subset-field", default=None, help="Join field: file_path | warc_file | url | warc_record_id.")
    p.add_argument("--subset-metadata", default=None, help="10k WARC metadata glob (for url/warc_record_id joins).")
    p.add_argument("--min-field", default=None, help="Keep only docs with rec[min_field] >= min_value (e.g. a band).")
    p.add_argument("--min-value", type=float, default=0.0, help="Threshold for --min-field.")
    return p.parse_args()


def _target_for(dataset: str, collection: Collection, region: str | None, source_glob: str | None) -> IndexTarget:
    """Registry target, or a synthetic one for a dataset built via --source-glob + --region."""
    if dataset in DATASETS:
        return get_target(dataset, collection)
    if not region or not source_glob:
        raise SystemExit(f"unknown dataset {dataset!r}; provide --region and --source-glob to build it")
    return IndexTarget(dataset=dataset, collection=collection, region=region, source=IndexSource.at(source_glob))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    target = _target_for(args.dataset, Collection(args.collection), args.region, args.source_glob)
    subset = None
    if args.subset_manifest:
        if not args.subset_field:
            raise SystemExit("--subset-manifest requires --subset-field")
        subset = build_subset_filter(args.subset_field, args.subset_manifest, args.subset_metadata)
    build_target(
        target,
        local_root=args.local_root,
        overwrite=args.overwrite,
        keys_only=args.keys_only,
        source_glob=args.source_glob,
        source_warc_manifest=args.source_warc_manifest,
        subset=subset,
        min_field=args.min_field,
        min_value=args.min_value,
    )


if __name__ == "__main__":
    main()
