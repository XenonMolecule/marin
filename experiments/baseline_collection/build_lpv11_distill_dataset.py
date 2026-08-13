# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the llm_pipeline_v1_1 useful-vs-NO_USEFUL classifier dataset (3000-WARC run).

Mirrors ``build_hq_distill_dataset.py`` (same per-WARC map-only stages, same output
schema) with the lpv11-specific differences:

* **Kept set** = the consolidated ``llm_pipeline_v1_1`` extraction (KEEP docs only —
  the pipeline never persists drops). Records carry no ``generated_text``, so
  ``reasoning_trace`` is always empty.
* **Raw HTML**: the 10k pool this run extracted from was deleted
  (``.agents/projects/10k_warc_cleanup.md``); coverage comes from the two surviving
  3000-WARC pools — ``baseline_3000_random-34884d`` (1423 of lpv11's WARCs) and
  ``baseline_3000-265ff5`` (477 more). WARCs with no surviving HTML are skipped.
* **No fed-length cap**: ``run_extract_standalone`` feeds EVERY decoded record to the
  multi-call pipeline (long docs are chunked, not dropped), so every non-empty HTML
  record not in the kept set is a negative.
* **Gap WARCs excluded**: the WARCs in ``missing_batches_llm_pipeline_v1_1.jsonl.gz``
  lost a batch during consolidation; their missing batches' kept docs would be
  mislabeled negative by the set difference, so those WARCs are dropped entirely.
* **Kept-set fix vs the hq build**: the hq assemble deduped by content id *before*
  the set difference, so a kept page whose extracted text duplicated another kept
  page leaked into the negatives. Here assemble emits ALL kept rows; ``htmljoin``
  dedups (first occurrence) when writing ``data/``, and ``negatives`` subtracts the
  full undeduped id set.

Stages (one durable output shard per WARC, ``skip_existing`` → preemption-proof):

``assemble``  (run in us-central1, where the consolidated extraction lives)
``htmljoin``  (run in us-central2, where the HTML pools live) → ``data/``
``negatives`` (run in us-central2) → ``data_no_useful/``
``stats``     (run in us-central2) → ``stats.json`` incl. the natural neg:pos ratio

Usage::

    R=us-central1; S=assemble  # or R=us-central2, S in {htmljoin,negatives,stats}
    uv run iris --cluster marin job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB --priority interactive --extra cpu \\
        --enable-extra-resources --region $R --job-name lpv11-distill-$S \\
        -- python experiments/baseline_collection/build_lpv11_distill_dataset.py $S
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pyarrow as pa
from fray import ResourceConfig
from marin.datakit.normalize import generate_id
from marin.utils import fsspec_glob
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext, counters
from zephyr.readers import load_jsonl

from experiments.baseline_collection.build_hq_distill_dataset import (
    content_id,
    normalize_record_id,
    remap_to_consolidated,
    warc_hash_from_path,
)

logger = logging.getLogger(__name__)

# --- Inputs ----------------------------------------------------------------

_RESOLVED_DIR = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/resolved"
RESOLVED_MANIFEST = f"{_RESOLVED_DIR}/resolved_llm_pipeline_v1_1.jsonl.gz"
MISSING_BATCHES_MANIFEST = f"{_RESOLVED_DIR}/missing_batches_llm_pipeline_v1_1.jsonl.gz"

# Surviving raw-HTML pools, in lookup-priority order. Together they cover 1900 of
# the run's 3000 WARCs; the extraction's own source pool no longer exists.
HTML_DIRS = (
    "gs://marin-us-central2/raw/commoncrawl/baseline_3000_random-34884d",
    "gs://marin-us-central2/raw/commoncrawl/baseline_3000-265ff5",
)

NO_USEFUL_MARKER = "[NO_USEFUL_CONTENT]"

# --- Outputs ---------------------------------------------------------------

DATASET_ROOT = "gs://marin-us-central2/datasets/llm_pipeline_v1_1_3000_distill"

_FINAL_COLUMNS = ("raw_html", "reasoning_trace", "final_output", "url", "warc_file", "warc_record_id", "snapshot")


def _dataset_root(tag: str | None) -> str:
    return DATASET_ROOT if not tag else f"{DATASET_ROOT}_{tag}"


def _meta_staging(tag: str | None) -> str:
    return f"{_dataset_root(tag)}/_staging_metadata"


def _final_data(tag: str | None) -> str:
    return f"{_dataset_root(tag)}/data"


def _no_useful_dir(tag: str | None) -> str:
    return f"{_dataset_root(tag)}/data_no_useful"


# --- Pure helpers (unit-tested in test_build_lpv11_distill_dataset.py) -----


def dedup_by_content(rows: Iterable[dict]) -> list[dict]:
    """Keep the first row per ``content_id`` (deterministic: manifest batch order)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        if r["content_id"] in seen:
            continue
        seen.add(r["content_id"])
        out.append(r)
    return out


def iter_negatives(html_records: Iterable[dict], kept_ids: set[str], warc_file: str, snapshot: str) -> Iterator[dict]:
    """Fed-but-not-kept HTML records as abstention rows (within-WARC html-deduped).

    No length cap: the multi-call pipeline chunks long docs instead of dropping
    them, so every non-empty record it decoded was genuinely fed.
    """
    seen_html: set[str] = set()
    for h in html_records:
        html = h.get("html") or ""
        if not html:
            continue
        nid = normalize_record_id(h.get("id") or "")
        if nid in kept_ids:
            continue
        hid = generate_id(html)
        if hid in seen_html:
            continue
        seen_html.add(hid)
        yield {
            "raw_html": html,
            "reasoning_trace": "",
            "final_output": NO_USEFUL_MARKER,
            "url": h.get("url") or "",
            "warc_file": (h.get("metadata") or {}).get("warc_file") or warc_file,
            "warc_record_id": nid,
            "snapshot": snapshot,
        }


# --- Manifest --------------------------------------------------------------


def _gap_warc_hashes() -> set[str]:
    """WARCs that lost a batch during consolidation (kept docs unrecoverable)."""
    hashes: set[str] = set()
    with fsspec.open(MISSING_BATCHES_MANIFEST, "rb") as f, gzip.open(f, "rt") as gz:
        for line in gz:
            if line.strip():
                hashes.add(json.loads(line)["warc_hash"])
    logger.info("Excluding %d gap WARCs from %s", len(hashes), MISSING_BATCHES_MANIFEST)
    return hashes


def _html_paths_by_hash() -> dict[str, str]:
    """warc_hash → HTML shard path, first pool in HTML_DIRS wins."""
    import re

    paths: dict[str, str] = {}
    for d in HTML_DIRS:
        for f in fsspec_glob(f"{d}/data-*.jsonl.gz"):
            m = re.search(r"data-([0-9a-f]+)\.jsonl\.gz$", f)
            if m:
                paths.setdefault(m.group(1), f)
    if not paths:
        raise RuntimeError(f"no HTML shards found under {HTML_DIRS}")
    logger.info("HTML pools cover %d WARC hashes", len(paths))
    return paths


def _warc_list(path: str) -> set[str]:
    """Read a newline-delimited WARC-hash allowlist (local or gs://)."""
    with fsspec.open(path, "rt", encoding="utf-8") as f:
        hashes = {line.strip() for line in f if line.strip()}
    if not hashes:
        raise RuntimeError(f"empty WARC list: {path}")
    logger.info("restricting to %d WARCs from %s", len(hashes), path)
    return hashes


def load_warcs(limit_warcs: int | None, warc_list: str = "", read_origin: bool = False) -> list[dict]:
    """Covered, gap-free WARCs with their canonical batch paths, sorted by hash.

    ``warc_list`` restricts to an explicit hash allowlist — used to build an
    EXTENSION dataset containing only WARCs absent from an earlier build, so the
    earlier dataset (and the frozen split derived from its shard indices) is
    never renumbered."""
    gaps = _gap_warc_hashes()
    html_paths = _html_paths_by_hash()
    allow = _warc_list(warc_list) if warc_list else None

    by_hash: dict[str, list[str]] = {}
    skipped_empty = 0
    with fsspec.open(RESOLVED_MANIFEST, "rb") as f, gzip.open(f, "rt") as gz:
        for line in gz:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("num_records") or 0) <= 0:
                skipped_empty += 1
                continue
            # read_origin: read the manifest's canonical ORIGIN path instead of the
            # us-central1 archive copy. Required for WARCs that were never mirrored by
            # the transfer step, and cheaper than mirroring when only a subset of the
            # extraction is usable (we pay egress once, for the batches we actually read).
            path = row["path"] if read_origin else remap_to_consolidated(row["path"])
            by_hash.setdefault(warc_hash_from_path(path), []).append(path)
    logger.info("Manifest: %d WARCs with kept docs (skipped %d empty batches)", len(by_hash), skipped_empty)

    no_html = sum(1 for h in by_hash if h not in html_paths)
    warcs = [
        {"warc_hash": h, "paths": sorted(paths), "html_path": html_paths[h]}
        for h, paths in sorted(by_hash.items())
        if h in html_paths and h not in gaps and (allow is None or h in allow)
    ]
    logger.info(
        "Usable WARCs: %d (no surviving HTML: %d, gap-excluded: %d)",
        len(warcs),
        no_html,
        len(by_hash) - no_html - len(warcs),
    )
    if not warcs:
        raise RuntimeError("no usable WARCs (manifest x HTML pools minus gaps is empty)")
    if limit_warcs is not None:
        warcs = warcs[:limit_warcs]
    return warcs


# --- Stage: assemble -------------------------------------------------------


def _warc_to_metadata(warc: dict) -> Iterator[dict]:
    """Yield ALL kept rows of one WARC (no dedup here — the negatives stage needs
    the complete kept-id set; ``htmljoin`` dedups when writing ``data/``)."""
    for path in warc["paths"]:
        for r in load_jsonl(path):
            text = r.get("text") or ""
            if not text:
                continue
            yield {
                "content_id": content_id(text),
                "final_output": text,
                "url": r.get("url") or "",
                "warc_file": r.get("warc_file") or "",
                "warc_record_id": r.get("warc_record_id") or "",
                "snapshot": r.get("snapshot") or "",
                "warc_hash": warc["warc_hash"],
                "html_path": warc["html_path"],
            }


def _assemble_one(spec: tuple[int, int, dict, str]) -> int:
    """Write one WARC's metadata shard (skip-existing, atomic tmp->rename). Returns rows written."""
    idx, total, warc, staging = spec
    out = f"{staging}/data-{idx:05d}-of-{total:05d}.jsonl.gz"
    fs, rpath = fsspec.core.url_to_fs(out)
    if fs.exists(rpath):
        return -1
    tmp = f"{out}.tmp"
    n = 0
    with fsspec.open(tmp, "wt", compression="gzip", encoding="utf-8") as f:
        for row in _warc_to_metadata(warc):
            f.write(json.dumps(row) + "\n")
            n += 1
    tfs, trpath = fsspec.core.url_to_fs(tmp)
    tfs.mv(trpath, rpath)
    return n


def run_assemble_standalone(warcs: list[dict], tag: str | None, workers: int) -> None:
    """Assemble on ONE box with a thread pool instead of a Zephyr fan-out.

    Zephyr needs the scheduler to place up to 200 worker actor-groups; when the CPU
    pools are contended it places one, and the stage crawls. This path is a single
    scheduling decision and parallelises internally — the reads are GCS-bound (the
    GIL is released), so threads scale fine. Same per-WARC output + skip-existing,
    so the two modes are interchangeable and resumable across each other."""
    staging = _meta_staging(tag)
    total = len(warcs)
    specs = [(i, total, w, staging) for i, w in enumerate(warcs)]
    done = skipped = rows = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n in ex.map(_assemble_one, specs):
            done += 1
            if n < 0:
                skipped += 1
            else:
                rows += n
            if done % 25 == 0:
                logger.info("assemble %d/%d WARCs (skipped=%d rows=%d)", done, total, skipped, rows)
    logger.info("standalone assemble DONE %d WARCs (skipped=%d rows=%d) -> %s", total, skipped, rows, staging)


def run_assemble(
    limit_warcs: int | None,
    tag: str | None,
    warc_list: str = "",
    read_origin: bool = False,
    standalone_workers: int = 0,
) -> None:
    warcs = load_warcs(limit_warcs, warc_list, read_origin)
    if standalone_workers:
        run_assemble_standalone(warcs, tag, standalone_workers)
        return
    out_template = f"{_meta_staging(tag)}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz"
    pipeline = Dataset.from_iterable(warcs).flat_map(_warc_to_metadata).write_jsonl(out_template, skip_existing=True)
    ctx = ZephyrContext(name="lpv11-distill-assemble", max_workers=200, resources=ResourceConfig(cpu=2, ram="16g"))
    ctx.execute(pipeline)
    logger.info("assemble done -> %s", _meta_staging(tag))


# --- Stage: htmljoin -------------------------------------------------------


def _enrich(meta: dict, html: str | None) -> dict:
    return {
        "raw_html": html,
        "reasoning_trace": "",
        "final_output": meta["final_output"],
        "url": meta["url"],
        "warc_file": meta["warc_file"],
        "warc_record_id": meta["warc_record_id"],
        "snapshot": meta["snapshot"],
    }


def _join_metadata_file(meta_path: str) -> Iterator[dict]:
    """Join one WARC's (content-deduped) kept rows to its HTML shard."""
    rows = dedup_by_content(load_jsonl(meta_path))
    if not rows:
        return
    need = {r["warc_record_id"]: r for r in rows}
    try:
        html_records = load_jsonl(rows[0]["html_path"])
    except FileNotFoundError:
        counters.increment("lpv11_distill_missing_html_file")
        logger.warning("no HTML shard %s for warc_hash=%s", rows[0]["html_path"], rows[0]["warc_hash"])
        return
    for h in html_records:
        if not need:
            break
        meta = need.pop(normalize_record_id(h.get("id") or ""), None)
        if meta is not None:
            yield _enrich(meta, h.get("html"))
    for meta in need.values():
        counters.increment("lpv11_distill_missing_html_record")
        yield _enrich(meta, None)


def run_htmljoin(tag: str | None) -> None:
    schema = pa.schema([(c, pa.string()) for c in _FINAL_COLUMNS])
    files = sorted(fsspec_glob(f"{_meta_staging(tag)}/*.jsonl.gz"))
    if not files:
        raise RuntimeError(f"no staging metadata under {_meta_staging(tag)}; run assemble first")
    out_template = f"{_final_data(tag)}/data-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_iterable(files)
        .flat_map(_join_metadata_file)
        .write_parquet(out_template, schema=schema, skip_existing=True)
    )
    ctx = ZephyrContext(name="lpv11-distill-htmljoin", max_workers=256, resources=ResourceConfig(cpu=2, ram="24g"))
    ctx.execute(pipeline)
    logger.info("htmljoin done -> %s", _final_data(tag))


# --- Stage: negatives ------------------------------------------------------


def _negatives_for_metadata_file(meta_path: str) -> Iterator[dict]:
    meta = list(load_jsonl(meta_path))
    if not meta:
        return
    kept = {m["warc_record_id"] for m in meta}  # ALL kept ids (pre-dedup) — see module docstring
    try:
        html_records = load_jsonl(meta[0]["html_path"])
    except FileNotFoundError:
        counters.increment("lpv11_distill_neg_missing_html_file")
        return
    yield from iter_negatives(html_records, kept, meta[0]["warc_file"], meta[0]["snapshot"])


def run_negatives(tag: str | None, limit_files: int | None) -> None:
    schema = pa.schema([(c, pa.string()) for c in _FINAL_COLUMNS])
    all_files = sorted(fsspec_glob(f"{_meta_staging(tag)}/*.jsonl.gz"))
    if not all_files:
        raise RuntimeError(f"no staging metadata under {_meta_staging(tag)}; run assemble first")
    total = len(all_files)
    # limit_files processes a prefix whose shard indices match the full run, so
    # the full run's skip_existing reuses smoke output.
    process = all_files[:limit_files] if limit_files else all_files
    out_template = f"{_no_useful_dir(tag)}/data-{{shard:05d}}-of-{total:05d}.parquet"
    pipeline = (
        Dataset.from_iterable(process)
        .flat_map(_negatives_for_metadata_file)
        .write_parquet(out_template, schema=schema, skip_existing=True)
    )
    ctx = ZephyrContext(name="lpv11-distill-negatives", max_workers=256, resources=ResourceConfig(cpu=2, ram="24g"))
    ctx.execute(pipeline)
    logger.info("negatives done (%d/%d WARCs) -> %s", len(process), total, _no_useful_dir(tag))


# --- Standalone (no-Zephyr) parquet stages ---------------------------------


def _write_parquet_shard(rows: Iterator[dict], out: str, schema: pa.Schema, batch: int = 512) -> int:
    """Stream rows into one parquet shard (bounded memory), atomic tmp->rename,
    skip-existing. Rows carry raw_html, so they are NEVER all materialized."""
    import pyarrow.parquet as pq

    fs, rpath = fsspec.core.url_to_fs(out)
    if fs.exists(rpath):
        return -1
    tmp = f"{out}.tmp"
    n = 0
    buf: list[dict] = []
    with fsspec.open(tmp, "wb") as fh:
        writer = pq.ParquetWriter(fh, schema)
        try:
            for r in rows:
                buf.append(r)
                if len(buf) >= batch:
                    writer.write_table(pa.Table.from_pylist(buf, schema=schema))
                    n += len(buf)
                    buf = []
            if buf:
                writer.write_table(pa.Table.from_pylist(buf, schema=schema))
                n += len(buf)
        finally:
            writer.close()
    tfs, trpath = fsspec.core.url_to_fs(tmp)
    tfs.mv(trpath, rpath)
    return n


def _parquet_stage_one(spec: tuple[str, int, int, str, str]) -> int:
    """Module-level (picklable) worker: build one parquet shard. Runs in a SEPARATE
    PROCESS — these stages json-parse ~47k records per WARC and json.loads holds the
    GIL, so threads collapse to ~1 core. Processes use the whole box."""
    stage, idx, total, meta, out_dir = spec
    schema = pa.schema([(c, pa.string()) for c in _FINAL_COLUMNS])
    emit = _join_metadata_file if stage == "htmljoin" else _negatives_for_metadata_file
    return _write_parquet_shard(emit(meta), f"{out_dir}/data-{idx:05d}-of-{total:05d}.parquet", schema)


def run_parquet_stage_standalone(
    stage: str, tag: str | None, workers: int, shard_start: int = 0, shard_end: int | None = None
) -> None:
    """htmljoin/negatives on ONE box with a thread pool (see run_assemble_standalone
    for why: Zephyr fan-out starves on a contended cluster). Same filenames +
    skip-existing as the Zephyr path, so the modes are interchangeable."""
    files = sorted(fsspec_glob(f"{_meta_staging(tag)}/*.jsonl.gz"))
    if not files:
        raise RuntimeError(f"no staging metadata under {_meta_staging(tag)}; run assemble first")
    total = len(files)
    out_dir = _final_data(tag) if stage == "htmljoin" else _no_useful_dir(tag)

    # Slice by GLOBAL index so several boxes can share the stage; the filename index
    # stays global, so slices produce exactly the same names as a single run would.
    work = [(stage, i, total, m, out_dir) for i, m in list(enumerate(files))[shard_start:shard_end]]
    logger.info("%s standalone: shards [%d,%s) of %d, %d procs", stage, shard_start, shard_end, total, workers)
    done = skipped = rows = 0
    ctx = mp.get_context("spawn")  # fresh procs -> no inherited gcsfs/gRPC state
    with ctx.Pool(workers) as pool:
        for n in pool.imap_unordered(_parquet_stage_one, work):
            done += 1
            if n < 0:
                skipped += 1
            else:
                rows += n
            if done % 25 == 0:
                logger.info("%s %d/%d shards (skipped=%d rows=%d)", stage, done, total, skipped, rows)
    logger.info("standalone %s DONE %d shards (skipped=%d rows=%d) -> %s", stage, len(work), skipped, rows, out_dir)


# --- Stage: stats ----------------------------------------------------------


def _parquet_dir_stats(dirpath: str) -> dict | None:
    """Footer-only rows/bytes/shards for a parquet dir, or None if empty."""
    import pyarrow.parquet as pq

    files = sorted(fsspec_glob(f"{dirpath}/*.parquet"))
    if not files:
        return None
    num_rows = size_bytes = 0
    for path in files:
        fs, resolved = fsspec.core.url_to_fs(path)
        size_bytes += fs.size(resolved)
        num_rows += pq.ParquetFile(path).metadata.num_rows
    return {"num_rows": num_rows, "size_bytes": size_bytes, "num_shards": len(files)}


def run_stats(tag: str | None) -> None:
    pos = _parquet_dir_stats(_final_data(tag))
    neg = _parquet_dir_stats(_no_useful_dir(tag))
    if pos is None or neg is None:
        raise RuntimeError("run htmljoin and negatives before stats")
    stats = {
        "positives": pos,
        "negatives": neg,
        # The deployment ratio: train fastText at this neg_per_pos (the dominant
        # lever per the hq classifier findings), never at 1:1.
        "natural_neg_per_pos": round(neg["num_rows"] / max(pos["num_rows"], 1), 3),
    }
    out = f"{_dataset_root(tag)}/stats.json"
    with fsspec.open(out, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("stats -> %s: %s", out, stats)


# --- Entry point -----------------------------------------------------------


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p_assemble = sub.add_parser("assemble", help="Stage 1: emit per-WARC kept-row metadata (run in us-central1).")
    p_assemble.add_argument("--limit-warcs", type=int, default=None, help="Smoke: only the first N WARCs.")
    p_assemble.add_argument(
        "--standalone-workers",
        type=int,
        default=0,
        help="Bypass Zephyr: assemble on THIS box with N threads. Use when the cluster cannot place "
        "Zephyr worker groups (contended CPU pools). Same output + skip-existing as the Zephyr path.",
    )
    p_assemble.add_argument(
        "--read-origin",
        action="store_true",
        help="Read batches from their canonical ORIGIN regional buckets instead of the us-central1 "
        "consolidated archive. Use when the transfer step has not mirrored these WARCs.",
    )
    p_assemble.add_argument(
        "--warc-list",
        default="",
        help="Newline-delimited WARC-hash allowlist (local or gs://). Use to build an EXTENSION "
        "dataset of only-new WARCs so an existing dataset's shard indices (and its frozen "
        "train/val/test split) are never renumbered.",
    )
    p_htmljoin = sub.add_parser("htmljoin", help="Stage 2: join HTML, write data/ parquet (run in us-central2).")
    p_negatives = sub.add_parser(
        "negatives", help="Stage 3: set-difference abstentions to data_no_useful/ (us-central2)."
    )
    for _p in (p_htmljoin, p_negatives):
        _p.add_argument(
            "--standalone-workers",
            type=int,
            default=0,
            help="Bypass Zephyr: run on THIS box with N threads (contended-cluster fallback).",
        )
        _p.add_argument("--shard-start", type=int, default=0, help="Standalone: process shards[start:end].")
        _p.add_argument("--shard-end", type=int, default=None, help="Standalone: process shards[start:end].")
    p_negatives.add_argument("--limit-files", type=int, default=None, help="Smoke: only the first N per-WARC files.")
    sub.add_parser("stats", help="Stage 4: row counts + natural ratio to stats.json (run in us-central2).")
    for p in sub.choices.values():
        p.add_argument("--tag", default=None, help="Suffix the dataset dir (e.g. 'smoke') to isolate test runs.")
    args = parser.parse_args()

    if args.stage == "assemble":
        run_assemble(args.limit_warcs, args.tag, args.warc_list, args.read_origin, args.standalone_workers)
    elif args.stage == "htmljoin":
        if args.standalone_workers:
            run_parquet_stage_standalone("htmljoin", args.tag, args.standalone_workers, args.shard_start, args.shard_end)
        else:
            run_htmljoin(args.tag)
    elif args.stage == "negatives":
        if args.standalone_workers:
            run_parquet_stage_standalone(
                "negatives", args.tag, args.standalone_workers, args.shard_start, args.shard_end
            )
        else:
            run_negatives(args.tag, args.limit_files)
    elif args.stage == "stats":
        run_stats(args.tag)


if __name__ == "__main__":
    main()
