# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the llm_pipeline_v1_1 (lpv11) extractor-benchmark export: paired
(HTML -> lpv11 gold text) train/dev/test/big_train sets, mirroring the
high_quality ``medium_12k`` benchmark in ``build_extractor_eval_set.py``.

Source is the CONSOLIDATED lpv11 extraction (3,000 done WARCs spanning 89
snapshots, 2013-2022) — KEEP docs only, pre-dedup. Raw HTML is re-decoded from
Common Crawl with the same record filtering/decoding the extraction used
(``_wanted_html`` mirrors ``download_warcs._download_one_warc``), so the ``html``
field is byte-faithful to what the teacher pipeline consumed.

Splits are whole-WARC-disjoint from each other AND from every WARC the old
high_quality benchmark used (train/dev/test/big_train/dev2/dev3), verified at
the record-id level. dev/test hold out 1 WARC per snapshot (old scheme); train
and big_train are snapshot-stratified draws.

Stages (subcommands), in run order::

    oldids    local          collect record ids of the old benchmark's splits
    oldwarcs  iris central2  map old ids -> the distill parquet WARC hashes they came from
    select    local          snapshot-stratified WARC -> split assignment
    assemble  iris central1  stage per-WARC metadata (text+provenance, no HTML)
                             from the consolidated archive into us-central2
    finalize  local          verify disjointness, sample exact doc ids per split
    join      iris central2  decode WARCs from CC (ttl-cache) + attach raw_html
    export    iris central2  write {split}.jsonl.gz in raw + teacher HTML variants

Egress: only ``assemble`` crosses regions (metadata, ~10 GB, <$1). ``join``
downloads WARCs from Common Crawl inside us-central2 (free ingress, read-through
``tmp/ttl=2d/`` cache). Final artifacts are small enough to pull locally.

Iris launch (CPU stages)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
        --cpu 8 --memory 12GB --priority interactive --extra cpu \
        --region <us-central1|us-central2> --job-name <name> \
        -- python experiments/baseline_collection/build_lpv11_eval_set.py <stage>
"""

import argparse
import gzip
import io
import json
import logging
import random
import re
import signal
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import warcio

from experiments.baseline_collection.download_warcs import (
    _fetch_warc_bytes,
    _s3_to_https,
    _warc_cache_path,
    _warc_path_hash,
)
from experiments.baseline_collection.pipelines.preprocessing import preprocess_html_for_extraction
from experiments.baseline_collection.run_extract_standalone import _normalize_record_id
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

SPEC = "llm_pipeline_v1_1"
SEED = 0

# Region → bucket (mirrors dedup_extracted._REGIONAL_BUCKETS; not imported to keep
# this script off the zephyr/fray dependency stack).
_REGIONAL_BUCKETS: dict[str, str] = {
    "us-central1": "gs://marin-us-central1",
    "us-east1": "gs://marin-us-east1",
    "us-east5": "gs://marin-us-east5",
    "us-west4": "gs://marin-us-west4",
    "europe-west4": "gs://marin-eu-west4",
}

POOL_MANIFEST = "experiments/distill/dclm_400m_1x_warcs.txt"
CONSOLIDATED_ROOT = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated"
DONE_WARCS_URI = f"{CONSOLIDATED_ROOT}/resolved/done_warcs_{SPEC}.txt"
RESOLVED_MANIFEST_URI = f"{CONSOLIDATED_ROOT}/resolved/resolved_{SPEC}.jsonl.gz"

# The old benchmark's two source parquets (per-WARC shards named data-{warc_hash}.parquet).
OLD_DISTILL_DIRS = (
    "gs://marin-us-central2/datasets/high_quality_3000_distill/data",
    "gs://marin-us-central2/datasets/high_quality_3000_distill_random/data",
)
DEFAULT_BENCHMARK_DIR = "~/Documents/School/Stanford/Research/jusText/benchmark"

# Mutable roots: a --tag suffixes all three (smoke runs stay out of the real dirs).
SOURCE_ROOT = "gs://marin-us-central2/datasets/extractor_eval_set/lpv11_source"
OUTPUT_TEACHER = "gs://marin-us-central2/datasets/extractor_eval_set/lpv11_12k"
OUTPUT_RAW = "gs://marin-us-central2/datasets/extractor_eval_set/lpv11_12k_rawhtml"


def _apply_tag(tag: str | None) -> None:
    global SOURCE_ROOT, OUTPUT_TEACHER, OUTPUT_RAW
    if tag:
        SOURCE_ROOT = f"{SOURCE_ROOT}_{tag}"
        OUTPUT_TEACHER = f"{OUTPUT_TEACHER}_{tag}"
        OUTPUT_RAW = f"{OUTPUT_RAW}_{tag}"


N_TRAIN, N_DEV, N_TEST, N_BIG = 10_000, 1_000, 1_000, 100_000
TRAIN_WARCS, BIG_WARCS = 50, 240
BIG_PER_WARC_CAP = 500
HOLDOUT_WARCS_PER_SNAPSHOT = 1  # for EACH of dev and test

SNAPSHOT_RE = re.compile(r"CC-MAIN-\d{4}-\d{2}")
ABSTAIN_MARKER = "[NO_USEFUL_CONTENT]"

HTML_FIELD_DESC = {
    "raw": "raw_html — CC bytes decoded exactly as the extraction saw them (utf-8, errors=replace)",
    "teacher": "preprocess_html_for_extraction(raw_html) — the lpv11 pipeline's own input transform",
}


def _snapshot(warc_path: str) -> str:
    m = SNAPSHOT_RE.search(warc_path)
    if m is None:
        raise ValueError(f"no CC-MAIN snapshot in {warc_path}")
    return m.group(0)


def _read_lines(uri: str) -> list[str]:
    with fsspec.open(uri, "rt") as f:
        return [line.strip() for line in f if line.strip()]


def _read_json(uri: str):
    compression = "gzip" if uri.endswith(".gz") else None
    with fsspec.open(uri, "rt", compression=compression) as f:
        return json.load(f)


def _write_json(uri: str, obj) -> None:
    compression = "gzip" if uri.endswith(".gz") else None
    with fsspec.open(uri, "wt", compression=compression) as f:
        json.dump(obj, f)


def _exists(uri: str) -> bool:
    fs, path = fsspec.core.url_to_fs(uri)
    return fs.exists(path)


# ---------------------------------------------------------------------------
# oldids (local): union of warc_record_ids across every old-benchmark split file.
# ---------------------------------------------------------------------------


def run_oldids(benchmark_dir: str) -> None:
    root = Path(benchmark_dir).expanduser()
    # ALL old split files, including the small domain sets (code/math/science/table) —
    # every record id ever used by the old benchmark is off-limits for the new WARCs.
    files = sorted(
        {
            *root.glob("datasets/*/*.jsonl.gz"),
            *root.glob("datasets_rawhtml/*/*.jsonl.gz"),
            *root.glob("big_train.jsonl.gz"),
            *root.glob("big_train_rawhtml.jsonl.gz"),
        }
    )
    if not files:
        raise FileNotFoundError(f"no old benchmark files under {root}")
    ids: set[str] = set()
    n_missing = 0
    for path in files:
        n_before = len(ids)
        with gzip.open(path, "rt") as f:
            for line in f:
                rid = json.loads(line).get("warc_record_id")
                if rid:
                    ids.add(_normalize_record_id(rid))
                else:  # some domain-set records lack provenance; nothing to exclude on
                    n_missing += 1
        logger.info("%s: +%d ids (total %d)", path.name, len(ids) - n_before, len(ids))
    if n_missing:
        logger.warning("%d old records had no warc_record_id (skipped)", n_missing)
    out = f"{SOURCE_ROOT}/old_benchmark_ids.json.gz"
    _write_json(out, sorted(ids))
    logger.info("wrote %d old-benchmark record ids -> %s", len(ids), out)


# ---------------------------------------------------------------------------
# oldwarcs (iris, us-central2): which distill-parquet WARCs did those ids come from?
# ---------------------------------------------------------------------------


def _shard_used(shard: str, old_ids: set[str]) -> tuple[str, list[str]] | None:
    """If this shard contributed rows to the old benchmark, return (shard, its WARC paths).

    Distill shards are index-named (data-00000-of-03000.parquet), so the WARC identity
    comes from the ``warc_file`` column, not the filename.
    """
    with fsspec.open(shard, "rb") as f:
        table = pq.read_table(f, columns=["warc_record_id", "warc_file"])
    shard_ids = {_normalize_record_id(v) for v in table.column("warc_record_id").to_pylist() if v}
    if not (shard_ids & old_ids):
        return None
    warc_files = sorted({v for v in table.column("warc_file").to_pylist() if v})
    return shard, warc_files


def run_oldwarcs() -> None:
    old_ids = {_normalize_record_id(i) for i in _read_json(f"{SOURCE_ROOT}/old_benchmark_ids.json.gz")}
    shards = [s for d in OLD_DISTILL_DIRS for s in fsspec_glob(f"{d}/*.parquet")]
    logger.info("scanning %d distill shards against %d old ids", len(shards), len(old_ids))
    used_shards: list[str] = []
    used_paths: set[str] = set()
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, hit in enumerate(ex.map(lambda s: _shard_used(s, old_ids), shards)):
            if hit:
                shard, warc_files = hit
                used_shards.append(shard)
                used_paths.update(warc_files)
            if (i + 1) % 500 == 0:
                logger.info("scanned %d/%d shards, %d used", i + 1, len(shards), len(used_shards))
    hashes = sorted({_warc_path_hash(p) for p in used_paths})
    _write_json(
        f"{SOURCE_ROOT}/used_old_warc_hashes.json",
        {"used_shards": used_shards, "used_warc_files": sorted(used_paths), "warc_hashes": hashes},
    )
    logger.info("old benchmark used %d WARCs -> %s/used_old_warc_hashes.json", len(hashes), SOURCE_ROOT)


# ---------------------------------------------------------------------------
# select (local): snapshot-stratified WARC -> split assignment.
# ---------------------------------------------------------------------------


@dataclass
class SelectedWarc:
    warc_hash: str
    warc_path: str
    snapshot: str
    split: str


def _stratified_draw(rng: random.Random, pools: dict[str, list[str]], n: int, split: str, out: list) -> None:
    """Round-robin over shuffled snapshots, popping one WARC per visit until n taken."""
    snaps = sorted(pools)
    rng.shuffle(snaps)
    taken = 0
    while taken < n:
        progressed = False
        for snap in snaps:
            if taken >= n:
                break
            if pools[snap]:
                path = pools[snap].pop()
                out.append(SelectedWarc(_warc_path_hash(path), path, snap, split))
                taken += 1
                progressed = True
        if not progressed:
            raise RuntimeError(f"pool exhausted at {taken}/{n} WARCs for split {split}")


def run_select(limit_warcs: int | None) -> None:
    pool_paths = _read_lines(POOL_MANIFEST)
    hash_to_path = {_warc_path_hash(p): p for p in pool_paths}
    done_hashes = _read_lines(DONE_WARCS_URI)
    used_old = set(_read_json(f"{SOURCE_ROOT}/used_old_warc_hashes.json")["warc_hashes"])

    candidates = [hash_to_path[h] for h in done_hashes if h in hash_to_path and h not in used_old]
    n_excluded = len(done_hashes) - len(candidates)
    logger.info(
        "done=%d, excluded (old-benchmark WARCs)=%d, candidates=%d", len(done_hashes), n_excluded, len(candidates)
    )

    pools: dict[str, list[str]] = defaultdict(list)
    for p in candidates:
        pools[_snapshot(p)].append(p)
    rng = random.Random(SEED)
    for snap in pools:
        pools[snap].sort()
        rng.shuffle(pools[snap])
    thin = {s: len(v) for s, v in pools.items() if len(v) < 2 * HOLDOUT_WARCS_PER_SNAPSHOT + 1}
    if thin:
        raise RuntimeError(f"snapshots too thin for dev+test holdout: {thin}")

    selected: list[SelectedWarc] = []
    for split in ("dev", "test"):  # 1 WARC per snapshot each, old holdout scheme
        for snap in sorted(pools):
            path = pools[snap].pop()
            selected.append(SelectedWarc(_warc_path_hash(path), path, snap, split))
    _stratified_draw(rng, pools, TRAIN_WARCS, "train", selected)
    _stratified_draw(rng, pools, BIG_WARCS, "big_train", selected)

    if limit_warcs:  # smoke mode: keep a tiny per-split prefix
        by_split: dict[str, list[SelectedWarc]] = defaultdict(list)
        for w in selected:
            if len(by_split[w.split]) < limit_warcs:
                by_split[w.split].append(w)
        selected = [w for ws in by_split.values() for w in ws]

    counts = Counter(w.split for w in selected)
    manifest = {
        "spec": SPEC,
        "seed": SEED,
        "holdout_warcs_per_snapshot": HOLDOUT_WARCS_PER_SNAPSHOT,
        "excluded_old_benchmark_warcs": n_excluded,
        "warc_split": dict(counts),
        "snapshots": len({w.snapshot for w in selected}),
        "warcs": [w.__dict__ for w in selected],
    }
    _write_json(f"{SOURCE_ROOT}/selection_warcs.json", manifest)
    logger.info(
        "selected %s over %d snapshots -> %s/selection_warcs.json", dict(counts), manifest["snapshots"], SOURCE_ROOT
    )


# ---------------------------------------------------------------------------
# assemble (iris, us-central1): stage per-WARC metadata into us-central2.
# ---------------------------------------------------------------------------

_REGIONAL_TO_ARCHIVE = {
    f"{bucket}/documents/baseline_llm_extraction/{SPEC}": f"{CONSOLIDATED_ROOT}/by_region/{region}/{SPEC}"
    for region, bucket in _REGIONAL_BUCKETS.items()
}


def _archive_path(regional_path: str) -> str:
    for src, dst in _REGIONAL_TO_ARCHIVE.items():
        if regional_path.startswith(src):
            return dst + regional_path[len(src) :]
    raise ValueError(f"unmapped regional path: {regional_path}")


def _assemble_one(warc: dict, winner_paths: list[str]) -> int:
    meta_uri = f"{SOURCE_ROOT}/staged_metadata/{warc['warc_hash']}.jsonl.gz"
    sidecar_uri = f"{SOURCE_ROOT}/sidecars/{warc['warc_hash']}.json.gz"
    if _exists(sidecar_uri):  # sidecar written last => metadata is complete
        return 0
    rows: list[dict] = []
    for path in winner_paths:
        with fsspec.open(_archive_path(path), "rb", compression="gzip") as f:
            for line in f:
                rec = json.loads(line)
                text = rec.get("text") or ""
                if not text.strip() or text.strip() == ABSTAIN_MARKER:
                    continue
                rows.append(
                    {
                        "warc_record_id": _normalize_record_id(rec.get("warc_record_id", "")),
                        "text": text,
                        "url": rec.get("url", ""),
                        "warc_file": rec.get("warc_file", ""),
                        "snapshot": rec.get("snapshot", ""),
                        "pipeline_id": rec.get("pipeline_id", SPEC),
                        "num_chunks": rec.get("num_chunks"),
                    }
                )
    with fsspec.open(meta_uri, "wt", compression="gzip") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(
        sidecar_uri, {"warc_hash": warc["warc_hash"], "n_docs": len(rows), "ids": [r["warc_record_id"] for r in rows]}
    )
    return len(rows)


def run_assemble(workers: int) -> None:
    selection = _read_json(f"{SOURCE_ROOT}/selection_warcs.json")
    warcs = selection["warcs"]
    winners: dict[str, list[str]] = defaultdict(list)
    wanted = {w["warc_hash"] for w in warcs}
    with fsspec.open(RESOLVED_MANIFEST_URI, "rt", compression="gzip") as f:
        for line in f:
            rec = json.loads(line)
            if rec["warc_hash"] in wanted and rec["is_valid"]:
                winners[rec["warc_hash"]].append(rec["path"])
    missing = wanted - set(winners)
    if missing:
        raise RuntimeError(f"{len(missing)} selected WARCs have no winner batches: {sorted(missing)[:5]}")

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_assemble_one, w, sorted(winners[w["warc_hash"]])): w for w in warcs}
        for i, fut in enumerate(futures):
            total += fut.result()
            if (i + 1) % 25 == 0:
                logger.info("assembled %d/%d WARCs", i + 1, len(warcs))
    logger.info("assemble complete: %d WARCs, %d newly staged docs", len(warcs), total)


# ---------------------------------------------------------------------------
# finalize (local): verify disjointness, sample exact doc ids per split.
# ---------------------------------------------------------------------------


def run_finalize() -> None:
    selection = _read_json(f"{SOURCE_ROOT}/selection_warcs.json")
    old_ids = {_normalize_record_id(i) for i in _read_json(f"{SOURCE_ROOT}/old_benchmark_ids.json.gz")}
    warcs = selection["warcs"]

    def _sidecar(w: dict) -> tuple[str, list[str]]:
        data = _read_json(f"{SOURCE_ROOT}/sidecars/{w['warc_hash']}.json.gz")
        return w["warc_hash"], data["ids"]

    with ThreadPoolExecutor(max_workers=32) as ex:
        ids_by_hash = dict(ex.map(_sidecar, warcs))

    # Whole-WARC disjointness vs the old benchmark, verified at the doc level.
    collisions = {h: len(set(ids) & old_ids) for h, ids in ids_by_hash.items() if set(ids) & old_ids}
    if collisions:
        raise RuntimeError(f"{len(collisions)} selected WARCs share record ids with the old benchmark: {collisions}")

    rng = random.Random(SEED + 11)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for w in warcs:
        by_split[w["split"]].append(w)

    chosen: dict[str, dict] = {}  # warc_hash -> {split, ids}

    def _sample_union(split: str, target: int, per_warc_cap: int | None) -> None:
        pool: list[tuple[str, str]] = []
        for w in by_split[split]:
            ids = ids_by_hash[w["warc_hash"]]
            if per_warc_cap is not None and len(ids) > per_warc_cap:
                ids = rng.sample(ids, per_warc_cap)
            pool.extend((w["warc_hash"], i) for i in ids)
        if len(pool) < target:
            logger.warning("split %s wanted %d docs but pool has %d — taking all", split, target, len(pool))
        picked = pool if len(pool) <= target else rng.sample(pool, target)
        per_warc: dict[str, list[str]] = defaultdict(list)
        for h, i in picked:
            per_warc[h].append(i)
        for w in by_split[split]:
            chosen[w["warc_hash"]] = {"split": split, "ids": sorted(per_warc.get(w["warc_hash"], []))}
        logger.info("split %s: %d docs across %d/%d WARCs", split, len(picked), len(per_warc), len(by_split[split]))

    _sample_union("dev", N_DEV, None)
    _sample_union("test", N_TEST, None)
    _sample_union("train", N_TRAIN, None)
    _sample_union("big_train", N_BIG, BIG_PER_WARC_CAP)

    counts = Counter(v["split"] for v in chosen.values() for _ in v["ids"])
    _write_json(f"{SOURCE_ROOT}/doc_selection.json.gz", {"seed": SEED, "counts": dict(counts), "warcs": chosen})
    logger.info("doc selection: %s -> %s/doc_selection.json.gz", dict(counts), SOURCE_ROOT)


# ---------------------------------------------------------------------------
# join (iris, us-central2): decode WARC -> attach raw_html to the chosen docs.
# ---------------------------------------------------------------------------


CACHE_COPY_CHUNK = 16 * 1024 * 1024


def _streamable_warc(warc_path: str) -> str | None:
    """Ensure the WARC's raw bytes sit in the in-region ``tmp/ttl=2d`` cache and
    return the cache URI, or None to use the in-memory fallback.

    Streams CC -> GCS in chunks so peak memory stays ~one chunk: the in-memory
    path (full bytes + BytesIO copy) OOM'd 8GB join jobs on >1GB WARCs. Bytes are
    the verbatim CC response either way.
    """
    cache = _warc_cache_path(warc_path)
    if cache is None:
        return None
    try:
        if _exists(cache):
            return cache
        with requests.get(_s3_to_https(warc_path), stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with fsspec.open(cache, "wb") as out:
                for chunk in resp.iter_content(CACHE_COPY_CHUNK):
                    out.write(chunk)
        return cache
    except Exception as e:
        logger.warning("stream-cache failed for %s (%s) — using in-memory fallback", warc_path, e)
        return None


def _wanted_html(warc_path: str, want: set[str]) -> dict[str, str]:
    """Decode a WARC keeping ONLY the wanted record ids' HTML.

    Mirrors ``download_warcs._download_one_warc`` record-for-record (response type,
    text/html content-type, utf-8 errors=replace decode) but drops unwanted pages
    immediately, reading the WARC as a stream from the in-region cache when possible.
    """
    cache = _streamable_warc(warc_path)
    if cache is not None:
        with fsspec.open(cache, "rb") as fh:
            return _scan_warc_stream(fh, want, warc_path)
    return _scan_warc_stream(io.BytesIO(_fetch_warc_bytes(warc_path)), want, warc_path)


def _scan_warc_stream(stream, want: set[str], warc_path: str) -> dict[str, str]:
    html_by_id: dict[str, str] = {}
    parse_errors = 0
    for record in warcio.ArchiveIterator(stream):
        try:
            if record.rec_type != "response" or record.http_headers is None:
                continue
            if "text/html" not in (record.http_headers.get_header("Content-Type") or "").lower():
                continue
            rid = _normalize_record_id(record.rec_headers.get_header("WARC-Record-ID") or "")
            if rid not in want:
                continue
            html_by_id[rid] = record.content_stream().read().decode("utf-8", errors="replace")
        except Exception:  # corrupt individual records: skip, matching the extraction's decoder
            parse_errors += 1
    if parse_errors:
        logger.warning("%s: skipped %d corrupt records", warc_path, parse_errors)
    return html_by_id


JOINED_SCHEMA = pa.schema(
    [
        ("raw_html", pa.string()),
        ("final_output", pa.string()),
        ("url", pa.string()),
        ("warc_record_id", pa.string()),
        ("warc_file", pa.string()),
        ("snapshot", pa.string()),
        ("pipeline_id", pa.string()),
        ("num_chunks", pa.int32()),
        ("split", pa.string()),
        ("warc_hash", pa.string()),
    ]
)


def _join_one(warc: dict, chosen: dict) -> tuple[int, int]:
    out_uri = f"{SOURCE_ROOT}/joined/{warc['warc_hash']}.parquet"
    if _exists(out_uri):
        return 0, 0
    entry = chosen.get(warc["warc_hash"])
    if entry is None or not entry["ids"]:
        _write_json(f"{SOURCE_ROOT}/joined/{warc['warc_hash']}.empty.json", {"n": 0})
        return 0, 0
    want = set(entry["ids"])

    meta_rows = []
    with fsspec.open(f"{SOURCE_ROOT}/staged_metadata/{warc['warc_hash']}.jsonl.gz", "rt", compression="gzip") as f:
        for line in f:
            rec = json.loads(line)
            if rec["warc_record_id"] in want:
                meta_rows.append(rec)

    html_by_id = _wanted_html(warc["warc_path"], want)

    rows = []
    for rec in meta_rows:
        html = html_by_id.get(rec["warc_record_id"])
        if html is None:
            continue
        rows.append(
            {
                "raw_html": html,
                "final_output": rec["text"],
                "url": rec["url"],
                "warc_record_id": rec["warc_record_id"],
                "warc_file": rec["warc_file"],
                "snapshot": rec["snapshot"],
                "pipeline_id": rec["pipeline_id"],
                "num_chunks": rec["num_chunks"] or 0,
                "split": entry["split"],
                "warc_hash": warc["warc_hash"],
            }
        )
    table = pa.Table.from_pylist(rows, schema=JOINED_SCHEMA)
    with fsspec.open(out_uri, "wb") as f:
        pq.write_table(table, f, compression="zstd")
    return len(rows), len(want) - len(rows)


def run_join(workers: int, shard_index: int, num_shards: int) -> None:
    selection = _read_json(f"{SOURCE_ROOT}/selection_warcs.json")
    chosen = _read_json(f"{SOURCE_ROOT}/doc_selection.json.gz")["warcs"]
    warcs = [w for i, w in enumerate(selection["warcs"]) if i % num_shards == shard_index]
    logger.info("join shard %d/%d: %d WARCs", shard_index, num_shards, len(warcs))
    joined = unmatched = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_join_one, w, chosen) for w in warcs]
        for i, fut in enumerate(futures):
            n, miss = fut.result()
            joined += n
            unmatched += miss
            if (i + 1) % 10 == 0:
                logger.info("joined %d/%d WARCs (%d docs, %d unmatched)", i + 1, len(warcs), joined, unmatched)
    logger.info("join complete: %d docs joined, %d chosen ids had no HTML record", joined, unmatched)


# ---------------------------------------------------------------------------
# export (iris, us-central2): split jsonl.gz files in both HTML variants.
# ---------------------------------------------------------------------------


def _content_flags(teacher_html: str) -> dict[str, bool]:
    lowered = teacher_html.lower()
    return {
        "has_table": "<table" in lowered,
        "has_code": "<code" in lowered or "<pre" in lowered,
        "has_list": "<ul" in lowered or "<ol" in lowered,
    }


PREPROCESS_TIMEOUT = 90  # seconds per page; the regex tail is unbounded on pathological pages


class _PreprocessTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _PreprocessTimeout


def _teacher_and_flags(raw_html: str) -> tuple[str, dict[str, bool], bool]:
    """Per-row transform, in a worker process: the preprocess regexes have a brutal
    tail (29s measured on one page; backtracking is unbounded in principle), so each
    page gets a SIGALRM cap — on timeout the teacher field falls back to the raw
    HTML and the row is counted in the manifest's ``preprocess_timeouts``."""
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(PREPROCESS_TIMEOUT)
    try:
        teacher = preprocess_html_for_extraction(raw_html or "")
        timed_out = False
    except _PreprocessTimeout:
        teacher = raw_html or ""
        timed_out = True
    finally:
        signal.alarm(0)
    return teacher, _content_flags(teacher), timed_out


def run_export(workers: int) -> None:
    shards = sorted(fsspec_glob(f"{SOURCE_ROOT}/joined/*.parquet"))
    logger.info("exporting from %d joined shards", len(shards))
    rng = random.Random(SEED + 23)
    rng.shuffle(shards)

    sinks: dict[tuple[str, str], TextIO] = {}  # (variant_root, split) -> open gzip text sink

    def _sink(root: str, split: str) -> TextIO:
        key = (root, split)
        if key not in sinks:
            sinks[key] = fsspec.open(f"{root}/{split}.jsonl.gz", "wt", compression="gzip").open()
        return sinks[key]

    counts: dict[str, int] = Counter()
    snap_dist: dict[str, Counter] = defaultdict(Counter)
    warc_split: dict[str, set] = defaultdict(set)
    seen_ids: set[str] = set()  # pre-dedup extraction can re-emit a record under two batch indices
    n_dups = 0
    n_timeouts = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for shard_idx, shard in enumerate(shards):
                with fsspec.open(shard, "rb") as f:
                    rows = pq.read_table(f).to_pylist()
                rng.shuffle(rows)
                transformed = pool.map(_teacher_and_flags, (r["raw_html"] for r in rows), chunksize=8)
                for row, (teacher, flags, timed_out) in zip(rows, transformed, strict=True):
                    if timed_out:
                        n_timeouts += 1
                        logger.warning("preprocess timeout: %s (%s)", row["warc_record_id"], row["url"][:80])
                    if row["warc_record_id"] in seen_ids:
                        n_dups += 1
                        continue
                    seen_ids.add(row["warc_record_id"])
                    split = row["split"]
                    base = {
                        "final_output": row["final_output"],
                        "url": row["url"],
                        "warc_record_id": row["warc_record_id"],
                        "snapshot": row["snapshot"],
                        **flags,
                        "split": split,
                        "warc_hash": row["warc_hash"],
                    }
                    _sink(OUTPUT_RAW, split).write(
                        json.dumps({"html": row["raw_html"], **base}, ensure_ascii=False) + "\n"
                    )
                    _sink(OUTPUT_TEACHER, split).write(json.dumps({"html": teacher, **base}, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    snap_dist[split][row["snapshot"]] += 1
                    warc_split[split].add(row["warc_hash"])
                if (shard_idx + 1) % 20 == 0:
                    logger.info("exported %d/%d shards (%s)", shard_idx + 1, len(shards), dict(counts))
                    # Observable heartbeat: sinks stay invisible until close, and job
                    # log fetching is unreliable — this tiny file is the progress signal.
                    _write_json(
                        f"{SOURCE_ROOT}/export_progress.json",
                        {"shards_done": shard_idx + 1, "total": len(shards), "counts": dict(counts)},
                    )
    finally:
        for s in sinks.values():
            s.close()

    for root, variant in ((OUTPUT_RAW, "raw"), (OUTPUT_TEACHER, "teacher")):
        manifest = {
            "source": f"{SOURCE_ROOT}/joined",
            "spec": SPEC,
            "scope": "lpv11 KEEP docs only (pre-dedup); WARC-disjoint splits, disjoint from the old hq benchmark",
            "html_field": HTML_FIELD_DESC[variant],
            "gold_field": "final_output (lpv11 `text`)",
            "flags_basis": "teacher-input html (preprocess_html_for_extraction)",
            "seed": SEED,
            "holdout_warcs_per_snapshot": HOLDOUT_WARCS_PER_SNAPSHOT,
            "big_train_per_warc_cap": BIG_PER_WARC_CAP,
            "order": "shard-shuffled, row-shuffled within shard (seeded)",
            "preprocess_timeouts": n_timeouts,
            "counts": dict(counts),
            "warc_split": {k: len(v) for k, v in warc_split.items()},
            "snapshot_distribution": {k: dict(sorted(v.items())) for k, v in snap_dist.items()},
        }
        _write_json(f"{root}/manifest.json", manifest)
        logger.info("wrote %s/manifest.json", root)
    if n_dups:
        logger.info("dropped %d duplicate warc_record_ids", n_dups)
    logger.info("export complete: %s", dict(counts))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["oldids", "oldwarcs", "select", "assemble", "finalize", "join", "export"])
    ap.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR, help="oldids: local jusText benchmark dir")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="thread count (default: 16 for assemble, 4 for join — each in-flight WARC holds ~1-2GB)",
    )
    ap.add_argument("--limit-warcs", type=int, default=None, help="select: smoke mode, WARCs per split")
    ap.add_argument("--tag", default=None, help="suffix for all GCS roots (smoke runs)")
    ap.add_argument("--shard-index", type=int, default=0, help="join: this job's shard (0-based)")
    ap.add_argument("--num-shards", type=int, default=1, help="join: total parallel join jobs")
    args = ap.parse_args()
    _apply_tag(args.tag)

    if args.stage == "oldids":
        run_oldids(args.benchmark_dir)
    elif args.stage == "oldwarcs":
        run_oldwarcs()
    elif args.stage == "select":
        run_select(args.limit_warcs)
    elif args.stage == "assemble":
        run_assemble(args.workers or 16)
    elif args.stage == "finalize":
        run_finalize()
    elif args.stage == "join":
        run_join(args.workers or 2, args.shard_index, args.num_shards)
    else:
        run_export(args.workers or 8)


if __name__ == "__main__":
    main()
