# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample documents that survived BOTH lpv11 (Generative Curation) and
lpv11_fastpipe_v2 (Generative Curation Cascade), paired with their raw HTML, as
input for the lpv11 hallucination eval (small-rephraser
``scripts/run_lpv11_hallucination_eval.py``, which reads ``html`` + ``final_output``).

Both corpora are the post-dedup, post-decon survivor trees over the same 10,364
WARCs (``grid_corpora.lpv11_10k`` / ``lpv11_fastpipe_v2_10k``, us-east5). They are
joined on the normalized WARC record id (``warc_record_id`` in lpv11, ``doc_id`` in
fastpipe; both bare UUIDs). The ``final_output`` field is lpv11's ``text``.

The draw is WARC-stratified: ``--num-warcs`` WARCs are chosen uniformly from the
pool, then ``--docs-per-warc`` docs uniformly from each WARC's intersection. Raw
HTML is re-decoded from Common Crawl, so the per-WARC layout bounds the WARC
downloads at ``--num-warcs`` instead of one per doc. Per-doc selection uses a
seeded hash priority (bottom-K per WARC), so a shard-parallel scan yields the same
sample as a single-process scan.

Stages (subcommands), in run order, all in us-east5::

    intersect   scan both trees once; write sample_docs.jsonl.gz (no HTML) + stats
    join        decode the chosen WARCs from CC (ttl-cache) and attach html
    export      merge joined shards into sample.jsonl.gz + manifest.json

``join --num-warcs N`` and ``export --num-warcs N --docs-per-warc K`` take a
smaller sample out of a larger ``intersect`` (the first N WARCs of the seeded
selection, the first K docs of each) without rescanning the trees.

Iris launch::

    uv run iris --cluster marin job run --no-wait --region us-east5 \\
        --cpu 8 --memory 14GB --enable-extra-resources --priority interactive --job-name <name> \\
        -- python -m experiments.baseline_collection.sample_lpv11_intersection <stage>
"""

import argparse
import gzip
import hashlib
import heapq
import json
import logging
import multiprocessing
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field

import fsspec
import numpy as np

from experiments.baseline_collection.build_lpv11_eval_set import _wanted_html
from experiments.baseline_collection.download_warcs import _warc_path_hash
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

LPV11_DEDUPED = "gs://marin-us-east5/documents/baseline_llm_pipeline_v1_1_deduped/10364warcs/deduped"
FASTPIPE_DEDUPED = "gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_decon_deduped/10364warcs/deduped"
POOL_MANIFEST = "experiments/distill/dclm_400m_1x_warcs.txt"
OUT_ROOT = "gs://marin-us-east5/users/michaelryan/analysis/lpv11_halluc_10k"

SEED = 0
NUM_WARCS = 500
DOCS_PER_WARC = 20

SNAPSHOT_RE = re.compile(r"CC-MAIN-\d{4}-\d{2}")
# Cheap pre-filters so shards are only json-parsed for records in chosen WARCs.
FASTPIPE_HASH_RE = re.compile(rb'"warc_hash":\s*"([0-9a-f]{12})"')
LPV11_WARC_RE = re.compile(rb'"warc_file":\s*"([^"]+)"')
FASTPIPE_SCORE_FIELDS = ("fasttext_score", "pooled_prob", "modernbert_prob")


@dataclass
class ShardCounts:
    """Per-WARC tallies from one scan pass."""

    fastpipe: Counter = field(default_factory=Counter)
    lpv11: Counter = field(default_factory=Counter)
    both: Counter = field(default_factory=Counter)


def _snapshot(warc_path: str) -> str:
    m = SNAPSHOT_RE.search(warc_path)
    if m is None:
        raise ValueError(f"no CC-MAIN snapshot in {warc_path}")
    return m.group(0)


def _priority(seed: int, record_id: str) -> str:
    """Seeded per-doc rank; bottom-K by this value is a uniform K-subset."""
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def select_warcs(manifest_lines: list[str], num_warcs: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen = rng.sample(manifest_lines, num_warcs)
    return [{"warc_path": p, "warc_hash": _warc_path_hash(p), "snapshot": _snapshot(p)} for p in chosen]


def merge_bottom_k(running: dict[str, list[tuple[str, dict]]], part: dict[str, list[tuple[str, dict]]], k: int) -> None:
    """Fold one worker's per-WARC bottom-K heaps into the running global bottom-K.

    Done per shard as results arrive so the parent never holds more than K rows
    (with text) per WARC; accumulating every shard's partials first OOMed a
    14 GB job at a sixth of the lpv11 tree.
    """
    for warc_hash, items in part.items():
        pooled = running.get(warc_hash, []) + items
        running[warc_hash] = heapq.nsmallest(k, pooled, key=lambda t: t[0])


@dataclass
class IdIndex:
    """Fastpipe survivor ids (64-bit hashed, sorted) with their scores.

    Two numpy arrays instead of a dict of ~4M Python objects: forked workers
    share the pages copy-on-write without refcount churn duplicating them.
    """

    keys: np.ndarray  # uint64, sorted
    scores: np.ndarray  # float64 (n, 3), NaN for null

    def lookup(self, record_id: str) -> tuple[float | None, ...] | None:
        key = _id_key(record_id)
        i = int(np.searchsorted(self.keys, key))
        if i >= len(self.keys) or self.keys[i] != key:
            return None
        return tuple(None if np.isnan(s) else float(s) for s in self.scores[i])


def _id_key(record_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(record_id.encode(), digest_size=8).digest(), "little")


def build_id_index(parts: list[tuple[np.ndarray, np.ndarray]]) -> IdIndex:
    keys = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, dtype=np.uint64)
    scores = np.concatenate([p[1] for p in parts]) if parts else np.zeros((0, 3), dtype=np.float64)
    order = np.argsort(keys, kind="stable")
    return IdIndex(keys[order], scores[order])


# Set once in the parent before the fork-based pool starts; read-only in workers.
_CHOSEN: dict[str, str] = {}
_IDS = IdIndex(np.zeros(0, dtype=np.uint64), np.zeros((0, 3), dtype=np.float64))
_SEED = SEED
_K = DOCS_PER_WARC


def _open_lines(path: str):
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rb") as f:
        yield from f


def _scan_fastpipe_shard(path: str) -> tuple[np.ndarray, np.ndarray, Counter]:
    keys: list[int] = []
    scores: list[list[float]] = []
    counts: Counter = Counter()
    for line in _open_lines(path):
        m = FASTPIPE_HASH_RE.search(line)
        if m is None:
            raise ValueError(f"{path}: record without warc_hash")
        warc_hash = m.group(1).decode()
        if warc_hash not in _CHOSEN:
            continue
        rec = json.loads(line)
        counts[warc_hash] += 1
        keys.append(_id_key(rec["doc_id"]))
        scores.append([float("nan") if rec.get(f) is None else rec[f] for f in FASTPIPE_SCORE_FIELDS])
    return np.array(keys, dtype=np.uint64), np.array(scores, dtype=np.float64).reshape(-1, 3), counts


def _scan_lpv11_shard(path: str) -> tuple[dict[str, list[tuple[str, dict]]], Counter, Counter]:
    heaps: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    n_lpv11: Counter = Counter()
    n_both: Counter = Counter()
    hash_cache: dict[bytes, str] = {}
    for line in _open_lines(path):
        m = LPV11_WARC_RE.search(line)
        if m is None:
            raise ValueError(f"{path}: record without warc_file")
        warc_file = m.group(1)
        warc_hash = hash_cache.get(warc_file)
        if warc_hash is None:
            warc_hash = hash_cache[warc_file] = _warc_path_hash(warc_file.decode())
        if warc_hash not in _CHOSEN:
            continue
        rec = json.loads(line)
        n_lpv11[warc_hash] += 1
        scores = _IDS.lookup(rec["warc_record_id"])
        if scores is None:
            continue
        n_both[warc_hash] += 1
        row = {
            "warc_hash": warc_hash,
            "warc_path": rec["warc_file"],
            "warc_record_id": rec["warc_record_id"],
            "url": rec["url"],
            "snapshot": rec["snapshot"],
            "num_chunks": rec["num_chunks"],
            "final_output": rec["text"],
            **dict(zip(FASTPIPE_SCORE_FIELDS, scores, strict=True)),
        }
        heap = heaps[warc_hash]
        item = (_priority(_SEED, rec["warc_record_id"]), row)
        if len(heap) < _K:
            heap.append(item)
        elif item[0] < max(heap, key=lambda t: t[0])[0]:
            heap.remove(max(heap, key=lambda t: t[0]))
            heap.append(item)
    return dict(heaps), n_lpv11, n_both


def _write_json(uri: str, obj) -> None:
    with fsspec.open(uri, "wt") as f:
        json.dump(obj, f, indent=2)


def _read_json(uri: str):
    with fsspec.open(uri, "rt") as f:
        return json.load(f)


def _shards(root: str, limit: int | None) -> list[str]:
    paths = sorted(fsspec_glob(f"{root}/data-*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no shards under {root}")
    return paths[:limit] if limit else paths


def run_intersect(num_warcs: int, docs_per_warc: int, seed: int, workers: int, limit_shards: int | None) -> None:
    global _CHOSEN, _IDS, _SEED, _K
    _SEED, _K = seed, docs_per_warc
    with open(POOL_MANIFEST) as f:
        manifest = [line.strip() for line in f if line.strip()]
    selection = select_warcs(manifest, num_warcs, seed)
    _CHOSEN = {w["warc_hash"]: w["warc_path"] for w in selection}
    _write_json(f"{OUT_ROOT}/selection_warcs.json", {"seed": seed, "pool": POOL_MANIFEST, "warcs": selection})
    logger.info("selected %d/%d WARCs (seed %d)", len(selection), len(manifest), seed)

    ctx = multiprocessing.get_context("fork")
    counts = ShardCounts()

    fastpipe_shards = _shards(FASTPIPE_DEDUPED, limit_shards)
    logger.info("pass A: %d fastpipe shards", len(fastpipe_shards))
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for i, (keys, scores, c) in enumerate(pool.map(_scan_fastpipe_shard, fastpipe_shards, chunksize=4)):
            counts.fastpipe.update(c)
            parts.append((keys, scores))
            if (i + 1) % 250 == 0:
                logger.info("pass A %d/%d shards, %d ids", i + 1, len(fastpipe_shards), sum(counts.fastpipe.values()))
    _IDS = build_id_index(parts)
    del parts
    logger.info("pass A done: %d fastpipe docs in chosen WARCs", len(_IDS.keys))

    lpv11_shards = _shards(LPV11_DEDUPED, limit_shards)
    logger.info("pass B: %d lpv11 shards", len(lpv11_shards))
    running: dict[str, list[tuple[str, dict]]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for i, (heaps, n_l, n_b) in enumerate(pool.map(_scan_lpv11_shard, lpv11_shards, chunksize=4)):
            merge_bottom_k(running, heaps, docs_per_warc)
            counts.lpv11.update(n_l)
            counts.both.update(n_b)
            if (i + 1) % 250 == 0:
                logger.info("pass B %d/%d shards, %d both", i + 1, len(lpv11_shards), sum(counts.both.values()))
    sample = {h: [rec for _, rec in items] for h, items in running.items()}

    n_rows = 0
    with fsspec.open(f"{OUT_ROOT}/sample_docs.jsonl.gz", "wt", compression="gzip") as f:
        for w in selection:
            for row in sample.get(w["warc_hash"], []):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    per_warc = {
        w["warc_hash"]: {
            "snapshot": w["snapshot"],
            "n_fastpipe": counts.fastpipe[w["warc_hash"]],
            "n_lpv11": counts.lpv11[w["warc_hash"]],
            "n_both": counts.both[w["warc_hash"]],
            "n_sampled": len(sample.get(w["warc_hash"], [])),
        }
        for w in selection
    }
    totals = {k: sum(v[k] for v in per_warc.values()) for k in ("n_fastpipe", "n_lpv11", "n_both", "n_sampled")}
    stats = {
        "seed": seed,
        "num_warcs": num_warcs,
        "docs_per_warc": docs_per_warc,
        "limit_shards": limit_shards,
        "shards_scanned": {"fastpipe": len(fastpipe_shards), "lpv11": len(lpv11_shards)},
        "totals": totals,
        "lpv11_in_fastpipe": totals["n_both"] / totals["n_lpv11"] if totals["n_lpv11"] else None,
        "fastpipe_in_lpv11": totals["n_both"] / totals["n_fastpipe"] if totals["n_fastpipe"] else None,
        "warcs_short_of_k": sum(1 for v in per_warc.values() if v["n_sampled"] < docs_per_warc),
        "per_warc": per_warc,
    }
    _write_json(f"{OUT_ROOT}/intersection_stats.json", stats)
    logger.info("intersect complete: %d rows sampled; totals %s", n_rows, totals)


# ---------------------------------------------------------------------------
# join: attach raw HTML per chosen WARC (resumable, one file per WARC).
# ---------------------------------------------------------------------------


def _load_sample_by_warc() -> dict[str, list[dict]]:
    by_warc: dict[str, list[dict]] = defaultdict(list)
    with fsspec.open(f"{OUT_ROOT}/sample_docs.jsonl.gz", "rt", compression="gzip") as f:
        for line in f:
            row = json.loads(line)
            by_warc[row["warc_hash"]].append(row)
    return by_warc


def _join_one(warc: dict, rows: list[dict]) -> tuple[int, int]:
    out_uri = f"{OUT_ROOT}/joined/{warc['warc_hash']}.jsonl.gz"
    fs, _ = fsspec.core.url_to_fs(out_uri)
    if fs.exists(out_uri):
        return 0, 0
    if not rows:
        return 0, 0
    html_by_id = _wanted_html(warc["warc_path"], {r["warc_record_id"] for r in rows})
    joined = [{"html": html_by_id[r["warc_record_id"]], **r} for r in rows if r["warc_record_id"] in html_by_id]
    with fsspec.open(out_uri, "wt", compression="gzip") as f:
        for row in joined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(joined), len(rows) - len(joined)


def _selected_warcs(num_warcs: int | None) -> list[dict]:
    """The chosen WARCs in selection order (a seeded random order), optionally the first N."""
    selection = _read_json(f"{OUT_ROOT}/selection_warcs.json")["warcs"]
    return selection[:num_warcs] if num_warcs else selection


def run_join(workers: int, shard_index: int, num_shards: int, num_warcs: int | None) -> None:
    selection = _selected_warcs(num_warcs)
    by_warc = _load_sample_by_warc()
    warcs = [w for i, w in enumerate(selection) if i % num_shards == shard_index]
    logger.info("join shard %d/%d: %d WARCs", shard_index, num_shards, len(warcs))
    joined = unmatched = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_join_one, w, by_warc.get(w["warc_hash"], [])) for w in warcs]
        for i, fut in enumerate(futures):
            n, miss = fut.result()
            joined += n
            unmatched += miss
            if (i + 1) % 10 == 0:
                logger.info("joined %d/%d WARCs (%d docs, %d without HTML)", i + 1, len(warcs), joined, unmatched)
    logger.info("join complete: %d docs joined, %d sampled ids had no HTML record", joined, unmatched)


# ---------------------------------------------------------------------------
# export: one shuffled jsonl.gz for the hallucination runner, plus a manifest.
# ---------------------------------------------------------------------------


def run_export(seed: int, num_warcs: int | None, docs_per_warc: int | None) -> None:
    """Merge joined WARC files, keeping the first ``docs_per_warc`` rows of each.

    Rows within a WARC are in ascending priority order (``intersect`` writes the
    bottom-K sorted, ``join`` preserves order), so a prefix is itself a uniform
    K'-subset of that WARC's intersection.
    """
    selection = _selected_warcs(num_warcs)
    stats = _read_json(f"{OUT_ROOT}/intersection_stats.json")
    present = {p.rsplit("/", 1)[1].removesuffix(".jsonl.gz") for p in fsspec_glob(f"{OUT_ROOT}/joined/*.jsonl.gz")}
    missing_warcs = [w["warc_hash"] for w in selection if w["warc_hash"] not in present]
    if missing_warcs:
        raise RuntimeError(f"{len(missing_warcs)} chosen WARCs have no joined file; rerun join: {missing_warcs[:5]}")

    rows = []
    n_sampled = n_joined = 0
    for w in selection:
        n_sampled += stats["per_warc"][w["warc_hash"]]["n_sampled"]
        with fsspec.open(f"{OUT_ROOT}/joined/{w['warc_hash']}.jsonl.gz", "rt", compression="gzip") as f:
            warc_rows = [json.loads(line) for line in f]
        n_joined += len(warc_rows)
        rows.extend(warc_rows[:docs_per_warc] if docs_per_warc else warc_rows)
    random.Random(seed + 23).shuffle(rows)
    with fsspec.open(f"{OUT_ROOT}/sample.jsonl.gz", "wt", compression="gzip") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "scope": "docs surviving BOTH lpv11 (Generative Curation) and lpv11_fastpipe_v2 (GC Cascade), post-dedup+decon",
        "sources": {"lpv11": LPV11_DEDUPED, "fastpipe_v2": FASTPIPE_DEDUPED},
        "join_key": "lpv11.warc_record_id == fastpipe_v2.doc_id (normalized WARC-Record-ID)",
        "html_field": "html — CC bytes decoded exactly as the extraction saw them (utf-8, errors=replace)",
        "gold_field": "final_output (lpv11 `text`)",
        "seed": seed,
        "num_warcs": len(selection),
        "docs_per_warc": docs_per_warc or stats["docs_per_warc"],
        "intersect_num_warcs": stats["num_warcs"],
        "intersect_docs_per_warc": stats["docs_per_warc"],
        "sampled_without_html": n_sampled - n_joined,
        "count": len(rows),
        "snapshot_distribution": dict(sorted(Counter(r["snapshot"] for r in rows).items())),
        "intersection_totals": stats["totals"],
    }
    _write_json(f"{OUT_ROOT}/manifest.json", manifest)
    logger.info("export complete: %d rows -> %s/sample.jsonl.gz", len(rows), OUT_ROOT)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("intersect")
    p.add_argument("--num-warcs", type=int, default=NUM_WARCS)
    p.add_argument("--docs-per-warc", type=int, default=DOCS_PER_WARC)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-shards", type=int, default=None, help="smoke runs: scan only the first N shards per tree")

    p = sub.add_parser("join")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--num-warcs", type=int, default=None, help="only the first N selected WARCs (default: all)")

    p = sub.add_parser("export")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num-warcs", type=int, default=None, help="only the first N selected WARCs (default: all)")
    p.add_argument("--docs-per-warc", type=int, default=None, help="cap rows per WARC (default: all joined)")

    args = ap.parse_args()
    if args.stage == "intersect":
        run_intersect(args.num_warcs, args.docs_per_warc, args.seed, args.workers, args.limit_shards)
    elif args.stage == "join":
        run_join(args.workers, args.shard, args.num_shards, args.num_warcs)
    elif args.stage == "export":
        run_export(args.seed, args.num_warcs, args.docs_per_warc)


if __name__ == "__main__":
    main()
