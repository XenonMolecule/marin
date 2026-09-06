# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild a full-provenance post-decon tree for a fast_curation spec whose dedup dropped it.

``dedup.py`` used to project records to ``{text, modernbert_prob}``, so every
``baseline_{spec}_decon_deduped/{n}warcs/deduped`` tree written before 2026-08-15 is URL-less — and
WebOrganizer (the topic axis of the quality x domain grid) needs ``url``. ``dedup.py`` now carries
every ``kept_text`` column through; this module repairs the trees that already exist.

It works because document identity is recoverable from text alone: the only text rewrite in the
whole downstream is ``normalize``'s whitespace compaction (runs longer than
``DEFAULT_MAX_WHITESPACE_RUN_CHARS`` are truncated to that length and the id recomputed), fuzzy-dedup
and decon drop whole documents, and ``generate_id`` is a pure xxh3_128 content hash. So applying the
same compaction to ``kept_text`` text reproduces the deduped tree's ids exactly, the survivor set can
be looked up in the per-WARC ``kept_text`` parquet (which carries ``doc_id, url, warc_hash, snapshot,
fasttext_score, modernbert_prob``), and the tree re-materialized FROM ``kept_text`` with every
column — exactly what the fixed ``dedup.py`` + decon would have produced, only re-sharded per WARC.
``text`` in the output is the compacted text, i.e. what was tokenized (12,934 of 77,051,628 lpv11
docs are affected; the first run without compaction found exactly those unmatched).

Stages, all in-region (zero egress)::

    index-kept     per kept_text shard:  {hi, lo, row}  (128-bit content id of every non-empty doc)
    ids-deduped    per deduped shard:    {hi, lo}       (the survivor set)
    keeplists      single process: join the two, pick ONE kept_text (shard,row) per survivor id
                   (lowest wins — exact-duplicate texts across WARCs share an id and dedup kept
                   exactly one; which copy is unrecoverable and the copies differ only in provenance),
                   assert every survivor matched, write one keep-list per kept_text shard
    materialize    per kept_text shard: filter to its keep-list, write jsonl.gz with ALL columns
    verify         recount the written tree against the keep-lists and the expected total

``index-kept``, ``ids-deduped``, ``materialize`` and ``verify`` are Zephyr fan-outs with per-file
skip-if-exists, so a killed coordinator is re-run and resumes for free. ``keeplists`` streams the
join in ``--num-passes`` slices of the id space so its peak RSS stays inside a CPU worker.

Run inside an Iris CPU job in the spec's region (the fan-outs need a long-lived coordinator)::

    uv run iris --cluster marin job run --no-wait --cpu 4 --memory 12GB --priority interactive \\
        --extra cpu --enable-extra-resources --region us-east5 --job-name recover-prov-index \\
        -- python -m experiments.fast_curation.recover_provenance index-kept --spec lpv11_fastpipe_v1

Output tree::

    {bucket}/documents/baseline_{spec}_decon_deduped_urls/{n}warcs/deduped/data-{warc_hash}.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import posixpath
import re
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import dupekit
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import DEFAULT_MAX_WHITESPACE_RUN_CHARS
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

from experiments.fast_curation.dedup import _kept_text_shard_paths
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

REGION_TO_BUCKET: dict[str, str] = {
    "us-east5": "gs://marin-us-east5",
    "us-central1": "gs://marin-us-central1",
    "us-central2": "gs://marin-us-central2",
    "us-east1": "gs://marin-us-east1",
    "us-west4": "gs://marin-us-west4",
}
DEFAULT_MAX_WORKERS = 200
DEFAULT_NUM_PASSES = 16
# Small and preemptible: filler beside accelerator jobs, not a claimant of hosts. A kept_text shard is
# a few hundred MB of text at most, so 8g leaves headroom for the parquet decode + hashing.
WORKER_RAM = "8g"
# Same rewrite normalize applies before hashing (dedup.py runs normalize_step at the default cap).
_WHITESPACE_RUN = re.compile(r"\s{" + str(DEFAULT_MAX_WHITESPACE_RUN_CHARS + 1) + r",}")


def compact_whitespace(text: str) -> str:
    """Truncate whitespace runs longer than the normalize cap, exactly as ``normalize`` does."""
    return _WHITESPACE_RUN.sub(lambda m: m.group(0)[:DEFAULT_MAX_WHITESPACE_RUN_CHARS], text)


@dataclass(frozen=True)
class RecoveryPaths:
    """Every path the recovery reads or writes, derived from (spec, region, pool size)."""

    kept_shards: list[str]
    deduped_dir: str
    root: str

    @property
    def kept_index_dir(self) -> str:
        return f"{self.root}/_recovery/kept_index"

    @property
    def survivor_ids_dir(self) -> str:
        return f"{self.root}/_recovery/survivor_ids"

    @property
    def keeplist_dir(self) -> str:
        return f"{self.root}/_recovery/keeplists"

    @property
    def keeplist_summary(self) -> str:
        return f"{self.root}/_recovery/keeplists_summary.json"

    @property
    def output_dir(self) -> str:
        return f"{self.root}/deduped"

    @property
    def verify_report(self) -> str:
        return f"{self.root}/verify_report.json"


def resolve_paths(spec_id: str, region: str, pool_n: int) -> RecoveryPaths:
    bucket = REGION_TO_BUCKET[region]
    kept = _kept_text_shard_paths(spec_id, region)
    if len(kept) != pool_n:
        raise ValueError(f"kept_text has {len(kept)} shards, expected --pool-n {pool_n}")
    return RecoveryPaths(
        kept_shards=kept,
        deduped_dir=f"{bucket}/documents/baseline_{spec_id}_decon_deduped/{pool_n}warcs/deduped",
        root=f"{bucket}/documents/baseline_{spec_id}_decon_deduped_urls/{pool_n}warcs",
    )


def stem(path: str) -> str:
    """``.../data-{warc_hash}.parquet`` / ``.../data-00001-of-02807.jsonl.gz`` -> the ``data-...`` stem."""
    name = posixpath.basename(path)
    for suffix in (".parquet", ".jsonl.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"unexpected shard name {name!r}")


def content_ids(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """xxh3_128 of each text (the datakit content id) as (hi, lo) uint64 arrays."""
    hi = np.empty(len(texts), dtype=np.uint64)
    lo = np.empty(len(texts), dtype=np.uint64)
    for i, text in enumerate(texts):
        h = dupekit.hash_xxh3_128(text.encode("utf-8"))
        hi[i] = h >> 64
        lo[i] = h & 0xFFFFFFFFFFFFFFFF
    return hi, lo


def _write_table(table: pa.Table, path: str) -> None:
    with fsspec.open(path, "wb") as fh:
        pq.write_table(table, fh, compression="zstd")


def _read_table(path: str, columns: list[str] | None = None) -> pa.Table:
    with fsspec.open(path, "rb") as fh:
        return pq.read_table(fh, columns=columns)


# --------------------------------------------------------------------------
# Stage: index-kept
# --------------------------------------------------------------------------


def index_kept_shard(kept_path: str, out_path: str) -> int:
    """Write ``{hi, lo, row}`` for every non-empty-text row of one kept_text shard.

    Ids are of the whitespace-compacted text, matching what normalize hashed.
    """
    table = _read_table(kept_path, columns=["text"])
    texts = table.column("text").to_pylist()
    rows = np.array([i for i, t in enumerate(texts) if t], dtype=np.uint32)
    hi, lo = content_ids([compact_whitespace(t) for t in texts if t])
    _write_table(pa.table({"hi": hi, "lo": lo, "row": rows}), out_path)
    return len(rows)


def ids_deduped_shard(deduped_path: str, out_path: str) -> int:
    """Write ``{hi, lo}`` for every record of one deduped jsonl.gz shard."""
    texts: list[str] = []
    with fsspec.open(deduped_path, "rb") as fh, gzip.open(fh, "rt", encoding="utf-8") as text_fh:
        for line in text_fh:
            if line.strip():
                texts.append(json.loads(line)["text"])
    hi, lo = content_ids(texts)
    _write_table(pa.table({"hi": hi, "lo": lo}), out_path)
    return len(texts)


def _fanout(name: str, jobs: list[tuple[str, str]], fn, region: str, max_workers: int, ram: str = WORKER_RAM) -> None:
    """Run ``fn(in_path, out_path) -> int`` over ``jobs`` on a Zephyr worker pool, skipping done outputs."""
    fs = fsspec.filesystem("gcs")
    todo = [(src, dst) for src, dst in jobs if not fs.exists(dst)]
    logger.info("%s: %d/%d outputs pending", name, len(todo), len(jobs))
    if not todo:
        return

    def run_shard(items: Iterator[tuple[str, str]], _: ShardInfo) -> Iterator[dict]:
        for src, dst in items:
            n = fn(src, dst)
            yield {"src": src, "n": n}

    ctx = ZephyrContext(
        name=name,
        resources=ResourceConfig(cpu=1, ram=ram, disk="10g", preemptible=True, regions=[region]),
        coordinator_resources=ResourceConfig(cpu=1, ram="3g", preemptible=False, regions=[region]),
        max_workers=max_workers,
        stage_runner_factory=InlineRunner,
    )
    ctx.execute(Dataset.from_list(todo).map_shard(run_shard))


def run_index_kept(paths: RecoveryPaths, region: str, max_workers: int) -> None:
    jobs = [(p, f"{paths.kept_index_dir}/{stem(p)}.parquet") for p in paths.kept_shards]
    _fanout("recover-index-kept", jobs, index_kept_shard, region, max_workers)


def run_ids_deduped(paths: RecoveryPaths, region: str, max_workers: int) -> None:
    shards = sorted(fsspec_glob(f"{paths.deduped_dir}/*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"no deduped shards under {paths.deduped_dir}")
    jobs = [(p, f"{paths.survivor_ids_dir}/{stem(p)}.parquet") for p in shards]
    _fanout("recover-ids-deduped", jobs, ids_deduped_shard, region, max_workers)


# --------------------------------------------------------------------------
# Stage: keeplists (single process)
# --------------------------------------------------------------------------


ID_DTYPE = np.dtype([("hi", np.uint64), ("lo", np.uint64)])


def _pass_of(hi: np.ndarray, num_passes: int) -> np.ndarray:
    """Slice of the id space a doc belongs to: the top bits of ``hi``. Both sides use this."""
    if num_passes & (num_passes - 1):
        raise ValueError(f"--num-passes must be a power of two, got {num_passes}")
    shift = np.uint64(64 - (num_passes.bit_length() - 1))
    return (hi >> shift).astype(np.int64)


def _ids(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Pack (hi, lo) into one structured array so numpy sorts/searches them as 128-bit ids."""
    out = np.empty(len(hi), dtype=ID_DTYPE)
    out["hi"] = hi
    out["lo"] = lo
    return out


def match_survivors(
    kept_ids: np.ndarray,
    kept_shard: np.ndarray,
    kept_row: np.ndarray,
    surv_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """One kept_text ``(shard, row)`` per survivor id present in ``kept``.

    Returns ``(shard, row, n_unmatched)``, aligned to the matched survivors in ``surv_ids`` order.
    Where an id occurs in several kept_text rows (exact duplicates across WARCs; the exact-dedup step
    kept one copy) the lowest ``(shard, row)`` wins.
    """
    if len(surv_ids) == 0 or len(kept_ids) == 0:
        return np.array([], dtype=np.uint32), np.array([], dtype=np.uint32), len(surv_ids)
    # Sort kept by (id, shard, row); the first row of each id group is its representative.
    order = np.lexsort((kept_row, kept_shard, kept_ids["lo"], kept_ids["hi"]))
    k_ids, k_shard, k_row = kept_ids[order], kept_shard[order], kept_row[order]
    first = np.ones(len(k_ids), dtype=bool)
    first[1:] = k_ids[1:] != k_ids[:-1]
    rep_ids, rep_shard, rep_row = k_ids[first], k_shard[first], k_row[first]

    pos = np.searchsorted(rep_ids, surv_ids)
    in_range = pos < len(rep_ids)
    hit = np.zeros(len(surv_ids), dtype=bool)
    hit[in_range] = rep_ids[pos[in_range]] == surv_ids[in_range]
    return rep_shard[pos[hit]], rep_row[pos[hit]], int((~hit).sum())


def _load_tables(paths: list[str], columns: list[str], threads: int) -> list[pa.Table]:
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(lambda p: _read_table(p, columns), paths))


def run_keeplists(paths: RecoveryPaths, expect_total: int | None, num_passes: int, threads: int) -> None:
    """Join survivor ids against the kept_text index and write one keep-list per kept_text shard."""
    fs = fsspec.filesystem("gcs")
    index_files = [f"{paths.kept_index_dir}/{stem(p)}.parquet" for p in paths.kept_shards]
    missing = [p for p in index_files if not fs.exists(p)]
    if missing:
        raise FileNotFoundError(f"index-kept incomplete: {len(missing)} of {len(index_files)} missing")
    surv_files = sorted(fsspec_glob(f"{paths.survivor_ids_dir}/*.parquet"))
    n_deduped = len(fsspec_glob(f"{paths.deduped_dir}/*.jsonl.gz"))
    if len(surv_files) != n_deduped:
        raise FileNotFoundError(f"ids-deduped incomplete: {len(surv_files)} of {n_deduped} present")

    logger.info("loading %d survivor id files", len(surv_files))
    surv = pa.concat_tables(_load_tables(surv_files, ["hi", "lo"], threads))
    surv_ids = np.sort(_ids(surv.column("hi").to_numpy(), surv.column("lo").to_numpy()))
    n_survivors = len(surv_ids)
    del surv
    if n_survivors > 1 and (surv_ids[1:] == surv_ids[:-1]).any():
        # The deduped tree went through exact dedup, so a repeated content id is corruption upstream.
        n_dup = int((surv_ids[1:] == surv_ids[:-1]).sum())
        raise ValueError(f"{n_dup} duplicate content ids among {n_survivors} survivors")
    if expect_total is not None and n_survivors != expect_total:
        raise ValueError(f"survivor ids {n_survivors} != --expect-total {expect_total}")
    logger.info("%d survivors, all ids distinct", n_survivors)
    surv_pass = _pass_of(surv_ids["hi"], num_passes)

    def load_index_slice(item: tuple[int, str], p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shard_idx, path = item
        t = _read_table(path, ["hi", "lo", "row"])
        hi = t.column("hi").to_numpy()
        keep = _pass_of(hi, num_passes) == p
        ids = _ids(hi[keep], t.column("lo").to_numpy()[keep])
        return ids, np.full(len(ids), shard_idx, dtype=np.uint32), t.column("row").to_numpy()[keep]

    all_shard: list[np.ndarray] = []
    all_row: list[np.ndarray] = []
    unmatched = 0
    for p in range(num_passes):
        # Every index file is re-read once per pass, filtered to this slice of the id space at load
        # time, so peak RSS is ~1/num_passes of the whole index rather than all of it.
        with ThreadPoolExecutor(threads) as pool:
            parts = list(pool.map(load_index_slice, enumerate(index_files), [p] * len(index_files)))
        sel = surv_pass == p
        shard, row, miss = match_survivors(
            np.concatenate([x[0] for x in parts]),
            np.concatenate([x[1] for x in parts]),
            np.concatenate([x[2] for x in parts]),
            surv_ids[sel],
        )
        del parts
        all_shard.append(shard)
        all_row.append(row)
        unmatched += miss
        logger.info(
            "pass %d/%d: %d survivors, %d matched, %d unmatched", p + 1, num_passes, int(sel.sum()), len(shard), miss
        )

    shard = np.concatenate(all_shard)
    row = np.concatenate(all_row)
    if unmatched:
        raise ValueError(f"{unmatched} of {n_survivors} survivors have no kept_text row; refusing to write keep-lists")
    if len(shard) != n_survivors:
        raise ValueError(f"matched {len(shard)} rows for {n_survivors} survivors")

    order = np.lexsort((row, shard))
    shard, row = shard[order], row[order]
    bounds = np.searchsorted(shard, np.arange(len(paths.kept_shards) + 1, dtype=np.uint32))
    per_shard: dict[str, int] = {}

    def write_one(i: int) -> None:
        rows = row[bounds[i] : bounds[i + 1]]
        _write_table(
            pa.table({"row": pa.array(rows, type=pa.uint32())}),
            f"{paths.keeplist_dir}/{stem(paths.kept_shards[i])}.parquet",
        )

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(write_one, range(len(paths.kept_shards))))
    for i, kp in enumerate(paths.kept_shards):
        per_shard[stem(kp)] = int(bounds[i + 1] - bounds[i])
    with fsspec.open(paths.keeplist_summary, "w") as fh:
        json.dump({"n_survivors": n_survivors, "n_kept_shards": len(paths.kept_shards), "per_shard": per_shard}, fh)
    logger.info("keep-lists written for %d shards, %d rows total", len(paths.kept_shards), n_survivors)


# --------------------------------------------------------------------------
# Stage: materialize + verify
# --------------------------------------------------------------------------


def materialize_shard(kept_path: str, out_path: str, keeplist_path: str) -> int:
    """Copy the keep-listed rows of one kept_text shard, every column, to jsonl.gz.

    ``text`` is whitespace-compacted so the output is byte-identical to the deduped tree's text.
    """
    rows = _read_table(keeplist_path, ["row"]).column("row").to_numpy()
    table = _read_table(kept_path)
    if len(rows) and int(rows.max()) >= table.num_rows:
        raise ValueError(f"{keeplist_path}: row {int(rows.max())} beyond {table.num_rows} rows of {kept_path}")
    subset = table.take(pa.array(rows, type=pa.int64())) if len(rows) else table.slice(0, 0)
    records = subset.to_pylist()
    with fsspec.open(out_path, "wb") as fh, gzip.open(fh, "wt", encoding="utf-8") as text_fh:
        for record in records:
            record["text"] = compact_whitespace(record["text"])
            text_fh.write(json.dumps(record, ensure_ascii=False))
            text_fh.write("\n")
    return len(records)


def run_materialize(paths: RecoveryPaths, region: str, max_workers: int) -> None:
    fs = fsspec.filesystem("gcs")
    if not fs.exists(paths.keeplist_summary):
        raise FileNotFoundError(f"keeplists not finished: {paths.keeplist_summary} missing")
    jobs = [(p, f"{paths.output_dir}/{stem(p)}.jsonl.gz") for p in paths.kept_shards]

    def fn(src: str, dst: str) -> int:
        return materialize_shard(src, dst, f"{paths.keeplist_dir}/{stem(src)}.parquet")

    _fanout("recover-materialize", jobs, fn, region, max_workers)


def count_shard(path: str, out_path: str) -> int:
    n = 0
    with fsspec.open(path, "rb") as fh, gzip.open(fh, "rt", encoding="utf-8") as text_fh:
        for line in text_fh:
            if line.strip():
                if "url" not in json.loads(line):
                    raise ValueError(f"{path}: record without url column")
                n += 1
    with fsspec.open(out_path, "w") as fh:
        fh.write(str(n))
    return n


def run_verify(paths: RecoveryPaths, region: str, max_workers: int, expect_total: int) -> None:
    """Recount every written shard against its keep-list and the expected survivor total."""
    with fsspec.open(paths.keeplist_summary) as fh:
        summary = json.load(fh)
    counts_dir = f"{paths.root}/_recovery/verify_counts"
    jobs = [(f"{paths.output_dir}/{s}.jsonl.gz", f"{counts_dir}/{s}") for s in summary["per_shard"]]
    _fanout("recover-verify", jobs, count_shard, region, max_workers, ram="4g")
    fs = fsspec.filesystem("gcs")
    mismatched = []
    total = 0
    with ThreadPoolExecutor(32) as pool:
        counted = list(pool.map(lambda job: int(fs.cat(job[1])), jobs))
    for (s, expected), n in zip(summary["per_shard"].items(), counted, strict=True):
        total += n
        if n != expected:
            mismatched.append((s, expected, n))
    report = {
        "shards": len(jobs),
        "docs": total,
        "expected_total": expect_total,
        "keeplist_total": summary["n_survivors"],
        "mismatched_shards": mismatched[:50],
        "n_mismatched": len(mismatched),
        "reconciles": total == expect_total == summary["n_survivors"] and not mismatched,
    }
    with fsspec.open(paths.verify_report, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("verify: %s", json.dumps(report, indent=2))
    if not report["reconciles"]:
        raise ValueError("rebuilt tree does not reconcile; see report")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["index-kept", "ids-deduped", "keeplists", "materialize", "verify"])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--region", default="us-east5", choices=sorted(REGION_TO_BUCKET))
    parser.add_argument("--pool-n", type=int, default=10364)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--num-passes", type=int, default=DEFAULT_NUM_PASSES, help="keeplists: id-space slices")
    parser.add_argument("--threads", type=int, default=32, help="keeplists/verify: GCS read/write threads")
    parser.add_argument(
        "--expect-total",
        type=int,
        default=None,
        help="document count the survivor set must equal (the tokenized cache's ledger total); required for verify",
    )
    args = parser.parse_args()
    paths = resolve_paths(args.spec, args.region, args.pool_n)
    if args.stage == "index-kept":
        run_index_kept(paths, args.region, args.max_workers)
    elif args.stage == "ids-deduped":
        run_ids_deduped(paths, args.region, args.max_workers)
    elif args.stage == "keeplists":
        run_keeplists(paths, args.expect_total, args.num_passes, args.threads)
    elif args.stage == "materialize":
        run_materialize(paths, args.region, args.max_workers)
    else:
        if args.expect_total is None:
            parser.error("verify requires --expect-total")
        run_verify(paths, args.region, args.max_workers, args.expect_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
