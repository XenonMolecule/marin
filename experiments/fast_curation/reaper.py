# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase C for the TEXT line: verify kept output, then reclaim the intermediate storage.

The TEXT line's presurvivors carry the full extracted ``text`` and are ~2x the size of the final
``kept/`` corpus; at 8M-WARC scale leaving them in place is ~600TB of transient storage. This reaper
trails Phase B and, for every WARC the central registry marks complete:

1. **verifies the kept artifact** — the parquet footer must parse and its row count must be sane
   (0 is legal: a WARC can keep nothing) in at least one region. Verification gates every delete:
   a registry entry alone proves a worker *ran*, not that its output is readable.
2. deletes that WARC's ``a_presurvivors/`` parquet and any leftover ``a_chunks/`` checkpoint dir in
   EVERY region (rescue re-decodes leave duplicates in TPU-fallback regions).
3. with ``--collapse-kept-duplicates``, keeps the canonical region's ``kept/``/``tombstones/`` copy
   (preference order: the region with the most output) and deletes the other regions' duplicates —
   cross-region re-scores after claim expiry produce identical content twice.

A WARC is only touched once its registry entry is older than ``--lag-minutes`` (default 60), so a
just-finished WARC keeps a re-run buffer. Deletes are idempotent; the reaper holds no claims and can
run repeatedly or concurrently. Run as a single small CPU job (or locally for a finished run)::

    python -m experiments.fast_curation.reaper --spec lpv11_fastpipe_v2 --dry-run
    python -m experiments.fast_curation.reaper --spec lpv11_fastpipe_v2 --collapse-kept-duplicates
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pyarrow.parquet as pq

from experiments.fast_curation import shard_worklist as sw
from experiments.fast_curation.spec import PipelineSpec, get_spec

logger = logging.getLogger(__name__)

# All buckets a run may have written to (primary + rescue/TPU-fallback regions).
RUN_BUCKETS = ("gs://marin-us-east5", "gs://marin-us-east1", "gs://marin-us-central2", "gs://marin-us-west4")
DELETE_WORKERS = 16


def _fs():
    return fsspec.filesystem("gcs")


def _ls_names(prefix: str) -> set[str]:
    """Object basenames under ``prefix`` (empty set if the prefix does not exist)."""
    fs = _fs()
    try:
        return {p.rsplit("/", 1)[-1] for p in fs.ls(prefix.replace("gs://", ""), refresh=True)}
    except FileNotFoundError:
        return set()


def _completed_older_than(spec: PipelineSpec, lag_minutes: float) -> set[str]:
    """WARC hashes whose Phase-B registry entry is older than the safety lag."""
    fs = _fs()
    prefix = f"marin-us-central1/{spec.subdir()}/_completed_b"
    cutoff = time.time() - lag_minutes * 60
    out = set()
    try:
        for info in fs.ls(prefix, refresh=True, detail=True):
            name = info["name"].rsplit("/", 1)[-1]
            if not name.startswith("data-"):
                continue
            mtime = info.get("mtime")
            ts = mtime.timestamp() if hasattr(mtime, "timestamp") else 0.0
            if ts and ts > cutoff:
                continue
            out.add(name[len("data-") :])
    except FileNotFoundError:
        pass
    return out


def _kept_row_count(path: str) -> int | None:
    """Row count from the parquet footer, or None if unreadable — the verification gate."""
    try:
        with fsspec.open(path, "rb") as fh:
            return pq.ParquetFile(fh).metadata.num_rows
    except Exception as e:
        logger.warning("kept UNREADABLE %s: %s", path, e)
        return None


def _ft_sidecar(spec: PipelineSpec, h: str, source_bucket: str, canon_bucket: str, dry_run: bool) -> bool:
    """Preserve the fastText scores held ONLY in the presurvivor before it is deleted.

    ``kept/`` carries all three scores for kept docs and ``tombstones/`` carries pooled+terminal for
    dropped docs, but a B-dropped doc's ``fasttext_score`` exists nowhere else. One small parquet
    per WARC (doc_id, fasttext_score, n_tokens; ~25 B/doc) keeps every computed score queryable
    after the ~2x-larger presurvivor is reclaimed.
    """
    out = f"{spec.namespace(canon_bucket)}/ft_scores/data-{h}.parquet"
    if dry_run:
        return True
    try:
        with fsspec.open(f"{spec.presurvivors_prefix(source_bucket)}/data-{h}.parquet", "rb") as fh:
            t = pq.read_table(fh, columns=["doc_id", "fasttext_score", "n_tokens"])
        with fsspec.open(out, "wb") as fh:
            pq.write_table(t, fh, compression="zstd")
        return True
    except Exception as e:
        logger.warning("ft sidecar failed for %s (%s); presurvivor NOT deleted", h, e)
        return False


def _delete(paths: list[str], namespace_tag: str, dry_run: bool) -> int:
    """Best-effort parallel delete; returns how many paths were removed.

    Every path MUST sit inside this spec's hash namespace — a construction bug upstream must be
    caught here, before anything outside the run's own tree can be touched.
    """
    for p in paths:
        if namespace_tag not in p:
            raise RuntimeError(f"REFUSING delete outside namespace {namespace_tag!r}: {p}")
    if dry_run or not paths:
        return len(paths)
    fs = _fs()

    def _rm(p: str) -> bool:
        try:
            fs.rm(p.replace("gs://", ""), recursive=True)
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning("delete failed %s: %s", p, e)
            return False

    with ThreadPoolExecutor(max_workers=DELETE_WORKERS) as ex:
        return sum(ex.map(_rm, paths))


def reap(
    spec: PipelineSpec, *, lag_minutes: float, collapse_kept: bool, dry_run: bool, preserve_dropped_scores: bool = False
) -> dict:
    """One reaping pass. Returns counters (also logged)."""
    done = _completed_older_than(spec, lag_minutes)
    logger.info("registry: %d WARCs past the %d-minute lag", len(done), int(lag_minutes))

    kept_by_bucket = {b: _ls_names(spec.kept_prefix(b)) for b in RUN_BUCKETS}
    sidecar_by_bucket = {b: _ls_names(f"{spec.namespace(b)}/ft_scores") for b in RUN_BUCKETS}
    pre_by_bucket = {b: _ls_names(spec.presurvivors_prefix(b)) for b in RUN_BUCKETS}
    chunk_by_bucket = {b: _ls_names(f"{spec.namespace(b)}/a_chunks") for b in RUN_BUCKETS}
    # Canonical preference: the bucket holding the most kept output wins ties for duplicates.
    canon_order = sorted(RUN_BUCKETS, key=lambda b: -len(kept_by_bucket[b]))

    stats = {
        "verified": 0,
        "unverified": 0,
        "sidecars": 0,
        "sidecar_failed": 0,
        "pre_deleted": 0,
        "chunks_deleted": 0,
        "kept_dupes_deleted": 0,
    }
    to_delete: list[str] = []

    # Incremental: only verify WARCs that still have something to reclaim. Without this, a looping
    # reaper re-reads EVERY completed WARC's kept footer each pass — 8M footer reads per hour at
    # production scale for zero deletes.
    def _reclaimable(h: str) -> bool:
        fname = f"data-{h}.parquet"
        if any(fname in pre_by_bucket[b] for b in RUN_BUCKETS):
            return True
        if any(f"data-{h}" in chunk_by_bucket[b] for b in RUN_BUCKETS):
            return True
        if collapse_kept and sum(fname in kept_by_bucket[b] for b in RUN_BUCKETS) > 1:
            return True
        return False

    done = {h for h in done if _reclaimable(h)}
    logger.info("of those, %d still have something to reclaim", len(done))
    holders_of = {}
    for h in sorted(done):
        holders_of[h] = [b for b in canon_order if f"data-{h}.parquet" in kept_by_bucket[b]]
    # Footer verification is one range-read per WARC; sequential it is ~45 min at 10k and unusable at
    # 8M, so fan it out (same pool width as the deletes).
    with ThreadPoolExecutor(max_workers=DELETE_WORKERS * 2) as ex:
        rows_of = dict(
            zip(
                holders_of,
                ex.map(
                    lambda h: (
                        _kept_row_count(f"{spec.kept_prefix(holders_of[h][0])}/data-{h}.parquet")
                        if holders_of[h]
                        else None
                    ),
                    holders_of,
                ),
                strict=True,
            )
        )
    verified = {}
    for h in sorted(done):
        holders = holders_of[h]
        if not holders or rows_of[h] is None:
            stats["unverified"] += 1
            continue  # registry says done but canonical kept unreadable/absent — touch NOTHING.
        stats["verified"] += 1
        verified[h] = holders

    # Preserve fastText scores (sidecar in the canonical kept bucket) BEFORE deleting the only place
    # they live. Existence comes from ONE upfront listing and creation is fanned out — sequential
    # per-WARC probes/writes made this pass hours long and would be unusable trailing an 8M run.
    # Off by default: kept docs carry every score in kept/ itself, and tombstones keep the B-stage
    # probs for dropped docs; the sidecar only adds dropped docs' fastText scores.
    need_sidecar = {}
    if preserve_dropped_scores:
        need_sidecar = {
            h: pre
            for h, holders in verified.items()
            if (pre := [b for b in RUN_BUCKETS if f"data-{h}.parquet" in pre_by_bucket[b]])
            and f"data-{h}.parquet" not in sidecar_by_bucket[holders[0]]
        }
    with ThreadPoolExecutor(max_workers=DELETE_WORKERS) as ex:
        ok_of = dict(
            zip(
                need_sidecar,
                ex.map(lambda h: _ft_sidecar(spec, h, need_sidecar[h][0], verified[h][0], dry_run), need_sidecar),
                strict=True,
            )
        )
    stats["sidecars"] = sum(ok_of.values())
    stats["sidecar_failed"] = len(ok_of) - stats["sidecars"]

    for h, holders in verified.items():
        fname = f"data-{h}.parquet"
        if not ok_of.get(h, True):
            continue  # sidecar failed: keep this WARC's presurvivors AND everything else untouched.
        for b in RUN_BUCKETS:
            if fname in pre_by_bucket[b]:
                to_delete.append(f"{spec.presurvivors_prefix(b)}/{fname}")
                stats["pre_deleted"] += 1
            if f"data-{h}" in chunk_by_bucket[b]:
                to_delete.append(f"{spec.namespace(b)}/a_chunks/data-{h}")
                stats["chunks_deleted"] += 1
        if collapse_kept:
            for b in holders[1:]:  # non-canonical duplicates only; canonical copy verified above.
                to_delete.append(f"{spec.kept_prefix(b)}/{fname}")
                to_delete.append(f"{spec.tombstones_prefix(b)}/data-{h}.jsonl.gz")
                stats["kept_dupes_deleted"] += 1

    n = _delete(to_delete, spec.subdir(), dry_run)
    logger.info(
        "%s: %s (%d delete ops%s)",
        "DRY-RUN would reclaim" if dry_run else "reclaimed",
        stats,
        n,
        ", not executed" if dry_run else "",
    )
    return stats


def reap_v3(spec: PipelineSpec, *, lag_minutes: float, dry_run: bool) -> dict:
    """V3 (storage_version=3) pass: shard-granular, catalog-verified, marker-compacting.

    For every shard with a ``_b_done`` sentinel older than the lag: read its catalog parquet,
    verify each row's ``kept_path`` parquet footer, then delete the shard's presurvivor files (in
    the shard's assigned region) and its now-redundant ``_a_marks``/``_b_marks`` trees — the
    catalog IS the durable record. A shard with any unverifiable kept file is left untouched.
    """
    fs = _fs()
    done = set()
    cutoff = time.time() - lag_minutes * 60
    try:
        for info in fs.ls(sw.sentinel_prefix(spec, "b").replace("gs://", ""), refresh=True, detail=True):
            name = info["name"].rsplit("/", 1)[-1]
            if not name.startswith("data-"):
                continue
            mtime = info.get("mtime")
            ts = mtime.timestamp() if hasattr(mtime, "timestamp") else 0.0
            if ts and ts > cutoff:
                continue
            done.add(int(name[len("data-") :]))
    except FileNotFoundError:
        pass
    index = {e["shard"]: e for e in sw.load_index(spec)}
    stats = {"shards_reaped": 0, "shards_unverified": 0, "pre_deleted": 0, "marks_deleted": 0}
    for s in sorted(done):
        bucket = sw.REGION_TO_BUCKET[index[s]["region"]]
        pre_prefix = spec.presurvivors_prefix(bucket)
        pre_names = _ls_names(pre_prefix)
        pairs = sw.load_shard(spec, s)
        if not any(f"data-{h}.parquet" in pre_names for _, h in pairs):
            continue  # already reaped — nothing left for this shard.
        try:
            with fsspec.open(sw.catalog_path(spec, s), "rb") as fh:
                rows = pq.read_table(fh).to_pylist()
        except Exception as e:
            logger.warning("shard %05d catalog unreadable (%s); untouched", s, e)
            stats["shards_unverified"] += 1
            continue
        with ThreadPoolExecutor(max_workers=DELETE_WORKERS * 2) as ex:
            counts = list(ex.map(lambda r: _kept_row_count(r["kept_path"]), rows))
        if len(rows) != len(pairs) or any(c is None for c in counts):
            logger.warning("shard %05d has unverifiable kept output; untouched", s)
            stats["shards_unverified"] += 1
            continue
        to_delete = [f"{pre_prefix}/data-{h}.parquet" for _, h in pairs if f"data-{h}.parquet" in pre_names]
        stats["pre_deleted"] += len(to_delete)
        for phase in ("a", "b"):
            to_delete.append(f"{sw.CENTRAL_BUCKET}/{spec.subdir()}/_{phase}_marks/shard-{s:05d}")
            stats["marks_deleted"] += 1
        _delete(to_delete, spec.subdir(), dry_run)
        stats["shards_reaped"] += 1
    logger.info("%s: %s", "DRY-RUN v3 would reclaim" if dry_run else "v3 reclaimed", stats)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="lpv11_fastpipe_v2")
    ap.add_argument("--lag-minutes", type=float, default=60.0, help="Only touch WARCs B-completed this long ago.")
    ap.add_argument("--collapse-kept-duplicates", action="store_true")
    ap.add_argument(
        "--preserve-dropped-scores",
        action="store_true",
        help="Write ft_scores/ sidecars (dropped docs' fastText scores) before deleting presurvivors.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--loop-seconds", type=float, default=None, help="Re-run every N seconds (trailing a live B).")
    args = ap.parse_args()

    spec = get_spec(args.spec)
    while True:
        if spec.storage_version == 3:
            reap_v3(spec, lag_minutes=args.lag_minutes, dry_run=args.dry_run)
            if args.loop_seconds is None:
                break
            time.sleep(args.loop_seconds)
            continue
        reap(
            spec,
            lag_minutes=args.lag_minutes,
            collapse_kept=args.collapse_kept_duplicates,
            dry_run=args.dry_run,
            preserve_dropped_scores=args.preserve_dropped_scores,
        )
        if args.loop_seconds is None:
            break
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    main()
