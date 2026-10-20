# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Reopen fused-run WARCs that finished with no input, so a relaunched fleet re-fetches them.

A fused worker that cannot download a WARC (Common Crawl S3 503s exhaust its retries) or that
abandons it after repeated stalls still closes the shard: it writes an empty ``kept/`` file,
a per-WARC marker with ``a_n_in == 0`` and, once the shard is done, the shard sentinel. This
tool finds those WARCs from the shard catalogs and undoes exactly that state:

- the WARC's ``_b_marks`` marker and ``_b_attempts`` counter (central bucket),
- every ``kept/data-<hash>.parquet`` in any region bucket, but only if it has 0 rows,
- the shard's ``_b_done`` sentinel, after its WARCs, so the shard becomes claimable again.

Workers then skip every still-marked WARC and process only the reopened ones; finishing the
shard rewrites its catalog and sentinel. Dry run by default; ``--apply`` deletes. Every applied
run writes a manifest of what it touched under ``{central}/{subdir}/_reopened/``.

    python -m experiments.fast_curation.reopen_failed_warcs --spec lpv11_fastpipe_v2_1_fused [--apply]
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pyarrow.parquet as pq

from experiments.baseline_collection.run_extract_standalone import _write_registry_summary
from experiments.fast_curation import shard_worklist as sw
from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)

FIRST_FUSED_SHARD = 404  # shards below are the two-phase rung 1, which has no fused catalog
CATALOG_COLUMNS = ["warc_hash", "a_n_in", "n_kept"]


@dataclasses.dataclass(frozen=True)
class FailedWarc:
    shard: int
    warc_hash: str


def failed_rows(shard: int, rows: list[dict]) -> list[FailedWarc]:
    """Catalog rows of one shard whose WARC produced no input documents."""
    failed = []
    for row in rows:
        if row["a_n_in"]:
            continue
        if row["n_kept"]:
            raise ValueError(f"shard {shard} WARC {row['warc_hash']}: a_n_in == 0 but n_kept == {row['n_kept']}")
        failed.append(FailedWarc(shard, row["warc_hash"]))
    return failed


def _central(spec) -> str:
    return f"{sw.CENTRAL_BUCKET}/{spec.subdir()}"


def _scan_shard(fs, spec, shard: int) -> list[FailedWarc]:
    with fs.open(sw.catalog_path(spec, shard)) as f:
        return failed_rows(shard, pq.read_table(f, columns=CATALOG_COLUMNS).to_pylist())


def _empty_kept_copies(fs, spec, warc_hash: str) -> list[str]:
    """Every region's ``kept/`` file for this WARC; raises if any copy holds rows."""
    paths = []
    for bucket in sw.REGION_TO_BUCKET.values():
        path = f"{spec.namespace(bucket)}/kept/data-{warc_hash}.parquet"
        if not fs.exists(path):
            continue
        with fs.open(path) as f:
            rows = pq.ParquetFile(f).metadata.num_rows
        if rows:
            raise ValueError(f"{path} has {rows} rows; refusing to reopen WARC {warc_hash}")
        paths.append(path)
    return paths


def _marker_paths(spec, w: FailedWarc) -> tuple[str, str]:
    root = _central(spec)
    return (
        f"{root}/_b_marks/shard-{w.shard:05d}/data-{w.warc_hash}",
        f"{root}/_b_attempts/shard-{w.shard:05d}/data-{w.warc_hash}",
    )


def _plan_warc(fs, spec, w: FailedWarc) -> list[str]:
    """Paths to delete for one WARC, after checking its marker records no input."""
    marker, attempts = _marker_paths(spec, w)
    with fs.open(marker, "r") as f:
        payload = json.load(f)
    if payload["a_n_in"] or payload["n_kept"]:
        raise ValueError(f"{marker}: marker records input ({payload['a_n_in']}) or kept docs ({payload['n_kept']})")
    paths = [marker, *_empty_kept_copies(fs, spec, w.warc_hash)]
    if fs.exists(attempts):
        paths.append(attempts)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--max-shard", type=int, default=32000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--apply", action="store_true", help="Delete (default is a dry run that only reports).")
    ap.add_argument(
        "--only",
        default=None,
        help="JSON list of {shard, warc_hash} objects: reopen only these of the failed WARCs.",
    )
    args = ap.parse_args()

    spec = get_spec(args.spec)
    fs = fsspec.filesystem("gcs")
    with ThreadPoolExecutor(args.threads) as ex:
        failed = [
            w for ws in ex.map(lambda s: _scan_shard(fs, spec, s), range(FIRST_FUSED_SHARD, args.max_shard)) for w in ws
        ]
        logger.info("%d WARCs with no input across %d shards", len(failed), len({w.shard for w in failed}))
        if args.only:
            with fsspec.open(args.only) as f:
                wanted = {FailedWarc(r["shard"], r["warc_hash"]) for r in json.load(f)}
            failed = [w for w in failed if w in wanted]
            logger.info("restricted to %d WARCs listed in %s", len(failed), args.only)
        plans = dict(zip(failed, ex.map(lambda w: _plan_warc(fs, spec, w), failed), strict=True))

    by_shard: dict[int, list[FailedWarc]] = collections.defaultdict(list)
    for w in failed:
        by_shard[w.shard].append(w)
    kept_files = sum(1 for paths in plans.values() for p in paths if "/kept/" in p)
    logger.info("plan: %d markers, %d empty kept files, %d shard sentinels", len(plans), kept_files, len(by_shard))
    if not args.apply:
        logger.info("dry run: nothing deleted (pass --apply)")
        return

    sentinels = {s: f"{sw.sentinel_prefix(spec, 'b')}/data-{s:05d}" for s in by_shard}
    manifest = f"{_central(spec)}/_reopened/{int(time.time())}.json"
    with fsspec.open(manifest, "w") as f:
        json.dump(
            {
                "warcs": {f"{w.shard}/{w.warc_hash}": p for w, p in plans.items()},
                "sentinels": sorted(sentinels.values()),
            },
            f,
        )
    logger.info("manifest %s", manifest)

    def reopen_shard(shard: int) -> None:
        fs.rm([p for w in by_shard[shard] for p in plans[w]])
        if fs.exists(sentinels[shard]):
            fs.rm(sentinels[shard])  # last: the shard is claimable only once its WARCs are unmarked

    with ThreadPoolExecutor(args.threads) as ex:
        for i, _ in enumerate(ex.map(reopen_shard, sorted(by_shard))):
            if i % 1000 == 0:
                logger.info("reopened %d / %d shards", i, len(by_shard))
    _write_registry_summary(sw.sentinel_prefix(spec, "b"))
    logger.info("reopened %d WARCs in %d shards", len(plans), len(by_shard))


if __name__ == "__main__":
    main()
