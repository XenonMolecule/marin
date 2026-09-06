# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Region failover for v3 sharded runs: move stuck shards to regions that have chips.

When a shard's assigned region loses its TPU capacity (e.g. the only chip pool is held
by another workload), its A output is stranded: B workers in other regions must not read
presurvivors cross-region. The fix is to REPROCESS Phase A in a chip-rich region — WARC
bytes re-fetched from Common Crawl are free ingress, and all GCS traffic stays in-region.

For each reassigned shard this tool, in order:

1. rewrites the shard's ``region`` in ``shards/_index.parquet`` (round-robin over the
   target regions) — shard membership is untouched, only the coordination metadata moves;
2. deletes its ``_a_done`` sentinel, ``_a_marks`` and any A/B claims, so relaunched A
   workers in the new region redo the shard from scratch;
3. deletes its old-region presurvivors (now orphaned).

Shards with a B claim fresher than ``--claim-fresh-minutes`` are left alone — a live
worker is finishing them in the original region. Existing ``_b_marks`` are preserved:
WARCs already scored keep their catalog rows and kept files in the original region, and
the new region's B worker skips them.

    python -m experiments.fast_curation.reassign_shards --spec lpv11_fastpipe_v2_1 \\
        --from-region us-central2 --to-regions us-east5 us-east1 us-west4 \\
        --max-shard 404 --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import logging

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fast_curation import shard_worklist as sw
from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)


def _fresh_b_claims(spec, fresh_minutes: float) -> set[int]:
    """Shard ids whose B claim was refreshed within the freshness window."""
    fs = fsspec.filesystem("gcs")
    now = datetime.datetime.now(datetime.UTC)
    fresh: set[int] = set()
    prefix = sw.claim_prefix(spec, "b").removeprefix("gs://")
    for f in fs.find(prefix, detail=True).values():
        mtime = f.get("mtime") or f.get("updated")
        if mtime is None:
            continue
        if isinstance(mtime, str):
            mtime = datetime.datetime.fromisoformat(mtime.replace("Z", "+00:00"))
        if (now - mtime).total_seconds() < fresh_minutes * 60:
            shard_part = f["name"].split("/shard-")[-1].split("/")[0]
            fresh.add(int(shard_part))
    return fresh


def reassign(
    spec,
    from_region: str,
    to_regions: list[str],
    max_shard: int,
    fresh_minutes: float,
    dry_run: bool,
    limit: int | None = None,
    only_unstarted: bool = False,
):
    for r in [from_region, *to_regions]:
        if r not in sw.REGION_TO_BUCKET:
            raise ValueError(f"unknown region {r!r}")
    fs = fsspec.filesystem("gcs")
    index = sw.load_index(spec)
    b_done = {
        int(p.split("/")[-1].replace("data-", "")) for p in fs.ls(sw.sentinel_prefix(spec, "b").removeprefix("gs://"))
    }
    fresh = _fresh_b_claims(spec, fresh_minutes)
    a_done = {
        int(p.split("/")[-1].replace("data-", "")) for p in fs.ls(sw.sentinel_prefix(spec, "a").removeprefix("gs://"))
    }

    movable = [
        e["shard"]
        for e in index
        if e["region"] == from_region and e["shard"] < max_shard and e["shard"] not in b_done and e["shard"] not in fresh
    ]
    if only_unstarted:
        # Restrict to shards with no A sentinel (already reset by a prior pass): moving them is
        # pure metadata — no live A output gets destroyed.
        movable = [s for s in movable if s not in a_done]
    if limit is not None:
        movable = movable[:limit]
    logger.info(
        "%s has %d incomplete shards under %d; %d actively claimed (left in place); moving %d -> %s",
        from_region,
        len(movable) + len(fresh & {e["shard"] for e in index if e["region"] == from_region}),
        max_shard,
        len(fresh),
        len(movable),
        to_regions,
    )
    if dry_run:
        logger.info("dry-run: shards %s", movable)
        return

    new_region = {s: to_regions[i % len(to_regions)] for i, s in enumerate(movable)}
    rewritten = [{**e, "region": new_region.get(e["shard"], e["region"])} for e in index]
    table = pa.table(
        {
            "shard": [e["shard"] for e in rewritten],
            "n_warcs": [e["n_warcs"] for e in rewritten],
            "region": [e["region"] for e in rewritten],
        }
    )
    with fsspec.open(sw.index_path(spec), "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    logger.info("index rewritten: %d shards reassigned", len(movable))

    old_bucket = sw.REGION_TO_BUCKET[from_region]
    namespace_tag = spec.subdir()
    to_delete: list[str] = []
    for s in movable:
        to_delete.append(f"{sw.sentinel_prefix(spec, 'a')}/data-{s:05d}")
        for prefix in (
            f"{sw.CENTRAL_BUCKET}/{namespace_tag}/_a_marks/shard-{s:05d}",
            f"{sw.claim_prefix(spec, 'a')}/shard-{s:05d}",
            f"{sw.claim_prefix(spec, 'b')}/shard-{s:05d}",
        ):
            try:
                to_delete.extend(f"gs://{p}" for p in fs.find(prefix.removeprefix("gs://")))
            except FileNotFoundError:
                pass
        for _, h in sw.load_shard(spec, s):
            to_delete.append(f"{spec.presurvivors_prefix(old_bucket)}/data-{h}.parquet")
    assert all(namespace_tag in p for p in to_delete), "refusing to delete outside the spec namespace"
    existing = [p for p in to_delete if fs.exists(p.removeprefix("gs://"))]
    logger.info("deleting %d objects (%d listed paths)", len(existing), len(to_delete))
    fs.rm([p.removeprefix("gs://") for p in existing])
    logger.info("done: relaunch Phase A workers in %s with --max-shard %d", to_regions, max_shard)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--from-region", required=True)
    ap.add_argument("--to-regions", nargs="+", required=True)
    ap.add_argument("--max-shard", type=int, required=True)
    ap.add_argument("--claim-fresh-minutes", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=None, help="Move at most this many shards.")
    ap.add_argument(
        "--only-unstarted",
        action="store_true",
        help="Move only shards with no _a_done sentinel (safe metadata-only move after a prior reset).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    spec = get_spec(args.spec)
    if spec.storage_version != 3:
        raise ValueError(f"{args.spec} is not a storage_version=3 spec")
    reassign(
        spec,
        args.from_region,
        args.to_regions,
        args.max_shard,
        args.claim_fresh_minutes,
        args.dry_run,
        args.limit,
        args.only_unstarted,
    )


if __name__ == "__main__":
    main()
