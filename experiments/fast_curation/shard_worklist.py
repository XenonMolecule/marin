# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sharded work-list for storage_version=3 runs — the unit of claim, accounting, and lookup.

The v2 contract (flat manifest + one claim/registry object per WARC, workers re-LISTing the
registry each pass) is O(WARCs x fleet x polls) in GCS operations — the exact class-B pattern of
the Jul-Aug incident, unusable at 8M. V3 replaces it:

* the run's WARCs are partitioned into **shards** of ~``WARCS_PER_SHARD``, deterministically:
  ``shard = int(sha256(warc_path)[:8], 16) % n_shards`` (the modulus is FROZEN per run — partition
  keys never change across resumes);
* shard membership is materialized ONCE as parquet files
  (``{central}/shards/shard-{s:05d}.parquet``: warc_path, warc_hash) plus one index
  (``shards/_index.parquet``: shard, n_warcs, region) assigning each shard a Phase-A region
  round-robin — no worker ever loads the full WARC list;
* workers claim SHARDS (``_claims_{phase}/shard-{s}``), keep per-WARC done markers inside the
  shard's own prefix (only the owner lists them), and stamp ``_{phase}_done/shard-{s}`` sentinels;
  fleet-wide progress is one LIST of the sentinel prefix (~n_shards objects, not n_WARCs);
* Phase B writes one **catalog** parquet per shard (one row per WARC: where the kept output lives
  and its counts) — the run's queryable lookup, ~n_shards small files in total.

Build a work-list (from an existing manifest file, or Common Crawl snapshot listings)::

    python -m experiments.fast_curation.shard_worklist build --spec lpv11_fastpipe_v2_1 \\
        --manifest experiments/distill/dclm_400m_1x.txt --n-shards 100 \\
        --regions us-east5 us-east1
    python -m experiments.fast_curation.shard_worklist build --spec lpv11_fastpipe_v2_1 \\
        --snapshots CC-MAIN-2022-33 CC-MAIN-2022-27 --n-warcs 100000 --n-shards 100 \\
        --regions us-east5 us-east1 us-central2 us-west4
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import logging
import random
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from experiments.baseline_collection.decode_warcs_clean import _load_manifest, _warc_path_hash
from experiments.fast_curation.spec import PipelineSpec, get_spec

logger = logging.getLogger(__name__)

CENTRAL_BUCKET = "gs://marin-us-central1"
WARCS_PER_SHARD = 1000  # sizing guide only; --n-shards is what is frozen into the layout.
CC_PATHS_URL = "https://data.commoncrawl.org/crawl-data/{snapshot}/warc.paths.gz"

REGION_TO_BUCKET = {
    "us-east5": "gs://marin-us-east5",
    "us-east1": "gs://marin-us-east1",
    "us-central2": "gs://marin-us-central2",
    "us-west4": "gs://marin-us-west4",
    "us-central1": "gs://marin-us-central1",
    # Bucket name is marin-eu-west4 (not marin-europe-west4) while the region is europe-west4.
    "europe-west4": "gs://marin-eu-west4",
}


def shard_of(warc_path: str, n_shards: int) -> int:
    """Deterministic shard id (stable across runs and resumes — never change the formula)."""
    return int(hashlib.sha256(warc_path.encode()).hexdigest()[:8], 16) % n_shards


def shards_root(spec: PipelineSpec) -> str:
    return f"{CENTRAL_BUCKET}/{spec.subdir()}/shards"


def shard_path(spec: PipelineSpec, s: int) -> str:
    return f"{shards_root(spec)}/shard-{s:05d}.parquet"


def index_path(spec: PipelineSpec) -> str:
    return f"{shards_root(spec)}/_index.parquet"


def sentinel_prefix(spec: PipelineSpec, phase: str) -> str:
    return f"{CENTRAL_BUCKET}/{spec.subdir()}/_{phase}_done"


def claim_prefix(spec: PipelineSpec, phase: str) -> str:
    return f"{CENTRAL_BUCKET}/{spec.subdir()}/_claims_{phase}"


def catalog_path(spec: PipelineSpec, s: int) -> str:
    return f"{CENTRAL_BUCKET}/{spec.subdir()}/catalog/shard-{s:05d}.parquet"


def load_shard(spec: PipelineSpec, s: int) -> list[tuple[str, str]]:
    """One shard's ``(warc_path, warc_hash)`` pairs."""
    with fsspec.open(shard_path(spec, s), "rb") as fh:
        t = pq.read_table(fh)
    return list(zip(t.column("warc_path").to_pylist(), t.column("warc_hash").to_pylist(), strict=True))


def load_index(spec: PipelineSpec) -> list[dict]:
    with fsspec.open(index_path(spec), "rb") as fh:
        return pq.read_table(fh).to_pylist()


def _cc_snapshot_paths(snapshot: str) -> list[str]:
    """All WARC paths of one Common Crawl snapshot (from the published listing; ~1MB gz)."""
    r = requests.get(CC_PATHS_URL.format(snapshot=snapshot), timeout=120)
    r.raise_for_status()
    lines = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read().decode().splitlines()
    return [f"s3://commoncrawl/{line.strip()}" for line in lines if line.strip()]


def build(
    spec: PipelineSpec,
    warc_paths: list[str],
    n_shards: int,
    regions: list[str],
    seed: int = 0,
) -> None:
    """Materialize the frozen shard layout. Refuses to overwrite an existing layout."""
    fs, idx = fsspec.core.url_to_fs(index_path(spec))
    if fs.exists(idx):
        raise RuntimeError(
            f"{index_path(spec)} already exists — the shard layout is FROZEN once built. "
            "Reusing a namespace with a different layout would orphan its claims and markers."
        )
    for r in regions:
        if r not in REGION_TO_BUCKET:
            raise ValueError(f"unknown region {r!r}; known: {sorted(REGION_TO_BUCKET)}")

    by_shard: dict[int, list[str]] = {s: [] for s in range(n_shards)}
    for w in warc_paths:
        by_shard[shard_of(w, n_shards)].append(w)

    # Region assignment round-robin over a seed-shuffled shard order, so early snapshots don't all
    # land in one region; the assignment is persisted in the index (workers read, never recompute).
    order = list(range(n_shards))
    random.Random(seed).shuffle(order)
    region_of = {s: regions[i % len(regions)] for i, s in enumerate(order)}

    def _write_shard(s: int) -> None:
        paths = sorted(by_shard[s])
        table = pa.table(
            {"warc_path": paths, "warc_hash": [_warc_path_hash(w) for w in paths]},
            schema=pa.schema([("warc_path", pa.string()), ("warc_hash", pa.string())]),
        )
        with fsspec.open(shard_path(spec, s), "wb") as fh:
            pq.write_table(table, fh, compression="zstd")

    # Fan out the shard writes: an 8M-pool layout is ~32k small parquets, hours sequential.
    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(_write_shard, range(n_shards)))
    index = pa.table(
        {
            "shard": list(range(n_shards)),
            "n_warcs": [len(by_shard[s]) for s in range(n_shards)],
            "region": [region_of[s] for s in range(n_shards)],
        }
    )
    with fsspec.open(index_path(spec), "wb") as fh:
        pq.write_table(index, fh, compression="zstd")
    sizes = sorted(len(v) for v in by_shard.values())
    logger.info(
        "built %d shards for %d WARCs (min/median/max %d/%d/%d WARCs) across regions %s -> %s",
        n_shards,
        len(warc_paths),
        sizes[0],
        sizes[len(sizes) // 2],
        sizes[-1],
        regions,
        shards_root(spec),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="materialize the frozen shard layout for a run")
    b.add_argument("--spec", default="lpv11_fastpipe_v2_1")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="Existing flat manifest file to shard.")
    src.add_argument("--snapshots", nargs="+", help="Common Crawl snapshots to draw WARCs from.")
    b.add_argument("--n-warcs", type=int, default=None, help="Sample this many WARCs (seeded, sorted pool).")
    b.add_argument("--n-shards", type=int, required=True)
    b.add_argument("--regions", nargs="+", required=True, help="Phase-A regions, assigned round-robin.")
    b.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = get_spec(args.spec)
    if spec.storage_version not in (3, 4):
        raise ValueError(f"{spec.spec_id} is not a sharded-worklist (storage_version 3/4) spec")
    if args.manifest:
        paths = _load_manifest(args.manifest)
    else:
        paths = []
        for snap in args.snapshots:
            got = _cc_snapshot_paths(snap)
            logger.info("%s: %d WARCs", snap, len(got))
            paths.extend(got)
    if args.n_warcs is not None:
        paths = random.Random(args.seed).sample(sorted(paths), args.n_warcs)
    build(spec, paths, args.n_shards, args.regions, seed=args.seed)


if __name__ == "__main__":
    main()
