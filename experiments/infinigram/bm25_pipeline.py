# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end streaming BM25 index build: resolve -> stream shards -> upload.

The BM25 backup mirror of :mod:`experiments.infinigram.pipeline`, but streaming:
cluster workers cap at 100 GiB disk, so we never stage the whole corpus. We read
each resolved shard from GCS in-region, recover url/warc provenance the same way
the infini-gram staging does (exact text-hash join to the raw batches), and feed
a ``(text, metadata)`` stream to :func:`~experiments.infinigram.bm25_build.
stream_build`. Each finished sub-index is uploaded to GCS, content-verified, and
deleted locally, so disk and RAM stay bounded no matter how large the corpus is.

One invocation builds exactly one index (one dataset, one collection), pinned to
the dataset's region.

    python -m experiments.infinigram.bm25_pipeline --dataset dclm --collection full
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Iterator

import fsspec
from marin.utils import fsspec_exists
from zephyr.readers import load_file

from experiments.infinigram.bm25_build import (
    TEXT_BYTES_BUDGET,
    Bm25ShardResult,
    bm25_index_dir,
    iter_shard_metrics,
    stream_build,
)
from experiments.infinigram.bm25_query import MANIFEST_NAME, Bm25Hit, load_local_index
from experiments.infinigram.bm25_sources import get_bm25_target
from experiments.infinigram.gcs_io import upload_dir
from experiments.infinigram.provenance import build_provenance_map, content_hash
from experiments.infinigram.resolve import resolve_target
from experiments.infinigram.targets import Collection, IndexTarget

logger = logging.getLogger(__name__)

# Content probe used to prove retrieval works on the freshly built shards.
_SMOKE_QUERY = "the United States of America"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _manifest_url(index_dir: str) -> str:
    return f"{index_dir.rstrip('/')}/{MANIFEST_NAME}"


def _doc_text(rec: dict) -> str | None:
    return rec.get("text") or rec.get("generated_text")


def _provenance_map(target: IndexTarget, shard_urls: tuple[str, ...]) -> dict[str, dict]:
    """Build content-hash -> {url, warc ids} for text-only tiers (empty otherwise).

    Mirrors the infini-gram staging: a first pass over the corpus collects the
    text hashes needing provenance, then the raw batches are scanned once. Reads
    are in-region (the job is region-pinned).
    """
    if not target.provenance_globs:
        return {}
    wanted: set[str] = set()
    for url in shard_urls:
        for rec in load_file(url):
            text = _doc_text(rec)
            if text:
                wanted.add(content_hash(text))
    return build_provenance_map(target.provenance_globs, wanted)


def _iter_docs(shard_urls: tuple[str, ...], prov_map: dict[str, dict], stats: dict) -> Iterator[dict]:
    """Yield documents (dicts with ``text``) from the resolved shards, provenance-joined."""
    matched = 0
    for url in shard_urls:
        for rec in load_file(url):
            text = _doc_text(rec)
            if not text:
                continue
            if prov_map:
                prov = prov_map.get(content_hash(text))
                if prov:
                    rec = {**rec, **prov}
                    matched += 1
            if rec.get("text") is None:
                rec = {**rec, "text": text}
            yield rec
    stats["provenance_matched"] = matched


def build_bm25_for_target(
    target: IndexTarget,
    *,
    local_root: str,
    text_bytes_budget: int = TEXT_BYTES_BUDGET,
    overwrite: bool = False,
    verify: bool = True,
) -> str:
    """Stream-build, verify, and upload the BM25 index for ``target``; return its GCS dir."""
    index_dir = bm25_index_dir(target)
    if fsspec_exists(_manifest_url(index_dir)) and not overwrite:
        raise FileExistsError(f"{_manifest_url(index_dir)} exists; pass --overwrite to rebuild {target.name}")

    resolved = resolve_target(target)
    prov_map = _provenance_map(target, resolved.shard_urls)
    save_dir = os.path.join(local_root, "bm25_index")

    uploaded: list[str] = []
    best_hit: list[Bm25Hit] = []  # best content-probe hit seen across shards
    fs = fsspec.filesystem("gcs")

    def on_shard_built(index: int, result: Bm25ShardResult) -> None:
        gcs_dir = f"{index_dir.rstrip('/')}/{os.path.basename(result.shard_dir)}"
        upload_dir(result.shard_dir, gcs_dir)
        uploaded.append(gcs_dir)
        if verify:
            sub = load_local_index([result.shard_dir], mmap=False)
            if sub.num_docs != result.doc_count:
                raise AssertionError(f"{gcs_dir}: loaded num_docs={sub.num_docs} != built {result.doc_count}")
            hits = sub.search(_SMOKE_QUERY, k=1)
            if hits and (not best_hit or hits[0].score > best_hit[0].score):
                best_hit[:] = hits[:1]
        shutil.rmtree(result.shard_dir, ignore_errors=True)

    stats: dict = {}
    build = stream_build(
        _iter_docs(resolved.shard_urls, prov_map, stats),
        save_dir,
        text_bytes_budget=text_bytes_budget,
        on_shard_built=on_shard_built,
    )
    if verify and not best_hit:
        raise AssertionError(f"content probe {_SMOKE_QUERY!r} returned no hits across {len(uploaded)} shards")
    shutil.rmtree(save_dir, ignore_errors=True)

    smoke = {}
    if best_hit:
        m = best_hit[0].metadata
        smoke = {"query": _SMOKE_QUERY, "top_score": round(best_hit[0].score, 4), "top_url": m.get("url")}

    manifest = {
        "engine": "bm25",
        "dataset": target.dataset,
        "collection": target.collection.value,
        "region": target.region,
        "shard_dirs": uploaded,
        "num_sub_indices": len(uploaded),
        "num_input_shards": resolved.shard_count,
        "input_bytes": resolved.total_bytes,
        "index_bytes": build.index_bytes,
        "index_to_input_ratio": round(build.index_bytes / resolved.total_bytes, 4) if resolved.total_bytes else None,
        "doc_count": build.doc_count,
        "provenance_joined": bool(target.provenance_globs),
        "provenance_matched": stats.get("provenance_matched", 0),
        "build_seconds": round(build.build_seconds, 2),
        "text_bytes_budget": text_bytes_budget,
        "shard_metrics": iter_shard_metrics(build.shards),
        "smoke": smoke,
        "source_shard_urls_sample": list(resolved.shard_urls[:5]),
        "git_sha": _git_sha(),
    }
    with fs.open(_manifest_url(index_dir), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "Done: %s -> %s (%d docs, %d shards, %.1fs)",
        target.name,
        index_dir,
        build.doc_count,
        len(uploaded),
        build.build_seconds,
    )
    return index_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream-build one BM25 backup index.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--collection", choices=[c.value for c in Collection], default=Collection.FULL.value)
    p.add_argument("--local-root", default=os.environ.get("BM25_LOCAL_ROOT", "/tmp/bm25/run"))
    p.add_argument(
        "--text-bytes-budget",
        type=int,
        default=int(os.environ.get("BM25_TEXT_BYTES_BUDGET", TEXT_BYTES_BUDGET)),
        help="Uncompressed text bytes buffered per bm25s sub-index (bounds build RAM).",
    )
    p.add_argument("--overwrite", action="store_true", help="Rebuild even if an index already exists.")
    p.add_argument("--no-verify", action="store_true", help="Skip per-shard content verification.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    target = get_bm25_target(args.dataset, Collection(args.collection))
    os.makedirs(args.local_root, exist_ok=True)
    build_bm25_for_target(
        target,
        local_root=args.local_root,
        text_bytes_budget=args.text_bytes_budget,
        overwrite=args.overwrite,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
