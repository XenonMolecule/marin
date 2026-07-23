# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: resolve -> stage -> build BM25 -> verify -> upload one index.

The BM25 backup mirror of :mod:`experiments.infinigram.pipeline`. It reuses the
exact same resolve/stage/provenance machinery (so the BM25 index covers the same
documents, with the same recovered url/warc provenance, as the infini-gram index)
but swaps the FM-index builder for a pure-pip ``bm25s`` index -- no gcc/sdsl
toolchain, so this job needs no ``run_in_toolchain.sh`` wrapper.

One invocation builds exactly one index (one dataset, one collection), pinned to
the dataset's region.

    python -m experiments.infinigram.bm25_pipeline --dataset llm_pipeline_v1 --collection small
"""

import argparse
import json
import logging
import os
import subprocess

import fsspec
from marin.utils import fsspec_exists

from experiments.infinigram.bm25_build import (
    SHARD_COMPRESSED_BYTES,
    Bm25BuildResult,
    bm25_index_dir,
    build_bm25_index,
    iter_shard_metrics,
)
from experiments.infinigram.bm25_query import MANIFEST_NAME, smoke_test_index
from experiments.infinigram.gcs_io import upload_dir
from experiments.infinigram.resolve import ResolvedTarget, resolve_target
from experiments.infinigram.stage import StagedCorpus, stage_corpus
from experiments.infinigram.targets import Collection, IndexTarget, get_target

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _manifest_url(index_dir: str) -> str:
    return f"{index_dir.rstrip('/')}/{MANIFEST_NAME}"


def _upload(
    build: Bm25BuildResult,
    resolved: ResolvedTarget,
    staged: StagedCorpus,
    smoke: dict,
    index_dir: str,
) -> None:
    """Upload every sub-index dir + the url_index, then a metrics-rich manifest."""
    for shard in build.shards:
        rel = os.path.basename(shard.shard_dir)
        upload_dir(shard.shard_dir, f"{index_dir.rstrip('/')}/{rel}")

    url_index_dest = f"{index_dir.rstrip('/')}/{os.path.basename(staged.url_index_path)}"
    fs = fsspec.filesystem("gcs")
    fs.put_file(staged.url_index_path, url_index_dest)

    target = resolved.target
    manifest = {
        "engine": "bm25",
        "dataset": target.dataset,
        "collection": target.collection.value,
        "region": target.region,
        "shard_dirs": [f"{index_dir.rstrip('/')}/{os.path.basename(s.shard_dir)}" for s in build.shards],
        "url_index": url_index_dest,
        "num_input_shards": resolved.shard_count,
        "input_bytes": resolved.total_bytes,
        "index_bytes": build.index_bytes,
        "doc_count": build.doc_count,
        "num_sub_indices": len(build.shards),
        "provenance_joined": bool(target.provenance_globs),
        "provenance_matched": staged.matched_provenance,
        # Efficiency metrics: end-to-end build time, per-shard breakdown, query latency.
        "build_seconds": round(build.build_seconds, 2),
        "index_to_input_ratio": round(build.index_bytes / resolved.total_bytes, 4) if resolved.total_bytes else None,
        "shard_metrics": list(iter_shard_metrics(build)),
        "smoke": smoke,
        "source_shard_urls_sample": list(resolved.shard_urls[:5]),
        "git_sha": _git_sha(),
    }
    with fsspec.open(_manifest_url(index_dir), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest %s", _manifest_url(index_dir))


def build_bm25_for_target(
    target: IndexTarget,
    *,
    local_root: str,
    shard_bytes: int = SHARD_COMPRESSED_BYTES,
    overwrite: bool = False,
    verify: bool = True,
) -> str:
    """Build, verify, and upload the BM25 index for ``target``; return its GCS dir."""
    index_dir = bm25_index_dir(target)
    if fsspec_exists(_manifest_url(index_dir)) and not overwrite:
        raise FileExistsError(f"{_manifest_url(index_dir)} exists; pass --overwrite to rebuild {target.name}")

    resolved = resolve_target(target)
    save_dir = os.path.join(local_root, "bm25_index")
    with stage_corpus(resolved, local_root) as staged:
        build = build_bm25_index(staged, save_dir, shard_bytes=shard_bytes)
        smoke = smoke_test_index(list(build.shard_dirs), doc_count=build.doc_count) if verify else {}
        _upload(build, resolved, staged, smoke, index_dir)
    logger.info("Done: %s -> %s (%d docs, %.1fs)", target.name, index_dir, build.doc_count, build.build_seconds)
    return index_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one BM25 backup index.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--collection", choices=[c.value for c in Collection], default=Collection.FULL.value)
    p.add_argument("--local-root", default=os.environ.get("BM25_LOCAL_ROOT", "/tmp/bm25/run"))
    p.add_argument(
        "--shard-bytes",
        type=int,
        default=int(os.environ.get("BM25_SHARD_BYTES", SHARD_COMPRESSED_BYTES)),
        help="Max compressed bytes of staged shards per bm25s sub-index (bounds build RAM).",
    )
    p.add_argument("--overwrite", action="store_true", help="Rebuild even if an index already exists.")
    p.add_argument("--no-verify", action="store_true", help="Skip the pre-upload smoke test.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    target = get_target(args.dataset, Collection(args.collection))
    os.makedirs(args.local_root, exist_ok=True)
    build_bm25_for_target(
        target,
        local_root=args.local_root,
        shard_bytes=args.shard_bytes,
        overwrite=args.overwrite,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
