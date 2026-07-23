# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: resolve -> stage -> build -> verify -> upload one index.

This module is both the reusable orchestrator (:func:`build_index_for_target`)
and the CLI that runs inside an Iris CPU job. One invocation builds exactly one
index (one dataset, one collection), pinned to the dataset's region.

    python -m experiments.infinigram.pipeline --dataset dclm --collection full
"""

import argparse
import json
import logging
import os
import shutil

import fsspec

from experiments.infinigram.build import build_index
from experiments.infinigram.gcs_io import download_dir
from experiments.infinigram.query import index_dirs_for, smoke_test_index
from experiments.infinigram.resolve import resolve_target
from experiments.infinigram.stage import stage_corpus
from experiments.infinigram.targets import Collection, IndexTarget, get_target
from experiments.infinigram.upload import upload_index

logger = logging.getLogger(__name__)

# Leave this much RAM for the OS / gsutil / page cache when sizing --mem.
_MEM_HEADROOM_GIB = 8


def _available_cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def _total_mem_gib() -> int:
    """Total RAM in GiB from /proc/meminfo (the indexer's --mem is a budget hint)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // (1024 * 1024)
    raise RuntimeError("could not read MemTotal from /proc/meminfo")


def build_index_for_target(
    target: IndexTarget,
    *,
    local_root: str,
    cpus: int | None = None,
    mem_gib: int | None = None,
    overwrite: bool = False,
    verify: bool = True,
) -> str:
    """Build, verify, and upload the index for ``target``; return its GCS dir."""
    cpus = cpus or _available_cpus()
    mem_gib = mem_gib or max(1, _total_mem_gib() - _MEM_HEADROOM_GIB)

    resolved = resolve_target(target)
    save_dir = os.path.join(local_root, "index")
    with stage_corpus(resolved, local_root) as staged:
        built = build_index(staged, save_dir, mem_gib=mem_gib, cpus=cpus)
        # Co-locate the url->id side table with the index so it uploads together.
        shutil.copy(staged.url_index_path, os.path.join(save_dir, os.path.basename(staged.url_index_path)))
        index_dir = upload_index(built, resolved, staged, overwrite=overwrite)

    # Verify AFTER upload (so an expensive build is never discarded on a toolchain
    # hiccup) and record the report to the index dir for out-of-band inspection.
    if verify:
        _verify_and_record(list(built.shard_dirs), index_dir)
    return index_dir


def verify_existing_target(target: IndexTarget, local_root: str) -> dict:
    """Download an already-uploaded index and validate it (no rebuild).

    Records the report to ``<index_dir>/validation.json`` and returns it.
    """
    index_dir = target.index_dir
    shard_dirs = index_dirs_for(target)  # from the uploaded manifest.json
    local_dirs = []
    for d in shard_dirs:
        dest = os.path.join(local_root, "verify", d.replace("gs://", ""))
        download_dir(d, dest)
        local_dirs.append(dest)
    _verify_and_record(local_dirs, index_dir)
    with fsspec.open(f"{index_dir.rstrip('/')}/validation.json", "r") as f:
        return json.load(f)


def _verify_and_record(shard_dirs: list[str], index_dir: str) -> None:
    report: dict
    try:
        report = smoke_test_index(shard_dirs)
        report["ok"] = True
    except Exception as e:
        logger.exception("verification failed")
        report = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    with fsspec.open(f"{index_dir.rstrip('/')}/validation.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote validation report -> %s/validation.json (ok=%s)", index_dir, report.get("ok"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one infini-gram-mini index.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--collection", choices=[c.value for c in Collection], default=Collection.FULL.value)
    p.add_argument("--local-root", default=os.environ.get("INFINIGRAM_LOCAL_ROOT", "/scratch/infinigram"))
    p.add_argument("--cpus", type=int, default=None, help="Override auto-detected CPU count.")
    p.add_argument("--mem-gib", type=int, default=None, help="Override auto-detected --mem budget.")
    p.add_argument("--overwrite", action="store_true", help="Rebuild even if an index already exists.")
    p.add_argument("--no-verify", action="store_true", help="Skip the post-upload smoke test.")
    p.add_argument("--verify-only", action="store_true", help="Validate an already-uploaded index; do not build.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    target = get_target(args.dataset, Collection(args.collection))
    os.makedirs(args.local_root, exist_ok=True)
    if args.verify_only:
        report = verify_existing_target(target, args.local_root)
        logger.info("Verify-only %s -> ok=%s", target.name, report.get("ok"))
        return
    index_dir = build_index_for_target(
        target,
        local_root=args.local_root,
        cpus=args.cpus,
        mem_gib=args.mem_gib,
        overwrite=args.overwrite,
        verify=not args.no_verify,
    )
    logger.info("Done: %s -> %s", target.name, index_dir)


if __name__ == "__main__":
    main()
