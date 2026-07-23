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
from marin.utils import fsspec_exists

from experiments.infinigram.build import build_index, plan_chunks
from experiments.infinigram.gcs_io import download_dir, upload_dir, upload_file
from experiments.infinigram.query import index_dirs_for, smoke_test_index
from experiments.infinigram.resolve import ResolvedTarget, resolve_target
from experiments.infinigram.stage import stage_corpus
from experiments.infinigram.targets import Collection, IndexTarget, get_target
from experiments.infinigram.upload import IndexStats, manifest_url, write_manifest

logger = logging.getLogger(__name__)

# Leave this much RAM for the OS / gsutil / page cache when sizing --mem.
_MEM_HEADROOM_GIB = 8
# Leave this much disk free (staging scratch, OS) when sizing the chunk budget.
_DISK_HEADROOM_GIB = 10


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            v = f.read().strip().split()[0]
        return None if v == "max" else int(v)
    except (FileNotFoundError, ValueError, IndexError):
        return None


def _cgroup_cpus() -> int | None:
    """CPU quota from the container cgroup (v2 then v1); None if unlimited.

    ``sched_getaffinity`` reports the whole node when the container is only
    quota-limited (not cpuset-pinned), so it over-reports on shared big nodes.
    """
    v2 = _read_int("/sys/fs/cgroup/cpu.max")  # first field is the quota (or "max")
    if v2 is not None:
        period = _read_int_field("/sys/fs/cgroup/cpu.max", 1) or 100000
        return max(1, round(v2 / period))
    quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and quota > 0 and period:
        return max(1, round(quota / period))
    return None


def _read_int_field(path: str, field: int) -> int | None:
    try:
        with open(path) as f:
            parts = f.read().strip().split()
        return int(parts[field])
    except (FileNotFoundError, ValueError, IndexError):
        return None


def _available_cpus() -> int:
    """CPUs the container may use — cgroup quota first, else affinity."""
    cg = _cgroup_cpus()
    if cg:
        return cg
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def _total_mem_gib() -> int:
    """Container memory limit in GiB (the indexer's --mem budget must fit the cgroup).

    Reads the cgroup limit (v2 ``memory.max`` then v1 ``memory.limit_in_bytes``);
    both report an enormous sentinel when unlimited, in which case we fall back to
    the node's MemTotal.
    """
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        limit = _read_int(path)
        if limit is not None and limit < (1 << 62):  # not the "unlimited" sentinel
            return limit // 1024**3
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // (1024 * 1024)
    raise RuntimeError("could not determine memory limit")


def _free_disk_gib(path: str) -> int:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) // 1024**3


# infini-gram-mini's indexing.py splits the corpus into batches so that peak SA
# RAM stays near the `--mem` argument (each of `cpus` parallel jobs handles
# ~mem/(12*cpus) bytes at ~12x RAM, so cpus jobs sum to ~mem); a bigger corpus
# just uses more batches, not more RAM. On top of --mem sits prepare()/rust
# overhead, so we budget --mem at a fraction of the container to leave slack.
_MEM_SAFETY = 0.5


def _index_mem_gib(container_mem_gib: int) -> int:
    return max(4, int(container_mem_gib * _MEM_SAFETY))


def build_index_for_target(
    target: IndexTarget,
    *,
    local_root: str,
    cpus: int | None = None,
    mem_gib: int | None = None,
    container_mem_gib: int | None = None,
    disk_budget_gib: int | None = None,
    overwrite: bool = False,
    verify: bool = True,
) -> str:
    """Build, verify, and upload the index for ``target``; return its GCS dir.

    The corpus is split into disk-budgeted chunks (usually one); each chunk is
    staged, indexed into its own shard dir, verified locally, uploaded, and then
    cleared before the next — bounding peak disk to a single chunk so 100s-of-GB
    corpora build on a modest CPU VM. Shard dirs are queried jointly.
    """
    cpus = cpus or _available_cpus()
    container_mem_gib = container_mem_gib or _total_mem_gib()
    mem_gib = mem_gib or _index_mem_gib(container_mem_gib)

    resolved = resolve_target(target)
    index_dir = target.index_dir
    if fsspec_exists(manifest_url(index_dir)) and not overwrite:
        raise FileExistsError(f"{manifest_url(index_dir)} exists; pass overwrite=True to rebuild {target.name}")

    budget = int((disk_budget_gib or max(1, _free_disk_gib(local_root) - _DISK_HEADROOM_GIB)) * 1024**3)
    chunks = plan_chunks(list(resolved.shard_bytes), budget)
    single = len(chunks) == 1
    logger.info(
        "%s: %d shards, %.1f GiB -> %d chunk(s); %d cpus, --mem %d GiB (of %d container), disk budget %.0f GiB",
        target.name,
        resolved.shard_count,
        resolved.total_bytes / 1024**3,
        len(chunks),
        cpus,
        mem_gib,
        container_mem_gib,
        budget / 1024**3,
    )

    stats = IndexStats()
    shard_dir_urls: list[str] = []
    chunk_reports: list[dict] = []
    combined_url_index = os.path.join(local_root, "url_index.jsonl.gz")

    with open(combined_url_index, "wb") as combined:
        for c, idxs in enumerate(chunks):
            chunk_resolved = ResolvedTarget(
                target=target,
                shard_urls=tuple(resolved.shard_urls[i] for i in idxs),
                shard_bytes=tuple(resolved.shard_bytes[i] for i in idxs),
            )
            shard_local = os.path.join(local_root, "index") if single else os.path.join(local_root, "index", f"{c:03d}")
            shard_url = index_dir.rstrip("/") if single else f"{index_dir.rstrip('/')}/{c:03d}"
            chunk_work = os.path.join(local_root, f"work_{c:03d}")

            with stage_corpus(chunk_resolved, chunk_work) as staged:
                # temp under chunk_work so it is reclaimed between chunks (bounds disk).
                built = build_index(
                    staged, shard_local, mem_gib=mem_gib, cpus=cpus, temp_dir=os.path.join(chunk_work, "_tmp")
                )
                stats.doc_count += staged.doc_count
                stats.provenance_matched += staged.matched_provenance
                stats.index_bytes += built.index_bytes
                with open(staged.url_index_path, "rb") as uf:  # gzip streams concatenate
                    shutil.copyfileobj(uf, combined)
                if verify:
                    chunk_reports.append(_verify_chunk(list(built.shard_dirs), c))
                upload_dir(shard_local, shard_url)
                shard_dir_urls.append(shard_url)
            shutil.rmtree(shard_local, ignore_errors=True)
            shutil.rmtree(chunk_work, ignore_errors=True)

    url_index_url = f"{index_dir.rstrip('/')}/url_index.jsonl.gz"
    upload_file(combined_url_index, url_index_url)
    write_manifest(
        index_dir,
        resolved,
        shard_dir_urls=shard_dir_urls,
        url_index_url=url_index_url,
        stats=stats,
        num_chunks=len(chunks),
    )
    if verify:
        _record_validation(index_dir, chunk_reports)
    return index_dir


def _verify_chunk(shard_dirs: list[str], chunk: int) -> dict:
    """Smoke-test one freshly built chunk locally; never raises (report only)."""
    try:
        report = smoke_test_index(shard_dirs)
        return {"chunk": chunk, "ok": True, **report}
    except Exception as e:
        logger.exception("chunk %d verification failed", chunk)
        return {"chunk": chunk, "ok": False, "error": f"{type(e).__name__}: {e}"}


def _record_validation(index_dir: str, chunk_reports: list[dict]) -> None:
    report = {"ok": all(r.get("ok") for r in chunk_reports), "chunks": chunk_reports}
    with fsspec.open(f"{index_dir.rstrip('/')}/validation.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote validation report -> %s/validation.json (ok=%s)", index_dir, report["ok"])


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
    # Verify all shard dirs jointly as a single logical unit.
    report = _verify_chunk(local_dirs, 0)
    _record_validation(index_dir, [report])
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one infini-gram-mini index.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--collection", choices=[c.value for c in Collection], default=Collection.FULL.value)
    p.add_argument("--local-root", default=os.environ.get("INFINIGRAM_LOCAL_ROOT", "/scratch/infinigram"))
    p.add_argument("--cpus", type=int, default=None, help="Override auto-detected CPU count.")
    p.add_argument(
        "--container-mem-gib",
        type=int,
        default=None,
        help="Container RAM limit (else cgroup); used to size per-job --mem.",
    )
    p.add_argument("--mem-gib", type=int, default=None, help="Override the PER-CPU-JOB --mem passed to indexing.py.")
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
        container_mem_gib=args.container_mem_gib,
        overwrite=args.overwrite,
        verify=not args.no_verify,
    )
    logger.info("Done: %s -> %s", target.name, index_dir)


if __name__ == "__main__":
    main()
