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
from zephyr.readers import load_file

from experiments.fsspec_paths import fsspec_exists
from experiments.infinigram.build import build_index, plan_chunks
from experiments.infinigram.gcs_io import download_dir, download_file, upload_dir, upload_file
from experiments.infinigram.provenance import build_provenance_map
from experiments.infinigram.query import index_dirs_for, smoke_test_index
from experiments.infinigram.resolve import ResolvedTarget, resolve_target
from experiments.infinigram.stage import collect_wanted_hashes, stage_corpus
from experiments.infinigram.targets import Collection, IndexTarget, get_target
from experiments.infinigram.upload import IndexStats, manifest_url, write_manifest

logger = logging.getLogger(__name__)

# Leave this much RAM for the OS / gsutil / page cache when sizing --mem.
_MEM_HEADROOM_GIB = 8
# Leave this much disk free (staging scratch, OS) when sizing the chunk budget.
_DISK_HEADROOM_GIB = 10
# Peak disk per chunk ~= staged gz (1x) + decompressed data (~3.5x) + build temp
# (~parts/merged) + index; empirically a ~12.7 GB gz corpus indexes through
# make-part on ~100 GB disk, so cap chunk gz at free_disk / this.
_DISK_FACTOR = 8.0


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


def _source_has_url(shard_url: str) -> bool:
    """Whether the corpus already carries url inline (peek the first record)."""
    for rec in load_file(shard_url):
        return bool(rec.get("url"))
    return False


def _mark(index_dir: str, phase: str) -> None:
    """Write a tiny progress marker to GCS so a crashed build's last phase is
    recoverable when task logs are unavailable (list ``<index_dir>/_progress/``)."""
    try:
        with fsspec.open(f"{index_dir.rstrip('/')}/_progress/{phase}", "w") as f:
            f.write(phase)
    except Exception as e:
        logger.warning("progress marker %s failed: %s", phase, e)


# The `--mem` arg only bounds indexing.py's make-part step; the merge/concat step
# peaks at ~3x the DECOMPRESSED corpus regardless of --mem, so total RAM is set by
# corpus size, not --mem. We therefore cap each chunk's gzip bytes so the merge
# fits the container: decompressed ~= GZ_INFLATE x gz, merge ~= MERGE_FACTOR x
# decompressed, kept under MEM_SAFETY x container. --mem itself (make-part budget)
# is a moderate fraction of the container.
_MEM_SAFETY = 0.8
_GZ_INFLATE = 3.5  # decompressed / gzip
# indexer merge loads the full suffix array (~5x decompressed) into RAM; measured
# empirically higher (a 12 GB gz chunk OOM'd a 160 GB box), so budget conservatively.
_MERGE_FACTOR = 8.0  # indexer peak RAM / decompressed
# --mem (make-part budget) as a fraction of container. Higher = fewer, larger
# batches = faster indexing; must stay under the container (make-part peaks ~--mem)
# with room for the merge (~MEM_SAFETY x container is reserved for that).
_INDEX_MEM_FRACTION = 0.35
# Hard cap on chunk size. Big-RAM containers only fit on preemptible/reserved TPU
# nodes that get reclaimed roughly every ~30 min, so chunks MUST finish faster than
# that or resume never accumulates. ~2 GB gz builds+compresses in ~10-12 min.
_MAX_CHUNK_GZ_BYTES = 2 * 1024**3


def _index_mem_gib(container_mem_gib: int) -> int:
    return max(4, int(container_mem_gib * _INDEX_MEM_FRACTION))


def _mem_chunk_budget_bytes(container_mem_gib: int) -> int:
    """Max gzip bytes per chunk so the indexer's merge fits the container RAM."""
    usable = container_mem_gib * _MEM_SAFETY * 1024**3
    return min(_MAX_CHUNK_GZ_BYTES, max(1, int(usable / (_MERGE_FACTOR * _GZ_INFLATE))))


def build_index_for_target(
    target: IndexTarget,
    *,
    local_root: str,
    cpus: int | None = None,
    mem_gib: int | None = None,
    container_mem_gib: int | None = None,
    disk_budget_gib: int | None = None,
    num_workers: int = 1,
    worker_index: int = 0,
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
        logger.info("%s already complete (manifest exists); nothing to do.", target.name)
        return index_dir
    # Only a single-worker build may auto-clear; under striding, clearing would wipe
    # other workers' chunks, so the launcher pre-clears once before fanning out.
    if overwrite and num_workers == 1:
        _clear_index_dir(index_dir)

    # Chunk boundaries must be STABLE across preemption restarts, so derive the
    # budget only from the explicitly-passed container mem + disk budget (never the
    # node's free disk, which varies by where a restart lands).
    disk_gib = disk_budget_gib or max(1, _free_disk_gib(local_root) - _DISK_HEADROOM_GIB)
    disk_chunk_bytes = int(disk_gib * 1024**3 / _DISK_FACTOR)
    mem_chunk_bytes = _mem_chunk_budget_bytes(container_mem_gib)
    chunk_budget = min(disk_chunk_bytes, mem_chunk_bytes)
    chunks = plan_chunks(list(resolved.shard_bytes), chunk_budget)
    single = len(chunks) == 1

    def _shard_url(c: int) -> str:
        return index_dir.rstrip("/") if single else f"{index_dir.rstrip('/')}/{c:03d}"

    logger.info(
        "%s: %d shards, %.1f GiB -> %d chunk(s) (chunk<=%.1f GiB gz); %d cpus, --mem %d of %d container",
        target.name,
        resolved.shard_count,
        resolved.total_bytes / 1024**3,
        len(chunks),
        chunk_budget / 1024**3,
        cpus,
        mem_gib,
        container_mem_gib,
    )

    # This worker owns chunks c where c % num_workers == worker_index (lets many
    # jobs share one huge corpus). Resume skips chunks already done by any worker.
    pending = [
        c
        for c in range(len(chunks))
        if c % num_workers == worker_index and not fsspec_exists(_chunk_done_url(_shard_url(c)))
    ]
    logger.info(
        "%s: worker %d/%d owns %d pending chunks of %d total",
        target.name,
        worker_index,
        num_workers,
        len(pending),
        len(chunks),
    )

    # Build the provenance map ONCE (not per chunk), and only if some chunk needs
    # it AND the source is text-only. Sources that already carry url inline (e.g.
    # filtered subsets) skip the join, which would otherwise waste ~40 min reading
    # the raw provenance tier for no gain.
    prov_map: dict[str, dict] | None = None
    if target.provenance_globs and pending and not _source_has_url(resolved.shard_urls[0]):
        _mark(index_dir, "building_provenance_map")
        wanted = collect_wanted_hashes(resolved.shard_urls)
        prov_map = build_provenance_map(target.provenance_globs, wanted)
        _mark(index_dir, f"provenance_map_{len(prov_map)}of{len(wanted)}")

    chunk_reports: list[dict] = []
    for c in pending:
        idxs = chunks[c]
        shard_url = _shard_url(c)
        chunk_resolved = ResolvedTarget(
            target=target,
            shard_urls=tuple(resolved.shard_urls[i] for i in idxs),
            shard_bytes=tuple(resolved.shard_bytes[i] for i in idxs),
        )
        shard_local = os.path.join(local_root, "index", f"{c:03d}")
        chunk_work = os.path.join(local_root, f"work_{c:03d}")

        _mark(index_dir, f"chunk{c:03d}_staging")
        with stage_corpus(chunk_resolved, chunk_work, prov_map=prov_map) as staged:
            _mark(index_dir, f"chunk{c:03d}_staged_{staged.doc_count}docs")
            built = build_index(
                staged, shard_local, mem_gib=mem_gib, cpus=cpus, temp_dir=os.path.join(chunk_work, "_tmp")
            )
            _mark(index_dir, f"chunk{c:03d}_built")
            if verify:
                chunk_reports.append(_verify_chunk(list(built.shard_dirs), c))
            upload_dir(shard_local, shard_url)
            upload_file(staged.url_index_path, f"{shard_url.rstrip('/')}/url_index.jsonl.gz")
            _write_chunk_done(
                shard_url,
                {"doc_count": staged.doc_count, "matched": staged.matched_provenance, "index_bytes": built.index_bytes},
            )
            _mark(index_dir, f"chunk{c:03d}_uploaded")
        shutil.rmtree(shard_local, ignore_errors=True)
        shutil.rmtree(chunk_work, ignore_errors=True)

    # Finalize (url-index reassembly + manifest) only once EVERY chunk is done --
    # under striding another worker may still be building. Whichever worker sees the
    # last chunk complete writes the manifest (idempotent if two race).
    if not all(fsspec_exists(_chunk_done_url(_shard_url(c))) for c in range(len(chunks))):
        logger.info("%s: this worker finished its chunks; others still building -- not finalizing", target.name)
        return index_dir

    stats = IndexStats()
    for c in range(len(chunks)):
        info = _read_chunk_done(_shard_url(c))
        stats.doc_count += info["doc_count"]
        stats.provenance_matched += info["matched"]
        stats.index_bytes += info["index_bytes"]

    url_index_url = f"{index_dir.rstrip('/')}/url_index.jsonl.gz"
    combined_url_index = os.path.join(local_root, "url_index.jsonl.gz")
    with open(combined_url_index, "wb") as combined:
        for c in range(len(chunks)):
            part = os.path.join(local_root, f"uidx_{c:03d}.jsonl.gz")
            download_file(f"{_shard_url(c).rstrip('/')}/url_index.jsonl.gz", part)
            with open(part, "rb") as pf:
                shutil.copyfileobj(pf, combined)
            os.remove(part)
    upload_file(combined_url_index, url_index_url)
    write_manifest(
        index_dir,
        resolved,
        shard_dir_urls=[_shard_url(c) for c in range(len(chunks))],
        url_index_url=url_index_url,
        stats=stats,
        num_chunks=len(chunks),
    )
    if verify and chunk_reports:
        _record_validation(index_dir, chunk_reports)
    return index_dir


def _chunk_done_url(shard_url: str) -> str:
    return f"{shard_url.rstrip('/')}/.chunk_done.json"


def _write_chunk_done(shard_url: str, info: dict) -> None:
    with fsspec.open(_chunk_done_url(shard_url), "w") as f:
        json.dump(info, f)


def _read_chunk_done(shard_url: str) -> dict:
    with fsspec.open(_chunk_done_url(shard_url), "r") as f:
        return json.load(f)


def _clear_index_dir(index_dir: str) -> None:
    """Delete an index dir's contents for a clean rebuild (fresh --overwrite)."""
    fs = fsspec.filesystem("gcs")
    if fs.exists(index_dir):
        fs.rm(index_dir, recursive=True)
        logger.info("cleared %s for rebuild", index_dir)


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
    p.add_argument(
        "--disk-budget-gib",
        type=int,
        default=None,
        help="Disk budget for chunk sizing (pass to keep chunks stable across restarts).",
    )
    p.add_argument("--num-workers", type=int, default=1, help="Total workers sharing this index's chunks.")
    p.add_argument("--worker-index", type=int, default=0, help="This worker's index in [0, num-workers).")
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
        disk_budget_gib=args.disk_budget_gib,
        num_workers=args.num_workers,
        worker_index=args.worker_index,
        overwrite=args.overwrite,
        verify=not args.no_verify,
    )
    logger.info("Done: %s -> %s", target.name, index_dir)


if __name__ == "__main__":
    main()
