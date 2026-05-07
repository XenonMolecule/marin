# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply DCLM's `bff` (bloom filter dedup) Rust binary, parallelized via Zephyr.

The marin Iris cluster only provides n2-highmem-2 (2 vCPU / 16 GiB) CPU VMs, so
we cannot run a single ``bff`` invocation over all input shards. Instead we
group input files and run one ``bff`` invocation per group on a separate
Zephyr worker. Each group has its own bloom filter (no cross-group dedup).

DCLM's published pipeline does the same thing for multi-node parallelism (the
``--shard-num``/``--total-shards`` flags subdivide the corpus into independent
bff runs). Cross-shard duplicates remain — within-group near-duplicates are
caught.

Pipeline:
  1. List input files, partition into groups of ``shards_per_group`` files.
  2. ``Dataset.from_list(groups).map(_dedup_one_group)`` dispatches via Zephyr.
  3. Each worker:
       - Installs cmake (rootless via pip)
       - Clones DCLM at ``dclm_commit`` and ``cargo build --release`` (cached
         after first invocation per worker process)
       - Stages the group's input files locally
       - Runs ``bff bff`` with DCLM-paper params (no ``--annotate``)
       - Uploads outputs to ``output_path``

bff params follow ``dclm_full.yaml`` — paragraph + document removal via
``--remove-type old-both`` at ``--filtering-threshold 0.8``.
"""

import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

import fsspec

from fray.v2.types import ResourceConfig
from zephyr import Dataset, ZephyrContext, zephyr_worker_ctx

logger = logging.getLogger(__name__)

DEFAULT_DCLM_REPO_URL = "https://github.com/mlfoundations/dclm.git"


@dataclass
class BffDedupConfig:
    input_path: str
    """Glob pattern for input JSONL files (e.g. ``gs://.../*.jsonl.gz``)."""

    output_path: str
    """GCS directory to write deduplicated output JSONL files."""

    shards_per_group: int = 10
    """How many input files to feed into a single bff invocation. Each group
    gets its own bloom filter (no cross-group dedup)."""

    expected_ngram_count_per_group: int = 5_000_000_000
    """Estimate of n-grams in one group (over-estimate is fine).

    For 10 input files × ~250M tokens each, 13-grams ≈ tokens, so ~2.5B per
    group. Default of 5B leaves comfortable headroom for the bloom filter."""

    fp_rate: float = 0.01
    """Per-ngram false-positive rate."""

    min_ngram_size: int = 13
    """DCLM paper setting."""

    max_ngram_size: int = 13
    """DCLM paper setting."""

    filtering_threshold: float = 0.8
    """If this fraction of a paragraph's n-grams already in the bloom, drop it."""

    remove_type: str = "old-both"
    """DCLM paper setting (paragraph + document removal in old-both semantics)."""

    bff_threads: int = 2
    """bff worker threads. Match worker VM CPU count (n2-highmem-2 = 2)."""

    dclm_repo_url: str = DEFAULT_DCLM_REPO_URL
    """Upstream DCLM source repo to clone."""

    dclm_commit: str = ""
    """Optional commit SHA to pin. Empty string = HEAD of main."""

    max_workers: int = 100
    """Zephyr max parallel worker count."""


def _list_input_files(input_glob: str) -> list[str]:
    """Resolve ``input_glob`` to a sorted list of GCS URIs (or local paths)."""
    if input_glob.startswith("gs://"):
        fs = fsspec.filesystem("gcs")
    else:
        fs = fsspec.filesystem("file")
    paths = sorted(fs.glob(input_glob))
    if input_glob.startswith("gs://"):
        paths = [f"gs://{p}" if not p.startswith("gs://") else p for p in paths]
    if not paths:
        raise FileNotFoundError(f"No input files matched {input_glob!r}")
    logger.info("Found %d input files matching %r", len(paths), input_glob)
    return paths


def _list_existing_output_basenames(output_path: str) -> set[str]:
    """Return basenames already present in ``output_path`` so we can skip them."""
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    base = output_path.rstrip("/")
    try:
        listed = fs.ls(base, detail=False)
    except FileNotFoundError:
        return set()
    return {p.split("/")[-1] for p in listed if p.endswith(".jsonl.gz")}


def _partition(files: list[str], group_size: int) -> list[list[str]]:
    """Split files into consecutive groups of size <= group_size."""
    return [files[i : i + group_size] for i in range(0, len(files), group_size)]


def _run(cmd: list[str] | str, cwd: str | None = None) -> None:
    """Run a subprocess; log+raise on failure."""
    logger.info("$ %s", " ".join(cmd) if isinstance(cmd, list) else cmd)
    subprocess.run(cmd, check=True, cwd=cwd)


def _ensure_cmake() -> None:
    """Install cmake into the venv if it's not on PATH (rootless container)."""
    from shutil import which

    if which("cmake"):
        return
    _run(["pip", "install", "--quiet", "cmake"])
    if not which("cmake"):
        raise RuntimeError("cmake still not on PATH after pip install")


def _build_bff(repo_dir: str, repo_url: str, commit: str) -> str:
    """Clone DCLM and build the bff Rust binary. Returns absolute path to binary."""
    if not os.path.isdir(repo_dir):
        _run(["git", "clone", "--depth=1", repo_url, repo_dir])
        if commit:
            _run(["git", "fetch", "--depth=1", "origin", commit], cwd=repo_dir)
            _run(["git", "checkout", commit], cwd=repo_dir)
    bff_src = os.path.join(repo_dir, "dedup", "bff")
    _run(["cargo", "build", "--release"], cwd=bff_src)

    candidates = [
        os.path.join(bff_src, "target", "release", "bff"),
        os.path.join(os.environ.get("CARGO_TARGET_DIR", "/root/.cargo/target"), "release", "bff"),
        "/root/.cargo/target/release/bff",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            logger.info("bff binary at %s", path)
            return path
    raise FileNotFoundError(f"bff binary not found in any of: {candidates}")


# Cache the built binary path per worker process so subsequent groups skip the
# install+build (10s cmake + 80s cargo cold, ~7s warm).
_WORKER_BFF_BIN: str | None = None
_WORKER_BUILD_DIR: str | None = None


def _get_or_build_bff(config: "BffDedupConfig") -> str:
    """Build the bff binary on first call per worker; reuse thereafter."""
    global _WORKER_BFF_BIN, _WORKER_BUILD_DIR
    if _WORKER_BFF_BIN and os.path.isfile(_WORKER_BFF_BIN):
        return _WORKER_BFF_BIN
    _ensure_cmake()
    if _WORKER_BUILD_DIR is None:
        _WORKER_BUILD_DIR = tempfile.mkdtemp(prefix="bff_build_")
    repo_dir = os.path.join(_WORKER_BUILD_DIR, "dclm")
    _WORKER_BFF_BIN = _build_bff(repo_dir, config.dclm_repo_url, config.dclm_commit)
    return _WORKER_BFF_BIN


def _dedup_one_group(group: list[str]) -> list[str]:
    """Zephyr map function: dedupe one group of input files. Returns uploaded GCS URIs."""
    ctx = zephyr_worker_ctx()
    config: BffDedupConfig = ctx.get_shared("bff_config")
    output_dir_gcs: str = ctx.get_shared("output_dir_gcs")
    already_done: set[str] = ctx.get_shared("already_done")

    # Skip files already deduped on a prior run.
    pending = [src for src in group if src.rsplit("/", 1)[-1] not in already_done]
    if not pending:
        logger.info("Group fully covered by existing outputs; skipping (%d files).", len(group))
        return []

    bff_bin = _get_or_build_bff(config)

    # Per-group scratch dirs.
    work = tempfile.mkdtemp(prefix="bff_group_")
    in_dir = os.path.join(work, "in")
    out_dir = os.path.join(work, "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    fs = fsspec.filesystem("gcs")

    # Stage inputs
    local_inputs: list[str] = []
    for src in pending:
        name = src.rsplit("/", 1)[-1]
        local = os.path.join(in_dir, name)
        if not os.path.exists(local):
            fs.get(src, local)
        local_inputs.append(local)
    total_in_bytes = sum(os.path.getsize(p) for p in local_inputs)
    logger.info(
        "Group: %d files (%.1f GB compressed) staged to %s",
        len(local_inputs),
        total_in_bytes / 1e9,
        in_dir,
    )

    # Run bff
    cmd = [
        bff_bin,
        "bff",
        "--inputs",
        in_dir,
        "--output-directory",
        out_dir,
        "--expected-ngram-count",
        str(config.expected_ngram_count_per_group),
        "--fp-rate",
        str(config.fp_rate),
        "--min-ngram-size",
        str(config.min_ngram_size),
        "--max-ngram-size",
        str(config.max_ngram_size),
        "--filtering-threshold",
        str(config.filtering_threshold),
        "--remove-type",
        config.remove_type,
        "--no-progress-bar",
        "--threads",
        str(config.bff_threads),
    ]
    _run(cmd)

    # Upload outputs
    uploaded: list[str] = []
    for name in sorted(os.listdir(out_dir)):
        local = os.path.join(out_dir, name)
        if not os.path.isfile(local) or not name.endswith(".jsonl.gz"):
            continue
        dst = f"{output_dir_gcs.rstrip('/')}/{name}"
        fs.put(local, dst)
        uploaded.append(dst)

    # Clean up scratch (keeps the build dir cached for subsequent groups).
    import shutil

    shutil.rmtree(work, ignore_errors=True)

    logger.info("Group done: %d output files uploaded.", len(uploaded))
    return uploaded


def bff_dedup(config: BffDedupConfig) -> None:
    """Run bff dedup over the input glob, parallelized per-group via Zephyr.

    Idempotent: groups whose outputs already exist in ``output_path`` are skipped.
    """
    all_inputs = _list_input_files(config.input_path)
    already_done = _list_existing_output_basenames(config.output_path)
    pending = [p for p in all_inputs if p.rsplit("/", 1)[-1] not in already_done]
    logger.info(
        "Plan: %d input files, %d already deduped, %d to process.",
        len(all_inputs),
        len(already_done),
        len(pending),
    )
    if not pending:
        logger.info("All %d output files already exist; nothing to do.", len(all_inputs))
        return

    groups = _partition(pending, config.shards_per_group)
    logger.info(
        "Partitioned %d pending files into %d groups of up to %d each.",
        len(pending),
        len(groups),
        config.shards_per_group,
    )

    pipeline = Dataset.from_list(groups).flat_map(_dedup_one_group)
    ctx = ZephyrContext(
        name="bff-dedup",
        max_workers=config.max_workers,
        # Each worker: bff bloom (~3GB for 2.5B ngrams) + bff binary (17MB) +
        # cargo cache (~3GB) + staged inputs (1-2GB compressed). 16 GB total VM
        # RAM (n2-highmem-2) is comfortable.
        resources=ResourceConfig(cpu=2, ram="14g", disk="50g"),
    )
    ctx.put("bff_config", config)
    ctx.put("output_dir_gcs", config.output_path)
    ctx.put("already_done", already_done)
    result = ctx.execute(pipeline).results

    # Write a stats sidecar so downstream consumers can read run metadata.
    stats = {
        "input_files_total": len(all_inputs),
        "input_files_already_deduped": len(already_done),
        "input_files_processed_this_run": len(pending),
        "groups_dispatched": len(groups),
        "shards_per_group": config.shards_per_group,
        "output_files_total_uploaded_this_run": sum(len(r) for r in result if isinstance(r, list)),
        "remove_type": config.remove_type,
        "min_ngram_size": config.min_ngram_size,
        "max_ngram_size": config.max_ngram_size,
        "filtering_threshold": config.filtering_threshold,
        "fp_rate": config.fp_rate,
        "expected_ngram_count_per_group": config.expected_ngram_count_per_group,
        "dclm_repo_url": config.dclm_repo_url,
        "dclm_commit": config.dclm_commit,
    }
    fs = fsspec.filesystem("gcs")
    stats_path = f"{config.output_path.rstrip('/')}/_bff_dedup_stats.json"
    with fs.open(stats_path, "w") as f:
        f.write(json.dumps(stats, indent=2))
    logger.info("bff dedup complete; stats → %s", stats_path)
