# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invoke infini-gram-mini's ``indexing.py`` on a staged local corpus.

The indexer, its prebuilt ``cpp_indexing`` binary, and the gcc-5 / isl / mpc /
mpfr toolchain it links against live in a vendored checkout pointed to by
``INFINIGRAM_MINI_DIR`` (baked into the job image; see ``packaging.md``). We only
ever shell out to it -- it is never imported into the marin ``uv`` environment.

Corpora above :data:`SHARD_BYTES_THRESHOLD` are split into independent shards
(each its own ``save_dir`` subdir), mirroring the repo's ``index_v2_dclm.py``.
Our datasets are far below that, so in practice every build is a single shard.
"""

import logging
import os
import resource
import subprocess
from dataclasses import dataclass

from experiments.infinigram.stage import StagedCorpus

logger = logging.getLogger(__name__)

# Per-shard corpus cap (the paper indexes <=700 GiB per shard under 2 TiB RAM).
SHARD_BYTES_THRESHOLD = 700 * 1024**3

# Files a SUCCESSFUL indexing.py leaves in save_dir. The final `cpp_indexing`
# compression step consumes the intermediate *.sdsl (suffix array / BWT) files and
# emits the compressed FM-indexes `data.fm9` / `meta.fm9` (+ offsets). Their
# presence is what proves compression actually ran (its absence = the silent
# libcilkrts failure that leaves a huge, unqueryable index).
_INDEX_FILES = (
    "data.fm9",
    "meta.fm9",
    "data_offset",
    "meta_offset",
)


@dataclass(frozen=True)
class BuildResult:
    """A finished local index (one or more shard dirs under save_dir)."""

    save_dir: str
    shard_dirs: tuple[str, ...]
    index_bytes: int


def _infinigram_dir() -> str:
    d = os.environ.get("INFINIGRAM_MINI_DIR")
    if not d:
        raise RuntimeError("INFINIGRAM_MINI_DIR is not set (should be baked into the job image).")
    return d


def plan_shards(byte_count: int) -> int:
    """Number of index shards for a corpus of this size (1 for all our datasets)."""
    return max(1, -(-byte_count // SHARD_BYTES_THRESHOLD))


def plan_chunks(shard_bytes: list[int], chunk_byte_budget: int) -> list[list[int]]:
    """Group shard indices into contiguous chunks each under ``chunk_byte_budget``.

    ``chunk_byte_budget`` is a GZIP-byte budget (the caller derives it from the
    tighter of the disk and memory limits). Each chunk is built, uploaded as its
    own index shard dir, and cleared before the next — bounding both peak disk and
    the indexer's peak RAM to a single chunk. A single shard larger than the budget
    gets its own chunk (shards are atomic). Returns lists of shard indices.
    """
    if chunk_byte_budget <= 0:
        raise ValueError(f"chunk_byte_budget must be positive, got {chunk_byte_budget}")
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_sum = 0
    for i, b in enumerate(shard_bytes):
        if cur and cur_sum + b > chunk_byte_budget:
            chunks.append(cur)
            cur, cur_sum = [], 0
        cur.append(i)
        cur_sum += b
    if cur:
        chunks.append(cur)
    return chunks


def _partition_shard_dirs(data_dir: str, num_shards: int, work_dir: str) -> list[str]:
    """Symlink the staged files into ``num_shards`` subdirs so each indexes independently."""
    files = sorted(f for f in os.listdir(data_dir) if ".json" in f)
    shard_data_dirs: list[str] = []
    for s in range(num_shards):
        sd = os.path.join(work_dir, f"shard_{s:02d}")
        os.makedirs(sd, exist_ok=True)
        for f in files[s::num_shards]:
            link = os.path.join(sd, f)
            if not os.path.exists(link):
                os.symlink(os.path.join(data_dir, f), link)
        shard_data_dirs.append(sd)
    return shard_data_dirs


def _max_open_files() -> int:
    """The container's hard RLIMIT_NOFILE. indexing.py's default (1048576) exceeds
    what a non-root process may set, so pass the real ceiling."""
    _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return 1048576 if hard == resource.RLIM_INFINITY else hard


def _run_indexing(data_dir: str, save_dir: str, temp_dir: str, *, mem_gib: int, cpus: int) -> None:
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    cmd = [
        "python",
        os.path.join(_infinigram_dir(), "src", "indexing.py"),
        "--data_dir",
        os.path.abspath(data_dir),
        "--save_dir",
        os.path.abspath(save_dir),
        "--temp_dir",
        os.path.abspath(temp_dir),
        "--mem",
        str(mem_gib),
        "--cpus",
        str(cpus),
        "--ulimit",
        str(_max_open_files()),
    ]
    logger.info("Running: %s", " ".join(cmd))
    # indexing.py invokes `./cpp_indexing` by relative path, so run it from src/.
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(_infinigram_dir(), "src"))
    if proc.returncode != 0:
        raise RuntimeError(f"indexing.py failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}")
    missing = [f for f in _INDEX_FILES if not os.path.exists(os.path.join(save_dir, f))]
    if missing:
        raise RuntimeError(f"index build produced no {missing} in {save_dir}; stderr tail:\n{proc.stderr[-2000:]}")


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def build_index(
    staged: StagedCorpus,
    save_dir: str,
    *,
    mem_gib: int,
    cpus: int,
    temp_dir: str | None = None,
) -> BuildResult:
    """Build an FM-index over ``staged`` into ``save_dir``.

    Single-shard corpora index straight into ``save_dir``; larger ones fan out
    into ``save_dir/NN`` shard subdirs that are queried jointly. Scratch (temp,
    shard-partition symlinks) lives beside ``save_dir`` so it is never uploaded.
    """
    work_base = os.path.dirname(os.path.abspath(save_dir))
    temp_dir = temp_dir or os.path.join(work_base, "_tmp")
    num_shards = plan_shards(staged.byte_count)
    logger.info("Building index: %.2f GiB -> %d shard(s)", staged.byte_count / 1024**3, num_shards)

    if num_shards == 1:
        _run_indexing(staged.data_dir, save_dir, temp_dir, mem_gib=mem_gib, cpus=cpus)
        shard_dirs = (save_dir,)
    else:
        shard_data_dirs = _partition_shard_dirs(staged.data_dir, num_shards, os.path.join(work_base, "_shards"))
        shard_dirs = tuple(os.path.join(save_dir, f"{s:02d}") for s in range(num_shards))
        for data_dir, out_dir in zip(shard_data_dirs, shard_dirs, strict=True):
            _run_indexing(
                data_dir, out_dir, os.path.join(temp_dir, os.path.basename(out_dir)), mem_gib=mem_gib, cpus=cpus
            )

    return BuildResult(save_dir=save_dir, shard_dirs=shard_dirs, index_bytes=_dir_bytes(save_dir))
