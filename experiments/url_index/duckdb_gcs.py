# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Let DuckDB read ``gs://`` parquet through the authenticated gcsfs filesystem.

DuckDB's built-in httpfs has no GCS credentials (it 403s), but gcsfs does. We
register gcsfs as a DuckDB filesystem so ``read_parquet('gs://...')`` routes
through it -- only when a path actually needs it, so local-only callers (tests)
never touch GCS.
"""

import duckdb
import fsspec


def maybe_register_gcs(con: duckdb.DuckDBPyConnection, paths: list[str]) -> None:
    """Register gcsfs with ``con`` if any of ``paths`` is a ``gs://`` URL."""
    if any(p.startswith("gs://") for p in paths):
        con.register_filesystem(fsspec.filesystem("gcs"))


def _container_memory_bytes() -> int | None:
    """The cgroup memory limit (v2 then v1), or None if unbounded/unreadable.

    DuckDB otherwise sizes its buffer pool off the *host* RAM, which on a shared
    node is hundreds of GiB -- it then blows past the container cgroup limit and
    gets OOM-killed (exit 137). We read the real limit and cap DuckDB below it.
    """
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        val = int(raw)
        if val <= 0 or val >= (1 << 62):  # cgroup v1 unlimited sentinel
            return None
        return val
    return None


def cap_memory(con: duckdb.DuckDBPyConnection, headroom_frac: float = 0.6) -> None:
    """Cap DuckDB ``memory_limit`` at ``headroom_frac`` of the container's cgroup limit.

    No-op when the limit is unknown (e.g. local dev), where DuckDB's default is fine.
    """
    b = _container_memory_bytes()
    if b is not None:
        con.execute(f"SET memory_limit = '{max(2.0, round(headroom_frac * b / 1024**3, 1))}GB'")
