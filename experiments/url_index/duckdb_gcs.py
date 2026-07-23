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
