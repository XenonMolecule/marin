# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed cache of edge-case searches, with favorites.

Keyed by a normalized query so re-running a query returns the saved result
instantly. Favorites are just a flag on a saved row. Small and self-contained
(one file under ``cache/``); no external deps.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

_DB_PATH = Path(__file__).parent / "cache" / "searches.sqlite"
_lock = threading.Lock()


def _key(query: str) -> str:
    return hashlib.sha1(query.strip().lower().encode()).hexdigest()


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS searches ("
        "query_key TEXT PRIMARY KEY, query TEXT, result_json TEXT, "
        "favorite INTEGER DEFAULT 0, created_at REAL, elapsed REAL, result_count INTEGER)"
    )
    return con


def save(query: str, result: dict) -> None:
    """Upsert the latest result for ``query`` (preserving any existing favorite flag)."""
    with _lock:
        con = _conn()
        fav = con.execute("SELECT favorite FROM searches WHERE query_key = ?", [_key(query)]).fetchone()
        con.execute(
            "INSERT OR REPLACE INTO searches"
            "(query_key, query, result_json, favorite, created_at, elapsed, result_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                _key(query),
                query,
                json.dumps(result),
                fav[0] if fav else 0,
                time.time(),
                result.get("elapsed"),
                len(result.get("results") or []),
            ],
        )
        con.commit()
        con.close()


def get(query: str) -> dict | None:
    """Return the cached result for ``query`` (or None)."""
    with _lock:
        con = _conn()
        row = con.execute("SELECT result_json FROM searches WHERE query_key = ?", [_key(query)]).fetchone()
        con.close()
    return json.loads(row[0]) if row and row[0] else None


def set_favorite(query: str, favorite: bool) -> None:
    with _lock:
        con = _conn()
        con.execute("UPDATE searches SET favorite = ? WHERE query_key = ?", [1 if favorite else 0, _key(query)])
        con.commit()
        con.close()


def delete(query: str) -> None:
    with _lock:
        con = _conn()
        con.execute("DELETE FROM searches WHERE query_key = ?", [_key(query)])
        con.commit()
        con.close()


def clear_results() -> int:
    """Drop cached results but keep the saved query terms + favorite flags.

    Also removes internal comparison artifacts (rows whose query carries the
    about-only mode marker). Returns the number of terms preserved.
    """
    with _lock:
        con = _conn()
        con.execute("DELETE FROM searches WHERE instr(query, char(0)) > 0 OR query LIKE '%about-only'")
        con.execute("UPDATE searches SET result_json = NULL, elapsed = NULL, result_count = NULL")
        con.commit()
        kept = con.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
        con.close()
    return kept


def list_saved(favorites_only: bool = False) -> list[dict]:
    """List saved searches (favorites first, then most-recent), without the full result blob."""
    with _lock:
        con = _conn()
        sql = "SELECT query, favorite, created_at, elapsed, result_count FROM searches"
        if favorites_only:
            sql += " WHERE favorite = 1"
        sql += " ORDER BY favorite DESC, created_at DESC"
        rows = con.execute(sql).fetchall()
        con.close()
    return [
        {"query": q, "favorite": bool(f), "created_at": c, "elapsed": e, "result_count": n} for (q, f, c, e, n) in rows
    ]
