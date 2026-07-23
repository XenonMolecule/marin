# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Interactive URL lookup across datasets: what did each pipeline extract?

Opens the local ``url_lookup_keys.duckdb`` (built by
:mod:`~experiments.url_index.consolidate`) and answers, for a URL or domain,
which datasets have it and their provenance -- sub-100ms, since url_key/domain
are indexed. ``--with-text`` additionally fetches the extracted text from the
per-region ``text.parquet`` store (local copies via ``--text-dir`` if present,
otherwise a small pruned read of the in-region parquet).

    # exact URL across all datasets
    python -m experiments.url_index.lookup --db /tmp/url_lookup_small.duckdb "www.example.com/page"
    # every page of a domain
    python -m experiments.url_index.lookup --db /tmp/url_lookup_small.duckdb --domain example.com
    # show the actual extracted text side by side
    python -m experiments.url_index.lookup --db /tmp/url_lookup_small.duckdb --with-text "example.com/page"
"""

import argparse
import logging
import os

import duckdb

from experiments.infinigram.targets import Collection
from experiments.url_index import layout
from experiments.url_index.duckdb_gcs import maybe_register_gcs
from experiments.url_index.keys import domain_of, registrable_domain, url_key

logger = logging.getLogger(__name__)

_LOOKUP_COLUMNS = "dataset, region, url_key, domain, warc_record_id, snapshot, warc_file, text_len"


def _collection(con: duckdb.DuckDBPyConnection) -> Collection:
    return Collection(con.execute("SELECT collection FROM index_meta").fetchone()[0])


def lookup_url(con: duckdb.DuckDBPyConnection, raw_url: str) -> list[dict]:
    """Rows for the exact normalized url_key of ``raw_url``, one per matching dataset doc."""
    key = url_key(raw_url)
    rows = con.execute(f"SELECT {_LOOKUP_COLUMNS} FROM docs WHERE url_key = ? ORDER BY dataset", [key]).fetchall()
    cols = [c.strip() for c in _LOOKUP_COLUMNS.split(",")]
    return [dict(zip(cols, r, strict=True)) for r in rows]


def lookup_domain(con: duckdb.DuckDBPyConnection, raw_domain: str) -> list[dict]:
    """Rows for every page whose registrable domain matches ``raw_domain``."""
    dom = registrable_domain(raw_domain) or domain_of(raw_domain)
    rows = con.execute(
        f"SELECT {_LOOKUP_COLUMNS} FROM docs WHERE domain = ? ORDER BY url_key, dataset", [dom]
    ).fetchall()
    cols = [c.strip() for c in _LOOKUP_COLUMNS.split(",")]
    return [dict(zip(cols, r, strict=True)) for r in rows]


def fetch_texts(rows: list[dict], collection: Collection, text_dir: str | None) -> dict[tuple, str]:
    """Fetch extracted text for matched rows, keyed by (dataset, url_key).

    Groups by (dataset, region) and issues one pruned parquet read per group,
    preferring a local ``{text_dir}/{collection}/{dataset}/text.parquet`` copy.
    """
    by_source: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        by_source.setdefault((r["dataset"], r["region"]), []).append(r["url_key"])

    out: dict[tuple, str] = {}
    con = duckdb.connect()
    for (dataset, region), keys in by_source.items():
        local = os.path.join(text_dir, collection.value, dataset, layout.TEXT_NAME) if text_dir else None
        path = local if local and os.path.exists(local) else layout.text_path(region, collection, dataset)
        maybe_register_gcs(con, [path])
        placeholders = ", ".join("?" for _ in keys)
        res = con.execute(
            f"SELECT url_key, text FROM read_parquet('{path}') WHERE url_key IN ({placeholders})", keys
        ).fetchall()
        for uk, text in res:
            out[(dataset, uk)] = text
    con.close()
    return out


def _print_rows(rows: list[dict], texts: dict[tuple, str] | None) -> None:
    if not rows:
        print("  (no matches in any dataset)")
        return
    for r in rows:
        line = (
            f"  [{r['dataset']:<16}] url_key={r['url_key']}  "
            f"len={r['text_len']}  snapshot={r['snapshot']}  rid={r['warc_record_id']}"
        )
        print(line)
        if texts is not None:
            text = texts.get((r["dataset"], r["url_key"]), "")
            snippet = text if len(text) <= 2000 else text[:2000] + f"… (+{len(text) - 2000} chars)"
            for ln in snippet.splitlines() or [""]:
                print(f"      | {ln}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Look up a URL or domain across all extraction datasets.")
    p.add_argument("queries", nargs="*", help="URLs (or domains with --domain) to look up.")
    p.add_argument("--db", required=True, help="Local url_lookup_*.duckdb from consolidate.py.")
    p.add_argument("--domain", action="store_true", help="Treat queries as registrable domains.")
    p.add_argument("--with-text", action="store_true", help="Also fetch and print the extracted text.")
    p.add_argument("--text-dir", default=None, help="Local dir of downloaded text.parquet copies (optional).")
    p.add_argument("--queries-file", default=None, help="File with one query per line (in addition to args).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    queries = list(args.queries)
    if args.queries_file:
        with open(args.queries_file) as f:
            queries.extend(ln.strip() for ln in f if ln.strip())
    if not queries:
        raise SystemExit("no queries given")

    con = duckdb.connect(args.db, read_only=True)
    collection = _collection(con)
    for q in queries:
        print(f"\n=== {q} ===")
        rows = lookup_domain(con, q) if args.domain else lookup_url(con, q)
        texts = fetch_texts(rows, collection, args.text_dir) if (args.with_text and rows) else None
        _print_rows(rows, texts)
    con.close()


if __name__ == "__main__":
    main()
