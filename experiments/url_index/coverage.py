# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pairwise coverage across datasets from their ``keys.parquet`` artifacts.

For a chosen identity key -- ``rid_h`` (source page; the default, right for
comparing *different* extractors) or ``text_h`` (identical extracted text; right
for tiers of the *same* pipeline) -- computes every unordered pair's
intersection, A-only, B-only, Jaccard and directional containment via a single
DuckDB self-join over the deduped ``(dataset, key)`` relation. Exact, no
approximation; seconds even at tens of millions of keys.

    python -m experiments.url_index.coverage \
        --keys gs://marin-us-central1/url_index/small/*/keys.parquet \
        --key rid_h --out-dir /tmp/coverage_small
"""

import argparse
import logging
import os

import duckdb

from experiments.url_index.duckdb_gcs import cap_memory, maybe_register_gcs

logger = logging.getLogger(__name__)

KEY_CHOICES = ("url_h", "rid_h", "text_h", "dom_h")


def _load(con: duckdb.DuckDBPyConnection, key_globs: list[str], key: str) -> None:
    """Materialize a deduped ``(dataset, key)`` relation from the keys parquets."""
    maybe_register_gcs(con, key_globs)
    cap_memory(con)
    globs = ", ".join(f"'{g}'" for g in key_globs)
    con.execute(
        f"CREATE TABLE d AS "
        f"SELECT DISTINCT dataset, {key} AS k "
        f"FROM read_parquet([{globs}]) WHERE {key} IS NOT NULL"
    )


def compute(key_globs: list[str], key: str) -> tuple[list[str], dict, list[dict]]:
    """Return (datasets, sizes, pair_rows) for the coverage of the given key.

    ``pair_rows`` has one row per ordered (a, b) with a != b: intersection,
    a_only, b_only, jaccard, and ``containment_a_in_b`` = |A∩B| / |A|.
    """
    con = duckdb.connect()
    _load(con, key_globs, key)
    sizes = {ds: n for ds, n in con.execute("SELECT dataset, count(*) FROM d GROUP BY dataset").fetchall()}
    datasets = sorted(sizes)

    inter = con.execute(
        "SELECT a.dataset, b.dataset, count(*) "
        "FROM d a JOIN d b USING (k) "
        "WHERE a.dataset <> b.dataset GROUP BY a.dataset, b.dataset"
    ).fetchall()
    con.close()

    inter_map = {(a, b): c for a, b, c in inter}
    rows: list[dict] = []
    for a in datasets:
        for b in datasets:
            if a == b:
                continue
            i = inter_map.get((a, b), 0)
            na, nb = sizes[a], sizes[b]
            union = na + nb - i
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "a_size": na,
                    "b_size": nb,
                    "intersection": i,
                    "a_only": na - i,
                    "b_only": nb - i,
                    "jaccard": round(i / union, 6) if union else 0.0,
                    "containment_a_in_b": round(i / na, 6) if na else 0.0,
                }
            )
    return datasets, sizes, rows


def _write_csvs(datasets: list[str], sizes: dict, rows: list[dict], out_dir: str, key: str) -> None:
    import csv

    os.makedirs(out_dir, exist_ok=True)
    long_path = os.path.join(out_dir, f"coverage_pairs_{key}.csv")
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["a", "b"])
        w.writeheader()
        w.writerows(rows)

    # Containment matrix: cell (row=a, col=b) = fraction of a contained in b.
    containment = {(r["a"], r["b"]): r["containment_a_in_b"] for r in rows}
    matrix_path = os.path.join(out_dir, f"containment_matrix_{key}.csv")
    with open(matrix_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a\\b", *datasets])
        for a in datasets:
            w.writerow([a] + [1.0 if a == b else containment.get((a, b), 0.0) for b in datasets])
    logger.info("Wrote %s and %s (sizes: %s)", long_path, matrix_path, sizes)


def _maybe_heatmap(datasets: list[str], rows: list[dict], out_dir: str, key: str) -> None:
    """Optional Jaccard heatmap PNG; silently skipped if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.info("matplotlib not available; skipping heatmap")
        return

    idx = {d: i for i, d in enumerate(datasets)}
    m = np.eye(len(datasets))
    for r in rows:
        m[idx[r["a"]], idx[r["b"]]] = r["jaccard"]
    fig, ax = plt.subplots(figsize=(1.4 * len(datasets) + 2, 1.4 * len(datasets) + 2))
    im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(datasets)), datasets, rotation=45, ha="right")
    ax.set_yticks(range(len(datasets)), datasets)
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title(f"Pairwise Jaccard ({key})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    png = os.path.join(out_dir, f"jaccard_heatmap_{key}.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", png)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise dataset coverage from keys.parquet artifacts.")
    p.add_argument("--keys", nargs="+", required=True, help="Glob(s) to keys.parquet files (gs:// or local).")
    p.add_argument("--key", choices=KEY_CHOICES, default="rid_h")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-heatmap", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    datasets, sizes, rows = compute(args.keys, args.key)
    _write_csvs(datasets, sizes, rows, args.out_dir, args.key)
    if not args.no_heatmap:
        _maybe_heatmap(datasets, rows, args.out_dir, args.key)


if __name__ == "__main__":
    main()
