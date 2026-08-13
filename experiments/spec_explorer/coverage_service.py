# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coverage / Jaccard across curation strategies from their ``keys.parquet``.

The ~264 MB ``keys.parquet`` set (one small UINT64-keyed table per dataset, one
per region) is mirrored to a local cache **once**; everything after is a fast
local DuckDB query — no repeated cross-region reads. From those keys we serve:

- the pairwise Jaccard / containment matrix for a chosen identity key
  (``url_h`` universal, ``rid_h`` source-page, ``text_h`` identical text), via
  :func:`experiments.url_index.coverage.compute`; and
- set-expression **counts** ("docs in dclm ∩ nemotron but not high_quality"),
  the primitive behind the set-difference document viewer (text bytes are
  resolved by the in-region worker in a later phase).

``stats.json:url_match_rate`` is carried alongside so the UI can warn when a low
``rid_h``/``text_h`` overlap is provenance-join loss, not true disjointness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb

from experiments.infinigram.targets import Collection
from experiments.spec_explorer.catalog import INDEX_DATASET_REGION
from experiments.url_index import coverage, layout

logger = logging.getLogger(__name__)

COLLECTION = Collection.SMALL
_CACHE_ROOT = Path(__file__).parent / "cache" / "keys" / COLLECTION.value
KEY_CHOICES = ("url_h", "rid_h", "text_h", "dom_h")


def _client():
    from google.cloud import storage

    return storage.Client()


def _local_keys(dataset: str) -> Path:
    return _CACHE_ROOT / dataset / layout.KEYS_NAME


def _local_stats(dataset: str) -> Path:
    return _CACHE_ROOT / dataset / layout.STATS_NAME


def ensure_keys_local(datasets: list[str] | None = None) -> list[str]:
    """Mirror each dataset's keys.parquet (+ stats.json) locally; return datasets present.

    Idempotent: skips files already cached. Downloads run via ADC (small UINT64
    tables, ~264 MB total for all ten datasets — a one-time cost).
    """
    datasets = datasets or list(INDEX_DATASET_REGION)
    client = _client()
    present: list[str] = []
    for ds in datasets:
        region = INDEX_DATASET_REGION.get(ds)
        if region is None:
            logger.warning("no index region for %s; skipping", ds)
            continue
        keys_dst = _local_keys(ds)
        if not keys_dst.exists():
            keys_dst.parent.mkdir(parents=True, exist_ok=True)
            src = layout.keys_path(region, COLLECTION, ds)  # gs://marin-<region>/...
            bucket, blob = src.replace("gs://", "").split("/", 1)
            logger.info("downloading keys for %s (%s)", ds, region)
            client.bucket(bucket).blob(blob).download_to_filename(str(keys_dst))
        # stats.json is tiny and optional
        stats_dst = _local_stats(ds)
        if not stats_dst.exists():
            try:
                s = layout.stats_path(region, COLLECTION, ds)
                b, bn = s.replace("gs://", "").split("/", 1)
                client.bucket(b).blob(bn).download_to_filename(str(stats_dst))
            except Exception as e:
                logger.debug("no stats.json for %s: %s", ds, e)
        present.append(ds)
    return present


def _key_globs(datasets: list[str]) -> list[str]:
    return [str(_local_keys(ds)) for ds in datasets if _local_keys(ds).exists()]


def url_match_rates(datasets: list[str]) -> dict[str, float | None]:
    """Per-dataset ``url_match_rate`` from cached stats.json (None if absent)."""
    out: dict[str, float | None] = {}
    for ds in datasets:
        p = _local_stats(ds)
        try:
            out[ds] = json.loads(p.read_text()).get("url_match_rate") if p.exists() else None
        except Exception:
            out[ds] = None
    return out


def matrix(key: str = "url_h", datasets: list[str] | None = None) -> dict:
    """Pairwise coverage for ``key``: sizes, Jaccard + containment + a_only/b_only.

    Returns a frontend-ready object: ordered ``datasets``, per-dataset ``sizes``,
    a ``jaccard`` and ``containment`` matrix (row a, col b), and the raw ``pairs``.
    """
    if key not in KEY_CHOICES:
        raise ValueError(f"key must be one of {KEY_CHOICES}, got {key!r}")
    present = ensure_keys_local(datasets)
    globs = _key_globs(present)
    if not globs:
        return {"datasets": [], "sizes": {}, "jaccard": [], "containment": [], "pairs": []}

    ds_list, sizes, pairs = coverage.compute(globs, key)
    idx = {d: i for i, d in enumerate(ds_list)}
    n = len(ds_list)
    jac = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    con = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for r in pairs:
        jac[idx[r["a"]]][idx[r["b"]]] = r["jaccard"]
        con[idx[r["a"]]][idx[r["b"]]] = r["containment_a_in_b"]
    return {
        "key": key,
        "datasets": ds_list,
        "sizes": sizes,
        "jaccard": jac,
        "containment": con,
        "pairs": pairs,
        "url_match_rate": url_match_rates(ds_list),
    }


def set_expression(include: list[str], exclude: list[str], key: str = "url_h", sample: int = 0, offset: int = 0) -> dict:
    """Count (and optionally sample) keys in (intersection of include) minus (union of exclude).

    ``include``: datasets a doc must appear in (intersection). ``exclude``:
    datasets it must NOT appear in. ``sample`` > 0 returns that many key values,
    ordered by a stable hash (reproducible), for the doc viewer to resolve later.
    ``offset`` pages through that same deterministic order (e.g. the "next 8").
    """
    if key not in KEY_CHOICES:
        raise ValueError(f"key must be one of {KEY_CHOICES}, got {key!r}")
    if not include:
        raise ValueError("need at least one dataset in `include`")
    present = set(ensure_keys_local(list({*include, *exclude})))
    include = [d for d in include if d in present]
    exclude = [d for d in exclude if d in present]
    globs = _key_globs([*include, *exclude])
    if not include or not globs:
        return {"count": 0, "sample": [], "include": include, "exclude": exclude, "key": key, "offset": offset}

    con = duckdb.connect()
    glist = ", ".join(f"'{g}'" for g in globs)
    con.execute(
        f"CREATE TABLE d AS SELECT DISTINCT dataset, {key} AS k " f"FROM read_parquet([{glist}]) WHERE {key} IS NOT NULL"
    )
    inc = ", ".join(f"'{d}'" for d in include)
    base = f"SELECT k FROM d WHERE dataset IN ({inc}) " f"GROUP BY k HAVING count(DISTINCT dataset) = {len(include)}"
    if exclude:
        exc = ", ".join(f"'{d}'" for d in exclude)
        result_sql = f"({base}) EXCEPT (SELECT k FROM d WHERE dataset IN ({exc}))"
    else:
        result_sql = base

    count = con.execute(f"SELECT count(*) FROM ({result_sql})").fetchone()[0]
    keys: list[str] = []
    if sample > 0:
        # Deterministic pseudo-random order via hash of the key; OFFSET pages through it.
        rows = con.execute(
            f"SELECT k FROM ({result_sql}) ORDER BY hash(k) LIMIT {int(sample)} OFFSET {int(offset)}"
        ).fetchall()
        keys = [str(r[0]) for r in rows]
    con.close()
    return {"count": int(count), "sample": keys, "include": include, "exclude": exclude, "key": key, "offset": offset}


def which_datasets_have(url_hs: list[int]) -> dict[int, list[str]]:
    """For each ``url_h``, the datasets whose keys contain it (i.e. that kept the doc).

    Uses the locally-cached ``keys.parquet`` set — the same data behind the
    coverage matrix — so the search's kept/dropped cross-reference is a fast
    local lookup, no worker round-trip.
    """
    wanted = [int(h) for h in url_hs if h is not None]
    if not wanted:
        return {}
    present = ensure_keys_local()
    globs = _key_globs(present)
    if not globs:
        return {h: [] for h in wanted}
    con = duckdb.connect()
    glist = ", ".join(f"'{g}'" for g in globs)
    values = ", ".join(str(h) for h in set(wanted))
    rows = con.execute(
        f"SELECT DISTINCT dataset, url_h FROM read_parquet([{glist}]) WHERE url_h IN ({values})"
    ).fetchall()
    con.close()
    out: dict[int, list[str]] = {h: [] for h in wanted}
    for ds, h in rows:
        out.setdefault(int(h), []).append(ds)
    for h in out:
        out[h] = sorted(out[h])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    m = matrix("url_h")
    print(f"datasets: {m['datasets']}")
    print(f"sizes: {m['sizes']}")
    for r in sorted(m["pairs"], key=lambda x: -x["jaccard"])[:6]:
        print(f"  {r['a']:>16} ∩ {r['b']:<16} jaccard={r['jaccard']:.4f} a_only={r['a_only']}")
    print("\nset-expr: dclm minus high_quality:", set_expression(["dclm"], ["high_quality"])["count"])


if __name__ == "__main__":
    main()
