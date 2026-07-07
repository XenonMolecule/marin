# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Cross-pipeline provenance audit for the 10k-WARC curation sweep.

All six pipelines (high_quality / dclm / nemotron_full / fineweb_edu, plus the
unfiltered resiliparse superset) extract the *same* 10,364 WARCs, and every doc
carries a ``url``. So each pipeline is a different keep/drop-and-rewrite decision
over one shared set of source URLs. This tool materialises that natural
experiment as a durable, reusable **per-URL membership table**:

    url, domain, kept_hq, kept_dclm, kept_nemo, kept_fwedu,
    hq_len, dclm_len, nemo_len, fwedu_len,
    hq_ft, hq_mbprob, dclm_ft, fineweb_score, nemo_quality, snapshot

From that one table we can answer, per source domain, "what fraction of URLs
does each pipeline keep?" and compute exact document-level set differences
(e.g. DCLM-minus-high_quality) that drive the eval gaps.

Design (mirrors matched_viewer.py — everything that touches the corpora runs
in-region on us-central2; cross-region reads are hard-refused):

  ``extract``  map-only, one durable parquet per input shard under
               ``.../per_method/{method}/{shard}.parquet`` (skip-if-exists →
               resumable). Streams each shard, emits url + domain + text_len +
               method-specific score columns. Memory-light per task.

  ``join``     single-node duckdb full-outer-join over all per-method parquets →
               the membership table + per-domain retention + set-difference
               summaries (small JSONs downloaded for local analysis).

Launch (us-central2, in-region, CPU):

    iris --cluster marin job run --region us-central2 --enable-extra-resources \\
        --memory 128GB --cpu 32 --extra marin \\
        -- python experiments/baseline_collection/provenance_audit_10k.py extract --method all

    iris --cluster marin job run --region us-central2 --enable-extra-resources \\
        --memory 128GB --cpu 16 --extra marin \\
        -- python experiments/baseline_collection/provenance_audit_10k.py join
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the battle-tested in-region guards + threaded GCS map from matched_viewer.
from experiments.baseline_collection.matched_viewer import (
    _assert_no_cross_region,
    _gcs_ls_glob,
    _open_read_gzip,
    _parallel_map,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("provenance_audit_10k")

# --------------------------------------------------------------------------- #
# 10k-WARC pipeline sources (all us-central2, confirmed present 2026-07-04)
# --------------------------------------------------------------------------- #
# NOTE: hq lives in us-central1; dclm/nemo/fwedu in us-central2. The join runs in
# us-central2, so the hq extract writes its (tiny) per-method parquet to a
# us-central1 workspace (pass --workspace) which is then copied into the
# central2 per_method dir before joining. WORKSPACE is the central2 default.
WORKSPACE = "gs://marin-us-central2/scratch/provenance_10k"
PER_METHOD_DIR = f"{WORKSPACE}/per_method"
MEMBERSHIP_DIR = f"{WORKSPACE}/membership"
SUMMARY_DIR = f"{WORKSPACE}/summary"

# fmt: off
SOURCES: dict[str, dict] = {
    # The FULL 10,364-WARC hq corpus the models trained on = the decon+dedup HF
    # export with url re-attached (19.97M docs, us-central1). NOT the 516-WARC
    # partial fastpipe_v3 run. This export has no fasttext/modernbert scores.
    "hq": {
        "root": "gs://marin-us-central1/documents/baseline_high_quality_hf_export/10364warcs/joined",
        "glob": "*.parquet",
        "format": "parquet",
        # schema: text,url,warc_record_id,warc_file,snapshot
        "scores": {},
        "snapshot_field": "snapshot",
    },
    "dclm": {
        "root": "gs://marin-us-central2/filtered/dclm_400m_1x_10k_dclm_resharded-1fe977",
        "glob": "*.jsonl.gz",
        "format": "jsonl",
        "scores": {"dclm_ft": "dclm_fasttext_score"},
        "snapshot_field": None,
    },
    "nemo": {
        "root": "gs://marin-us-central2/filtered/dclm_400m_1x_10k_nemotron_full-96bad9",
        "glob": "*.jsonl.gz",
        "format": "jsonl",
        "scores": {"nemo_quality": "nemotron_quality"},  # categorical str
        "snapshot_field": None,
    },
    "fwedu": {
        "root": "gs://marin-us-central2/filtered/dclm_400m_1x_10k_fineweb_edu-0d49e9",
        "glob": "**/*.jsonl.gz",
        "format": "jsonl",
        "scores": {"fineweb_score": "fineweb_score"},
        "snapshot_field": "dump",
    },
    # Unfiltered superset (denominator). 10,364 shards, 1:1 with WARCs — the
    # 2h bottleneck, so it is opt-in (extract --method resiliparse).
    "resiliparse": {
        "root": "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f",
        "glob": "*.jsonl.gz",
        "format": "jsonl",
        "scores": {},
        "snapshot_field": None,
    },
}
# fmt: on

FILTER_METHODS = ["hq", "dclm", "nemo", "fwedu"]

# Minimal multi-label public suffixes so bbc.co.uk / archiveofourown.org map to
# their registered domain rather than the bare host. Covers the named eval
# domains + common ccTLDs; unknown suffixes fall back to last-2-labels.
_MULTI_SUFFIXES = frozenset(
    {
        "co.uk",
        "org.uk",
        "gov.uk",
        "ac.uk",
        "me.uk",
        "co.jp",
        "or.jp",
        "ne.jp",
        "com.au",
        "net.au",
        "org.au",
        "gov.au",
        "edu.au",
        "co.nz",
        "com.br",
        "co.in",
        "co.za",
        "com.cn",
        "org.cn",
        "com.mx",
        "co.kr",
    }
)


def registered_domain(url: str) -> str:
    """Best-effort eTLD+1 for retention grouping. Not a full PSL, but correct for
    the eval-relevant domains (bbc.co.uk, archiveofourown.org, github.com, ...)."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if not host:
        return ""
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    last2 = ".".join(labels[-2:])
    if last2 in _MULTI_SUFFIXES:
        return ".".join(labels[-3:])
    return last2


# --------------------------------------------------------------------------- #
# Extract pass — one parquet per input shard
# --------------------------------------------------------------------------- #

_ARROW_SCHEMA_CACHE: dict[str, pa.Schema] = {}


def _score_value(rec: dict, field: str):
    """Coerce a stored score to float; leave categorical (nemotron_quality) as str."""
    v = rec.get(field)
    if v is None:
        return None
    if field == "nemotron_quality":
        return str(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _iter_jsonl(shard_path: str):
    with _open_read_gzip(shard_path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_parquet(shard_path: str):
    _assert_no_cross_region(shard_path)
    fs = fsspec.filesystem("gcs")
    with fs.open(shard_path, "rb") as fh:
        table = pq.read_table(fh)
    yield from table.to_pylist()


def extract_shard(method: str, shard_path: str, out_path: str) -> int:
    """Stream one input shard → a parquet of (url, domain, text_len, *scores).

    Returns the number of non-empty rows emitted. Skip-if-exists is handled by
    the caller so a preempted run resumes cleanly.
    """
    src = SOURCES[method]
    score_cols = src["scores"]
    snap_field = src["snapshot_field"]
    it = _iter_parquet(shard_path) if src["format"] == "parquet" else _iter_jsonl(shard_path)

    urls: list[str] = []
    domains: list[str] = []
    lens: list[int] = []
    snaps: list = []
    score_data: dict[str, list] = {c: [] for c in score_cols}

    for rec in it:
        url = rec.get("url")
        text = rec.get("text") or ""
        if not url or not text:
            continue
        urls.append(url)
        domains.append(registered_domain(url))
        lens.append(len(text))
        snaps.append(rec.get(snap_field) if snap_field else None)
        for col, field in score_cols.items():
            score_data[col].append(_score_value(rec, field))

    cols: dict[str, list] = {"url": urls, "domain": domains, "text_len": lens, "snapshot": snaps}
    cols.update(score_data)
    table = pa.table(cols)
    _assert_no_cross_region(out_path)
    fs = fsspec.filesystem("gcs")
    with fs.open(out_path, "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    return len(urls)


def _shard_out_path(method: str, shard_path: str, per_method_dir: str) -> str:
    # Flatten nested shard names (fineweb_edu is nested) into a unique key.
    root = SOURCES[method]["root"]
    rel = shard_path[len(root) :].lstrip("/").replace("/", "__")
    for suffix in (".jsonl.gz", ".parquet"):
        if rel.endswith(suffix):
            rel = rel[: -len(suffix)]
            break
    return f"{per_method_dir}/{method}/{rel}.parquet"


def run_extract(args: argparse.Namespace) -> int:
    methods = FILTER_METHODS if args.method == "all" else [args.method]
    per_method_dir = f"{args.workspace}/per_method"
    fs = fsspec.filesystem("gcs")

    for method in methods:
        src = SOURCES[method]
        shards = _gcs_ls_glob(f"{src['root']}/{src['glob']}")
        if args.limit_shards:
            shards = shards[: args.limit_shards]
        logger.info("[%s] %d shards -> %s", method, len(shards), per_method_dir)

        def _do(shard_path: str, method=method) -> int:
            out_path = _shard_out_path(method, shard_path, per_method_dir)
            exists_path = out_path[len("gs://") :] if out_path.startswith("gs://") else out_path
            if not args.overwrite and fs.exists(exists_path):
                return -1  # already done
            return extract_shard(method, shard_path, out_path)

        results = _parallel_map(_do, shards, f"extract {method}", max_workers=args.threads)
        done = sum(1 for r in results if r == -1)
        emitted = sum(r for r in results if r >= 0)
        logger.info(
            "[%s] %d shards skipped (cached), %d rows emitted from %d fresh shards",
            method,
            done,
            emitted,
            len(results) - done,
        )
    return 0


# --------------------------------------------------------------------------- #
# Join pass — duckdb full-outer-join → membership table + summaries
# --------------------------------------------------------------------------- #


def run_join(args: argparse.Namespace) -> int:
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    # gcsfs-backed httpfs: register the fsspec filesystem so duckdb reads gs://.
    con.register_filesystem(fsspec.filesystem("gcs"))

    methods = FILTER_METHODS
    # Per-method deduped url→(len,scores). A url can recur across shards (rare);
    # take max text_len as the representative.
    for m in methods:
        glob = f"{PER_METHOD_DIR}/{m}/*.parquet"
        _assert_no_cross_region(glob)
        score_cols = list(SOURCES[m]["scores"].keys())
        agg = ", ".join([f"max({c}) AS {c}" for c in score_cols])
        agg = (", " + agg) if agg else ""
        con.execute(
            f"""
            CREATE TEMP VIEW {m}_raw AS SELECT * FROM read_parquet('{glob}');
            CREATE TEMP TABLE {m}_t AS
              SELECT url,
                     any_value(domain) AS domain,
                     max(text_len) AS {m}_len,
                     any_value(snapshot) AS {m}_snapshot
                     {agg}
              FROM {m}_raw GROUP BY url;
        """
        )
        n = con.execute(f"SELECT count(*) FROM {m}_t").fetchone()[0]
        logger.info("[join] %s: %d unique urls", m, n)

    # Union all URLs once (domain is url-derived, so identical across methods),
    # then LEFT JOIN each method — unambiguous vs chained FULL OUTER JOINs.
    # Score columns are built dynamically from SOURCES (hq has none).
    score_sql = "".join(f", {m}.{c}" for m in methods for c in SOURCES[m]["scores"])
    con.execute(
        f"""
        CREATE TEMP TABLE all_urls AS
        SELECT url, any_value(domain) AS domain FROM (
            SELECT url, domain FROM hq_t    UNION ALL
            SELECT url, domain FROM dclm_t  UNION ALL
            SELECT url, domain FROM nemo_t  UNION ALL
            SELECT url, domain FROM fwedu_t
        ) GROUP BY url;

        CREATE TEMP TABLE membership AS
        SELECT
            a.url, a.domain,
            (hq.url IS NOT NULL)    AS kept_hq,
            (dclm.url IS NOT NULL)  AS kept_dclm,
            (nemo.url IS NOT NULL)  AS kept_nemo,
            (fwedu.url IS NOT NULL) AS kept_fwedu,
            hq.hq_len, dclm.dclm_len, nemo.nemo_len, fwedu.fwedu_len,
            coalesce(hq.hq_snapshot, fwedu.fwedu_snapshot) AS snapshot
            {score_sql}
        FROM all_urls a
        LEFT JOIN hq_t    hq    ON a.url = hq.url
        LEFT JOIN dclm_t  dclm  ON a.url = dclm.url
        LEFT JOIN nemo_t  nemo  ON a.url = nemo.url
        LEFT JOIN fwedu_t fwedu ON a.url = fwedu.url
    """
    )
    total = con.execute("SELECT count(*) FROM membership").fetchone()[0]
    logger.info("[join] membership rows (union of 4 filters): %d", total)
    # Log per-method kept counts (sanity: dclm~5.9M, nemo~17.7M, fwedu~2.35M, hq~19.97M).
    per_method = con.execute(
        "SELECT sum(kept_hq::INT), sum(kept_dclm::INT), sum(kept_nemo::INT), sum(kept_fwedu::INT) FROM membership"
    ).fetchone()
    logger.info("[join] kept counts hq/dclm/nemo/fwedu = %s", per_method)

    # Summaries FIRST (the small payoff), then the big durable COPY last so a COPY
    # failure never costs us the summaries.
    # --- Per-domain retention summary -------------------------------------- #
    dom = (
        con.execute(
            """
        SELECT domain,
               count(*) AS n_urls,
               sum(kept_hq::INT)    AS n_hq,
               sum(kept_dclm::INT)  AS n_dclm,
               sum(kept_nemo::INT)  AS n_nemo,
               sum(kept_fwedu::INT) AS n_fwedu
        FROM membership GROUP BY domain
        HAVING count(*) >= 20
        ORDER BY n_urls DESC
        LIMIT 5000
    """
        )
        .fetch_arrow_table()
        .to_pylist()
    )
    _write_json(f"{SUMMARY_DIR}/domain_retention.json", dom)
    logger.info("[join] wrote domain_retention.json (%d domains)", len(dom))

    # --- Pairwise set-difference counts + membership-pattern buckets ------- #
    patterns = (
        con.execute(
            """
        SELECT kept_hq, kept_dclm, kept_nemo, kept_fwedu, count(*) AS n
        FROM membership GROUP BY 1,2,3,4 ORDER BY n DESC
    """
        )
        .fetch_arrow_table()
        .to_pylist()
    )
    _write_json(f"{SUMMARY_DIR}/membership_patterns.json", patterns)

    # --- Named eval-domain retention (the headline table) ------------------ #
    named = (
        con.execute(
            """
        SELECT domain, count(*) n_urls,
               round(100.0*sum(kept_hq::INT)/count(*),1)    hq_pct,
               round(100.0*sum(kept_dclm::INT)/count(*),1)  dclm_pct,
               round(100.0*sum(kept_nemo::INT)/count(*),1)  nemo_pct,
               round(100.0*sum(kept_fwedu::INT)/count(*),1) fwedu_pct
        FROM membership
        WHERE domain IN (
            'bbc.co.uk','bbc.com','archiveofourown.org','fanfiction.net',
            'github.com','github.io','stackoverflow.com','stackexchange.com','arxiv.org',
            'wikipedia.org','reddit.com','nytimes.com','washingtonpost.com',
            'theguardian.com','medium.com','wordpress.com','blogspot.com','tumblr.com'
        )
        GROUP BY domain ORDER BY n_urls DESC
    """
        )
        .fetch_arrow_table()
        .to_pylist()
    )
    _write_json(f"{SUMMARY_DIR}/named_domain_retention.json", named)
    logger.info("[join] named-domain retention: %s", json.dumps(named, indent=2)[:2000])

    # Durable membership parquet last (sharded by duckdb COPY). If the gs:// COPY
    # is unsupported in this env, the summaries above are already safe.
    out_glob = f"{MEMBERSHIP_DIR}"
    _assert_no_cross_region(out_glob + "/x")
    try:
        con.execute(
            f"COPY membership TO '{out_glob}' (FORMAT parquet, COMPRESSION zstd, PER_THREAD_OUTPUT true, OVERWRITE_OR_IGNORE true)"
        )
        logger.info("[join] wrote membership parquet → %s", out_glob)
    except Exception as e:
        logger.error("[join] membership COPY failed (summaries already written): %s", e)
        raise

    return 0


def _write_json(path: str, obj) -> None:
    _assert_no_cross_region(path)
    with fsspec.open(path, "w") as out:
        out.write(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- #
# Phase 0c — extraction-length comparison on co-kept URLs (mangling probe)
# --------------------------------------------------------------------------- #

# Domains grouped by the Phase-0 verdict, so length ratios in HQ-LOSES domains
# can be read against HQ-WINS controls.
LENGTHCOMPARE_DOMAINS: dict[str, list[str]] = {
    "news (HQ loses)": ["bbc.co.uk", "bbc.com", "theguardian.com", "cnn.com", "nytimes.com", "reuters.com"],
    "fiction (HQ loses, coverage)": ["archiveofourown.org", "fanfiction.net"],
    "social (HQ loses, coverage)": ["reddit.com"],
    "academic (HQ loses, coverage)": ["arxiv.org"],
    "code (HQ wins, control)": ["github.com", "stackoverflow.com", "stackexchange.com"],
    "reference (HQ wins, control)": ["wikipedia.org"],
}


def run_lengthcompare(args: argparse.Namespace) -> int:
    """For URLs co-kept by HQ and each other method, compare extraction lengths.

    A systematically SMALLER hq_len on the SAME url (esp. in news, where HQ still
    loses despite high retention) is a mangling/over-stripping signal; comparable
    or larger length rules length-truncation out. Controls: code/wiki (HQ wins).
    Also selects sample URLs (co-kept by hq+dclm+nemo) for downstream text diffs.
    """
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.register_filesystem(fsspec.filesystem("gcs"))
    glob = f"{MEMBERSHIP_DIR}/*.parquet"
    _assert_no_cross_region(glob)
    con.execute(f"CREATE TEMP VIEW m AS SELECT * FROM read_parquet('{glob}')")

    all_domains = sorted({d for ds in LENGTHCOMPARE_DOMAINS.values() for d in ds})
    dom_list = ", ".join(f"'{d}'" for d in all_domains)
    # Per-domain median lengths on co-kept URLs, HQ vs each comparison method.
    rows = (
        con.execute(
            f"""
        SELECT domain,
          count(*) FILTER (WHERE kept_hq AND kept_dclm) AS n_hq_dclm,
          median(hq_len)   FILTER (WHERE kept_hq AND kept_dclm) AS hq_len_vs_dclm,
          median(dclm_len) FILTER (WHERE kept_hq AND kept_dclm) AS dclm_len,
          median(hq_len::DOUBLE / nullif(dclm_len,0)) FILTER (WHERE kept_hq AND kept_dclm) AS ratio_hq_dclm,
          count(*) FILTER (WHERE kept_hq AND kept_nemo) AS n_hq_nemo,
          median(hq_len)   FILTER (WHERE kept_hq AND kept_nemo) AS hq_len_vs_nemo,
          median(nemo_len) FILTER (WHERE kept_hq AND kept_nemo) AS nemo_len,
          median(hq_len::DOUBLE / nullif(nemo_len,0)) FILTER (WHERE kept_hq AND kept_nemo) AS ratio_hq_nemo
        FROM m WHERE domain IN ({dom_list})
        GROUP BY domain
    """
        )
        .fetch_arrow_table()
        .to_pylist()
    )
    by_domain = {r["domain"]: r for r in rows}

    print(
        f'\n{"domain":22} {"grp":34} {"n(hq&dclm)":10} {"hq_len":8} {"dclm":8} {"ratio":6} {"hq_len2":8} {"nemo":8} {"r_nemo":6}'
    )
    for grp, doms in LENGTHCOMPARE_DOMAINS.items():
        for d in doms:
            r = by_domain.get(d)
            if not r or not r["n_hq_dclm"]:
                continue

            def f(x):
                return f"{x:.0f}" if isinstance(x, (int, float)) and x is not None else "-"

            def g(x):
                return f"{x:.2f}" if isinstance(x, (int, float)) and x is not None else "-"

            print(
                f'{d:22} {grp:34} {r["n_hq_dclm"]:>10} {f(r["hq_len_vs_dclm"]):>8} {f(r["dclm_len"]):>8} '
                f'{g(r["ratio_hq_dclm"]):>6} {f(r["hq_len_vs_nemo"]):>8} {f(r["nemo_len"]):>8} {g(r["ratio_hq_nemo"]):>6}'
            )
    _write_json(f"{SUMMARY_DIR}/length_compare.json", rows)

    # Sample URLs for text diffing: news domains, co-kept by hq+dclm+nemo.
    news = ", ".join(f"'{d}'" for d in LENGTHCOMPARE_DOMAINS["news (HQ loses)"])
    sample = (
        con.execute(
            f"""
        SELECT url, domain, hq_len, dclm_len, nemo_len FROM m
        WHERE domain IN ({news}) AND kept_hq AND kept_dclm AND kept_nemo
        ORDER BY hash(url) LIMIT {args.sample}
    """
        )
        .fetch_arrow_table()
        .to_pylist()
    )
    _write_json(f"{SUMMARY_DIR}/newsdiff_sample_urls.json", sample)
    logger.info("[lengthcompare] wrote length_compare.json + %d sample news URLs", len(sample))
    return 0


# --------------------------------------------------------------------------- #
# Composition — the dilution test: knowledge-density per TOKEN, not per URL
# --------------------------------------------------------------------------- #

# Categories treated as knowledge-dense (what ARC/Jeopardy/OpenBookQA reward).
KNOWLEDGE_CATEGORIES = ("reference_wiki", "academic", "code_tech")


def run_composition(args: argparse.Namespace) -> int:
    """At a fixed token budget, what matters is char/token SHARE per category, not
    URL retention. Compute each corpus's char-weighted category composition and the
    make-up of each membership pattern (esp. HQ-only mass vs DCLM-only 'secret sauce').
    """
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.register_filesystem(fsspec.filesystem("gcs"))
    glob = f"{MEMBERSHIP_DIR}/*.parquet"
    _assert_no_cross_region(glob)
    con.execute(f"CREATE TEMP VIEW m AS SELECT *, {_category_case_sql()} AS category FROM read_parquet('{glob}')")

    # 1. Char-weighted category composition per method (share of that corpus's chars).
    comp = {}
    for meth in FILTER_METHODS:
        rows = (
            con.execute(
                f"""
            SELECT category, sum({meth}_len) AS chars, count(*) FILTER (WHERE kept_{meth}) AS docs
            FROM m WHERE kept_{meth} GROUP BY category
        """
            )
            .fetch_arrow_table()
            .to_pylist()
        )
        tot = sum(float(r["chars"] or 0) for r in rows) or 1.0
        comp[meth] = {
            r["category"]: {"char_share": round(100.0 * float(r["chars"] or 0) / tot, 2), "docs": r["docs"]}
            for r in rows
        }
        kshare = sum(comp[meth].get(c, {}).get("char_share", 0) for c in KNOWLEDGE_CATEGORIES)
        comp[meth]["_knowledge_char_share"] = round(kshare, 2)
        comp[meth]["_total_chars"] = int(tot)
    _write_json(f"{SUMMARY_DIR}/composition_by_method.json", comp)

    # 2. Membership-pattern make-up: what is the HQ-only mass, and the DCLM-only set?
    patterns = (
        con.execute(
            """
        SELECT
          CASE
            WHEN kept_hq AND NOT kept_dclm AND NOT kept_nemo AND NOT kept_fwedu THEN 'HQ_only'
            WHEN kept_dclm AND NOT kept_hq THEN 'DCLM_not_HQ'
            WHEN kept_nemo AND NOT kept_hq THEN 'NEMO_not_HQ'
            WHEN kept_hq AND kept_dclm THEN 'HQ_and_DCLM'
            ELSE 'other_pattern' END AS patt,
          category,
          count(*) AS docs,
          median(coalesce(hq_len, dclm_len, nemo_len)) AS median_len
        FROM m GROUP BY 1, 2
    """
        )
        .fetch_arrow_table()
        .to_pylist()
    )
    _write_json(f"{SUMMARY_DIR}/composition_patterns.json", patterns)

    # 3. Log the headline: knowledge char-share per method + HQ-only category mix.
    logger.info("[composition] knowledge char-share: %s", {m: comp[m]["_knowledge_char_share"] for m in FILTER_METHODS})
    return 0


# --------------------------------------------------------------------------- #
# Dev set — raw HTML docs (decoded from CommonCrawl) for extraction-spec testing
# --------------------------------------------------------------------------- #
DEVSET_DIR = f"{WORKSPACE}/devset"
DEVSET_CANDIDATES = f"{DEVSET_DIR}/candidates"  # url -> category + membership
DEVSET_DECODED = f"{DEVSET_DIR}/decoded"  # per-WARC raw HTML for candidate urls
WARC_MANIFEST_REL = "experiments/distill/dclm_400m_1x.txt"


def run_devset_select(args: argparse.Namespace) -> int:
    """Pick candidate URLs from the membership table, per content category, capped.

    Rare categories (fiction/arxiv) are taken whole; common ones are down-capped.
    The decode pass keeps raw HTML only for these candidates (bounds output)."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.register_filesystem(fsspec.filesystem("gcs"))
    glob = f"{MEMBERSHIP_DIR}/*.parquet"
    _assert_no_cross_region(glob)
    # Per-category cap via row_number; rare categories fall under the cap → all kept.
    con.execute(
        f"""
        CREATE TEMP TABLE cand AS
        WITH tagged AS (
            SELECT url, domain, kept_hq, kept_dclm, kept_nemo, kept_fwedu,
                   {_category_case_sql()} AS category
            FROM read_parquet('{glob}')
        )
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY category ORDER BY hash(url || 'devset')) AS rn
            FROM tagged
        ) WHERE rn <= {args.cap_per_category}
    """
    )
    n = con.execute("SELECT count(*) FROM cand").fetchone()[0]
    bycat = con.execute("SELECT category, count(*) FROM cand GROUP BY category ORDER BY 2 DESC").fetchall()
    logger.info("[devset-select] %d candidate urls; by category: %s", n, bycat)
    out = DEVSET_CANDIDATES
    _assert_no_cross_region(out + "/x")
    con.execute(
        f"COPY (SELECT url, domain, category, kept_hq, kept_dclm, kept_nemo, kept_fwedu FROM cand) "
        f"TO '{out}' (FORMAT parquet, COMPRESSION zstd, PER_THREAD_OUTPUT true, OVERWRITE_OR_IGNORE true)"
    )
    logger.info("[devset-select] wrote candidates → %s", out)
    return 0


def _even_spaced_warcs(manifest_path: str, n: int) -> list[str]:
    """Evenly-spaced sample across the TIME-SORTED manifest (not the head) so all
    CommonCrawl snapshots are represented proportionally."""
    with open(manifest_path) as f:
        warcs = [ln.strip() for ln in f if ln.strip()]
    total = len(warcs)
    if n >= total:
        return warcs
    idx = sorted({round(i * total / n) for i in range(n)})
    return [warcs[min(j, total - 1)] for j in idx]


def run_devset_decode(args: argparse.Namespace) -> int:
    """Decode an evenly-spaced sample of WARCs from CommonCrawl and keep raw HTML
    only for candidate URLs. One durable parquet per WARC (skip-existing)."""
    from experiments.baseline_collection.decode_warcs_clean import _decode_one_warc, _warc_path_hash

    # Load candidate url -> category/membership (in-region).
    cand_glob = f"{DEVSET_CANDIDATES}/*.parquet"
    _assert_no_cross_region(cand_glob)
    cand_tbl = pq.ParquetDataset(cand_glob.replace("gs://", ""), filesystem=fsspec.filesystem("gcs")).read(
        columns=["url"]
    )
    candidates = frozenset(cand_tbl.column("url").to_pylist())
    logger.info("[devset-decode] %d candidate urls loaded", len(candidates))

    manifest = str(Path(__file__).resolve().parents[2] / WARC_MANIFEST_REL)
    warcs = _even_spaced_warcs(manifest, args.num_warcs)
    logger.info("[devset-decode] decoding %d evenly-spaced WARCs", len(warcs))
    fs = fsspec.filesystem("gcs")

    def _do(warc_path: str) -> int:
        wh = _warc_path_hash(warc_path)
        out_path = f"{DEVSET_DECODED}/{wh}.parquet"
        exists = out_path[len("gs://") :] if out_path.startswith("gs://") else out_path
        if not args.overwrite and fs.exists(exists):
            return -1
        recs = _decode_one_warc(warc_path)
        keep = [
            {
                "url": r["url"],
                "warc_hash": r.get("warc_hash"),
                "snapshot": r.get("snapshot"),
                "html": (r.get("html") or "")[: args.html_cap],
                "text_body": (r.get("text_body") or "")[: args.html_cap],
            }
            for r in recs
            if r.get("url") in candidates and (r.get("html") or "")
        ]
        _assert_no_cross_region(out_path)
        with fs.open(out_path, "wb") as fh:
            pq.write_table(pa.Table.from_pylist(keep), fh, compression="zstd")
        return len(keep)

    results = _parallel_map(_do, warcs, "devset-decode", max_workers=args.threads)
    kept = sum(r for r in results if r and r >= 0)
    logger.info("[devset-decode] kept %d candidate HTML docs from %d WARCs", kept, len(warcs))
    return 0


# --------------------------------------------------------------------------- #
# Comprehensive domain differential — where HQ over/under-keeps vs the others
# --------------------------------------------------------------------------- #

# Content categories by domain substring (priority-ordered; first match wins).
# Heuristic, but captures the long tail that a top-N domain list misses (top-5000
# domains are only ~39% of the 32.5M union URLs).
CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "fiction",
        [
            "fanfiction",
            "archiveofourown",
            "wattpad",
            "fictionpress",
            "literotica",
            "deviantart",
            "quotev",
            "royalroad",
            "booksie",
            "wuxiaworld",
            "novelupdates",
            "webnovel",
            "fictionhub",
            "adult-fanfiction",
            "mediaminer",
            "asianfanfics",
            "storiesonline",
        ],
    ),
    (
        "academic",
        [
            "arxiv",
            "biorxiv",
            "ssrn",
            "researchgate",
            "academia.edu",
            "jstor",
            "sciencedirect",
            "springer",
            "ncbi.nlm.nih",
            "plos",
            "pubmed",
            "semanticscholar",
            "citeseerx",
            "iopscience",
            "ieee",
            "acm.org",
            "aps.org",
            "mdpi",
            "tandfonline",
            "wiley",
            "oup.com",
            "sagepub",
        ],
    ),
    (
        "news",
        [
            "bbc.",
            "cnn.",
            "nytimes",
            "theguardian",
            "reuters",
            "washingtonpost",
            "forbes",
            "bloomberg",
            "huffpost",
            "huffingtonpost",
            "npr.org",
            "apnews",
            "aljazeera",
            "telegraph",
            "dailymail",
            "foxnews",
            "nbcnews",
            "cbsnews",
            "usatoday",
            "wsj.",
            "economist",
            "theatlantic",
            "politico",
            "news",
            "tribune",
            "gazette",
            "herald",
            "chron.",
            "latimes",
        ],
    ),
    (
        "forum_social",
        [
            "reddit",
            "forum",
            "stackexchange",
            "quora",
            "4chan",
            "phpbb",
            "disqus",
            "vbulletin",
            "proboards",
            "tapatalk",
            "groups.google",
            "boards.",
            "community.",
        ],
    ),
    (
        "code_tech",
        [
            "github",
            "gitlab",
            "bitbucket",
            "stackoverflow",
            "sourceforge",
            "npmjs",
            "pypi",
            "readthedocs",
            "apache.org",
            "w3.org",
            "developer.",
            "docs.",
            "mozilla",
            "kernel.org",
            "gnu.org",
            "freedesktop",
            "codeproject",
            "geeksforgeeks",
            "mathworks",
            "msdn",
        ],
    ),
    (
        "reference_wiki",
        [
            "wikipedia",
            "wikia",
            "fandom",
            "wiktionary",
            "wikihow",
            "wikimedia",
            "wikinews",
            "wikibooks",
            "britannica",
            "wikidot",
            "wikispaces",
        ],
    ),
    ("qa_help", ["answers.", "ask.", "support.", "help.", "faq", "howto"]),
    (
        "ecommerce",
        [
            "amazon.",
            "ebay.",
            "etsy",
            "aliexpress",
            "alibaba",
            "shopify",
            "walmart",
            "bestbuy",
            "target.com",
            "craigslist",
        ],
    ),
    (
        "blog",
        [
            "blogspot",
            "wordpress",
            "tumblr",
            "typepad",
            "medium.com",
            "livejournal",
            "blogs.",
            "blog.",
            "ghost.io",
            "substack",
            "wixsite",
            "weebly",
        ],
    ),
]


def _category_case_sql(col: str = "domain") -> str:
    """Build a priority-ordered SQL CASE mapping domain → content category."""
    whens = []
    for cat, pats in CATEGORY_PATTERNS:
        ors = " OR ".join(f"{col} LIKE '%{p}%'" for p in pats)
        whens.append(f"WHEN {ors} THEN '{cat}'")
    return "CASE " + " ".join(whens) + " ELSE 'other' END"


def run_domaindiff(args: argparse.Namespace) -> int:
    """Comprehensive retention differential: per-category (all URLs, captures the
    long tail), per-TLD, and a large top-domain table. Where does HQ over/under-keep?"""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads TO {args.threads}")
    con.register_filesystem(fsspec.filesystem("gcs"))
    glob = f"{MEMBERSHIP_DIR}/*.parquet"
    _assert_no_cross_region(glob)
    con.execute(f"CREATE TEMP VIEW m AS SELECT *, {_category_case_sql()} AS category FROM read_parquet('{glob}')")

    def retention_by(expr: str, having: str = "count(*) >= 200", limit: str = "") -> list[dict]:
        return (
            con.execute(
                f"""
            SELECT {expr} AS grp, count(*) AS n_urls,
                round(100.0*sum(kept_hq::INT)/count(*),1)    AS hq_pct,
                round(100.0*sum(kept_dclm::INT)/count(*),1)  AS dclm_pct,
                round(100.0*sum(kept_nemo::INT)/count(*),1)  AS nemo_pct,
                round(100.0*sum(kept_fwedu::INT)/count(*),1) AS fwedu_pct
            FROM m GROUP BY {expr} HAVING {having} ORDER BY n_urls DESC {limit}
        """
            )
            .fetch_arrow_table()
            .to_pylist()
        )

    # 1. Category retention over ALL 32.5M urls (the comprehensive view).
    cats = retention_by("category", having="count(*) >= 1")
    _write_json(f"{SUMMARY_DIR}/category_retention.json", cats)
    logger.info("[domaindiff] category retention: %s", json.dumps(cats, default=str)[:1500])

    # 2. TLD retention (orthogonal cut).
    tld_expr = "regexp_extract(domain, '([^.]+)$')"
    tlds = retention_by(tld_expr, having="count(*) >= 1000")
    _write_json(f"{SUMMARY_DIR}/tld_retention.json", tlds)

    # 3. Large top-domain table (30k) for head/mid detail.
    top = retention_by("domain", having="count(*) >= 50", limit=f"LIMIT {args.top_domains}")
    _write_json(f"{SUMMARY_DIR}/domain_retention_full.json", top)
    logger.info("[domaindiff] wrote %d categories, %d tlds, %d domains", len(cats), len(tlds), len(top))
    return 0


# --------------------------------------------------------------------------- #
# Phase 0c — fetch same-URL text per method (for rewrite/mangling inspection)
# --------------------------------------------------------------------------- #


def _scan_shard_for_urls(method: str, shard_path: str, targets: frozenset[str], cap: int) -> list[dict]:
    src = SOURCES[method]
    it = _iter_parquet(shard_path) if src["format"] == "parquet" else _iter_jsonl(shard_path)
    out = []
    for rec in it:
        url = rec.get("url")
        if url in targets:
            text = rec.get("text") or ""
            out.append({"url": url, "method": method, "text": text[:cap], "full_len": len(text)})
    return out


def run_fetchtext(args: argparse.Namespace) -> int:
    """Fetch each method's extracted text for a target URL list (region-matched).

    Run per method in its own region (hq->central1, dclm/nemo->central2). Output is
    small (few hundred URLs x cap chars), downloaded locally for the diff.
    """
    _assert_no_cross_region(args.url_list)
    with fsspec.open(args.url_list, "r") as f:
        payload = json.load(f)
    targets = frozenset(row["url"] if isinstance(row, dict) else row for row in payload)
    logger.info("[fetchtext] %d target urls", len(targets))

    for method in args.methods.split(","):
        src = SOURCES[method]
        shards = _gcs_ls_glob(f"{src['root']}/{src['glob']}")
        logger.info("[fetchtext] %s: scanning %d shards", method, len(shards))
        results = _parallel_map(
            lambda p, m=method: _scan_shard_for_urls(m, p, targets, args.cap),
            shards,
            f"fetchtext {method}",
            max_workers=args.threads,
        )
        recs: dict[str, dict] = {}
        for rs in results:
            for r in rs:
                recs.setdefault(r["url"], r)
        out_path = f"{args.workspace}/newsdiff/{method}_text.jsonl.gz"
        _assert_no_cross_region(out_path)
        import gzip
        import io

        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for r in recs.values():
                gz.write(json.dumps(r, ensure_ascii=False).encode() + b"\n")
        with fsspec.open(out_path, "wb") as out:
            out.write(buf.getvalue())
        logger.info("[fetchtext] %s: matched %d / %d urls → %s", method, len(recs), len(targets), out_path)
    return 0


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="map each pipeline's shards → per-method parquet")
    ex.add_argument("--method", default="all", choices=[*SOURCES.keys(), "all"])
    ex.add_argument("--threads", type=int, default=64)
    ex.add_argument("--limit-shards", type=int, default=0, help="debug: cap shards per method")
    ex.add_argument("--overwrite", action="store_true")
    ex.add_argument(
        "--workspace",
        default=WORKSPACE,
        help="output workspace (per_method written here). Must be in-region with the source: "
        "use gs://marin-us-central1/scratch/provenance_10k for hq.",
    )

    jn = sub.add_parser("join", help="duckdb full-outer-join → membership table + summaries")
    jn.add_argument("--threads", type=int, default=16)

    lc = sub.add_parser("lengthcompare", help="Phase 0c: extraction-length comparison on co-kept URLs")
    lc.add_argument("--threads", type=int, default=16)
    lc.add_argument("--sample", type=int, default=400, help="sample news URLs (co-kept hq+dclm+nemo) for text diff")

    dd = sub.add_parser("domaindiff", help="comprehensive retention differential: category + TLD + top-domain")
    dd.add_argument("--threads", type=int, default=16)
    dd.add_argument("--top-domains", type=int, default=30000)

    ft = sub.add_parser("fetchtext", help="Phase 0c: fetch per-method text for a target URL list (region-matched)")
    ft.add_argument("--methods", required=True, help="comma list, e.g. hq or dclm,nemo")
    ft.add_argument("--url-list", required=True, help="gs path to newsdiff_sample_urls.json (region-matched)")
    ft.add_argument("--workspace", default=WORKSPACE, help="output workspace (region-matched to sources)")
    ft.add_argument("--cap", type=int, default=12000, help="max chars of text kept per doc")
    ft.add_argument("--threads", type=int, default=64)

    cp = sub.add_parser("composition", help="dilution test: knowledge char-share per token + pattern make-up")
    cp.add_argument("--threads", type=int, default=16)

    ds = sub.add_parser("devset-select", help="pick candidate URLs per category (capped) for the HTML dev set")
    ds.add_argument("--threads", type=int, default=16)
    ds.add_argument("--cap-per-category", type=int, default=80000)

    dc = sub.add_parser("devset-decode", help="decode evenly-spaced WARCs → raw HTML for candidate URLs")
    dc.add_argument("--num-warcs", type=int, default=4000)
    dc.add_argument("--html-cap", type=int, default=200000)
    dc.add_argument("--threads", type=int, default=48)
    dc.add_argument("--overwrite", action="store_true")

    args = p.parse_args()
    return {
        "extract": run_extract,
        "join": run_join,
        "lengthcompare": run_lengthcompare,
        "domaindiff": run_domaindiff,
        "fetchtext": run_fetchtext,
        "composition": run_composition,
        "devset-select": run_devset_select,
        "devset-decode": run_devset_decode,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
