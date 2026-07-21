# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Phase C: find the pretraining docs (and domains) carrying content hq loses evals on.

Given the curated keyword vocab from hq-worse eval examples (Phase A/B), scan each
pipeline's corpus IN-REGION for docs containing those keywords, cross with the
membership table, and rank the domains/website-types that dclm/nemo keep but hq drops.
That domain list feeds the real dev set; the matched docs form a separate held-out set.

Two failure modes the aggregate distinguishes, per (url, domain):
  COVERAGE gap  dclm/nemo keep an eval-relevant doc but kept_hq is FALSE (hq never had it).
  QUALITY  gap  hq keeps the url (kept_hq TRUE) but hq's extraction does NOT carry the
                content (hq scan didn't match) while dclm's/nemo's does — i.e. hq
                extracted the same page worse. (Consistent with the ablation finding.)

STRICT IN-REGION (no bulk egress): each corpus is scanned in its own region and writes
hits to that region's bucket. Only small keyword lists + the aggregation move regions.

Stages:
  estimate-df  sample-scan one corpus → per-keyword doc frequency → drop generic keywords
               (the DF threshold) and compute IDF weights → keywords_final.json.
  scan         full in-region scan of one corpus → per-shard parquet of matched docs
               {url, domain, matched_keywords, score, snippet}. IDF-weighted.
  aggregate    duckdb join all corpora's hits + the membership table → ranked domains
               (coverage vs quality gap) + the held-out doc set.

Launch (per corpus, in its region — hq=us-central1, dclm/nemo/fwedu=us-central2):
    iris --cluster marin job run --region us-central2 --enable-extra-resources \\
        --cpu 32 --memory 96GB --extra cpu \\
        -- python experiments/baseline_collection/keyword_corpus_scan.py scan --method dclm
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.baseline_collection.provenance_audit_10k import (
    SOURCES,
    WORKSPACE,
    _assert_no_cross_region,
    _gcs_ls_glob,
    _iter_jsonl,
    _iter_parquet,
    _parallel_map,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("keyword_corpus_scan")

# Keyword lists live in us-central2 (WORKSPACE region) so the dclm/nemo/fwedu scans read
# them in-region; the hq scan (us-central1) gets a copied local list (see orchestration).
KEYWORDS_CURATED = f"{WORKSPACE}/keywords_curated.json"
KEYWORDS_FINAL = f"{WORKSPACE}/keywords_final.json"
DF_THRESHOLD = 0.02  # drop keywords matching >2% of sampled docs (too generic)
MIN_HITS = 3  # a doc must match >=3 distinct eval keywords
MIN_TOKENS = 50  # ...and be a real doc (skip titles/snippets)
MIN_DENSITY = 0.03  # ...and be genuinely ABOUT eval topics (IDF mass per token). Calibrated 2026-07-07.
SNIPPET_LEN = 400
AGG_OUT = f"{WORKSPACE}/keyword_agg"  # ranked domains + held-out set


def _bucket_of(method: str) -> str:
    return SOURCES[method]["root"].split("/")[2]  # marin-us-central1 | marin-us-central2


def _hits_dir(method: str) -> str:
    """Region-local hits dir for a method (keeps hq hits in us-central1, others in central2)."""
    return f"gs://{_bucket_of(method)}/scratch/provenance_10k/keyword_hits/{method}"


def _read_json_gcs(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def _write_json_gcs(path: str, obj: dict) -> None:
    _assert_no_cross_region(path)
    with fsspec.open(path, "w") as f:
        json.dump(obj, f)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _build_automaton(keywords: list[str]) -> dict:
    """Stdlib token matcher (no external deps → available in every distributed worker).

    Single-word keywords go in a set (O(1) membership); multi-word phrases are indexed by
    their first token so a doc only checks phrases whose first word actually appears.
    Returns lowercased keyword strings as hit keys (matching the IDF dict). Tokenizing both
    sides makes matching word-boundary-correct by construction ('sparkling' ≠ 'park').
    """
    singles: set[str] = set()
    phrases: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for kw in keywords:
        toks = tuple(_TOKEN_RE.findall(kw.lower()))
        if not toks:
            continue
        if len(toks) == 1:
            singles.add(toks[0])
        else:
            phrases.setdefault(toks[0], []).append((kw.lower(), toks))
    return {"singles": singles, "phrases": phrases}


def _match_tokens(toks: list[str], matcher: dict) -> set[str]:
    """Distinct keyword hits given pre-tokenized doc tokens (word-boundary-correct)."""
    hits = matcher["singles"] & set(toks)
    phrases = matcher["phrases"]
    if phrases:
        hits = set(hits)
        for i, t in enumerate(toks):
            for kw, ptoks in phrases.get(t, ()):  # only phrases whose first word appears
                if tuple(toks[i : i + len(ptoks)]) == ptoks:
                    hits.add(kw)
    return hits


def _match(text_lower: str, matcher: dict) -> set[str]:
    return _match_tokens(_TOKEN_RE.findall(text_lower), matcher)


def _score_doc(text: str, matcher: dict, idf: dict[str, float]) -> tuple[set[str], int, float]:
    """Return (hits, n_tokens, density). density = IDF-weighted eval-keyword mass per token,
    so a doc that is genuinely ABOUT eval topics scores high while a merely-long doc that
    incidentally contains many keywords does not."""
    toks = _TOKEN_RE.findall(text.lower())
    hits = _match_tokens(toks, matcher)
    n = len(toks)
    density = sum(idf.get(h, 0.0) for h in hits) / n if n else 0.0
    return hits, n, density


def _iter_docs(method: str, shard_path: str):
    src = SOURCES[method]
    it = _iter_parquet(shard_path) if src["format"] == "parquet" else _iter_jsonl(shard_path)
    for rec in it:
        url, text = rec.get("url"), rec.get("text")
        if url and text:
            yield url, text


# --------------------------------------------------------------------------- #
# Stage 1: estimate document frequency on a sample → drop generic keywords + IDF
# --------------------------------------------------------------------------- #
def run_estimate_df(args: argparse.Namespace) -> int:
    kw = _read_json_gcs(args.keywords or KEYWORDS_CURATED)["keywords"]
    automaton = _build_automaton([k.lower() for k in kw])
    shards = _gcs_ls_glob(f"{SOURCES[args.method]['root']}/{SOURCES[args.method]['glob']}")[: args.sample_shards]
    logger.info("[estimate-df] %s: %d sample shards, %d keywords", args.method, len(shards), len(kw))

    def _shard_df(shard):
        df: Counter = Counter()
        n = 0
        for _url, text in _iter_docs(args.method, shard):
            n += 1
            for k in _match(text.lower(), automaton):
                df[k] += 1
        return df, n

    total_df: Counter = Counter()
    n_docs = 0
    for df, n in _parallel_map(_shard_df, shards, "estimate-df", max_workers=args.threads):
        total_df.update(df)
        n_docs += n
    keep = [k for k in kw if total_df.get(k.lower(), 0) / max(n_docs, 1) < DF_THRESHOLD]
    idf = {k.lower(): math.log((n_docs + 1) / (total_df.get(k.lower(), 0) + 1)) for k in keep}
    top_drop = [(k, total_df.get(k.lower(), 0)) for k in sorted(kw, key=lambda k: -total_df.get(k.lower(), 0))[:12]]
    logger.info("[estimate-df] n_docs=%d; keep %d / %d; top generic dropped: %s", n_docs, len(keep), len(kw), top_drop)
    _write_json_gcs(KEYWORDS_FINAL, {"n_docs_sampled": n_docs, "keywords": keep, "idf": idf})
    logger.info("[estimate-df] wrote %s", KEYWORDS_FINAL)
    return 0


# The full scan runs as a distributed Zephyr pipeline (keyword_scan_zephyr.py), not here.


# --------------------------------------------------------------------------- #
# Stage 3: aggregate — join hits + membership → ranked domains (coverage vs quality)
# --------------------------------------------------------------------------- #
def run_aggregate(args: argparse.Namespace) -> int:
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))  # gcsfs-backed: read gs:// via gcloud creds
    con.execute("SET enable_progress_bar=false; SET preserve_insertion_order=false;")
    membership = f"{WORKSPACE}/membership/*.parquet"
    # Per-method hit globs (hq lives in us-central1; a small cross-region read here only).
    hit_globs = {m: f"{_hits_dir(m)}/*.parquet" for m in args.methods}
    # Union all hits, tagging the scanning method; a url can be hit by several corpora.
    union = " UNION ALL ".join(
        f"SELECT url, domain, subjects, tasks, best_frac, n_examples, '{m}' AS hit_method FROM read_parquet('{g}')"
        for m, g in hit_globs.items()
    )
    con.execute(f"CREATE TEMP TABLE hits AS {union}")
    # FineWeb-CC's kept-URL set (recovered re-extraction) → the 4th independent verifier.
    fwcc = f"{WORKSPACE}/fwcc_membership/kept_fwcc.parquet"
    con.execute(f"CREATE TEMP TABLE fwcc AS SELECT DISTINCT url FROM read_parquet('{fwcc}')")
    con.execute(
        f"""CREATE TEMP TABLE mem AS
        SELECT m.url, m.kept_hq, m.kept_dclm, m.kept_nemo, m.kept_fwedu, (f.url IS NOT NULL) AS kept_fwcc
        FROM read_parquet('{membership}') m LEFT JOIN fwcc f ON m.url = f.url"""
    )
    # Per-url: which corpora's scan matched it (content present) + membership (kept).
    con.execute(
        """CREATE TEMP TABLE perurl AS
        SELECT h.url, any_value(h.domain) AS domain,
               max(CASE WHEN hit_method='hq' THEN 1 ELSE 0 END)   AS hit_hq,
               max(CASE WHEN hit_method='dclm' THEN 1 ELSE 0 END) AS hit_dclm,
               max(CASE WHEN hit_method='nemo' THEN 1 ELSE 0 END) AS hit_nemo,
               max(CASE WHEN hit_method='fwedu' THEN 1 ELSE 0 END) AS hit_fwedu,
               max(CASE WHEN hit_method='resiliparse' THEN 1 ELSE 0 END) AS hit_resiliparse,  -- raw unfiltered pool = the universe
               max(h.best_frac) AS best_frac, max(h.n_examples) AS n_examples,
               arg_max(h.subjects, h.n_examples) AS subjects, arg_max(h.tasks, h.n_examples) AS tasks
        FROM hits h GROUP BY h.url"""
    )
    con.execute(
        """CREATE TEMP TABLE joined AS
        SELECT p.*, COALESCE(m.kept_hq,false) AS kept_hq, COALESCE(m.kept_dclm,false) AS kept_dclm,
               COALESCE(m.kept_nemo,false) AS kept_nemo, COALESCE(m.kept_fwedu,false) AS kept_fwedu,
               COALESCE(m.kept_fwcc,false) AS kept_fwcc,
               (COALESCE(m.kept_dclm,false)::INT + COALESCE(m.kept_nemo,false)::INT
                + COALESCE(m.kept_fwedu,false)::INT + COALESCE(m.kept_fwcc,false)::INT) AS verifier_score,  -- # independent quality filters (not hq) that kept the page (now /4 incl fwcc)
               -- UNIVERSE = eval-content exists in SOME extraction (resiliparse=raw pool dominates).
               ((p.hit_resiliparse=1 OR p.hit_dclm=1 OR p.hit_nemo=1 OR p.hit_fwedu=1)
                AND NOT COALESCE(m.kept_hq,false)) AS coverage_gap,
               ((p.hit_resiliparse=1 OR p.hit_dclm=1 OR p.hit_nemo=1 OR p.hit_fwedu=1)
                AND COALESCE(m.kept_hq,false) AND p.hit_hq=0) AS quality_gap
        FROM perurl p LEFT JOIN mem m USING (url)"""
    )
    # Ranked domains: how much eval-relevant content each domain carries that hq misses.
    domains = con.execute(
        """SELECT domain,
               count(*) AS n_matched_docs,
               sum(CASE WHEN coverage_gap THEN 1 ELSE 0 END) AS n_coverage_gap,
               sum(CASE WHEN quality_gap  THEN 1 ELSE 0 END) AS n_quality_gap,
               sum(CASE WHEN hit_hq=1 THEN 1 ELSE 0 END)     AS n_hq_has,
               round(avg(CASE WHEN coverage_gap OR quality_gap THEN verifier_score END),2) AS avg_verifiers,
               sum(CASE WHEN (coverage_gap OR quality_gap) AND verifier_score>=3 THEN 1 ELSE 0 END) AS n_all3_agree,
               sum(CASE WHEN (coverage_gap OR quality_gap) AND verifier_score=0 THEN 1 ELSE 0 END) AS n_frontier,
               round(sum(CASE WHEN coverage_gap OR quality_gap THEN best_frac ELSE 0 END),1) AS missing_score
        FROM joined GROUP BY domain
        HAVING (n_coverage_gap + n_quality_gap) > 0
        ORDER BY (n_coverage_gap + n_quality_gap) DESC LIMIT 3000"""
    ).fetchall()
    cols = [
        "domain",
        "n_matched_docs",
        "n_coverage_gap",
        "n_quality_gap",
        "n_hq_has",
        "avg_verifiers",
        "n_all3_agree",
        "n_frontier",
        "missing_score",
    ]
    _assert_no_cross_region(f"{AGG_OUT}/domains.parquet")
    with fsspec.filesystem("gcs").open(f"{AGG_OUT}/domains.parquet", "wb") as fh:
        pq.write_table(pa.table({c: [r[i] for r in domains] for i, c in enumerate(cols)}), fh, compression="zstd")
    # Held-out set: the missing eval-relevant docs, annotated (separate; not for training).
    con.execute(
        f"""COPY (SELECT url, domain, subjects, tasks, n_examples, best_frac, verifier_score,
                  coverage_gap, quality_gap, kept_hq, kept_dclm, kept_nemo, kept_fwedu
           FROM joined WHERE coverage_gap OR quality_gap ORDER BY verifier_score DESC, n_examples DESC)
        TO '{AGG_OUT}/heldout_missing_docs.parquet' (FORMAT parquet)"""
    )
    totals = con.execute(
        "SELECT count(*), sum(coverage_gap::INT), sum(quality_gap::INT) FROM joined WHERE coverage_gap OR quality_gap"
    ).fetchone()
    logger.info(
        "[aggregate] %d domains ranked; held-out missing docs=%d (coverage=%d, quality=%d) → %s",
        len(domains),
        totals[0],
        totals[1],
        totals[2],
        AGG_OUT,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("estimate-df", help="sample-scan one corpus → per-keyword DF → drop generics")
    e.add_argument("--method", required=True, choices=list(SOURCES))
    e.add_argument("--sample-shards", type=int, default=8)
    e.add_argument("--threads", type=int, default=32)
    e.add_argument("--keywords", default=None, help="override curated keywords path")
    e.set_defaults(func=run_estimate_df)
    a = sub.add_parser("aggregate", help="join hits + membership → ranked domains + held-out set")
    a.add_argument("--methods", nargs="+", default=["hq", "dclm", "nemo", "fwedu"])
    a.set_defaults(func=run_aggregate)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
