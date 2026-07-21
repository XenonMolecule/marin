# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Stratified sample of the 0-vote RESIDUAL for eyeballing + an independent judge pass.

The residual = docs that appear (eval-relevant) in the raw pool (resiliparse) but which hq
dropped AND no independent quality filter (dclm/nemo/fwedu) kept — verifier_score=0. Mostly
junk by expectation; the question is whether a usable practical-knowledge seam hides inside.

This joins the resiliparse hits (which carry the doc snippet + the eval subjects the doc
matched — i.e. the LINK to the eval set) with the membership table, keeps the 0-vote
residual, buckets each doc into a coarse category by domain, and takes a random sample per
category so we can compare needle-rates across registers. Also grabs a small 3/3 sample
(all filters kept, hq alone dropped) as a high-confidence contrast.

Runs in-region (us-central2, where resiliparse hits + membership live). Writes a small
sample parquet to pull locally for the judge pass.
"""

from __future__ import annotations

import logging
import sys

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("sample_residual")

WORKSPACE = "gs://marin-us-central2/scratch/provenance_10k"
MEMBERSHIP = f"{WORKSPACE}/membership/*.parquet"
RES_HITS = f"{WORKSPACE}/keyword_hits/resiliparse/*.parquet"
OUT = f"{WORKSPACE}/residual_sample.parquet"
PER_CATEGORY = 200

# Coarse domain→category buckets (regex on registered domain). Practical-expertise registers
# the academic-leaning filters tend to drop, plus reference, plus the blog platforms and rest.
CATEGORY_CASE = r"""
  CASE
    WHEN regexp_matches(domain,'recipe|food|cook|kitchen|baking|cuisine|chow|wine|brew|eat|grocer') THEN 'cooking'
    WHEN regexp_matches(domain,'fitness|health|diet|nutrition|calorie|weight|myfitness|fatsecret|sparkpeople|fitbit|nih|clinic|medic') THEN 'health_fitness'
    WHEN regexp_matches(domain,'soccer|sport|golf|piano|guitar|music|garden|aquarium|marine|fish|hunt|craft|hobby|arts|photo') THEN 'hobby_sport'
    WHEN regexp_matches(domain,'wiki|library|slpl|ipl|nasa|answers|princeton|\.edu|\.gov|reference') THEN 'reference'
    WHEN regexp_matches(domain,'blogspot|wordpress|typepad|tumblr|livejournal|blog') THEN 'blog_platform'
    ELSE 'other'
  END
"""


def main() -> int:
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false; SET enable_progress_bar=false;")
    logger.info("[sample] loading membership + resiliparse hits")
    con.execute(
        f"CREATE TEMP TABLE mem AS SELECT url, kept_hq, kept_dclm, kept_nemo, kept_fwedu FROM read_parquet('{MEMBERSHIP}')"
    )
    # A url can recur across resiliparse shards; collapse to one row keeping the richest match.
    con.execute(
        f"""CREATE TEMP TABLE res AS
        SELECT url, any_value(domain) AS domain, arg_max(subjects, n_examples) AS subjects,
               arg_max(tasks, n_examples) AS tasks, max(best_frac) AS best_frac,
               arg_max(snippet, n_examples) AS snippet
        FROM read_parquet('{RES_HITS}') GROUP BY url"""
    )
    con.execute(
        f"""CREATE TEMP TABLE tagged AS
        SELECT r.url, r.domain, r.subjects, r.tasks, r.best_frac, r.snippet,
               (COALESCE(m.kept_dclm,false)::INT + COALESCE(m.kept_nemo,false)::INT + COALESCE(m.kept_fwedu,false)::INT) AS verifier_score,
               {CATEGORY_CASE} AS category
        FROM res r LEFT JOIN mem m USING (url)
        WHERE NOT COALESCE(m.kept_hq,false)"""  # hq dropped it
    )
    # Sample: PER_CATEGORY from the 0-vote residual per bucket, + a 3/3 high-confidence contrast.
    con.execute(
        f"""CREATE TEMP TABLE sample AS
        SELECT * EXCLUDE (rn) FROM (
          SELECT *, row_number() OVER (PARTITION BY (verifier_score>=3), category ORDER BY random()) AS rn
          FROM tagged WHERE verifier_score=0 OR verifier_score>=3
        ) WHERE rn <= {PER_CATEGORY}"""
    )
    n = con.execute("SELECT count(*) FROM sample").fetchone()[0]
    breakdown = con.execute(
        "SELECT (verifier_score>=3) AS is3, category, count(*) FROM sample GROUP BY 1,2 ORDER BY 1,2"
    ).fetchall()
    logger.info("[sample] %d docs; breakdown (is_3of3, category, n): %s", n, breakdown)
    con.execute(f"COPY sample TO '{OUT}' (FORMAT parquet)")
    logger.info("[sample] wrote %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
