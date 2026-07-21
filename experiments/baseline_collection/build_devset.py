# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build the spec-iteration dev set: candidate selection across 14 category-subsets.

Selects ~cap_per_category candidate URLs per subset from the membership table, tagging each
with verifier_score (0-4 = kept_dclm+nemo+fwedu+fwcc, all independent of hq) so downstream
stratification (clean 4/4 positives vs 1-2/4 boundary vs 0/4 junk) is available. Domain-based
subsets come from the membership table; the random-timespan subset is drawn from the CC-timeline
sample; a junk subset is drawn from 0-vote docs as negative anchors.

Selection only (raw HTML + per-pipeline text are attached in a later decode step). The PREVIEW
GATE renders these candidates for the user to confirm before any judge spend. In-region us-central2.

Subsets (see plan): 6 eval-confirmed gaps + 4 hypothesized gaps + 4 controls.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("build_devset")

WORKSPACE = "gs://marin-us-central2/scratch/provenance_10k"
MEMBERSHIP = f"{WORKSPACE}/membership/*.parquet"
FWCC = f"{WORKSPACE}/fwcc_membership/kept_fwcc.parquet"
CANDIDATES = f"{WORKSPACE}/devset/candidates"
RESILIPARSE_HITS = f"{WORKSPACE}/keyword_hits/resiliparse/*.parquet"
PREVIEW = f"{WORKSPACE}/devset/match_preview_sample.parquet"
HITS_80 = f"{WORKSPACE}/keyword_hits_frac80/resiliparse/*.parquet"  # the tighter 0.8-threshold matches
HARD_DIR = f"{WORKSPACE}/devset/hard"
HARD_SAMPLE = f"{HARD_DIR}/sample.parquet"  # the ~1k hand-label subset (tool + extraction fetch)
HARD_SAMPLE_FULL = f"{HARD_DIR}/sample_full.parquet"  # oversampled pool retained for LLM soft-labeling
HARD_URLS = f"{HARD_DIR}/urls.parquet"
PINNED_URLS = f"{HARD_DIR}/pinned_urls.parquet"  # already-labeled urls: force-kept in the label set forever
HARD_EXTRACTIONS = f"{HARD_DIR}/extractions"  # /{method}/*.parquet  (matched pool)
TIMESPAN = "gs://marin-us-east5/documents/internet_timespan_sample/v1"
# The random pool lives in us-east5 (timespan's region) to keep all reads/writes in-region; the
# tool build pulls these two small parquets to the laptop (few MB) — no cross-region GCS copy.
RANDOM_WORKSPACE = "gs://marin-us-east5/scratch/provenance_10k_devset"
RANDOM_SAMPLE = f"{RANDOM_WORKSPACE}/random_sample.parquet"
RANDOM_EXT = f"{RANDOM_WORKSPACE}/random_resiliparse.parquet"

# Priority-ordered dev-set subsets → domain substrings. A url is assigned to the FIRST match,
# so put the specific registers (math/legal/science) BEFORE the broad ones (howto/qa). The four
# non-domain subsets (random_timespan, junk) are handled separately.
DEVSET_CATEGORIES: list[tuple[str, list[str]]] = [
    # --- eval-confirmed gaps ---
    (
        "fiction",
        [
            "fanfiction",
            "archiveofourown",
            "wattpad",
            "fictionpress",
            "literotica",
            "royalroad",
            "quotev",
            "booksie",
            "wuxiaworld",
            "novelupdates",
            "webnovel",
            "asianfanfics",
            "storiesonline",
        ],
    ),
    ("arxiv", ["arxiv.org", "biorxiv", "medrxiv"]),
    # medical/health BEFORE science so consumer-health sites don't fall into 'science' (MMLU:
    # clinical_knowledge, anatomy, professional/college medicine, medical_genetics, virology, nutrition).
    (
        "medical",
        [
            "mayoclinic",
            "webmd",
            "healthline",
            "medlineplus",
            "medscape",
            "drugs.com",
            "patient.info",
            "clevelandclinic",
            "verywellhealth",
            "everydayhealth",
            "rxlist",
            "healthgrades",
            "medicalnewstoday",
            "nih.gov/health",
            "cdc.gov",
            "who.int",
            "uptodate",
            "merckmanuals",
        ],
    ),
    # --- hypothesized gaps (put specific ones early so they win over science/edu/reference) ---
    (
        "math",
        [
            "math.stackexchange",
            "mathoverflow",
            "khanacademy",
            "mathworld",
            "wolframalpha",
            "wolfram.com",
            "proofwiki",
            "artofproblemsolving",
            "brilliant.org",
            "cut-the-knot",
            "mathisfun",
            "purplemath",
            "encyclopediaofmath",
            "aops.com",
        ],
    ),
    (
        "legal",
        [
            "openjurist",
            "justia",
            "law.cornell",
            "courtlistener",
            "findlaw",
            "leagle",
            "casetext",
            "supremecourt",
            "oyez.org",
            "law.com",
            "caselaw",
            "ecfr.gov",
            "govinfo.gov",
            "regulations.gov",
        ],
    ),
    (
        "science",
        [
            "biomedcentral",
            "sciencedaily",
            "phys.org",
            "plos",
            "ncbi.nlm.nih",
            "pubmed",
            "jove.com",
            "livescience",
            "scientificamerican",
            "nature.com",
            "sciencemag",
            "eurekalert",
            "nih.gov",
            "nasa.gov",
            "usgs.gov",
            "sciencenews",
        ],
    ),
    (
        "educational",
        [
            "coursera",
            "edx.org",
            "ocw.mit",
            "study.com",
            "sparknotes",
            "coursehero",
            "quizlet",
            "cliffsnotes",
            "shmoop",
            "chegg",
            "gutenberg",
            "bartleby",
            "toppr",
            "byjus",
        ],
    ),
    # MMLU social-science / humanities registers (topic, not format). Kept high-precision.
    (
        "business_econ",
        [
            "investopedia",
            "bloomberg",
            "forbes",
            "wsj.com",
            "economist.com",
            "marketwatch",
            "fool.com",
            "nasdaq.com",
            "seekingalpha",
            "businessinsider",
            "hbr.org",
            "cnbc",
            "inc.com",
            "entrepreneur.com",
        ],
    ),
    (
        "history",
        [
            "history.com",
            "historyextra",
            "worldhistory.org",
            "ancient.eu",
            "historytoday",
            "smithsonianmag",
            "historynet",
            "britishmuseum",
        ],
    ),
    (
        "philosophy_religion",
        [
            "plato.stanford",
            "iep.utm",
            "biblegateway",
            "biblehub",
            "gotquestions",
            "patheos",
            "catholic.org",
            "sacred-texts",
            "philosophynow",
            "islamqa",
        ],
    ),
    (
        "social_science",
        [
            "psychologytoday",
            "verywellmind",
            "simplypsychology",
            "pewresearch",
            "brookings",
            "jstor",
            "apa.org",
        ],
    ),
    (
        "arts_humanities",
        [
            "metmuseum",
            "tate.org",
            "artsy",
            "wikiart",
            "allmusic",
            "poetryfoundation",
            "litcharts",
            "gradesaver",
        ],
    ),
    (
        "reference_expository",
        [
            "wikipedia",
            "wikia",
            "fandom",
            "wikimili",
            "answers.com",
            "ipl.org",
            "britannica",
            "infoplease",
            "encyclopedia",
            "wikidot",
            "wikibooks",
            "reference.com",
            "thefreedictionary",
        ],
    ),
    (
        "qa_forum",
        [
            "stackexchange",
            "reddit",
            "quora",
            "justanswer",
            "fixya",
            "ask.",
            "answerbag",
            "avvo",
            "forum",
            "phpbb",
            "vbulletin",
            "proboards",
            "boards.",
            "community.",
        ],
    ),
    (
        "howto",
        [
            "wikihow",
            "allrecipes",
            "instructables",
            "ehow",
            "bettycrocker",
            "food.com",
            "myrecipes",
            "blogspot",
            "wordpress",
            "typepad",
            "tumblr",
            "livejournal",
            "medium.com",
            "substack",
        ],
    ),
    # product/commerce/reviews — a web FORMAT hq's spec explicitly drops ("light reviews"); worth
    # a labelled register so we can confirm we *want* it dropped vs. over-dropping real reviews.
    (
        "product_commerce",
        [
            "amazon.",
            "ebay.",
            "etsy.",
            "walmart",
            "aliexpress",
            "yelp.",
            "tripadvisor",
            "trustpilot",
            "bestbuy",
            "target.com",
            "cnet.com",
            "consumerreports",
        ],
    ),
    # --- controls ---
    (
        "news",
        [
            "bbc.",
            "cnn.",
            "nytimes",
            "theguardian",
            "reuters",
            "washingtonpost",
            "npr.org",
            "apnews",
            "aljazeera",
            "foxnews",
            "nbcnews",
            "cbsnews",
            "usatoday",
            "politico",
            "latimes",
        ],
    ),
    (
        "code",
        [
            "github",
            "gitlab",
            "stackoverflow",
            "sourceforge",
            "npmjs",
            "pypi",
            "readthedocs",
            "geeksforgeeks",
            "codeproject",
            "kernel.org",
            "gnu.org",
            "developer.",
            "docs.",
        ],
    ),
]
# Tabular is not cleanly domain-identifiable; seeded by data/statistics domains here and refined at
# decode time via the has_table content flag.
TABULAR_DOMAINS = [
    "data.gov",
    "census.gov",
    "statista",
    "worldbank",
    "bls.gov",
    "tradingeconomics",
    "kaggle",
    "data.world",
    "ourworldindata",
    "knoema",
]


def _case_sql(col: str = "domain") -> str:
    whens = []
    for cat, pats in [*DEVSET_CATEGORIES, ("tabular", TABULAR_DOMAINS)]:
        ors = " OR ".join(f"{col} LIKE '%{p}%'" for p in pats)
        whens.append(f"WHEN {ors} THEN '{cat}'")
    return "CASE " + " ".join(whens) + " ELSE 'other' END"


def _domain_register(domain: str) -> str:
    """Python mirror of `_case_sql` first-match-wins domain→register for non-SQL paths (random pool)."""
    d = (domain or "").lower()
    for cat, pats in [*DEVSET_CATEGORIES, ("tabular", TABULAR_DOMAINS)]:
        if any(p in d for p in pats):
            return cat
    return "other"


def run_select(args: argparse.Namespace) -> int:
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"CREATE TEMP TABLE fwcc AS SELECT DISTINCT url FROM read_parquet('{FWCC}')")
    con.execute(
        f"""CREATE TEMP TABLE mem AS
        SELECT m.url, m.domain, m.snapshot, m.kept_hq, m.kept_dclm, m.kept_nemo, m.kept_fwedu,
               (f.url IS NOT NULL) AS kept_fwcc,
               (m.kept_dclm::INT + m.kept_nemo::INT + m.kept_fwedu::INT + (f.url IS NOT NULL)::INT) AS verifier_score,
               {_case_sql('m.domain')} AS category
        FROM read_parquet('{MEMBERSHIP}') m LEFT JOIN fwcc f ON m.url = f.url"""
    )
    cap = args.cap_per_category
    # Per-category cap via deterministic hash order. 'other' is excluded here; junk is sampled below.
    con.execute(
        f"""CREATE TEMP TABLE cand AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY category ORDER BY hash(url || 'devset')) AS rn
            FROM mem WHERE category <> 'other'
        ) WHERE rn <= {cap}"""
    )
    # Junk/negative anchors: 0-vote docs that hq ALSO dropped, sampled from the 'other' bucket.
    con.execute(
        f"""INSERT INTO cand
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, 'junk' AS category2, row_number() OVER (ORDER BY hash(url || 'junk')) AS rn
            FROM (SELECT * EXCLUDE (category), 'junk' AS category FROM mem
                  WHERE category = 'other' AND verifier_score = 0 AND NOT kept_hq)
        ) WHERE rn <= {cap}"""
    )
    n = con.execute("SELECT count(*) FROM cand").fetchone()[0]
    bycat = con.execute(
        "SELECT category, count(*) n, round(avg(verifier_score),2) avgv, sum(kept_hq::INT) hq_has FROM cand GROUP BY category ORDER BY 2 DESC"
    ).fetchall()
    logger.info("[select] %d candidates. by category (n, avg_verifier, hq_already_has):", n)
    for c, cc, av, hh in bycat:
        logger.info("    %-22s n=%-6d avg_verifier=%-5s hq_has=%d", c, cc, av, hh)
    con.execute(
        f"COPY (SELECT url, domain, category, snapshot, verifier_score, kept_hq, kept_dclm, kept_nemo, kept_fwedu, kept_fwcc FROM cand) "
        f"TO '{CANDIDATES}' (FORMAT parquet, COMPRESSION zstd, PER_THREAD_OUTPUT true, OVERWRITE_OR_IGNORE true)"
    )
    logger.info("[select] wrote candidates → %s", CANDIDATES)
    logger.info("[select] NOTE: random-timespan subset (%s) + tabular has_table refinement handled at decode.", TIMESPAN)
    return 0


def run_preview_sample(args: argparse.Namespace) -> int:
    """Stratified sample of MATCHED docs (per category x verifier tier) with FULL text + a
    re-verification of each recorded eval-subject match on the full text (bug audit)."""
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"CREATE TEMP TABLE fwcc AS SELECT DISTINCT url FROM read_parquet('{FWCC}')")
    con.execute(
        f"""CREATE TEMP TABLE hits AS
        SELECT url, any_value(domain) AS domain, arg_max(subjects, n_examples) AS subjects,
               max(best_frac) AS best_frac, arg_max(snippet, n_examples) AS snippet
        FROM read_parquet('{RESILIPARSE_HITS}') GROUP BY url"""
    )
    con.execute(
        f"""CREATE TEMP TABLE j AS
        SELECT h.url, h.domain, h.subjects, h.best_frac, h.snippet,
               (COALESCE(m.kept_dclm,false)::INT + COALESCE(m.kept_nemo,false)::INT
                + COALESCE(m.kept_fwedu,false)::INT + (f.url IS NOT NULL)::INT) AS verifier_score,
               {_case_sql('h.domain')} AS category
        FROM hits h LEFT JOIN read_parquet('{MEMBERSHIP}') m ON h.url = m.url
        LEFT JOIN fwcc f ON h.url = f.url
        WHERE NOT COALESCE(m.kept_hq, false)"""  # hq dropped it (the missing docs)
    )
    con.execute(
        f"""CREATE TEMP TABLE samp AS SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY category, verifier_score ORDER BY hash(url || 'prev')) AS rn FROM j
        ) WHERE rn <= {args.per_cell}"""
    )
    # Also persist just the sampled urls, so a separate Zephyr pass can attach full text fast.
    con.execute(f"COPY (SELECT url FROM samp) TO '{WORKSPACE}/devset/preview_urls.parquet' (FORMAT parquet)")
    con.execute(f"COPY samp TO '{PREVIEW}' (FORMAT parquet)")
    n = con.execute("SELECT count(*) FROM samp").fetchone()[0]
    bd = con.execute("SELECT category, count(*) FROM samp GROUP BY 1 ORDER BY 2 DESC").fetchall()
    logger.info("[preview-sample] %d docs (per_cell=%d). by category: %s", n, args.per_cell, bd)
    logger.info("[preview-sample] wrote %s", PREVIEW)
    return 0


_FETCH_URLS: frozenset | None = None


def _fetch_keep(rec: dict) -> dict | None:
    global _FETCH_URLS
    url, text = rec.get("url"), rec.get("text")
    if not url or not text:
        return None
    if _FETCH_URLS is None:
        from zephyr.execution import zephyr_worker_ctx

        _FETCH_URLS = frozenset(zephyr_worker_ctx().get_shared("fetch_urls"))
    return {"url": url, "full_text": text[:40000]} if url in _FETCH_URLS else None


def run_fetch_text(args: argparse.Namespace) -> int:
    """Zephyr pass: attach FULL resiliparse text for the preview-sample urls (fast, distributed)."""
    import pyarrow.parquet as pq
    from fray import ResourceConfig
    from zephyr import Dataset, ZephyrContext

    urls_tbl = pq.ParquetDataset(
        f"{WORKSPACE}/devset/preview_urls.parquet".replace("gs://", ""), filesystem=fsspec.filesystem("gcs")
    ).read(columns=["url"])
    urls = list(urls_tbl.column("url").to_pylist())
    logger.info("[fetch-text] %d preview urls", len(urls))
    src = "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f/*.jsonl.gz"
    out = f"{WORKSPACE}/devset/preview_text/t-{{shard:05d}}-of-{{total:05d}}.parquet"
    ds = Dataset.from_files(src).load_jsonl().map(_fetch_keep).filter(lambda x: x is not None).reshard(8)
    ctx = ZephyrContext(
        name="devset-fetch-text",
        max_workers=200,
        resources=ResourceConfig(cpu=1, ram="4g", regions=["us-central2"], preemptible=True),
    )
    ctx.put("fetch_urls", urls)
    ctx.execute(ds.write_parquet(out, skip_existing=True))
    logger.info("[fetch-text] wrote %s", out)
    return 0


def run_hard_sample(args: argparse.Namespace) -> int:
    """Stratified hard-label candidate sample from the 0.8-threshold matches.

    Stratified by (category, verifier_score, kept_hq) so BOTH docs hq kept and dropped appear in
    every register — the gold set can then measure hq's precision (are its keeps good?) as well as
    its recall (are its drops actually bad?), instead of only auditing drops.
    """
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"CREATE TEMP TABLE fwcc AS SELECT DISTINCT url FROM read_parquet('{FWCC}')")
    con.execute(
        f"""CREATE TEMP TABLE hits AS
        SELECT url, any_value(domain) AS domain, arg_max(subjects, n_examples) AS subjects,
               max(best_frac) AS best_frac, arg_max(snippet, n_examples) AS snippet
        FROM read_parquet('{HITS_80}') GROUP BY url"""
    )
    con.execute(
        f"""CREATE TEMP TABLE j AS
        SELECT h.url, h.domain, h.subjects, h.best_frac, h.snippet,
               COALESCE(m.kept_hq,false)::INT AS kept_hq,
               (COALESCE(m.kept_dclm,false)::INT + COALESCE(m.kept_nemo,false)::INT
                + COALESCE(m.kept_fwedu,false)::INT + (f.url IS NOT NULL)::INT) AS verifier_score,
               {_case_sql('h.domain')} AS category
        FROM hits h LEFT JOIN read_parquet('{MEMBERSHIP}') m ON h.url = m.url
        LEFT JOIN fwcc f ON h.url = f.url"""
    )
    # Rank docs within each (register, verifier, kept_hq) stratum by a stable hash.
    con.execute(
        """CREATE TEMP TABLE ranked AS
        SELECT *, row_number() OVER (PARTITION BY category, verifier_score, kept_hq ORDER BY hash(url || 'hard')) AS rn
        FROM j"""
    )
    # Oversampled full pool (metadata only) — retained on GCS for the later LLM-judge soft-labeling pass.
    con.execute(f"CREATE TEMP TABLE pool_full AS SELECT * FROM ranked WHERE rn <= {args.full_per_cell}")
    nf = con.execute("SELECT count(*) FROM pool_full").fetchone()[0]
    con.execute(
        f"COPY (SELECT url, domain, category, kept_hq, verifier_score, best_frac, subjects FROM pool_full) "
        f"TO '{HARD_SAMPLE_FULL}' (FORMAT parquet)"
    )
    # HAND-LABEL subset (~1k). Low rn preserves whatever was already displayed/labeled; a global
    # round-robin over rn caps the total at --label-budget while keeping every stratum represented.
    con.execute(
        f"""CREATE TEMP TABLE samp AS SELECT * EXCLUDE (rn, rr) FROM (
            SELECT *, row_number() OVER (ORDER BY rn, hash(url || 'lbl')) AS rr
            FROM ranked WHERE rn <= {args.per_cell}
        ) WHERE rr <= {args.label_budget}"""
    )
    # Force-include any already-labeled urls that the budget/round-robin would otherwise drop, so a
    # doc you've hand-labeled can never fall out of the label set on a re-run (columns match `j`).
    n_pinned = 0
    if fsspec.filesystem("gcs").exists(PINNED_URLS.replace("gs://", "")):
        con.execute(f"CREATE TEMP TABLE pins AS SELECT DISTINCT url FROM read_parquet('{PINNED_URLS}')")
        n_pinned = con.execute(
            """INSERT INTO samp
            SELECT url, domain, subjects, best_frac, snippet, kept_hq, verifier_score, category FROM j
            WHERE url IN (SELECT url FROM pins) AND url NOT IN (SELECT url FROM samp)"""
        ).fetchone()[0]
        # Pinned urls NOT in the keyword pool (e.g. content-scan tabular/code) come from membership.
        n_pinned += con.execute(
            f"""INSERT INTO samp
            SELECT mm.url, mm.domain, '' AS subjects, 0.0 AS best_frac, '' AS snippet,
                   mm.kept_hq::INT AS kept_hq,
                   (mm.kept_dclm::INT + mm.kept_nemo::INT + mm.kept_fwedu::INT + (f.url IS NOT NULL)::INT) AS verifier_score,
                   {_case_sql('mm.domain')} AS category
            FROM read_parquet('{MEMBERSHIP}') mm LEFT JOIN fwcc f ON mm.url = f.url
            WHERE mm.url IN (SELECT url FROM pins) AND mm.url NOT IN (SELECT url FROM samp)"""
        ).fetchone()[0]
        logger.info("[hard-sample] pinned %d already-labeled/enriched urls back into the label set", n_pinned)
    # Enrich data-poor hq-KEPT registers (fiction/arxiv/math) from the BROADER membership (not just
    # the keyword-matched pool), so hq's precision is measurable where it keeps almost nothing.
    n_enrich = 0
    if args.enrich_per_register > 0:
        con.execute(
            f"""CREATE TEMP TABLE enrich AS SELECT * EXCLUDE (rn) FROM (
                SELECT url, domain, '' AS subjects, 0.0 AS best_frac, '' AS snippet, 1 AS kept_hq,
                       verifier_score, category,
                       row_number() OVER (PARTITION BY category ORDER BY hash(url || 'enrich')) AS rn
                FROM (
                    SELECT mm.url, mm.domain,
                           (mm.kept_dclm::INT + mm.kept_nemo::INT + mm.kept_fwedu::INT + (f.url IS NOT NULL)::INT) AS verifier_score,
                           {_case_sql('mm.domain')} AS category
                    FROM read_parquet('{MEMBERSHIP}') mm LEFT JOIN fwcc f ON mm.url = f.url
                    WHERE mm.kept_hq
                ) WHERE category IN ('fiction', 'arxiv', 'math')
            ) WHERE rn <= {args.enrich_per_register}"""
        )
        n_enrich = con.execute(
            """INSERT INTO samp
            SELECT url, domain, subjects, best_frac, snippet, kept_hq, verifier_score, category FROM enrich
            WHERE url NOT IN (SELECT url FROM samp)"""
        ).fetchone()[0]
        logger.info("[hard-sample] enriched %d hq-kept fiction/arxiv/math docs from membership", n_enrich)
    n = con.execute("SELECT count(*) FROM samp").fetchone()[0]
    n_kept = con.execute("SELECT sum(kept_hq) FROM samp").fetchone()[0]
    bd = con.execute(
        "SELECT category, count(*) AS n, sum(kept_hq) AS hq_kept FROM samp GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    logger.info(
        "[hard-sample] full pool %d docs (soft-label). LABEL set %d: %d hq-kept + %d hq-dropped.",
        nf,
        n,
        n_kept,
        n - n_kept,
    )
    for c, cc, hk in bd:
        logger.info("    %-22s n=%-5d hq_kept=%d hq_dropped=%d", c, cc, hk, cc - hk)
    con.execute(f"COPY samp TO '{HARD_SAMPLE}' (FORMAT parquet)")
    con.execute(f"COPY (SELECT url FROM samp) TO '{HARD_URLS}' (FORMAT parquet)")
    logger.info("[hard-sample] wrote %s (label) + %s (full) + %s", HARD_SAMPLE, HARD_SAMPLE_FULL, HARD_URLS)
    return 0


def run_fetch_extractions(args: argparse.Namespace) -> int:
    """Zephyr: fetch one pipeline's extracted text for the hard-sample urls (in that pipeline's region)."""
    import pyarrow.parquet as pq
    from fray import ResourceConfig
    from zephyr import Dataset, ZephyrContext

    from experiments.baseline_collection.provenance_audit_10k import SOURCES

    src = SOURCES[args.method]
    urls_path = getattr(args, "urls", None) or HARD_URLS
    urls = list(
        pq.ParquetDataset(urls_path.replace("gs://", ""), filesystem=fsspec.filesystem("gcs"))
        .read(columns=["url"])
        .column("url")
        .to_pylist()
    )
    region = src["root"].split("/")[2].removeprefix("marin-")
    out_dir = getattr(args, "out", None) or HARD_EXTRACTIONS
    out = f"{out_dir}/{args.method}/e-{{shard:05d}}-of-{{total:05d}}.parquet"
    ds = Dataset.from_files(f"{src['root']}/{src['glob']}")
    ds = ds.load_parquet() if src["format"] == "parquet" else ds.load_jsonl()
    ds = ds.map(_fetch_keep).filter(lambda x: x is not None).reshard(8)
    ctx = ZephyrContext(
        name=f"fetch-{args.method}",
        max_workers=200,
        resources=ResourceConfig(cpu=1, ram="4g", regions=[region], preemptible=True),
    )
    ctx.put("fetch_urls", urls)
    ctx.execute(ds.write_parquet(out, skip_existing=True))
    logger.info("[fetch-extractions] %s (%s) -> %s", args.method, region, out)
    return 0


_NUM_RE = re.compile(r"^-?[\d,.$%()]+$")
_CODE_TOKENS = (
    "def ",
    "class ",
    "function",
    "import ",
    "#include",
    "public ",
    "private ",
    "static ",
    "var ",
    "const ",
    "let ",
    "return ",
    "console.log",
    "<?php",
    "});",
    "print(",
    "for (",
    "if (",
    "while (",
    "#!/",
    "</",
    "=>",
    "&&",
    "||",
    "== ",
    "!= ",
)


def _tabular_signal(text: str) -> float:
    """Fraction of non-blank lines that look like table rows (>=3 delimited cells, >=1 numeric)."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 6:
        return 0.0
    rows = 0
    for ln in lines:
        cells = [c for c in re.split(r"\t|\s{2,}|\s*\|\s*", ln.strip()) if c]
        if len(cells) >= 3 and any(_NUM_RE.match(c.replace(",", "")) for c in cells):
            rows += 1
    return rows / len(lines) if rows >= 6 else 0.0


def _code_signal(text: str) -> float:
    """Fraction of non-blank lines that look like source code."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 6:
        return 0.0
    code = 0
    for ln in lines:
        s = ln.strip()
        if any(t in s for t in _CODE_TOKENS) or s.endswith(("{", "}", ";", ":")) or re.match(r"^\s{2,}\S", ln):
            code += 1
    return code / len(lines) if code >= 6 else 0.0


def _scan_tabular(rec: dict) -> dict | None:
    url, text = rec.get("url"), rec.get("text")
    if not url or not text:
        return None
    sig = _tabular_signal(text)
    return {"url": url, "signal": round(sig, 3), "snippet": text[:1400]} if sig >= 0.30 else None


def _scan_code(rec: dict) -> dict | None:
    url, text = rec.get("url"), rec.get("text")
    if not url or not text:
        return None
    sig = _code_signal(text)
    return {"url": url, "signal": round(sig, 3), "snippet": text[:1400]} if sig >= 0.30 else None


def run_dclm_wins(args: argparse.Namespace) -> int:
    """Find the highest-quality docs DCLM KEEPS but hq DROPS — DCLM's 'wins', ranked by DCLM's own
    fastText score (dclm_ft). One per domain for diversity; surfaces non-obvious registers where hq
    is losing to DCLM, beyond the domain-based searches."""
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"CREATE TEMP TABLE fwcc AS SELECT DISTINCT url FROM read_parquet('{FWCC}')")
    con.execute(
        f"""CREATE TEMP TABLE w AS
        SELECT m.url, m.domain, m.dclm_ft, m.fineweb_score, m.dclm_len,
               (m.kept_nemo::INT + m.kept_fwedu::INT + (f.url IS NOT NULL)::INT) AS other_agree,
               {_case_sql('m.domain')} AS register
        FROM read_parquet('{MEMBERSHIP}') m LEFT JOIN fwcc f ON m.url = f.url
        WHERE m.kept_dclm AND NOT m.kept_hq AND m.dclm_ft IS NOT NULL AND COALESCE(m.dclm_len, 0) >= 500"""
    )
    con.execute(
        f"""COPY (
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (PARTITION BY domain ORDER BY dclm_ft DESC) AS rn FROM w
            ) WHERE rn = 1 ORDER BY dclm_ft DESC LIMIT {args.n}
        ) TO '{args.out}' (FORMAT parquet)"""
    )
    n, tot = (
        con.execute(f"SELECT count(*) FROM read_parquet('{args.out}')").fetchone()[0],
        con.execute("SELECT count(*) FROM w").fetchone()[0],
    )
    byreg = con.execute(
        f"SELECT register, count(*) FROM read_parquet('{args.out}') GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    logger.info("[dclm-wins] %d DCLM-keep/hq-drop docs total; top %d (1/domain) by dclm_ft -> %s", tot, n, args.out)
    logger.info("[dclm-wins] by register: %s", byreg)
    return 0


def _fetch_decoded(rec: dict) -> dict | None:
    """Emit the FULL raw html (+ text_body) for urls in the shared set (from the decoded_10k store).
    Raw html lets downstream preserve <pre>/<code> LINE STRUCTURE that the flattened text_body loses."""
    global _FETCH_URLS
    url = rec.get("url")
    if not url:
        return None
    if _FETCH_URLS is None:
        from zephyr.execution import zephyr_worker_ctx

        _FETCH_URLS = frozenset(zephyr_worker_ctx().get_shared("fetch_urls"))
    if url not in _FETCH_URLS:
        return None
    return {"url": url, "html": (rec.get("html") or "")[:120000], "full_text": (rec.get("text_body") or "")[:40000]}


def run_fetch_decoded(args: argparse.Namespace) -> int:
    """Zephyr (us-east5): pull the full text_body for a url list from the decoded_10k HTML store."""
    import pyarrow.parquet as pq
    from fray import ResourceConfig
    from zephyr import Dataset, ZephyrContext

    urls = list(
        pq.ParquetDataset(args.urls.replace("gs://", ""), filesystem=fsspec.filesystem("gcs"))
        .read(columns=["url"])
        .column("url")
        .to_pylist()
    )
    decoded = "gs://marin-us-east5/documents/bert_pipeline/decoded_10k/*.parquet"
    ds = Dataset.from_files(decoded).load_parquet().map(_fetch_decoded).filter(lambda x: x is not None).reshard(2)
    ctx = ZephyrContext(
        name="fetch-decoded",
        max_workers=200,
        resources=ResourceConfig(cpu=1, ram="8g", regions=["us-east5"], preemptible=True),
    )
    ctx.put("fetch_urls", urls)
    ctx.execute(ds.write_parquet(f"{args.out}/d-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True))
    logger.info("[fetch-decoded] %d urls -> %s", len(urls), args.out)
    return 0


def run_pipeline_flags(args: argparse.Namespace) -> int:
    """Join urls to membership+fwcc -> each pipeline's keep decision, for computing per-pipeline F1.
    With --normalize, match on a normalized url (strip scheme/www/query/fragment/trailing-slash) and
    OR-aggregate flags across collisions, to recover docs the exact join missed on url quirks."""
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"CREATE TEMP TABLE fwcc AS SELECT DISTINCT url FROM read_parquet('{FWCC}')")
    con.execute(f"CREATE TEMP TABLE q AS SELECT DISTINCT url FROM read_parquet('{args.urls}')")
    if args.normalize:
        con.execute(
            r"CREATE MACRO nrm(u) AS rtrim(split_part(split_part(regexp_replace(lower(u), '^https?://(www\.)?', ''), '?', 1), '#', 1), '/')"
        )
        con.execute(
            f"""COPY (
                SELECT q.url,
                       max((m.url IS NOT NULL)::INT)::BOOL AS in_membership,
                       max(COALESCE(m.kept_hq, false)::INT)::BOOL AS kept_hq,
                       max(COALESCE(m.kept_dclm, false)::INT)::BOOL AS kept_dclm,
                       max(COALESCE(m.kept_nemo, false)::INT)::BOOL AS kept_nemo,
                       max(COALESCE(m.kept_fwedu, false)::INT)::BOOL AS kept_fwedu,
                       max((f.url IS NOT NULL)::INT)::BOOL AS kept_fwcc
                FROM q LEFT JOIN read_parquet('{MEMBERSHIP}') m ON nrm(q.url) = nrm(m.url)
                       LEFT JOIN fwcc f ON nrm(q.url) = nrm(f.url)
                GROUP BY q.url
            ) TO '{args.out}' (FORMAT parquet)"""
        )
    else:
        con.execute(
            f"""COPY (
                SELECT q.url, (m.url IS NOT NULL) AS in_membership,
                       COALESCE(m.kept_hq, false) AS kept_hq, COALESCE(m.kept_dclm, false) AS kept_dclm,
                       COALESCE(m.kept_nemo, false) AS kept_nemo, COALESCE(m.kept_fwedu, false) AS kept_fwedu,
                       (f.url IS NOT NULL) AS kept_fwcc
                FROM q LEFT JOIN read_parquet('{MEMBERSHIP}') m ON q.url = m.url LEFT JOIN fwcc f ON q.url = f.url
            ) TO '{args.out}' (FORMAT parquet)"""
        )
    n = con.execute(f"SELECT count(*), sum(in_membership::INT) FROM read_parquet('{args.out}')").fetchone()
    logger.info("[pipeline-flags] %d urls (%d in membership, normalize=%s) -> %s", n[0], n[1], args.normalize, args.out)
    return 0


def run_pipeline_labels(args: argparse.Namespace) -> int:
    """Like pipeline-flags but also carries each pipeline's raw score + extracted length + domain.

    For per-dataset/per-domain F1 vs the human dev-set labels AND for sweeping each classifier's own
    threshold (dclm_ft / nemo_quality / fineweb_score are the raw scores behind kept_dclm/nemo/fwedu).
    Normalized-url match (strip scheme/www/query/fragment/trailing-slash); OR keeps, max scores/lengths
    across collisions. `in_membership=false` marks docs outside the 10k pool (sampled past the cutoff)."""
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET temp_directory='/tmp/duckdb_spill';")  # allow spilling so the 32M/41M scans stay in budget
    con.execute("SET memory_limit='3GB';")
    con.execute(f"CREATE TEMP TABLE q AS SELECT DISTINCT url FROM read_parquet('{args.urls}')")
    con.execute(
        r"CREATE MACRO nrm(u) AS rtrim(split_part(split_part(regexp_replace(lower(u), '^https?://(www\.)?', ''), '?', 1), '#', 1), '/')"
    )
    # membership carries the keeps + scores; fwcc is a separate url-only keep list. Do the two joins
    # separately (each streams the big table probing the tiny q) and merge — never materialize 41M urls.
    con.execute(
        f"""CREATE TEMP TABLE mem AS
            SELECT q.url,
                   max((m.url IS NOT NULL)::INT)::BOOL AS in_membership,
                   any_value(m.domain) AS domain,
                   max(COALESCE(m.kept_hq, false)::INT)::BOOL AS kept_hq,
                   max(COALESCE(m.kept_dclm, false)::INT)::BOOL AS kept_dclm,
                   max(COALESCE(m.kept_nemo, false)::INT)::BOOL AS kept_nemo,
                   max(COALESCE(m.kept_fwedu, false)::INT)::BOOL AS kept_fwedu,
                   max(m.dclm_ft) AS dclm_ft,
                   max(m.nemo_quality) AS nemo_quality,
                   max(m.fineweb_score) AS fineweb_score,
                   max(m.hq_len) AS hq_len, max(m.dclm_len) AS dclm_len,
                   max(m.nemo_len) AS nemo_len, max(m.fwedu_len) AS fwedu_len,
                   any_value(m.snapshot) AS snapshot
            FROM q LEFT JOIN read_parquet('{MEMBERSHIP}') m ON nrm(q.url) = nrm(m.url)
            GROUP BY q.url"""
    )
    con.execute(
        f"""CREATE TEMP TABLE fw AS
            SELECT q.url, max((f.url IS NOT NULL)::INT)::BOOL AS kept_fwcc
            FROM q LEFT JOIN read_parquet('{FWCC}') f ON nrm(q.url) = nrm(f.url)
            GROUP BY q.url"""
    )
    con.execute(
        f"""COPY (
            SELECT mem.*, COALESCE(fw.kept_fwcc, false) AS kept_fwcc
            FROM mem LEFT JOIN fw ON mem.url = fw.url
        ) TO '{args.out}' (FORMAT parquet)"""
    )
    n = con.execute(f"SELECT count(*), sum(in_membership::INT) FROM read_parquet('{args.out}')").fetchone()
    logger.info("[pipeline-labels] %d urls (%d in 10k membership) -> %s", n[0], n[1], args.out)
    return 0


def run_hq_lookup(args: argparse.Namespace) -> int:
    """Join arbitrary urls to membership -> {url, kept_hq, verifier_score, in_membership}.
    Used to find which mined hard-negatives hq KEPT (= hq false positives, the valuable ones)."""
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"CREATE TEMP TABLE fwcc AS SELECT DISTINCT url FROM read_parquet('{FWCC}')")
    con.execute(f"CREATE TEMP TABLE q AS SELECT DISTINCT url FROM read_parquet('{args.urls}')")
    con.execute(
        f"""COPY (
            SELECT q.url, COALESCE(m.kept_hq, false) AS kept_hq,
                   (COALESCE(m.kept_dclm,false)::INT + COALESCE(m.kept_nemo,false)::INT
                    + COALESCE(m.kept_fwedu,false)::INT + (f.url IS NOT NULL)::INT) AS verifier_score,
                   (m.url IS NOT NULL) AS in_membership
            FROM q LEFT JOIN read_parquet('{MEMBERSHIP}') m ON q.url = m.url LEFT JOIN fwcc f ON q.url = f.url
        ) TO '{args.out}' (FORMAT parquet)"""
    )
    n = con.execute(
        f"SELECT count(*), sum(in_membership::INT), sum(kept_hq::INT) FROM read_parquet('{args.out}')"
    ).fetchone()
    logger.info("[hq-lookup] %d urls: %s in membership, %s hq-kept -> %s", n[0], n[1], n[2], args.out)
    return 0


def run_content_topk(args: argparse.Namespace) -> int:
    """In-region: rank content-scan candidates by signal, keep the top-N above a high threshold
    (the 0.30 scan threshold is recall-oriented and matches too much to pull; this cuts to precision)."""
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    src = f"{WORKSPACE}/devset/content_scan/{args.shape}/*.parquet"
    out = f"{WORKSPACE}/devset/content_scan/{args.shape}_top.parquet"
    # Streaming top-N (no GROUP BY — that hash-aggregates every url and OOMs on the huge candidate set);
    # dedup the small result downstream.
    con.execute(
        f"""COPY (
            SELECT url, signal, snippet FROM read_parquet('{src}')
            WHERE signal >= {args.min_signal} ORDER BY signal DESC LIMIT {args.n}
        ) TO '{out}' (FORMAT parquet)"""
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    logger.info("[content-topk] %s: %d docs (signal>=%.2f) -> %s", args.shape, n, args.min_signal, out)
    return 0


def run_content_scan(args: argparse.Namespace) -> int:
    """Zephyr: scan the resiliparse universe for docs whose STRUCTURE matches a page-shape register
    (tabular/code) — these registers aren't domain-identifiable, so content is the only signal.
    Emits candidates {url, signal, snippet} above threshold; rank/cap/verify happens downstream."""
    from fray import ResourceConfig
    from zephyr import Dataset, ZephyrContext

    from experiments.baseline_collection.provenance_audit_10k import SOURCES

    src = SOURCES["resiliparse"]
    region = src["root"].split("/")[2].removeprefix("marin-")
    fn = {"tabular": _scan_tabular, "code": _scan_code}[args.shape]
    out = f"{WORKSPACE}/devset/content_scan/{args.shape}/c-{{shard:05d}}-of-{{total:05d}}.parquet"
    ds = Dataset.from_files(f"{src['root']}/{src['glob']}")
    ds = ds.load_parquet() if src["format"] == "parquet" else ds.load_jsonl()
    ds = ds.map(fn).filter(lambda x: x is not None).reshard(8)
    ctx = ZephyrContext(
        name=f"content-{args.shape}",
        max_workers=200,
        resources=ResourceConfig(cpu=1, ram="4g", regions=[region], preemptible=True),
    )
    ctx.execute(ds.write_parquet(out, skip_existing=True))
    logger.info("[content-scan] %s -> %s", args.shape, out)
    return 0


def _arxiv_metric(rec: dict) -> dict | None:
    u, t = rec.get("url"), rec.get("text")
    if not u or not t or "arxiv.org" not in u:
        return None
    tl = t.lower()
    nd = t.count("$")
    refs = tl.count("arxiv:")
    return {
        "url": u,
        "n_chars": len(t),
        "n_dollar": nd,
        "arxiv_refs": refs,
        "is_listing": int("authors and titles" in tl or refs >= 5),
        "heavy_latex": int(nd >= 15),
    }


def run_arxiv_audit(args: argparse.Namespace) -> int:
    """Zephyr: over one pipeline's whole corpus, measure LaTeX-garbling + listing rate on arxiv docs."""
    from fray import ResourceConfig
    from zephyr import Dataset, ZephyrContext

    from experiments.baseline_collection.provenance_audit_10k import SOURCES

    src = SOURCES[args.method]
    region = src["root"].split("/")[2].removeprefix("marin-")
    out = f"{WORKSPACE}/devset/arxiv_audit/{args.method}/a-{{shard:05d}}-of-{{total:05d}}.parquet"
    ds = Dataset.from_files(f"{src['root']}/{src['glob']}")
    ds = ds.load_parquet() if src["format"] == "parquet" else ds.load_jsonl()
    ds = ds.map(_arxiv_metric).filter(lambda x: x is not None).reshard(4)
    ctx = ZephyrContext(
        name=f"arxiv-{args.method}",
        max_workers=200,
        resources=ResourceConfig(cpu=1, ram="4g", regions=[region], preemptible=True),
    )
    ctx.execute(ds.write_parquet(out, skip_existing=True))
    logger.info("[arxiv-audit] %s -> %s", args.method, out)
    return 0


def run_random_sample(args: argparse.Namespace) -> int:
    """Uniform random pool from the 2013-2026 timespan HTML sample (us-east5, in-region).

    Draws N docs by hash(doc_id) — no keyword-match, no verifier tier, no pipeline decision — so
    this is the unbiased natural-web anchor for the keep/drop policy (junk/redirects INCLUDED, on
    purpose). The pipelines never processed these, so we run resiliparse_main ourselves for a doc
    to judge. Writes random_sample.parquet (metadata) + random_resiliparse.parquet (extracted text).
    """
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    from experiments.baseline_collection.extractors import resiliparse_main
    from experiments.baseline_collection.provenance_audit_10k import registered_domain

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    # Pick doc_ids first (cheap — no html), then fetch html for only those N.
    con.execute(
        f"""CREATE TEMP TABLE ids AS SELECT doc_id FROM (
            SELECT doc_id, row_number() OVER (ORDER BY hash(doc_id || 'random-devset')) AS rn
            FROM read_parquet('{TIMESPAN}/*.parquet')
        ) WHERE rn <= {args.n}"""
    )
    rows = con.execute(
        f"""SELECT t.url, t.snapshot, t.html
        FROM read_parquet('{TIMESPAN}/*.parquet') t SEMI JOIN ids USING (doc_id)"""
    ).fetchall()
    logger.info("[random-sample] drew %d docs from timespan; extracting resiliparse_main…", len(rows))

    meta, ext, empty = [], [], 0
    for url, snapshot, html in rows:
        text = resiliparse_main(html or "", url)
        if not text.strip():
            empty += 1
        dom = registered_domain(url)
        meta.append(
            {
                "url": url,
                "domain": dom,
                "category": "random",  # grouping bucket in the tool
                "reg_guess": _domain_register(dom),  # register dropdown default (mostly 'other')
                "snapshot": snapshot,
                "verifier_score": None,  # pipelines never scored these
                "best_frac": 0.0,
                "subjects": "",
            }
        )
        ext.append({"url": url, "full_text": text[:40000]})
    logger.info("[random-sample] %d/%d extracted empty (redirects/js/junk — kept as DROP candidates)", empty, len(rows))
    pq.write_table(pa.Table.from_pylist(meta), RANDOM_SAMPLE.replace("gs://", ""), filesystem=fsspec.filesystem("gcs"))
    pq.write_table(pa.Table.from_pylist(ext), RANDOM_EXT.replace("gs://", ""), filesystem=fsspec.filesystem("gcs"))
    logger.info("[random-sample] wrote %s + %s", RANDOM_SAMPLE, RANDOM_EXT)
    return 0


def run_coverage(args: argparse.Namespace) -> int:
    """Symmetric per-category coverage: among docs kept by >=threshold of ALL 5 pipelines
    (hq/dclm/nemo/fwedu/fwcc counting equally), what % did each pipeline include?"""
    import json

    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"CREATE TEMP TABLE fwcc AS SELECT DISTINCT url FROM read_parquet('{FWCC}')")
    con.execute(
        f"""CREATE TEMP TABLE m AS
        SELECT {_case_sql('mm.domain')} AS category,
               mm.kept_hq::INT AS hq, mm.kept_dclm::INT AS dclm, mm.kept_nemo::INT AS nemo,
               mm.kept_fwedu::INT AS fwedu, (f.url IS NOT NULL)::INT AS fwcc,
               (mm.kept_hq::INT + mm.kept_dclm::INT + mm.kept_nemo::INT + mm.kept_fwedu::INT + (f.url IS NOT NULL)::INT) AS cons
        FROM read_parquet('{MEMBERSHIP}') mm LEFT JOIN fwcc f ON mm.url = f.url"""
    )
    t = args.threshold
    pipes = ["hq", "dclm", "nemo", "fwedu", "fwcc"]
    cov = ", ".join(
        f"round(100.0*sum({p}) FILTER (WHERE cons>={t}) / nullif(sum((cons>={t})::INT),0), 1) AS {p}" for p in pipes
    )
    rows = con.execute(
        f"SELECT category, count(*) AS n_all, sum((cons>={t})::INT) AS n_good, {cov} FROM m GROUP BY category ORDER BY n_good DESC"
    ).fetchall()
    cols = ["category", "n_all", f"n_good(>={t}of5)", *pipes]
    out = [dict(zip(cols, r, strict=False)) for r in rows]
    logger.info("[coverage] symmetric coverage of >=%d-of-5 docs, by category (pipeline = %% of good docs it kept):", t)
    logger.info("  %s", " | ".join(cols))
    for r in out:
        logger.info("  %s", r)
    with fsspec.open(f"{WORKSPACE}/devset/coverage.json", "w") as fh:
        json.dump(out, fh)
    logger.info("[coverage] wrote %s/devset/coverage.json", WORKSPACE)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select", help="pick ~cap/category candidate urls with verifier_score")
    s.add_argument("--cap-per-category", type=int, default=3000)
    s.set_defaults(func=run_select)
    v = sub.add_parser("preview-sample", help="stratified matched-doc sample (category x tier)")
    v.add_argument("--per-cell", type=int, default=6)
    v.set_defaults(func=run_preview_sample)
    ft = sub.add_parser("fetch-text", help="Zephyr: attach full resiliparse text for preview urls")
    ft.set_defaults(func=run_fetch_text)
    hs = sub.add_parser("hard-sample", help="stratified hard-label candidates from the 0.8 matches")
    hs.add_argument("--per-cell", type=int, default=20, help="cap per (register,verifier,kept_hq) cell before budget")
    hs.add_argument("--label-budget", type=int, default=1500, help="max keyword-matched hand-label docs")
    hs.add_argument("--full-per-cell", type=int, default=60, help="cap per cell for the oversampled soft-label pool")
    hs.add_argument(
        "--enrich-per-register", type=int, default=25, help="hq-kept fiction/arxiv/math pulled from membership"
    )
    hs.set_defaults(func=run_hard_sample)
    fe = sub.add_parser("fetch-extractions", help="Zephyr: fetch one pipeline's text for hard-sample urls")
    fe.add_argument("--method", required=True, choices=["hq", "dclm", "nemo", "fwedu", "resiliparse"])
    fe.add_argument("--urls", default=None, help="url-list parquet (default: hard-sample urls)")
    fe.add_argument("--out", default=None, help="output dir (default: hard extractions dir)")
    fe.set_defaults(func=run_fetch_extractions)
    rs = sub.add_parser("random-sample", help="uniform random pool from the 2013-2026 timespan HTML sample")
    rs.add_argument("--n", type=int, default=150)
    rs.set_defaults(func=run_random_sample)
    cv = sub.add_parser("coverage", help="symmetric per-category coverage of >=threshold-of-5 docs")
    cv.add_argument("--threshold", type=int, default=2)
    cv.set_defaults(func=run_coverage)
    cs = sub.add_parser("content-scan", help="find tabular/code docs by page STRUCTURE (not domain)")
    cs.add_argument("--shape", required=True, choices=["tabular", "code"])
    cs.set_defaults(func=run_content_scan)
    dw = sub.add_parser("dclm-wins", help="highest-dclm_ft docs DCLM keeps but hq drops (DCLM's wins)")
    dw.add_argument("--n", type=int, default=500)
    dw.add_argument("--out", required=True)
    dw.set_defaults(func=run_dclm_wins)
    fd = sub.add_parser("fetch-decoded", help="pull full text_body for a url list from decoded_10k (us-east5)")
    fd.add_argument("--urls", required=True)
    fd.add_argument("--out", required=True)
    fd.set_defaults(func=run_fetch_decoded)
    pf = sub.add_parser("pipeline-flags", help="join urls to membership+fwcc for all pipeline keep decisions")
    pf.add_argument("--urls", required=True)
    pf.add_argument("--out", required=True)
    pf.add_argument("--normalize", action="store_true", help="match on normalized url (recover url-quirk misses)")
    pf.set_defaults(func=run_pipeline_flags)
    pl = sub.add_parser("pipeline-labels", help="pipeline keeps + raw scores + lengths per url (for per-domain F1)")
    pl.add_argument("--urls", required=True)
    pl.add_argument("--out", required=True)
    pl.set_defaults(func=run_pipeline_labels)
    hq = sub.add_parser("hq-lookup", help="join urls to membership for kept_hq/verifier (find hq false-positives)")
    hq.add_argument("--urls", required=True)
    hq.add_argument("--out", required=True)
    hq.set_defaults(func=run_hq_lookup)
    ck = sub.add_parser("content-topk", help="rank content-scan candidates by signal, keep top-N")
    ck.add_argument("--shape", required=True, choices=["tabular", "code"])
    ck.add_argument("--min-signal", type=float, default=0.55)
    ck.add_argument("--n", type=int, default=300)
    ck.set_defaults(func=run_content_topk)
    aa = sub.add_parser("arxiv-audit", help="measure LaTeX-garbling/listing rate on a pipeline's arxiv docs")
    aa.add_argument("--method", required=True, choices=["hq", "dclm", "nemo", "fwedu", "resiliparse"])
    aa.set_defaults(func=run_arxiv_audit)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
