# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Edge-case search: Gemini query expansion -> BM25 retrieval -> Gemini rerank
-> kept/dropped cross-reference.

The pipeline probes how different extraction pipelines handle a topic or edge
case. Retrieval runs in-region on the worker (one fan-out ``bm25_multi`` call
over all warm datasets); the kept/dropped cross-reference is a local lookup over
the cached coverage keys. Designed to finish in well under 25s (the two Gemini
calls dominate); ``on_progress`` reports each stage for a live progress bar.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from experiments.spec_explorer import coverage_service, gemini_client, worker_manager
from experiments.spec_explorer.catalog import index_datasets
from experiments.url_index.keys import u64, url_key

logger = logging.getLogger(__name__)

# The core comparison datasets are consolidated in us-central1; fastpipe (us-east5)
# joins once its worker is up.
SEARCH_REGION = "us-central1"
CANDIDATE_CAP = 40  # candidates handed to the reranker (bounded for latency + prompt size)
BM25_K = 15  # hits per (dataset, query) before pooling
RERANK_SNIPPET_CHARS = 1500  # fuller text resolved per candidate so the reranker judges real content
SHUFFLE_CANDIDATES = False  # keep BM25-relevance order (A/B: more precise, no dataset bias — see rerank step)


def _seed(query: str) -> int:
    """Stable per-query seed (process-independent, unlike ``hash``)."""
    return int(hashlib.sha1(query.encode()).hexdigest()[:8], 16)


def _url_h(url: str) -> int | None:
    uk = url_key(url) if url else ""
    return u64(uk) if uk else None


def _apply_crossref(results: list[dict]) -> list[dict]:
    """Set ``kept_by``/``dropped_by`` on each result from the local coverage keys
    and return the per-strategy keep/discard ``keep_summary``.

    Uses ``index_datasets()`` as the full universe, so any dataset registered since
    a result was produced shows up (in ``dropped_by`` when it didn't keep the doc).
    Shared by live search and :func:`backport_result` so both stay consistent.
    """
    all_ds = index_datasets()
    for c in results:
        c["url_h"] = str(_url_h(c.get("url") or "")) if c.get("url") else c.get("url_h")
    hs = [int(c["url_h"]) for c in results if c.get("url_h")]
    kept = coverage_service.which_datasets_have(hs)
    for c in results:
        if c.get("url_h"):
            k = kept.get(int(c["url_h"]), [])
            c["kept_by"] = sorted(k)
            c["dropped_by"] = sorted(set(all_ds) - set(k))
        else:
            c["kept_by"], c["dropped_by"] = [], []
    kept_c: Counter = Counter()
    drop_c: Counter = Counter()
    for c in results:
        kept_c.update(c["kept_by"])
        drop_c.update(c["dropped_by"])
    return [
        {"dataset": ds, "kept": kept_c.get(ds, 0), "dropped": drop_c.get(ds, 0)}
        for ds in all_ds
        if (kept_c.get(ds, 0) + drop_c.get(ds, 0)) > 0
    ]


def backport_result(result: dict, *, reembed: bool = True) -> dict:
    """Re-run only the cross-reference (+ keep_summary, + extraction embedding) on a
    stored result, so datasets registered after it was cached appear in it.

    No Gemini and no BM25 — just the local coverage-key lookup and (if ``reembed``)
    an extraction resolve for kept datasets whose worker is warm. Idempotent.
    """
    results = result.get("results") or []
    result["keep_summary"] = _apply_crossref(results)
    if reembed:
        _autoload_extractions(results)
    result["backported"] = True
    return result


def backport_results(result_objs: list[dict], *, reembed: bool = True) -> None:
    """Backport many stored result objects in place, efficiently.

    Recomputes each object's cross-ref + keep_summary, then (if ``reembed``) embeds
    extractions for ALL of them in a single bulk pass — one worker call per dataset
    total, instead of one per saved search.
    """
    for obj in result_objs:
        obj["keep_summary"] = _apply_crossref(obj.get("results") or [])
        obj["backported"] = True
    if reembed:
        all_results = [c for obj in result_objs for c in obj.get("results") or []]
        if all_results:
            _autoload_extractions(all_results)


def search(
    query: str,
    *,
    top_k: int = 12,
    model: str = gemini_client.DEFAULT_MODEL,
    on_progress=None,
    example_mode: bool = False,
) -> dict:
    """Run the full search pipeline for ``query`` and return a structured result.

    Default (``example_mode=False``) is the plain about-topic search: expand into
    topic queries and rank by relevance + quality. Opt-in ``example_mode=True`` also
    retrieves for the "actual example / primary source" facet and tells the reranker
    to prefer real examples (primary text) over commentary — useful sometimes, but
    it can pull descriptions-of rather than the work itself, so it is not the default.
    """

    def emit(stage: str, **info):
        if on_progress:
            on_progress(stage, info)

    t0 = time.time()
    timing: dict[str, float] = {}

    # 1. Gemini expands the NL query into BM25 queries + keywords.
    emit("expand")
    s = time.time()
    plan = gemini_client.expand_query(query, model=model)
    timing["expand"] = round(time.time() - s, 2)
    # example_queries surface primary sources / actual instances (the underserved
    # case), bm25_queries surface pages about the topic. In example_mode retrieve
    # for both (examples first, de-duplicated); otherwise about-topic only.
    example_queries = (plan.get("example_queries") or []) if example_mode else []
    about_queries = plan.get("bm25_queries") or []
    bm25_queries = list(dict.fromkeys([*example_queries, *about_queries])) or [query]

    # 2. In-region BM25 fan-out over all warm datasets (one proxy call).
    emit("retrieve", bm25_queries=bm25_queries)
    s = time.time()
    datasets = worker_manager.datasets_in(SEARCH_REGION)
    retrieved = worker_manager.bm25_multi(SEARCH_REGION, datasets, bm25_queries, k=BM25_K)
    candidates = (retrieved.get("hits") or [])[:CANDIDATE_CAP]
    timing["retrieve"] = round(time.time() - s, 2)

    if not candidates:
        return {
            "query": query,
            "plan": plan,
            "results": [],
            "candidate_count": 0,
            "used_datasets": retrieved.get("used_datasets", []),
            "pending_datasets": retrieved.get("pending_datasets", []),
            "timing": timing,
            "elapsed": round(time.time() - t0, 2),
        }

    for c in candidates:
        c["url_h"] = str(_url_h(c.get("url") or "")) if c.get("url") else None

    # 3. Enrich each candidate with a fuller snippet from the dataset it was
    # retrieved via, so the reranker judges on real content (not a 200-char preview).
    emit("enrich", candidate_count=len(candidates))
    s = time.time()
    _resolve_snippets(candidates, RERANK_SNIPPET_CHARS)
    timing["enrich"] = round(time.time() - s, 2)

    # 4. Rerank on the enriched snippets. Candidates are kept in BM25-relevance
    # order by default: an A/B test showed that prior keeps results on-topic (e.g.
    # it excluded a Haskell page from an "ocaml" search that the shuffled order let
    # in), and it introduces no dataset bias — Gemini is never told which dataset a
    # candidate came from, and the dataset skew is content-driven (a reverse-order
    # probe returned the same datasets). Flip SHUFFLE_CANDIDATES to debias order.
    emit("rerank", candidate_count=len(candidates))
    s = time.time()
    order = list(range(len(candidates)))
    if SHUFFLE_CANDIDATES:
        random.Random(_seed(query)).shuffle(order)
    shuffled = [candidates[i] for i in order]
    ranked = gemini_client.rerank(
        query,
        [{"id": j, "snippet": c.get("snippet") or c.get("preview") or ""} for j, c in enumerate(shuffled)],
        model=model,
        snippet_chars=RERANK_SNIPPET_CHARS,
        prefer_examples=example_mode,
    )
    results = []
    for r in ranked[:top_k]:
        j = r.get("id")
        if not isinstance(j, int) or j >= len(shuffled):
            continue
        c = dict(shuffled[j])
        c["rerank_score"] = r.get("score")
        c["reason"] = r.get("reason")
        results.append(c)
    timing["rerank"] = round(time.time() - s, 2)

    # 4. Kept/dropped cross-reference over the local coverage keys.
    emit("crossref", result_count=len(results))
    s = time.time()
    keep_summary = _apply_crossref(results)
    timing["crossref"] = round(time.time() - s, 2)

    # 5. Pre-resolve + embed each result's extracted text (from datasets whose
    # worker is up) so "view extractions" is instant. One worker call per dataset.
    emit("resolve", result_count=len(results))
    s = time.time()
    _autoload_extractions(results)
    timing["resolve"] = round(time.time() - s, 2)

    elapsed = round(time.time() - t0, 2)
    emit("done", elapsed=elapsed)
    return {
        "query": query,
        "plan": plan,
        "example_mode": example_mode,
        "results": results,
        "candidate_count": len(candidates),
        "keep_summary": keep_summary,
        "used_datasets": retrieved.get("used_datasets", []),
        "pending_datasets": retrieved.get("pending_datasets", []),
        "timing": timing,
        "elapsed": elapsed,
    }


def _resolve_snippets(candidates: list[dict], chars: int) -> None:
    """Attach ``candidate["snippet"]`` = fuller text from the dataset it was retrieved via.

    One worker call per dataset (all its url_hs at once), in parallel; datasets
    whose worker is down keep the short BM25 preview. Best-effort — a slow/failed
    dataset just falls back to the preview.
    """
    ready = set(worker_manager.partition_datasets({c.get("dataset") for c in candidates if c.get("dataset")})[0])
    by_ds: dict[str, list[tuple[int, int]]] = {}
    for i, c in enumerate(candidates):
        ds, h = c.get("dataset"), c.get("url_h")
        if h and ds and ds in ready:
            by_ds.setdefault(ds, []).append((int(h), i))

    def _one(ds: str) -> None:
        try:
            rows = worker_manager.resolve_text(ds, [h for h, _ in by_ds[ds]], max_chars=chars, timeout=22)
            text_by_h = {r["url_h"]: r["text"] for r in rows}
            for h, i in by_ds[ds]:
                t = text_by_h.get(str(h))
                if t:
                    candidates[i]["snippet"] = t
        except Exception as e:
            logger.warning("enrich %s failed: %s", ds, e)

    if by_ds:
        with ThreadPoolExecutor(max_workers=min(6, len(by_ds))) as ex:
            list(ex.map(_one, list(by_ds)))


def _autoload_extractions(results: list[dict], max_chars: int = 8000) -> None:
    """Resolve each result's text across its kept datasets (running workers only) and embed it.

    Groups by dataset so each region worker is hit once (all url_hs at once), in
    parallel, and attaches ``result["extractions"] = {dataset: {text, ...}}``.
    """
    ready = set(worker_manager.partition_datasets({ds for c in results for ds in c.get("kept_by", [])})[0])
    by_ds: dict[str, list[int]] = {}
    for c in results:
        if not c.get("url_h"):
            continue
        for ds in c["kept_by"]:
            if ds in ready:
                by_ds.setdefault(ds, []).append(int(c["url_h"]))

    resolved: dict[tuple[str, str], dict] = {}

    def _one(ds: str) -> None:
        try:
            for row in worker_manager.resolve_text(ds, by_ds[ds], max_chars=max_chars, timeout=22):
                resolved[(row["url_h"], ds)] = row
        except Exception as e:
            logger.warning("autoload %s failed: %s", ds, e)

    if by_ds:
        with ThreadPoolExecutor(max_workers=min(6, len(by_ds))) as ex:
            list(ex.map(_one, list(by_ds)))
    for c in results:
        h = c.get("url_h")
        c["extractions"] = {ds: resolved[(h, ds)] for ds in c["kept_by"] if (h, ds) in resolved} if h else {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import json
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "ocaml tutorials"
    worker_manager.reconnect()
    out = search(q, on_progress=lambda stage, info: logger.info("stage=%s %s", stage, info))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    for r in out["results"][:8]:
        print(f"  [{r.get('rerank_score')}] {r.get('url')}")
        print(f"      kept_by={r.get('kept_by')} dropped_by={r.get('dropped_by')}")
        print(f"      {(r.get('preview') or '')[:100]!r}")


if __name__ == "__main__":
    main()
