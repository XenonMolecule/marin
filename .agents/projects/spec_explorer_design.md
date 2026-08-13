# Spec Explorer — Design Doc

Status: **Phase 1 shipped** (2026-07-23). Living document; expand as phases land.

## Purpose

A durable **spec-selection instrument**: help decide which data-curation extraction
method (`dclm`, `high_quality`/HQ, `nemotron_full`, `resiliparse`, `llm_pipeline_v1`,
`llm_simple_v1`, `fastpipe_v3` bands, …) is best, by comparing them across evals,
coverage overlap, extracted text, and their actual prompts. Built to keep expanding.

Lives at `experiments/spec_explorer/` (in-repo so it imports the existing query/spec
modules directly). Most analysis at the **300-WARC scale**.

## Why two tiers

Measured index/text footprint at 300-WARC scale (2026-07-23), region-pinned:

| Artifact | us-central1 | us-central2 | us-east5 | Total |
|---|---|---|---|---|
| BM25 (`bm25_indices/small`) | 22.0 GB | 1.6 GB | 10.3 GB | ~34 GB |
| URL index (`url_index/small`, mostly `text.parquet`) | 13.8 GB | 1.1 GB | 10.6 GB | ~25.5 GB |
| **Heavy total** | 35.8 | 2.7 | 20.9 | **~60 GB** |
| `keys.parquet` only (coverage inputs) | 156 MB | 18 MB | 90 MB | **~264 MB** |

10k (FULL) scale is ~34× the WARCs → `text.parquet` projected ~1–2 TB. So the heavy
data must be read **in-region**; only the ~264 MB `keys.parquet` set and small eval
metadata travel.

- **Light tier** (`app.py`): always-on local Flask. Eval plots, winners table, spec
  display, coverage matrix (from consolidated keys). No bulk data.
- **Heavy tier** (`worker.py`, Phases 3–4): in-region Iris CPU worker, fired up on
  demand from the UI, for BM25 search + extracted-text viewing. Ephemeral (CPU-only,
  torn down after use). One worker per region touched (central1/central2/east5).

Datasets → region: `dclm`,`nemotron_full`→us-central2; `high_quality`,`llm_pipeline_v1`,
`llm_simple_v1`→us-central1; `fastpipe_v3(+bands)`→us-east5.

## Dataflow & reuse (do not reimplement)

- **Eval join key** = `run_name_core` (`curation-<method>-<tag>-<budget>-d<H>-L<L>-B<B>`),
  parsed in `catalog.parse_run_name`. Eval sources:
  - loss (paloma ↓, uncheatable bpb ↓): training summary `eval` dict at fixed
    us-central1 `data_curation_{warc_scaling,fixed_model,10k_natural}_results/<run>.json`.
  - core_v2 (↑): per-region `data_curation_10k_core_results/<run>_summary.json` →
    `dclm.Core_v2` + `raw_results` subtasks (~7 KB each).
  - olmo_bpb (↓): per-region `olmo_bpb_results/<run>/results.json` (~8 KB) → macro +
    reader-side category groups (imported from `olmo_bpb_tasks_set`) + individual tasks.
  - olmes (↑): the consolidated `metadata/olmes_base_summary.csv` (macro only). **Never**
    the per-run `olmes_base_results/<run>/results.json` — those are ~98 MB (lm-eval
    per-sample dumps; ~64 GB total, mostly cross-region egress).
  - **Listing cost control**: nested result dirs listed with a `/` delimiter (enumerate
    run-dirs, not every artifact) → full scan drops from >10 min to ~20 s.
- **Specs**: `extraction_specs.SPECS` (single-prompt), `pipelines.pipeline_specs.PIPELINES`
  (+ `prompts/*.txt`, two-stage/one-call), fastpipe bands = ModernBERT keep-top-X%
  thresholds (not prompts). Assembled by `specs_service.describe_spec`.
- **Coverage** (Phase 2): `url_index.coverage.compute(keys_globs, key)` → `a_only` =
  "docs in A not B"; `key=url_h` is the universal cross-dataset key. Consolidate the
  ~264 MB keys set into a local DuckDB once. Surface `stats.json:url_match_rate` so low
  `rid_h`/`text_h` overlaps aren't misread as disjointness (can be provenance-join loss).
- **Text / lookup** (Phase 3): `url_index.lookup.lookup_url` + `fetch_texts` (routed to
  the owning region's worker).
- **BM25 search** (Phase 3–4): `infinigram.bm25_query.open_bm25_index(ds, SMALL).search`
  → hits carry `url`, `preview`, ids, score. Runs on the in-region worker.
- **Gemini** (Phase 4): reuse `run_gemini_crosscheck.gemini_backoff` REST helper; needs
  `GEMINI_API_KEY` in env (NOT in `.env` today — export before search, and pass into any
  Iris worker via `EnvironmentSpec`).
- **Iris** (Phase 3): `IrisClient.remote(...).submit(Entrypoint.from_command, ResourceSpec,
  EnvironmentSpec)`; tunnel pattern mirrored from `warc_scaling_dashboard._get_iris_client`.

## Frontend

Vanilla-JS tabbed SPA (repo convention), Plotly via CDN, "laboratory-instrument"
aesthetic (warm near-black, off-white ink, LOCKED method colors as the only chroma,
amber reserved for UI chrome). Tabs: Performance · Winners · Specs · Coverage · Search ·
Compare. Method colors mirror `plot_core_v2_random_ladder.METHOD_STYLE`.

## Performance budget

- Eval refresh: ~20 s, cached; explicit-refresh only.
- Search (Phase 4, user requirement): live progress bar + **<25 s** end-to-end via
  concurrent BM25 fan-out, capped rerank candidates (~40–60), gemini-flash, pre-warmed
  mmap'd worker indices.

## Open items / risks

- fastpipe naming: eval method `fastpipe_v3_100` ↔ index dataset `fastpipe_v3` (100%);
  reconcile in the coverage/search wiring.
- Datasets span 3 regions → search/text fan out to per-region workers; MVP does 2.
- Light-tier → worker tunnel auth: reuse the `iris job run` controller-address pattern.
- infini-gram exact-substring deferred (C++ toolchain); add as an Iris-worker mode later.

## Worker transport — how the light tier reaches the in-region worker (hard-won)

The worker is a plain **`http.server`** (in `worker.py`) that registers its
address as an Iris endpoint. Reaching it took a real investigation; the findings:

- The Iris **actor** system's proxy (`/iris.actor.ActorService/*`) is **not on
  the deployed controller** (image `iris-controller:latest` predates it) — a
  GET returns 404 while the sibling `/proxy/*` route returns 401 (exists). So we
  do **not** use `ActorClient`/`ActorServer`.
- The controller's **generic HTTP endpoint proxy IS deployed**:
  `GET/POST {dashboard}/proxy/<name-with-`/`-as-`.`>/<subpath>` forwards to the
  registered endpoint's address, stripping the client `Authorization` before
  forwarding. The worker runs an HTTP server; the light tier calls it through
  this proxy (`worker_manager.call_worker`).
- **Ports:** the gRPC controller (job ops: submit/status/terminate) is on
  **10000**; the dashboard + endpoint proxy is on **8080**. We tunnel to both.
- **Auth:** the dashboard proxy is auth-**optional** — send **no**
  `Authorization` header (a GCP access token is actively rejected with 401; the
  no-token path is anonymous and works).
- **Endpoint lease:** a registered endpoint is dropped after a few minutes
  unless refreshed, so the worker **re-registers every 45 s** (keepalive thread).
- **Restart safety:** `worker_manager.reconnect()` rebuilds worker handles from
  RUNNING `spec-worker-*` jobs, so restarting the light tier doesn't orphan a
  worker (state is job identity + wire name, fetched via `client.status`).
- **BM25 warm-up:** the worker pre-warms every dataset's BM25 index at startup
  (mirrors shards in-region, ~2–3 min for ~22 GB) and reports readiness in
  `/health`; searches hit warm mmap'd indices (~ms). One fan-out `bm25_multi`
  call runs all `(dataset, query)` searches in-region to stay under the proxy's
  30 s cap.

## Search — iteration log (v1 → v3)

Gemini expand → in-region `bm25_multi` → Gemini rerank → local kept/dropped
cross-ref over the cached keys. Latency iterations, each measured on real
queries:

- **v1:** ~40 s. Rerank ranked *all* 60 candidates with reasons.
- **v2:** ~24 s, but truncated JSON on some queries (`flat earth`).
- **v3 (shipped):** **~10 s.** Root cause of both slowness *and* truncation was
  gemini-2.5-flash **thinking tokens** eating the output budget — set
  `thinkingConfig.thinkingBudget = 0` (rerank 27 s → 2.3 s). Also: reranker
  returns only the top ~15 (not a full ranking), candidates capped at 40,
  snippets ≤260 chars, and parse failures fail fast (no 5× retry).
  `search_cache.py` (SQLite) caches every search with favorites.

## Phase status

1. **Done** — Performance plots, winners table, spec display.
2. **Done** — Coverage Jaccard/containment matrix + set-difference sampling.
3. **Done** — In-region HTTP worker + on-demand Iris launch + text viewer
   (reached via the controller endpoint proxy; not the actor proxy).
4. **Done** — Gemini edge-case search (progress bar, ~10 s) + cache/favorites.
5. **Done** — Head-to-head compare view (eval deltas + coverage + spec diff +
   sample docs).
6. Later — infini-gram exact-substring mode; FULL (10k) scale; fastpipe search
   (its us-east5 worker); parallelize `bm25_multi` to shave the ~5 s retrieve.
