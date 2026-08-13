# Spec Explorer

An interactive dashboard for choosing the best data-curation **spec** (extraction
method) by comparing methods across evals, coverage, extracted text, and their
actual prompts — the durable spec-selection instrument.

Most analysis runs at the **300-WARC scale**. See the design doc:
[`.agents/projects/spec_explorer_design.md`](../../.agents/projects/spec_explorer_design.md).

## Run it

```bash
uv run python -m experiments.spec_explorer.app     # -> http://localhost:8096
```

First load shows empty plots. Click **refresh evals** (top-right) to scan GCS and
build the eval cache (~20s; cached to `cache/eval_matrix.json`, so later loads are
instant). The scan reads only small metadata JSON/CSV — no bulk data leaves its
region (OLMES uses the consolidated `olmes_base_summary.csv`, never the 98 MB
per-run lm-eval dumps).

## Architecture (two-tier hybrid)

- **Light tier** — this Flask app (`app.py`), always-on and local. Serves the SPA
  and read-only `/api/*` endpoints for eval plots, the winners table, spec display,
  and (Phase 2) the coverage matrix from the tiny consolidated `keys.parquet` set.
- **Heavy tier** (Phases 3–4) — an in-region Iris CPU worker, fired up on demand
  from the UI, for BM25 search and extracted-text viewing (the index/text data is
  ~60 GB at 300 scale, region-pinned, so it is read where it lives, never mirrored).

## Modules

| File | Role |
|---|---|
| `catalog.py` | Shared constants: regions, method colors/labels, model sizes, dataset→region map, `run_name_core` parsing. |
| `eval_aggregate.py` | Scans all eval families → normalized `(method, budget, dim, n, family, task, value)` rows. |
| `specs_service.py` | Assembles display-ready specs from the single-prompt, pipeline, threshold, and external registries. |
| `app.py` | Flask backend: cache+refresh, `/api/*`, static SPA. |
| `static/` | Tabbed SPA (`index.html`, `app.js`, `styles.css`), Plotly via CDN. |
| `cache/` | Runtime cache (gitignored). |

## Eval families

`paloma` (val loss ↓), `uncheatable` (bpb ↓), `core_v2` (DCLM Core v2 + subtasks ↑),
`olmo_bpb` (macro + Code/Math/QA categories + individual tasks ↓), `olmes` (macro ↑).
All joined on `run_name_core`.

## Tabs (all live)

- **Performance** — interactive Plotly crossover plots per eval family, with
  hoverable 1/2-epoch marks and unified hover.
- **Winners** — direction-aware winners table, best-per-column starred.
- **Specs** — the actual prompts/thresholds for every method, phase-tabbed.
- **Coverage** — Jaccard / containment heatmap (toggle datasets, fastpipe off by
  default) + set-difference explorer with side-by-side extracted text.
- **Search** — Gemini expand → in-region BM25 → Gemini rerank → kept/dropped
  cross-ref, live progress, ~10s, with saved searches + favorites.
- **Compare** — head-to-head A-vs-B: eval deltas, corpus overlap/containment,
  spec diff, and sample docs.

## The in-region worker

`worker.py` runs as an on-demand Iris CPU job (launch/stop from the Coverage
tab, or it auto-reconnects). It holds all BM25 indices + `text.parquet` in
us-central1 and is reached through the controller's HTTP endpoint proxy — see
the design doc's "Worker transport" section for the (non-obvious) details:
port 8080, no auth header, and a light-tier keepalive that re-registers the
endpoint every 40s. It pre-warms BM25 at startup (~2-3 min); `/health` reports
readiness.

The worker runs on a **preemptible** node; the light tier's keepalive + an
overnight monitor (`cache/`-adjacent) re-launch it if it dies.
