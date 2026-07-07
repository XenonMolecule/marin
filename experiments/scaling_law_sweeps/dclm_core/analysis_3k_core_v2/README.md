# 3k Core_v2 rerun — scale vs. sampling analysis

Rescued from the cloud VM (scratch/ and /tmp never sync home). Source-of-truth
results live in GCS; the JSONs here are derived caches so analysis can start
without re-pulling.

## Final result

The 10k Core_v2 lift over the 3k biased sample is **mostly scale, with a real
but secondary head-biased-sampling effect** (scale wins ~2–4×). Baseline rank
DCLM ≥ Nemotron ≥ Resiliparse is stable across all three regimes (random-3k,
biased-3k, 10k). See `core_v2_scale_vs_sampling.html` for the full report.

## Files

- `core_v2_scale_vs_sampling.html` — the report (also published as a claude.ai artifact).
- `threeway.json` — `{method: {width: {biased3k|random3k|tenk: Core_v2*100}}}`. The analysis-ready table.
- `all_scores_3k.json` — 154 rows: `{method, width, budget, core_v2, run}` for the 3k rerun.
- `all_scores_10k.json` — same schema for the 10k sweep.
- `run_region.json` — run_name → region map.

## Source of truth (GCS, region-independent)

- 3k Core_v2 summaries: `gs://marin-us-central1/metadata/data_curation_3k_core_results/{run_name}_summary.json` (154 files). Field: `d["dclm"]["Core_v2"]` (×100 for tables).
- 10k Core_v2 summaries: `gs://marin-us-central1/metadata/data_curation_10k_core_results/` (191 files, methods suffixed `_10k`).

## Re-run

```
iris --cluster marin job run --no-wait --cpu 2 --memory 3GB --disk 9GB --region us-east5 \
  --job-name <name> -e WANDB_API_KEY <key> -e HF_TOKEN <tok> -- \
  python -m experiments.scaling_law_sweeps.dclm_core.launch_dclm_core_sweep \
  --methods <9 methods> \
  --results-prefix gs://marin-us-central1/metadata/data_curation_3k_core_results/ \
  --samples-prefix gs://marin-us-central1/tmp/ttl=30d/dclm_3k_core/ \
  --no-log-samples --child-priority batch --wave-size 20 --wave-delay 60 \
  --keepalive-max 86400 --launch
```

Add `--hidden-sizes W --region-float` to float a width; `--cells B:W` to target
cells. Always `export SSL_CERT_FILE=$(.venv/bin/python -m certifi)` first.
Proven recipe = datasets-OFFLINE + model-ONLINE + retry; do **not** add
`HF_HUB_OFFLINE`.
