# Paper Figures

How to (re)generate each figure in the paper. One section per figure. Run all
commands from the repo root.

## Loss vs. tokens trained, per model size

Three side-by-side panels (157M, 998M, 2.9B params) showing validation loss
vs. tokens trained for the four canonical curation methods. One figure per
metric: LIMA, Paloma macro, Uncheatable-Eval macro.

```bash
# (1) Refresh local data from gs:// — pulls summaries + LIMA sidecars into
#     scratch/fm_summaries/ and scratch/fm_lima_sidecar/. Run this when new
#     fixed-model runs have landed and you want the latest numbers.
uv run --with matplotlib --with numpy python \
    experiments/scaling_law_sweeps/plot_loss_vs_tokens_by_size.py --pull-only

# (2) Plot from local data (fast, no network). Drop --pull-only above and
#     re-run this whenever you tweak styling.
uv run --with matplotlib --with numpy python \
    experiments/scaling_law_sweeps/plot_loss_vs_tokens_by_size.py
```

You can also combine the two: `--pull` (without `-only`) refreshes data and
then plots in a single invocation.

Outputs `loss_vs_tokens__{lima_loss,paloma_macro_loss,uncheatable_macro_loss}.{png,pdf}`
in `scratch/plots/loss_vs_tokens/`.

The script's `--help` covers `--methods`, `--ours-method`, `--hidden-sizes`,
`--metrics`, and `--output-dir` for variants.

## WARC-scaling method comparison (single cell, shaded)

One single-panel figure per `(N, hidden_size, metric)` cell from the WARC-scaling
sweep, comparing five curation methods: DCLM, Resiliparse (dedup), and the three
spec-driven quality tiers (LQ/MQ/HQ, ours). Background is shaded by which method
currently has the lowest loss — a natural green→yellow→red gradient emerges
because the quality tiers take over the lead as data scales. Title is omitted on
purpose: model size and N are meant to be stated in the caption.

```bash
# (1) Refresh local data from gs:// — pulls per-run summaries into
#     scratch/warc_summaries/. Run when new WARC-scaling runs have landed.
uv run --with matplotlib --with numpy python \
    experiments/scaling_law_sweeps/plot_warc_scaling_method_comparison.py --pull-only

# (2) Plot from local data at the canonical N=100 / d=512 cell on uncheatable
#     macro loss (paper default).
uv run --with matplotlib --with numpy python \
    experiments/scaling_law_sweeps/plot_warc_scaling_method_comparison.py \
    --n-warcs 100 --hidden-size 512

# Other cells, same defaults:
uv run --with matplotlib --with numpy python \
    experiments/scaling_law_sweeps/plot_warc_scaling_method_comparison.py \
    --n-warcs 500 --hidden-size 1024

# All three metrics in one go, no shading:
uv run --with matplotlib --with numpy python \
    experiments/scaling_law_sweeps/plot_warc_scaling_method_comparison.py \
    --n-warcs 100 --hidden-size 512 \
    --metrics lima paloma uncheatable --no-shade-best
```

Outputs `warc_scaling__N{N}_d{d}__{metric_short}{_shaded}.{png,pdf}` in
`scratch/plots/warc_scaling_paper/`.

Defaults worth knowing:
- `--metrics uncheatable_macro_loss` (single metric, paper-canonical).
- `--shade-best` is **on by default**; pass `--no-shade-best` for the plain version.
- `--methods dclm resiliparse_dedup low_quality med_quality high_quality`.

The script's `--help` covers `--methods`, `--metrics`, `--n-warcs`,
`--hidden-size`, `--shade-best/--no-shade-best`, `--results-prefix`,
`--results-gs`, `--output-dir`, and `--suffix`.

## DCLM CORE table (extraction methods × model scale)

Four-row, three-column LaTeX table reporting DCLM CORE for each extraction
method at three fixed model scales (157M / 998M / 2.9B). Per-cell number is
the bootstrap mean over 1000 resamples (examples-within-task, recomputing
the centered aggregation each iter); the `±` is the bootstrap stdev. Bold
row defaults to "ours" (configurable).

The pipeline is two stages: (1) per-run bootstrap that reads per-example
sample logs from a CORE eval's partial dir and writes a small summary
JSON; (2) a Jinja2-based renderer that assembles the summaries into the
LaTeX table.

### (1) Refresh: bootstrap each run

Run this on a worker colocated with the partials' bucket region so
sample-level reads stay free (squad alone is ~21 MB per partial × 12+
runs). For partials in `gs://marin-us-central1/.../partial/<run_name>/`,
pin the job to `us-central1`. One invocation per run:

```bash
uv run iris --cluster us-central1 job run \
    --region us-central1 \
    --cpu 1 --memory 4GB --disk 5GB \
    --priority batch --no-wait \
    -- python -m experiments.scaling_law_sweeps.dclm_core.bootstrap_core \
        --run-input gs://marin-us-central1/metadata/data_curation_core_results/partial/<run_name>/ \
        --output-json gs://marin-us-central1/metadata/data_curation_core_bootstrap/<run_name>_bootstrap.json \
        --n-bootstrap 1000 --seed 0
```

For local prototyping (e.g. while a run is still 21/22 partials), pull
the partials with `gcloud storage cp -r` first and point `--run-input`
at the local copy. Note: `Core_v2` is `N/A` until all 22 tasks are
present (matches deterministic behavior); the bootstrap output still
populates `per_task_observed_mean` and `per_task_bootstrap_stdev` for
sanity-checking variance scales.

CLI escape hatch when prototyping a metric swap without editing
`task_mapping.py`: `--metric-override boolq=acc_norm` (repeatable).

### (2) Render: assemble the table

Pull all per-run bootstrap JSONs locally, then render. Free pull from
the bootstrap bucket is small (~few KB per run):

```bash
mkdir -p scratch/bootstrap
gcloud storage cp \
    "gs://marin-us-central1/metadata/data_curation_core_bootstrap/*_bootstrap.json" \
    scratch/bootstrap/

uv run --with jinja2 --with pyyaml python \
    -m experiments.scaling_law_sweeps.dclm_core.render_core_table \
    --layout experiments/scaling_law_sweeps/dclm_core/layouts/dclm_core_3000warc.yaml
```

Outputs:
- `scratch/dclm_core_table.tex` — paper-ready table
- `scratch/dclm_core_table.csv` — one row per cell with raw bootstrap
  mean/stdev, the derived `run_name`, and the `is_bold` decision

### Layout YAML

`experiments/scaling_law_sweeps/dclm_core/layouts/dclm_core_3000warc.yaml`
defines rows (methods + display labels + group + bold-marker) and
columns (header + `hidden_size` + `budget`). The renderer derives each
cell's `run_name` deterministically via the launcher convention:

```
curation-{method}-{experiment_tag}-{budget:.0e}-d{hidden_size}-L{L}-B{B}
```

where `(L, B)` come from
`fixed_model_plan._candidate_for_fixed_model(hidden_size, budget)`. If a
`(hidden_size, budget)` cell was planner-rejected (`_candidate_for_fixed_model`
returns `None`), the renderer fails loud rather than silently mis-naming.

To add a column (new model scale) or a row (new extraction method),
edit the YAML — no code changes. To change the bolding rule, set
`bold_mode: ours | max | none`. To swap the caption or label, edit the
`caption:` and `label:` fields. The Jinja2 template lives at
`experiments/scaling_law_sweeps/dclm_core/templates/dclm_core_table.tex.j2`
if the table structure itself needs to change (e.g. extra metric column).

## Quality-tier winner regions (LQ / MQ / HQ across (N, FLOPs))

Three side-by-side panels comparing the three quality tiers (LQ, MQ, HQ) over
the (parameters N, compute C = 6·N·D) plane, one panel per WARC budget
(100 / 500 / ~8M, "Common Crawl"). Each cell is colored by which method has
the lowest predicted loss under the Option B fit. Cells where the bootstrap
consensus winner is uncertain (< 95% across 1000 resamples) render as a 50/50
blend of the top-1 and top-2 methods (peach for LQ/MQ, sage for MQ/HQ),
giving an at-a-glance trust-region view. Boundary lines mark the transition
between optimal methods; the 8M panel also draws the Chinchilla-optimal line
(D = 20·N) for reference.

The pipeline is three stages: (1) refresh the gold CSV; (2) launch the
Option B bootstrap on Iris (Zephyr fan-out); (3) aggregate the per-shard
results locally, then render.

### (1) Refresh the gold (pull latest landed runs)

```bash
# Rebuild the streamlined CSV from the newest result JSONs in GCS.
uv run python experiments/scaling_law_sweeps/export_csv.py --refresh

# Upload it as the gold the cluster workers read (the lib filters to the
# 3 tiers internally; row count in the filename is just a sanity tag).
gcloud storage cp scratch/exports/warc_scaling_streamlined.csv \
  gs://marin-us-central1/scratch/bootstrap_optionB/gold_298rows.csv
```

### (2) Launch the bootstrap on Iris

```bash
TS=$(date +%s)
uv run iris --cluster marin job run \
  --region us-central1 --cpu 2 --memory 3GB --disk 8GB \
  --priority interactive --no-wait --job-name boot-optionB-full \
  -- python experiments/scaling_law_sweeps/run_bootstrap_optionB_zephyr.py \
     --n-bootstraps 1000 --max-workers 100 --n-restarts 100 \
     --output gs://marin-us-central1/scratch/bootstrap_optionB/run_$TS
```

Note the printed `run_$TS` path. The coordinator computes the central fit
(~7 min) then fans out; ~1 hour wall at ~15 shards/min. Monitor by polling
the shard count:

```bash
gcloud storage ls gs://marin-us-central1/scratch/bootstrap_optionB/run_$TS/ \
  | grep -c seeds-
```

A smoke run first (`--n-bootstraps 8 --max-workers 8 --n-restarts 20`)
validates the cluster path in ~3 min before committing to the full 1000.

### (3) Aggregate into the trust npz (local, after all 1000 shards land)

```bash
uv run python experiments/scaling_law_sweeps/aggregate_bootstrap_optionB.py \
  --run gs://marin-us-central1/scratch/bootstrap_optionB/run_<TS> \
  --out scratch/plots/bootstrap_results/optionB_bootstrap_zephyr.npz
```

Prints the per-panel consensus + %-uncertain and writes the npz with the
central fit (`central_params_json`, `central_tpw_json`) AND the bootstrap
trust/consensus grids the figure consumes. Embedding the central fit in the
npz is what keeps the rendered boundary line and the bootstrap winners
self-consistent — the plot script reuses both via `--central-from-npz`
instead of refitting locally on a CSV that may have drifted since.

### (4) Render the figure (canonical V3 args)

V3 is the paper-locked figure version. It's frozen at
`experiments/scaling_law_sweeps/plot_quality_winner_regions__optionB_trust_blend__V3_2026_05_25.py`
(see `.agents/projects/blend_v3_2026_05_25_milestone.md`). Render either
that snapshot (reproducible verbatim) or the live script with V3 args:

```bash
uv run --with matplotlib --with numpy --with pandas --with scipy --with plotly python \
  experiments/scaling_law_sweeps/plot_quality_winner_regions__optionB_trust_blend__V3_2026_05_25.py \
  --warcs 100 500 7925398 \
  --log-n-ranges 7.8,10 7.8,10.699 7.8,13 \
  --log-c-ranges 17,21 17,22 17,30 \
  --min-tokens 4e7 \
  --bootstrap-npz scratch/plots/bootstrap_results/optionB_bootstrap_zephyr.npz \
  --central-from-npz \
  --trust-threshold 0.95 \
  --smooth-sigma 2.0 \
  --boundary-min-logc none none 22.5 \
  --chinchilla-panels 3 \
  --suffix V3
```

Outputs `quality_winner_regions__OPTIONB_TRUST_BLEND__W{w1}_W{w2}_W{w3}__V3.{png,pdf}`
in `scratch/plots/quality_winner_regions/`.

Render is cache-hit (~5 s) if `--central-from-npz` is set (no refit). Without
`--central-from-npz` the script refits locally from
`scratch/exports/warc_scaling_streamlined.csv` and caches the result in
`scratch/plots/fit_cache/`. Cache key includes the data hash + the source
of `_predict_optionB` and `fit_w1`, so any code edit auto-invalidates.

Knobs worth knowing:
- `--blend-mode top2_mix` (default, V2+) — pure consensus color above the
  trust threshold, 50/50 of top-1 and top-2 below. `weighted` reverts to V1's
  full RGB mix.
- `--trust-threshold` controls the binary cutoff (0.95 in V3). Lower (0.80,
  0.90) widens the confident region.
- `--smooth-sigma 2.0` Gaussian-smooths the per-method fractions before the
  argmax → kills the discrete-vote pixelation visible at small B.
- `--boundary-min-logc` hides the transition line below a per-panel
  log10 C floor (V3 uses `none none 22.5` to suppress the unfocused
  low-compute fragment in panel 3).
- `--chinchilla-panels 3` draws the Chinchilla-optimal line on panel 3 only.
- Per-panel y-axis ranges are set via `--log-c-ranges` (V3: `17,21 17,22 17,30`)
  to crop each panel to its data-relevant compute window.

### Gotchas worth remembering

- Cluster is `marin` (single multi-region), not `us-central1` — that name no
  longer resolves post-merge.
- Keep the Iris entrypoint at `--memory<4GB` and `--disk<10GB` or Iris demands
  `--enable-extra-resources`.
- Bootstrap tunables live in
  `experiments/scaling_law_sweeps/bootstrap_optionB_lib.py`:
  `WARCS_PANELS`, `LOG_N_RANGES`, `LOG_C_RANGES`, `N_GRID`, `N_RESTARTS_DEFAULT`.
- If the gold CSV changes between cluster fit and local render, the central
  fit embedded in the npz **must** be used (`--central-from-npz`); otherwise
  the boundary line and the bootstrap winners drift apart.
