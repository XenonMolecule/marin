# Runbook: launch a 3000-WARC fixed-model sweep on a new dataset

This is the recipe for the `expFM_natural` sweep — 5 model sizes × 7 compute
budgets = 35 cells at N=3000 (full corpus, natural epoching). Same shape as
`llm_curated_dedup`. Use this when you want a single-corpus
isoflop-style scaling-law fit on a new tokenized cache.

## 0. Prerequisites

You need:
- A **tokenized cache** in Levanter format under `gs://marin-{region}/tokenized/{cache_hash}/train/`
  with `input_ids/`, `shard_ledger.json`, `.stats.json`. Llama-3.1-8B
  tokenizer, seq_len=4096.
- The cache's `total_tokens` (read from `train/.stats.json`).
- Access to **at least 2 of {us-central1, us-central2, us-east5}** for the cache.
  Do NOT mirror to eu-west4 — Iris's TPU scheduler will otherwise dispatch
  v6e-{4,8,16,32} alternatives there and the children will fail with
  `ValueError: No cache available for component <method> in train split`.
  See the launch step for the hard region restriction.

## 1. Mirror the cache (~$0.02/GB intra-NA × 2 destinations)

If your cache lives in only one region, mirror it. ~177 GB cache → ~$7 total
for c2 + e5.

```bash
SRC=gs://marin-us-central1/tokenized/<cache_hash>
gcloud storage cp -r $SRC gs://marin-us-east5/tokenized/
gcloud storage cp -r $SRC gs://marin-us-central2/tokenized/

# Verify all three match (byte counts):
for region in us-central1 us-east5 us-central2; do
  gcloud storage du -s gs://marin-$region/tokenized/<cache_hash>/train/
done
```

## 2. Register the method (1 file)

Edit `experiments/scaling_law_sweeps/curation_plan.py`:

```python
# In _D_OBS_DEFAULTS:
"<cache_hash>": <total_tokens>,  # e.g., 43_644_701_678

# In METHODS:
"<method_name>": _method(
    "<method_name>",
    "<cache_hash>",
),
```

`<method_name>` is what you'll pass to `--methods` at launch and what
appears in run names / wandb / plots. Pick something short and stable
(e.g. `llm_curated_dedup`).

## 3. Wire visibility (4 files, all ~1-line edits)

| File | Add |
|---|---|
| `plot_warc_scaling_sweep.py` | `"<method_name>": "<method_name>"` to `_FM_METHOD_MAP` (keeps it as its own base name on cross-method plots) |
| `plot_curation_isoflop.py` | `"<method_name>": "#<hex>"` to `COMPARE_COLORS` (pick a color that contrasts with the existing palette) |
| `warc_scaling_dashboard.py` | append `"<method_name>"` to `FM_PROGRESS_METHODS` (Progress tab will count cells) |
| `export_csv.py` | add `"<method_name>"` to `CANONICAL_METHODS` (CSV export will include it) |
| `plot_fixed_model_sweep.py` | append `"<method_name>"` to the default `--methods` list (side-by-side plots will show it) |

You can launch without these — runs will train and write summaries — but
no dashboard / plot / CSV will surface the data until they're wired.

## 4. Launch the parent

You need WANDB_API_KEY and HF_TOKEN. **Hard-restrict regions to the 3
NA regions** so children don't dispatch to eu-west4.

```bash
uv run iris --cluster marin job run --no-wait \
  --cpu 4 --memory 3GB --priority interactive \
  --job-name <method_name>-fm \
  -e WANDB_API_KEY $WANDB_API_KEY \
  -e HF_TOKEN $HF_TOKEN \
  -- python experiments/scaling_law_sweeps/launch_fixed_model_sweep.py \
  --methods <method_name> \
  --child-priority batch \
  --force-memory-gb 64 \
  --allowed-regions us-central1 us-central2 us-east5
```

Notes:
- **Parent priority `interactive`** so the coordinator doesn't get
  preempted by other batch jobs.
- **Children priority `batch`** — they're long-running and capacity-flexible.
- **`--force-memory-gb 64`** prevents OOM kills on B=2048+ cells. The
  default 24 GiB host memory tier is too small for the larger batch sizes.
- The parent enumerates 35 plans (5 hidden_dims × 7 budgets) from
  `fixed_model_plan.enumerate_fixed_model_plans` and submits each as an
  independent TPU child running `run_curation_train_standalone.py`.

## 5. Monitor

**Iris**:
```bash
uv run iris --cluster marin job list --prefix /michaelryan/<method_name>-fm
```
States: `pending` → `running` → `succeeded`/`failed`/`killed`. Iris keeps
job records for ~24 h after completion.

**Wandb**: group `data-curation-fixed-model`, run names match
`curation-<method_name>-expFM_natural-<budget>-d<H>-L<L>-B<batch>`.

**Summaries (canonical truth)**: `gs://marin-us-central1/metadata/data_curation_fixed_model_results/curation-<method_name>-expFM_natural-*.json`.

**Dashboard** (warc-scaling dashboard surfaces FM data via `FM_PROGRESS_METHODS`):
```bash
uv run python -m experiments.scaling_law_sweeps.warc_scaling_dashboard
# Open http://localhost:8091
```

## 6. Regenerate plots

The dashboard's Plots tab regenerates `plot_warc_scaling_sweep` plots
automatically. For the 3-panel side-by-side **fixed-model** plots
(`scratch/plots/fixed_model/lima_loss/method_comparison_side_by_side_*.html`),
run separately:

```bash
gcloud storage cp 'gs://marin-us-central1/metadata/data_curation_fixed_model_results/*.json' scratch/fm_summaries/
gcloud storage cp 'gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results/*.json' scratch/fm_lima_results/
uv run python experiments/scaling_law_sweeps/plot_fixed_model_sweep.py \
  --results-prefix scratch/fm_summaries/ \
  --lima-sidecar-prefix scratch/fm_lima_results/ \
  --output-dir scratch/plots/fixed_model
```

## 7. CSV export (analysis)

```bash
uv run python experiments/scaling_law_sweeps/export_csv.py --refresh
# scratch/exports/warc_scaling_streamlined.csv  (analysis-focused, 11 cols)
# scratch/exports/warc_scaling_complete.csv     (reproducibility, 25 cols)
```

The streamlined CSV's `eval_lima_loss` column is sourced from the FM LIMA
sidecar prefix when the main summary lacks `eval/lima/loss` (older runs
predate LIMA being in the validation set). For new runs (post-2026-04-22)
LIMA is logged inline.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: No cache available for component <method> in train split` | Worker landed in a region without the cache (typically eu-west4 v6e-* alternatives) | Use `--allowed-regions us-central1 us-central2 us-east5` |
| `Exit code 137: OOM killed` on B=2048+ cells | Default 24 GiB host memory tier | `--force-memory-gb 64` |
| Loss diverges mid-training (e.g. 3.0 → 6+ → partial recovery) | Optimizer state corruption from preempt+resume race | Re-run with `--run-suffix v2` to force a fresh checkpoint dir |
| Runs marked "running" in wandb but no heartbeat for hours | Worker died without sending a final wandb event | Check `iris job list` — if the child is gone or `failed`, the wandb state is stale; relaunch the cell |
| iris parent disappears after ~24 h, children stop being managed | iris job retention timeout on the parent CPU coordinator | Re-launch a fresh parent (children survive independently as long as they're alive) |

## Surgical relaunch of specific cells

If only some cells fail, relaunch just those without re-doing healthy ones:

```bash
# Pass any number of substrings — a plan's run_name_core must contain at
# least one to be submitted.
... launch_fixed_model_sweep.py \
  --methods <method_name> \
  --filter-name-contains-any 2e+19-d512-L6-B128 9e+18-d2432-L24-B8 \
  --no-skip-if-done   # only if you want to re-run cells that already have summaries
```

Pair with `--run-suffix v2` to force a fresh checkpoint dir (avoids
resuming from a corrupted optimizer state). The summary writer dual-writes
to BOTH the canonical path AND the `-v2`-suffixed path, so cross-method
plots auto-pick up the rerun.

## Cost notes

Compute cost depends heavily on capacity contention. As a rough order of
magnitude for the full 35-cell grid on TPU v4/v5p (the AdamH heuristic
chooses `vm_count` per cell):
- Smallest cells (d=512 / B=8-32): ~10-30 min on v4-8/v5p-8.
- Largest corner (d=3584 / B≥256 / 3e+20 budget): several hours on v5p-64.
- Wall-clock for the full sweep with full TPU access: ~12-24 h. With
  capacity contention (typical), 2-4 days.

Storage egress: the 3-region mirror is ~$7 for a 177 GB cache. Cache
storage cost in 3 regions: ~$0.022/GB/month × 3 = ~$11/month for the
177 GB cache. Plan to delete the mirrors when the sweep is done.
