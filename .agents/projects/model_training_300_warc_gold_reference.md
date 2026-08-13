# Gold reference: `model-training-300-warc` → the N=3000 equivalent

Canonical end-to-end record of how a new LLM-extraction spec goes from a finished
WARC extraction to trained models with CORE v2 + OLMo bpb evals, plus the exact
N=3000 adaptation for `llm_pipeline_v1_1`.

Written 2026-08-01. Reference spec: `llm_pipeline_v1_1` (markdown-extraction
upgrade, `two_stage_v5.2+mdx_d5`).

---

## Part 1 — The N=300 gold reference (what we actually did)

Chain: **inventory → resolve → transfer → dedup → decon → tokenize → register →
train → eval**.

Completed methods on the seed-0 random 300 (`lpv1_300_done.txt`, 300 paths):

| method | cache | tokens | pin |
|---|---|---:|---|
| `llm_pipeline_v1_random_300` | `llm_pipeline_v1_decon_300warcs-b834c5` | 3,353,097,168 | us-east5 |
| `llm_pipeline_v1_1_random_300` | `llm_pipeline_v1_1_decon_300warcs-1563f5` | 4,261,414,828 | us-central1 |
| `llm_simple_v1_random_300` | `llm_simple_v1_decon_300warcs-b8a945` | 4,259,336,845 | us-central1 |
| `med_quality_random_300` | `med_quality_300warcs-b87e1b` | 2,761,298,543 | us-central1 |

Training: `launch_warc_scaling_sweep.py --methods <base> --n-warcs 300` = **36
cells** (`expWARC_natural`), 5 widths × 9 budgets minus 9 HP-rejected corners.
Evals: CORE v2 36/36 + OLMo base_easy 36/36.

---

## Part 2 — What CHANGES at N=3000

### 2.1 The sweep changes. This is the single biggest difference.

`warc_scaling_plan.WARC_COUNTS = (100, 300, 500, 1000, 2000)`. There is no 3000.
`hidden_sizes_for(3000)` and `budgets_for(3000)` both raise `ValueError`
(`warc_scaling_plan.py:262-266`, `:277-278`), so
`launch_warc_scaling_sweep.py --n-warcs 3000` **hard-fails**.

N=3000 belongs to the **fixed-model sweep** (`fixed_model_plan.py`,
`EXPERIMENT_TAG = "expFM_natural"`):

- `TARGET_HIDDEN_SIZES = (512, 1536, 2432)` — 157M / 998M / 2.9B
- `BUDGETS = (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)`
- 3 × 7 = **21 cells**, no rejections

Run name: `curation-llm_pipeline_v1_1_random_3000-expFM_natural-<budget>-d<dim>-L<L>-B<B>`

The FM sweep does no name-splitting — it takes a literal `METHODS` key and reads
N from the method's own `sampled_warcs=3000`. So **do NOT** add a 3000 entry to
`warc_scaling_plan.WARC_METHOD_BASE_NAMES`; that list is only for N ∈ WARC_COUNTS.

`plot_warc_scaling_sweep.py` splices the FM runs back in as the 3000 anchor row
(`_per_n_size_lineup`: `3000 → [512, 1536, 2432]`), which is why 3000 still
appears on the WARC-scaling ladder despite living in a different sweep.

⚠️ `RUNBOOK_FM_NEW_DATASET.md` says "5 sizes × 7 = 35 cells". **Stale** — 3328
and 3584 were dropped 2026-05-31 (`fixed_model_plan.py:50-54`). It is 21.

### 2.2 The manifest: NESTED, not the baseline 3k draw

Three 3000-path manifests exist and they are **not** the same set:

| manifest | paths | note |
|---|---:|---|
| `random_subsets/random_warcs_3000_nested.txt` | 3000 | **what lpv1_1 ran**; guarantees 100⊂300⊂500⊂1000⊂2000⊂3000 |
| `random_subsets/random_warcs_3000_independent.txt` | 3000 | byte-identical to the baseline file below |
| `subsets/baseline_warcs_3000_random.txt` | 3000 | what `dclm_random_3000` / `nemotron_full_random_3000` / `resiliparse_random_dedup_3000` were built on |

**nested ∩ baseline = 1423/3000 (47.4%).** Verified empirically: all 2998
lpv1_1 completion markers fall inside nested, zero outside. `dashboard.py:48`
maps `llm_pipeline_v1_1 → random_warcs_3000_nested.txt`.

The N=300 gold-reference set nests **300/300** into nested but only **298/300**
into the baseline draw. User decision (2026-08-01): **use nested** — it is what
actually ran and it preserves the ladder lineage.

**Residual caveat to state in any writeup:** at N=3000 the lpv1_1 point and the
dclm/nemotron/resiliparse 3k baselines sit on different (47% overlapping) uniform
seed-0 draws from the same 10,364-WARC pool. Both are unbiased samples so they
remain statistically comparable at this size, but it is NOT the same-sample
paired comparison that N=300 had.

### 2.3 The manifest must be comment-free

`dedup_extracted.py::_load_manifest_hashes` (`:123-135`) skips only *blank*
lines — `#` headers get SHA-256'd as if they were WARC paths, and the
strict-done check then `sys.exit(2)`. `random_warcs_3000_nested.txt` has 5
header lines.

Built for this run:
```bash
grep -v '^#' experiments/distill/random_subsets/random_warcs_3000_nested.txt \
  | grep -v '^[[:space:]]*$' \
  > experiments/distill/random_subsets/lpv11_3000_done.txt   # 3000 lines
```

### 2.4 Resources scale, config mostly does not

Nothing in dedup/decon/tokenize sizes resources by N. Two knobs matter:

- `--target-partition-bytes 67108864` (64 MB) — the N≥1000 value.
- `--fuzzy-ram-gb 96` — the default is 64. Peak fuzzy memory is driven by CC
  **graph density**, not corpus size, and the graph grows superlinearly with N.
  64g cleared llm_pipeline_v1 at N=300; launch 3000 at 96g. Upstream
  reshape/normalize/minhash cache and skip on relaunch, so a later bump only
  costs the fuzzy time already burned.

Expected wall clock at N=3000: **~6–8 h** total (reshape ~2 h, normalize+minhash
~1 h, fuzzy ~4 h worst case, apply ~1 h). Tokenized cache ~180 GB.

If preemption corrupts a CC iteration you get a deterministic
`Lost/corrupted structure for node`. Fix = delete `{bucket}/fuzzy/` then relaunch
with `--no-fuzzy-preemptible`. **Never fall back to exact-only dedup.**

---

## Part 3 — The N=3000 runbook

All stages `--region us-central1` (the consolidation archive is forced there by
`transfer_region.py:42` `DEST_BUCKET = "marin-us-central1"`).

### Step 0 — gate

3000/3000 in `gs://marin-us-central1/documents/baseline_llm_extraction/llm_pipeline_v1_1/_completed/`.
Registry is single-region, so a plain listing is authoritative — no cross-region
dedup needed (the 4.5% steal-mode overlap applies to batch shards, not markers).

Then **manually stop the extraction fleet** — coordinators and workers linger
after 100% and starve the next run.

### Step 1 — inventory → resolve → transfer

Must be re-run: the existing `resolved_llm_pipeline_v1_1.jsonl.gz` only covers
the N=300 inventory, and dedup exits 2 unless all 3000 hashes are present.

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run \
    --priority interactive --memory 2GB --cpu 2 --no-wait \
    --job-name "extract-inventory-coord-lpv11-${ts}" \
    -- python experiments/baseline_collection/consolidate/launch_inventory.py \
        --spec llm_pipeline_v1_1 --priority batch
# then launch_resolve.py, then launch_transfer.py (same shape)
```
Coordinators `while True: sleep(3600)` and never exit — kill them once children
are SUCCEEDED.

### Step 2 — dedup

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
    --cpu 4 --memory 16GB --disk 20GB --priority interactive \
    --extra cpu --enable-extra-resources --region us-central1 \
    --job-name "dedup-extracted-lpv11-3000-${ts}" \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
    -- python experiments/baseline_collection/dedup_extracted.py \
        --spec llm_pipeline_v1_1 --n 3000 \
        --manifest experiments/distill/random_subsets/lpv11_3000_done.txt \
        --target-partition-bytes 67108864 --fuzzy-ram-gb 96
```
Output: `gs://marin-us-central1/documents/baseline_llm_pipeline_v1_1_deduped/3000warcs/deduped/`
Done marker: `.../3000warcs/stats/dedup_stats.json`

### Step 3 — decon (n=15, DF≤10, CORE v2)

```bash
-- python experiments/baseline_collection/decon_apply.py \
   --input-path   gs://marin-us-central1/documents/baseline_llm_pipeline_v1_1_deduped/3000warcs/deduped/ \
   --decon-source gs://marin-us-central1/decontamination/dclm_core_v2/ \
   --output-path  gs://marin-us-central1/documents/baseline_llm_pipeline_v1_1_decon_deduped/3000warcs/deduped/ \
   --ngram-length 15 --df-threshold 10
```
No `--spec`/`--n` — spec and N enter only via the literal paths. Expect ~0 docs
dropped (measured genuine contamination 0.0087–0.05%).

**Decon source coverage verified 2026-08-01 — 21 files vs 22 CORE tasks is CORRECT,
not a gap.** `task_mapping.CORE_TASK_MAP` has 22 entries; the source has 21 jsonl
files. The difference is naming plus one genuine dedup:
- DCLM name → lm-eval name: `agi_eval_lsat_ar`→`agieval_lsat_ar`,
  `openbook_qa`→`openbookqa`, `squad`→`squad_completion`,
  `bigbench_*`→`bigbench_*_{generate_until,multiple_choice}` (6 tasks).
- `hellaswag` and `hellaswag_zeroshot` are the same dataset at different shot
  counts. Decon matches on eval **text**, which is identical, so one file covers
  both. That is the entire 22→21 delta.

### Step 4 — tokenize (Llama-3.1-8B, hardcoded)

```bash
-- python experiments/baseline_collection/tokenize_deduped_extracted.py \
    --spec llm_pipeline_v1_1_decon --n 3000 --region us-central1
```
`--region us-central1` is **mandatory** on both the script and the iris job — the
script default is us-east5 and would silently glob an empty prefix.

Output: `gs://marin-us-central1/tokenized/llm_pipeline_v1_1_decon_3000warcs-<hash>/`
Token count: `train/.stats.json:total_tokens`.

### Step 5 — register

`curation_plan.py`:
```python
# _D_OBS_DEFAULTS
"llm_pipeline_v1_1_decon_3000warcs-<hash>": <total_tokens>,

# METHODS
"llm_pipeline_v1_1_random_3000": _method(
    "llm_pipeline_v1_1_random_3000",
    "llm_pipeline_v1_1_decon_3000warcs-<hash>",
    sampled_warcs=3000,
    pin_region="us-central1",
),
```

### Step 6 — train (21 cells)

```bash
uv run iris --cluster marin job run --no-wait \
  --cpu 4 --memory 3GB --priority interactive \
  --job-name lpv11-3000-fm \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python experiments/scaling_law_sweeps/launch_fixed_model_sweep.py \
     --methods llm_pipeline_v1_1_random_3000 \
     --child-priority batch \
     --allowed-regions us-central1
```
Parent priority **interactive**, children **batch**, never `production`.
`--allowed-regions` must match the cache footprint or children strand.

### Step 7 — evals

**CORE v2** — the FM sweep auto-enumerates, no manifest needed:
```bash
-- python -m experiments.scaling_law_sweeps.dclm_core.launch_dclm_core_sweep \
   --methods llm_pipeline_v1_1_random_3000 \
   --results-prefix gs://marin-us-central1/metadata/data_curation_3k_core_results/ \
   --launch --child-priority batch
```

**OLMo bpb** — manifest-driven, two passes:
```bash
-- python -m experiments.scaling_law_sweeps.olmo_bpb.launch_olmo_bpb_manifest \
   --manifest experiments/core_eval_manifests/llm_pipeline_v1_1_random_3000.txt --launch --tasks all
# then
   --tasks qa_rc --merge --done-marker _qa_rc.done
```
CSV columns `run_name,region,output_path,hf_dir` (hf_dir blank → largest
`hf/step-N` auto-resolved).

Submit coordinators **backgrounded** — a foreground `iris job run` hits the 2-min
bash timeout mid workspace-bundle-upload and the job is never created. Keep total
submit concurrency ≤4 (one SSH tunnel each).

### Step 8 — plots (these will NOT pick up 3000 for free)

`plot_core_v2_random_ladder.py` and `plot_olmo_bpb_random_ladder.py` hardcode
`NS = [100,300,500,1000,2000]` and `-expWARC_natural-` in `_STEM_RE`/`NAME_RE`.
Add 3000 to `NS`/`N_COLOR` and generalize the regex+glob to accept
`expFM_natural`. In `NAME_RE`, `llm_pipeline_v1_1` must precede
`llm_pipeline_v1` in the alternation or the shorter branch wins.

Also: `_FM_METHOD_MAP` in `plot_warc_scaling_sweep.py`,
`RANDOM_3K_METHODS`/`METHOD_ALIAS`/`METHOD_COLORS` in
`plot_core_v2_by_size_10k.py`, `FM_PROGRESS_METHODS` in
`warc_scaling_dashboard.py`, `CANONICAL_METHODS` in `export_csv.py`.
`llm_pipeline_v1_1` is already in both `METHOD_STYLE` dicts (`#8c564b`, marker `X`).

---

## ACTUAL RESULTS — N=3000 run completed 2026-08-01

| Stage | Result |
|---|---|
| Extraction | 3000/3000 (registry markers, 0 outside manifest) |
| Inventory | 5/5 regions refreshed |
| Resolve | `done_warcs`=3000; 552,085 unique keys from 583,772 copies (26,415 steal dups) |
| Transfer | all 5 children terminal; 3000 WARCs in `by_region/` |
| Dedup | 936 shards, 54.1 GB (from 62.8 GB reshape, ~86% retained) |
| Decon | 936/936; removed 29.8 MB = **0.055%** |
| Tokenize | **36,648,839,862 tokens**, 25,161,812 docs |

Cache: `gs://marin-us-central1/tokenized/llm_pipeline_v1_1_decon_3000warcs-aa7070/`
(`.executor_status` = SUCCESS, ledger `is_finished` = true)

Scaling check vs N=300 (4,261,414,828 tok): 36.65B / 4.26B = **8.6× for 10× WARCs**
→ ~86% per-WARC retention relative to N=300, i.e. the expected sublinear scaling as a
bigger corpus surfaces more near-duplicate pairs. Sanity band predicted beforehand was
35–45B; landing at 36.6B confirms nothing was truncated.

Registered in `curation_plan.py`:
```python
"llm_pipeline_v1_1_decon_3000warcs-aa7070": 36_648_839_862,
"llm_pipeline_v1_1_random_3000": _method(
    "llm_pipeline_v1_1_random_3000",
    "llm_pipeline_v1_1_decon_3000warcs-aa7070",
    sampled_warcs=3000,
    pin_region="us-central1",
),
```

### Gate-design lessons from this run (reusable)

Every stage in this chain has a *proxy* signal that is correlated with completion and a
*true* terminal marker. Gating on the proxy silently corrupts downstream data:

- **inventory** — files already existed from a prior run; existence is meaningless.
  Gate on **mtime > cutoff**, per region.
- **resolve** — gate on freshness AND `done_warcs`==3000. "Fresh but short" must be a
  hard failure (dedup would otherwise exit 2 hours later, after transfer had run).
- **transfer** — ⚠ the sharpest trap. A `data-<hash>/` dir appears when its FIRST batch
  lands, so the unique-WARC count reached 3000/3000 while europe-west4 was still copying.
  Gate on **all children terminal**, not the count. (Observed live: count hit 3000 with
  `children_running=1` and eu-west4 at 2263/2281 dirs.)
- **dedup** — gate on `stats/dedup_stats.json`. Track `fuzzy/metadata/cc/it_N` for
  progress; a mid-write `it_N` directory measures write progress, NOT convergence
  (it_10 read 1.66 GB vs 3.94 GB purely because it was being written).
- **decon** — two passes; filter pass cannot start until the global DF merge finishes,
  so `filter_pass=0` while `df_pass` climbs is correct, not a stall.
- **tokenize** — gate on `train/.stats.json` (holds `total_tokens`).

Fuzzy CC convergence signature (healthy): per-iteration state stays ~constant while the
delta halves — 1.93 → 1.28 → 0.75 → 0.45 MB here, converging by ~it_10.

## Extraction tail (state at 2026-08-01)

2998/3000. The two stragglers are the known wedged tail:

- `69c0fce25586` — 1.23 GB (near pool max), 380 batches, 18 gaps (145–161, 163),
  stale `_claimed` in all 5 regions.
- `f44012907554` — 0.095 GB, 30 batches, one gap at `batch_0007`.

Unstick tool: `fill_missing_batches.py --pipeline llm_pipeline_v1_1 --manifest <file>`
— downloads the WARC once, writes only missing batches directly, bypasses
steal/claim, then writes `_done` + the `_completed` marker.

Regions with data: us-central1 2382 · eu-west4 2281 · us-east5 2179 ·
us-west4 2060 · us-east1 1396 · us-central2 0.
