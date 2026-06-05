# Resiliparse per-N fuzzy dedup — handoff

**Goal**: produce `baseline_resiliparse_{N}_deduped` tokenized caches at
`N ∈ {100, 500, 1000, 2000, 3000}` matching the per-N dedup semantics used for the
LLM-extracted quality bands (`dedup_extracted.py`). Once tokenized, register
in `curation_plan.py` and launch training with `launch_warc_scaling_sweep.py`.

## Why per-N (and not slice the existing 3000-WARC deduped corpus)

Fuzzy dedup is N-dependent. At `N=100` we only want to drop duplicates *within*
those 100 WARCs. If we slice the existing 3000-WARC deduped output to first-100,
we'd be removing docs that got dedup'd against the *other* 2900 WARCs — those
docs would have survived a true 100-WARC dedup. The LLM-extracted specs
(`low_quality`, `med_quality`, etc.) all use per-N dedup via
`dedup_extracted.py`, so resiliparse must too for fair method comparison.

## What's already done

- Raw resiliparse extraction at 3000 WARCs:
  `gs://marin-us-central2/extracted/baseline_resiliparse-19bdaa/data-*-of-03000.jsonl.gz`
  (one shard per WARC, schema `{text, url}`)
- Pre-built dedup driver: **`experiments/baseline_collection/dedup_resiliparse_warc_scaling.py`**
  (planted alongside this handoff — mostly an adapter of `dedup_extracted.py`)
- Tokenize driver works as-is for spec=resiliparse if input lives at the
  expected path (see Phase 2c below)

## Pipeline overview

```
raw resiliparse (3000 shards, us-central2)
  └── filter to first N shards (in driver)
        └── reshape → normalize → minhash → fuzzy → apply
              └── deduped/data-*-of-YYYY.jsonl.gz (us-central2)
                    └── (mirror to us-central1)
                          └── tokenize (us-central1)
                                └── mirror to us-central2/us-east5/eu-west4
                                      └── register in curation_plan.py
                                            └── train via launch_warc_scaling_sweep.py
```

## Phase-by-phase plan

### Phase 1 — Run per-N dedup (PRIMARY WORK)

For each `N ∈ {100, 500, 1000, 2000, 3000}`:

```bash
ts=$(date +%s)
# target-partition-bytes: 16MB for 100W, 32MB for 500W, 64MB for 1k+ (see header)
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
    --cpu 4 --memory 16GB --disk 20GB \
    --priority interactive \
    --extra cpu --enable-extra-resources \
    --region us-central2 \
    --job-name "dedup-resiliparse-${N}warcs-${ts}" \
    -e WANDB_API_KEY ${WANDB_API_KEY} \
    -e HF_TOKEN $HF_TOKEN \
    -- python experiments/baseline_collection/dedup_resiliparse_warc_scaling.py \
        --n ${N} --target-partition-bytes 33554432   # adjust per N
```

**Key choices baked into the driver**:
- Output goes to `gs://marin-us-central2/documents/baseline_resiliparse_deduped/{n}warcs/`
  to match input region (avoids cross-region reads during dedup).
- `worker_resources=ResourceConfig(ram="64g")` on the fuzzy step (matches
  `dedup_extracted.py` after we hit OOM at 32g on `med_low_quality`).
  At `N≥1000` for resiliparse expect to bump to 96g or 128g — resiliparse is
  high-retention, so the LSH key space grows fast. Edit the
  `compute_fuzzy_dups_attrs_step(worker_resources=...)` call if you see
  `ZephyrWorkerError: Worker job terminated permanently (all retries
  exhausted). Workers likely crashed (OOM...)` in parent logs. Cached
  upstream stages skip on relaunch.

**Wall clock estimate** (cluster compute is free; this is just elapsed time):
| N    | Reshape | Normalize+Minhash | Fuzzy (worst case) | Apply | Total |
|------|---------|-------------------|--------------------|-------|-------|
| 100  | 15 min  | 15 min            | 30 min             | 5 min | ~1 hr |
| 500  | 30 min  | 20 min            | 45 min             | 10 min| ~1.5 hr |
| 1000 | 45 min  | 30 min            | 90 min             | 20 min| ~2.5 hr |
| 2000 | 90 min  | 45 min            | 150 min            | 30 min| ~4 hr |
| 3000 | 120 min | 60 min            | 240 min (multi-pool CC) | 60 min | ~6–8 hr |

All 5 can run in parallel — independent jobs.

**Verify after each run**: check `dedup_stats.json` lands at
`gs://marin-us-central2/documents/baseline_resiliparse_deduped/{n}warcs/stats/dedup_stats.json`.

### Phase 2 — Copy deduped output to us-central1 (so tokenize can read it)

```bash
N=...  # 100, 500, etc.
gcloud storage cp -r \
    "gs://marin-us-central2/documents/baseline_resiliparse_deduped/${N}warcs/deduped" \
    "gs://marin-us-central1/documents/baseline_resiliparse_deduped/${N}warcs/deduped"
```

Why this intra-US copy: the tokenize script reads its input from us-central1
by convention; running tokenize in us-central2 would work but mixes regions.
The deduped output is ~5–140 GB depending on N (much smaller than raw
resiliparse — most duplicates get dropped). Intra-US egress is effectively
free.

### Phase 2c — Tokenize (reuses existing `tokenize_deduped_extracted.py`)

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
    --cpu 2 --memory 8GB --disk 10GB \
    --priority interactive \
    --extra cpu --enable-extra-resources \
    --region us-central1 \
    --job-name "tokenize-resiliparse-${N}warcs-${ts}" \
    -e WANDB_API_KEY ${WANDB_API_KEY} \
    -e HF_TOKEN $HF_TOKEN \
    -- python experiments/baseline_collection/tokenize_deduped_extracted.py \
        --spec resiliparse --n ${N}
```

The existing tokenize script reads from
`gs://marin-us-central1/documents/baseline_resiliparse_deduped/{n}warcs/deduped/data-*.jsonl.gz`
which matches the path established in Phase 2 (no code change needed). Output:
`gs://marin-us-central1/tokenized/resiliparse_{n}warcs-{cache_hash}/`.

### Phase 3 — Mirror tokenized cache to all training regions

```bash
N=...
CACHE_DIR=$(gcloud storage ls gs://marin-us-central1/tokenized/ \
    | grep "resiliparse_${N}warcs-" | head -1 | sed 's|.*/tokenized/||; s|/$||')
for dst in \
    gs://marin-us-central2/tokenized/${CACHE_DIR} \
    gs://marin-us-east5/tokenized/${CACHE_DIR} \
    gs://marin-eu-west4/tokenized/${CACHE_DIR}; do
  gcloud storage cp -r "gs://marin-us-central1/tokenized/${CACHE_DIR}" "${dst}" &
done
wait
```

Cost: dominated by eu-west4 (~$0.12/GB cross-continent). See cost table below.

### Phase 4 — Register in `curation_plan.py`

Add to `_D_OBS_DEFAULTS` (token count from
`gs://marin-us-central1/tokenized/resiliparse_${N}warcs-{hash}/train/.stats.json`):

```python
    "resiliparse_100warcs-XXXXXX": 1_234_567_890,
    "resiliparse_500warcs-XXXXXX": 4_321_098_765,
    "resiliparse_1000warcs-XXXXXX": 9_876_543_210,
    "resiliparse_2000warcs-XXXXXX": 19_876_543_210,
    "resiliparse_3000warcs-XXXXXX": 29_876_543_210,
```

Add to `METHODS`:

```python
    "resiliparse_100_dedup": _method("resiliparse_100_dedup", "resiliparse_100warcs-XXXXXX", sampled_warcs=100),
    "resiliparse_500_dedup": _method("resiliparse_500_dedup", "resiliparse_500warcs-XXXXXX", sampled_warcs=500),
    "resiliparse_1000_dedup": _method("resiliparse_1000_dedup", "resiliparse_1000warcs-XXXXXX", sampled_warcs=1000),
    "resiliparse_2000_dedup": _method("resiliparse_2000_dedup", "resiliparse_2000warcs-XXXXXX", sampled_warcs=2000),
    "resiliparse_3000_dedup": _method("resiliparse_3000_dedup", "resiliparse_3000warcs-XXXXXX", sampled_warcs=3000),
```

Note the `_dedup` suffix on the method name to disambiguate from the
existing non-deduped `resiliparse_{N}` methods registered already. Also add
`"resiliparse_dedup"` (or the base name you pick) to
`WARC_METHOD_BASE_NAMES` in `experiments/scaling_law_sweeps/warc_scaling_plan.py`
if not present, plus a color entry in
`experiments/scaling_law_sweeps/plot_curation_isoflop.py:COMPARE_COLORS` so
plots distinguish dedup from non-dedup resiliparse.

### Phase 5 — Launch training

Per spec, per N:

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
    --priority interactive \
    --memory 4GB --cpu 4 --enable-extra-resources \
    --region us-central1 \
    --job-name "curate-train-coord-resiliparse_${N}_dedup-${ts}" \
    -e WANDB_API_KEY ${WANDB_API_KEY} \
    -e HF_TOKEN $HF_TOKEN \
    -- python experiments/scaling_law_sweeps/launch_warc_scaling_sweep.py \
        --methods resiliparse_dedup \
        --n-warcs ${N} \
        --child-priority interactive
```

⚠️ **DO NOT set `MARIN_PREFIX`** on the coord — it pollutes children's
region detection (see general curation_playbook.md for the bug story).
Children should auto-detect region from VM metadata.

## Cost summary

Cluster compute is free on Marin. The only real cost is the eu-west4 mirror
(cross-continent egress ~$0.12/GB):

| N    | Tokenized cache | eu-west4 mirror egress |
|------|----------------:|-----------------------:|
| 100  | ~5 GB           | ~$0.60                 |
| 500  | ~21 GB          | ~$2.50                 |
| 1000 | ~45 GB          | ~$5.50                 |
| 2000 | ~90 GB          | ~$11                   |
| 3000 | ~140 GB         | ~$17                   |
| **Total** | **~300 GB** | **~$37** |

The intra-US copy from us-central2 → us-central1 (Phase 2) is essentially
free (~$0.02/GB but typically waived).

## Smoke-test recommendation

**Before launching 1k/2k/3k, do N=100 end-to-end** (Phase 1 → Phase 5).
This validates the script port from `dedup_fuzzy_document` (deleted API) to
`compute_minhash_attrs_step + compute_fuzzy_dups_attrs_step` (current API).
~1 hour wall clock; if it works, the larger N's are mechanical.

Things to verify after the N=100 smoke:
- [ ] `stats/dedup_stats.json` shows reasonable retention (resiliparse
      typically drops 40-60% of docs via fuzzy dedup at scale, so dedup
      output should be smaller than raw input)
- [ ] `deduped/data-*-of-*.jsonl.gz` files exist and are non-empty
- [ ] `train/.stats.json` after tokenize has `total_tokens > 0`
- [ ] After training, result JSONs appear at
      `gs://marin-us-central1/metadata/data_curation_warc_scaling_results/curation-resiliparse_100_dedup-expWARC_natural-*.json`

## Pointers

- **Driver script**: `experiments/baseline_collection/dedup_resiliparse_warc_scaling.py`
  (planted alongside this doc — mostly a copy of `dedup_extracted.py` with the
  upstream input swapped for raw resiliparse and the manifest-filter step
  swapped for "first N shards of the 3000-shard tree")
- **Reference implementation**: `experiments/baseline_collection/dedup_extracted.py`
  (LLM-extracted specs — newer API, working today)
- **Old (broken) reference**: `experiments/baseline_collection/dedup/dedup_resiliparse.py`
  — depends on the deleted `dedup_fuzzy_document` symbol, won't run as-is.
  Useful only for cross-checking that the fuzzy params and apply-step logic
  match (they're identical in the new driver).
- **General playbook**: `.agents/projects/curation_playbook.md` covers the
  shared phases (mirror, register, train) that apply equally to resiliparse
  and LLM-extracted specs.
