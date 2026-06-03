# Curation playbook: extraction → tokenized training dataset

How to take a finished LLM extraction at `(spec, N)` and turn it into a
tokenized dataset registered for training. Use this every time a new
`(spec, N)` is ready: e.g. med_low_quality at 100W, every spec at 500W, 1k,
3k, etc.

Throughout, replace `{spec}` (e.g. `high_quality`) and `{n}` (e.g. `100`,
`500`, `1000`, `3000`) with your target.

---

## Quick orientation (read first)

- **`{n}` is a stable prefix**, not an arbitrary subset. `--n 500` = first
  500 WARCs of `experiments/distill/baseline_warcs_3000.txt`. So
  N=500 ⊃ N=100. Tokenized caches for different N's are independent though
  — each gets its own content-hashed cache directory.
- **Recommended sequence**: do **N=500 end-to-end first** before
  parallelizing 1k/2k. Gives one calibration cycle on worker memory and
  wall clock, then 1k/2k can launch with confidence.
- **Monitoring** during/after:
  - Results JSONs: `gs://marin-us-central1/metadata/data_curation_warc_scaling_results/curation-{spec}_{n}-*.json`
  - WandB: project `marin`, entity `marin-community`, group `data-curation-warc-scaling`
  - Dashboard: `uv run python -m experiments.scaling_law_sweeps.warc_scaling_dashboard`
    → http://localhost:8091 (Progress, Region, Plots tabs)
- **iris CLI listings are truncated**. To see all children of a coord:
  `iris --config lib/iris/examples/marin.yaml job list --prefix /michaelryan/curate-train-coord-...`
  Without the prefix flag, iris only shows the most recent ~50–100 jobs
  cluster-wide, which often hides your training children behind extraction
  workers.

---

## low_quality-specific notes

- **Legacy path**: low_quality was the first spec extracted (before the
  multi-spec refactor) so its data lives at the *unprefixed*
  `gs://marin-{region}/documents/baseline_llm_extraction/data-*/` —
  unlike `med_quality`, `high_quality`, etc. which are under
  `documents/baseline_llm_extraction/{spec}/data-*/`. The Phase 1
  inventory script already handles this (the `if spec == "low_quality"`
  branch in the prereq check). Any hand-rolled GCS scan for low_quality
  must NOT add the `{spec}/` segment.
- **Phase 1 may already be done.** Before re-running consolidation, check
  if `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/resolved/resolved_low_quality.jsonl.gz`
  exists. If yes, Phase 1 has been staged already; only re-run if you need
  to pick up incrementally-extracted WARCs.
- **Phase 2 memory at N≥500**: low_quality is the *highest-retention* spec
  (least content gets minhash-deduped out, so the LSH key space is
  largest). The current 64g `compute_fuzzy_dups_attrs_step` default barely
  cleared med_low_quality at N=100. **At N=500+ for low_quality, expect to
  bump fuzzy worker ram to 96g or 128g on first try** — saves a 2-hour
  OOM-and-relaunch cycle. Edit
  `experiments/baseline_collection/dedup_extracted.py`'s
  `compute_fuzzy_dups_attrs_step(worker_resources=ResourceConfig(...))`.
- **Cost ballpark per N** (compute is free on Marin — only the
  cross-region egress for the eu-west4 mirror copy costs anything;
  intra-US mirror hops are essentially free):

  | N    | Tokenized cache | eu-west4 mirror egress |
  |------|----------------:|-----------------------:|
  | 100  | ~6 GB           | ~$0.75                 |
  | 500  | ~30 GB          | ~$4                    |
  | 1000 | ~60 GB          | ~$7                    |
  | 2000 | ~120 GB         | ~$15                   |

---

## Prerequisites

All N priority WARCs must be **`_done` somewhere across regions** before
starting. Quick check:

```bash
.venv/bin/python <<'EOF'
import hashlib
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage as gcs_storage
from rigging.filesystem import REGION_TO_DATA_BUCKET

SPEC = "med_low_quality"   # ← change
N = 100                     # ← change

priority = set()
with open("experiments/distill/baseline_warcs_3000.txt") as f:
    for i, line in enumerate(f):
        if i >= N: break
        line = line.strip()
        if line: priority.add(hashlib.sha256(line.encode()).hexdigest()[:12])

REGIONS = list(REGION_TO_DATA_BUCKET.values())
client = gcs_storage.Client()
def count(spec):
    prefix = "documents/baseline_llm_extraction/" if spec == "low_quality" else f"documents/baseline_llm_extraction/{spec}/"
    glob = f"{prefix}data-*/_done"
    found = set()
    def scan(b):
        s = set()
        try:
            for blob in client.bucket(b).list_blobs(match_glob=glob):
                p = blob.name[len(prefix):].split("/")
                if len(p) >= 2 and p[0].startswith("data-") and p[1] == "_done":
                    h = p[0][5:]
                    if len(h) == 12: s.add(h)
        except Exception: pass
        return s
    with ThreadPoolExecutor(max_workers=6) as pool:
        for s in pool.map(scan, REGIONS): found |= s
    return found & priority

done = count(SPEC)
print(f"{SPEC}: {len(done)}/{N} priority done")
print(f"missing: {sorted(priority - done)[:20]}")
EOF
```

Don't proceed until all N priority WARCs report `_done`.

---

## Phase 1 — Consolidation

Phase 1 is **idempotent and per-spec** — staged once per spec, reused for
every `(spec, N)` pair thereafter. If you already ran Phase 1 for this spec
at a smaller N, just rerun the same launchers to pick up incremental WARCs.

### 1a. Inventory all 5 regions

Each region scans its `documents/baseline_llm_extraction/{spec}/data-*/`
tree, decompresses every batch to count records + sum chars (~10-20 min
per region in parallel).

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run \
    --priority interactive --memory 2GB --cpu 2 --no-wait \
    --job-name "extract-inventory-coord-{spec}-${ts}" \
    -- python experiments/baseline_collection/consolidate/launch_inventory.py \
        --spec {spec} --priority batch
```

Output: `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/inventories/inventory_{region}_{spec}.jsonl.gz` × 5.

### 1b. Resolve duplicates (pick canonical per batch + emit done-WARC list)

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run \
    --priority interactive --memory 2GB --cpu 2 --no-wait \
    --job-name "extract-resolve-coord-{spec}-${ts}" \
    -- python experiments/baseline_collection/consolidate/launch_resolve.py \
        --spec {spec} --priority batch
```

Output:
  - `resolved_{spec}.jsonl.gz` — canonical (warc_hash, batch_idx) → path, filtered to done-only
  - `done_warcs_{spec}.txt` — newline-delimited list of WARC hashes that have `_done` somewhere
  - `integrity_report_{spec}.json` — counts, duplicates, gaps

### 1c. Transfer non-central1 batches to us-central1 archive

Reads `done_warcs_{spec}.txt`. Only transfers batches whose WARC is fully done — no partial WARCs (those would re-transfer as they grow).

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run \
    --priority interactive --memory 2GB --cpu 2 --no-wait \
    --job-name "extract-transfer-coord-{spec}-${ts}" \
    -- python experiments/baseline_collection/consolidate/launch_transfer.py \
        --spec {spec} --priority batch
```

Output: `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/{region}/{spec}/data-{hash}/batch_*.jsonl.gz`. Cost: $1–10 depending on cross-region distribution (eu-west4 dominates).

When all 5 transfer-child jobs are SUCCEEDED, Phase 1 is complete.

---

## Phase 2 — Marin fuzzy dedup

Reshape (manifest-filter to first-N WARCs) → normalize (parquet + exact-doc
dedup) → minhash → fuzzy CC → apply (drop non-canonical near-dups). Uses
the Marin library `compute_minhash_attrs` + `compute_fuzzy_dups_attrs`
(286 perms / 26 bands / 5-char ngram / Jaccard ~0.75).

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
    --cpu 4 --memory 16GB --disk 20GB \
    --priority interactive \
    --extra cpu \
    --enable-extra-resources \
    --region us-central1 \
    --job-name "dedup-extracted-{spec}-{n}-${ts}" \
    -e WANDB_API_KEY 53bbc2cb719bfeb0684b439f6265ad6557ec0397 \
    -e HF_TOKEN $HF_TOKEN \
    -- python experiments/baseline_collection/dedup_extracted.py \
        --spec {spec} --n {n} \
        --target-partition-bytes 16777216
```

**`--target-partition-bytes` cheat sheet** (controls normalize parquet shard
count → downstream MinHash + fuzzy parallelism width):

| N    | target_partition_bytes | rationale |
|------|-----------------------:|-----------|
| 100  | 16 MB (16777216)       | small corpus, want many shards for parallelism |
| 500  | 32 MB                  | balanced |
| 1000 | 64 MB                  | default value; OK overhead |
| 3000 | 64 MB                  | default value; gives ~300-1000 shards depending on spec |

**Worker memory (2026-05-11 update)**: `compute_fuzzy_dups_attrs_step` now
defaults to `ram="64g"` (was 32g — bumped after med_low_quality OOM'd in
stage1-Reduce of fuzzy-dups). High-retention specs (low_quality and
med_low_quality, where the LSH key space scales with corpus size) at
N=500+ may need further bumps to 96g or 128g. If you see
`ZephyrWorkerError: Worker job terminated permanently (all retries
exhausted). Workers likely crashed (OOM...)`, edit
`experiments/baseline_collection/dedup_extracted.py` (the
`compute_fuzzy_dups_attrs_step` call's `worker_resources=ResourceConfig(...)`)
and relaunch. Cached upstream stages (reshape/normalize/minhash) will be
skipped on relaunch.

Output paths under `gs://marin-us-central1/documents/baseline_{spec}_deduped/{n}warcs/`:
  - `reshape/data-XXXXX-of-00200.jsonl.gz` — text-only, manifest-filtered (REUSED across re-runs at same N)
  - `normalize/outputs/main/part-XXXXX-of-YYYYY.parquet` — parquet with xxh3_128 ids
  - `minhash/outputs/<basename>.parquet` — MinHash bucket attrs
  - `fuzzy/outputs/source_000/<basename>.parquet` — per-doc {dup_cluster_id, is_cluster_canonical}
  - `deduped/data-XXXXX-of-YYYYY.jsonl.gz` — final deduped jsonl.gz (text-only) — **this is what tokenize reads**
  - `stats/dedup_stats.json` — pipeline-end summary

**Wall clock**: 30 min – 2 hours at 100W (depending on cluster contention).
Linear-ish in N up to ~1000W. At 3000W expect 4–8 hours per spec (fuzzy CC
graph size grows with corpus).

### 2a. Re-running with different params

`dedup_extracted.py` uses `override_output_path` for every step, which means
**outputs go to the same fixed paths regardless of hyperparams**. So if you
change `--target-partition-bytes`, you must manually delete the downstream
intermediates first (preserve reshape if it's still valid):

```bash
for dir in normalize minhash fuzzy deduped stats; do
  gcloud storage rm -r "gs://marin-us-central1/documents/baseline_{spec}_deduped/{n}warcs/${dir}/"
done
```

Reshape is preserved unless you also bumped the manifest or `--n`. See
task #22 for the path to fix this properly via `output_path_prefix` +
content hashing.

### 2b. Watching progress

```
spec/100warcs/
  reshape/             R=200 means reshape done
  normalize/outputs/main/  N=X parquet shards (depends on data size / partition_bytes)
  minhash/outputs/         M=N once minhash done (1:1 with normalize)
  fuzzy/metadata/cc/it_K/  K iterations of CC (up to 10)
  fuzzy/outputs/source_*/  F=N once fuzzy converges + per-shard aggregator runs
  deduped/             D=N once apply step finishes
  stats/dedup_stats.json   S=1 means pipeline COMPLETE
```

If logs show "stage1-Reduce → Scatter K/1024 complete ... W/W workers alive,
D dead", that's CC iterating. K → 1024 within minutes-to-hours per iteration.

---

## Phase 3 — Tokenize

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
    --cpu 2 --memory 8GB --disk 10GB \
    --priority interactive \
    --extra cpu \
    --enable-extra-resources \
    --region us-central1 \
    --job-name "tokenize-{spec}-{n}warcs-${ts}" \
    -e WANDB_API_KEY 53bbc2cb719bfeb0684b439f6265ad6557ec0397 \
    -e HF_TOKEN $HF_TOKEN \
    -- python experiments/baseline_collection/tokenize_deduped_extracted.py \
        --spec {spec} --n {n}
```

Output: `gs://marin-us-central1/tokenized/{spec}_{n}warcs-{hash}/train/`. The
`-{hash}` suffix is content-hashed by the executor — different (spec, n)
pairs (or rerunning after changing tokenize params) produce different
hashes. Read `train/.stats.json:total_tokens` for the final token count.

**Wall clock**: 10-30 min at 100W; up to a few hours at 3000W.

---

## Phase 4 — Mirror tokenized cache to training regions

Iris places batch-priority training children wherever TPU capacity exists
— in practice that's `us-central1`, `us-central2`, `us-east5`, and
`eu-west4`. Each child writes a region lock that pins it to that region
permanently. Any region in the mirror set must have the cache present, or
the child fails the `_assert_all_components_local` check and exits before
training starts (no egress, just wasted boot time).

```bash
CACHE_DIR={spec}_{n}warcs-XXXXXX   # ← grab from gs://marin-us-central1/tokenized/
# Mirror to all regions iris might place into. us-central1 is the source.
for dst in \
    gs://marin-us-central2/tokenized/${CACHE_DIR} \
    gs://marin-us-east5/tokenized/${CACHE_DIR} \
    gs://marin-eu-west4/tokenized/${CACHE_DIR}; do
  gcloud storage cp -r "gs://marin-us-central1/tokenized/${CACHE_DIR}" "${dst}" &
done
wait
```

**2026-05-11 update**: `eu-west4` was added to the mirror set after we
discovered iris was pinning ~9 of 11 low_quality_100 plans there with no
cache present. Cross-continent egress is ~$0.12/GB (vs $0.02/GB intra-US),
so eu-west4 dominates the bill. Cost: $1–5 per spec at 100W; ~$30–150 at
3000W with all 3 mirror regions.

---

## Phase 5 — Register in curation_plan.py

Edit `experiments/scaling_law_sweeps/curation_plan.py`:

### 5a. Add token count to `_D_OBS_DEFAULTS`

Find the existing block of `_D_OBS_DEFAULTS` entries (around lines 118-183)
and append before the closing `}`:

```python
    "{spec}_{n}warcs-XXXXXX": 1_234_567_890,   # ← from train/.stats.json:total_tokens
```

### 5b. Add a `METHODS` entry

Find the existing `METHODS` dict (around lines 233-358) and append before
the closing `}`:

```python
    "{spec}_{n}": _method(
        "{spec}_{n}",
        "{spec}_{n}warcs-XXXXXX",
        sampled_warcs={n},
    ),
```

If a future ExpC sweep should include this method, also add `{spec}_{n}` to
`EXPC_METHOD_NAMES` (around line 365).

---

## Phase 6 — Launch training

For the WARC-scaling sweep (the 2-dim grid we use for the quality bands),
**use `launch_warc_scaling_sweep.py`** — NOT the broader
`launch_curation_sweep.py` (which enumerates the full 78-plan IsoFLOP grid).
The warc-scaling launcher emits 11 plans per (spec, N): 2 hidden dims
(d256/d512) × 7 budgets (3e15..3e18) after filtering.

```bash
ts=$(date +%s)
uv run iris --config lib/iris/examples/marin.yaml job run \
    --priority interactive --no-wait \
    --memory 4GB --cpu 4 --enable-extra-resources \
    --region us-central1 \
    --job-name "curate-train-coord-{spec}_{n}-${ts}" \
    -e WANDB_API_KEY 53bbc2cb719bfeb0684b439f6265ad6557ec0397 \
    -e HF_TOKEN $HF_TOKEN \
    -- python experiments/scaling_law_sweeps/launch_warc_scaling_sweep.py \
        --methods {spec} \
        --n-warcs {n} \
        --child-priority batch
```

**⚠️ DO NOT set `MARIN_PREFIX`** on the coord (it propagates to children
and overrides VM region detection — children think they're in us-central1
when they're actually in eu-west4, fail the `_assert_all_components_local`
check, and don't run). Children should auto-detect region from GCP
metadata. See `region_tracker.detect_current_region()`.

If you want to target a subset of plans (e.g. resubmit just 2 specific
budget/dim cells), add `--only-budgets 1e17 3e17 --only-hidden-sizes 256`.

Defaults for things you can override:
  - `--wandb-project marin --wandb-entity marin-community --wandb-group data-curation-warc-scaling`
  - `--tracker-prefix gs://marin-us-central1/metadata/region_locks/data_curation_warc_scaling/`
  - `--results-prefix gs://marin-us-central1/metadata/data_curation_warc_scaling_results/`

**Child priority**: `batch` is the right default for overnight training.
Use `interactive` if you want children to outprioritize extraction TPU
workers (which run at batch). The parent coordinator stays at
`interactive` so it doesn't get preempted while submitting children.

**TPU shape competition**: training children primarily want `v5p-8` /
`v4-8` / `v6e-4` (any of those — `device_variant_constraint`). Extraction
TPU workers use the same shapes for some parents — if extraction holds
60+ v5p-8 slots at batch priority, training children at batch will wait
indefinitely. Bumping child priority to `interactive` is the simplest
unblock.

The coordinator submits 11 TPU children per method/N pair. It keeps
itself alive (`while True: sleep`) so the iris parent-child relationship
stays intact. Stop with `iris job stop` when all children are submitted
(~couple minutes).

---

## Troubleshooting

### "stuck in reshape forever"
Reshape is many small GCS reads (9508 files at 100W). Slow due to small-
file IO. Normal: 30-60 min at 100W on a contested cluster.

### "fuzzy CC has only N workers, where N = parquet shard count"
Known issue (task #22). Zephyr's actor pool sizes to the upstream
`from_list` partition count, not `max_parallelism`. Fix: smaller
`--target-partition-bytes` → more shards → more workers.

### "STRICT-DONE FAILURE"
Some priority hashes aren't in `resolved_{spec}.jsonl.gz`. Either re-run
Phase 1 (more WARCs landed since last consolidation), or the extraction
isn't done for that hash. Check `done_warcs_{spec}.txt`.

### Cache hash unknown
The `{cache_hash}` suffix after `{spec}_{n}warcs-` is determined by the
executor at tokenize time. List `gs://marin-us-central1/tokenized/` after
tokenize completes to discover it.

---

## Operational notes

- **Naming convention**: dataset names match `{spec}_{n}` (e.g.
  `high_quality_100`, `med_quality_500`). Cache dir is
  `{spec}_{n}warcs-{hash}`.
- **Reshape preservation**: Phase 2's reshape step output is N-specific.
  Re-running Phase 2 at the same (spec, N) reuses the existing reshape
  (saves ~30-40 min) if you've manually deleted normalize+downstream.
- **No global lock**: the consolidation Phase 1 outputs are append-only
  (rewriting resolved manifest is safe). Two simultaneous Phase 1 runs
  for the same spec would race but the resolver is deterministic so end
  state is the same.
- **Mirror only after tokenize**: don't mirror intermediate dedup outputs;
  those are large and you only need the tokenized cache for training.
