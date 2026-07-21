# Scope: CORE-v2 decontamination for dclm / nemotron / fineweb / fineweb_edu

**Status:** EXECUTING dclm + nemotron only (2026-07-13, per Michael). Owner: Michael Ryan.

## EXECUTION LOG (dclm + nemotron, 10k natural sweep only)

Decision: run decon for the two most-competitive baselines (dclm, nemotron) only;
if numbers move, do fineweb/fineweb_edu after. Fair decon-vs-non-decon comparison
via NEW method names (old non-decon runs kept intact — run_name embeds method name,
not cache hash, so new names = fresh runs).

Pipeline (all us-central2 except training):
1. **decon_apply** (RUNNING, launched 2026-07-13):
   - dclm: input `filtered/dclm_400m_1x_10k_dclm_resharded-1fe977/`, out
     `documents/baseline_dclm_decon/10364warcs_core_v2/survivors/`
   - nemotron: input `filtered/dclm_400m_1x_10k_nemotron_full-96bad9/`, out
     `documents/baseline_nemotron_decon/10364warcs_core_v2/survivors/`
   - source `gs://marin-us-central2/decontamination/dclm_core_v2/`, n=15 df<=10
   - iris jobs `decon-apply-{dclm,nemotron}-10k` (us-central2, cpu4, interactive)
2. **tokenize** (script READY, not launched): `tokenize_10k_decon.py`, DECON_METHOD
   env=dclm|nemotron, MARIN_PREFIX=gs://marin-us-central2. Cache names
   `dclm_400m_1x_10k_dclm_decon`, `dclm_400m_1x_10k_nemotron_full_decon`.
3. **register** (TODO): after tokenize, read `{cache}/train/.stats.json:total_tokens`
   + real -HASH; add to curation_plan `_D_OBS_DEFAULTS` + METHODS as
   `dclm_10k_decon` / `nemotron_10k_decon` (sampled_warcs=EXPC_SAMPLED_WARCS,
   pin_region=us-east5 after mirror); add both to launch_10k_natural METHOD_NAMES.
4. **mirror** (TODO): LEAN mirror us-central2 -> us-east5 — copy ONLY
   `train/input_ids/` + `train/shard_ledger.json` + `train/.stats.json` (Levanter
   reads only those at runtime; part-*/ build dirs NOT needed — curation_plan.py
   resiliparse precedent). dclm input_ids=15.6GB (full cache 31GB); nemo ~22GB.
   Lean total ~38GB, ~$1-3 egress. us-east5 v5p runs the WHOLE grid + already holds
   the non-decon caches (co-located comparison). Set method pin_region="us-east5".
5. **launch** (TODO): `launch_10k_natural.py --methods dclm_10k_decon
   nemotron_10k_decon --child-priority batch` (pin_region on the methods forces
   us-east5) = 2x38=76 runs. Sweep gives each cell v4+v5p variants; us-east5 v5p
   covers all cells.

PROGRESS (live):
- dclm: decon-apply ✓ (100/100 shards) → tokenize ✓ cache
  `dclm_400m_1x_10k_dclm_decon-177776` = 7,327,476,298 tok (non-decon was
  7,331,583,927; 4.1M / 0.056% dropped) → LEAN MIRROR to us-east5 ✓
  (train/input_ids 15.58GB + ledger + stats + root .artifact/.executor_* files).
- nemotron: decon-apply ✓ (24390/24390 shards) → tokenize RUNNING
  (tokenize-nemotron-decon-10k). Cache will be `dclm_400m_1x_10k_nemotron_full_decon-*`.
- Validated: `launch_10k_natural.py --methods dclm_10k --dry-run` = 38 cells, each with
  v4+v5p variants (up to v5p-128). us-east5 v5p covers the whole grid.
- nemotron: tokenize ✓ cache `dclm_400m_1x_10k_nemotron_full_decon-e271a5` =
  10,125,029,374 tok (non-decon 10,130,086,896; 5.06M/0.05% dropped) → LEAN MIRROR
  us-east5 ✓ (input_ids 21.6GB). Total lean egress ~37GB.
- REGISTERED: curation_plan _D_OBS + METHODS (`dclm_10k_decon`, `nemotron_10k_decon`,
  pin us-east5) + launch_10k_natural METHOD_NAMES. New file tokenize_10k_decon.py.
- LAUNCHED (batch): coordinator `launch-10k-decon` = 76 canonical cells (38+38).

## ACTUAL non-decon scale (2026-07-13, per Michael "registry may be lying")

Counted DISTINCT finished checkpoint dirs across ALL 6 regions (runs float regions):
**dclm_10k = 44, nemotron_10k = 44** — NOT the frozen-grid 38 that curation_plan /
launch_10k_natural / view_10k_natural_grid report. All 38 frozen cells DID finish
(zero gaps) so the 38-cell decon sweep is a complete matched scaling law. The 6
extras per method (identical set for both):
- `2e19-d512-B64` + `…-B64-ARCHIVED` — superseded batch (frozen uses B128) + archived dup
- `9e21-d2432`, `9e21-d3584` — the 9e21 cells DROPPED 2026-05-31 for v5p cost (already finished)
- `9e20-d1536`, `2e21(1.8e21)-d1536` — high-budget points at MEDIUM width; frozen grid
  restricts 9e20/1.8e21 to big widths only.

DECISION (Michael): add the 2 d1536 points to decon (skip 9e21 + archived). New file
`launch_10k_decon_extras.py` reproduces them (9e20 div=1 -> B1024/v5p-256;
1.8e21 div=2 -> B1024/v5p-64). Coordinator `launch-10k-decon-extras` launched.
NOTE: 9e20-d1536 needs v5p-256 (bigger than the frozen grid's v5p-128 peak) — pinned
us-east5; may sit pending if that capacity isn't free (harmless at batch).

TOTAL DECON RUNS: 76 canonical + 4 extras = 80. Evals (CORE/OLMES/olmo_bpb) after.

## PAUSED 2026-07-15 to free capacity for N=100 evals (per Michael)

Decon is a confirmed no-op on Core (mean Δ≈+0.001, no systematic finding), so Michael
de-prioritized it in favor of the N=100 WARC evals. PAUSED the decon TRAINING fleet:
`iris job stop /michaelryan/launch-10k-decon --include-children` → killed BOTH coordinators
(launch-10k-decon + launch-10k-decon-extras, prefix match) + 25 running big cells. 55/80
DONE runs PRESERVED (markers + hf/ intact). Freed ~25 us-east5 v5p slices (incl v5p-128/64).
Decon Core evals (37/40) left running; decon monitor retired.

RESUME decon training later (temp checkpoints in pinned us-east5 → resumes; skip_if_done
skips the 55 done):
  iris --cluster marin job run --no-wait --region us-east5 --cpu 2 --memory 8GB \
    --priority interactive --extra cpu --enable-extra-resources --job-name launch-10k-decon \
    -e WANDB_API_KEY <k> -e HF_TOKEN <t> -- python experiments/scaling_law_sweeps/launch_10k_natural.py \
    --methods dclm_10k_decon nemotron_10k_decon --child-priority batch
  # + launch_10k_decon_extras.py for the 2e21-d1536 pair (skip_if_done skips the done extras)

## MONITORING (live, 2026-07-14)

Monitor task tracks DONE markers in us-east5; emits at 20/40/60/80 milestones + failures
+ 3h stall. Progress ~21/80 early on, draining fast.

FAILURES SELF-RECOVER (no manual recovery needed). Transient `failed`/`cancelled` states are
preemption churn on the preemptible TPUs: observed `3e+20-d1024-L11-B512` go failed -> back to
`pending` (auto-requeued) on its own; ~1 cell is failed at any moment as preemptions rotate.
Confirmed benign (non-decon twins all have DONE markers). So NO end-recovery resubmit pass —
the fleet drains to 80 by itself. Monitor alerts only if CONCURRENT failures >=3 (systematic
signal) or a 3h stall (would be the 9e20-d1536 v5p-256 capacity wait). Then launch evals.
6. **eval** (TODO, later): CORE_tasks + OLMES + olmo_bpb over the 76 new checkpoints.

Note: finelog log-fetch is down (known) — rely on `iris job list` state + GCS output.

---

**Original scoping (2026-07-13):** No work launched. Owner: Michael Ryan.

## Question

`high_quality` and `resiliparse` got CORE-v2 decontamination; `dclm`, `nemotron`,
`fineweb` (=fineweb_cc), and `fineweb_edu` did not. What would it take to bring the
latter four to parity? (Retraining is the expensive part.)

## How decontam works (the mechanism)

Two CPU-only Zephyr stages wrapping `marin.processing.classification.decon`
(bloom filter of eval n-grams; GPT-3/Dolma rule):

- **Mark** — `experiments/baseline_collection/decon_extracted.py` (DECONTAMINATE
  mode). Read-only; writes an attributes tree + bloom. `--ngram-length 13`.
- **Apply/remove** — `experiments/baseline_collection/decon_apply.py`. Two passes:
  (1) document-frequency of each eval n-gram across the corpus; (2) drop any doc
  containing an eval n-gram with **DF ≤ 10** (a *distinctive* eval passage =
  genuine leakage). `--ngram-length 15 --df-threshold 10`. Writes survivor
  `data-*.jsonl.gz`. This is the actual removal → tokenizer input.
- **Inspect** — `decon_inspect.py` measures flag/genuine rate with an exact index
  (no bloom FPs) → `inspect_summary.json`.

Decon **source** = DCLM CORE-v2 eval items (22 tasks), built by
`experiments/scaling_law_sweeps/dclm_core/dump_core_v2_decon_source.py` →
`gs://marin-us-central1/decontamination/dclm_core_v2/`.

**How a method becomes "decon'd":** there is NO per-method decon flag. A
`CurationMethod` just points `tokenized_rel_path` at a tokenized cache. hq /
resiliparse point at `*_decon_10364warcs-*` caches; the four targets point at
plain caches. Adding decon = new corpus → new cache hash → re-tokenize →
re-register → retrain.

## Current state — the mark/inspect was PARTLY done already

Flag-rate inspection has already been run for three of the four (dirs under
`gs://marin-us-central2/documents/baseline_{dclm,nemotron,fineweb_edu}_decon/10364warcs_core_v2/inspect/`).
**No removal, no re-tokenize, no retrain.** Contamination is negligible:

| method | total docs | flagged | genuine (DF≤10) | genuine % |
|---|---|---|---|---|
| dclm | 5,931,419 | 2,187 | 1,258 | 0.021% |
| nemotron | 17,758,166 | 3,549 | 2,264 | 0.013% |
| fineweb_edu | 2,349,555 | 2,066 | 1,173 | 0.050% |
| fineweb (cc) | — | not measured | not measured | — |

`fineweb_cc` has no decon dir yet (it's sourced from HF FineWeb, HF dedup trusted).

## Work remaining (per method)

1. **(fineweb_cc only)** run `decon_inspect.py` to measure flag rate — cheap CPU.
2. **Apply removal** — `decon_apply.py` over the filtered corpus, in-region
   us-central2, CPU. Cheap (billions of n-grams but a small bloom; hq/resiliparse
   precedent). Produces `*_decon_deduped` survivors.
3. **Re-tokenize** the survivors → new cache hash. Tokenize job (Llama-3.1-8B
   tokenizer, `default_tokenize`), in-region.
4. **Re-register** in `experiments/scaling_law_sweeps/curation_plan.py`: new cache
   hash in `_D_OBS_DEFAULTS` (with `total_tokens` from `train/.stats.json`) + point
   the `_method(...)` at it. Mirror the new cache to whichever regions the sweep
   dispatches to.
5. **Retrain** the sweep grid (TPU — free on TRC).
6. **Re-eval** every new checkpoint: CORE_tasks + CORE-v2 + OLMES + olmo_bpb.

Steps 1–4 are hours of cheap in-region CPU. Steps 5–6 are the cost.

## Retraining magnitude

Grid = `launch_10k_natural.py`: WIDTHS (512/1024/1536/2432/3584) × 12 budgets
(3e16…3e20) + big-width extension (9e20, 1.8e21), minus corner drops =
**38 cells/method**. All four are currently 38/38 done.

| sweep | dclm | nemotron | fineweb (cc) | fineweb_edu |
|---|---|---|---|---|
| **10k natural** (`launch_10k_natural.py`) | 38 | 38 | 38 | 38 |
| 3k fixed-model (`launch_fixed_model_sweep.py`) | ~21 (`dclm`) | ~21×2 (`nemotron_org`, `nemotron_full_bos_fixed`) | — | ~21 (`fineweb_edu`) |
| WARC-scaling (`launch_warc_scaling_sweep.py`) | ~113 (N=100/500/1000/2000) | ~113 (`nemotron_full`) | — | — (dropped) |

- **10k sweep only (the canonical comparison):** 38 × 4 = **152 training runs** +
  ~152 checkpoints × 4 eval suites.
- **+ 3k fixed-model parity:** +~84 runs.
- **+ WARC-scaling parity:** +~226 runs (dclm + nemotron only).

Full parity across all three sweeps ≈ **450+ runs**. If we only care about the 10k
natural comparison (where hq/resiliparse decon lives), it's **152 runs**.

## Dependency chain (what regenerates)

corpus (decon survivors) → `default_tokenize` → `tokenized/{new_hash}/` →
`curation_plan._D_OBS_DEFAULTS` + `METHODS` → `run_curation_train_standalone.py`
→ `checkpoints/isoflop-curation/{run_name}` → eval manifests
(`experiments/core_eval_manifests/`) → CORE/CORE-v2/OLMES/olmo_bpb results.
`run_name` embeds the method name, not the cache hash, so old checkpoints go
silently stale — they must be regenerated, not detected as different.

## Cost / region notes

- Decon + tokenize: all in-region us-central2 (where the four corpora live). CPU,
  cheap. No cross-region egress.
- Training: TPU, free on TRC. New caches must be mirrored to the sweep's dispatch
  regions (fineweb_edu/fineweb_cc are currently us-central2-only; dclm/nemotron 10k
  caches likewise) — pin `--allowed-regions` accordingly.
- Real $ cost is only the cache-mirror egress and any eval-cache egress.

## The honest caveat (recommend deciding before launching)

**Contamination is 0.01–0.05% of docs (1–2k of 2–18M).** Removing it will not
measurably move any training curve. The reason to do this is **not** better models —
it's **methodological consistency / defensibility**: hq and resiliparse were
decontaminated, so an apples-to-apples curation comparison (and a "all methods
decontaminated" claim in a writeup) wants the other four treated identically. If
the comparison table doesn't need that claim, the 152+ retrain is scientifically
near-null. Decide the *why* (parity-for-publication vs. actual leakage worry)
before spending the TPU-hours — the flag rates say leakage is a non-issue.

If we proceed, the cheapest defensible scope = **10k natural sweep only, 152 runs**,
since that's the sweep where hq/resiliparse decon actually lives.
