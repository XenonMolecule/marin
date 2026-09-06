# Handoff: train + evaluate the four OlmixExact mixtures

Owner hand-off, 2026-08-26. Everything below is verified against cluster `marin`.

**Goal.** Vendor four solved OLMIX mixtures into the repo, register them as training methods,
launch their training sweeps, and evaluate the resulting checkpoints on the 60-task OLMo
Base-Easy suite.

**Target artifact.** `_olmixexact_lambda0p01` — the olmix paper's exact 51-task devset
(Table 9), KL regulariser λ=0.01, solved at R=30B, k=20, from 363 proxy runs per corpus.

---

## 0. State at handoff

| grid_corpus | swarm | evals | λ=0.01 fit | vendored |
|---|---|---|---|---|
| `resiliparse_10k` | 363 | 363 | done, 363 runs | **yes** |
| `lpv11_fastpipe_v1_10k` | 363 | 363 | running | no |
| `dclm_10k` | 363 | 363 | running (refit) | no |
| `high_quality_10k` | 363 | 363 | running (refit) | no |

Fit outputs land at:

```
gs://marin-us-east5/metadata/olmix/<grid_corpus>/fit_olmix_exact_kl0p01/mix_R3e+10_k20.json       # resiliparse, lpv11
gs://marin-us-east5/metadata/olmix/<grid_corpus>/fit_olmix_exact_kl0p01_all4/mix_R3e+10_k20.json  # dclm, high_quality refits
```

### Before vendoring anything, verify the fit used all 363 runs

The dclm and high_quality fits were first run against only two of the four result buckets. They
fit on 96 and 201 runs respectively and still reported `status=optimal`. A short collection is
not surfaced as an error anywhere downstream.

```bash
uv run iris --cluster marin job logs /michaelryan/<fit-job-id> 2>&1 \
  | grep -oE "collected [0-9]+ usable runs"
# REQUIRED: collected 363 usable runs
```

---

## 1. Vendor the mixture into the repo

Training sweeps do **not** read the GCS solve artifact. They read a vendored file in the repo,
and the two documents differ:

| | solve artifact (GCS) | vendored (repo) |
|---|---|---|
| weight map key | `mixture` | `weights` |
| provenance | — | `source`, `swarm_runs` |
| diagnostics | `interaction_matrix`, `log_c`, `domains`, `dead_domains` | stripped |

`GridMixCurationMethod.load_weights` does `json.loads(...)["weights"]`, so copying the GCS file
straight in raises `KeyError: 'weights'` at sweep-planning time.

```bash
uv run python -m experiments.data_mixing.vendor_olmix_mixture \
  --src gs://marin-us-east5/metadata/olmix/<grid_corpus>/fit_olmix_exact_kl0p01/mix_R3e+10_k20.json \
  --grid-corpus <grid_corpus> \
  --mixture-tag _olmixexact_lambda0p01
```

Writes `experiments/data_mixing/mixtures/<grid_corpus>_R3e10_k20_olmixexact_lambda0p01.json`.

Tests: `tests/data_mixing/test_vendor_olmix_mixture.py` (8 tests, round-trips through the real
`GridMixCurationMethod.load_weights`).

**Naming.** GCS dirs use `kl0p01`; the vendored tag uses `lambda0p01`. This is deliberate repo
convention — a `k`-prefixed tag beside `_k20` reads as a second repetition factor, which it is
not. Keep the vendored tag exactly `_olmixexact_lambda0p01`.

---

## 2. Register the training method

Add one entry per corpus to `METHODS` in `experiments/scaling_law_sweeps/curation_plan.py`.
`_grid_mix_method` builds the mixture path from `grid_corpus` + `mixture_tag`, so the filename
from step 1 must match exactly.

```python
"resiliparse_10k_mix_olmixexact_lambda0p01": _grid_mix_method(
    "resiliparse_10k_mix_olmixexact_lambda0p01",
    "resiliparse_decon_10364warcs-beaaf5",
    "resiliparse_10k",
    mixture_tag="_olmixexact_lambda0p01",
),
```

Base tokenized cache hashes:

| grid_corpus | base_cache_hash |
|---|---|
| `dclm_10k` | `dclm_400m_1x_10k_dclm-3df0ba` |
| `high_quality_10k` | `high_quality_decon_10364warcs-6451c8` |
| `resiliparse_10k` | `resiliparse_decon_10364warcs-beaaf5` |
| `lpv11_fastpipe_v1_10k` | `lpv11_fastpipe_v1_decon_10364warcs-a16e729` |

Only dclm and high_quality have existing `*_mix` arms. resiliparse and lpv11 will be the first
grid-mix methods for those corpora — give their registration an extra read.

**Grid cache layouts differ between corpora:**

```
gs://marin-us-east5/datakit/store/resiliparse_10k_gridv1/cluster=C/quality=Q/sub=S
gs://marin-us-east5/datakit/store/lpv11_fastpipe_v1_10k_gridv1/cluster=C/quality=Q      # no sub= level
```

---

## 3. Launch the training sweep

**CORRECTED 2026-08-27.** This section originally prescribed
`launch_curation_sweep.py --experiments C` (73 IsoFLOP cells/corpus). That grid is NOT
comparable to the existing mix arms — every prior `*_mix*` method trained on the frozen
38-cell `expFM_natural` grid — and a full expC fleet launched from this doc had to be
killed. Use `launch_10k_natural.py` with `--max-budget 9e20` (the 36-cell variant; the two
2e+21 stragglers are dropped per the lpv11 36/38 close-out). Register new methods in its
`METHOD_NAMES` first. Run it **as an Iris job**, never locally.

```bash
set -a; source .env; set +a
uv run iris --cluster marin job run --no-wait --priority interactive \
  --cpu 4 --memory 2GB --extra cpu --no-preemptible \
  --job-name 10k-natural-olmixexact \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python experiments/scaling_law_sweeps/launch_10k_natural.py \
       --methods dclm_10k_mix_olmixexact_lambda0p01 high_quality_10k_mix_olmixexact_lambda0p01 lpv11_fastpipe_v1_10k_mix_olmixexact_lambda0p01 \
       --max-budget 9e20 --child-priority batch \
       --allowed-regions us-east5 us-central1 us-central2
# resiliparse separately: its grid store exists ONLY in us-east5
#   --methods resiliparse_10k_mix_olmixexact_lambda0p01 --allowed-regions us-east5
```

Region constraints come from grid-store presence (`datakit/store/<corpus>_gridv1`), not the
base caches: resiliparse us-east5 only; lpv11/dclm/hq also us-central1 + us-central2
(copied 2026-08-27); dclm/hq additionally us-west4 + europe-west4. `--methods` is
space-separated, not comma-separated.

Children float across regions (soft preference for all regions, so they do not inherit the
parent's region), request both a v4 and a v5p shape so Iris places on whichever has capacity,
and region-lock on first write via the tracker.

**Do not stop the coordinator.** Stopping a sweep coord orphan-kills every child with
`Terminated by user`, even with `--no-include-children`. It must also be `--no-preemptible`: on
a preemptible host one preemption takes down the whole fleet.

---

## 4. Evaluate on OLMo Base-Easy (all 60)

The 60 tasks are `resolve_tasks('all+qa_rc')` (56, from
`experiments/scaling_law_sweeps/olmo_bpb/olmo_bpb_tasks_set.py`) plus the 4 MMLU category tasks.

Build a manifest of `run_name,region,output_path` covering trained runs that lack
`results.json`, then launch one wave per corpus:

```bash
uv run iris --cluster marin job run --no-wait --region us-east5 --priority interactive \
  --cpu 4 --memory 1GB --extra cpu --no-preemptible \
  --job-name olmix-bpb-<corpus> \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.scaling_law_sweeps.olmo_bpb.launch_olmo_bpb_manifest \
       --manifest <manifest.txt> --tasks "$TASKS" \
       --results-subpath 'metadata/olmix_swarm_bpb/{run_name}/' \
       --launch --child-priority interactive \
       --keepalive-timeout 172800
```

- **`--keepalive-timeout` is mandatory.** Default is 43200 s (12 h). On expiry the parent exits
  **state 4 (finished) with no error** and Iris kills every still-pending child. A truncated
  wave is indistinguishable from a successful one in the job state. This silently dropped 120 of
  363 lpv11 evals. Size it from the measured rate; 172800 (48 h) is a safe default.
- **Judge completion by the GCS result count, never by parent job state.**
- **Adding tasks to already-evaluated runs?** Use `--merge` plus `--done-marker`. Without them
  the child overwrites the existing `results.json` instead of extending it.
- **Region pinning is per-row**, from the manifest. A child's TPU variant must exist in its
  pinned region — a mismatch is rejected at submit time. Keep one wave per region when hardware
  differs (`DEFAULT_TPU_VARIANTS` does not exist in us-west4, which is v5litepod-only).
- Each eval is ~20 min for all 60 tasks.

---

## 5. Traps

**Fits need all four regions.** Swarm results are spread across four buckets:

```
--region us-east5 --region us-central1 --region us-west4 --region europe-west4
```

The region key is `europe-west4`. `eu-west4` is only the bucket name and raises
`KeyError: 'eu-west4'`.

**Never sum per-bucket counts.** The same run leaves a result directory in more than one bucket,
so summing over-reports — it produced an impossible "388/363" and "364/363". Union unique
`-iNNNN-` indices instead:

```bash
for b in us-east5 us-central1 us-west4 eu-west4; do
  gcloud storage ls "gs://marin-$b/metadata/olmix_swarm_bpb/" 2>/dev/null
done | grep -oE "olmix-<corpus>-s42-K363-i[0-9]{4}" | sort -u | wc -l
```

**Never let a failed read count as a value.** A failed `gcloud storage ls` returning 0 reads as
"no progress" and can trigger a fallback. Probe the bucket root for reachability (a missing
prefix and a network failure both exit non-zero), and hold on failure rather than acting.

**A capacity wait is not a stall.** Gate any watchdog on the queue — pending children > 0 means
the job still wants capacity — never on a timer. A timer-based fallback fired during a pure
capacity wait and restarted training into the region the evals were waiting on.

**Core v2 is inside the `olmix_exact` objective.** It contains the 10 DCLM Core v2 tasks
(arc_easy, arc_challenge, csqa, hellaswag, winogrande, piqa, coqa, jeopardy, squad, lambada), so
a mixture optimised against it must **not** be reported as beating natural *on Core v2*. Use the
42-task `marin` devset (`--devset marin`) for that claim. Both read the same evals, so run both.

**Capacity shapes schedules.** us-central1 is v5p-only and thin (3–8 small slices), so a 363-run
wave pinned there takes days; us-east5 v6e is the fast path. CPU workers total ~11 cluster-wide,
so idle coordinators starve real work — retire them once their children are done.

**HF Hub rate limits kill evals, not just training.** 1000 requests / 5 min per token, shared
across every job. The launcher's `--wave-size 8 / --wave-delay 420` throttle exists for this.
`--hub-offline` does not work — the runner's model load is inherently online (levanter probes
gpt2); three pilots confirmed this.

**Mixture block floor.** Levanter truncates weights below `1/mixture_block_size` (65535). The
loader drops those cells and renormalises — measured ~0.01% of mass, logged at load. Expected.

---

## Reference

- Solve driver: `experiments/data_mixing/run_olmix_fit.py` (`--devset marin|olmix_exact`)
- Devset definitions: `experiments/data_mixing/olmix_tasks.py`
  (`build_target_tasks` = 42, `build_olmix_exact_tasks` = 51)
- Vendoring: `experiments/data_mixing/vendor_olmix_mixture.py`
- Method registry: `experiments/scaling_law_sweeps/curation_plan.py`
- Mixture loader: `experiments/scaling_law_sweeps/data_curation_math.py::GridMixCurationMethod`
- Sweep coordinator: `experiments/scaling_law_sweeps/launch_curation_sweep.py`
- BPB eval launcher: `experiments/scaling_law_sweeps/olmo_bpb/launch_olmo_bpb_manifest.py`
