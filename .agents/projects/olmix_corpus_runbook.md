# OLMIX-BASE per-corpus runbook

How we actually ran the 24-topic x 5-quality mixture optimization end to end for `dclm_10k`
and `high_quality_10k` (726 proxy runs, 726 evals, both fits landed 2026-08-01), and what to
repeat — and avoid — for `nemotron_full_10k`, `fineweb_edu_10k`, `fineweb_cc_10k`, and
`resiliparse` when their grids are ready.

Read alongside:
- `.claude/plans/i-am-currently-working-curried-moonbeam.md` — the design decisions and why each
  hyperparameter is what it is. This runbook is the *operational* half; it does not re-argue the method.
- `experiments/data_mixing/` — all code referenced below.

---

## 0. The shape of the thing

Four phases, two of which overlap:

```
validate grid caches
      -> sample swarm (K=363) + train 363 proxy models        [~2 h per run, many in parallel]
      -> BPB-eval each finished checkpoint on 42 tasks        [~11 min per run, overlaps training]
      -> fit 42 log-linear laws + constrained solve           [~25-60 min on 7 cores]
```

Training and eval overlap by design — evals run on checkpoints as they land, not after the
whole swarm finishes. The fit needs **every** run evaluated, so it is the only hard barrier.

Per corpus this is 363 proxy runs at `d512 / 157M params / 3e18 FLOPs` (≈4.3e9 tokens,
~32.8k steps, `mixture_block_size=32768`) on 4-chip slices, plus 363 42-task BPB evals.

---

## 1. Prerequisites

The 120 per-cell Levanter caches must exist and load. `experiments/data_mixing/validate_grid_domains.py`
asserts every non-empty cell has a loadable `TreeCache`, reads real marin-tokenizer token counts
from each `CacheLedger`, and emits the `cells.json` the sampler keys off.

Do not skip it. `distribution.json` token counts are **gte-base** tokens and are only a sizing
hint; the solve's availability caps use the marin-tokenizer counts from the ledgers.

Measured totals (marin tokenizer, live domains ≈ 99.8% of corpus):

| corpus | non-empty cells | corpus tokens |
|---|---:|---:|
| dclm_10k | 118 | 7.33 B |
| high_quality_10k | 120 | 21.30 B |

**Region note.** Cells are read locally by the trainer — no cross-region cache reads. dclm /
fineweb_edu / high_quality exist in us-east5; nemotron_full and fineweb_cc exist **only in
us-central1**, so their swarms must run there unless the caches are mirrored first.

---

## 2. Swarm: sample + train

One coordinator per (corpus, region), submitted **as an Iris job** — never locally, or Iris
orphan-kills the children when the parent exits.

```bash
iris --cluster marin job run --job-name olmix-coord-<corpus>-<region> \
  --region <region> --cpu 4 --memory 1GB --extra cpu --priority interactive --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.data_mixing.launch_olmix_swarm \
       --corpus <corpus> --region <region> --launch \
       --index-start 0 --index-end 363
```

`--index-start/--index-end` split one swarm across regions. Ranges **must** be disjoint;
completion records are bucket-local and cannot deduplicate across regions on their own.

### Claims are what make this safe

`experiments/data_mixing/index_claims.py` gives every index an atomic claim in a single global
registry (`gs://marin-us-central1/metadata/olmix_claims`), created with GCS
`ifGenerationMatch=0` so exactly one coordinator wins. **Use it.** Before claims existed, one
measured hour had 53 of 57 completions be re-runs of finished work and 21 run names trained twice.

Two operational consequences:

- Launching a *fresh full-range* coordinator is safe — it will skip everything already claimed
  or complete. This is the fix for the cursor gap below.
- A coordinator that dies holds its claims until the 8 h TTL. If runs are stranded, reap first:
  `reap_dead_claims(corpus, live_run_names=<from iris>, done_run_names=<from GCS>)`. It only
  releases claims whose run is in **neither** set, so it cannot free a live run.

### The cursor gap — the failure that will bite you again

Coordinators compute their `todo` list once and advance a cursor that never revisits. **Any
index whose child dies is skipped permanently**, and the coordinator then exits looking healthy.
We finished dclm at 358/363 with *zero live children* and no error anywhere.

Detect it by reconciling **live children against remaining work**, not by watching the counter:

```bash
iris --cluster marin query \
  "SELECT SUBSTR(job_id,14,40) j, state FROM jobs WHERE job_id LIKE '/michaelryan/olmix-%/%' AND state IN (1,3)"
```

If `remaining > 0` and live children is 0, you are stalled. Fix: find the missing indices from
the results listing, reap their claims, launch a coordinator over a range covering them. Runs
resume from their temp checkpoints — one of our five stranded runs finished ~15 min after relaunch.

---

## 3. Evals: the part that goes wrong quietly

Evals are a separate pass over finished checkpoints, driven by a manifest.

```bash
# 1. build the manifest (all regions; --region is repeatable and MUST be)
python -m experiments.data_mixing.build_final_eval_wave --tag w<N> --exclude <inflight.txt>

# 2. one wave per region-group, because --tpu-variants applies to the whole wave
iris --cluster marin job run --job-name olmix-bpb-w<N> --region <region> \
  --cpu 4 --memory 1GB --extra cpu --priority interactive --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.scaling_law_sweeps.olmo_bpb.launch_olmo_bpb_manifest \
       --manifest experiments/core_eval_manifests/olmix_swarm_bpb_w<N>_<group>.txt \
       --tasks "<the 42 tasks>" \
       --results-subpath 'metadata/olmix_swarm_bpb/{run_name}/' \
       --launch --child-priority interactive \
       [--tpu-variants v5litepod-4]     # us-west4 ONLY
```

`build_final_eval_wave.py` exists precisely to encode the three traps below; prefer it over
hand-rolling a manifest.

### Trap 1 — a whole region can be silently unrunnable (cost us 132 evals)

`DEFAULT_TPU_VARIANTS = ("v5p-8", "v4-8", "v6e-4")`. **us-west4 has only `v5litepod-*`.** A child
requesting an absent variant is rejected at *submit* time; the launcher caught the exception and
filed it under `incomplete`, which normally means "HF export not written yet" — a self-healing
condition. So 20% of the sweep read as ordinary lag for hours.

Second, independent blocker: the child reads `eval_datasets/olmo_in_loop_evals/` and
`core_tasks_hub_cache/` from **its own checkpoint's bucket**. us-west4 had neither. Staging is
~554 MB, i.e. cents:

```bash
gcloud storage rsync -r gs://marin-us-central1/eval_datasets/olmo_in_loop_evals/ \
                        gs://marin-<region>/eval_datasets/olmo_in_loop_evals/
gcloud storage rsync -r gs://marin-us-central1/eval_datasets/core_tasks_hub_cache/ \
                        gs://marin-<region>/eval_datasets/core_tasks_hub_cache/
# verify the destination task-dir count matches the source (118 dirs)
```

**The diagnostic that finds this: group un-evaluated runs by region.** A region that is merely
*behind* is lag; a region with *zero completions ever* while others progress is config.
`launch_olmo_bpb_manifest` now separates `rejected` from `incomplete` and logs rejections
per-region at ERROR — trust that line.

### Trap 2 — `--skip-existing` cannot see in-flight work

It skips runs whose `results.json` exists. A run queued or mid-eval in another wave looks
un-evaluated and gets submitted twice. Always pass the currently pending/running run names as
`--exclude`. (We launched two waves over the same manifest once; this is that fix.)

### Trap 3 — waves finish without covering everything

Every wave is a snapshot. Runs that finish training afterwards have no wave. After each wave
completes, re-run the builder; it reports `trained / evaluated / in_flight -> needs_eval` and
breaks `needs_eval` down by region. Keep going until it says 0.

---

## 4. Fit + solve

**Locked target: `--requested-tokens 3.0e10 --repetition-factor 20`** (user, 2026-08-01).

```bash
iris --cluster marin job run --job-name olmix-fit-<corpus> --region us-east5 \
  --cpu 7 --memory 10GB --enable-extra-resources --extra cpu --extra mixing \
  --priority interactive --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.data_mixing.run_olmix_fit \
       --corpus <corpus> \
       --region us-east5 --region us-central1 --region europe-west4 --region us-west4 \
       --manifest gs://marin-us-east5/metadata/olmix/<corpus>/swarm_s42_K363.json \
       --output-prefix gs://marin-us-east5/metadata/olmix/<corpus> \
       --requested-tokens 3.0e10 --repetition-factor 20 --workers 7
```

- **CPU workers are 8 core / 14 GB.** Requesting 32 GB makes the job permanently unschedulable
  and it just sits pending — that cost us an hour before we noticed.
- **`--workers` matters a lot.** One task fit is ~236 s on the real design (363 runs x ~75 live
  domains), so 42 tasks serial is ~2.75 h *per corpus*. The fan-out is bit-identical to serial
  (`fit_log_linear` reseeds to 42 on entry; asserted in `tests/data_mixing/test_olmix_fit_parallel.py`).
- Add `--sweep` only if you want the (R, k) grid; it **re-fits** all 42 tasks a second time, so
  it roughly doubles runtime. Skip it for a known (R, k).
- Do **not** run two fits for the same corpus concurrently with `--write-csv` — they race on
  `ratios.csv` / `metrics.csv`.

Outputs land in `metadata/olmix/<corpus>/`: `mix_R3e+10_k20.json` (mixture, natural, interaction
matrix, per-task `log_c`, fit quality), plus `ratios.csv` / `metrics.csv` in olmix's own schema
so the fit can be cross-checked against upstream `olmix fit`.

### Sanity checks on the fit output

| field | dclm | high_quality | what a bad value means |
|---|---|---|---|
| `runs` | 363 | 363 | < K means evals are missing |
| `live_domains` | 75 / 118 | 73 / 120 | live must be >> 1 and `runs >= live+1` |
| `regression_fit.average_bpb` | 0.957 | 0.983 | << 0.9 means the laws don't describe the data |
| `n_degenerate_tasks` | 0 | 0 | **non-zero is a broken task — fix it, never drop it** |
| `skipped.duplicate_run` | 21 | 26 | cross-region duplicates; collector keeps the first, this is fine |

`skipped.no_bpb > 0` simply means some runs aren't evaluated yet — do not fit until it's 0.

### Re-solving at a different (R, k) without refitting

The fit is the expensive part; the solve is seconds. You can rebuild the fitted params exactly
from `mix_*.json` (`log_c` + `interaction_matrix` over the live domains, in `manifest.domains`
order) and call `solve_with_natural_reinsertion` directly. Verified against a stored sweep cell
to 3.9e-10. Useful for exploring; produce the **official** number with a real job.

---

## 5. Choosing (R, k) for a new corpus

`R` is a token count only — model size enters the solve nowhere. And only the ratio matters:

```
cap_j / natural_j  =  k * N_total / R      (identical for every cell)
max feasible R     =  k * N_live
```

So compute `k*N_total/R` first. If it is near 1, the constraint — not the fit — will dictate the
answer. At `R=20B, k=4` dclm's ceiling was **1.47x**, every cell pinned to it, and the proposed
mixture collapsed to the natural distribution (q3 1.01x vs 1.26x unconstrained). At `R=30B, k=5`
it went *below* natural (0.91x): the recommendation inverted. At `k=20` the ceiling is 4.89x and
both corpora recover their true preference (dclm q3 1.25x, hq 1.22x) while moving only 0.9% /
1.7% of mass from the unconstrained optimum.

**Rule of thumb: keep `k*N_total/R` >= ~4, and sanity-check the implied repetition** of the
unconstrained mixture (`epochs_j = x_j*R/N_j`) before trusting a result. A corpus small relative
to `R` cannot express a preference no matter what the fit says — report those arms as
data-constrained rather than as findings about quality.

---

## 6. Monitoring

`experiments/data_mixing/swarm_monitor.py --interval 1800` prints one line per cycle and fires a
completion trigger at `K/K trained + K/K evaluated`. Two invariants it encodes, both learned from
false alarms:

- **Count distinct `-iNNNN-` indices, not objects.** A run trained in two regions leaves two
  result files for one index (hq once read 344 files for 318 real runs).
- **Completion counts are append-only.** A count that goes *down* proves a bad read, never lost
  work. A failed read must stay distinguishable from an empty one (`None` vs `0`) — conflating
  them produced every false alarm we had.

Its `running=` figure only matches `olmix-coord%` job names; coordinators named otherwise are
undercounted. Query the job table directly when the number matters.

Other things that will look like bugs and aren't:
- **finelog is often down** (`StatsError: Not Found`), so `iris job logs` fails. Use
  `iris job bug-report <job>` and GCS artifacts instead.
- **Flat counters with many runs in flight are normal** — proxy runs start in batches and land in
  batches ~2 h later.
- **The controller intermittently refuses new jobs** — `launch_job` times out server-side while
  queries work fine. It cleared on its own after ~3 h. Probe with a trivial job
  (`--cpu 1 --memory 1GB -- python -c "print(1)"`) to tell "controller is refusing everything"
  from "my job is malformed" *before* burning an afternoon on retries.
- **Orphaned SSH tunnels accumulate** (one per `iris` CLI call, never reaped) and eventually
  contend; `pkill -f "compute ssh iris-controller-marin"` is safe — no cluster job depends on them.

### Finished-but-lingering coordinators

After a swarm completes, old coordinators can keep dispatching children that re-train indices
which already have results, occupying a whole region. We found 26 such children holding all of
us-west4 while the last 10 evals starved. Verify every index they're running already has a
results JSON, then stop them **with explicit user permission** —
see `feedback_never_kill_running_jobs`.

---

## 7. Per-corpus checklist

1. `validate_grid_domains.py` green; note `N_total` and cell count.
2. Compute `k*N_total/R` at the locked `R=30B, k=20`. If < ~4, flag it before spending compute.
3. Stage eval datasets into every region the corpus will train in (§3, Trap 1).
4. Launch swarm coordinator(s) as Iris jobs, disjoint index ranges, claims on.
5. Roll eval waves as checkpoints land; one wave per region-group; always `--exclude` in-flight.
6. Loop the wave builder until `needs_eval = 0`.
7. Fit with `--workers 7` at the locked (R, k); check the table in §4.
8. Report the mixture **with** its implied repetition and `k*N_total/R`, so a supply-constrained
   result is never mistaken for a quality finding.
