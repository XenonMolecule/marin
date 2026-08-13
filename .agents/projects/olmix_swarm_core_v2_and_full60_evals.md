# OLMIX swarm: DCLM Core v2 + full OLMo Base-Easy evals over all 726 proxy runs

Launched 2026-08-02. Goal (user): enable mixing toward targets other than the 42-task
devset — (a) **DCLM Core v2** (higher priority) and (b) the **full** OLMo Base-Easy suite.
No retraining: the swarm stands, only evals are added.

Read alongside `.agents/projects/olmix_corpus_runbook.md` (the swarm's operational manual);
this note covers only the two new eval families.

---

## STATUS

- **Full canonical OLMo Base-Easy (60-task): COMPLETE 2026-08-03 22:52 UTC.**
  751/751 result files covering **726/726 distinct runs** (363 dclm + 363 hq), and all 751
  verified: 60 tasks each, no lost originals, `limit=None`, no non-finite values, and every
  original bpb **byte-identical** to the pre-merge backup. The 42-task fit path is unaffected
  (the collector selects by task name and ignores the 18 extras) — regression-checked in §2.
  The backup at `metadata/olmix_swarm_bpb_backup_pre_full60/` is now redundant; delete only
  with user sign-off.
- **DCLM Core v2: IN PROGRESS** — **456/726 as of 2026-08-04 11:01 UTC**, ~92 concurrent
  across eu-west4 (56) / us-west4 (16) / us-east5 (15) / us-central1 (4). Live waves:
  `olmix-corev2-w2-rest` + `w4-usw4` (main), `w5-rest` + `w5-usw4` (sweeper over the 178
  transient failures). All parents `--no-preemptible`, 24 h keep-alive, attempt 0.
  Everything for all 726 is submitted — no manual step gates completion, though a further
  sweeper pass may be needed for anything that fails twice.

  Two blockers were cleared 2026-08-03/04, both worth remembering:
  1. **us-west4 needed `--memory-gb 64`, not free capacity** (§7e). The hardcoded 192GB
     request cannot fit a v5e VM, so that region completed **zero** Core v2 evals all day
     while its slices sat idle.
  2. **europe-west4 genuinely was capacity** — the user freed `extract-lpv11-10k-v6e4euw`
     (67 x v6e-4) and concurrency went 14 -> 87 immediately.

## Conclusions so far (read this first; sections below show the working)

Several sections record a hypothesis and then correct it. These are the **current** positions:

1. **Run the Core v2 fleet — it is justified.** The 42-task devset explains only ~48% of the
   true variance in `Core_v2` (r = -0.70 corrected for measurement noise), so a
   devset-optimal mixture is *not* Core-v2-optimal. §4c.
2. **Optimise the 9-task mean, report the 22-task `Core_v2`.** 9 of 22 clear snr ≥ 2.5, and the
   set has not changed across **n = 30 / 59 / 115 / 313** (a 10x range, corpus-balanced at
   n=313: 142 dclm / 171 hq): `bigbench_qa_wikidata` 18.2, `boolq` 7.5, `lambada_openai` 6.5,
   `squad` 4.5, `coqa` 4.2, `bigbench_dyck_languages` 4.2, `bigbench_cs_algorithms` 4.2,
   `arc_easy` 3.1, `commonsense_qa` 3.1. Restricting the target is a large win:

   | target (n=313) | se_noise | sd_swarm | reliability |
   |---|---:|---:|---:|
   | all 22 (standard `Core_v2`) | 0.624 pts | 1.469 pts | 0.820 |
   | **the 9 signal tasks** | 0.425 pts | **2.671 pts** | **0.975** |

   The 9-task mean carries **1.8x the real signal with 32% less noise** — 97.5% signal vs 82%.
   Fit the 9; quote `Core_v2` as the headline. Note both reliabilities *rose* with n (22-task
   was 0.713 at n=115), as expected: a better estimate of true spread leaves less variance
   looking like noise. `Core_v2` at n=313 spans **-1.82 to 9.04** — some mixtures score below
   the random baseline, so the target discriminates more widely than early samples showed. §4.
3. **Do NOT substitute a bpb proxy for Core v2**, and do not stage `boolq`/`openbookqa` for
   that purpose. bpb over the *same* Core v2 datasets predicts Core v2 **worse** (r²=0.33)
   than the disjoint devset (0.485). Having variance ≠ predicting the target. §4b correction.
4. **Keep the whole fleet at `per_device_eval_parallelism=1`.** p16 is ~5-6x faster but its
   scores differ in the last bits, and a part-p1/part-p16 fleet would put a systematic offset
   into the design matrix. §7.
5. **`--hub-offline-after-load` is implemented but REJECTED** (default off; leave it). §9.
6. **us-west4 unblocked itself at 11:33 UTC — no egress needed.** It had looked hopeless for
   ~7 h. (Had it stayed blocked, the 141 missing runs are a *random* subset, so fitting on
   K=257/328 would still have been sound.) §7b.
7. **Launch every coordinator with `--no-preemptible`.** One preempted parent orphan-killed
   219 children — the single largest loss of the run, and a gap in the standard launch
   pattern. §7d.
8. **CURRENT FLEETS (relaunched 2026-08-03 13:55 UTC, `-w2`)** — all four originals were
   preempted **twice**, each event orphan-killing every child (219, then 286). Stopped at a
   zero-loss moment (0 children running) and relaunched with `--no-preemptible`; verified the
   parents now sit on `marin-cpu-vm-e2-highmem-2-ondemand`, not `v5p-preemptible` TPU spare
   capacity. Backfill parents also moved to `--child-priority interactive` to end the
   starvation in §7c — backfill went 548 → 602 within 25 min of the change.

   | parent | scope | notes |
   |---|---|---|
   | `olmix-corev2-w2-rest` | 384 | from `build_swarm_core_v2_wave --tag w2` |
   | `olmix-corev2-w2-usw4` | 141 | `--tpu-variants v5litepod-4` |
   | `olmix-full60-w2-rest` | remainder | interactive |
   | `olmix-full60-w2-usw4` | remainder | interactive, `v5litepod-4` |

   State at relaunch: Core v2 201/726, full-60 548/751; nothing lost (completions are
   marker-guarded, per-task partials persist).
9. Outstanding actions: loop the Core v2 sweeper to `needs_eval=0`; re-run the signal +
   correlation analyses at n≈300. §8.

## 0. Fleet inventory (verified 2026-08-02)

| fact | value |
|---|---|
| distinct proxy runs | **726** (363 `dclm_10k` + 363 `high_quality_10k`) |
| checkpoint dirs | 818 (92 indices trained in 2+ regions) |
| HF exports | **818/818 present, all at `hf/step-32843`** — zero export gaps |
| existing 42-task BPB results | **751 dirs = 726/726 distinct runs** (25 cross-region dupes) |
| regions | us-east5 148 / us-central1 174 / europe-west4 289 / us-west4 207 (ckpt dirs) |

Runs are `olmix-<corpus>-s42-K363-i<NNNN>-w<hash>`. Checkpoints:
`gs://marin-<region>/checkpoints/olmix-swarm/<run_name>/hf/step-32843`.

**141 of the 726 exist ONLY in us-west4** — see the TPU-variant blocker in §3.

---

## 1. Canonical results paths

Deliberately separated so neither fleet can collide with the canonical curation results
or with each other. Everything is written **in the checkpoint's own bucket** (in-region).

| eval family | path | shape |
|---|---|---|
| 42-task BPB (pre-existing, complete) | `gs://marin-<r>/metadata/olmix_swarm_bpb/<run>/results.json` | `tasks{<t>:{bpb,...}}` |
| **full 60-task BPB** (this work) | **same file**, merged in place; marker `_full60.done` | 42 → **60** tasks |
| **DCLM Core v2** (this work) | **`gs://marin-<r>/metadata/olmix_swarm_core_v2/<run>_summary.json`** | `dclm.{raw,centered}_results`, `dclm.Core_v2` |
| Core v2 full result (ttl) | `gs://marin-<r>/tmp/ttl=30d/olmix_swarm_core_v2/<run>.json` | + `lm_eval_raw` |

Pre-change backup of every existing BPB result (in-region copy, 751 files):
`gs://marin-<r>/metadata/olmix_swarm_bpb_backup_pre_full60/`. **Delete only once the
60-task merge is verified complete.**

---

## 2. The full-60 BPB backfill

The 42-task devset was built by *removing* 18 staged tasks from the 56-task suite
(`olmix_tasks.build_target_tasks`): 10 Core-v2-overlapping + 5 olmix-dropped + 3
non-olmix variants. Backfilling those 18 gives the full suite, 42 + 18 = **60 tasks**.

All 18 verified staged in all four region buckets before launch. The runner merges
(`{**existing, **new}` — existing preserved, new wins), so the 42 the fit reads are
untouched; only `averages.macro_bpb` changes meaning (now over 60).

The 18: `codex_humaneval/gold_bpb_0shot`, `codex_mbpp/gold_bpb_0shot`,
`minerva_math_500/gold_bpb_0shot`, `coqa/bpb_0shot`, `jeopardy/bpb_5shot`,
`lambada/bpb_0shot`, `squad/bpb_5shot`, `arc_challenge/rc_5shot`, `arc_easy/rc_5shot`,
`csqa/rc_5shot`, `hellaswag/rc_5shot`, `winogrande/rc_5shot`, `piqa/rc_5shot`,
`qasper_yesno/rc_5shot`, `lab_bench_dbqa/rc_3shot`, `lab_bench_protocolqa/rc_3shot`,
`medqa_en/rc_5shot`, `sciriff_yesno/rc_5shot`.

Manifests cover all **751** result dirs (not 726) so cross-region duplicates are both
upgraded — whichever copy the collector picks then has 60 tasks.

**Regression-checked: the existing 42-task fit is unaffected.** The merge rewrites files the
current fit reads, so this was verified rather than assumed — 120 upgraded runs were parsed
with `collect_olmix_swarm.task_bpb_scores` (the exact function the fit calls) and all 120
still resolve every one of the 42 target tasks, `incomplete=0`. The collector selects by task
name, so the 18 extra entries are simply ignored. Only `averages.macro_bpb` changes meaning
(now a mean over 60), and nothing in the fit path reads it. A pre-merge backup of all 751
files exists (§1) if a byte-level comparison is ever needed.

---

## 3. Code change: `launch_10k_manifest.py` gained three flags

The Core v2 manifest launcher had **no `--tpu-variants`**, and `DEFAULT_TPU_VARIANTS =
("v5p-8","v4-8","v6e-4")` does not exist in us-west4 (only `v5litepod-*`). Children
asking for an absent variant are rejected at *submit* time and filed under `incomplete`,
which normally means the benign "HF export not written yet" — this is the exact trap that
cost 132 evals in the BPB sweep (runbook §3 Trap 1). Without the flag, **141 runs (19% of
the fleet) were unrunnable**.

Added: `--tpu-variants`, `--scores-subpath`, `--samples-subpath`, plus a warning when the
manifest holds us-west4 rows and no `v5litepod-*` is requested. Verified: all 8 pilot
children submitted, us-west4 included, zero rejections.

---

## 4. Will Core v2 actually have signal at 157M / 3e18?

The proxies are d512 / 157M / ~4.3e9 tokens — small enough that this had to be checked
before spending the fleet. Two measurements, both from data already on disk:

**(a) Core v2 discriminates at this exact scale.** Eight `3e18-d512-L6-B32` curation runs
on different corpora span `Core_v2` **0.25 → 8.57** (×100). The aggregate is alive.

**(b) Within-swarm spread is ~half the between-corpus spread.** Over the 38 BPB tasks
shared between the swarm and those corpora, `CV_swarm / CV_corpus` has **median 0.49**
(mean 0.48, range 0.25–0.73). Swarm-internal bpb CV is itself large: median **21.7%**
(dclm) / **18.1%** (hq).

Scaling the measured per-task Core v2 between-corpus ranges by ~0.5 and comparing against
the binomial noise floor `sqrt(p(1-p)/n)/(1-baseline)` gives an expected
signal-to-noise per task:

| tier | tasks |
|---|---|
| **strong** (≳5×) | bigbench_qa_wikidata, bigbench_cs_algorithms, lambada_openai, boolq, coqa, bigbench_dyck_languages, commonsense_qa, arc_easy, squad |
| **usable** (~2.5–5×) | bigbench_operators, piqa, hellaswag, hellaswag_zeroshot |
| **weak** (~1.2–2.5×) | arc_challenge, winogrande, openbook_qa, bigbench_language_identification |
| **dead** (≲1.2×) | jeopardy, bigbench_repeat_copy_logic (n=32!), copa (n=100), winograd (n=273), agi_eval_lsat_ar (n=230) |

So ~13 of 22 tasks should carry real signal. **The aggregate `Core_v2` target is safe.**
Per-task log-linear fitting over all 22 would let ~5 noise tasks vote with weight 5/22 in
the flat mean — decide task inclusion from the *measured* swarm spread once results land,
not from this estimate.

These are projections from 6–8 corpora, not swarm measurements. Recompute per-task spread
directly from the 726 Core v2 results before fitting.

### MEASURED at n=30 (2026-08-03 09:15) — the projection was right

`collect_swarm_core_v2` over the first 30 real swarm results (10 dclm / 20 hq).
`Core_v2` spans **2.87 → 8.95**, mean 6.49, sd 1.37 — the aggregate target is clearly viable.

| tier | tasks (snr) |
|---|---|
| **strong** ≥5 | bigbench_qa_wikidata 17.1, boolq 7.1, lambada_openai 6.1, squad 5.2 |
| **usable** 2.5-5 | bigbench_cs_algorithms 4.3, bigbench_dyck_languages 4.0, coqa 3.9, arc_easy 3.0, commonsense_qa 2.6 |
| **weak** 1.5-2.5 | piqa 2.1, hellaswag 1.9, hellaswag_zeroshot 1.8 |
| **DEAD** <1.5 | jeopardy, bigbench_operators, arc_challenge, bigbench_repeat_copy_logic, winogrande, agi_eval_lsat_ar, openbook_qa, winograd, bigbench_language_identification, copa |

**9 of 22 clear snr ≥ 2.5 — and they are exactly the 9 the §4 projection named**, derived
before any swarm Core v2 result existed (between-corpus range x the measured 0.49
within-swarm ratio, against the binomial floor). All 5 predicted-dead tasks measured dead.
The method was slightly optimistic only in the middle band (piqa / hellaswag / bigbench_operators
landed one tier lower). That is strong validation for reusing this projection on a new corpus
*before* spending its fleet.

Caveats: n=30 gives sd-of-sd ~±13%, so borderline tasks (commonsense_qa 2.6, piqa 2.1) may
cross the line; the sample is corpus-unbalanced (10/20). Re-run at n≈300 before locking a
task set.

**Confirmed stable at n=59** (14 dclm / 45 hq): the *same 9* tasks clear snr ≥ 2.5, and the
ordering barely moves — bigbench_qa_wikidata 16.7, boolq 6.7, lambada_openai 6.0, squad 4.7,
coqa 3.8, bigbench_dyck_languages 3.6, bigbench_cs_algorithms 3.4, arc_easy 2.9,
commonsense_qa 2.5. The same 10 stay dead. `Core_v2` mean 6.62, sd 1.31, range 2.87-9.04.
Doubling n changed no verdict, so the 9-task set is robust. Two things still to watch:
`commonsense_qa` sits exactly on the 2.5 line, and the sample is now heavily hq-weighted
(45 vs 14), so the spreads are currently dominated by high_quality — recheck once dclm fills in.

---

## 4b. bpb is far better conditioned than accuracy on the SAME tasks (measured)

From the first 163 completed full-60 runs (96 dclm / 67 hq), across-swarm CV of the 18
backfilled tasks — **none are flat**, and the ones that are *dead in Core v2 accuracy space
are alive in bpb space*:

| task | Core v2 centered-acc range (8 corpora) | bpb CV across swarm (dclm) |
|---|---:|---:|
| `squad` | 1.4 pts — dead | **19.1%** — strong |
| `jeopardy` | 0.2 pts — dead | **14.2%** — usable |
| `coqa` | 5.2 pts | **15.3%** — strong |
| `hellaswag` | 3.3 pts | 7.5% |

Full ranking (dclm CV%): codex_humaneval 26.3, medqa_en 26.2, codex_mbpp 22.5, lambada 22.0,
minerva_math_500 20.9, squad 19.1, sciriff_yesno 16.4, qasper_yesno 16.4, coqa 15.3,
jeopardy 14.2, piqa 12.9, lab_bench_protocolqa 9.8, arc_easy 9.8, arc_challenge 9.5,
csqa 7.6, hellaswag 7.5, winogrande 7.4, lab_bench_dbqa 7.1. dclm shows systematically
higher CV than hq — the dclm swarm spans a wider behavioural range.

**Why this matters for the Core v2 target.** Accuracy at 157M is quantised, near-baseline,
and carries a binomial noise floor that centering *inflates* (boolq's 62% baseline multiplies
its noise by 2.6). bpb is a continuous mean over thousands of documents, so its measurement
noise is negligible and every task retains signal. The backfill has therefore already put
**10 of the 22 Core v2 datasets into bpb space** (arc_easy, arc_challenge, commonsense_qa,
hellaswag, winogrande, piqa, coqa, jeopardy, lambada, squad).

`boolq` and `openbookqa` are **already staged** in the OLMo bundle in every region but are
not in the 56-task suite — adding `boolq/rc_5shot,openbookqa/rc_5shot` in a second merge wave
would take that to **12 of 22** for ~10 min/run (~125 slice-hours over 751 runs). The
remaining 10 (`copa`, `winograd`, `agi_eval_lsat_ar`, the six `bigbench_*`,
`hellaswag_zeroshot`) are **not** staged and would have to be generated in oe-eval request
format via the `build_mmlu_bpb_requests.py` recipe.

### CORRECTION (2026-08-03 10:32): the bpb-over-Core-v2-datasets route is WORSE, not better

The paragraph above proposed fitting bpb over the Core v2 datasets as a better-conditioned
route to a Core-v2-targeted mixture. **Measured on the 59 runs with both metrics, that is
wrong** — against `Core_v2`, with r corrected for its 0.773 reliability:

| proxy | corrected r | r² |
|---|---:|---:|
| devset 42-task bpb | -0.696 | 0.485 |
| full 60-task bpb | -0.701 | 0.492 |
| **the 10 Core v2 datasets in bpb** | **-0.576** | **0.331** |

Measuring the *same datasets* in bpb predicts Core v2 accuracy **worse** than the broader,
disjoint devset. Two reasons: bpb and multiple-choice accuracy are different functionals — bpb
scores how well the gold continuation is modelled, accuracy asks only whether the gold option
out-ranks its alternatives, and a model can improve every option's bpb without changing the
argmax — and 10 tasks make a noisier aggregate than 42.

**Consequence:** to optimise Core v2, fit on **Core v2 accuracy itself** (transformed to a
decreasing quantity per §5), restricted to the 9 signal-carrying tasks from §4. Do not
substitute a bpb proxy. Also note adding the 18 backfilled tasks barely changes the proxy
quality (0.485 → 0.492), so the full-60 suite is a *different* target from the devset, not a
better Core v2 stand-in.

Staging `boolq` + `openbookqa` in bpb is therefore **not** recommended for Core v2 targeting.
It remains valid only if the full 60-task suite is itself the objective.

## 4c. The 42-task devset is only a HALF-proxy for Core v2 — the fleet is justified

Measured on the 59 runs that have both metrics (2026-08-03 10:20):

| | |
|---|---|
| `r(devset42 macro_bpb, Core_v2)` | **-0.612** (correctly signed: lower bpb -> higher Core v2), r² = 0.375 |
| reliability of `Core_v2` | **0.773** — from `se(Core_v2) = sqrt(Σ se_i²)/22 = 0.626 pts` against an observed swarm sd of 1.314 pts |
| attenuation-corrected | **r = -0.696, r² ≈ 0.485** |
| per corpus | hq r=-0.675 (n=45); dclm r=-0.439 (n=14, too thin to lean on) |

So the devset explains only **~half** the true, noise-free Core v2 variation across the swarm.
The relationship is real and directional — a devset-better mixture does tend to be Core-v2-better
— but a devset-*optimal* mixture is **not** Core-v2-optimal. This is the quantitative
justification for running the Core v2 fleet at all: at r² ≈ 0.9 the right advice would have been
to skip it and reuse the existing fit.

Mechanistically this is expected: `olmix_tasks` builds the devset by *removing* every Core v2
task, leaving it code/math-heavy (19 code + 8 math of 42), while Core v2 is dominated by
commonsense QA and reading comprehension. They measure different capabilities and the swarm's
mixtures trade off between them.

Caveats: n=59, hq-weighted 45/14; the noise correction assumes independent per-task binomial
error, which overstates reliability if tasks share failure modes (so the true r² is if anything
*lower*, strengthening the conclusion). Re-run at n≈300.

## 5. What fitting on Core v2 requires (not yet done)

`olmix_solve.solve_mixture` minimizes `sum(exp(t@x))`, which is **convex — `cp.Maximize`
is not DCP and ECOS will refuse it**. So accuracy cannot be fed in directly. Fit on a
*decreasing* transform, e.g. `y = 1 - centered_accuracy` (error); then the law
`exp(log_c) + exp(t·x)`, the dropped-`log_c` constant, and `Minimize` all stay correct.

Also needs changing: `collect_olmix_swarm.task_bpb_scores` (hard-codes
`tasks[t]["bpb"]`; Core v2 is `doc["dclm"]["centered_results"][t]`), the path template in
`sources_from_regions` (flat `<run>_summary.json`, not `<run>/results.json`), and the task
list in `olmix_tasks.build_target_tasks`. Note `compute_core` returns the **string**
`"N/A due to missing tasks: [...]"` rather than a float when a task is absent — that will
raise on `float()`, not silently NaN.

**Scientific caveat, stated once:** `olmix_tasks.py` exists to keep Core v2 *out* of the
objective, because it is the held-out test set. Optimizing a mixture against Core v2 makes
the "optimized beats natural" comparison in-sample. That is a legitimate experiment (does
the method transfer to a different target?) but it is no longer a held-out result — report
it as such, and keep the 42-task devset fit as the clean comparison.

---

## 6. Launch commands (all four parents, 2026-08-02)

Parents are `--priority interactive`, children `batch` (full60) / `interactive` (pilot),
per `feedback_batch_queue_parents`. Parents must run **as Iris jobs** or Iris orphan-kills
the children. Parent memory must stay **< 4GB** or iris demands `--enable-extra-resources`.

```bash
set -a; source .env; set +a
TASKS18="codex_humaneval/gold_bpb_0shot,...,sciriff_yesno/rc_5shot"   # the 18 above

# full-60 BPB backfill — 609 rows
iris --cluster marin job run --job-name olmix-full60-rest \
  --region us-central1 --cpu 4 --memory 1GB --extra cpu --priority interactive --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.scaling_law_sweeps.olmo_bpb.launch_olmo_bpb_manifest \
     --manifest experiments/core_eval_manifests/olmix_swarm_bpb_full60_rest.txt \
     --tasks "$TASKS18" --merge --done-marker _full60.done \
     --results-subpath 'metadata/olmix_swarm_bpb/{run_name}/' \
     --launch --child-priority batch
# ...and _usw4.txt (142 rows) with  --tpu-variants v5litepod-4

# Core v2 pilot — 6 + 2 rows
iris --cluster marin job run --job-name olmix-corev2-pilot-rest ... \
  -- python -m experiments.scaling_law_sweeps.dclm_core.launch_10k_manifest \
     --manifest experiments/core_eval_manifests/_pilot_core_v2_rest.txt \
     --scores-subpath 'metadata/olmix_swarm_core_v2/{run_name}_summary.json' \
     --samples-subpath 'tmp/ttl=30d/olmix_swarm_core_v2/{run_name}.json' \
     --launch --child-priority interactive --wave-size 6
# ...and _pilot_core_v2_usw4.txt with  --tpu-variants v5litepod-4
```

Full Core v2 fleet launched 2026-08-03 00:20 once the pilot returned its timing (§7):

```bash
iris --cluster marin job run --job-name olmix-corev2-w1-rest ... \
  -- python -m experiments.scaling_law_sweeps.dclm_core.launch_10k_manifest \
     --manifest experiments/core_eval_manifests/olmix_swarm_core_v2_rest_w1.txt \
     --scores-subpath 'metadata/olmix_swarm_core_v2/{run_name}_summary.json' \
     --samples-subpath 'tmp/ttl=30d/olmix_swarm_core_v2/{run_name}.json' \
     --launch --child-priority batch --wave-size 8 --wave-delay 120 --keepalive-timeout 86400
# ...and _usw4_w1.txt (139) with  --tpu-variants v5litepod-4
```

The `_w1` manifests are the full set **minus the 8 pilot runs** (579 + 139), because
`--skip-existing` only sees *finished* evals — a pilot still mid-eval would be resubmitted
under the same child job name and collide. 579 + 139 + 8 = **726**, full coverage.

Never pass `--log-samples` on the full fleet: finals are ~1 GB each → ~726 GB.

---

## 7. Cost: MEASURED from the pilot (2026-08-03)

A Core v2 eval is a **sequential 22-task pass over 92,241 docs** (MC tasks expand ~4
loglikelihood requests/doc) on one 4-chip slice at `per_device_eval_parallelism=1`
(effective eval batch **4**). The only *a priori* in-repo quantification was a comment that
a single `bigbench_language_identification` task "can take 30-90 min", with parent
keep-alive budgets of 6–12 h — which implied ~4,400 slice-hours for the fleet.

**The real number is ~2.5 h per model.** From 87 *completed* `dclm-core-*` jobs in the
controller DB:

```sql
SELECT COUNT(*), AVG((finished_at_ms-started_at_ms)/60000.0), MIN(...), MAX(...)
FROM jobs WHERE name LIKE '%dclm-core-%' AND state=4;
-- n=87  avg=158.3 min  min=112.8  max=614.6   (mode ~143-146 min)
```

| | |
|---|---|
| per model | **~113-158 min** (those 87 include d1024-d3584; 157M should sit near the floor) |
| fleet cost | **~1,400-1,900 slice-hours**, i.e. ~15-40 h wall-clock at 50-100 concurrent |
| for contrast, a 42-task BPB eval | ~11 min |

**Do not extrapolate per-model runtime from the first few partials.** `CORE_TASK_MAP` runs
in a fixed order whose *early* tasks are the cheap ones: the first nine landed in ~9 min
(`bigbench_qa_wikidata`, 20,321 docs, in under 3 min), which projected to ~45 min/model and
was wrong by 3-4x. The cost is concentrated later — `hellaswag` at 10-shot, `squad`,
`coqa`, `boolq`, `bigbench_language_identification` — and in `arc_challenge`, which ran ~1
min on one worker and 25+ min on another. A partial mtime that has not moved in 20-30 min
is **normal**, not a hang. Calibrate against the SQL above, not against the early tasks.

### `per_device_eval_parallelism=16` is ~5-6x faster (MEASURED, matched A/B)

Both arms launched together on the *same* checkpoint (`olmix-dclm_10k-...-i0001-wccc4a18e`,
us-east5), same start time, scratch result paths, `--no-skip-existing`:

| arm | result |
|---|---|
| **p16** | **all 22 tasks in 26.9 min** (4.4 min model load, first partial 06:16:18, done 06:38:50); `Core_v2 = 4.9847` |
| p1 (control) | at **28 min still had not finished its FIRST task** (`hellaswag_zeroshot`) — zero partials |
| p1 (historical, 87 completed jobs) | avg 158 min, min 113 |

An effective eval batch of 4 (`per_device_eval_parallelism=1` x 4 chips) badly under-utilises
the slice for a 157M model, and the gap widens on the large-request tasks where the cost is.
At p16 the whole fleet would be **~250-350 slice-hours instead of ~1,400-1,900**.

### ...but scores are NOT batch-invariant. DO NOT adopt it. (verified)

The speedup is real; the assumption that batching cannot move scores was **wrong**, and the
check caught it. Diffing the same checkpoint's per-task partials, p16 vs p1:

| task | p16 | p1 | delta |
|---|---:|---:|---:|
| `arc_easy` acc | 0.406987 | 0.408670 | **-0.0017** |
| `hellaswag_zeroshot` acc | 0.268672 | 0.269169 | -0.0005 |
| `bigbench_qa_wikidata` acc | 0.156587 | 0.155996 | +0.0006 |
| `jeopardy` acc | 0.000000 | 0.000000 | 0 (both floor) |

Same checkpoint, same data, same code — so every delta is a pure eval-configuration artifact.
The cause is numerical, not logical: changing the batch shape changes XLA reduction order and
padding, which perturbs logprobs in the last bits and flips argmax on near-tied multiple-choice
options. Standard for batched harness eval on TPU; it just means **p16 and p1 results are not
comparable to each other**.

Why that is disqualifying here even though the deltas are tiny: the whole point of the swarm is
to regress *small* per-task differences across 726 models. `arc_easy`'s 0.17 pt offset is ~2.5%
of that task's across-swarm spread — negligible as noise, but this is not noise, it is a
**systematic offset between two sub-populations** of the design matrix. Evaluating part of the
fleet at p16 and part at p1 would inject a confound correlated with *when* a run was evaluated,
which is exactly the kind of artifact a log-linear fit will happily absorb into domain weights.

**Decision: keep the entire fleet at `per_device_eval_parallelism=1`,** including sweeper waves
and the us-west4 backlog. Consistency beats a 5-6x speedup here. This reverses the earlier
"use p16 for new waves, it's purely additive" plan — it is additive in compute but *not* in
measurement.

p16 remains the right choice for any **fresh, self-consistent** fleet (a new corpus's swarm
evaluated entirely at p16), where no cross-configuration comparison arises. `--limit` stays off
so scores remain DCLM-comparable.

Track progress *inside* a running eval via the partial dir, which is the only per-task
signal available (the summary is written once, at the end):

```bash
gcloud storage ls -l "gs://marin-<r>/tmp/ttl=30d/olmix_swarm_core_v2/partial/*/*.json"
```

Wave sizing: the Core sweep's hard-won lesson is **waves of ≤5–8, never 40** — 38 bunched
cold-starts once produced 15 hard-failures and 23 hung children with 0/38 completing. The
full fleet uses `--wave-size 8 --wave-delay 120`.

---

## 7b. RESOLVED (2026-08-03 11:33): us-west4 unblocked itself

**Outcome first:** by 11:33 UTC us-west4 was scheduling again — `olmix-full60-usw4` had 11
running and 11 complete after ~7 h of zero. The extraction fleet released enough v5e-4
capacity on its own. **No egress was spent and none is needed**; the 141 us-west4-only Core v2
runs should complete in place.

Two lessons from how wrong the pessimism was: a starved region is not necessarily starved
*indefinitely* — this one looked hopeless right up until it wasn't, and the parent restart
that appeared to be pure damage (§7d) actually helped by resubmitting children straight into
freshly-freed capacity. Prefer waiting over spending when the blocker is someone else's
transient load, and re-check before escalating a cost decision.

The diagnosis below is kept because the *method* (distinguishing a missing pool from a starved
one) is what made it safe to wait rather than pay.

### Original analysis — us-west4 could not run these evals

As of 2026-08-03 06:10 UTC, **283 children are pending in us-west4 with zero started** for
90+ min (141 Core v2 + 142 full-60), while the other three regions run normally.

Diagnosed, and it is *not* a config error:

- `tpu_v5e-preemptible_4-us-west4-a` exists in `scaling_groups`, and the region has run these
  evals before (`olmix-bpb-w12-usw4` has 10 finished children).
- But **no `v5e-preemptible-4` worker is provisioned in us-west4**. The region's pools are
  `v5e-preemptible-16` (88 workers), `v5e-preemptible-32` (40), `v5e-serving-4` (16),
  `v5e-serving-8` (7) — and *every one of them is 100% busy*, largely with this project's own
  `extract-lpv11-10k-*` fleet (~110 running children). Regional quota is exhausted, so the
  autoscaler cannot bring up the 4-chip preemptible pool the evals ask for.

```bash
# the query that shows it: pools present, and how many of each are busy
iris --cluster marin query "SELECT SUBSTR(w.worker_id,11,32) pool, COUNT(t.task_id) running \
  FROM workers w LEFT JOIN tasks t ON t.current_worker_id=w.worker_id AND t.state=3 \
  WHERE w.worker_id LIKE '%us-west4%' GROUP BY pool"
```

**Why it cannot simply be moved:** 141 of the 726 runs exist *only* in us-west4. Evaluating
them elsewhere means cross-region checkpoint reads. Measured, not guessed: one HF export is
**643 MB**, so 141 x 0.643 GB = **90.7 GB**. At this project's recorded egress rate of
$0.08-0.12/GB that is **$7.3-10.9** — at or over the hard $10 cap, so it needs explicit user
sign-off (`feedback_cost_confirmation`). (US-to-US inter-region may bill nearer $0.02/GB
≈ $1.80; confirm the actual rate before assuming the high end.) One copy would unblock **both**
stuck fleets, since the 142 blocked backfill runs are the same checkpoints.

**Update 2026-08-03 09:36 — waiting is no longer a safe default.** The us-west4 pools are
still 100% busy (v5e-16 136 running, v5e-32 24, serving-4 16, serving-8 7) and the
`extract-lpv11-10k-*` fleet has **grown** from ~110 to **228 running + 44 pending**. It is not
draining, so "wait for capacity" has no known horizon.

**ACTION REQUIRED ~2026-08-03 22:10 UTC — re-run both full-60 parents (with
`--no-preemptible`, §7d, and `--child-priority interactive`, §7c).** They started 04:44 UTC
with the default `--keepalive-timeout 43200` (12 h), but were preempted and **restarted at
~10:10 UTC**, which **resets the keep-alive clock** — so the real deadline is ~22:10 UTC, not
the 16:44 implied by the original launch. Always recompute from the *current attempt's*
`started_at_ms` (`state=3` row in `task_attempts`), never from the original submit time. When a parent
exits, iris orphan-kills any child that has not detached. Given §7c (backfill starved to 2
running) and the us-west4 block, a large number of children will still be pending then:

- `full60-rest`: ~282 pending as of 07:23
- `full60-usw4`: 142 pending, none ever started

No data is lost — `--skip-existing` keys off the `_full60.done` marker, so re-running each
parent verbatim resumes exactly where it stopped. But it **must** be re-run or those runs
silently never get their 60-task upgrade. Consider `--child-priority interactive` on the
re-run so the short jobs stop queueing behind Core v2 (§7c).

The Core v2 `w1-*` parents started 05:20 UTC with `--keepalive-timeout 86400` (24 h), so their
deadline is ~2026-08-04 05:20 UTC; the sweeper (§8) covers whatever they drop.

**Read deadlines from the DB, not from CLI log lines.** `iris` prints log timestamps in
**UTC-5** while GCS mtimes and `date -u` are UTC; conflating them put my first estimate of
this deadline 5 hours early. Compute it from `started_at_ms` instead:
```sql
SELECT SUBSTR(job_id,14,26) p,
       ROUND((<now_ms> - started_at_ms)/3600000.0, 2) hrs_running,
       ROUND((43200000 - (<now_ms> - started_at_ms))/3600000.0, 2) hrs_left
FROM jobs WHERE job_id IN ('/<user>/<parent>');
```

**The blocked runs are a RANDOM subset, so losing them costs precision, not validity.**
They are contiguous index blocks (dclm 201-353, hq 250-284) because coordinators were split
by `--index-start/--index-end`. That *looks* like it should bias the design, but index order
is i.i.d. draw order: `sample_flat_dirichlet_swarm` draws each mixture independently and
`sort_and_deduplicate` — despite the name — only drops near-duplicates, it does not sort.
Verified against the real mixtures in `swarm_s42_K363.json`:

| corpus | blocked | remaining K | entropy blocked vs rest | t |
|---|---:|---:|---|---:|
| dclm_10k | 106 | **257** | 1.768 vs 1.859 | -1.25 |
| high_quality_10k | 35 | **328** | 1.905 vs 1.853 | +0.45 |

Mean max-weight and active-domain counts match too. Both remaining K are far above the
`runs >= live_domains + 1` (~76) the fit needs, so **fitting without us-west4 is sound** —
wider intervals, no bias. Note dclm loses 29% of its runs vs hq's 10%, so the two corpora's
fits differ in precision; do not read a dclm-vs-hq difference as corpus behaviour without
accounting for that.

Options, in preference order:

1. **Wait / proceed without them.** Costs nothing, and the fit stays valid at K=257/328.
   *Recommended.* If us-west4 frees up the runs land anyway.
2. Pay the egress (§ above: 90.7 GB, $2-11 depending on the real inter-region rate) and
   re-emit those 141 rows pinned to another region. Optional, not needed for validity.
3. Reprioritise or pause the extraction fleet — **user's call only**; never stop those jobs
   without explicit permission (`feedback_never_kill_running_jobs`).

Do **not** try `--tpu-variants v5litepod-16`: v5e-16 is multi-host and `run_dclm_core_eval`
assumes a single-host slice. The idle-looking `v5e-serving-*` pools are also fully busy and
are a serving reservation, not general eval capacity.

## 7d. LAUNCH PARENTS WITH `--no-preemptible`. One worker preemption killed 219 children.

The single most expensive failure of this run, and it is a gap in the standard launch pattern
in `olmix_corpus_runbook.md` §2/§3 — so it has probably bitten before and will again.

At ~10:50 UTC one worker died. Both `olmix-corev2-w1-rest` and `olmix-full60-rest` were
running **on that same worker**, and it was `marin-tpu-v5p-preemptible-8-us-central1-...`:

```
task_id                              att state started_at_ms  finished_at_ms  error
/michaelryan/olmix-corev2-w1-rest/0   0    7   1785734428168  1785751828868   Worker marin-tpu-v5p-preemptible-8-...
/michaelryan/olmix-corev2-w1-rest/0   1    3   1785751838645
/michaelryan/olmix-full60-rest/0      0    7   1785732226147  1785751828868   Worker marin-tpu-v5p-preemptible-8-...
/michaelryan/olmix-full60-rest/0      1    3   1785751838645
```

Iris orphan-kills a preempted parent's children, so that one event killed **219 of 579** Core
v2 children (`state=6 KILLED`, `error: "Parent task preempted"`) plus the backfill's in-flight
set. It is the dominant loss mode — larger than the 100 task-load failures and the 55 worker
failures combined.

Why it happened: the parents are CPU-only coordinators (`--cpu 4 --memory 1GB --extra cpu`),
and `iris job run --help` claims "CPU-only jobs pinned to non-preemptible" — but they were
scheduled onto a **preemptible TPU** VM's spare CPU capacity anyway. The default did not hold.

**Always pass `--no-preemptible` when launching a coordinator**, and prefer separate parents on
separate workers so one preemption cannot take out two fleets:

```bash
iris --cluster marin job run --job-name <parent> --no-preemptible \
  --region us-central1 --cpu 4 --memory 1GB --extra cpu --priority interactive --no-wait ...
```

Mitigating factors, so this is churn rather than loss: the parents auto-restarted (attempt 1),
re-read their manifests and resubmitted under `--skip-existing`, and every finished per-task
partial survived. Nothing must be recomputed that was already computed — but a lot of
in-flight work was discarded.

**Decode job states before drawing conclusions** (`iris.rpc.job_pb2`): 1 PENDING, 2 BUILDING,
3 RUNNING, 4 SUCCEEDED, 5 FAILED, **6 KILLED**, 7 WORKER_FAILED, 8 UNSCHEDULABLE; tasks add
9 ASSIGNED, 10 PREEMPTED. I spent a while treating state 7 as "preemption" (it is
WORKER_FAILED) and did not notice state 6 at all until it became the largest bucket.

## 7e. The 192GB child request makes every v5e region silently unusable

**Symptom:** us-west4 completed **zero** Core v2 evals across an entire day while its
`v5e-serving-4` slices sat *idle*, and the pool was even scaling itself down (28 → 14
workers) for lack of satisfiable demand. Meanwhile the full-60 bpb fleet — same region, same
4-chip slices — finished all 751 runs there without trouble.

**Cause:** `submit_one` hardcoded `memory="192GB"`, and a v5e VM has **exactly** 192 GiB
(`ct5lp-hightpu-4t`: 112 vCPU / 192 GiB). A request equal to total capacity can never be
placed. Compare the pools in `lib/iris/examples/marin.yaml`:

| pool | RAM | fits 192GB? |
|---|---:|---|
| `v5e-serving` / `v5e-preemptible` | **192 GB** | **no** |
| `v6e-preemptible` | 720 GB | yes |
| `v5p-preemptible` | 448 GB | yes |

That is why it worked everywhere except v5e regions, and why the bpb children (64GB, via
`--memory-gb`) were fine in the very same pool.

**Fix:** `--memory-gb` now exists on `launch_10k_manifest` (default unchanged at 192).
Confirmed by test: a 64GB child scheduled on `v5e-serving-4-us-west4-a` within minutes, and
the 141-run fleet then ran there at 16 concurrent. No OOM — 64GB is ample for a 157M proxy;
the 192GB was copy-pasted from the curation *training* spec.

**The general lesson, which cost hours here:** "region has free workers but my jobs won't
schedule" is not always contention or a variant mismatch. Check the *resource* request
against the pool's actual capacity before asking anyone to free compute — I twice advised
freeing us-west4 TPUs that could never have been used. The diagnostic:

```bash
# what the pool actually offers
grep -A 6 "^  v5e-serving:" lib/iris/examples/marin.yaml
# and confirm a sibling fleet with a smaller ask IS running there
```

## 7c. A slow fleet starves a fast one in the same priority band

Both fleets were launched at `--child-priority batch`. Once the Core v2 waves ramped up, the
backfill collapsed:

| time | full-60 running | Core v2 running | full-60 completions |
|---|---:|---:|---|
| 06:30 | 25 | 24 | ~20 per 7 min |
| 07:23 | **2** | **116** | **0 in 26 min** |

Nothing failed — the scheduler is simply band-fair, and a Core v2 child occupies a 4-chip
slice for ~2.5 h while a bpb child needs ~11 min. Every slice Core v2 takes is one the fast
fleet cannot cycle through, so the short-job fleet stalls behind the long-job fleet.

**Lesson for next time: put the short fleet in a higher band than the long one** (bpb
`--child-priority interactive`, Core v2 `batch`). Interleaving them at equal priority converts
a 3 h job into a multi-day one for no benefit.

Not worth fixing mid-flight here: the pending backfill children are already submitted, so
raising their priority would mean either stopping them (needs sign-off) or submitting
duplicates that re-do the same merge. Both fleets still complete; the backfill just finishes
later. See the keep-alive deadline below — that is the part that actually needs an action.

## 8. Monitoring / resume

**The endgame loop.** Core v2 will not finish in one pass — children die transiently (§9)
and the launched waves are snapshots that miss anything failing afterwards. Drive it with:

```bash
python -m experiments.data_mixing.build_swarm_core_v2_wave --tag w2 [--exclude inflight.txt]
# prints "trained=726 evaluated=N in_flight=M -> needs_eval=K" + needs_eval by region,
# writes one manifest per region-group, and prints the exact launch command for each.
```

"Summary exists" is a **safe** completion test: `run_dclm_core_eval` returns early when
fewer than 22 partials are on disk, *before* writing either the final JSON or the summary
sibling. So a `_summary.json` implies all 22 tasks landed and `dclm.Core_v2` is a real float,
never the `"N/A due to missing tasks"` string. (The collector still type-checks it.)

Repeat until it reports `needs_eval=0`. Always pass the currently-pending run names via
`--exclude`: `--skip-existing` only sees *finished* summaries, so an in-flight run looks
un-evaluated and would be submitted twice. us-west4 gets its own manifest and
`--tpu-variants v5litepod-4` automatically.

Then read the results and decide the task set:

```bash
python -m experiments.data_mixing.collect_swarm_core_v2 --out scratch/core_v2_signal
# per task: sd across the swarm vs binomial noise floor -> snr -> strong/usable/weak/DEAD
```

- Both launchers are **resumable**: `--skip-existing` is on by default and keys off the
  done-marker (`_full60.done`) / scores JSON. A killed parent can simply be resubmitted.
- **Runs excluded from a wave need their own recovery.** The `_w1` manifests deliberately
  omit the 8 pilot runs, so a pilot failure is invisible to that wave — the two that died
  were resubmitted via `_pilot_core_v2_retry.txt` with `--name-suffix -r2`. The sweeper
  above has no such blind spot; prefer it over hand-built manifests.
- Progress:
  ```bash
  iris --cluster marin query "SELECT SUBSTR(job_id,14,17) p, state, COUNT(*) n \
    FROM jobs WHERE job_id LIKE '/michaelryan/olmix-full60-%/%' GROUP BY p,state"
  gcloud storage ls gs://marin-<r>/metadata/olmix_swarm_core_v2/ | wc -l
  ```
- **Count artifacts, not job states** (`feedback_iris_vs_gcs_done`,
  `feedback_health_check_artifact_not_process`). Specifically: a Core v2 child that ends with
  fewer than 22 partials writes **neither** the final JSON nor the summary and still returns
  normally (the guard at `run_dclm_core_eval.py:677` is a bare `return`), so it would be
  recorded as state 4 / success having produced nothing. Job state 4 is therefore not evidence
  of a result. The sweeper keys on `_summary.json` existence, so it catches this correctly.
- **Filter job queries by parent, not just run name.** Every run name appears under several
  fleets at once — `dclm-core-<run>`, `olmobpb-<run>` from the full-60 wave, `olmobpb-<run>`
  from the July waves, plus training jobs. A bare `WHERE job_id LIKE '%<run>%'` returns all of
  them, and it is very easy to read another fleet's finished/364-min row as the eval's status.
  Always anchor on the parent: `job_id LIKE '/<user>/olmix-corev2-w1-rest/%'`.
- **`_error.txt` is never cleaned up — always check its mtime.** The bpb result dirs held 7
  error markers during the full-60 wave; 6 were dated 2026-07-31, left from the *original*
  42-task waves whose runs later succeeded. Only one was current, and that run retried and
  finished. Reading the count alone would have reported 7 phantom failures. Pair the marker
  with `_full60.done` / the task count in `results.json` before believing it:
  ```bash
  gcloud storage ls -l "gs://marin-<r>/metadata/olmix_swarm_bpb/*/_error.txt"   # note the DATE
  ```
- A region with *zero* completions while others progress is **config, not lag** — but check
  before acting. `full60-usw4` sat at 142 pending / 0 started for 35+ min while `rest`
  progressed, which looks exactly like the Trap-1 rejection. It was **not**: the children
  were present in the `jobs` table (so they passed submit, not rejected), and
  `tpu_v5e-preemptible_4-us-west4-a` exists in `scaling_groups`. It was capacity
  contention — the extraction fleet held us-west4's v5e-16/32. Two positive checks that
  distinguish the cases:
  ```bash
  # rejected children never reach the jobs table at all
  iris --cluster marin query "SELECT state, COUNT(*) FROM jobs WHERE job_id LIKE '/<user>/<parent>/%' GROUP BY state"
  # and confirm the pool the child asks for actually exists
  iris --cluster marin query "SELECT name FROM scaling_groups WHERE name LIKE '%us-west4%'"
  ```
- **Read partial mtimes against the real clock.** Core v2 writes one partial per task in
  the fixed `additional_aggregation.json` order, so the count tells you exactly which task
  a child is on. The last five (`squad`, `coqa`, `boolq`,
  `bigbench_language_identification`, and `hellaswag` at 10-shot) are the expensive ones,
  so a long gap late in the run is normal, not a hang.
- **A stalled partial mtime is usually preemption, not a hang.** `copa` (100 docs) appeared
  to run 8 min; in fact the child had been preempted and restarted, and the gap was the
  restart plus model reload. Children run on `*-preemptible-*` workers, so this is routine.
  Confirm before acting — `jobs` only shows the *current* attempt:
  ```bash
  iris --cluster marin query "SELECT SUBSTR(task_id,48,24) run, attempt_id, state, \
    started_at_ms, finished_at_ms, SUBSTR(COALESCE(error,''),1,40) err \
    FROM task_attempts WHERE task_id LIKE '/<user>/<parent>/%' ORDER BY run, attempt_id"
  ```
  State 7 with a `Worker ...` error is a preemption. The per-task partials make this cheap:
  a restart re-does at most the one in-flight task, and attempt 1 resumed with its 5
  finished tasks already banked. Budget ~3-5 min of model reload per preemption when
  estimating fleet wall-clock (so ~60-90 min/model realistic, vs ~45-60 min of pure compute).
- **`ValueError: Failed to load task {...}` hits ONLY the HF-`datasets`-backed tasks, at a
  few percent per attempt.** Across the `w1` fleet the failures were `arc_challenge` (4),
  `arc_easy` (3), `piqa` (1), `copa` (1) — every one loaded through
  `datasets.load_dataset` from the offline cache. The 12 `custom_tasks/` tasks (jeopardy,
  squad, coqa, winograd, commonsense_qa, agi_eval_lsat_ar, the six `bigbench_*`), which read
  local JSONL, have **never** failed. That split is the diagnostic: it points at the offline
  `datasets` cache path (`HF_DATASETS_OFFLINE=1` + `HF_DATASETS_CACHE`), not at the Hub, the
  network, or the model.
  - **Decompose attempts by cause; job-state counts mislead.** A job in state 5 may be
    retrying, and job state does not distinguish a routine preemption from a hard task-load
    death, so counting states gave me three different wrong rates (3%, 25%, 43%) before I
    classified the attempts themselves:
    ```sql
    SELECT CASE WHEN error LIKE '%Failed to load task%' THEN 'task-load'
                WHEN error LIKE '%Worker%'              THEN 'preemption'
                WHEN error IS NULL OR error='' THEN 'none' ELSE 'other' END cause,
           state, COUNT(*) n
    FROM task_attempts WHERE task_id LIKE '/<user>/<parent>/%' GROUP BY cause, state;
    ```
    On `w1-rest` mid-flight: **144 running / 111 task-load / 107 preemption / 3 succeeded**.
    Two roughly equal churn sources. Preemption is inherent to the preemptible pools and is
    what the per-task partials exist to absorb; only the task-load half is fixable.
  - Load-sensitive: near-zero while few children ran, ~30% of attempts once ~150 were
    concurrent. Keep the wave throttle.
  - Two unconfirmed suspects, in order of what to try first if the rate ever climbs:
    1. **Hub 429s.** `run_dclm_core_eval` sets `HF_DATASETS_OFFLINE=1` for *datasets*, but
       deliberately leaves `HF_HUB_OFFLINE` unset, so anything in task construction that
       resolves a repo/tokenizer/metric through `huggingface_hub` still goes out. Rate limits
       on the shared token are a known failure mode in this project
       (`feedback_dclm_core_no_hub_offline`: Hub 429s kill training runs too), and the
       load-sensitivity fits. Note the `_error.txt` files from the July bpb waves have
       `huggingface_hub/utils/_http.py raise_for_status` at the top of the traceback — the
       same signature.
    2. Stale `_builder.lock` / `.incomplete_info.lock` files in `dclm_core_hf_cache`, which
       can make `datasets` treat a cached build as incomplete and attempt a re-download that
       offline mode then refuses.
    Either way the mitigation is the same and already applied: throttle the waves, and let
    the sweeper re-run what fails.
  - **It is not a missing-retry bug.** Both branches of `_load_tasks` are retried —
    `eval_harness.py:1150` wraps the string branch and `_get_task_and_rename` wraps the dict
    branch internally at `:1179` — via `_call_with_retry(max_retries=10, base_delay=5,
    max_delay=300)`, which explicitly handles HTTP 429. That budget is ~20 min of backoff,
    which matches the ~26 min a failing child sits before dying. So these are failures that
    **survive a 20-minute retry window**, i.e. sustained rate-limiting under ~118 concurrent
    children, not a blip. Adding more retries would not help.
  - **`--hub-offline-after-load`: implemented, TESTED, and REJECTED.** The flag exists in
    `run_dclm_core_eval.py` / `launch_10k_manifest.py` (default **off** — leave it off). On a
    single-job trial it cleared `arc_easy`, `arc_challenge`, `copa`, `commonsense_qa` and
    `piqa` — exactly the tasks dying in the main fleet — and then **hung on `openbook_qa`,
    which normally takes ~10 s** (pilot timeline: `piqa` 05:10:51 → `openbook_qa` 05:11:01).
    24+ min on a 10 s task is the `_call_with_retry` backoff burning down, and the job then
    died with `ValueError: Failed to load task {'task_alias': 'openbook_qa'...}` — confirmed
    by the recorded error, not inferred. The flag converted one task's Hub-assisted
    resolution into a hard offline failure: it moved the problem rather than solving it, which
    is precisely the risk that kept it default-off. Note `openbookqa/` *is* present in
    `dclm_core_hf_cache`, so presence in the cache is not sufficient — lm-eval still needs the
    Hub to resolve something for that task.
    **Do not enable it on a wave.** Anyone revisiting this should first confirm whether
    `dclm_core_hf_cache` can satisfy *every* task with `HF_HUB_OFFLINE=1` — test all 22 with
    `--limit 1` before trusting it.
  - Original reasoning behind that flag, kept because the diagnosis still stands:
    `_prepare_offline_dataset_cache`
    sets `HF_DATASETS_OFFLINE=1`, but modern `datasets` routes through `huggingface_hub` and
    respects `HF_HUB_OFFLINE`, which this script deliberately leaves unset because setting it
    at import time breaks `from_hf`'s registry probe
    (`feedback_dclm_core_no_hub_offline`). The model is loaded *before* the task loop, so
    `HF_HUB_OFFLINE=1` could be set **after** model load and before the loop, making task
    construction fully local and removing the Hub from the hot path entirely. Risk: any task
    whose data is genuinely absent from `dclm_core_hf_cache` would then fail deterministically
    instead of eventually. **Validate on a single job (as with the p16 A/B) before applying
    to a wave.**
- **It is not a broken region.** Two eu-west4 pilot children died on `arc_challenge_10shot`. The
  regional `dclm_core_hf_cache` is byte-identical across all four buckets (verified
  recursively, not just the top level — the top-level diff shows only a stray `.lock`), and
  eu-west4 has historically produced a complete 22-task summary *including* `arc_challenge`
  (`Core_v2=11.76`). Root cause is Hub throttling: `run_dclm_core_eval` deliberately does
  **not** set `HF_HUB_OFFLINE` (`feedback_dclm_core_no_hub_offline`), so `get_task_dict`
  still reaches the Hub, and the Hub's ~1000-req/5-min window is the real throughput limit.
  This one was self-inflicted — the full-60 backfill launcher has **no wave throttling** and
  put 751 children up at once alongside the Core v2 fleet.
  - Levanter swallows the underlying exception: `eval_harness.py:1162` logs it via
    `logger.exception` but re-raises a bare `ValueError`, and that message is all iris
    records. With finelog down there is no way to see the real cause — diagnose by
    elimination (cache present? region ever succeeded?) rather than by hunting the traceback.
  - **Recovery is just re-running the parent.** `--skip-existing` skips finished runs, and
    per-task partials survive, so a resubmitted child resumes and re-does only the failed
    task onward. Loop the parent until every run has a `_summary.json` (runbook Trap 3).
    Never kill the running children to do it.
- Children showed `priority_band=3` while running despite `--child-priority interactive`
  (pending ones showed 2). Work flowed regardless — 655 of my tasks running at once — so
  this was not worth chasing; see `feedback_iris_budget_demotes_priority_band` if
  throughput ever actually stalls.
- finelog is often down; use `iris job bug-report <job>` instead of `iris job logs`.

## 9. Gotchas hit while setting this up

- **Local `gcsfs`/`aiohttp` fails SSL cert verification on this laptop**, silently returning
  zero results from GCS listings. Fix: `export SSL_CERT_FILE=$(.venv/bin/python -m certifi)`
  (and `REQUESTS_CA_BUNDLE`). Cost ~20 min of phantom "empty bucket" results.
- **`--name-suffix -r2` fails; use `--name-suffix=-r2`.** Suffix values conventionally start
  with `-`, and argparse reads a leading-dash value as the next flag: the parent dies
  instantly with `error: argument --name-suffix: expected one argument` and submits zero
  children. It looks like a scheduling failure (parent in state 5, no children) rather than a
  CLI typo. Cost three launches here; the wave builder now emits the `=` form.
- `iris job run --memory 4GB` is rejected for coordinator jobs; keep parents at 1GB.
- The `dclm_core_hf_cache` differs across regions by exactly one stray `.lock` file — the
  33 real entries are byte-identical, so cross-region Core v2 scores remain comparable.
- `olmo_in_loop_evals/oe_eval_tasks` has 118 dirs in us-central1/us-west4 but 61 in
  us-east5/eu-west4 — the difference is the 57 per-subject MMLU dirs, which nothing uses
  (the fit uses the 4 `mmlu_<category>` dirs, present everywhere). Benign.

## 10. BLEnD cultural-knowledge sweep (2026-08-05)

Third benchmark over the same 726 swarm checkpoints. **Diagnostics only** — BLEnD is
deliberately NOT part of the olmix mixture objective, so it writes to its own tree and
cannot leak into a fit via `--tasks all`.

| | |
|---|---|
| Suite | 16 country/region tasks, `blend_<country>/rc_5shot` |
| Form | OLMES rc cloze, 5-shot, `compute_gold_bpb: true`, `limit: null` |
| Size | ~1,120–1,450 requests/country, ~20k total (≈1/5 of Core v2) |
| Builder | `experiments/scaling_law_sweeps/olmo_bpb/build_blend_bpb_requests.py` |
| Source | `nayeon212/BLEnD`, `mc_questions_file_v1.1.json` (5,081 docs) |
| Results | `gs://<bucket>/metadata/olmix_swarm_blend/<run_name>/results.json` |

Countries: Algeria, Assam, Azerbaijan, China, Ethiopia, Greece, Indonesia, Iran, Mexico,
North_Korea, Northern_Nigeria, South_Korea, Spain, UK, US, West_Java. Task dirs are
**lowercase** (`blend_north_korea`); the HF `dataset_name` keeps the capitalised form.

### Staging (Trap 1 again)

BLEnD was present in us-east5 / us-central1 / europe-west4 but **missing from us-west4**,
where 107 checkpoints live. That is the identical shape of the bug that hid 132 us-west4
Core v2 evals: a child asking for an absent task fails at task-load, which reads as
ordinary transient churn rather than missing data. Staged us-central1 → us-west4 with
`gcloud storage cp -r` (the builder forbids `rsync -d`), 712,754 bytes, US-to-US.
**All four regions verified byte-identical: 16 dirs / 32 files, matching sizes.**

Check before any future wave:

```bash
for B in marin-us-east5 marin-us-central1 marin-eu-west4 marin-us-west4; do
  echo "$B $(gcloud storage ls gs://$B/eval_datasets/olmo_in_loop_evals/oe_eval_tasks/ | grep -c blend)"
done   # must print 16 four times
```

### Launch

Manifests built by preferring non-us-west4 placement for the 50 dual-region checkpoints
(us-east5 > us-central1 > europe-west4 > us-west4), which shrinks the v5e-only tail from
~157 to 107:

- `olmix_swarm_blend_rest.txt` — 619 rows (us-east5 172, eu-west4 314, us-central1 133)
- `olmix_swarm_blend_usw4.txt` — 107 rows, needs `--tpu-variants v5litepod-4`

Launch script: `scratchpad/launch_blend_full.sh <child-priority> <suffix>`. Parents are
`--priority interactive --no-preemptible`; children default `batch`, escalatable to
`interactive` by launching a second parent with a new suffix — `--skip-existing` means the
new wave picks up only what is left, so **no need to kill the first parent**.

Wave throttle relaxed from the 8/420s default to `--wave-size 16 --wave-delay 90`: the
default would have spent ~9 h merely submitting 619 children. 16 per 90 s keeps the HF Hub
well under its 1000-request / 5-min per-token cap.

### What went wrong in the BLEnD wave (2026-08-05) — read before the next launch

**1. Escalating priority by ADDING a parent doubles the launch rate.**
Batch children were starved (measured: `running=1` against `pending=143 -> 222 -> 312`,
because band 3 held 772 running tasks of which 683 were *my own* extraction fleet). The fix
was interactive priority — but launching `-i1` parents while the `-b1` parents were still
alive put ~37 children/min against the Hub instead of ~15. Adding a fifth parent
(`euw2usw4`) made it worse.

Result: **87 BLEnD children killed** by `Job exceeded max_task_failures`, every one of them
`We had to rate limit you, you hit the quota of 1000 api requests per 5 minutes period` —
and the blast radius reached **20 failures in `olmix-corev2-coord` curation TRAINING** plus
one job belonging to another user. The cap is per TOKEN across every job, exactly as
`feedback_dclm_core_no_hub_offline` warns.

**When escalating priority, STOP the old parents in the same breath.** `--skip-existing`
makes the new wave pick up only what is left, so the old wave is pure duplicate load.

**2. A failure query that omits state 6 reports "no failures" during a mass kill.**
Terminal states are `5 FAILED, 6 KILLED, 7 WORKER_FAILED, 8 UNSCHEDULABLE`. A child that
exhausts `max_retries_failure` lands in **6**, not 5. Spot-checks using `state IN (5,7,8)`
returned clean while 87 children were dying. Always use `IN (5,6,7,8)`.

**3. Progress counted as raw `results.json` across buckets over-counts once any run is
evaluated in two regions.** Verify completion with DISTINCT run names (the collector
dedups); the raw count can pass the target while runs are still missing.

### `--hub-offline`: PROMISING BUT NOT YET RELIABLE (measured, do not trust the single smoke)

`launch_olmo_bpb_manifest --hub-offline` sets `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`.
This removes the failure mode instead of pacing around it.

The historical "NO HF_HUB_OFFLINE" verdict predated the fallback now at
`run_olmo_bpb_eval.py:223-237`: `HFCheckpointConverter.from_hf` probes every registered
config's default HF repo and raises offline on a cache miss, but the runner now catches that
and resolves `LevConfig` from the checkpoint's own local `config.json`. Everything else the
eval needs is already local — requests are staged raw text, and the checkpoint carries its
own tokenizer.

**Measured on a 13-child wave, 2026-08-05: 7 succeeded, 6 HARD-FAILED** with
`Check your internet connection or see how to run the library in offline mode`. A single
smoke checkpoint passed (16/16 tasks, 5,001 docs, ~180 s, matching the online smoke exactly)
— **and that one pass was not predictive.** Do not generalise a one-checkpoint offline smoke
to a wave; the failure is checkpoint/cache dependent, so a ~54% success rate looked like
100% at n=1.

Consequence: the staged `core_tasks_hub_cache` does **not** cover every repo the load path
touches (cf. `feedback_eval_hub_cache_needs_all_registry_tokenizers`). To make offline real,
warm the cache for every registry tokenizer + `gpt2` + the Llama-3.1 mistral-regex repo,
mirror to all four regions, and only then re-measure on a FULL wave rather than one child.

**When offline DOES succeed it is bit-identical to online** — verified by running one
checkpoint both ways: `max |bpb diff| = 0.000e+00` across all 16 tasks, same 5,001 docs. So
the failure mode is purely availability (cache coverage), never silent numeric drift, and
offline-produced results can be mixed freely with online ones in the same dataset.

Until that is done: **run waves online, and control Hub load by not overlapping parents**
(see the incident above) rather than by going offline. An offline cache miss is a hard child
failure, not a silent online fetch, so a partial cache converts a rate-limit risk into a
guaranteed loss.

### BLEnD result (COMPLETE, 726/726 — 2026-08-05)

363 `dclm_10k` + 363 `high_quality_10k`, all 16 countries, 0 partial, 0 missing, bpb range
1.483–6.953. Results: `gs://<bucket>/metadata/olmix_swarm_blend/<run>/results.json`.
Collector + analysis: `experiments/data_mixing/collect_swarm_blend.py`.

| | dclm_10k | high_quality_10k |
|---|---|---|
| PC1 share | 92.0% | 87.0% |
| eigenvalues 2–4 | 3.4 / 1.2 / 0.8% | 3.5 / 1.5 / 1.3% |
| residual after PC1 | 8.0% | 13.0% |
| per-country sd | 0.23–0.40 | 0.11–0.15 |

**BLEnD is ~90% a single general-quality axis at 157M proxy scale.** PC1 loadings are nearly
uniform (0.23–0.26 across all 16 countries), i.e. the dominant axis is "the model got better
at everything", not any cultural contrast. `high_quality` mixtures move BLEnD roughly half
as much in absolute terms as `dclm` ones (sd ~0.12 vs ~0.27) — expected from a filtered,
more homogeneous corpus — and its larger residual *share* is mostly a smaller PC1, not more
cultural signal. The collector's canned verdict flips between the two corpora only because
of an arbitrary 0.9 threshold; do not read that as a qualitative difference.

**The residual does not replicate across the two independent swarms**, so it cannot be
treated as culture-specific structure:

- residual correlation-matrix agreement between corpora: **r = 0.110**
- PC2 / PC3 loading agreement: |r| = 0.35 / 0.38
- pair replication: **35/120 vs 28.0 expected under independence → 1.25x**, i.e. mostly noise

Three linguistically coherent pairs do replicate in both swarms — UK–US (+0.305 / +0.666),
Mexico–Spain (+0.220 / +0.581), Indonesia–West Java (+0.167 / +0.161, and West Java is *in*
Indonesia). These were selected after seeing the ranking, so they are suggestive, not
established. A pre-registered test of just those pairs on a future swarm would settle it.

**Conclusion: cultural coverage is NOT a separable mixing target at this scale.** Keeping
BLEnD out of the olmix objective — which the separate results path enforces structurally —
is the right call on evidence, not just convention. Revisit only at larger proxy scale,
where per-country signal may separate from the general-quality axis.

Difficulty ordering is stable and sane throughout (easiest US 1.76 / UK 1.87 / Mexico 1.85;
hardest Ethiopia 2.43 / Assam 2.42 / South Korea 2.35) — the expected Anglocentric gradient
for web-corpus models, which is the main sanity check that the benchmark is wired correctly.

### IMPORTANT CAVEAT on the BLEnD result — it is all English, and gold-bpb ignores the distractors

Verified from the staged requests, 2026-08-05 (user raised the question):

**BLEnD as staged is entirely ENGLISH.** Prompt (`"The following are questions about everyday
life in the United States."`), questions, and answers are all English; `native_id` carries the
`-en-` variant marker. Only the cultural *content* varies by country. Therefore the UK–US and
Mexico–Spain residual correlations reported above are **cultural/topical**, NOT linguistic —
an earlier framing of them as "linguistic family" pairs was wrong. Language being constant is
a design strength: any per-country signal is cultural by construction.

**Gold-bpb discards what makes BLEnD a cultural benchmark.** Each doc carries four choices
drawn from *other countries' correct answers* — `blend_choice_countries` records which country
each distractor came from, e.g. `{A: Assam, B: US, C: Northern_Nigeria, D: Spain}`. That
construction is exactly the cultural-discrimination test. `run_olmo_bpb_eval` keeps only the
gold continuation (oe-eval's ce_loss/bpb filter) and scores its bpb, so **the distractors are
never used**. Many golds are culture-neutral common English words (`soccer`, `cake`, `parks`),
shared verbatim across countries.

So "~90% one general-quality axis" may reflect the **weakness of the bpb instrument** as much
as any absence of cultural signal. Before concluding that cultural coverage is unmixable,
re-run BLEnD in **multiple-choice accuracy** form — the staged `requests.jsonl.gz` already
contains all four choices and the gold index, so no rebuild is needed, only a scorer that
ranks the choices instead of filtering to gold.

**Data-quality note:** 127/5001 = **2.5% of gold answers are the boilerplate string
`not-applicable`**, unevenly distributed — azerbaijan 5.6%, ethiopia 5.5%, algeria 4.7% down
to mexico 0.9%. Those docs score boilerplate prediction, not cultural knowledge. Tested as an
explanation for the residual instability and REJECTED as inconclusive: corr(na%, per-country
sd) = +0.45 (dclm) but −0.14 (hq), and algeria is high-boilerplate yet among the most stable.
n=16 countries is too small to resolve it. Exclude these docs if BLEnD is ever rebuilt.

### BLEnD-objective mixtures: the answer is NO, Science & Tech is not upweighted (2026-08-05)

Four fits, R=30B / k=20, λ ∈ {0.05, 0.01}, both corpora, via `run_olmix_fit --metric blend`.
All valid: 363 runs each, 0 degenerate tasks, per-task R² 0.82–0.90, live domains above the
identifiability bar (dclm 75/118 need 76; hq 73/120 need 74).

Outputs: `gs://marin-us-east5/metadata/olmix/<corpus>/blend_lambda{005,001}/mix_R3e+10_k20.json`

**Science & Tech. (c19) weight, BLEnD vs the other objectives:**

| corpus / λ | natural | bpb | core_v2 | **BLEnD** | BLEnD/nat |
|---|---:|---:|---:|---:|---:|
| dclm λ=0.05 | 8.33% | 15.87% | 11.73% | **6.53%** | 0.78x |
| dclm λ=0.01 | 8.33% | 23.56% | 21.68% | **5.02%** | 0.60x |
| hq λ=0.05 | 7.03% | 17.93% | 9.95% | **7.39%** | 1.05x |
| hq λ=0.01 | 7.03% | 31.24% | 16.47% | **6.37%** | 0.91x |

bpb and Core v2 both push Science & Tech to 1.4–3.7x natural. **BLEnD leaves it at or below
natural in all four fits.** Software Dev. behaves the same way: bpb 2.2–2.5x vs BLEnD
0.42–0.73x.

**Where BLEnD puts the weight instead — everyday/social life, consistently:**

- **Education & Jobs**: 2.6x–4.2x natural in ALL FOUR fits (top topic in three of them)
- **Social Life**: 2.2x–3.5x (dclm), **Entertainment** 1.4x–1.6x (hq), **History** ~2.1x (dclm)
- **Fashion & Beauty** 4.6x–9.3x and **Travel** 2.2x–3.0x (hq), **Food & Dining** 1.2x–1.9x

This is coherent with the benchmark: BLEnD asks about school snacks, popular sports, festival
foods, everyday customs. The mixture that best predicts those answers loads on everyday-life
prose, not technical writing. The upweighted set differs in flavour between corpora (hq picks
Fashion/Travel, dclm picks Social Life/History) but the axis is the same.

**Divergence exceeds the bpb↔core_v2 baseline in every fit**, i.e. BLEnD is not a restatement
of either:

| corpus / λ | TV(BLEnD, bpb) | TV(BLEnD, core_v2) | TV(bpb, core_v2) ref |
|---|---:|---:|---:|
| dclm λ=0.05 | 0.256 | 0.235 | 0.192 |
| dclm λ=0.01 | 0.461 | 0.548 | 0.435 |
| hq λ=0.05 | — | — | — |
| hq λ=0.01 | 0.556 | 0.512 | 0.491 |

**IMPORTANT correction to the PC1 finding above.** "BLEnD is ~90% one axis" describes
across-swarm *outcome variance* — the 16 countries rise and fall together. It does NOT mean
that axis responds to the same DATA as the bpb devset's axis, and these mixtures prove it does
not: TV(BLEnD, bpb) exceeds TV(bpb, core_v2) in every fit. "One axis" and "the same axis as
general quality" are different claims; conflating them understated BLEnD's distinctiveness as
a mixing signal. The earlier conclusion that cultural coverage is not a separable *target*
still holds for the per-country contrast; what is separable is the everyday-life vs technical
prose direction, which is a topic axis rather than a cultural one.

These mixtures are NOT registered as trainable arms. Registering one would mean training on a
diagnostic whose gold-bpb scoring ignores BLEnD's cross-country distractors (see the caveat
section above) — worth doing only alongside the MC-accuracy rebuild.
