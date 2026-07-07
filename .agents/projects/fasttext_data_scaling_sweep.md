# fastText useful-classifier — data-size scaling sweep (at the correct 12:1 ratio)

**Question we never answered:** does *more training data* help the fastText stage-1
classifier **once the negative:positive ratio is correct (12:1)**? The existing sweeps
varied size only at the wrong (1:1) ratio — where 200k→1M *hurt* (0.472→0.425) — and
varied ratio only at one size (~190k pos → 0.597). The `1M @ 12:1` run (`natratio1Mpos`)
was started but never landed a number. This sweep fills the missing cell.

See [[fasttext_useful_classifier_findings]] for the full prior findings.

## CLUSTER LIMIT (discovered 2026-06-24 at launch) — sweep is DOWNWARD only

us-central2 caps **every** VM at **100 GB disk** (`disk: 100GB` for all groups in
`lib/iris/examples/marin.yaml`, incl. the 720 GB-RAM pods) and the only non-TPU VM is
`n2-highmem-2` (**2 vCPU / 16 GB RAM**). The dataset lives **only** in us-central2 (not
mirrored), so we can't move to a bigger-disk region without a forbidden cross-region read.
The 80-WARC `train.txt` is **14.85 GB gzipped → ~70–90 GB uncompressed**, i.e. it nearly
fills 100 GB by itself — so **160/320 WARCs cannot be materialized** with the full-HTML
recipe. Going *above* the shipped data size needs Phase 2 (document truncation) or a
bigger-disk environment.

RAM is fine: `min_count=500` sizes the weight matrix to the *pruned* vocab (~1 GB), so peak
RAM is a few GB even at 80 WARCs — well within 16 GB.

## Design (downward sweep)

Scale the **number of front WARCs**, hold everything else at the **shipped recipe**.
`--max-per-class` is set huge so each run takes ~all useful docs per WARC (~2,367/WARC),
so size is driven purely by WARC count. The **80-WARC point already exists as the shipped
winner (`body_strip_natratio200kpos_mc500`, F1 0.594)** — reuse it as the anchor; the new
jobs fill in the previously-empty sub-190k region (we had zero 12:1 points below ~190k).

Fixed across all runs (the shipped recipe):
- `--representation body_strip`
- `--neg-per-pos 12 --train-sample front`
- `--fixed-config --epoch 5 --lr 0.1 --dim 100 --word-ngrams 2 --minn 0 --maxn 0 --loss softmax`
- `--min-count 500`  ← the "~1GB model" knob (singleton markup pruning, ~0 quality loss)
- Eval: frozen snapshot-stratified ~12:1 test (`full_prep_body_strip/test/*.txt.gz`), best-F1 over threshold sweep.

| train-warcs | ≈ useful (pos) | disk req | status |
|---|---|---|---|
| 10 | ~24k | 30 GB | LAUNCHED 2026-06-24 (`ft-scale-w10`) |
| 20 | ~47k | 45 GB | LAUNCHED 2026-06-24 (`ft-scale-w20`) |
| 40 | ~95k | 70 GB | LAUNCHED 2026-06-24 (`ft-scale-w40`) |
| **80** | **~189k** | — | **REUSE shipped `body_strip_natratio200kpos_mc500` → F1 0.594** |
| 160 | ~379k | (~150 GB) | BLOCKED by 100 GB disk cap — Phase 2 (truncation) |
| 320 | ~758k | (~300 GB) | BLOCKED by 100 GB disk cap — Phase 2 (truncation) |

min_count fixed at 500 by decision 2026-06-24 (match shipped). Caveat: at fixed min_count,
vocabulary grows with data, so a positive result blends "more data" with "bigger effective
vocab" — acceptable because we care about the shipped-recipe curve.

## Phase 2 (only if the downward curve is still climbing at 80 WARCs)

To reach 160/320 WARCs within the 100 GB disk cap, add a `--max-chars` truncation knob to
`to_fasttext_text` (fastText bigrams mostly use the head of the doc). First validate
truncation is ~free by re-running the 80-WARC point truncated and comparing to 0.594; then
run 160/320 truncated as their own consistent curve. Until then, "does *more* data than
shipped help?" stays open.

## Execution

CPU-only, `us-central2` (dataset is local there — no cross-region egress). Each job:
`pilot` (prep+train+self-eval) `&&` `eval` (natural-ratio) → leaderboard merges both.
Output dirs: `gs://marin-us-central2/classifiers/useful_fasttext/body_strip_scale_w{N}_mc500/`.

Launcher: `scratch/launch_fasttext_scaling.sh`. After all 5 finish, run the `leaderboard`
subcommand to regenerate `LEADERBOARD.md` / `leaderboard.json`, then plot natural-F1 vs
train_pos.

## Reporting convention — x-axis is TOTAL TRAINING LINES (pos+neg), series = ratio

Report/plot natural-F1 vs **total training lines** (= positives + negatives = docs the model
sees = the disk-bound quantity), NOT positives. Faceted/colored by neg:pos ratio.

Why this matters (recomputed 2026-06-24 from `leaderboard.json`):
- In line-terms the "1M run" (`dclm1m`, 1:1) is only **1.77M lines** — *smaller* than the 12:1
  winner's **2.46M lines** (F1 0.597). So "scaled to 1M → worse" was an artifact of the
  *positives* axis. The model trained on the **most lines is also the best**. Data didn't
  hurt — **1:1-composed data hurt**.
- Lines alone don't predict F1: at ~1.75M lines, 8:1 → 0.556 vs 1:1 → 0.425. Ratio is the
  lever; hold it fixed to read a clean size curve.

Three series on one axis:
- 1:1 — dclm200k→dclm1m: 0.38M→1.77M lines, **declining** 0.472→0.425.
- ratio sweep (fixed ~192k pos) — 0.39M→3.21M lines, rising 0.477→0.578 (this is really
  "add negative lines", i.e. raise ratio).
- **12:1 (this sweep)** — ~0.31M→2.46M lines, fixed ratio: the clean size curve. w80=shipped (0.594).

## Results (natural-ratio best F1 vs total training lines, fixed 12:1)

FRONT arm (matches shipped recipe) — COMPLETE, PLATEAUS ~0.59:
| warcs | pos | total lines | natural F1 |
|---|---|---|---|
| 10 | 26,414 | 335,443 | **0.546** |
| 20 | ~53k | 678,493 | **0.569** |
| 40 | ~106k | 1,419,899 | **0.593** |
| 80 (shipped) | 189,388 | 2,456,898 | **0.594** |

KEY: front curve flattens hard — w40→w80 (1.7× data) gained +0.001. fastText saturates ~0.59 by
~1.4M lines; shipped model already at the ceiling. Open Q the strat+trunc push answers: is the
plateau MODEL CAPACITY (→ stratified w320 also flattens; gains need ModernBERT) or FRONT's diversity
ceiling (→ stratified breaks past 0.594)?

STRATIFIED arm (diversity control; launched 2026-06-25 — ft-scale-w{10,20,40}-strat):
| warcs | total lines | natural F1 |
|---|---|---|
| 10 | 522,178 | **0.561** |
| 20/40 | ~0.7M/~1.4M | running |
| 80 (anchor) | ~2.5M | 0.576 (existing sweep_strat_n12) |

STRATIFIED UNTRUNCATED results (lines → natF1): 522k→0.561, 1.005M→0.586, **1.8M→0.611**.

BREAKTHROUGH (revises earlier "tracks front" read): stratified KEEPS CLIMBING where front PLATEAUED.
- front: 1.42M→0.593, 2.46M→0.594 (flat).
- strat: 1.005M→0.586, 1.8M→**0.611** (+0.025, still rising) — ABOVE the shipped 0.594.
Gap front→strat grows with scale: tie@522k → +0.006@1M → +0.017@1.8M. DIVERSITY breaks the plateau:
front's redundant early-snapshot data saturates; diverse data keeps teaching fastText. (Compare on lines axis;
"wN" strat ≠ "wN" front — strat WARCs are ~1.27× denser.)

Matched truncation penalty (same 40 strat WARCs): untruncated 0.611 vs 8k-trunc 0.489 = **−0.122**. Severe.

UNTRUNCATED BIG RUNS — first attempt FAILED with `[Errno 28] No space left on device` (train.txt overflowed
the tmpfs, which is sized to --memory; stratified denser than projected):
- w160-strat untruncated train.txt >220GB → relaunched `ft-scale-w160-strat-untrunc2` at **--memory 340GB** (running).
- w320-strat untruncated ≈475GB > 400GB v4 worker RAM → INFEASIBLE in-region (v6e=720GB/v5p=448GB are out-of-region
  → forbidden cross-region reads of the us-central2 dataset). RAM is the hard wall. Truncated w320-tc8k = lower bound only.

At fixed 12:1, front F1 **rises** with data (0.546 → 0.569 → 0.594), opposite of the 1:1 series
(0.472 → 0.425). Decelerating toward the anchor. Front>stratified at w80 (0.594 vs 0.576); the
stratified arm tests whether that gap holds across scales / is just sampling noise.
NOTE: pilot self-test F1 understates natural F1 (w10 pilot 0.492 vs natural 0.546) — only quote eval_natural.

## Reaching >w80 (190k pos): disk-bypass investigation (2026-06-25)

fastText (src/fasttext.cc) requires ONE seekable input file: `train()` rejects stdin ("Cannot use
stdin for training!"), opens a single `args_->input`, and `trainThread` seeks to `threadId*size/nthreads`.
No multi-file/dir/glob. BUT it streams via ifstream+seek — never loads to RAM — so the constraint is
seekable *storage*, not memory. Paths to w160/w320 within the 100GB local cap:
- **gcsfuse-mounted train.txt** — recipe-faithful, no truncation; needs gcsfuse in task image; ~6× in-region (free) re-reads.
- **`--max-chars` truncation** — fast, fits local disk; recipe change (validate ≈free at w80).
- multi-file/stdin streaming — NOT possible (fastText is single-file).

## THE PUSH: stratified + truncated, scaling to w160/w320 (2026-06-25)

Goal: drive natural F1 higher with MORE + MORE-DIVERSE data (user: better classifier → significantly
cheaper downstream training). Implemented `--max-chars` head-truncation in
`fasttext_useful_classifier.py` (to_fasttext_text/fasttext_line/PilotConfig/run_eval + CLI on both
pilot & eval; unit-tested). Truncation to 8000 chars shrinks train.txt ~4× (w320 ~300GB→~55GB, fits the
100GB local cap) AND cuts training tokens ~4× (w320 trains in hours). Eval truncates the test the SAME
way (`--max-chars 8000`) to avoid train/serve skew. Used stratified sampling (diverse across all 35 snapshots).

Launched `ft-scale-w{40,80,160,320}-strat-tc8k` (batch, us-central2, 8cpu/96GB):
- w40 = free-ness check vs the in-flight UNtruncated stratified w40 (does 8k-char truncation cost F1?).
- w80/160/320 = the new diverse, above-shipped points (w160/w320 were impossible untruncated).

| warc-scaling option to reach >w80 | status |
|---|---|
| `--max-chars 8000` truncation | running (w40/80/160/320-strat-tc8k); fast; validate free-ness at w40 |
| **tmpfs/RAM (UNtruncated)** | **running (w160/w320-strat-untrunc)** — see below |
| gcsfuse (recipe-faithful) | DEAD: /dev/fuse absent in task container, can't self-mount (no privilege) |

### KEY ENV FINDING (probe ft-envprobe, 2026-06-25): train.txt is already in RAM, not disk

Container probe: `/dev/shm`=64M (Docker default, useless); **`/tmp` is a RAM-backed tmpfs sized to
`--memory`** (200GB request → /tmp=201G); overlay `/`=97G boot disk; `/dev/fuse` ABSENT. Python tempdir=`/tmp`.
So `tempfile.TemporaryDirectory()` → `/tmp` → **train.txt lives in tmpfs/RAM, NOT the 100GB disk** — the
disk cap never bound it. Real limit = worker RAM (v4=400GB). So UNtruncated w160/w320 just need big `--memory`:
- w160 untrunc (~150GB train.txt) → `--memory 220GB` (comfortable). LAUNCHED `ft-scale-w160-strat-untrunc`.
- w320 untrunc (~325GB train.txt) → `--memory 384GB` (tight vs 400GB worker; may OOM). LAUNCHED `ft-scale-w320-strat-untrunc`.
Both NO --max-chars (full docs), stratified. Untruncated = slow (no token reduction): w320 could be ~10-15h.
If w320-untrunc OOMs, truncated w320-tc8k is the fallback.

NOTE: 3 pre-existing RUF001 ambiguous-unicode lint warnings (×/≥/— in _leaderboard_markdown) remain;
unrelated to this change — clean up before any commit.

## TRUNCATION VERDICT: NOT free — costs ~0.10 F1 (2026-06-25)

w40-strat-tc8k (8000-char truncated) = **0.489** @ 1.8M lines vs untruncated curve ~**0.59** at the same
lines → **8k-char head-truncation costs ~0.10 F1**. Likely cause: useful pages are LONG (substantive),
no_useful are SHORT (boilerplate/nav) → truncation destroys the document-LENGTH signal separating classes.
=> Truncated arm (w80/160/320-tc8k) is a pessimistic LOWER BOUND, not the real curve. UNTRUNCATED (RAM/tmpfs)
is the answer — w160/w320-strat-untrunc are the critical path. (And since tmpfs removed the disk limit,
truncation has no remaining purpose.) The w40 free-ness check did its job before we trusted big truncated points.

## train-from-prep: the durable-shard shortcut (2026-06-25)

The slow ~2min/WARC prep (parquet read + body_strip regex) was ALREADY DONE: `full_prep_body_strip/train/`
has **2930 untruncated gz shards (one per WARC, 764 GiB)** — same prep that built the eval test set (test/=35).
So we never need to re-prep: pick N shards, concat into tmpfs train.txt, train. Added `train-from-prep`
subcommand (selects N stratified shards, concats at neg_per_pos=12 per shard, optional max_chars, trains,
saves model+metrics; NO train.txt.gz re-upload — shards are the durable copy). Unit-tested concat (12:1 cap +
truncation + neg-starvation). Eliminates prep (~5-11h) → w160 untrunc drops ~25h→~7h; data durable so a
preemption only loses the train step.

ROOT CAUSE of Larry's kill confirmed: CPU-only iris jobs DEFAULT to **non-preemptible (=reserved pool)**.
Fix: pass **`--preemptible`** to land on `tpu_v4-preemptible_8` (off the reserved hero-run capacity).

Launched (train-from-prep, untruncated, stratified, 12:1, mc500, cpu64, --preemptible):
- `ft-w80-prep`  (mem 200GB) — running on v4-preemptible-8.
- `ft-w160-prep2` (mem 340GB) — running on preemptible (was `-prep`/`-prep-np`, see below).

PREEMPTIBLE vs NON-PREEMPTIBLE (2026-06-25): non-preemptible avoids cloud-preempt but the scheduler
DEMAND-ROUTES CPU jobs to `tpu_v4-reserved_8` — the HOT hero-run pool (demand=8) — not the idle
reserved_16..1024. So non-preemptible = guaranteed hero-run collision (worse than preemptible's
probabilistic cloud preempt). Stay PREEMPTIBLE. fastText can't checkpoint → a preempt loses the train
step (~2.5h/~5h), but train-from-prep keeps PREP durable, so just relaunch training on preempt. To fully
avoid preempt-loss: wait out the hero run, or find a non-reserved_8 stable pool (scheduler won't route there).

w320 untrunc cross-region price-out (NOT executed): copy 320 train + 35 test shards ≈ **93 GB** from
marin-us-central2 → marin-us-east5 (same continent ≈ $0.02/GB) ≈ **~$2**, then run on v6e (720GB RAM) in
us-east5-b. (Earlier "$15-40" was wrong — that was inter-continent.)

## HEADLINE RESULT (2026-06-26): stratified+untruncated beats shipped, still climbing

Stratified UNTRUNCATED curve (natF1 vs total lines, 12:1, mc500):
| lines | natF1 |
|---|---|
| 522k (w10) | 0.561 |
| 1.005M (w20) | 0.586 |
| 1.8M (w40) | 0.611 |
| **3.58M (w80, via train-from-prep)** | **0.6191** (P0.538 R0.728) |

vs FRONT (plateaued): 1.42M→0.593, 2.46M→**0.594 (shipped)**. So stratified-untruncated w80 = **0.619 beats
shipped 0.594 by +0.025**, monotonic & not yet plateaued. Diversity + no-truncation are the levers.
w80 model is deployable (mc500, 1.05GB) — a shippable upgrade, not just a research number.
w80 model.bin durable at body_strip_scale_w80_strat_prep_mc500/ (got preempted DURING eval; re-evaled
the durable model with a separate eval-only job → never retrained).

w160 (7.2M lines) would likely push higher but can't complete: preemptible exhausts retries (cloud preempt
mid-train), non-preemptible gets reclaimed for production/hero work (reserved_8 contested). Needs genuinely
free stable capacity. TODO: split train and eval into SEPARATE jobs in the script so a preemption during
eval never re-triggers training (bit us on w80; worked around manually).

## PLOTS + PROJECTIONS (2026-06-26) — scratch/ft_scaling_plots/

Script: scratch/ft_scaling_plots/plot_ft_scaling.py (reads small eval JSONs in json_data/; uv run --with matplotlib,scipy,numpy).
- scaling_f1_vs_lines.png — F1 vs lines, all series, saturating fit + projected w160/w320.
- operating_curves_by_scale.png — full retained-vs-excluded curves at w10/20/40/80.
- exclusion_vs_scale.png — excl@90/95/97.5/99% recall vs scale + log-linear projection to w160/w320 + w320 resource note.

F1 projection (saturating fit, strat-untrunc): w80 0.619 → **w160 ~0.630 → w320 ~0.636**, ceiling **~0.647**.
=> fastText F1 near saturation; w80 already ~75% of headroom.

BUT exclusion at HIGH recall (the cascade operating point) is NOT saturating — scale helps most there:
| recall | shipped | strat w80 | proj w160 | proj w320 |
|---|---|---|---|---|
| 95% | 0.78 | 0.80 | ~0.83 | ~0.86 |
| 97.5% | 0.69 | 0.72 | ~0.78 | ~0.83 |
| **99%** | **0.52** | **0.58** | **~0.66** | **~0.73** |
=> at 99% recall, w320 projects ~0.73 excluded vs shipped 0.52 — the filtering win is concentrated exactly
where a stage-1 filter operates. (99% projection is a steep-tail log-linear extrapolation — validate with w160.)
w80 model.bin = 1.05 GB (mc500), deployable drop-in.

w320 resources/time: untruncated train.txt ~475GB > 400GB v4 (us-central2) RAM → needs v6e (720GB) in
us-east5-b (out-of-region) + ~93GB shard copy (~$2 egress); prep already done (full_prep shards); ~10-12h
train+eval @ cpu64. w160 (~258GB) fits us-central2 v4 @ 340GB — running (ft-w160-prep-np3).

## Status

- [x] 2026-06-24 — plan written.
- [x] 2026-06-24 — hit 100 GB disk cap; descoped to downward sweep (10/20/40 WARCs).
- [x] 2026-06-24 — launched `ft-scale-w10/20/40` (batch, us-central2, 2cpu/16GB). Each: pilot && eval.
- [ ] await completion → `python experiments/baseline_collection/fasttext_useful_classifier.py leaderboard`,
      then plot natural-F1 vs train_pos with the 4 points (24k/47k/95k/189k).
- [ ] decide on Phase 2 (truncation) to reach >190k.
