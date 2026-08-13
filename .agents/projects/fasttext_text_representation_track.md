# fastText TEXT-representation track (lpv11) — prepared plan, NOT launched

Companion to `.agents/projects/w320_fasttext_runbook_for_lpv11.md`. The whole lpv11 classifier
lineup trains on `body_strip` HTML, which is ~78.3% markup by character count. This is the fastText
arm of the text-vs-HTML representation experiment: retrain the **deployed stage-1 filter** on
main-content TEXT (XenonMolecule Rust resiliparse fork) and see whether it beats the HTML model on
the identical frozen documents.

**Number to beat: 0.8110 best-F1** on the frozen 7k (6,477 docs / 1,513 useful), measured for
`body_strip_scale_w640_sub0p22_strat_prep_mc500`. Its more-often-quoted 0.7990 is the *full*
4,089,328-row test — a different denominator; do not mix them.

Artifacts prepared by this plan (new files only, nothing existing was touched):

| file | role |
|---|---|
| `scratch/text_repr/extract_prep_text_rs.py` | prep-shard → text-shard extractor (Rust fork), sharded + single-file modes |
| `scratch/text_repr/score_fasttext_text_frozen7k.py` | frozen-7k scorer with provable doc-set alignment |
| `scratch/text_repr/launch_fasttext_text_track.sh` | the stage-by-stage command sequence (run with `bash`, one stage at a time) |

If the track is adopted, `extract_prep_text_rs.py` should be promoted into
`experiments/baseline_collection/`.

---

## 1. The exact training inputs, verified against GCS (2026-08-12)

fastText does **not** train on the survivor preshard the neural models use. It trains on the
natural-ratio `full_prep` shards, so it needs its own extraction over exactly this set:

**Selection file** (frozen, written by `train-from-prep --dry-run`, seed 0, `stratified`):

```
gs://marin-us-central2/classifiers/useful_fasttext_lpv11/full_prep_body_strip/_dryrun_selected_640_stratified.txt
gs://marin-us-east5/classifiers/useful_fasttext_lpv11/full_prep_body_strip/_dryrun_selected_640_stratified.txt
```
Both copies are **byte-identical**: 640 unique indices, min 64 / max 1892. Verified nesting
`w160 ⊂ w256 ⊂ w640` (all three prefix-nested at seed 0, as the runbook claims).

**What exists today:**

| location | split | shards | gz size |
|---|---|---|---|
| `gs://marin-us-central2/.../full_prep_body_strip/train/` | train | **1,715** (superset) | 641.1 GiB |
| `gs://marin-us-east5/.../full_prep_body_strip/train/`   | train | **640** (exactly the selection) | 241.2 GiB |
| both regions | test | **89** | 33.1 GiB |
| us-central2 only | val | 89 | 33.2 GiB |

Verified: the 640 objects in us-east5 are **set-identical** to the selection file (0 missing, 0
extra). The us-central2 1,715-shard train dir **contains all 640**. No drift from the runbook.

**Confirmation that this is what w640 was trained on** —
`…us-east5/…/body_strip_scale_w640_sub0p22_strat_prep_mc500/metrics.json`:
`train_warcs 640 · train_sample stratified · sample_seed 0 · neg_per_pos 3.5 · line_keep_frac 0.22 ·
max_chars null · min_count 500 · epoch 5 · lr 0.1 · dim 100 · word_ngrams 2 · loss softmax ·
n_missing_shards 0 · split_counts 1,462,564 useful / 4,170,558 no_useful (5,633,122 lines)`.

**Frozen test assets** (both already mirrored to us-central2 — no copy needed):
- full test: 89 shards, `n_useful 880,407 / n_no_useful 3,208,921 = 4,089,328 rows @ 3.64:1`
- frozen 7k: `…/full_prep_body_strip_test7k/test_sample_7k.txt.gz`, 59,268,551 bytes, present in
  **both** us-east5 and us-central2 at identical size.
- HTML baseline per-doc scores: `gs://marin-us-east5/.../eval/ft_w640_on_frozen7k.json` (156 KB).
- The w640 `model.bin` was mirrored into us-central2 on 2026-08-11 (1.32 GiB).

**Drift flags:** none material. Two cosmetic ones: (a) the per-WARC prep manifest reports 0 counts
for 3 shards (1 test, 2 val) that a re-run marked `skipped` — the authoritative test counts come
from `eval_natural.json`, not the manifest; (b) one train shard's gzip ISIZE footer wraps past
4 GiB, corrected in the sizing below.

**Region decision: run the whole track in us-central2.** Everything needed is already there (the
640 selected shards as a subset of 1,715, the 89 test shards, the 7k sample, the HTML model), the
raw HTML lives there, and — see §3 — the text `train.txt` no longer needs a v6e, which was the only
reason the HTML track ever went to us-east5. Total cross-region traffic for the whole plan:
**16 MB** (mirroring the Rust extractor artifact) + 156 KB (the baseline score JSON).

---

## 2. Sizing the text extraction

Doc counts from the per-WARC prep manifest; byte counts from object sizes + gzip ISIZE footers
(one shard's footer wrap corrected).

| set | shards | docs | gz | uncompressed HTML | text @21.7% | text gz (est) |
|---|---|---|---|---|---|---|
| train (the 640) | 640 | **30,118,669** (6,643,803 pos / 23,474,866 neg, 3.53:1) | 259.0 GB (241.2 GiB) | **1,655.2 GB** | 359.2 GB | ~112 GB |
| test (frozen 89) | 89 | **4,089,328** (880,407 pos / 3,208,921 neg, 3.64:1) | 35.5 GB (33.1 GiB) | **226.4 GB** | 49.1 GB | ~15 GB |
| frozen 7k sample | 1 file | 6,477 scored | 0.059 GB | ~0.37 GB | 0.08 GB | ~0.03 GB |
| **total to extract** | **729** | **34,207,997** | **294.5 GB** | **1,881.6 GB** | **408.3 GB** | **~110–145 GB** |

Per-WARC: 405 MB gz / **2.59 GB uncompressed** (train), 399 MB / 2.54 GB (test); compression 6.4x on
HTML. Text gz will compress worse (~3.2x, no repeated markup), hence the ~127 GB midpoint estimate.
Storage cost in-region ≈ $2.5/month. Egress: **$0**.

**Compute:** 34,207,997 docs ÷ 291.8 docs/s/core (measured for this extractor) = **32.6 core-hours**
of extraction, + ~7 core-h of gzip (1.88 TB in / 0.41 TB out) + ~2 core-h of Python string work ≈
**45–55 core-hours**.

**Job shape: 8 independent jobs × 16 processes (128 readers), strided disjoint slices.**
Rationale, all from the runbook's measured lessons:
- Zephyr fan-out is unusable here — a contended cluster places ~1 worker group out of 200.
- More boxes is not better: **14 concurrent reader boxes caused a throughput COLLAPSE** (bucket
  throttling); 4 boxes × 24 procs was the shape that worked. 8 × 16 sits in that band and holds
  aggregate read at ~80–150 MB/s.
- Slices are **strided** (`specs[i::N]`), so they are provably disjoint (the preshard lesson: an
  overlap duplicates work/docs) and long/short shards spread evenly rather than piling into one
  straggler slice.
- Every shard is atomic tmp→rename with skip-existing, so any job can be preempted and resubmitted.
Expected **~45–60 min per job**, all 8 in parallel.

**The `<frameset>` segfault matters at this scale.** The Rust extractor hard-crashes on
`<frameset>` documents — 0.108% of a 100k CC sample, i.e. **~37,000 crashes** across 34.2M docs, at
~50 per shard. Isolating them one-process-per-doc (the `score_resiliparse_rs.py` fallback) is
hopeless at that rate, so `extract_prep_text_rs.py` **pre-filters** on the substring `<frameset>`
and emits empty text (prep lines are already lowercased, so the plain substring test is exact; and
these are redirect/parking framesets whose main content genuinely is empty). A `BrokenProcessPool`
recovery path remains as a backstop: rebuild the pool, resubmit whatever has no output yet.

---

## 3. The `/tmp` tmpfs wall — the TEXT model no longer hits it

iris mounts `/tmp` with a bare Docker `--tmpfs`, so it is **50% of HOST RAM, not of `--memory`**.
That is the wall that made untruncated HTML w256 impossible and forced HTML w640 onto a us-east5
v6e. Host ceilings from `lib/iris/examples/marin.yaml`:

| family | host RAM | `/tmp` tmpfs |
|---|---|---|
| v4 (us-central2-b) | 400 GiB | **214.7 GB** |
| v5p (us-central1-a, us-east5-a) | 448 GiB | 240.5 GB |
| v6e (euw4-a, us-east1-d, us-east5-b) | 720 GiB | ~381 GB (probed) |

**Sizing estimator, calibrated.** Per-shard kept-line fraction (`keep_frac`, then the 3.5 neg cap,
per shard) × that shard's uncompressed bytes, summed:

| arm | predicted lines | measured lines | predicted bytes | measured |
|---|---|---|---|---|
| w160 | 6,658,901 | **6,658,901** (exact) | 354.7 GB | 350.4 GB (+1.2%) |
| w256 | 10,348,501 | — | 557.1 GB | 553 GB (+0.7%) |
| w640 sub-0.22 | 5,628,964 | **5,633,122** (−0.07%) | 304.7 GB | — |

Applying the −1.2% calibration:

- **HTML w640 sub-0.22 `train.txt` ≈ 301 GB** → over the v4 wall (214.7) and the v5p wall (240.5);
  under the v6e wall (381) with 80 GB headroom. Hence the us-east5 detour.
- **TEXT w640 sub-0.22 `train.txt` ≈ 301 × 0.217 = 65 GB** (allow 60–80 GB, since 21.7% is a
  corpus-level char ratio). **3.3x headroom under the plain v4 wall.** The detour is gone.

Runbook sizing rule (`post-cap bytes + ~130 GB slack ≤ min(tmpfs, request)`): 65 + 130 = 195 GB
→ request **`--memory 200GB`**, which is under the 214.7 GB v4 tmpfs ceiling. (`--memory 160GB`
is also defensible — fastText's own footprint at dim 100 / 2M buckets is only 1–2 GB — and
schedules more easily on a contended cluster.)

**Bonus, for later, NOT part of the controlled comparison:** because text is 4.6x smaller, arms that
were physically impossible in HTML now fit.

| arm | lines | HTML train.txt | TEXT train.txt | fits |
|---|---|---|---|---|
| w640 keep 0.22 (the control) | 5.63 M | 301 GB | **65 GB** | anything |
| w640 keep 0.50 | 12.79 M | 684 GB | **148 GB** | v4 (67 GB headroom) |
| w640 keep 1.00 | 25.59 M | 1,368 GB | **297 GB** | v6e only |

A 2.3x-more-lines text arm on a plain v4 is the first time this corpus could be scaled past the
tmpfs wall at all. Given "all four HTML fastText arms sit within 0.008 F1 — fastText is saturated by
~3.4M lines", expect little; but it is now cheap to check.

---

## 4. Command sequence

`scratch/text_repr/launch_fasttext_text_track.sh <stage>`, run with **`bash`** (zsh globs unquoted
`gs://` wildcards locally and kills the command before gsutil runs). Stages:

0. `stage0_mirror_artifact` — copy the 16 MB prebuilt Rust extractor into us-central2.
1. `stage1_extract_shards` — 8 jobs × `--cpu 16 --memory 64GB --extra cpu`, batch + preemptible.
   `stage1b_extract_frozen7k` — 1 tiny job for the line-aligned 7k twin.
2. `stage2_verify` — **gate**: 640/89 shards present, index set == the frozen selection,
   measured `text_char_fraction` ≈ 0.217, 7k twin present.
3. `stage3_train` — `train-from-prep --representation resiliparse --prep-tag rs`
   (→ `full_prep_resiliparse_rs`), `--shard-indices <the same 640 CSV>`, `--line-keep-frac 0.22`,
   `--neg-per-pos 3.5`, untruncated, `--epoch 5 --lr 0.1 --dim 100 --word-ngrams 2 --minn 0
   --maxn 0 --loss softmax --min-count 500`, `-e USEFUL_FT_ROOT <root>`,
   `--cpu 64 --memory 200GB --extra cpu --extra dclm --enable-extra-resources`, batch+preemptible.
4. `stage4_eval_full` — **separate job, never `&& eval`** (a preemption during eval re-triggers
   training). Must pass `--test-glob` explicitly: the default resolves to `full_prep_body_strip`.
5. `stage5_score_frozen7k` — the headline number.
6. `stage6_compare` — `cascade_analysis.py` over the two aligned per-doc JSONs.

Note `train-from-prep` only uses `--representation` to build the path, so no code change is needed:
`resiliparse` + `--prep-tag rs` resolves to `full_prep_resiliparse_rs`, and the recorded
`config.representation` is honest.

Priority is `--priority batch --preemptible` throughout. CPU-only iris jobs **default to
non-preemptible**, which bin-packs them onto the reserved v4 hero-run pool; escalating to
`interactive` needs the user's sign-off (that is how the HTML w256 run was authorized).

---

## 5. Confound checklist — what a reviewer will ask

Held fixed by construction (each is asserted somewhere in the prepared scripts):

1. **Same WARCs.** `--shard-indices` is fed the very same 640-index CSV; `stage2_verify` diffs the
   text shard indices against the selection file. Not re-derived — re-running the dry run would
   re-shuffle.
2. **Same documents inside each WARC.** Same `--line-keep-frac 0.22`, same `--sample-seed 0`; the
   per-shard RNG is seeded `(seed << 20) ^ shard_index`, so the identical docs are kept. Expected
   `metrics.json split_counts` ≈ **1,462,564 / 4,170,558** — the same line counts as the HTML run.
   A different count means a confound leaked in; stop and find it.
3. **Same class ratio.** `--neg-per-pos 3.5` (the measured natural ratio). Ratio is the dominant
   lever — balanced training cost ~0.12 F1 on this task.
4. **Same hyperparameters.** The locked DCLM recipe, `min_count 500`, untruncated (no `--max-chars`;
   8k head-truncation cost ~0.10–0.12 F1, and length is itself signal — note text is 4.6x shorter,
   so any lurking truncation would bite the two arms *differently*).
5. **Same test documents, in the same order.** The 89 frozen test shards are extracted 1:1 and
   `run_eval` must report exactly `880,407 / 3,208,921 = 4,089,328`. For the frozen 7k, the sample
   is selected from the **HTML** file and applied positionally to the row-aligned text twin.
6. **Empty extractions must not vanish.** This is the sharpest trap. `score_fasttext_on_bert_test.py`
   and `modernbert read_fasttext` skip lines with empty text — and main-content extraction returns
   "" for a real slice of junk pages (plus every pre-filtered `<frameset>` page). Silently dropping
   them would delete mostly-negative documents from the evaluation and *inflate* the text model's
   score, while also changing the shuffle and breaking index alignment with the published JSONs.
   Hence `score_fasttext_text_frozen7k.py`, which keeps them and scores them at the class prior
   (which is also what deployment would do), and asserts `6477 docs / 1513 useful`.
7. **Same text normalization.** The extractor applies the identical `\s+ → " "`, `.strip()`,
   `.lower()` that `to_fasttext_text` applies, so the only difference is markup removal.

Judgement calls that must be settled before launch, not after:

8. **`preserve_formatting` must match the neural track.** The default here is `markdown` (what
   `score_resiliparse_rs.py` and the planner stage `extract_resiliparse_rs` use). If the agent
   producing the neural text data uses `minimal_html` or plain, change `--preserve-formatting` to
   match — otherwise "text vs HTML" silently becomes "two different texts".
9. **Extraction source: prep shards, not the labeled parquet.** Chosen so that the text arm sees
   *exactly* the HTML the incumbent sees (same docs, same lowercasing, same whitespace collapse,
   same order, same labels) — one variable, markup, changes. Costs: `body_strip` already dropped
   `<head>` (so no `<title>`) and collapsed newlines (so `<pre>` structure is gone) *before*
   extraction. If the neural track extracts from raw parquet HTML instead, the two arms are not
   comparable and one of them must move. Reading prep also costs 295 GB instead of ~780 GB.
10. **The operating point must be re-derived, never reused.** The deployed HTML threshold 0.0125
    (and the best-F1 threshold 0.406/0.48) are properties of that model's score distribution. Take
    the text model's threshold from its own `eval_natural.preds.parquet` at the target recall.
11. **Noise floor.** The frozen-7k noise floor is ~0.015 F1, and all four HTML fastText arms sit
    within 0.008 of each other. A text-vs-HTML delta under ~0.015 is a tie; prefer the
    exclusion-at-fixed-recall columns (`excl@R≥0.99 / 0.975 / 0.95`), which is what a stage-1 filter
    is actually operated on, over peak F1.
12. **No `&& eval`.** Chaining eval after training means a preemption during eval re-runs the whole
    train step; that is why stages 3, 4 and 5 are separate jobs.
