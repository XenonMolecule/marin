# Cascade Pipeline Planner: TEXT-classifier stages + remaining lpv11 models (2026-08-17)

Goal: every finished lpv11 classifier appears as a planner stage; TEXT-trained classifiers
(input = resiliparse-rs main-content text, XenonMolecule Rust fork) are modelled as filters that
DEPEND on that extraction and are charged for it exactly once per pipeline.

## Inventory vs planner (2026-08-17)

Already wired: fastText lpv11 w80/w160/w256/w640 (HTML), mb base-1M/10M (HTML), ettin68-10M (HTML),
pooled 1M-r5/10M/10M-e3/big-10M (HTML).

To add — DONE training (hf/ present):
- HTML arch-sweep 1M: ettin68-1M, ettin17-1M, ettin32-1M, pruned8-1M, tiny2-1M, tiny4-1M (HF-format,
  `score_modernbert_useful`); funnelbert-1M (eqx, needs config_class dispatch in `score_pooled_useful`).
- TEXT: text-ettin68-10M (0.8805, best), text-base-1M, text-ettin68-1M (HF); text-pooled-10M,
  text-pooled-1M, text-pooled-90M (eqx; 90M lives in us-central2 → mirror hf/ to us-east5 once);
  fastText TEXT w640 (`resiliparse_scale_w640_sub0p22_strat_prep_mc500_TEXT`, us-central2).
Still training (no hf/): mb-clf-lpv11-text-base-10M (75%), mb-clf-lpv11-pooled-90M HTML (68%) → ask user.

## Train/serve parity for TEXT models (the trap)

The sample's existing `text_resiliparse_rs` column is `extract_plain_text(raw_html)` — the
extractor-comparison text (Levenshtein vs targets). TEXT classifiers trained on
`extract_plain_text(body_strip HTML already lowercased+ws-collapsed, main_content=True,
preserve_formatting="markdown")` → `\s+`→" " → strip → lower (`extract_prep_text_rs.html_to_text_line`;
neural corpora via `extract_text_shards._one_line`, same modulo `__empty__` placeholder for empty docs).
So a NEW side artifact `clf_text_resiliparse_rs/` (row-aligned, same shard names) built from
`_preprocess(stripped_html)` is the scorer input; neural scorers map "" → "__empty__", fastText feeds "".

## Engine model for text stages

`Stage.requires_text = "<extractor key>"` (here `resiliparse_rs`). In `evaluate`, a per-key
`paid` mask records docs whose text extraction has been charged. A filter with `requires_text`
charges the text extractor's throughput/device for reaching docs not yet paid, then marks them.
The terminal extractor with the same key charges only unpaid docs (0 if a text filter preceded).
Reach sets shrink monotonically, so extraction is charged once, on the first text stage's reach.
Breakdown gets a "<extractor> text (for classifiers)" line. Tests in engine.test.js.

## Throughput
ettin68 36.4 (precise). Sweep archs re-measured precisely in one job
(`arch_inference_benchmark --archs ettin17,ettin32,pruned8,tiny2,tiny4,funnelbert,mb-base --warmup 15 --iters 50`).
TEXT twins reuse the arch number (same arch/ctx; text is shorter so this is conservative) and pay
resiliparse-rs extraction (291.8 docs/s/core CPU) via the engine. fastText TEXT measured by the scorer.

## Steps
1. build_clf_text_resiliparse_rs.py → side artifact (CPU us-east5).
2. Scorers: input kind per model (html|text); pooled scorer config_class dispatch (funnelbert).
3. Score all; join; precise benchmark; registry+engine+app+tests; precompute; download.

## Status 2026-08-17 ~20:15Z
- DONE: `comparison_sample.py` (shared paths/reader, `ClassifierInput.{HTML,TEXT}`); scorers refactored
  onto it (`score_modernbert_useful` MODELS now (run_id, ctx, input); `score_pooled_useful` MODELS
  (run_id, input) + `_EQX_ARCHS` config_class dispatch incl. funnelbert; `score_fasttext_useful --input`).
- DONE: `build_clf_text_resiliparse_rs.py` → `clf_text_resiliparse_rs/` (200 shards, 99,996 docs, 1.4%
  empty, mean 3.8k chars, 62 framesets screened, 0 panics; 44.7s on 16 cores). Install the extractor in
  the PARENT before the pool — per-worker installs race on the .so ("file too short").
- DONE: engine `requires_text` accounting (+4 tests, 14/14 pass); registry: `Stage.requires_text`,
  families `arch_sweep` / `text` / `fasttext_text`, 14 new stages (46 classifiers); app.js badges/labels.
- RUNNING: 14 scoring jobs (7 HTML sweep, 3 mb-text, 3 pooled-text, 1 fastText-text) + precise sweep
  benchmark; all TPU jobs pending on us-east5 v6e capacity at launch. text-pooled-90M hf/ mirrored to
  us-east5 (99 MiB, one-time).
- NEXT: when scores land → `score_modernbert_useful join` → full precompute (`--extra cpu --extra
  extraction-bakeoff`) → download to web/data/. Drop precise sweep throughputs into `_BERT_CEIL` /
  `_FUNNELBERT_TP` and fastText TEXT timing into its Stage (clear `throughput_assumed`).
- ASK USER: text-base-10M (75%) and pooled-90M HTML (68%) still training — worth adding when done?

## DONE 2026-08-18 06:20Z — dashboard refreshed (53 stages / 46 classifiers) at :8092

**One-extraction verdict: NO RETRAIN.** Production contract = resiliparse-rs ONCE on raw HTML; TEXT
classifiers see lower(collapse(that)). Same checkpoints, same 99,996 lpv11-covered docs, as-trained vs
deployment input (`compare_text_input_scores.py`; JSON at
`…/high_quality_200warc/clf_text_input_score_comparison.json`):
- text-ettin68-10M: F1@.5 0.8719 -> 0.8719 (+-0.0000), best-F1 +0.0005, P@R.97 -0.0044, keep-agree 99.38%
- text-base-1M: +0.0015 / +0.0012 / +0.0075 · fastText TEXT: +0.0008 / +0.0010 / +0.0054 (noise floor 0.015)
- Strata (champion): identical 62,686 (0); sim>=.9 20,411 (-0.0014); [.5,.9) 11,577 (-0.0126, precision-side);
  sim<.5 4,738 (+0.0561 — raw extraction recovers content body_strip's 1M-char cap/stripping dropped);
  only-deploy-empty 360, gold 3.9% (F1->0 but = 14 useful docs of 20,900); only-trained-empty 224,
  gold 19.6% (+0.80 F1: 44 useful docs recovered). Net favours deployment.
- The dashboard's TEXT stages therefore use the DEPLOYMENT columns (`*_textraw_prob_*`,
  `fasttext_lpv11_textraw_prob_w640`); the as-trained `*_text_prob_*` columns remain as diagnostics.
- `extract_clf_text(text_body)` is NOT an alternative: its output is lowercased/ws-collapsed and unusable
  as pipeline output text, so it would force a second extraction.

Other landings: precise sweep throughputs (mb-base 21.57 cross-check): tiny2 516.1, tiny4 192.6,
ettin17 172.3, funnelbert 92.9, ettin32 85.1, pruned8 54.9. fastText TEXT 4875.5 docs/s/core (13.4x HTML).
funnelbert eqx config round-trip stringifies StrEnum fields -> coerced in `load_eqx_config`.
funnelbert end-to-end scoring ran at 5.8 docs/chip/s vs 92.9 forward ceiling (host tokenization bound).
Backend now needs `uv run --with flask …` (post-sync venv has no flask).
Open: text-base-10M (trainer session `new_classifiers` will ping when hf/ lands) — add on deployment input;
pooled-90M HTML likewise.

## Pipeline shape: HTML classifiers -> THE extractor -> TEXT classifiers (2026-08-18, user call)
There is exactly ONE extractor and it is REQUIRED, so it is not a filter and not terminal — it sits in the
middle. Placement is derived from what a stage READS, not chosen: `requires_text` => runs after the
extractor, everything else runs before it. The extractor is charged on the docs the HTML filters passed;
TEXT stages then run on its output; final kept is after the TEXT stages.
Rejected en route (my two earlier attempts): charging extraction implicitly inside the first TEXT
classifier's card (hid the cost), then letting extractors be inserted as chain steps (allowed a nonsensical
second extractor at the end). A TEXT classifier is only offered/valid when the extractor it was trained on
is selected; switching extractors drops incompatible TEXT filters with a flash rather than stranding the
pipeline, and any residual invalid state renders an explanatory card instead of blanking the flow.
engine.js returns `stages` (pre) + `postStages` (post) with `filterIdx` back-references; breakdown rows are
pre + extractor + post and sum exactly to the totals (asserted). 16 engine tests pass.

### Follow-up bug (2026-08-18): panels indexed `r.stages` by filter index
Splitting the rows into `stages` (pre-extractor) + `postStages` broke three call sites that still did
`r.stages[filterIdx]`: `showBorderline`, `catchItHtml`, `deathStage`. Invisible until a pipeline whose
filters are ALL text stages (pipeline_config-3: fastText-TEXT -> pooled-90M -> ettin68-10M), where
`r.stages` is empty and `r.stages[0].threshold` throws — "show pipeline losses" silently did nothing.
Fixed with `rowFor(r, filterIdx)` / `rowsInOrder(r)`; `deathStage` now also walks EXECUTION order and
attributes docs the extractor abstained on ("no threshold recovers this one").

## COMPLETE 2026-08-19 — 55 stages / 48 classifiers live at :8092
Final two models added: `mb-clf-lpv11-text-base-10M` (frozen-7k .8836) and `mb-clf-lpv11-pooled-90M` HTML
(.8331, hf/ was us-central2-only -> mirrored 99 MiB; the scoring job survived 7 preemptions, 55 min compute).
Every one of the 47 registry score columns is scored, joined and in the served matrix (verified against GCS,
not against notes).

**base-10M-TEXT does NOT break the one-extraction tie.** as-trained -> deployment on the same 99,996 docs:
base-10M .8730 -> .8729 (-0.0001, best-F1 +0.0002, P@R.97 -0.0058) vs ettin68 .8719 -> .8719. Stratum by
stratum the two are near-identical (sim>=.9 -0.0017/-0.0014; [.5,.9) -0.0122/-0.0126; sim<.5 +0.053/+0.056;
only-deploy-empty F1->0 on 360 docs both), i.e. **the input skew is a property of the extraction paths, not
of model capacity** — which is why the fastText null generalized after all.
On the deployment input the gap is 0.0010 (0.8729 vs 0.8719), tighter than frozen-7k's 0.0031. Claim:
**ettin68-10M-TEXT MATCHES mb-base-10M-TEXT at 1.69x throughput** (36.4 vs 21.57) — equal accuracy for 59%
of the inference cost. Not "more accurate"; the registry comment says so explicitly.

## ⬆ import button (2026-08-19)
Export had no counterpart, which is why configs could not cross origins (localStorage is per-port) and why
a downloaded `pipeline_config-*.json` previously had to be read by hand. Added `⬆ import` beside `⬇ export`:
file picker -> JSON parse -> validate (filters array, extractor, capacity, known extractor id) -> adopt.
Both import and the saved-library dropdown now go through one `loadPipeline(p, source)` which normalizes,
re-renders the palette (its TEXT options depend on the selected extractor — the dropdown previously skipped
this) and syncs the hash so an imported config is immediately shareable as a link. Malformed input flashes a
reason instead of throwing; stages this build can't run are dropped with a count in the flash.

## Optimal-config search (2026-08-22) — EARLY-EXIT cascades

Task: beat config-3 (F1 0.8658 on the 10k UI matrix / 0.8690 on 100k, 25.9 days) under <=1 month at
1000 TPU chips + 5000 CPU cores, minimizing runtime too. Method: harness reproducing engine.js semantics
EXACTLY (verified bit-for-bit on config-3), thresholds tuned on one half of the 100k sample and every
number reported on the held-out half; a bitset evaluator (intersection => AND+popcount) made ~10^5
candidate pipelines tractable.

**What the physics turned out to be.** resiliparse-rs is the only affordable extractor (LLM extractors
need 80+ years). CPU is nearly idle (2.5 d for extraction of all 317B docs); TPU is the whole budget.
ettin68-text at 36.4 docs/chip/s can touch only **30% of the corpus in 30 days**, so an intersection
cascade's F1 is pinned by *gate recall at the keep-rate the budget buys*: text_pooled_90M is the best gate
at every keep-rate (94.8% of gold at 30% keep) and gate PAIRS add only +0.002 — saturated. Hence no
meaningful F1 gain is available from re-thresholding: 30d tops out at ~0.871, i.e. config-3 within noise.

**The win is structural: early exit.** Give the cheap gate two thresholds — accept outright at/above `hi`,
drop below `lo`, send only the uncertain band to the strong model. The gate is already right on the
confident tails, so ettin68 adjudicates ~9-14% of docs instead of 30%:
  5d 0.8569 | 10d 0.8657 | **15d 0.8693** | 25d 0.8709   (real engine, full 100k)
vs config-3's 0.8690 at 25.9d. **Same F1 in 42% of the runtime**; F1 differences are all inside the 0.015
noise floor, the runtime difference is not. Configs: `~/Downloads/pipeline_config-EE-{5,10,15,25}d.json`.

Implemented as `mode: "band"` in engine.js (+4 tests, 20 pass) with `hi`/`lo` controls in the UI. Accepted
docs skip every later filter but the EXTRACTOR is still charged for them — they need text to enter the
corpus — and an abstaining extractor still removes them.

Open lever, queued behind TPU capacity: the text models are scored at ctx 8192 but the extracted text
averages ~926 tokens (median ~334), so the forward pass is mostly padding. Same checkpoint at ctx 2048
measures 141.5 docs/chip/s (3.9x) with only 9.2% of docs truncated; at 1024, 244.4 (6.7x, 20% truncated).
Columns `bert_lpv11_textraw_prob_ettin68_10M_c{1024,2048,4096}` + base-10M c2048 are scoring now — if
accuracy holds, ettin68 could run on the WHOLE corpus in ~26 days, or the early-exit band gets ~4x cheaper.

## RESULT (2026-08-23): 5x faster at equal F1, via short eval context + early exit

Verified with the REAL engine on both matrices (1000 chips / 5000 cores / 317B docs):

| config | 100k F1 | 10k F1 | runtime |
|---|---|---|---|
| config-3 (baseline) | 0.8690 | 0.8644 | 25.9 d |
| **BEST**: pooled_90M band[.992/.066] -> ettin68@**2048**@.438 | **0.8718** | 0.8649 | **9.6 d** |
| FAST: pooled_90M band[.940/.066] -> ettin68@**1024**@.381 | 0.8690 | 0.8647 | 5.0 d |
| FASTEST: pooled_90M band[.839/.184] -> ettin68@1024 | 0.8671 | 0.8637 | 2.9 d |
| simple: ettin68@1024 alone, no gate | 0.8705 | 0.8638 | 15.0 d |

Configs in `~/Downloads/pipeline_config-{BEST-10d,FAST-5d,FASTEST-3d}.json`.

**Every F1 difference is inside the 0.015 noise floor — the deliverable is RUNTIME: 25.9 d -> 5.0 d at
identical 100k F1 (0.8690), or 9.6 d for the best measured F1.** Two independent levers, both verified:
1. **Short eval context.** Same checkpoint, shorter window. ettin68-text best-F1 0.8736@8192 / 0.8720@2048
   / 0.8705@1024 while throughput goes 36.4 -> 141.5 -> 244.4 docs/chip/s. The text averages ~926 tokens
   (median ~334) so an 8192 forward is mostly padding; 20% of docs truncate at 1024 and it costs 0.0031.
2. **Early exit** (`mode:"band"`), see the section above.

**We are at the ceiling.** Held-out solo ceiling (best single model, unlimited compute) = 0.8737 (ettin68)
/ 0.8742 (base-10M); the unconstrained early-exit optimum = 0.8731/0.8736. BEST scores 0.8735 held-out.
Nothing meaningful is left on the accuracy axis with these checkpoints — further gains need a better model,
not a better cascade. The runtime floor is 2.5 d = resiliparse-rs over all 317B docs on 5000 cores.

Search methodology (harness in the session scratchpad, `search/`): reproduces engine.js semantics
bit-for-bit (verified against config-3), thresholds tuned on one half of the 100k sample and reported on
the held-out half, both split directions checked. Bitset evaluator (intersection = AND+popcount) and an
O(1) prefix-sum evaluator for early exit made ~10^5-10^6 candidates tractable.

### fastText DOES belong (2026-08-23, user challenge)
The frontier search that produced BEST enumerated only gate->final PAIRS, so it never tried a fastText
prefilter — an omission in the search, not a finding. A general optimizer over the full structure
(`[HTML gate] -> EXTRACT -> [fastText-TEXT] -> [pooled gate] -> [strong model]`, every position off /
threshold / band, random restarts + coordinate descent) says fastText buys ~30% runtime at equal F1:

| config | 100k F1 | 10k F1 | runtime |
|---|---|---|---|
| config-3 | 0.8690 | 0.8644 | 25.9 d |
| BEST, no fastText | 0.8718 | 0.8649 | 9.6 d |
| **7d + fastText** | 0.8708 | 0.8646 | **7.0 d** |
| **5d + fastText** | 0.8696 | 0.8644 | **5.0 d** (config-3's F1, 5.2x faster) |
| 3d + fastText | 0.8669 | 0.8616 | 2.9 d |

Two mechanisms: fastText-TEXT is ~free on CPU (4875 docs/s/core) and shrinks the band the strong model
sees; and an HTML pooled gate BEFORE extraction cuts the extraction bill, breaking what I had wrongly
called a hard 2.5 d CPU floor (a 2 d config extracts only 28% of docs, cpu 0.72 d).
Files: `~/Downloads/pipeline_config-{3d,5d,7d}-ft.json`, plus `pipeline_config-BEST-10d.json`.
Caveat: at this point +-0.002 is search-restart noise (a 60-restart refinement scored 0.8666 at 3 d where
an 8-restart run scored 0.8686), so read the F1 column as flat and choose on runtime.

## PRODUCTIONIZED (2026-08-26): TEXTONLY-7d -> `lpv11_fastpipe_v2`, pilot on random-300

User picked **TEXTONLY-7d** (`~/Downloads/pipeline_config-TEXTONLY-7d.json`) as the production config.
Implemented in `experiments/fast_curation/` as spec **`lpv11_fastpipe_v2`** (hash `32b74664f1`); see
`VERSIONS.md` for the ledger entry. Structure collapses to TWO phases (no Phase C):

- **Phase A (CPU, `cpu_phase_a._process_one_text`)**: decode → resiliparse-rs on raw html (every doc,
  `ResiliparseRsPool` reused from Phase C) → `normalize_text` → fastText-TEXT gate (0.0048) →
  tokenize@8192 → `a_presurvivors/` carrying the extracted `text` (PRESURVIVOR_TEXT_SCHEMA).
- **Phase B (TPU, `tpu_phase.process_warc_text_b`, `--mode textb`)**: pooled-90M scores all
  (ctx 8192, its calibrated native ctx); band `[lo 0.079, hi 0.8883]`; ettin68@2048 on the band only
  (ids derived by `batch_format.truncate_ids` slice+[SEP], parity-tested vs real tokenize@2048);
  writes final `kept/` (KEPT_SCHEMA_TEXT, `modernbert_prob=NaN` for hi-accepts) + band tombstones.
- `pooled_hi` + `modernbert_max_length` are namespace-defining; `modernbert_threshold` late-bound
  over band docs only. Hashes pinned in `test_spec.py`; band routing covered by
  `test_process_warc_text_b_band_routing`.

Models mirrored (byte-verified equal sizes) to us-east5 (canonical), us-central1, us-east1, eu-west4:
fastText `resiliparse_scale_w640_sub0p22_strat_prep_mc500_TEXT/model.bin` (919MB, canonical home now
us-east5, copied from its us-central2 training home), `mb-clf-lpv11-text-pooled-90M/hf` (104MB),
`mb-clf-lpv11-text-ettin68-10M/hf` (277MB). resiliparse-rs artifact already mirrored in all 4.

Pilot plan (user): **300 random WARCs** (`experiments/distill/random_subsets/random_warcs_300.txt`,
seed-0 nested chain → scaling to 10,364 extracts only new WARCs) → then 10,364 → then bigger. Pilot
runs single-region us-east5; other regions are model-ready. v6e in us-east5 is preemptible-only, so
Phase B canary/fleet run `--preemptible` (a `--no-preemptible` submit fails constraints).

### gigatoken (2026-08-26): parity EXACT, wired behind `--tokenizer-impl gigatoken`

Measured on 4,921 real deployment-distribution docs (clf_text fromraw shards) + adversarial cases,
macOS M-series: **byte-exact id parity at both ctx 8192 and 2048** through the HFCompat adapter
(`gigatoken.Tokenizer(hf).as_hf()`, `truncation=True, max_length=N`, same `max_length*8` char cap).
Throughput vs our real baseline (`backend.encode_batch_fast`, NOT the slow transformers wrapper the
1000x headline compares against): **0.86x at max_length=8192** (our Phase A setting — no win),
**3.3x at 2048**. The native `encode_batch` API is 0.28x once Python-side special-token assembly is
added — the compat path is the right one. Linux worker numbers may differ; that is what the pilot
A/B measures.

Wiring: marin-core extra `gigatoken==0.10.0` (exact pin — ids are the classifier input);
`preprocess.load_gigatoken` / `tokenize_trunc_batch_gigatoken` / `assert_gigatoken_parity` (worker
STARTUP GATE: adversarial-text id comparison vs HF, RuntimeError on divergence — a "close" tokenizer
silently shifts every cascade score); `cpu_phase_a --tokenizer-impl {hf,gigatoken}` (launcher adds
the extra); `timing_a` JSON records `tokenizer_impl` so a split fleet A/Bs `tokenize_s` per WARC.
Gotcha fixed: `load_gigatoken` initially called `load_tokenizer` while holding the same
non-reentrant `_LOCK` — self-deadlock; resolve the HF tokenizer before locking.

### Arrow fast path (2026-08-26, user push): 23x — Python materialization WAS the bottleneck
The compat-path numbers above were misleading twice over: cold-start inflated gigatoken's cost, and
BOTH paths were dominated by building per-doc Python lists. Warm, at WARC scale (15k real docs, 51M
chars, ctx 8192): HF encode_batch_fast 8.9s; gigatoken native `encode_batch` -> awkward ragged
(uint32 buffers) -> vectorized truncate to ml-2 + [CLS]/[SEP] concat -> `ak.to_arrow` ->
`list<int32>` = **0.39s (23x), byte-exact ids AND n_tokens**, and the parquet write is ~2x faster
from the arrow source. Productionized as THE gigatoken path: `tokenize_trunc_batch_gigatoken` now
returns `(pa.list_(int32) column, int32 lengths)`; the HF impl got the same columnar contract
(`tokenize_trunc_batch_arrow`); `write_presurvivors_text_columns` writes ids without row dicts;
`MODERNBERT_CLS_TOKEN_ID` pinned in spec.py; parity gate covers ids + n_tokens + wrong-special-token
divergence. Ids never exist as Python objects between tokenizer and parquet.

### First real WARC through the TEXT pipeline (canary, 8-cpu worker, 6 extract procs)
57,160 decoded -> 614 empty + 136 crashed extractions -> 46,211 presurvivors (81% fastText keep —
consistent with TEXTONLY-7d's loose 0.0048 gate; pooled does the real culling). Timing: extract
443s, decode 159s, fastText 28s, tokenize 5.6s (old compat path; arrow ~1s), wall 655s. **Extraction
is Phase A's bottleneck at 21 docs/s/proc vs the 292/s single-core benchmark** — likely pickling
multi-MB html into 25-doc pool tasks; the next lever if Phase A needs to be faster, tokenize is not.

### Ops trap hit at first canary (2026-08-26): resiliparse-rs artifact is CPython-minor-pinned
Post-merge workers run py3.12; the published artifact was a py3.11 build → Phase A fails fast at
`_assert_artifact_matches_spec` (by design). Fork master had moved past the pinned commit, so
`build_resiliparse_rs.py` gained `--commit` to rebuild the EXACT pinned `850891b` for py3.12
(`rp-rs-build-py312b`, us-east5). Note the build must be launched as `-- python -m ...` directly —
the docstring's `bash -lc` pattern escapes the uv env post-merge (ModuleNotFoundError: fsspec).
