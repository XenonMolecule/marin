# Fast-curation pipeline versions

Human-readable ledger of what each `fastpipe_vN` actually means. The authoritative
machine-readable definition is `SPECS` in `spec.py`; this file explains the *why*. Keep
the two in sync — update both in the same commit when you bump a version.

Each version's `compute_version()[:10]` hash is recorded so any GCS namespace
(`gs://marin-<region>/documents/fast_curation/fastpipe_vN-<hash>/...`) can be traced back
to its exact configuration. The hash covers every namespace-defining field **except**
`modernbert_threshold` (that threshold is a cheap late-bound re-filter over stored probs,
not a recompute trigger).

| field | meaning |
|---|---|
| fastText model + threshold | useful-vs-not filter on `body_strip` text; **below threshold → dropped, no record** |
| JustText version | the XenonMolecule/jusText fork tag; produces the training `text` (output content depends on it) |
| JustText max_html_chars | pages with raw html longer than this **skip JustText → dropped**; a quality knob (too low loses long articles/books) whose purpose is to bound lxml DOM-parse cost |
| JustText timeout | per-doc wall-clock guard (s); a doc exceeding it is hard-killed → dropped (guards the rare page lxml's C parser hangs on) |
| tokenizer / pad / max_length / single_window | how the `body_strip` text is tokenized for ModernBERT |
| ModernBERT checkpoint | the useful-vs-not classifier run on (pre-tokenized) `body_strip` |
| step_order | the cascade order |

Recompute any hash with
`python -c "from experiments.fast_curation.spec import get_spec; print(get_spec('fastpipe_vN').version())"`.

---

## lpv11_fastpipe_v2_1  (ACTIVE, lpv11 TEXT line — 8M-scale storage contract)

- **Status**: identical cascade semantics to v2 (same models, thresholds, band, ettin@2048 — same
  corpus content); new namespace for the scale contract.
- **Hash**: `944ca6bc38` (`storage_version=3` is namespace-defining).
- **Changed vs v2 (storage/orchestration only)**: sharded work-list + shard claims + per-shard
  catalog (`shard_worklist.py`; O(shards) GCS traffic instead of O(WARCs x fleet)); presurvivors/
  kept carry NO `input_ids` (Phase B re-tokenizes from `text` via gigatoken, parity-gated);
  text parquets at zstd level 12; `n_tokens` computed in Phase B. First run: the 100k shakeout.

## lpv11_fastpipe_v2  (lpv11 TEXT line — proven at 10,364 WARCs)

- **Status**: new line — the TEXTONLY-7d config from the planner's optimal-cascade search
  (`.agents/projects/planner_text_classifier_stages.md`, 2026-08-23). Equal F1 to config-3 within the
  0.015 noise floor at ~5x less TPU time, via early exit + short eval context.
- **Hash**: `32b74664f1` → namespace `gs://marin-<region>/documents/fast_curation/lpv11_fastpipe_v2-32b74664f1/`.
- **Naming**: briefly launched as `lpv11_textpipe_v1` (same hash — `spec_id` is path-only, not hashed);
  the 300-WARC pilot's outputs started under that prefix and are renamed in-region to this one.
- **Shape**: 2-phase. Extraction runs FIRST (Phase A, every decoded doc); every classifier consumes
  `lower(collapse(resiliparse-rs(raw_html)))` — the `*_textraw_*` deployment representation the
  thresholds were tuned on. Phase B (TPU) is terminal and writes `kept/` directly; there is no Phase C.
- **Cascade**: `decode → resiliparse_rs → clf_text → fastText → tokenize → pooled_band → ettin68@2048`
  | stage | model | operating point |
  |---|---|---|
  | fastText TEXT | `resiliparse_scale_w640_sub0p22_strat_prep_mc500_TEXT` (w640) | ≥ 0.0048 |
  | pooled 90M TEXT (band) | `mb-clf-lpv11-text-pooled-90M` | accept ≥ 0.8883, drop < 0.079 |
  | ettin68 10M TEXT @2048 | `mb-clf-lpv11-text-ettin68-10M` | ≥ 0.4378 (band docs only) |
  | extract | resiliparse-rs @ `850891b` (in Phase A) | — |
- **Early-exit band**: pooled ≥ hi is kept outright (NO terminal prob is ever stored for it), < lo is
  dropped, the middle goes to ettin68. Hence **`pooled_hi` and `pooled_threshold` are BOTH
  namespace-defining**; `modernbert_threshold` stays late-bound but only over BAND docs.
- **Short eval context**: `modernbert_max_length=2048` (namespace-defining). Phase A tokenizes once at
  8192 (the pooled model's calibrated ctx); Phase B derives the 2048 ids via `truncate_ids`
  (slice + re-append `[SEP]`), exactly reproducing `tokenize(max_length=2048)` — parity-tested.
- **fastText model canonical path is us-east5** (mirrored from its us-central2 training home) so the
  region rebucketing convention holds.

---

## lpv11_fastpipe_v1  (lpv11 line)

- **Status**: new line — the first cascade targeting `llm_pipeline_v1_1` instead of the 8B `high_quality` run.
- **Hash**: `2224e3e476` → namespace `gs://marin-<region>/documents/fast_curation/lpv11_fastpipe_v1-2224e3e476/`.
- **Why a separate line, not a v4**: lpv11 and the 8B agree at only **0.325 F1** (keep rates 21.5% vs 4.8%). A cascade mixing an lpv11-trained stage with hq-trained ones would have its filters optimizing for different definitions of "useful". Every stage here is lpv11-targeted; `test_spec.py` asserts it.
- **Cascade**: `decode → body_strip → fastText → tokenize → pooled → modernbert → resiliparse_rs`
  | stage | model | threshold |
  |---|---|---|
  | fastText | `useful_fasttext_lpv11/body_strip_scale_w640_sub0p22_strat_prep_mc500` | 0.130 |
  | **pooled** (new) | `mb-clf-lpv11-pooled-10M` | 0.178 |
  | ModernBERT | `mb-clf-lpv11-base-10M-c8192` | 0.410 |
  | extract | resiliparse-rs @ `850891b` (Rust `_extract_rs`) | — |
- **Pooled stage**: runs inside Phase B *before* ModernBERT on the same tokens (both use the ModernBERT tokenizer/pad, so Phase A is unchanged). At 3932.6 vs 21.5 docs/chip/s it costs ~0.7% of the TPU time while culling ~24% of pre-survivors. **`pooled_threshold` IS namespace-defining** — unlike `modernbert_threshold`, a pooled drop means ModernBERT never scores the doc, so there is no stored prob to re-threshold against. Retuning it requires a re-run.
- **resiliparse-rs vs jusText**: 291.8 vs 9.43 docs/s/core (**31x**), and the better lpv11 approximator (Levenshtein 0.726 vs 0.702; closer on 59.7% of docs). jusText was the pipeline's dominant cost. Pinned by fork commit because the extracted text IS the training text.
- **Thresholds** are the operating points the cascade planner resolved on the 100k comparison sample (fastText/pooled @ recall 0.95, ModernBERT @ 0.93) → F1 0.850 vs lpv11.

---

## ⚠ Hash-stability note (2026-08-11)

`_namespace_fields()` now **omits unset (None) fields**. Adding an optional field would otherwise change the hash of every spec that predates it, silently re-pointing a live namespace at an empty directory. Adding the pooled/extractor fields moved `fastpipe_v3` `da3893385e → 47cfdfc9d2`, which would have orphaned **3,602 already-extracted WARCs**. `test_spec.py` now pins published hashes so this fails loudly instead.

**`fastpipe_v3`'s real hash is `da3893385e`, NOT the `6855733850` recorded below.** Both namespaces exist in GCS: `6855733850` holds 82 kept parquet (abandoned), `da3893385e` holds ~3,602 (us-east5 3,020 + us-central1 582). A namespace-defining field changed after v3 was documented without a version bump. The section below is left as written for the record; treat `da3893385e` as canonical.

---

## fastpipe_v3  (ACTIVE, hq line)

- **Status**: active — the version run over the full 10,364-WARC pool, multi-region.
- **Hash**: `6855733850` → namespace `gs://marin-<region>/documents/fast_curation/fastpipe_v3-6855733850/`.
- **Changed vs v2**: `justext_max_html_chars` **3M → 50M** and `justext_timeout` **None → 60s**. Everything else identical to v2 (same models, thresholds, tokenizer, `V2_STEP_ORDER`).
- **Rationale**: v2's 3M-char html cap silently **dropped genuinely long documents** (long articles/books/docs pages — exactly the long-context training data we want). Raising the cap to 50M keeps them; a 60s per-doc timeout (hard-kill, since lxml's C parser ignores SIGALRM) guards the rare pathological page that would otherwise hang a worker. The 50M cap also fixed the Phase A OOM/oversized-parquet-cell failures that killed ~all v2 Phase A workers at 24GB (Phase A now also runs at 64GB and frees the decoded WARC before writing).
- **Note**: v2's partial output (~69/10364 WARCs) is **discarded** — v3 is a fresh namespace, not a resume of v2.

## fastpipe_v2

- **Status**: superseded by v3 (only ~69 WARCs were produced before the long-doc-drop issue was found; discarded).
- **Hash**: `34e3b152e0`. (The earlier live run used `f78c2b2b7a`, before `max_html_chars`/`timeout` became spec fields; that data is abandoned.)
- **Changed vs v1**: `step_order` → `decode → body_strip → fasttext → tokenize → modernbert → justext` (run ModernBERT **before** JustText so JustText, the dominant CPU cost, only touches ModernBERT-survivors ≈5× fewer docs). Output is intended to be identical to v1; only the compute order (and the 3-phase A/B/C layout) differs.
- **Rationale**: pure efficiency reorder of v1.

## fastpipe_v1

- **Status**: initial prototype (single-phase). Superseded by the v2/v3 3-phase layout.
- **Hash**: `2d1089842d`.
- **fastText**: `gs://marin-us-east5/classifiers/useful_fasttext/body_strip_scale_w320_strat_prep_mc500/model.bin`, threshold **0.0368**.
- **JustText**: `xenon-v4.2.0`, `max_html_chars=3M`, `timeout=None`.
- **Tokenizer**: `answerdotai/ModernBERT-base`, pad `50283`, `max_length=8192`, `single_window=True`.
- **ModernBERT**: `gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-base-10M-c8192/hf`, threshold **0.1974** on `P(useful)`.
- **step_order**: `decode → body_strip → fasttext → justext → tokenize → modernbert`.
- **Rationale**: first end-to-end version; reproduces the cascade the classifiers were trained/calibrated on. Single-window 8192 (chunked ModernBERT deferred — see `.agents/projects/chunked_modernbert_classifier.md`).
