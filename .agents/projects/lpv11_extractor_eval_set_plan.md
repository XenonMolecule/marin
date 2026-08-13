# Plan: extractor-eval train/dev/test export for llm_pipeline_v1_1

**Status:** ✅ COMPLETE (overnight 2026-08-05→06). Local artifacts in
`jusText/benchmark/datasets{,_rawhtml}/lpv11/` — dev 1000 / test 1000 / train 9999
/ big_train 100000, both variants, line-counts + gzip verified. GCS canonical:
`gs://marin-us-central2/datasets/extractor_eval_set/lpv11_12k{,_rawhtml}/`.
Script `build_lpv11_eval_set.py` UNCOMMITTED on `multi-spec-extraction`.
v1 export job was stopped (wedged ≥2h50m on unbounded preprocess regex); v2 added
90s SIGALRM fallback (25 pages hit it, flagged in manifest `preprocess_timeouts`)
+ GCS heartbeat. train=9999 not 10000: 1 of 9 duplicate warc_record_ids landed in
train's sample; export dedupes globally (documented, harmless).

## Overnight progress log

- Script: `experiments/baseline_collection/build_lpv11_eval_set.py` (7 stages:
  oldids/oldwarcs/select/assemble/finalize/join/export; `--tag smoke` isolates
  smoke dirs). Lint + pyrefly clean. NOT yet committed.
- DONE: `oldids` real (115,048 old-benchmark record ids → `lpv11_source/`).
- DONE: smoke `select`(2 WARCs/split) + `assemble` (8 WARCs staged, job
  `lpv11-evalset-smoke-assemble` SUCCEEDED).
- Smoke `finalize` correctly ABORTED on a real collision (smoke used an empty
  used-old stub; WARC c004d88c743b shares 420 ids with the old benchmark) —
  disjointness backstop verified working.
- GOTCHA CAUGHT: old distill shards are INDEX-named (`data-00000-of-03000.parquet`),
  not warc-hash-named — first oldwarcs run produced junk "hashes" (done∩used=0,
  statistically impossible given the smoke collision; that mismatch was the tell).
  Fixed by reading the shards' `warc_file` column instead of parsing filenames.
- IN FLIGHT: `lpv11-evalset-oldwarcs-r2` (us-central2) with the warc_file-based
  mapping; output unblocks real select AND the smoke re-select. User confirmed
  benchmarks are siloed, so old-benchmark exclusion is insurance, not required —
  kept because it's already built and free.
- SMOKE COMPLETE: full chain green; export schema verified in both variants.
- oldwarcs r2 done: old distill shards are INDEX-named → mapped via `warc_file`
  column; 477 used WARCs, done∩used=140 (matches expectation).
- FULL RUN: select (468 WARCs / 89 snapshots: dev 89, test 89, train 50,
  big_train 240), assemble 468/468, finalize exact (1000/1000/10000/100000,
  zero old-benchmark collisions), join 468/468.
- Join OOM saga: workers=4@12GB OOM → workers=2@8GB OOM (BytesIO copy doubles
  >1GB WARCs) → 12-shard fan-out workers=1: 10/12 ok, s0+s2 OOM → added
  STREAMING decode (`_streamable_warc`: chunked CC→ttl-cache copy + warcio reads
  from GCS stream, peak ~16MB) → retries completed. Streaming path is now the
  default decode.
- Export: serial preprocess has brutal tail (29s on one page) → ProcessPool
  (workers=8), smoke regression byte-identical, ~3.3x faster. 9 duplicate
  warc_record_ids found in joined data (pre-dedup re-emission) → export dedupes
  globally, final counts exact.
- IN FLIGHT: full export job `lpv11-evalset-export` → `lpv11_12k{,_rawhtml}/`.
- REMAINING: verify manifests/counts → pull to jusText
  `benchmark/datasets_rawhtml/lpv11/` + `benchmark/datasets/lpv11/` → memory.
Goal: replicate the `high_quality` extractor-benchmark export (the train/dev/test
jsonl.gz sets living in `~/Research/jusText/benchmark/datasets*/general/`) for the
`llm_pipeline_v1_1` (lpv11) extraction.

## How the original was built (recovered lineage)

Two-script pipeline, both in `experiments/baseline_collection/`:

1. **`build_hq_distill_dataset.py`** — built
   `gs://marin-us-central2/datasets/high_quality_3000_distill/data/` (3000 per-WARC
   parquet shards; `raw_html, reasoning_trace, final_output, url, warc_file,
   warc_record_id, snapshot`). Two iris CPU jobs:
   - `assemble` (us-central1): per-WARC metadata from the consolidated extraction
     archive via `resolved/resolved_high_quality.jsonl.gz` → staged to us-central2.
   - `htmljoin` (us-central2): join staged metadata to the decoded-HTML download
     (`raw/commoncrawl/baseline_3000-265ff5/data-{hash}.jsonl.gz`) on normalized
     record id (`<urn:uuid:X>` ↔ `warc_record_id`).
2. **`build_extractor_eval_set.py`** — sampled that parquet into WARC-disjoint,
   snapshot-stratified splits (seed 0, `held_out_indices` k=1 per snapshot →
   35 dev + 35 test WARCs; train 10k / dev 1k / test 1k docs; `big_train` 100k with
   reserve=128 + per-warc-cap=500; later `dev2`/`dev3` via `--mode new_dev`).
   Two html modes: `body_strip` (8B teacher input) and `raw`. Output
   `gs://marin-us-central2/datasets/extractor_eval_set/medium_12k{,_rawhtml}/`,
   pulled locally with one `gcloud storage cp` into the jusText benchmark dirs.

## What exists for lpv11 today (verified 2026-08-05)

- 10k-pool extraction in flight: 4,429 WARCs in
  `gs://marin-us-central1/documents/baseline_llm_extraction/llm_pipeline_v1_1/_completed/`.
- The **random-3k run (3,004-WARC manifest
  `experiments/distill/subsets/baseline_warcs_3000_random.txt`) is DONE and
  CONSOLIDATED**: `…/baseline_llm_extraction_consolidated/by_region/<r>/llm_pipeline_v1_1/`
  + winner manifest `resolved/resolved_llm_pipeline_v1_1.jsonl.gz` (+
  `done_warcs_llm_pipeline_v1_1.txt`) — the cross-region duplicate-copy problem
  (4.5% of (warc,batch) keys in >1 region) is already resolved there.
- lpv11 records carry `warc_record_id/url/warc_file/snapshot` **first-class**
  (`run_extract_standalone.py` `_process_batch_pipeline`, ~line 956) plus
  `pipeline_id`/`num_chunks` → the html join is a pure **ID join**, no exact-text
  matching (this is exactly the fix the HQ HF-export postmortem recommended).
- **No reasoning trace and no abstentions persisted** (KEEP docs only). Fine for
  the benchmark — it needs only html + final_output.
- Decoded raw HTML exists in us-central2 **only for the random-3k**
  (`raw/commoncrawl/baseline_3000_random-34884d/`, 3,003 shards, `{id, html, url,
  metadata.warc_file}`) and head-3k. The rest of the 10k pool has raw WARCs only
  (`dclm_400m_1x_10k-ee2365`); html would need re-decoding.
- **Teacher preprocessing differs from HQ:** lpv1.1 sees
  `pipelines/preprocessing.py::preprocess_html_for_extraction` (gold-safe: keeps
  prose-bearing scripts/head-meta), NOT `fasttext_useful_classifier.body_strip`.
  A "teacher input" html variant must use that function.

## CORRECTION (2026-08-05, verified locally)

The consolidated lpv11 done set (`done_warcs_llm_pipeline_v1_1.txt`, 3,000 hashes)
is **NOT** the `baseline_warcs_3000_random.txt` draw. Mapped via `_warc_path_hash`:
all 3,000 match the 10k manifest (`dclm_400m_1x_warcs.txt`) at positions 1–10,363;
only 1,423 ∩ random-3000 manifest (the known manifest-divergence number); 908 ∩
head-3005. Done pool spans **89 snapshots, 2013→2022, min 10 / med 35 WARCs per
snapshot**.

**Consequence:** the pre-decoded HTML store `baseline_3000_random-34884d` covers
only the biased 1,423-WARC subset (its non-head-3k part has NO 2013–2016
snapshots). Do NOT restrict selection to it. Instead **decode HTML for the
selected WARCs directly from the in-region 10k WARC pool**
`gs://marin-us-central2/raw/commoncrawl/dclm_400m_1x_10k-ee2365/` (us-central2 →
zero egress; reuse `decode_warcs_clean.py` / `_download_one_warc` with the U+FFFD
decoder fix). This replaces the `htmljoin`-against-download step.

## Agreed split structure (user, 2026-08-05)

- **dev = 1 WARC per snapshot (89), test = 1 WARC per snapshot (89)** — the old
  `held_out_indices` scheme over the done pool's 89 snapshots.
- **train = ~50 WARCs**, snapshot/year-stratified random sample.
- Stage ALL kept docs from selected WARCs (no cap-sampling needed at ~230 WARCs);
  sample final doc counts afterward in-region.
- Disjointness from the old benchmark: exclude the 908 done∩head-3005 WARCs from
  selection (covers old train/dev/test/big_train/dev2 sources) + id-filter against
  dev3's record ids (dev3 came from the random-3000 distill parquet).
- big_train (100k) if wanted: draw from the same selected train WARCs or widen
  train WARC count; decide at build time.

**Cost:** staging ~230 WARCs × ~20MB gz metadata ≈ 4–5GB central1→central2 →
**<$0.50**. WARC decode + join all in-region us-central2 (free egress, small CPU
job, ~180GB local reads). Local pull of final artifact (raw-html variant,
~112k docs × ~70KB) ≈ 8GB ≈ **$1–2**. Total ≈ **$2**.

## Proposed steps

1. **Join stage** — small adaptation of the `build_hq_distill_dataset.py` pattern
   (or a leaner sibling script `build_lpv11_eval_source.py`):
   read `resolved_llm_pipeline_v1_1.jsonl.gz`, per WARC pull winner batches from the
   consolidated us-central1 archive, join on record id to
   `baseline_3000_random-34884d` html in us-central2 → per-WARC parquet
   `gs://marin-us-central2/datasets/llm_pipeline_v1_1_3000_eval_source/data/`
   with `raw_html, final_output, url, warc_record_id, warc_file, snapshot,
   pipeline_id, num_chunks`.
   - **Cost control:** lpv11-3k pre-dedup text ≈ 170 GB (137.8B tok @10k × 3/10 ×
     4.31 chars/tok) → full-corpus staging central1→central2 ≈ $14–19, over the $10
     cap. Instead **pick the split WARCs first** (train ~32 + dev 35 + test 35
     [+ ~224 for big_train]) and stage only those → ~$1–2. Only build full-corpus if
     the user separately wants an lpv11 distill dataset (needs cost sign-off).
2. **Split stage** — `build_extractor_eval_set.py` needs a `--source-dir` for
   `splits`/`big_train` modes (currently hardwired to `USEFUL_DIR`) and a third
   `HtmlMode` (or replacement of BODY_STRIP) using `preprocess_html_for_extraction`.
   Same seed 0, k_holdout=1, 10k/1k/1k, WARC-disjoint; output
   `gs://marin-us-central2/datasets/extractor_eval_set/lpv11_12k{,_rawhtml}/`.
3. **Local pull** — `gcloud storage cp` into
   `jusText/benchmark/datasets_rawhtml/lpv11/` (+ body-variant dir if built).

## Open questions for the user

1. **Corpus scope:** random-3k now (recommended: done, consolidated, html on hand,
   snapshots 2013→2022) vs waiting for the 10k run (needs WARC re-decode for html).
2. **Split sizes:** same 10k/1k/1k (+100k big_train?) or different?
3. **Disjointness vs the old benchmark:** random-3k shares ~888 WARCs with head-3k.
   Enforce whole-WARC disjointness from the existing medium_12k splits (id-filter as
   in `new_dev`) so cross-corpus train/eval stays leak-free? Recommended: at minimum
   exclude the old dev/dev2/dev3/test WARCs from the new TRAIN split.
4. **HTML variants:** raw only, or also the lpv11-teacher-input variant
   (`preprocess_html_for_extraction`)?
