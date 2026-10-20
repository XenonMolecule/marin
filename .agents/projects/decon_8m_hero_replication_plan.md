# Decontaminate the 8M pool the way the hero run did

Goal: drop every document of the 8M c075 pool
(`gs://marin-us-central2/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/8582115warcs/deduped_c075/`,
9,951,767,085 docs) that the hero run's decontamination would drop, then re-tokenize.

## What the hero run does (branch `upstream/rav/hero_store_dup_075` @ c2f981fe3d)

- Eval corpus `aa-index-v4.1.1-v3` (`experiments/datakit/decontam/prepare_eval_corpus.py`):
  the 9 Artificial Analysis Intelligence Index v4.1.1 benchmarks (required) plus the
  lm-eval-harness tasks of every bundle in `experiments/evals/task_configs.py` (339 configured,
  809 leaf tasks staged, 118 failed to load and were skipped). Excludes wikitext, lambada*,
  code2text_*, jsonschema_bench_*, swde, realtoxicityprompts. Reading-comprehension passages are
  dropped (question + answer kept).
- `lib/marin/src/marin/datakit/decon.py`, `FEATURE_FILTER_VERSION = 3`: bloom of word 13-grams
  (plus an exact feature for short alphabetic paragraphs of >= 3 tokens); a doc is
  `contaminated` when >= 50% of some paragraph's 13-grams hit the bloom
  (`experiments/datakit/reference_pipeline.py`: `NGRAM_LENGTH = 13`, `OVERLAP_THRESHOLD = 0.5`).
- Boilerplate drop sets (`experiments/datakit/decontam/config.py`) remove common n-grams from
  matching: source-local (in >= 0.5% of a 5,000-doc sample, min 5 docs) and global (>= 50 docs of
  a 1M-doc sample, in >= 3 sources), computed across all 292 hero sources.
- The hero data store drops every contaminated doc.
- Production decon ran on CoreWeave (`s3://marin-us-east-02a/marin`); none of its blooms or drop
  sets are in GCS.

## What already exists in GCS

- The exact eval corpus: `gs://marin-us-central2/datakit/decontam/evals/aa-index-v4.1.1-v3/`
  (AA manifest `complete`; LMH manifest `complete_with_failures`, 809 artifacts). No rebuild
  needed.
- Blooms in us-central2 (`datakit/bloom/_combined_fixed_1f292909`, `..._388e03b1`) are from the
  SFT decon pipeline (5- and 8-grams) and do not match. us-west4's `_combined_fixed_a0dd549a` is
  the 2026-08-04 10% test run (13-gram but feature-filter v2, older corpus). Build our own.

## Steps

1. **Port** `lib/marin/src/marin/datakit/decon.py` (+283 lines: v3 features, required /
   best-effort manifest validation) and the decontam config/loader from the hero branch, with
   their tests (`tests/datakit/test_decon.py`). Add an explicit `id_field` parameter so the marker
   reads our `doc_id` (the pool is jsonl.gz keyed by `doc_id`; datakit parquet uses `id`).
2. **Bloom**: `build_eval_bloom_step` over the us-central2 eval corpus, 13-gram, threshold 0.5,
   v3, hero exclusions, required AA names. ~1.5M eval records; small job.
3. **Drop sets** (decision needed, see below).
4. **Smoke**: the 32-file subset in `tmp/ttl=7d/tok-smoke-8m/input/` plus planted eval items
   (a GSM8K test question, an MMLU item, an HLE item) must be flagged; inspect the flagged sample.
5. **Mark** the full pool (`decon_to_parquet`, one attributes parquet per input file, flagged
   sample 8 per shard). Reads 26.5 TB in-region; attributes ~0.5 TB.
6. **Filter**: drop contaminated ids, write
   `documents/baseline_lpv11_fastpipe_v2_1_fused_c075_herodecon_deduped/8582115warcs/deduped/`
   (~26 TB). Report docs/chars removed per source family.
7. **Tokenize** with `tokenize_lpv11.py` into
   `tokenized/lpv11_fastpipe_v2_1_fused_c075_herodecon_8582115warcs` (~18 h, ~33 TB); then the
   undecontaminated cache can be deleted.

## Decision: drop sets

- **A. Recompute on our pool** with snapshots as sources (125 CC-MAIN crawls, CC-NEWS,
  CC-SUPPLEMENTAL; output files group by crawl because units follow WARC path order). Same
  rules; the "in >= 3 sources" test then means "recurs across crawls". Close to the hero
  behavior, not byte-identical.
- **B. Use the hero production combined drop set** from CoreWeave. Exact replication, but needs
  read access to `s3://marin-us-east-02a/marin` (no credentials in `.env`); the set itself is
  small (n-gram hashes).

## Cost and time (estimates)

Compute is TRC (free). Storage: filtered text ~26 TB (~$530/month) and a new ~33 TB cache
(~$670/month), offset by deleting the undecontaminated cache. About 2 days end to end, most of
it the re-tokenize.
