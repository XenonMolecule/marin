# Dedupe observations — resiliparse and llm_curated (2026-04-29)

Fuzzy-document dedupe applied to the raw resiliparse and llm_curated WARC
extractions. Used marin's native pipeline
(`marin.processing.classification.deduplication.{exact,fuzzy}`) at default
parameters.

## Pipeline parameters

| Parameter | Value |
|---|---|
| MinHash permutations | 286 |
| LSH bands | 26 |
| n-gram size | 5 chars |
| MinHash seed | 42 |
| Approx. Jaccard threshold | ~0.75 |
| CC max iterations | 10 |
| Tokenizer | `meta-llama/Meta-Llama-3.1-8B` |

This is more aggressive than DCLM's BFF (word-13-gram, 0.8 ngram-overlap) and
Nemotron-CC's typical (260 perms, 20 bands). Numbers should run higher than
those published comparisons.

## Headline reductions

| Source | Original tokens | Deduped tokens | Token Δ | Original docs | Deduped docs | Doc Δ |
|---|---:|---:|---:|---:|---:|---:|
| **llm_curated** | 56.01 B | 43.64 B | **−22.1 %** | 103.7 M | 76.20 M | −26 % |
| **resiliparse** | 142.65 B | 94.33 B | **−33.9 %** | 141.6 M | 80.29 M | −42 % |

## Surprising finding: token reduction < doc reduction

| Source | Doc reduction | Token reduction | Avg tokens/doc (orig → dedup) |
|---|---|---|---|
| llm_curated | 26 % | **22 %** | 540 → 573 (+6 %) |
| resiliparse | 42 % | **34 %** | 1007 → 1175 (+17 %) |

Removed duplicates are **systematically shorter than non-duplicates**. The
gap is much larger for resiliparse (8-point gap) than llm_curated (4-point
gap). Avg tokens-per-doc rises +17 % for resiliparse vs +6 % for
llm_curated.

This is consistent with the long-tail in the cluster-size histogram: the
biggest clusters are nav menus, scrape artifacts, templated boilerplate —
all short. Resiliparse's raw HTML extraction picks up much more of this than
the LLM-curated path (which already discards many short outputs via
`min_output_chars=50` and `[NO_USEFUL_CONTENT]` filters).

## Cluster-size histograms

```
                     resiliparse        llm_curated
size 1     :      74,721,322          71,542,286     ← truly unique
size 2     :       4,312,707           3,395,931
size 3-5   :       1,620,677           1,157,575
size 6-10  :         391,012             260,838
size 11-100:         319,117             199,045
size >100  :          52,824              26,156     ← templated/boilerplate

Total      :      81,417,659          76,581,831
```

Same shape on both sources. Resiliparse has 2× the count in the >100 bin
(52k vs 26k clusters), confirming the "more boilerplate noise" picture.

## Side observation: extraction methods reflect their philosophies

- **Resiliparse**: aggressive HTML→text, kept short pages → long tail of
  boilerplate near-duplicates → larger token reduction post-dedup.
- **llm_curated**: LLM-rewritten with implicit deduplication via paraphrase
  generation and explicit length filters → less raw boilerplate → smaller
  reduction after additional dedup.

Net: dedup is more impactful on raw web extraction than on LLM-curated
text. If you want the cleanest input for the smallest hit, raw extraction +
fuzzy doc dedup gets you most of the way there.

## Pipeline implementation notes

What worked well:

- **Marin's native `dedup_fuzzy_document`** runs end-to-end on Iris/Zephyr.
  No Ray cluster needed, no GPU needed, no NeMo Curator setup.
- **`StepRunner` cache discipline**: when a downstream step fails, only
  failed steps re-run on resubmit. Prep + exact + fuzzy + apply all cached
  via `.executor_status=succeeded`. Saved hours when the stats helper
  crashed both times — only stats reran.

What needed iterating:

- **OOM at 32 GB worker RAM** for resiliparse fuzzy MinHash. Workers spiked
  to ~32 GB on the long-document tail (rare ~10 MB pages produce huge
  shingle sets at char-5-gram). Bumped to 64 GB; peaks landed at ~18–20 GB.
  Document this 32→64 jump as a permanent tail-safety margin for
  resiliparse-shaped data.
- **Production-priority preemption**: a concurrent `interactive`-priority
  job in us-central1 was preempted ~every 10–15 min by a `production`-tier
  tokenization fleet. Each preemption killed the marin driver and lost all
  in-flight fuzzy work — `dedup_fuzzy_document` has no checkpoint
  resumption (the `it_*/` parquets are overwritten on restart, not read).
  Eventually the production fleet went quiet and a clean run completed.
  Future production-blocked workloads need either `--priority production`
  (consumes those resources) or external coordination.
- **`cluster_size_histogram` worker RAM** undersized at 8 GB — value-counts
  pass over the CC final-iteration parquet OOM'd one shard which retried
  3× and aborted the helper for both runs. Bumped to 32 GB; both stats
  steps then ran in 8–17 min.
- **`it_N/` string-sort bug** in `cluster_size_histogram` — sorted strings
  put `"it_10"` between `"it_1"` and `"it_2"`, so `iter_dirs[-1]` returns
  `"it_9"` (last alphabetically, not last numerically). Both stats JSONs
  report `cc_final_iteration: 9` even though iter 10 ran for both. Numbers
  are approximately correct (iter 10 only flips a small number of
  `component_id`s). Trivial fix: natural-sort by extracting the int.
- **Marin executor's bucket-region check rejects `marin-tmp-*`** — the
  executor only accepts buckets matching `^marin-{region}/...` for region
  inference. Tokenization couldn't read the deduped trees from
  `marin-tmp-us-central{1,2}/...` directly; needed a same-region copy to
  permanent regional buckets first. (Free egress within region.)

## Wall-clock breakdown

| Source | prep | exact | fuzzy | apply | stats | tokenize | end-to-end |
|---|---|---|---|---|---|---|---|
| llm_curated | 5 min | 5–15 min | ~80 min (10 CC iters) | <5 min | 17 min (after RAM bump) | **18 min** | ~7 h (incl. preempt thrash) |
| resiliparse | 5 min | 30 min | ~3 h (10 CC iters, 2048 shards) | <10 min | 17 min (after RAM bump) | **2 h 14 min** | ~12 h |

Resiliparse fuzzy CC iterations averaged ~21 min/iter at 10× the doc count
of llm_curated. CC iter time scales much better than expected — iteration
cost is roughly linear in graph edge count, and LSH bucket counts grow
sub-linearly with doc count for typical text.

## Final artifacts

Deduped jsonl trees (permanent regional buckets):

```
gs://marin-us-central1/documents/baseline_llm_curated_deduped/   (200 shards)
gs://marin-us-central2/documents/baseline_resiliparse_deduped/   (3000 shards)
```

Tokenized caches (permanent regional buckets, Llama-3.1-8B tokenizer):

```
gs://marin-us-central1/tokenized/baseline_llm_curated_deduped-c444e2/train/
gs://marin-us-central2/tokenized/baseline_resiliparse_deduped-471baf/train/
```

Stats JSONs:

```
gs://marin-tmp-us-central1/ttl=14d/michaelryan/dedup_llm_curated/stats_*/dedup_stats.json
gs://marin-tmp-us-central2/ttl=14d/michaelryan/dedup_resiliparse/stats_*/dedup_stats.json
```

Source code:

```
experiments/baseline_collection/dedup/
├── prep_with_id.py            # adds synthetic id, projects text column
├── apply_dedup.py             # filter + cluster_size_histogram
├── dedup_resiliparse.py       # StepSpec wiring, us-central2
├── dedup_llm_curated.py       # StepSpec wiring, us-central1
├── tokenize_deduped.py        # post-dedup tokenization
└── promote_and_tokenize.sh    # one-shot copy-and-tokenize script
```

## Suggested follow-ups

1. **Train and compare**: point a fixed-model sweep at the new caches and
   see whether the deduped data improves paloma loss at fixed compute. If
   the curation_mystery_report's D11 hypothesis holds (effective unique
   content dominates), deduped resiliparse should do worse at small budgets
   (less data, more epoching) but better at large budgets.
2. **Inspect the >100-cluster bin**: pull a sample of 5–10 large clusters
   from the CC parquet, lookup record IDs, fetch their text. Characterize
   what's driving heavy duplication — likely boilerplate-heavy pages or
   scraper output. Could inform paragraph-level dedup as a future step.
3. **Patch the `it_N` string-sort bug** in
   `apply_dedup.py:cluster_size_histogram` — 5-line fix. Re-run stats on
   both sources to get the actual iter-10 distribution. Cosmetic.
4. **Try a more conservative fuzzy config** for an apples-to-apples
   comparison with DCLM's BFF or Nemotron-CC's MinHash — bump n-gram size
   to 24 chars or use word n-grams (would require dupekit extension), and
   raise Jaccard threshold to 0.8. Probably halves the dup count.
5. **Marin library improvements** worth filing: (a) checkpoint resumption
   in `connected_components` to survive driver preemption, (b) accept
   `marin-tmp-*` buckets in the executor's region-inference, (c)
   `apply_dedup.py:cluster_size_histogram` natural-sort for `it_N/` dirs.
