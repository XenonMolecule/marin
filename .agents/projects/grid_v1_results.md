# Quality × Domain grid — RESULTS (2026-07-29)

24 WebOrganizer topics × 5 calibrated quality buckets, per corpus.
**ALL 5 CORPORA COMPLETE AND VERIFIED — 87,867,267 documents.**

Supersedes `grid_v1_topic_results.md` (topic axis only).

## Artifacts

```
gs://marin-us-central1/datakit/cluster_assign/{corpus}_gridv1/*.parquet
    id, native_id, url, cluster_24, topic_prob, token_length
gs://marin-us-central1/datakit/quality/{corpus}_gridv1/outputs/main/*.parquet
    source, id, score (CALIBRATED [0,1]), quality_bucket (0..4)
gs://marin-us-central1/datakit/quality/{corpus}_gridv1/outputs/samples/*.parquet
    + text, systematic 2% — for spot-checking cells
gs://marin-us-central1/metadata/grid_v1/{corpus}/distribution.json
    per-stage totals + the 24x5 grid in BOTH docs and tokens
```

Models: `WebOrganizer/TopicClassifier` (gte-base, JAX port, HF parity 3.3e-6) and
datakit's `pooled_junkgate2` pooled fast-transformer, calibration applied
(raw 0.2495/0.4419/0.6194/0.7514 → bucket edges 0.2/0.4/0.6/0.8).

## Headline numbers

| corpus | docs | score mean | q0 (junk) | q4 (top) | cells |
|---|---:|---:|---:|---:|---:|
| fineweb_edu | 2,349,555 | **0.589** | 2,082 | 12,429 | 118/120 |
| high_quality | 19,968,996 | 0.560 | 62,982 | **61,038** | **120/120** |
| dclm | 5,931,419 | 0.517 | 73,080 | 12,838 | 118/120 |
| fineweb_cc | **41,859,131** | 0.489 | 824,129 | 41,859 | 119/120 |
| nemotron | 17,758,166 | 0.486 | 387,401 | 38,820 | 120/120 |
| **total** | **87,867,267** | | **1,349,674** | **166,984** | |

Every corpus's grid total equals its topic total *exactly*, so co-partitioning held
across all 48,002 shards. high_quality's 19,968,996 also matches its independently
known corpus size to the document.

**fineweb_cc is the largest corpus of the five** at 41.86M documents — nearly as
many as the other four combined — and its profile confirms the prediction that a
permissive CC filter would resemble dclm/nemotron rather than fineweb_edu: mean
0.489 (vs nemotron's 0.486), topped by Entertainment (3.47M), Finance & Business
(3.42M), Sports & Fitness (3.31M) and Home & Hobbies (2.98M). General consumer web,
not an educational slice.

## Findings

**1. Quality is near-orthogonal to topic — which is what makes the grid useful.**
Per-topic mean quality spans only ~1.74–2.64 on a 0–4 scale, so quality is not a
proxy for domain. But the variation is real and interpretable: expository topics
(Science & Tech., History, Politics, Crime & Law) sit high; social, entertainment
and lifestyle topics sit low. Had the two axes been strongly correlated most of the
120 cells would be empty and the grid would be decorative.

Measured directly on dclm: Software Dev. mean 0.521, rank 9/23, *above* Finance
(0.503), Entertainment (0.489) and Literature (0.485). Full topic spread 1.46x,
versus 3.8x for the older `sonnet46-thr05` fastText (`stackv2_code` 0.219 vs
`pubmed` 0.725). Choosing the calibrated fast-transformer over that fastText was
load-bearing, not cosmetic.

**2. high_quality is the best general-purpose substrate.** Only corpus with all
120 cells populated, and 61,038 q4 documents — roughly 5x dclm or fineweb_edu. For
code it is not close: 3,347 q4 Software Dev. docs vs dclm's 342 (~10x).

**3. fineweb_edu is a specialist, not an upgrade.** Higher *rate* of quality than
dclm but fewer high-quality docs in absolute terms (1,117,593 vs 1,275,995 at
q3+q4), because dclm is 2.5x larger. It wins raw high-quality counts in only 6 of
24 topics — Science & Tech., Health, History, Industrial, Education & Jobs, Art &
Design — and loses the other 18, often heavily (Politics 62,465 vs 188,806).
Science & Tech. + Health + History = 52% of its tokens.

**4. nemotron is weakest per document but large.** Lowest mean (0.486) and by far
the most junk (387,401 in q0), yet still 2.74M docs at q3+q4 through sheer size.
Its Adult column is a clean demonstration of the quality axis working: 272,939
Adult documents, of which **4** reach q4.

**5. Adult content spans ~174x across corpora** by token share: fineweb_edu 0.01%,
high_quality 0.13%, dclm 1.14%, nemotron 1.74%.

**6. Doc share and token share diverge enough to matter.** dclm Politics is 9.29%
of documents but 12.01% of tokens; Literature 4.99% → 6.67%. Mixing on document
counts systematically under-weights long-form topics. Both are stored per cell.

## Cost

$0.76 one-time egress for 37.8 GB of mirrors (dclm, fineweb_edu, nemotron) plus
$1.09 for fineweb_cc — **$1.85 total**. high_quality is native to us-central1.
Mirrors are permanent, so re-runs are free. All TPU compute was free (TRC).

## Operational lessons

* **Only `v5p-8` schedules as one task.** Iris derives replicas as chips/8, so
  v5p-16/32 are gang-scheduled and never placed by preemption — 32 gang tasks sat
  PENDING 12 min against 116 free-on-paper hosts, while single-host requests were
  ASSIGNED in <30 s every time.
* **`interactive` preempts the extraction fleet's `batch` children automatically.**
  No coordination needed. But a CPU-only task will happily be placed on a TPU host
  with spare cores, so a CPU-bound job at `interactive` silently evicts accelerator
  work. Quality therefore runs at `batch`.
* **Zephyr is the right fan-out for the CPU stage** — small preemptible actors pack
  onto spare cores beside a running TPU job rather than claiming worker slots.
  Expect the coordinator to trip `max_task_failures` on a large corpus and just
  re-run; done markers make that free.
* **Ordering beats parallelism when running as filler.** Three concurrent quality
  jobs against a ~19-host CPU pool throttled each other to 0.73 shards/min; pausing
  the one whose grid could not yet be assembled restored the others to 5.75/min and
  261/min.
* Merges must run in-region — gcsfs cannot complete a TLS handshake from the dev
  laptop though `gcloud` can.

## Bugs found and fixed (all mine; tests added)

1. **Calibration never applied** — buckets were digitized from *raw* scores against
   edges defined in calibrated space. Materially wrong (q0 nearly tripled once
   fixed) and it would have quietly destroyed the cross-type coherence that
   justified this model. Cost ~6,400 shards of rework. Test pins the raw-0.20 case.
2. **Empty shards treated as fatal** — 7 of fineweb_edu's 1,469 and 13,541 of
   fineweb_cc's 21,531 shards are legitimately empty. Skipping them would have
   silently mis-aligned the store's positional join; now they emit zero-row files.
3. **`--output-base` moved the mirror input**, not just the output tree.
4. **Worker RAM at the module default** (8g) rather than upstream's production 16g.
   Killed nemotron and fineweb_edu; high_quality survived because fewer shards mean
   fewer distinct compiled shapes.
5. **`counters.pipeline.update_counter`** — upstream API absent here. Raised *after*
   each shard's outputs were written, so data completed correctly while every
   worker crashed and reloaded the 53 MB model per shard.
6. **zsh word-splitting** — `set -- $spec` does not split unquoted expansions in
   zsh, so three jobs received a malformed `--dataset`.
7. **Monitor could truncate, re-alarm, fabricate, and mis-signal** — fixed to strip
   iris chatter by pattern, dedupe on the failure set, ignore its own failed
   queries, and require `ok>0` before calling a wave drained (otherwise a
   wholly-failed wave reads as complete and triggers a merge over missing data).

## Materializing cells (2026-07-30)

The grid was an attribute table; training a mixture needs each cell as its own
Levanter cache, because `DatasetComponent` has no row-range or filter field. Three
new pieces, following upstream rather than reinventing it:

* `grid_tokenize.py` — `{id, input_ids}` parquet, one file per attribute shard,
  same basename, same row order. Tokenizes through Levanter's `BatchTokenizer`
  with `enforce_bos`/`enforce_eos`, the path training uses. Not
  `default_tokenize`, which bundles shards into size-balanced groups and so cannot
  be positionally joined.
* `experiments/datakit/store/` — upstream's `datakit_store` copied verbatim except
  for a documented fork note, plus `store_compat.py` for the symbols this checkout
  lacks. The shuffle, subshard plan and artifact are upstream's.
* `grid_store.py` / `verify_grid_store.py` — wiring and an independent recount.

Nothing reorders tokens or splits documents. A document moves from one file to
another intact; Levanter tracks boundaries with the same JaggedArray offsets.

### Five deviations from upstream, all deliberate

1. `StoragePath`, `read_artifact`, `write_artifact`, `deterministic_hash` don't
   exist here — `store_compat` supplies them. The hash must be stable **across
   processes** (blake2b, not Python's seeded `hash()`), or a retry scatters one
   document's tokens across two subshards.
2. Upstream's five artifact classes aren't ported; minimal shapes stand in.
3. **decon is optional.** Ours were deconned upstream (high_quality) or
   deliberately not (the rest). Passing an all-false decon table would falsely
   record that decontamination ran at store time.
4. **dedup is optional**, same reasoning.
5. **The per-bucket ledger merge is replaced.** Upstream writes a ledger over the
   `sub=*` child caches and reads it as a virtual concatenation; this levanter's
   `TreeCache.load(d)` opens a single tensorstore at `d`, so that ledger would
   load as an *empty* cache — a cell that looks fine and trains on nothing.
   `_finalize_buckets` instead resolves each cell concretely: one subshard means
   the sub cache *is* the cell (our case — hottest cell ~2.3B tokens vs the ~651B
   that forced upstream's 32-way split), more than one means a real physical
   concatenation via levanter's `consolidate_shard_caches`.

### Smoke result (4 dclm shards, real labels, us-central1)

108 of 120 cells, 230,116 documents, 280,468,138 marin tokens, every cell at
`n_shards=1`, zero documents dropped (`records_in == records_out`). Verified by
`verify_grid_store`, which recounts the grid straight from the topic and quality
parquets rather than trusting the artifact: cell membership matches exactly, token
totals match the tokenized parquets, and every cell opens through `TreeCache.load`.
1,219 tokens/doc against the gte-based estimate of 1,160, so the corpus-level
projections hold.

### Bugs the end-to-end tests caught

All four would have produced a store that looked correct.

1. **Every cell reported 0 tokens.** This checkout's `SerialCacheWriter` commits
   `field_counts={}`, so reading token counts off the ledger yields zero — which
   would have zeroed both the mixture weights and the next build's sizing hint.
   Tokens are now accumulated in the reducer.
2. **Empty shards failed the id check.** `pc.all` over an empty array returns
   *null*, so two zero-row shards read as mismatched. 13,541 of fineweb_cc's
   21,531 shards are empty, so this would have failed the largest corpus outright.
3. **Consolidation hardcoded an int32 exemplar**, which tensorstore refuses
   against an int64 cache. The dtype is now carried from the reducer.
4. **A tokenize shard longer than its attribute shards** ran off the end of the
   positional slices and surfaced as a bare `IndexError` from a reduce task. Now
   checked against Parquet metadata before streaming.

## ALL FIVE CORPORA MATERIALIZED AND CROSS-REGION-VERIFIED (2026-07-30)

| corpus | cells | docs | tokens | store size | us-east5 cost |
|---|---:|---:|---:|---:|---:|
| dclm | 118/120 | 5,931,419 | 7.33B | 15.47 GB | $0.31 |
| high_quality | 120/120 | 19,968,996 | 21.30B | 44.57 GB | $0.90 |
| fineweb_edu | 118/120 | 2,349,555 | 2.35B | ~9 GB | ~$0.18 |
| fineweb_cc | 119/120 | 41,859,131 | 28.00B | 59.28 GB | $1.19 |
| nemotron | 120/120 | 17,758,166 | 10.13B | 21.37 GB | $0.43 |
| **total** | | **87,867,267** | **69.11B** | ~150 GB | **~$3.01** |

Every corpus's per-cell store matches its topic-stage grid total exactly (docs
column above == the earlier grid totals in "Headline numbers"), so co-partitioning
held across tokenization AND the shuffle. Observed on-disk ratio is a consistent
**~2.10 bytes/token** (tensorstore compression, not raw int32) across all five —
dclm and high_quality calibrated it, fineweb_cc and nemotron's actuals landed
within the ±7% band projected from it.

Every store is now present, verified against its own source attribute tables
(`verify_grid_store.py`), replicated to **us-east5**, and independently
re-verified byte-for-byte via a fresh CRC32C comparison run *after* the copy job
reported success (not just trusting its own exit code) — a second bug
(`replicate_prefix.py`'s key-offset truncation, see below) justified the extra
step. Total replication cost **~$3.01**, each corpus confirmed individually under
budget before running.

Each corpus's `.artifact.json` (`ClusteredStoreData`) is a ready-to-use Levanter
mixture input: one `DatasetComponent`-shaped cache per populated `(cluster_24,
quality_bucket)` cell, in both us-central1 and us-east5. Mixing/training code
does not exist yet in this project — that was explicitly scoped as later work —
but every corpus is wired and waiting for it.

### Two more bugs found materializing the remaining three corpora

6. **`pc.equal` has no kernel for `(null, null)`.** `grid_label.py` builds each
   attribute table's `id` column from a bare Python list
   (`pa.table({"id": docs.ids, ...})`), so a genuinely empty shard infers the
   column as pyarrow's `null` type rather than `string` — there is nothing to
   infer a type from. This crashed the store on fineweb_edu (7 empty shards) and
   would have crashed it on fineweb_cc (13,541 empty shards, the majority of the
   corpus) had it not been caught first. Fixed by casting `id` columns to string
   at read time in the store; the already-written production parquet files
   needed no changes. Regression test reproduces the exact PyArrow error.
7. **`replicate_prefix.py` truncated every relative key by `len("gs://")`
   characters.** `fsspec`'s `find()` returns keys with the scheme stripped, but
   the offset for slicing a relative path was computed from the scheme-bearing
   root. `.artifact.json` (14 chars) became `fact.json` (9 chars) — exactly 5
   chars short — and the resulting copy 404'd immediately. Fixed by slicing from
   `fs._strip_protocol(root)`; regression test uses a stub filesystem that
   reproduces gcsfs's scheme-stripping behavior with no network dependency.

### Tokenizer-loading root causes (three distinct failures, same symptom class)

Tokenizing nemotron and fineweb_edu surfaced three unrelated problems that all
looked like "the HF Hub is flaky":

- **Thundering herd**: every worker in a wave calls `AutoTokenizer.from_pretrained`
  independently at startup — a few hundred simultaneous requests against a
  1000-req/5-min quota. Fixed by mirroring the tokenizer to GCS once
  (`grid_tokenize.py --prewarm-tokenizer`); every worker after that reads GCS,
  never the Hub.
- **Corrupted shared cache**: Iris injects `HF_HOME=~/.cache/huggingface` into
  every job by default, and worker VMs are long-lived and reused across task
  attempts (sometimes concurrently). One process's interrupted download left a
  half-written cache entry that every later process on that worker tripped over
  — reproduced as two different symptoms on the same worker with zero
  concurrency either time. Fixed by unconditionally overriding `HF_HOME` to a
  fresh per-process temp directory (the Iris default is exactly the poisoned
  shared path, so a conditional override never fires).
- **The prewarm step itself needed the Hub the first time**, which kept hitting
  the still-live quota from nemotron's old (unfixed) 256-worker job. Resolved by
  uploading the tokenizer directly from an already-resolved local HF cache
  (verified via vocab size 128,000+256 matching `_TOKENIZER_SIZES`, BOS/EOS
  tokens matching Llama-3 convention, and the blob filename's hash matching an
  independently computed SHA-256 of its contents) straight to the GCS mirror,
  bypassing the Hub entirely for the one-time setup too.

## Still open

1. **id-join coverage measurement on high_quality** (plan Phase 0 step 7). Not
   load-bearing for these five corpora, since each was labelled on the layer we
   want. It matters before relying on transferring labels across dedup layers,
   which is how resiliparse is planned.
2. **resiliparse** — needs its 10,364-WARC post-decon tree rebuilt first (does not
   exist in any region), carrying url/warc_record_id through dedup+decon, plus
   sign-off on ~$12.25 egress.
3. **Mixing/training launcher** — does not exist yet. Every store above is ready
   to be consumed by one the moment it does.
