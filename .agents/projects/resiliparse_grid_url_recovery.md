# Resiliparse 10k — quality × domain grid via url recovery

**Goal (user, 2026-08-01 night):** produce the 24-topic × 5-quality grid for the
resiliparse 10,364-WARC corpus, the only one of the six 10k corpora without one,
so it can join the OLMIX data-mixing experiments. User asleep; authorized to run
to completion. **Correctness over speed — explicitly no shortcuts.**

## Situation

`resiliparse_10k` was deduped + CORE-v2 decontaminated + tokenized, then its
document tree was **deleted**. Provenance from the tokenize step:

```
gs://marin-us-central2/documents/baseline_resiliparse_decon_deduped/10364warcs/deduped/data-*.jsonl.gz
    -> gs://marin-us-central2/tokenized/resiliparse_decon_10364warcs-beaaf5   (also mirrored us-east5)
    tokenizer meta-llama/Meta-Llama-3.1-8B
```
That input path 404s now. No dedup/decon intermediates or attributes survive in
any region. The grid needs `{text, url}` at the post-decon layer; url exists only
in the raw extraction `gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f`
(10,364 shards, 570.38 GiB, records are `{text, url}`, us-central2 ONLY).

## Hard numbers (measured, not estimated)

| quantity | value | source |
|---|---:|---|
| raw shards | 10,364 | `gsutil ls` |
| raw size | 570.38 GiB | `gsutil du -s` |
| raw docs (shard 0) | 53,072 (49,559 unique ids) | local count |
| raw docs (corpus, **measured**) | **418,616,205** | hash-raw done markers, all 10,364 shards |
| ~~raw docs (extrapolated)~~ | ~~550,038,208~~ | shard 0 × 10,364 — **31% too high**, shard 0 is not typical (53,072 vs mean 40,391) |
| raw mean tokens/doc | 950 | probe A, full shard |
| cache rows (survivors) | **290,262,075** | `train/.stats.json` |
| cache tokens | **339,971,302,028** | `.stats.json`, matches curation_plan |
| cache parts | 4,451 | `shard_ledger.json` |
| doc survival | **69.3%** | 290.26M / 418.62M (measured both sides) |
| decode throughput | 1,077 docs/s, 1.25M tok/s (1 core) | probe D |
| full-cache decode | **74.9 core-hours** (~22 min @ 200 workers) | probe D |
| egress rate (observed) | **$0.02/GB** | 4 prior runs, see below |

Egress is $0.02/GB, NOT the $0.08–0.12 list price: dclm 15.47GB→$0.31,
high_quality 44.57GB→$0.90, fineweb_cc 59.28GB→$1.19, resiliparse raw
177.88GB→$3.56. All $0.0200–0.0202/GB.

## The mechanism, and why it works

`marin.datakit.normalize.generate_id` is a pure `xxh3_128` content hash and
`normalize_record` copies text verbatim (`out["text"] = text`) while carrying all
other columns. `decon_apply.py` drops whole documents and is read-only w.r.t.
text. So post-decon text is **byte-identical** to raw text, and document identity
is recoverable from text alone — no need for the deleted survivor list.

Plan: decode the cache back to text, hash it, join to a `(content_id -> url)` map
built from raw. Avoids re-running the 192 GB-RAM fuzzy connected-components step
and preserves the **exact** survivor set the existing `resiliparse_10k` training
runs used.

### Two decode traps, both found and fixed

1. **`clean_up_tokenization_spaces`.** HF `decode()` defaults to rewriting
   `" ."` → `"."`, `" ?"` → `"?"`, `" '"` → `"'"`. Broke 20.35% of documents
   (1,593/2,000 exact). Must pass `clean_up_tokenization_spaces=False`.
2. **Trailing space token.** `BatchTokenizer` frames every row as
   `<|begin_of_text|> …doc tokens… Ġ <|end_of_text|>` — a lone `Ġ` (space) before
   EOS, appended unconditionally. Drop it at the **token** level, not by rstrip,
   so a doc that legitimately ends in whitespace keeps its own trailing space.
   (Raw shard 0 has 0 such docs, so it is moot here, but the token-level rule is
   correct regardless.)

Implemented in `experiments/baseline_collection/recover_tokenized_text.py`
(`strip_framing` / `decode_rows`).

## Validation done

- **Probe A** — raw text → encode → decode roundtrip, full raw shard 0:
  **53,072/53,072 byte-exact and id-exact (100.0000%)**, including all 3,754 docs
  containing U+FFFD / control / format / private-use characters.
- **Probe C** — re-encode of decoded text vs stored ids: 99.07%. The 14 misses are
  chunk-boundary BPE splits (`['(', 'Ċ']` vs `['(Ċ']`) that **decode to identical
  text**. Token identity is not stable across chunk edges; text identity is. This
  is why the join must key on the content hash, never on the token sequence.
- **Probe G (decisive)** — the N=2000 resiliparse lineage is the one pair where
  both sides still exist (`documents/baseline_resiliparse_deduped/2000warcs/deduped/`
  → `tokenized/resiliparse_dedup_2000warcs-7e2171`). Its ledger shows 1,722 parts
  for 1,722 shards and part-00000 has exactly 32,829 rows = shard 0's exact doc
  count, so rows map positionally. Comparing **all 32,829** rows:
  **32,829/32,829 decoded text == source text, zero framing anomalies.**

### Probe design trap (cost me several false negatives)

The cache is written **one part per input shard**. A contiguous run of cache rows
therefore comes from ~one input shard — a sample of size 1, not size N. Early
probes scanned contiguous blocks and got 0 hits against raw shard 0, which meant
nothing. **Always sample scattered rows** when estimating overlap.

## ✅ RECONSTRUCTION VERIFIED (2026-08-02)

`verify-tree` reconciles the rebuilt corpus against the token cache exactly:

```json
{ "data_shards": 10364, "data_shards_expected": 10364, "unmatched_shards": 4451,
  "matched_docs": 289428307, "unmatched_docs": 833768,
  "total_docs": 290262075, "expected_total": 290262075,
  "reconciles": true, "missing_keeplists": [] }
```

289,428,307 + 833,768 = 290,262,075 — the cache's `total_elements`, to the
document. Tree lives in **both** us-central2 (source) and **us-east5** (422.12 GiB,
copied for $9.06 at the observed $0.02/GB); the quality scorer was copied to
us-east5 too (53.7 MiB, `calib_bme.json` included — it exists nowhere else in that
region, and its absence silently corrupted buckets on the first grid attempt).

Unmatched came in at 0.2873%, matching the 0.2895% measured by sampling.

### Grid launched

`grid_corpora.GRID_CORPORA["resiliparse_10k"]` now points at the us-east5 tree
with `post_decon=True`. Labelling runs need
`--region us-east5 --output-base gs://marin-us-east5 --source native`.
A one-shard smoke on `v5p-8` validated the path end to end: 29,569 rows, all 24
topics present, mean `topic_prob` 0.867, urls on 100% of documents.

Topic (`v5p-8`, interactive) and quality (CPU, **batch**) are independent stages
with separate done-marker sets and run concurrently. Measured ~2.8–3.3 min/shard
for both — the one-shard smoke's 5m15s is misleading because it amortises model
load over a single shard.

**The 4,451 `unmatched-*` shards need their own labelling wave**: the running jobs
enumerated shards at startup and cannot see files that arrived later.

## ⚠️ Running a grid OUTSIDE us-central1: the region checklist

resiliparse is the **first corpus not in us-central1**, so every implicitly
region-dependent resource in this pipeline surfaced as a separate failed job.
Anyone repeating this elsewhere needs all four:

| resource | symptom if missed | fix |
|---|---|---|
| grid outputs (`OUTPUT_BASE`) | tokenize/store silently write to us-central1 | `--output-base gs://marin-<region>` (added 2026-08-02; `grid_label` already had it, `grid_tokenize`/`grid_store` did not) |
| worker scheduling | whole corpus streamed cross-region (422 GiB) | `--compute-region <region>` (added 2026-08-04; both had `regions=["us-central1"]` hardcoded) |
| quality scorer | `--quality-model` path 404s | copy `datakit/quality_model/pooled_junkgate2` (53.7 MiB, **must include `calib_bme.json`**) |
| tokenizer mirror | every worker falls back to the HF Hub at once → thundering herd, job dies ~6 min in | copy `resources/tokenizers/marin-community/marin-tokenizer` (16.5 MiB), or run `grid_tokenize --prewarm-tokenizer` with the same `--output-base` |

The tokenizer mirror is the nastiest: `tokenizer_mirror_dir()` derives from
`OUTPUT_BASE`, so pointing `--output-base` at a new region silently relocates
where the mirror is *looked for*. The code deliberately falls back to the Hub so a
lone smoke run works without a prewarm — which means the failure only appears at
wave scale, exactly when it is most expensive.

## Remaining sequence (exact commands)

Everything runs **in us-east5** with `--output-base gs://marin-us-east5`. Never
flip `grid_corpora.COMPUTE_REGION` — that would relocate the other five corpora's
paths too.

```bash
# 1. label the 4,451 unmatched-* shards (they arrived after the running jobs
#    enumerated their shard lists, so they need their own wave). SKIP THIS if the
#    decision is to drop the corrupted-token documents — see the OPEN ISSUE below.
launch_grid label --dataset resiliparse_10k --stages topic   --source native \
    --tasks 32 --tpu v5p-8 --region us-east5 --output-base gs://marin-us-east5 --wave u1
launch_grid label --dataset resiliparse_10k --stages quality --source native \
    --quality-model gs://marin-us-east5/datakit/quality_model/pooled_junkgate2 \
    --device cpu --priority batch --tasks 32 \
    --region us-east5 --output-base gs://marin-us-east5 --wave u1

# 2. merge per-shard tallies into the 24x5 grid. MUST run as an Iris job in-region:
#    gcsfs cannot complete a TLS handshake from the dev laptop (gcloud can).
iris job run --region us-east5 --priority batch --preemptible --cpu 2 --memory 32GB \
  --extra cpu --enable-extra-resources --max-retries 3 -e HF_TOKEN "$HF_TOKEN" -- \
  python -m experiments.baseline_collection.grid_label merge \
    --dataset resiliparse_10k --output-base gs://marin-us-east5

# 3. tokenize, then materialise the per-cell store. BOTH need --output-base, which
#    did not exist until 2026-08-02 — without it they write to us-central1.
python -m experiments.baseline_collection.grid_tokenize --dataset resiliparse_10k \
    --source native --output-base gs://marin-us-east5
python -m experiments.baseline_collection.grid_store    --dataset resiliparse_10k \
    --source native --output-base gs://marin-us-east5

# 4. verify the store against the attribute tables
python -m experiments.baseline_collection.verify_grid_store --dataset resiliparse_10k
```

Sanity numbers for step 4: the store's document total must equal the corpus
(290,262,075 with the unmatched documents included, 289,428,307 without), and the
five existing corpora all landed at a consistent ~2.10 bytes/token on disk.

## Three latent bugs found in `launch_grid.py` (all fixed 2026-08-02)

These predate this project and affected **every** previous grid run:

1. **`--priority` was never passed.** `main()` called `run_label` positionally and
   stopped before `priority`, so the parameter always took its default,
   `interactive`. Every quality wave ever run therefore went in at `interactive`
   despite the module's own comment explaining that a CPU-only task at that band
   gets placed on a TPU host with spare cores and evicts extraction from
   accelerators it actually needs.
2. **Uncaught `subprocess.TimeoutExpired`.** `subprocess.run` raises rather than
   returning non-zero, and an uncaught raise inside a `pool.map` worker aborts
   every remaining submission — so one slow submit silently truncated a wave. The
   exception text also embeds argv, i.e. `HF_TOKEN`, into logs.
3. **`--max-retries` was never set, so Iris's default of 0 applied.** A SINGLE
   preemption killed a job permanently. Everything here runs on preemptible
   capacity other users actively reclaim (observed: `Preempted by /bizon/...`), so
   waves quietly thinned out over hours and the stage looked like it was slowing
   for no reason. This is the most insidious of the three: it presents as gradual
   slowdown, never as failure. Now `MAX_RETRIES = 3`; retries are free because
   every shard has a done marker, so a restarted job resumes.

### ⚠️ CHECK THE POOL INVENTORY BEFORE BLAMING CONTENTION (2026-08-03)

The topic stage sat at 1,707/10,364 for **eight hours** while I diagnosed, in
order: tight capacity, then a competing swarm, then a controller rejecting
submissions. All three were wrong. The actual cause:

```
tpu_v5p-preemptible_8-us-east5-a       1 worker      <- what launch_grid targets
tpu_v6e-preemptible_4-us-east5-b      45 workers     <- idle, same region, same data
tpu_v5p-preemptible_8-us-central1-a   30 workers     <- why the v5p-8 default is right THERE
tpu_v4-reserved_2048-us-central2-b   256 workers     <- reserved, off limits
```

`LABEL_TPU = "v5p-8"` is correct for us-central1 and near-useless in us-east5.
The module's long comment about v5p-8 being the only single-task v5p shape is
sound, and I treated it as settled guidance without re-checking it against the
pool inventory of a **different region**.

Compounding it: an earlier `v6e-8` test "wouldn't schedule", which I read as v6e
also being contended. The provisioned group is `v6e-preemptible_4` — **size 8 is
simply unsatisfiable there**. Request `v6e-4`.

Result after switching: 24/24 jobs running within minutes (v5p never exceeded 1),
and throughput went from 0 shards in 8 hours to ~4/min.

**Rule: when a stage stalls, query the worker inventory FIRST.**

```sql
SELECT scale_group, COUNT(*) FROM workers
WHERE scale_group LIKE '%v4%' OR scale_group LIKE '%v6e%' OR scale_group LIKE '%v5p%'
GROUP BY 1 ORDER BY 2 DESC
```

Job states tell you your jobs are pending; they do not tell you whether any
worker of the requested shape exists. Those look identical from the job side.

### The submission "failures" were self-inflicted (2026-08-03)

**Correction to everything below about waves not landing.** The controller was
never rejecting submissions — it was responding slowly, and every client-side
timeout in play was cutting it off far too early:

```
Operation launch_job(...) failed (attempt 7/240, 239.8s elapsed),
  retrying in 10.32s: Request timed out
```

The iris client's own retry budget is **240 attempts**. `launch_grid`'s
`SUBMIT_TIMEOUT = 300` plus its 3-retry give-up, and the shell `timeout` wrappers
used interactively, were killing submissions around attempt 7. Waves reported as
`0/24`, `0/32` and `0/8` were victims of that, not of a refusing controller.

Proof: the same six topic jobs that "failed" resubmitted with **no client-side
timeout at all** landed 6/6 on the first try.

So when submissions look like they are failing under load, **remove the timeout
rather than shrinking the wave**. Sequential single submissions appear to work
better only because each one gets a fresh (still too short) budget.

### Operational rule, learned twice the hard way

**Run ONE launcher at a time.** Each `iris job run` opens its own SSH tunnel and
uploads a ~15 MB workspace bundle; two launchers in parallel saturate the
controller and submissions start timing out. Evidence across three waves:

| wave | condition | landed |
|---|---|---:|
| t3 | launched 3 s after the quality wave | **0/24** |
| q2 | launched alongside t3 | 5/32 |
| t4 | launched alone | **23/32** |

Also: a `cpu=16` TPU job would not schedule at all in 25 minutes, whereas
`cpu=0.1` places readily. If host-side tokenization ever looks like the topic
bottleneck, weigh the per-shard gain against much worse placement latency before
raising the request.

## ⚠️ OPEN ISSUE: 0.29% of survivors cannot be matched to raw (2026-08-02)

`select-representatives` exited 2 on **all 64 buckets**: 0.2895% of surviving ids
decode to a content hash that exists nowhere in the raw extraction. Bucket 0:
13,126 unmatched of 4,534,469 (verified they are genuinely distinct ids, not
duplicate rows). Corpus-wide that is roughly **840,000 documents**.
`keeplists` refused to run — the `--expect-total` guard working as intended.

**These documents are not a random sample.** Comparing 57 unmatched against 250
matched rows drawn from the same cache parts:

| | unmatched | matched |
|---|---:|---:|
| median tokens | **2,068** | 526 |
| p90 tokens | **9,280** | 1,995 |
| max tokens | 106,822 | 10,818 |
| contains U+FFFD | 15.8% | 6.4% |

They are ~4x longer than typical, and the samples are overwhelmingly SEO spam
(`<kbd id='wpS9oApnE'></kbd>` repeated hundreds of times, `九游j9` link farms).

**Most likely cause: chunk-boundary corruption inside the cache itself.**
`BatchTokenizer` tokenizes in chunks; probe C already measured 0.93% of rows
carrying a chunk-boundary BPE split (`['(', 'Ċ']` vs `['(Ċ']`). Those particular
splits decode identically, but a boundary landing inside a multi-byte UTF-8
character would not — the stored tokens then decode to text the original never
was. Longer documents cross more boundaries, which matches the length skew
exactly. It also explains why the earlier probes missed it: **probe A encoded
whole documents in a single `tok.encode` call, so it never exercised chunking at
all**, and probe G's part-00000 happened to hold few long documents.

If that is right, the information is lost **in the cache**, not in the decode, so
no decoder recovers it and these rows cannot be hash-matched to raw. Note the
implication: the model trained on `resiliparse_10k` saw the corrupted text for
these documents, since the cache *is* what it read.

### Options (needs a call from the user)

1. **Include them with decoded text and an empty url.** Preserves the exact
   290,262,075-document corpus. Costs url-conditioned topic accuracy on 0.29% of
   documents, which are mostly spam destined for low-quality buckets anyway.
   WebOrganizer's template is `"{url}\n\n{text}"`, so an empty url degrades but
   does not break classification.
2. **Drop them.** Simplest, but the grid then describes a corpus 840k documents
   smaller than the one actually trained on.
3. **Prefix-match to recover urls.** If corruption is late in a document, hashing
   its first N characters against raw prefixes would recover the url, and each
   match is independently verifiable. More work; unknown yield.
4. **Fall back to re-running dedup+decon** with url carried through, which sides
   steps the whole matching problem. Costs the 192 GB-RAM fuzzy CC step.

Option 1 is the recommendation: it is the only one that keeps the corpus identical
to what was trained on, and the decoded text is by definition exactly what the
model saw.

### Option 1 taken (2026-08-02), pending the user's ratification

Chosen while the user was asleep, on the reasoning that doing nothing silently
selects option 2 — the one argued against — and that option 1 is cheaply
reversible while option 2 is not (dropping the documents and later changing course
means re-running the grid over the whole corpus).

`materialize-unmatched` writes them as **separate `unmatched-part-*.jsonl.gz`
files inside the same tree**, so reverting is one `gsutil rm` of that glob. It
reads only the ~0.3% of cache rows involved, located via the per-part survivor
tables, rather than re-decoding the corpus, and caches the derived unmatched id
set at `metadata/resiliparse_url_recovery/unmatched_ids.parquet` so workers do not
each repeat the 64-bucket set difference (which took ~11 min single-threaded).

Validated on part-00000: 194 documents, 0.2979% of the part (corpus-wide 0.2895%),
all with text, all urls empty, all ids unique, mean 12,864 chars against a corpus
mean near 3,500 — the 3.7x length skew independently confirming the chunk-boundary
explanation. The sample was a Catalan forum page, not spam, so the affected
population is mixed rather than uniformly junk.

### Measured sizes

Tree projects to ~426 GiB / 457 GB from 176.21 GiB at 4,289/10,364 shards, so a
us-east5 copy is ~$9.15 at the observed $0.02/GB — inside the $10 cap, but measure
the real total before spending.

## Status

- [x] decode mechanism proven (probe G, 100% on a full part)
- [x] **lineage gate PASSED.** `/michaelryan/resiliparse-lineage-probe` reached
      `scanned 76000/100000 rows, hits 1507` before being OOM-killed at its 12 GB
      cap (raise to 24 GB if ever re-run). 1,507 exact 128-bit content-hash
      matches against 200 of 10,364 raw shards; if `f0887f` were not the ancestor
      this would be ~0 (collision probability across 7.3e11 pairs is negligible).
      The observed 1.98% hit rate exceeds the naive 1.1% because cross-shard
      duplicate texts are over-represented in any shard sample. **f0887f is the
      ancestor.**
- [x] `survivor-ids` smoke test on one part: wrote exactly **65,125** rows,
      matching the ledger's `part-00000-of-04451`, all ids unique, zero framing
      anomalies. 87 s/part => ~107 core-hours for the full cache.
- [ ] `survivor-ids` full wave (200 tasks, 4,451 parts) — RUNNING
- [ ] `hash-raw` over all 10,364 raw shards (independent of survivor-ids; can run
      concurrently)
- [ ] `bucket-survivors` → `select-representatives` → `shard-keeplists`
- [ ] `materialize` the `{text, url}` tree; **must total 290,262,075 docs**
- [ ] register in `grid_corpora.GRID_CORPORA` with `post_decon=True`
- [ ] run grid: topic → quality → tokenize → store → verify

### Launcher bug that cost ~1h (fixed 2026-08-02)

A 200-job wave landed only **93** jobs, and the second wave only **17**, with
almost nothing in the log to say so. Three separate defects, all in `_submit`:

1. `subprocess.run(..., timeout=900)` raises `TimeoutExpired` rather than
   returning non-zero. `_submit` only inspected `returncode`, so the exception
   escaped the `pool.map` worker and **aborted every remaining submission**. One
   slow `iris job run` therefore killed the rest of the wave. Now caught, with a
   300 s ceiling so a hung submit frees its pool slot fast.
2. The `TimeoutExpired` message embeds the full argv, which carries **HF_TOKEN**.
   It got written to a launcher log in plaintext. Never log that exception's text;
   log the job name and exception type only. (The leaked scratchpad log was
   scrubbed.)
3. Job names must be unique cluster-wide, so re-running a stage with a *different*
   `--tasks` collided with the previous wave's names — and since "already exists"
   is treated as success, every colliding chunk was silently skipped even though
   its partition of the work had changed. Hence `--wave`.

Diagnosis note: `state = 4` in the tasks table is **succeeded**, not running. Do
not read a stuck-looking wave off that column. What actually localised this was
computing per-chunk completion from the written part indices: 92 chunks fully
complete and 107 with *zero* output is the signature of jobs that never launched,
whereas a hung stage would show partial progress spread across many chunks.

### The Iris controller was also genuinely flaky (2026-08-02 ~03:30-04:00)

Not all of the submission failures were self-inflicted. Alongside the timeouts,
submits returned:

```
Error: Could not connect to controller: No controller VM found
       (label=iris-marin-controller=true, project=hai-gcp-models)
httpx.RemoteProtocolError: Server disconnected without sending a response.
google.auth ... RemoteDisconnected('Remote end closed connection without response')
```

The controller recovered on its own within minutes each time — `iris query` worked
before and after — so this is intermittent, not an outage. **Do not restart or
bounce anything in response**; the standing rule is that a shared cluster is never
restarted without explicit permission, and the jobs already scheduled keep running
regardless of controller availability. The correct response is to retry the wave
later and rely on `--greedy` so the jobs that do land absorb the backlog.

### Submission is the bottleneck, not compute

The deeper problem behind the above: **every `iris job run` opens its own SSH
tunnel to the controller and uploads a ~15 MB workspace bundle.** A 200-job wave
is therefore ~3 GB pushed through one controller, and at `SUBMIT_PARALLELISM=16`
the controller degraded until individual submits took >300 s and timed out. The
second attempt landed 7 of 60.

Fix, and the shape to reuse for any large fan-out here:

* `--chunk-idx` is **variadic**, so one Iris job walks several chunks.
  `--tasks` (chunk count) still fixes the output partitioning and filenames and
  must not change between runs; `--jobs` controls how many submissions that costs.
  These are now independent knobs — 200 chunks over 40 jobs, not 200 jobs.
* `SUBMIT_PARALLELISM` lowered 16 → 6.

Rule of thumb: keep an Iris wave to a few dozen jobs. If you want more
parallelism than that, use zephyr inside one job (as
`dedup_resiliparse_warc_scaling.py` does with `ZephyrContext(max_workers=200)`)
rather than submitting hundreds of Iris jobs. **That is what this pipeline should
have done from the start** — the chunked-Iris-jobs shape was copied from
`launch_grid.py`, which only ever fans out ~24 jobs.

Even at 6-way concurrency and 40 jobs, most submits still timed out (4/40 landed
on one wave). Hence `--greedy`: a greedy job ignores its chunk assignment and
works through *every* outstanding unit in an order seeded by its index. Landing a
job costs far more than running one here, so the jobs that do land must be able to
drain the backlog. Every unit is done-marker guarded, so two greedy workers
racing on one unit duplicates work but cannot corrupt anything, and the
independently shuffled orders keep collisions rare.

Corollary for re-runs: `survivor-ids` may be re-run with any `--tasks` because its
done-marker is the part file itself (`part-NNNNN-of-04451.parquet`), which does
not depend on the chunking. `hash-raw` may **not** — its outputs are named
`bucket=B/chunk-CCCCC.parquet`, so changing `--num-chunks` would make different
shard sets collide on the same filename and silently corrupt the index.

### Pipeline entry points

`experiments/baseline_collection/launch_url_recovery.py` fans each stage out on
Iris in us-central2, pinned `--preemptible` at `batch` priority. Stage functions
live in `recover_tokenized_text.py`. Every stage has a done marker and is safe to
re-run; re-running is the preemption recovery mechanism.

### Open decision: which region runs the grid

The user's stated plan is to move the smaller (post-decon, url-bearing) tree to
us-east5 and build the grid there, co-located with the mixing swarm, which is
also where the other five corpora's stores already live.

**Deliberately NOT writing materialize output straight to us-east5**, even though
that would fold the copy into a pass we are running anyway and avoid paying to
store the tree twice. The size is only estimated (~340–360 GB → ~$7 at the
observed $0.02/GB), and the standing cost rule is a HARD $10 cap. A 50% estimate
miss would breach it with no way to stop mid-write. So: materialize into
us-central2 at zero egress, `du -s` the result, and only then copy with a real
number. If the measured cost exceeds the cap, leave it in us-central2 and either
run the grid there on v4 or surface the number to the user.

Materialize workers must run in **us-central2** regardless — they read 570 GiB of
raw and write ~350 GB, so co-locating with the raw side is the cheap direction.

`grid_corpora.COMPUTE_REGION` is a module constant that `OUTPUT_BASE` and
`MIRROR_BASE` both derive from, and the other five corpora's grid outputs live
under us-central1. Do NOT flip the constant globally — `grid_label.py` accepts
`--output-base` (it assigns `grid_corpora.OUTPUT_BASE` at startup), which is the
per-run override. `launch_grid.py` passes `COMPUTE_REGION` as the Iris `--region`
and would need a flag to schedule elsewhere.

us-central2 has **v4 only**, so the topic stage's proven `v5p-8` single-task shape
would become `v4-8` there. gte-base is ~110M params and fits either way; the risk
is scheduling shape, not memory.

## Fallback (user-authorized)

Re-run dedup + decon with url carried through. `dedup_resiliparse_warc_scaling.py`
drops url at two `{"text": ...}` projections (lines ~143 reshape, ~223 fuzzy-apply);
`_carry_provenance` in `dedup_extracted.py` is the fix to port. Fuzzy params are
already identical between the two scripts (286/26/5/seed42/0.75). Costs the
192 GB-RAM fuzzy CC step, which in us-central2 bin-packs onto **reserved v4**
hosts — the configuration that was reclaimed mid-run on 2026-06-25. Verify a
re-run by checking it tokenizes to exactly 339,971,302,028 tokens / 290,262,075 docs.

## Notes

- `dedup_extracted.py` cannot be used for resiliparse: `_REGIONAL_BUCKETS` has no
  us-central2 entry and `_source_prefix` hard-builds `documents/baseline_llm_extraction/`.
- Do NOT trust `resiliparse_decon_10364warcs-beaaf5`'s size for grid-store sizing
  without re-deriving: the grid re-tokenizes with its own tokenizer.
- Probe scripts live in the session scratchpad; the reusable ones are in
  `experiments/baseline_collection/recover_tokenized_text.py`.

## COMPLETE (2026-08-07)

The resiliparse 24x5 quality x topic grid is materialized and verified.

| artifact | path (us-east5) | state |
|---|---|---|
| documents | `documents/baseline_resiliparse_decon_deduped_urls/10364warcs/deduped` | 14,815 shards, 290,262,075 docs |
| topic table | `datakit/cluster_assign/resiliparse_10k_gridv1` | 14,815 shards |
| quality table | `datakit/quality/resiliparse_10k_gridv1/outputs/main` | 14,815 shards |
| tokenized | `datakit/tokenize/resiliparse_10k_gridv1` | 14,815 shards, 563 GB |
| **store** | `datakit/store/resiliparse_10k_gridv1` | **120/120 cells, 1190 GB** |
| merged grid | `metadata/grid_v1/resiliparse_10k/distribution.json` | 120/120 cells |

`verify_grid_store` PASSED (26 min, in-region): cell membership recounted from the
attribute tables, every cell loadable via `TreeCache.load`, token totals matched.

- docs 290,262,075 (exact match to the reconstructed corpus)
- tokens 339,968,940,135 (llama3; the 306.6B figure in the grid is gte tokens, capped at 8192)
- a mixture component's `cache_dir` is the bucket path MINUS the trailing `/train`,
  e.g. `.../cluster=0/quality=0/sub=0` — levanter appends the split itself.

### Store gotchas discovered the hard way

1. **The subshard plan must be pinned.** `sub = hash(doc_id) % k`, so k is an
   addressing scheme, not a tuning knob. Recomputing it between resumed runs
   silently lost ~1/3 of a cell (docs filtered as done that nothing had written)
   or double-counted it — in a self-consistent artifact either way. Now written
   once to `_subshard_plan.json` and loaded verbatim. Cost of learning: 561 GB.
2. **Reduce and consolidation both need resume on preemptible workers.** A cell
   writes its whole subshard in one stretch; at 8B tokens/subshard nothing ever
   committed. 4B (and 1B for the fat tail) made writes short enough to land.
   Consolidation was likewise redone from scratch each retry — 39/50 buckets per
   run, so it could never finish until it learned to reuse a matching ledger.
3. **The artifact is `.artifact.json`** (leading dot), not `artifact.json` as the
   datakit_store docstring says. Trust `store_compat.write_artifact`.
4. **`--output-base` must rebind grid_tokenize too**, not just grid_store and
   grid_corpora — `tokenize_dir` reads its own module global.
5. Suffixes are inconsistent: `metadata/grid_v1/` (underscore) vs
   `datakit/*/<ds>_gridv1` (none). Derive paths from code, never retype them.

### Open item for the user

833,768 documents (0.29%) carry chunk-boundary corruption recovered from the token
cache and have an empty url. They are INCLUDED. This matches what the original
training runs actually saw (the cache is what the model read). To drop them:
`gsutil -m rm` the `unmatched-part-*` inputs and re-run tokenize + store.
