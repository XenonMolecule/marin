# llm_pipeline_v1_1 — 24 x 5 token projection to the 10,364-WARC pool

**Date:** 2026-07-31 · **Status:** DONE — stopped at 6.5% shard coverage, estimate converged
**Artifact:** https://claude.ai/code/artifact/b6f8d29e-270b-443f-8959-13202d0cd3ae

A 10%-progress check on the in-flight 3,005-WARC `llm_pipeline_v1_1` extraction:
score a random WARC sample with the same WebOrganizer topic classifier and
calibrated quality scorer used for grid_v1, then project each of the 120
(topic, quality) cells up to the full pool.

## Headline

**137.8B llama3 tokens, pre-dedup, at 10,364 WARCs**, across ~97M documents. 77%
of that mass sits in the middle quality bucket q2; the top bucket q4 holds only
~0.2%.

Final measurement: 124,197 of a projected 1,905,829 batch shards (6.5%), spanning
966 of the 1,000 sampled WARCs. The run was stopped there deliberately — the
estimate had converged and the jobs were holding interactive TPU capacity away
from the extraction fleet.

**Convergence, as coverage tripled:**

| shard coverage | shard-weighted | WARC-weighted cross-check |
|---:|---:|---:|
| 2.0% | 138.3B | 115.2B (40 WARCs) |
| 4.2% | 136.8B | 115.2B (86 WARCs) |
| 5.5% | 137.3B | 118.3B (126 WARCs) |
| 6.5% | **137.8B** | 124.3B (219 WARCs) |

The shard-weighted number is flat to within 1%. The WARC-weighted one climbs
steadily toward it, which is the predicted signature of its size bias unwinding:
each additional multi-region WARC that completes is a *larger* one, and adding
large WARCs pulls the WARC-average up. That the two converge from opposite
directions is the evidence that the batch-sampling assumption holds.

## What made this non-trivial

The run is **not consolidated**, so there is no corpus directory to point a
labeller at. Its output is `data-{hash}/batch_NNNN.jsonl.gz` spread over five
regional buckets, and steal mode fragments it further. Facts established while
building the sample, all worth reusing:

| | |
|---|---|
| WARCs in `_completed` | 2,971 of 3,005 |
| Eligible after a batch-contiguity check | 2,965 |
| Unique batch shards | 545,232 → **183.9 shards/WARC** |
| Surviving docs per batch | **48.3** (from 250 inputs — ~19% keep) |
| Chars per doc | ~5,700–7,700 |
| Chars per **llama3** token | **4.31** |
| **(warc, batch) keys present in >1 region** | **4.5%** — dedupe or double-count |
| Regions a single WARC's batches span | **3.3 on average** |

## The estimator, and why it is shard-weighted

The obvious estimator — average over WARCs, scale by 10,364/N — is **biased
here**. A WARC only counts once every region holding a piece of it has reported,
and the WARCs that finish first are the least-fragmented, which are the
*smaller* ones. Measured on identical tallies: WARC-weighted read **115.2B**
against shard-weighted **136.8B**.

So the projection is

    projected_cell = (tokens in cell / shards measured) x total shards in the finished run

with `total shards` measured off the inventory (183.9 x 10,364 = 1,905,829)
rather than assumed. A batch is a contiguous slice of one WARC's records and
which batches were stolen where is unrelated to their content, so the tallied
batches are an unbiased sample. The WARC-weighted number is still printed as a
cross-check — the two converging is the evidence that assumption holds.

Interval is a bootstrap resampling **whole tallies** (documents inside a WARC
are correlated, so a per-document interval would be far too tight), with a
finite-population correction for the fraction of shards measured.

Tokens are **llama3** tokens: each cell's exact character count converted by a
chars-per-token ratio measured on that cell's own systematic 1-in-200
llama3-tokenized subsample, backing off to the topic row then global. The topic
model's own `gte_tokens` are capped at its 8192 context and would undercount the
long tail — they are recorded but not used for the projection.

## Caveats to state when presenting

1. **Pre-dedup.** This is raw extraction output. Dedup + decontamination will cut
   it, and near-duplicate rates *rise* with corpus size, so 137B is an upper
   bound on trainable tokens.
2. **Linear in WARCs.** Nothing here models the sublinearity that dedup adds.
3. Coverage was 4.2% of shards at the time of the headline number; the estimate
   moved only 138.3B → 136.8B between 2.0% and 4.2% coverage, so it is stable.

## Cross-corpus comparison

`grid_compare.py` merges the exact `metadata/grid_v1/{corpus}/distribution.json`
tables with this projection into one dataset; `grid_compare_page.py` renders a
three-tab explorer (quality buckets / topics / full 24x5).
**Tool:** https://claude.ai/code/artifact/a5642742-725a-4831-8d43-e555ccf3896e

Comparison is in **gte tokens** — grid_v1 only ever stored the topic tokenizer's
length, so that is the one unit measured identically everywhere. It is capped at
8192 and is not a training-token count. lpv1_1 reads 121.5B gte against its 137B
llama3, i.e. llama3 runs ~13% higher; expect a similar gap for the others.

Token share by bucket (q0/q1/q2/q3/q4):

| corpus | gte tok | q0 | q1 | q2 | q3 | q4 |
|---|---:|---:|---:|---:|---:|---:|
| llm_pipeline_v1_1 | 121.5B | 0.68% | 6.66% | 77.49% | 15.01% | 0.16% |
| fineweb_cc | 28.3B | 0.34% | 6.74% | 66.39% | 26.39% | 0.14% |
| high_quality | 22.5B | 0.10% | 1.59% | 63.86% | 34.11% | 0.34% |
| nemotron | 10.2B | 0.35% | 7.53% | 69.85% | 21.84% | 0.43% |
| dclm | 6.9B | 0.12% | 3.29% | 69.58% | 26.81% | 0.21% |
| fineweb_edu | 2.3B | 0.01% | 1.00% | 39.53% | 58.91% | 0.55% |

**The size/rate trade is the whole story.** lpv1_1 has the *worst* quality rate
of the six (highest q0, lowest q3 share) but is so much larger that it still
carries the most absolute q3+q4 mass — 18.4B gte tokens against high_quality's
7.7B. fineweb_edu is the inverse: the best rate by far (58.9% q3) on only 1.4B
q3+q4 tokens. q4 is scarce everywhere, peaking at 0.55% of tokens.

Caveat on that comparison: lpv1_1 is pre-dedup and projected, high_quality is
post-dedup+decon, so lpv1_1's absolute lead will shrink once it is deduped.

## Code

```
experiments/baseline_collection/
  grid_projection_manifest.py   # cached region listings -> per-region group manifests
  grid_projection.py            # score a (warc, region) group -> one tally JSON
  launch_grid_projection.py     # per-region Iris fan-out (data-local, no egress)
  grid_projection_report.py     # tallies -> 24x5 grid + projection + bootstrap
  grid_projection_page.py       # projection.json -> standalone HTML
  grid_compare.py               # + grid_v1 distributions -> one comparison dataset
  grid_compare_page.py          # compare.json -> three-tab explorer
```

Rebuild/refresh:

```bash
gcloud storage rsync gs://marin-us-central1/metadata/lpv1_1_projection/tallies/ tallies/
python -m experiments.baseline_collection.grid_projection_report \
    --tallies tallies --sample sample.json --total-shards 1905829 --out report
python -m experiments.baseline_collection.grid_projection_page \
    --projection report/projection.json --out report/index.html
```

## Operational notes

* Jobs run **in the region their data lives in** and write only kilobyte tallies
  to us-central1, so no batch data crosses a region boundary. Total egress for
  the whole exercise was a few megabytes.
* Throughput is **I/O-bound, not TPU-bound**. The first wave read shard files
  serially and held a v5p-8 at ~1% of its FLOPs (~11 docs/s). Adding a 32-way
  threaded prefetch in `iter_buffered` is what made it viable — reach for that
  before adding jobs.
* `iris job run` submits at roughly one job per minute because each opens its own
  SSH tunnel. Launch regions in parallel processes rather than a sequential loop,
  but keep total concurrency low: three launchers at `SUBMIT_PARALLELISM=16` put
  ~26 tunnels on the one controller and **every** submission blew its 900 s
  timeout. The constant is now 4.
* **A region's preferred pool can be entirely unavailable.** us-east5's `v5p-8`
  requests sat PENDING for two hours (that pool is where the curation training
  sweeps live) and contributed nothing, while its `v6e` sat free. The launcher
  now takes `--tpu` to reroute. Moving us-east5 to `v6e-4` took it from 0 to 222
  groups in ~15 minutes.
* Per-region results are consistent to a fraction of a point (q0 3.55–3.63%,
  chars/doc 5967–6168 across five regions, two TPU families), so switching
  accelerator did not perturb labels — and region is independent of content,
  which is the assumption the shard-weighted estimator needs.
* Fixed in `launch_grid_projection.py`: a submission that hits its timeout used
  to raise out of `ThreadPoolExecutor.map` and take the whole wave down with it
  (eu-west4 lost 2 of 12 jobs that way).
