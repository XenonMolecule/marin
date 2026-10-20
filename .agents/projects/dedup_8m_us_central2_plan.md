# 8M fuzzy dedup in us-central2: launch plan

Runs `experiments/fast_curation/rdedup/` (see `reliable_dedup_runbook.md`) over the whole
consolidated corpus once extraction and the last re-fetch are done. Same code and rule as the
2M run (c075 only); what changes is the region, the worker shape and the scale.

## Input

`gs://marin-us-central2/documents/fast_curation/lpv11_fastpipe_v2_1_fused-2a076ddfcb/kept/`,
one parquet per WARC. Expected count **8,582,115**: the 8,529,971-WARC pool (119 CC-MAIN crawls
through 2026-12, six 2026 crawls 2026-17..39, CC-SUPPLEMENTAL-2026-22) plus 52,144 CC-NEWS
WARCs. About 109 TB compressed, ~62B kept docs (4.2x the 2M run's 14.82B).

Gate before `init`: met 2026-10-17. All re-fetches are done, every region's output is in
us-central2 (count 8,582,115, size+crc32c verified), and the source copies are deleted.
Still without data: 5,978 WARCs, of which ~5,800 are empty at Common Crawl (header-only
~450-byte files, mostly 2014-2015 crawls), 101 hang the extractor every time
("abandoned", deterministic), and 53 hit download failures (CC rate limiting). Audit:
`gs://marin-us-central1/.../_reopened/audit-empty-warcs.json`. `--expect-files 8582115` enforces the count.

Provenance survives dedup: output records keep `snapshot` (`CC-MAIN-YYYY-WW`,
`CC-SUPPLEMENTAL-2026-22`, `CC-NEWS-YYYY-MM`), `warc_hash`, `url` and the quality scores, plus
`exact_copies` and `fuzzy_cluster_size`. Checked on the 2M output; any source can be filtered
in or out after dedup.

## How the 2M run went (2000987warcs, sealed 2026-10-02)

- Launcher in us-central2, workers in us-east1 (where the data was).
- 14.82B docs in; 6.87B exact duplicates (46%); 5.71B fuzzy-cluster members in 1.13B
  components, the largest 927M; c075 kept 4.45B docs / 27.33T chars (6.35T tokens).
- pass1 through bucket took a few hours; CC + canonical ~6 h; the c075 tail (text, solve 1,913
  CPU-hours, apply) took ~2.5 days because east1 CPU was held by our own extraction fleet.
- Fix that unblocked it: `--worker-tpu v6e-8 --chunk 1` (each worker a whole TPU host, chips
  idle). For a TPU device `replicas` is the host count of ONE slice, so `--chunk` must be 1.
- Final launcher: `--max-workers 8 --chunk 1 --check-interval 60 --worker-tpu v6e-8
  --worker-cpus 100 --worker-memory-gb 350`.

## Worker shape in us-central2

The only Iris CPU group is 8 vCPU / 16 GB with `max_slices: 6` per zone, too small: c075_solve
budgets 15 GB per task and c075_text 14 GB. Workers run on whole v4-8 hosts (240 CPU / 400 GB):
`--worker-tpu v4-8 --worker-cpus 200 --worker-memory-gb 360 --chunk 1`. The launcher's
preemptible constraint is soft, so workers prefer the preemptible v4 pool and fall back to
reserved v4 (user OK'd both, 2026-10-16). Reserved hosts can be reclaimed for hero runs;
workers are stateless, so a reclaimed task is retried after the 8-minute stale timeout.
Validated 2026-10-16 by a 300-file smoke run (`tmp/ttl=14d/rdedup/smoke300-c2`) on 2 v4-8
workers: every stage sealed; 2,184,320 records in, 35,667 exact duplicates, c075 kept 2,101,480.

## Plan sizing (defaults, `--unit-gb 8`)

13,643 units, 1,393 exact partitions, 12,073 LSH partitions, 2,729 node partitions,
34,824 c075 partitions.

## Cost and time (estimates)

- Compute: TRC TPU hosts, no charge.
- GCS reads: shuffles cost maps x partitions GETs. The c075 shuffle (13.6k x 34.8k ~ 475M),
  LSH (165M) and ~13 CC rounds (~290M) dominate: ~1B class-B GETs, ~$400. This grows with the
  square of the corpus (the 2M run was ~17x cheaper).
- Scratch: pass1 stash ~14 TB, c075 member text ~60 TB at peak, all under `tmp/ttl=14d/`.
  The bucket has 7-day soft delete, so deleted scratch is billed another week: ~$300-450.
- Output: `deduped_c075/` ~30 TB, kept in us-central2.
- Total ~$700-900. Needs the user's explicit OK at launch (cost cap $10).
- Wall time: ~1-1.5 days with ~40 v4-8 hosts; 2-3 days with 15-20. c075_solve is
  memory-bound at ~23 tasks per host.
- Expected result: c075 ~18T tokens (2M-anchored refit gave 16.9T for 7.93M WARCs; the pool is
  now 8% larger plus CC-NEWS, which will dedup heavily).

## Launch commands

```bash
set -a; source .env; set +a
R=gs://marin-us-central2/tmp/ttl=14d/rdedup/8m-v1
O=gs://marin-us-central2/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/8582115warcs
# 1. Freeze the plan (in-region job; lists 8.58M objects, ~15 min).
uv run iris --cluster marin job run --region us-central2 --cpu 8 --memory 16GB \
  --enable-extra-resources --priority batch --max-retries 1 --no-wait --job-name rdedup-init-8m-v1 -- \
  python -m experiments.fast_curation.rdedup.launcher init --run-root $R --output-root $O \
  --spec lpv11_fastpipe_v2_1_fused --bucket gs://marin-us-central2 --expect-files 8582115
# 2. Launcher (control plane) -> v4-8 workers in us-central2.
uv run iris --cluster marin job run --region us-central2 --cpu 1 --memory 8GB \
  --enable-extra-resources --no-preemptible --priority batch --no-wait --job-name rdedup-launch-8m-v1 -- \
  python -m experiments.fast_curation.rdedup.launcher run --run-root $R --region us-central2 \
  --tag 8m1 --max-workers 60 --chunk 1 --check-interval 60 \
  --worker-tpu v4-8 --worker-cpus 200 --worker-memory-gb 360
# 3. Progress.
uv run python -m experiments.fast_curation.rdedup.launcher status --run-root $R
```

The launcher adds one worker job per minute while nothing is pending, so it grows to whatever
the v4 pools grant (up to `--max-workers`). Killing and relaunching it loses nothing.

## After it seals

- Check `stats/rdedup_stats.json`: records_in equals the input row total, conservation passed
  at every seal.
- Token count job (`count_deduped_docs`) into `stats/deduped_c075_token_count.json`.
- Delete the run root (the TTL would do it in 14 days; deleting early saves scratch cost).

## Result (sealed 2026-10-18)

9,951,767,085 docs / 72,712,186,187,505 chars kept from 61,211,321,760 records (exact dups
39,375,554,171; c075 removed 11,884,000,504). Largest component 4,557,202,931 (needed u8
component sizes). Per source (`stats/by_source/summary.json`): CC-MAIN 9.743B docs / 71.76T
chars, CC-NEWS 205.4M / 0.92T, CC-SUPPLEMENTAL 3.76M / 33.4B. Measured llama3 ratio on this
corpus is 4.259 chars/token (32-file smoke), so ~17.07T tokens before decon.

## Next: decon + tokenize (ready to launch)

Same chain as every earlier training set: `decon_apply.py` (CORE-v2 source, 15-grams, drop docs
holding an eval n-gram whose corpus DF <= threshold), then `tokenize_lpv11.py` (llama3, direct
`tokenize()`; the executor-based `tokenize_deduped_extracted.py` no longer imports on this
branch). Smoke 2026-10-18 on 32 output files passed both steps (`tmp/ttl=7d/tok-smoke-8m`).

Cost: decon output ~26 TB (~$530/month), token cache ~1.98 bytes/token -> ~34 TB (~$670/month).

```bash
set -a; source .env; set +a
IN=gs://marin-us-central2/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/8582115warcs/deduped_c075/
DECON=gs://marin-us-central2/documents/baseline_lpv11_fastpipe_v2_1_fused_c075_decon_deduped/8582115warcs/deduped/
CACHE=gs://marin-us-central2/tokenized/lpv11_fastpipe_v2_1_fused_c075_decon_8582115warcs
uv run iris --cluster marin job run --no-wait --region us-central2 --cpu 4 --memory 16GB --disk 20GB \
  --enable-extra-resources --priority batch --max-retries 3 --job-name decon-tokenize-8m \
  -e HF_TOKEN "$HF_TOKEN" -e MARIN_PREFIX gs://marin-us-central2 -e ZEPHYR_MAX_WORKERS 1000 -- \
  bash -c "python experiments/baseline_collection/decon_apply.py --input-path $IN \
    --decon-source gs://marin-us-central2/decontamination/dclm_core_v2/ --output-path $DECON \
    --ngram-length 15 --df-threshold <T> && \
    python -m experiments.fast_curation.tokenize_lpv11 --input-glob '${DECON}data-*.jsonl.gz' --cache-path $CACHE"
```

## Tokenized without decon (2026-10-19)

User chose to tokenize before decontaminating. `tokenize_lpv11.py` wrote
`gs://marin-us-central2/tokenized/lpv11_fastpipe_v2_1_fused_c075_8582115warcs`:
**17,152,574,682,528 llama3 tokens**, 9,951,767,085 docs (matches the dedup exactly), 33.09 TB,
4.239 chars/token. ~18.5 h on ~1,000 Zephyr workers (4,130 shards). A decontaminated version
means running `decon_apply.py` on `deduped_c075/` and tokenizing its output (a Levanter cache
keeps no doc ids, so it cannot be filtered in place).
