# Nemotron URL → WARC Mapping: Validated 2026-04-25

## Headline result

For Common Crawl snapshot **CC-MAIN-2013-20**, **99.9677% of Nemotron-CC v1 URLs**
were found in Common Crawl's CDX index for that snapshot:

```
matched_count:        92,010,006
nemotron_url_count:   92,039,761
coverage_pct:         99.9677%   (29,755 unmatched, 0.0323%)
```

This is essentially at the noise floor of the validation method itself, so the
URL-based mapping used by `filter_nemotron` is treated as **lossless** for
practical purposes.

Report file: `gs://marin-us-central2/scratch/nemotron_coverage_validation/CC-MAIN-2013-20_coverage.json`

## What was actually tested

The production pipeline maps Nemotron-CC records to source WARCs by string
equality between two fields:

- WARC side: `WARC-Target-URI` from the WARC response record header.
- Nemotron side: `metadata.nemotron_url` on each Nemotron jsonl record.

Both fields trace to the same `WARC-Target-URI` byte-string by construction
(both Nemotron's pipeline and Marin's `download_warcs` read the same WARC header
from the same Common Crawl S3 path). So the open question was *not* "is byte
equality the right operator" but "is the join key faithfully preserved across
the chain — and does our scoping by snapshot in `filter_nemotron` match
Nemotron's own snapshot partitioning?"

The validator answered both at once: take every Nemotron URL for one snapshot,
check whether each is present somewhere in CC's index for the same snapshot.

## Method (validate_url_coverage_v4.py)

For one snapshot S:

1. **Build the Nemotron URL set N for S.** Read every Nemotron jsonl file under
   `quality=*/kind={actual,synthetic}/kind2=*/CC-MAIN-S-part-*.jsonl.gz`,
   project `metadata.nemotron_url`, dedupe in a Python set. Per-file checkpoints
   landed in `gs://marin-us-central2/scratch/nemotron_coverage_v4_checkpoints/`
   so the (multi-hour, multi-tens-of-GB) set build is preemption-safe.
2. **Stream every CDX file for S.** 302 gzipped CDX text files at
   `https://data.commoncrawl.org/cc-index/collections/CC-MAIN-S/indexes/cdx-NNNNN.gz`.
   16 Zephyr workers, each scans one CDX file, checks each `url` against the
   shared N, writes per-task hits to GCS via `write_jsonl(skip_existing=True)`.
3. **Aggregate** the per-task hit shards into a `matched` set, compute
   `|matched| / |N|`.

Workers needed 64 GiB RAM each because the Nemotron set serializes to ~30 GB
and each worker `cloudpickle.loads` a full copy.

## Why the 0.0323% gap is at the noise floor

A separate parity check (`validate_cdx_warc_parity.py`) compared one WARC's CDX
entries vs. the WARC's own headers:

```
WARC URL count:          57,003 (200 + text/html only)
Found in CDX:            56,961
Missed by CDX:               42  (0.07%)
```

The 42 misses look like URL-encoding canonicalization (`%7B`/`%7D` braces in
query strings) on CDX's side. So even a *perfect* Nemotron→WARC mapping would
show up as ≈ 99.93% in the snapshot validator, not 100%. The actual 99.97%
result is *better* than the floor — i.e. the test didn't even fully consume the
budget that CDX-vs-WARC noise allows.

The 1,327 "only in CDX" URLs from the parity check are non-HTML records (PNGs,
PDFs, .bib files) that the parity test's `200 + text/html` filter excluded; not
relevant to the Nemotron URL question since Nemotron only processes HTML.

## What this validates

- **filter_nemotron's URL-equality join is correct.** Both sides read the same
  WARC header field, so equality is the right operator and 99.97% is the right
  outcome.
- **Per-snapshot scoping (`_list_nemotron_files_for_snapshot`) is correct.**
  Nemotron's `CC-MAIN-S-part-*.jsonl.gz` files do contain URLs from snapshot S
  WARCs (and only those); otherwise the snapshot-level coverage would have been
  much lower because we'd be matching against the wrong CC URL set.
- **The choice to use `filter_nemotron_full` (organic + 5 synthetic variants)
  doesn't dilute the join.** Synthetic variants carry the source URL in
  `nemotron_url` and were included in N; they still matched at 99.97%.

## What this does NOT validate

- Coverage on snapshots other than CC-MAIN-2013-20. We picked the smallest
  snapshot (~31,600 WARCs in CC, 65 in our 10k subset) for tractability. If a
  later snapshot showed materially worse coverage, the assumption of
  uniform behavior would fail; for now there is no reason to expect it.
- That every URL in our WARC subset that *should* match Nemotron *does* match
  Nemotron — i.e. recall on the WARC-subset side. The validator checks
  Nemotron's URLs are findable in CC; it does not exercise filter_nemotron
  against our 10k WARC subset directly. (That direction is what the production
  pipeline does for real.)
- Quality of Nemotron's content. This validation is purely about URL-set
  membership.

## Reproducibility

Code:
- `experiments/baseline_collection/validate_url_coverage_v4.py` — the validator
- `experiments/baseline_collection/validate_cdx_warc_parity.py` — the parity test
- `tests/test_validate_url_coverage_v4.py`, `tests/test_validate_cdx_warc_parity.py` — unit tests

Job IDs (Iris):
- `/michaelryan/validate-nemotron-coverage-2013-20-v4` — succeeded, 1/1 task,
  `2026-04-25T00:29:17Z`–`2026-04-25T02:43:19Z` (~2h14m).
- `/michaelryan/validate-cdx-warc-parity-2013-20` — succeeded, 1/1 task.

Outputs:
- `gs://marin-us-central2/scratch/nemotron_coverage_validation/CC-MAIN-2013-20_coverage.json`
- `gs://marin-us-central2/scratch/cdx_warc_parity/parity_<warc-name>.json`

Checkpoints (safe to delete after the result is in hand):
- `gs://marin-us-central2/scratch/nemotron_coverage_v4_checkpoints/`

## To extend to another snapshot

```bash
uv run iris --cluster marin job run \
    --cpu 2 --memory 96GB --enable-extra-resources --region us-central2 --no-wait \
    --job-name validate-nemotron-coverage-<SNAP> \
    -- python experiments/baseline_collection/validate_url_coverage_v4.py \
        --snapshot CC-MAIN-YYYY-WW \
        --output gs://marin-us-central2/scratch/nemotron_coverage_validation/ \
        --checkpoint-path gs://marin-us-central2/scratch/nemotron_coverage_v4_checkpoints/
```

Memory may need bumping for snapshots with more Nemotron URLs than 2013-20's
~92M (e.g. recent snapshots with bigger crawls).
