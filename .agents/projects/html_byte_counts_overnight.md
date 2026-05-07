# Plain-text HTML byte count over 3000-WARC baseline (overnight 2026-04-26)

## Goal

Compute accurate tokens/byte for the 3000 WARCs by measuring real HTML
payload bytes (UTF-8) and chars across all 156.43 M HTML records (the
older "~177 M" estimate was approximate).

## Job

- **ID**: `/michaelryan/count-html-bytes-3000warc`
- **Cluster**: marin (Iris), region `us-central2` only
- **Resources**: cpu=16, memory=32GB, disk=20GB, batch priority
- **Retries**: 20 (preemption-resilient)
- **Submitted**: 2026-04-26 ~02:02 UTC

## Safety guarantees

- **No cross-region egress**: reads `gs://marin-us-central2/raw/...`,
  writes `gs://marin-us-central2/metadata/...`, runs in `us-central2`.
- **CPU-only**: `--extra cpu`, no GPU/TPU.
- **Easy cleanup**: all output under one prefix
  `gs://marin-us-central2/metadata/baseline_3000_html_byte_counts/`.
  To remove: `gcloud storage rm -r gs://marin-us-central2/metadata/baseline_3000_html_byte_counts/`

## What it does

`experiments/baseline_collection/count_html_bytes.py`:
1. Lists 3000 `data-*.jsonl.gz` files in `baseline_3000-265ff5/`
2. Filters out files already checkpointed
3. ProcessPoolExecutor with 16 workers; each worker streams one
   jsonl.gz and accumulates: `n_records`, `html_chars` (Python str len),
   `html_utf8_bytes` (encoded), `url_utf8_bytes`, `uncompressed_jsonl_bytes`
4. Writes one tiny JSON checkpoint per file: `data-<hash>.json`
5. After all files done, aggregates into `_summary.json`

Per-file checkpointing means preemption loses at most one in-progress file.

## Important context

The "raw" pool is *not* raw `.warc.gz` — it's already de-WARCed JSONL.gz
where each record carries the response-body HTML as a Python string in
`rec["html"]` (download_warcs.py decodes UTF-8 with `errors="replace"`).
So we're measuring the post-HTTP-strip, post-decompress, decoded-text
size. That's what tokenizers see.

## Expected ETA

Each file ~744 MB gzipped, ~3-4 GB uncompressed. ProcessPool of 16 ×
~30 MB/s effective gunzip+JSON parse → ~3000 files × ~25s/file / 16 =
~80 min wall clock if no preemptions. With preemptions, anywhere
2-8 hours.

## Tomorrow morning

Read the summary:

    gcloud storage cat gs://marin-us-central2/metadata/baseline_3000_html_byte_counts/_summary.json

Expected fields: `n_files`, `n_records`, `html_utf8_bytes`,
`html_chars`, `url_utf8_bytes`, `uncompressed_jsonl_bytes`.

Then divide the 3.63T raw-token figure by `html_utf8_bytes` to get the
honest tokens/byte for raw HTML (with tags, decoded, no compression).

## Monitoring commands

    # Status
    uv run iris --config lib/iris/examples/marin.yaml job summary /michaelryan/count-html-bytes-3000warc

    # Logs (tail last 200 lines)
    uv run iris --config lib/iris/examples/marin.yaml job logs /michaelryan/count-html-bytes-3000warc | tail -200

    # Count completed checkpoints
    gcloud storage ls gs://marin-us-central2/metadata/baseline_3000_html_byte_counts/ | wc -l

    # If something goes wrong, kill it
    uv run iris --config lib/iris/examples/marin.yaml job stop /michaelryan/count-html-bytes-3000warc
