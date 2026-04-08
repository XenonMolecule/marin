# Baseline Dataset Collection Pipeline

Compare data curation approaches on identical Common Crawl input by extracting
the Nemotron-CC, DCLM, and FineWeb-Edu outputs for the same set of WARC files.

## Problem

We want to evaluate a new data curation pipeline against three baselines
(Nemotron-CC, DCLM, FineWeb-Edu) on the **exact same** Common Crawl input
documents. Each baseline publishes its outputs but uses different metadata
formats, so we need a pipeline that:

1. Selects a set of WARC files known to be in all three baselines
2. Downloads those WARCs (needed for our own pipeline anyway)
3. Extracts matching records from each baseline dataset
4. Produces a unified per-document comparison dataset

The baselines join on different keys:

| Dataset       | Join key to WARC                | Location on GCS                                           |
|---------------|---------------------------------|-----------------------------------------------------------|
| DCLM          | `WARC-Record-ID` in `metadata`  | Not yet downloaded (parquet on HuggingFace)               |
| FineWeb-Edu   | `file_path` (WARC S3 path) + `url` | `gs://marin-us-central2/raw/fineweb-edu/`             |
| Nemotron-CC v1| `nemotron_url` only (no WARC ID)| `gs://marin-us-central2/raw/nemotro-cc-eeb783/`          |

**Key constraint**: Nemotron-CC v1 has URLs but no WARC record IDs or WARC
filenames, so we must extract URLs from the raw WARCs to join against it.

## Goals

- Produce a reproducible, extensible pipeline where adding 10 new WARC files
  doesn't reprocess the existing 3000
- Extract per-WARC metadata (record IDs, URLs, WARC filenames) as a reusable
  intermediate artifact
- Filter each baseline dataset to just the documents from our selected WARCs
- Preserve baseline-specific metadata (Nemotron quality tier, FineWeb-Edu score,
  DCLM fastText score) for downstream analysis

**Non-goals**: Reproducing the Nemotron/DCLM/FineWeb pipelines from scratch.
We use their published outputs directly.

## Proposed Solution

### Directory structure

```
experiments/baseline_collection/
    warc_file_lists/           # Input WARC file lists (text, one path per line)
        dclm_fineweb_overlap.txt   # Output of find_overlapping_warcs.py
    pipeline.py                # Main pipeline definition (ExecutorStep DAG)
    extract_warc_metadata.py   # Step 1: WARC -> {record_id, url, warc_file} JSONL
    filter_nemotron.py         # Step 2a: Join metadata against Nemotron-CC v1
    filter_dclm.py             # Step 2b: Join metadata against DCLM
    filter_fineweb_edu.py      # Step 2c: Join metadata against FineWeb-Edu
```

### Pipeline DAG

```
warc_file_list.txt (input, first 3000 lines of dclm_400m_1x_warcs.txt)
        |
        v
[download_warcs]  ──────────────────────────> extracted HTML (~2.2 TB for 3000 WARCs)
        |                                      (for user's own pipeline)
        v
[extract_warc_metadata]  ─────────────────> metadata.jsonl per WARC shard
    {record_id, url, warc_file}               (small, ~150 MB total)
        |
        ├──> [filter_nemotron]   ──> nemotron_subset.jsonl.gz
        |       (join on url)         {text, url, nemotron_quality, nemotron_kind}
        |       |
        |       └──> [tokenize_nemotron]  ──> tokenized (Levanter cache)
        |
        ├──> [filter_dclm]       ──> dclm_subset.jsonl.gz
        |       (join on record_id)   {text, url, dclm_fasttext_score, ...}
        |       |
        |       └──> [tokenize_dclm]      ──> tokenized (Levanter cache)
        |
        └──> [filter_fineweb]    ──> fineweb_edu_subset.jsonl.gz
                (join on warc_file    {text, url, score, int_score, ...}
                 then url)
                |
                └──> [tokenize_fineweb]   ──> tokenized (Levanter cache)
```

All tokenization uses `meta-llama/Meta-Llama-3.1-8B` (the standard Marin
tokenizer, compatible with Delphi training runs). Token counts per baseline
are a key output metric — they reflect each pipeline's data retention rate.

### Core design: per-WARC-file granularity

Each WARC file is its own unit of work. This makes the pipeline:

- **Incremental**: Adding 10 WARCs doesn't reprocess the existing 3000.
  Each WARC produces its own metadata shard; the downstream joins just
  read more shards.
- **Embarrassingly parallel**: Zephyr flat_map over the WARC list.
- **Resumable**: Zephyr's `skip_existing=True` on write_jsonl handles restarts.

### Step 1: extract_warc_metadata

For each WARC file, parse all HTML response records and emit:

```python
{
    "warc_record_id": "d02dd231-b2ab-4a01-80e8-...",  # stripped from <urn:uuid:...>
    "url": "http://example.com/page",                  # WARC-Target-URI
    "warc_file": "crawl-data/CC-MAIN-2022-49/...",     # normalized path
    "snapshot": "CC-MAIN-2022-49",                      # extracted from path
}
```

This is ~50KB per WARC file (25K records × ~100 bytes each, compressed).
For 3000 WARCs: ~150MB total. Tiny artifact, hugely valuable — it's the
universal join key for all three baselines.

**Implementation**: Zephyr flat_map over WARC paths. Each worker downloads
one WARC, scans records, emits metadata rows. Same pattern as download_warc.py
but only extracting headers, not HTML content.

### Step 2a: filter_nemotron (join on URL)

Nemotron-CC v1 is organized as:
```
gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/
    data-jsonl/quality={high,medium-high,...}/kind={actual,synthetic}/
        kind2=actual/CC-MAIN-YYYY-WW-part-NNNNN.jsonl.gz
```

Each record has `{id, text, metadata: {nemotron_url, nemotron_language}}`.

**Join strategy**: For each snapshot in our WARC list, read the corresponding
Nemotron partition files. Build a URL set from our metadata, then filter
Nemotron records where `nemotron_url` is in the set.

**Output per record**:
```python
{
    "url": "...",
    "text": "...",
    "nemotron_quality": "high",      # from directory structure
    "nemotron_kind": "actual",       # actual vs synthetic
    "nemotron_id": "...",
}
```

### Step 2b: filter_dclm (join on WARC-Record-ID)

DCLM-baseline has full WARC headers in its `metadata` dict including
`WARC-Record-ID`. The join is exact and unambiguous — record IDs are
globally unique UUIDs.

**Complication**: DCLM-baseline-1.0 is **not partitioned by snapshot**.
It's 10 global shards, mixed across all crawls. There's no way to read
"just CC-MAIN-2022-49" without scanning everything. The `warcinfo` field
has `isPartOf` (snapshot) but NOT the WARC filename, so we can narrow to
snapshot but not to specific files.

**Approach**: Full scan with hash-set filter.

1. Load our ~75M extracted record IDs into a set (1.2GB as packed UUIDs)
2. Scan all DCLM-baseline parquet files, checking each record's
   `metadata.WARC-Record-ID` against the set
3. Emit matching records with their quality scores
4. Parallelize across DCLM's ~100 shard files via Zephyr

This is a one-time ~4TB scan but it's embarrassingly parallel and the
hash lookup is O(1). On 100 workers reading in parallel, this should
take ~30 minutes.

**Prerequisite**: DCLM-baseline-1.0 must be downloaded to GCS first
(one-time, 4TB). Check if it's already on the cluster.

**Future optimization**: Build a record_id index (just the UUID column +
shard pointer) once, then all future joins are cheap lookups.

### Step 2c: filter_fineweb_edu (join on file_path + url)

FineWeb-Edu has `file_path` (WARC S3 path) and `url` columns, partitioned
by `dump` (snapshot). Two-stage filter:

1. Filter parquet where `file_path` matches our WARC files (fast, columnar)
2. Within those, keep all records (they're from our WARCs)

**Output per record**:
```python
{
    "url": "...",
    "text": "...",
    "fineweb_score": 3.5,
    "fineweb_int_score": 3,
    "dump": "CC-MAIN-2022-49",
}
```

## Implementation Outline

1. Create `experiments/baseline_collection/` directory with pipeline.py
2. Implement `download_warcs` step — reuse existing `download_and_extract_warcs`
   from `marin.datakit.download.commoncrawl.download_warc`. Input: first 3000
   lines of `dclm_400m_1x_warcs.txt`. Output: ~2.2 TB extracted HTML on
   `gs://marin-us-central2/`.
3. Implement `extract_warc_metadata` — Zephyr flat_map over WARC list,
   emit {record_id, url, warc_file, snapshot} per HTML record. One output
   shard per WARC file for incrementality.
4. Implement `filter_nemotron` — Read metadata shards, build URL set per
   snapshot, scan matching Nemotron partitions, filter and emit
5. Implement `filter_fineweb_edu` — Read metadata shards, build warc_file
   set per snapshot, read FineWeb-Edu parquet with file_path filter
6. Implement `filter_dclm` — Full scan of 27,838 DCLM shards with hash-set
   filter on record_id
7. Tokenize each baseline subset with `meta-llama/Meta-Llama-3.1-8B` using
   the standard `default_tokenize` from `experiments/defaults.py`. Token
   counts are a key output metric (data retention rate per pipeline).
8. Wire all into pipeline.py as ExecutorStep DAG, test on 10 WARCs first

## Notes

- **The metadata step is the linchpin.** It costs almost nothing (header
  parsing only, no HTML extraction) but enables all three joins. It should
  be a separate ExecutorStep so it's cached and reusable.
- **URL normalization matters for Nemotron join.** Nemotron stores URLs as-is
  from the WARC. We should normalize both sides (lowercase scheme/host,
  strip trailing slash, sort query params) to maximize match rate.
- **DCLM will need its own download step.** DCLM-baseline-1.0 is not yet on
  our cluster. We may want to download only the snapshots we need.
- **Synthetic Nemotron data is excluded for now.** It has no URL or WARC ID.
  Can be added later by matching against the organic data's URLs.
- **Per-WARC sharding in the metadata step** means adding 10 WARCs later
  just appends 10 new shards — no reprocessing of existing 3000.

## Future Work

- Extend to Nemotron-CC v2.1 (has `warc_record_id` but covers 2025 snapshots)
- Add synthetic data matching (Nemotron synthetic → organic → WARC)
- Build unified comparison table with all four pipelines' quality scores
  per document
- Training ablations on subsets selected by each pipeline
