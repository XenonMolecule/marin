# URL Index — cross-dataset URL lookup + pairwise coverage

Two indexes over the extraction datasets (dclm, nemotron_full, high_quality,
fineweb, fineweb_edu, resiliparse, llm_pipeline_v1, llm_simple_v1, …) at **both**
scales — `small` (random-300 WARC) and `full` (10,364 WARC):

1. **URL lookup** — type a URL or domain, instantly see what every dataset
   extracted for it, side by side. Sub-100 ms, local.
2. **Pairwise coverage** — |A∩B|, A-only, B-only, Jaccard, containment (is
   `high_quality` ⊆ `llm_pipeline_v1`? is `fineweb_edu` disjoint from
   `nemotron_full`?) as an N×N matrix.

It reuses the infinigram registry (`experiments/infinigram/targets.py`) as the
single source of truth for where each dataset lives, and its text-hash provenance
recovery for the LLM tiers that dropped `url`.

## Artifacts (one dir per dataset, in the dataset's own region)

`gs://marin-{region}/url_index/{collection}/{dataset}/`

| file | columns | purpose |
|------|---------|---------|
| `keys.parquet` | `dataset, rid_h, text_h, dom_h` (UINT64) | coverage set-math |
| `meta.parquet` | `dataset, url_key, domain, warc_record_id, snapshot, warc_file, text_len` | portable lookup routing (no text) |
| `text.parquet` | meta + `text`, sorted `(domain, url_key)` | in-region text store for `--with-text` |
| `stats.json` | doc/url-match counts | provenance completeness |

**Identity keys.** Coverage defaults to `rid_h` = `xxh3_64` of the normalized
`warc_record_id` — the *source page*, the right question across different
extractors (their text differs). `text_h` (identical extracted text) is right for
tiers of the *same* pipeline. `dom_h` is domain-level coverage.

## 1. Build (in-region Iris jobs)

One region-pinned CPU job per (dataset, collection). `--only-landed` ships
everything currently available at both scales; not-yet-landed tiers just fail
their own job until the data lands.

```bash
iris job run --cluster marin --region us-central1 -- \
    python -m experiments.url_index.launch_build_iris --only-landed
```

`--dry-run` first to see the target list. Build one target directly (inside an
in-region job) with `python -m experiments.url_index.build --dataset X --collection small`.

**Standalone build jobs** (what was actually used to ship) — one `iris job run`
per (dataset, collection), sized per dataset, on the larger **preemptible** CPU
pool (the non-preemptible pool is only e2-highmem-2):

```bash
iris --cluster marin job run --region us-central1 --preemptible --extra cpu \
    --cpu 8 --memory 32GB --disk 64GB --max-retries 3 --enable-extra-resources -- \
    python -m experiments.url_index.build --dataset llm_pipeline_v1 --collection small
```

**Huge raw tiers** (e.g. `resiliparse`, ~500M docs) use `--keys-only`: only
`keys.parquet` is written (coverage), skipping the multi-TB text store. The text
is dropped right after its hash is computed, so the staging parquet stays small.

**All-in-one** (`orchestrate.py`): a single in-cluster job that submits every
build child, waits, then runs the analysis below and uploads results — handy once
the cluster's per-VM resource limits are known.

## 2. Consolidate the lookup index (local, one per collection)

Reads only the compact `meta.parquet` (megabytes, no text) from both regions:

```bash
python -m experiments.url_index.consolidate \
    --meta 'gs://marin-us-central1/url_index/small/*/meta.parquet' \
           'gs://marin-us-central2/url_index/small/*/meta.parquet' \
    --collection small --out url_lookup_small.duckdb
# ...and again with full/ for the 10k index.
```

## 3. Look up URLs

```bash
# exact URL across all datasets
python -m experiments.url_index.lookup --db url_lookup_small.duckdb "www.example.com/page"
# every page of a domain
python -m experiments.url_index.lookup --db url_lookup_small.duckdb --domain example.com
# with the actual extracted text side by side
python -m experiments.url_index.lookup --db url_lookup_small.duckdb --with-text "example.com/page"
# a batch from a file
python -m experiments.url_index.lookup --db url_lookup_small.duckdb --queries-file urls.txt
```

`--with-text` reads the extracted text from the in-region `text.parquet`. At
300-WARC scale the text stores are small (~1–2 GB total); copy them next to the
db and pass `--text-dir <dir>` (layout `<dir>/{collection}/{dataset}/text.parquet`)
for fully-offline text. At 10k scale leave them in-region and run `--with-text`
from an in-region VM.

## 4. Pairwise coverage

```bash
python -m experiments.url_index.coverage \
    --keys 'gs://marin-us-central1/url_index/small/*/keys.parquet' \
           'gs://marin-us-central2/url_index/small/*/keys.parquet' \
    --key rid_h --out-dir coverage_small
```

Writes `coverage_pairs_{key}.csv` (every ordered pair), `containment_matrix_{key}.csv`
(cell `a\b` = fraction of A contained in B), and a Jaccard heatmap PNG (if
matplotlib is present). Re-run with `--key text_h` to compare same-pipeline tiers.
Check each LLM tier's `stats.json` `url_match_rate` — a low `rid_h` overlap can be
join-loss, not true disjointness.

**One-shot analysis** (`analyze.py`) does consolidate + coverage (both keys) +
upload + logs the matrices, over the datasets that built. Run it locally for
`small` (tiny keys) or as an in-region Iris job for `full` (keeps the large
`resiliparse` keys read in-region):

```bash
python -m experiments.url_index.analyze --collection full \
    --datasets dclm high_quality nemotron_full resiliparse \
    --out-prefix gs://marin-us-central1/url_index/analysis/full
```

## Current data availability (2026-07-23)

The canonical document tiers are asymmetric right now, so the shipped indexes are:
- **small (random-300):** `llm_pipeline_v1` (two-stage) and `llm_simple_v1`
  (one-call) — a real pipeline-vs-pipeline comparison. Both use the deduped
  (non-decon) `300warcs` tier. The 300-warc *baseline* trees (dclm/nemotron/HQ/
  fineweb) aren't materialized as documents yet (only tokenized caches / 10k).
- **full (10,364):** `dclm`, `high_quality`, `nemotron_full`, and `resiliparse`
  (keys-only). `fineweb_edu`/`llm_pipeline_v1` full tiers weren't landed yet.

As the missing tiers land, re-run their build + re-analyze — nothing else changes.

## Adding a new dataset

The registry is `experiments/infinigram/targets.py` (shared with infinigram).
**Add one `DatasetSpec`** and both tools pick it up automatically:

```python
DatasetSpec(
    dataset="my_new_pipeline",
    region="us-central1",              # region its tier lives in
    full=IndexSource.at(".../10364warcs/deduped/data-*.jsonl.gz", landed=False),
    small=IndexSource.at(".../300warcs/deduped/data-*.jsonl.gz", landed=False),
    # Only if the tier is text-only (url dropped by dedup) — recovered by text hash:
    provenance_globs=_llm_provenance_globs("my_new_pipeline"),
),
```

- Use `IndexSource.under(".../foo-*/...")` instead of `.at(...)` when the path has
  a confighash segment to resolve at run time.
- Omit `provenance_globs` for baseline tiers that carry `url`/`warc_record_id`
  inline.
- Then `launch_build_iris --datasets my_new_pipeline` (or `--only-landed` once it
  lands), re-consolidate, and re-run coverage. Nothing else changes.

## Tests

```bash
uv run pytest experiments/url_index/tests/test_url_index.py
```

Correctness runs the real build path on synthetic shards through
consolidate→lookup→coverage and checks against brute-force Python set ops;
efficiency asserts point-lookup < 50 ms, batch-of-10 < 300 ms, coverage matrix
< 5 s.

## Design notes

- **Sub-second, at scale.** Lookup keys are indexed (`url_key`, `domain`) in a
  local text-free DuckDB (~0.5–1 GB even at 10k). Coverage is exact DuckDB set
  math over UINT64 hash columns — a single self-join, seconds at tens of millions
  of keys. No bloom filters (approximate) or extra deps needed.
- **No cross-region reads.** Every build runs in the dataset's own region
  (`resolve_target` refuses otherwise); only the compact `keys`/`meta` artifacts
  and the local index cross regions. Text stays in-region unless you deliberately
  download it.
- **URL normalization** (`keys.py`): scheme + fragment dropped, host lowercased and
  de-`www`'d, trailing slash trimmed, query kept verbatim; `domain` = registrable
  domain (eTLD+1) via the vendored `public_suffix_list.dat` (dated snapshot —
  refresh occasionally from https://publicsuffix.org/list/).
