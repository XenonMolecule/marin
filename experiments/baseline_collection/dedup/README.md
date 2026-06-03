# Baseline dedup pipeline

Document-level deduplication for the baseline extractions (`llm_curated`,
`resiliparse`). Dedup runs on **extracted text**, not tokens — tokenization is
a separate, downstream step (see [Tokenize the survivors](#4-tokenize-the-survivors)).

## TL;DR

```text
raw extraction (jsonl with `text`)
  └─ prep      add synthetic `id`, project to {id, text, url}
      ├─ exact   xxh3-128 hash over `text`            ── emits sidecar of dup ids
      └─ fuzzy   MinHash-LSH over `text`              ── emits sidecar of dup ids
           └─ deduped (apply)  join prep ⨝ (exact ∪ fuzzy), drop dups → jsonl.gz
                └─ stats        dedup_stats.json + cluster-size histogram
                                   ▼  separate launch, in-region
                          tokenize_deduped.py → Llama-3.1-8B cache
```

There is **no paragraph dedup** — only document-level exact + fuzzy.

## Files

| File | Role |
|---|---|
| `dedup_llm_curated.py` | Driver for `llm_curated` (us-central1). Builds the StepSpec DAG and runs it. |
| `dedup_resiliparse.py` | Driver for `resiliparse` (us-central2). Same DAG, different region/source. |
| `prep_with_id.py` | `prep` step — adds a synthetic `id` and projects to `{id, text, url}`. |
| `apply_dedup.py` | `deduped` step — joins prepped shards with the dup-id sidecars and writes the deduped jsonl tree; also computes the cluster-size histogram. |
| `tokenize_deduped.py` | Downstream tokenization of the deduped trees (run separately). |

The actual exact/fuzzy logic lives in marin, not here:
`marin.processing.classification.deduplication.{exact,fuzzy}`. The drivers just
wire those into a DAG.

## Dedup parameters (both sources)

- **Exact:** `dedup_exact_document` — xxh3-128 hash over the `text` field.
- **Fuzzy:** `dedup_fuzzy_document` — MinHash-LSH + connected components, marin
  defaults: `286` perms, `26` bands, `5`-char n-grams, seed `42`,
  approx Jaccard threshold ≈ `0.75`.

> The fuzzy defaults are **more aggressive** than DCLM's BFF (word-13-gram,
> 0.8 overlap) and Nemotron-CC's typical (260 perms / 20 bands), so dedup rates
> will read slightly higher than those published comparisons.

Dedup keys on the **`text`** column for both sources (the cleaned,
post-processed output that gets tokenized in training — *not* `generated_text`,
which `llm_curated` keeps only for debugging).

## How the sidecar / apply pattern works

`dedup_*_document` **never delete anything**. They emit a sidecar parquet of
duplicate ids at `{dedup_output}/data/<shard>.parquet` with shape
`{id, attributes: {dup_doc: True}}`. The `deduped` step (`apply_dedup`) joins
each prepped input shard against the **union** of the exact and fuzzy sidecars
and writes the surviving rows to a new `jsonl.gz` tree (the original `text`
column is preserved, so the downstream tokenizer reads the right field).

Cluster-size stats come from the fuzzy connected-components final iteration at
`{fuzzy_output}/metadata/cc/it_<N>/*.parquet`.

## Running

Each driver runs the whole DAG via `StepRunner().run(build_steps())`. They are
**region-pinned** — `check_path_in_region` asserts the source bucket is local,
so you must launch on the matching cluster (cross-region reads are blocked by
design):

| Source | Cluster / region | Source path |
|---|---|---|
| `llm_curated` | us-central1 | `gs://marin-us-central1/documents/baseline_llm_curated-050243` |
| `resiliparse` | us-central2 | `gs://marin-us-central2/extracted/baseline_resiliparse-19bdaa` |

> **Ray is retired** — launch with `iris job run`, not `ray_run.py`. (Some
> in-tree docstrings still show the old `ray_run.py` command; ignore those.)

### 1–3. Run dedup (prep → exact/fuzzy → apply → stats)

```bash
# llm_curated (us-central1)
uv run iris --cluster us-central1 job run \
    --region us-central1 \
    --priority batch --no-wait \
    -e WANDB_API_KEY <YOUR_WANDB_API_KEY> \
    -e HF_TOKEN <YOUR_HF_TOKEN> \
    -- python -m experiments.baseline_collection.dedup.dedup_llm_curated

# resiliparse (us-central2)
uv run iris --cluster us-central2 job run \
    --region us-central2 \
    --priority batch --no-wait \
    -e WANDB_API_KEY <YOUR_WANDB_API_KEY> \
    -e HF_TOKEN <YOUR_HF_TOKEN> \
    -- python -m experiments.baseline_collection.dedup.dedup_resiliparse
```

Sizing knobs live inside each driver's `build_steps()` (`max_parallelism` and
the fuzzy `worker_resources` ram). resiliparse uses higher parallelism and 64 GB
fuzzy workers; llm_curated uses 32 GB. Bump these in the driver if you hit OOM
or want more throughput — don't pass them on the command line.

### Output location & the 14-day TTL

The deduped tree is written under a **temp bucket** with a 14-day TTL:
`gs://marin-tmp-us-central{1,2}/ttl=14d/michaelryan/dedup_{llm_curated,resiliparse}/...`.
**It auto-deletes after 14 days.** Either tokenize before then, or copy the
`deduped/` tree to permanent storage first
(`gs://marin-us-central1/documents/baseline_llm_curated_deduped`,
`gs://marin-us-central2/documents/baseline_resiliparse_deduped`).

Check `dedup_stats.json` (written by the `stats` step) for the exact output
paths, dedup rates, and the cluster-size histogram.

### 4. Tokenize the survivors

Separate launch, **in-region** (the temp/permanent deduped trees are regional —
cross-region reads are expensive and blocked elsewhere in the pipeline):

```bash
# llm_curated → cache on us-central1
uv run iris --cluster us-central1 job run --region us-central1 \
    --priority batch --no-wait \
    -e WANDB_API_KEY <YOUR_WANDB_API_KEY> -e HF_TOKEN <YOUR_HF_TOKEN> \
    -- python experiments/baseline_collection/dedup/tokenize_deduped.py --only llm_curated

# resiliparse → cache on us-central2
uv run iris --cluster us-central2 job run --region us-central2 \
    --priority batch --no-wait \
    -e WANDB_API_KEY <YOUR_WANDB_API_KEY> -e HF_TOKEN <YOUR_HF_TOKEN> \
    -- python experiments/baseline_collection/dedup/tokenize_deduped.py --only resiliparse
```

`tokenize_deduped.py` points `default_tokenize` (Llama-3.1-8B, `text_key="text"`)
at `baseline_{llm_curated,resiliparse}_deduped/data-*.jsonl.gz`. The resulting
caches have the same shape as the pre-dedup `baseline_*` caches but with roughly
**26% / 42% fewer tokens** for llm_curated / resiliparse respectively.

## Why dedup before tokenize (and not after)

Fuzzy dedup operates on 5-char n-grams of the raw `text`, and exact dedup hashes
that text. Running it first means near-duplicates collapse on real text rather
than token streams, and the tokenizer never spends compute on documents that are
about to be thrown away. Tokenizing first would be both wrong (wrong granularity
for MinHash) and wasteful.

## Related runbooks

- `.agents/projects/curation_playbook.md` — full `consolidate → dedup →
  tokenize → mirror → register → train` for llm_curated / quality bands.
- `.agents/projects/resiliparse_dedup_handoff.md` — resiliparse per-N runbook.
- `.agents/projects/dedup_observations.md` — notes / observed dedup rates.
