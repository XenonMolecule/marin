# Dedup pipeline

The general document-level dedup pipeline for baseline extractions. One script,
`experiments/baseline_collection/dedup_extracted.py`, deduplicates the first-N
WARCs of **any** extraction spec — you pass the spec name with `--spec`, it runs
the whole DAG in a single shot. Dedup operates on **extracted text**;
tokenization is a separate downstream step.

## Why fuzzy dedup

LLM extraction is **stochastic** — two extractions of "the same page" come out as
different strings, so exact-match dedup catches almost nothing. The MinHash-LSH
fuzzy pass (~0.75 Jaccard over 5-char n-grams) is what actually collapses
near-duplicate pages into one canonical record. The fuzzy pass is the point of
this pipeline; don't skip it.

Dedup is **N-dependent**: deduping the first-N WARCs only collapses duplicates
*within* those N. The N=100 deduped tree is not a prefix of the N=500 tree —
each (spec, N) is its own run and its own output dir.

## Pipeline stages

`build_steps()` wires six steps into a `StepRunner` DAG. Each writes to a fixed,
predictable path under the output bucket (no content-hashed cache dir — rerunning
only changes with `(spec, n)`, which already changes the bucket).

| Step | What it does |
|---|---|
| **reshape** | Read the first-N WARC hashes from the manifest, pull those done batches from the consolidated archive, emit a flat 200-shard `data-XXXXX-of-00200.jsonl.gz` tree (text-only schema). |
| **normalize** | `datakit normalize_to_parquet`: jsonl.gz → `NormalizedData` parquet with `xxh3_128` ids and `DedupMode.EXACT` (so exact-duplicate docs are collapsed here for free). Default 64 MB target partitions (vs marin's 256 MB) → ~4× more parquet shards, which gives the downstream MinHash + fuzzy stages ~4× more parallelism width. Pure parallelism knob (`--target-partition-bytes`); outputs are identical regardless. |
| **minhash** | Per-shard MinHash bucket attrs: 286 perms, 26 bands, 5-char n-grams, seed 42. Co-partitioned 1:1 with normalize output. |
| **fuzzy** | Global LSH + connected-components across all minhash shards → per-doc cluster markers `{id, attributes: {dup_cluster_id, is_cluster_canonical}}`. Exactly one row per cluster has `is_cluster_canonical=True`; singletons get no row. `cc_resume=True` resumes from the last complete CC iteration on disk (survives mid-run preemption). |
| **deduped** | Apply step: join the normalize parquet with the fuzzy attrs, drop `is_cluster_canonical=False`, keep canonicals + singletons. Writes deduped `jsonl.gz` (the `text` field) — this is what the tokenizer reads. |
| **stats** | `dedup_stats.json` with every step's output path + the fuzzy params. |

## Parameters

Fuzzy / MinHash (marin defaults): **286** perms, **26** bands, **5**-char
n-grams, seed **42**, approx Jaccard threshold ≈ **0.75**, on the **`text`**
field. There is **no paragraph dedup** — document level only.

> These defaults are more aggressive than DCLM's BFF (word-13-gram, 0.8 overlap)
> and Nemotron-CC's typical (260 perms / 20 bands), so dedup rates read slightly
> higher than those published comparisons.

## Output layout (permanent, regional)

```text
gs://marin-{region}/documents/baseline_{spec}_deduped/{n}warcs/
    reshape/data-XXXXX-of-00200.jsonl.gz
    normalize/outputs/main/part-*.parquet
    minhash/outputs/<basename>.parquet
    fuzzy/outputs/source_000/<basename>.parquet
    deduped/data-XXXXX-of-YYYYY.jsonl.gz   <-- tokenize reads here
    stats/dedup_stats.json
```

The bucket is derived from `--region`; outputs are permanent (no TTL).

## Running

`dedup_extracted.py` is region-portable: it reads raw input AND writes outputs
in the same region (no cross-region reads). The raw extraction for the spec must
already be present in that region. Supported regions: `us-central1` (default),
`us-east5`, `us-west4`.

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
    --cpu 4 --memory 16GB --disk 20GB \
    --priority interactive --extra cpu --enable-extra-resources \
    --region us-central1 \
    --job-name dedup-<spec>-<N>warcs \
    -e WANDB_API_KEY <YOUR_WANDB_API_KEY> -e HF_TOKEN <YOUR_HF_TOKEN> \
    -- python experiments/baseline_collection/dedup_extracted.py \
       --spec <spec> --n <N> [--region us-central1]
```

Flags:

- `--spec` *(required)* — the extraction spec name. The output bucket and
  tokenizer entry are both keyed on it.
- `--n` *(required)* — number of priority (first-N) WARCs.
- `--manifest` — WARC manifest defining the first-N ordering
  (default `experiments/distill/baseline_warcs_3000.txt`).
- `--region` — region whose `marin-*` bucket holds raw input and receives the
  deduped outputs (default `us-central1`).
- `--target-partition-bytes` — normalize parquet partition size; controls
  downstream parallelism width (default 64 MB).

Sizing for the heavy fuzzy stage lives in `build_steps()` (`max_parallelism`,
64 GB preemptible workers, `cc_resume`); bump there if you hit OOM rather than on
the command line.

## Tokenize the survivors

Separate launch, in-region (the deduped tree is regional):

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
    --cpu 4 --memory 16GB --disk 20GB \
    --priority interactive --extra cpu --enable-extra-resources \
    --region us-central1 --job-name tokenize-<spec>-<N>warcs \
    -e WANDB_API_KEY <YOUR_WANDB_API_KEY> -e HF_TOKEN <YOUR_HF_TOKEN> \
    -- python experiments/baseline_collection/tokenize_deduped_extracted.py \
       --spec <spec> --n <N>
```

This reads `…/baseline_{spec}_deduped/{n}warcs/deduped/data-*.jsonl.gz` and
writes a Llama-3.1-8B cache to
`gs://marin-{region}/tokenized/{spec}_{n}warcs-<cache_hash>/`. The canonical
`{spec}_{n}warcs` name is the entry to paste into `_D_OBS_DEFAULTS`.

## Why dedup before tokenize

Fuzzy dedup works on 5-char n-grams of raw `text`, and the normalize step's exact
pass hashes that text. Running dedup first collapses near-duplicates on real text
rather than token streams, and the tokenizer never spends compute on documents
that are about to be dropped. Tokenizing first would be both wrong (wrong
granularity for MinHash) and wasteful.

> **Ray is retired** — launch everything with `iris job run`, not `ray_run.py`.

## Related runbooks

- `.agents/projects/curation_playbook.md` — full `consolidate → dedup →
  tokenize → mirror → register → train` walkthrough.
- `.agents/projects/dedup_observations.md` — notes / observed dedup rates.
