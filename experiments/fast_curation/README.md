# fast_curation — fast multi-stage web-content extraction

A fast classifier+extractor cascade that approximates the slow LLM extractor:

```
(CPU) download WARC → clean-decode → body_strip → fastText useful-filter
      → JustText extract → pre-tokenize          ┐
                                                  ├─ survivor parquet (per WARC)
(TPU) claim survivors → ModernBERT useful-filter ┘ → kept parquet + tombstones
(later) consolidate kept → dedup → decon → tokenize → train (curation benchmark)
```

A doc is **kept** iff it passes fastText (`>= 0.0368`) **and** ModernBERT (`>= 0.1974`).

## Versioning

`spec.py :: SPECS` is the authoritative definition of each `fastpipe_vN`; `VERSIONS.md` is the
human ledger. The `compute_version()[:10]` hash is baked into every output path, so a config
change lands in a fresh namespace and never collides. The `modernbert_threshold` is the one
field NOT in the hash — the TPU phase stores `P(useful)` for every survivor (kept + tombstone),
so re-thresholding ModernBERT is a cheap offline re-filter, not a TPU re-run.

## On-disk contract (under `{bucket}/documents/fast_curation/{spec_id}-{ver}/`)

| path | written by | content |
|---|---|---|
| `cpu_survivors/data-{warc_hash}.parquet` | CPU | one row per fastText survivor (`SURVIVOR_SCHEMA`: doc_id, url, warc_hash, snapshot, fasttext_score, **text**=JustText, **input_ids** ragged int32, n_tokens) |
| `kept/data-{warc_hash}.parquet` | TPU | survivors with `modernbert_prob >= threshold` (+ all survivor fields) |
| `tombstones/data-{warc_hash}.jsonl.gz` | TPU | `{doc_id, modernbert_prob}` for every dropped survivor |
| `timing_cpu/` `timing_tpu/` | both | per-WARC timing JSONs |
| `_phase1_start.json` `_phase1_end.json` | CPU | phase wall-clock sentinels |
| `_claims/data-{h}/_claimed` | TPU | atomic per-WARC claim markers |
| `_xla_cache/` | TPU | shared persistent XLA compile cache |
| `gs://marin-us-central1/{subdir}/_completed/data-{h}` | TPU | central completed-WARC registry (cross-region convention) |

**Token truncation is spec-governed.** `fastpipe_v1` is single-window (`input_ids` truncated to
8192). A future *chunked* spec must store the full untruncated sequence; `n_tokens` records the
true length and `text` is retained so re-tokenization never re-runs the CPU phase.

## Running

Phase 1 (one Iris CPU job; Zephyr fans out the worker fleet):

```bash
python -m experiments.fast_curation.launch_cpu --spec fastpipe_v1 --limit 100   # canary
python -m experiments.fast_curation.launch_cpu --spec fastpipe_v1               # full 10,364
```

Phase 2 (independent TPU workers; coordinate via GCS; can start while Phase 1 runs):

```bash
python -m experiments.fast_curation.launch_tpu --num-workers 1 --priority interactive \
    --no-preemptible --limit 5            # canary
python -m experiments.fast_curation.launch_tpu --num-workers 32 --priority batch   # fleet
```

Timing / progress:

```bash
python -m experiments.fast_curation.timing --spec fastpipe_v1 [--deep]
```

## Multi-region (mirror BEFORE launch — never let the job copy cross-region)

The ModernBERT checkpoint + fastText model must be in the worker's region. To add a region,
do a **one-time manual** copy before launching there (the TPU worker `assert`s the checkpoint
is local and fails fast otherwise):

```bash
gcloud storage cp -r \
  gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-base-10M-c8192/hf \
  gs://marin-<region>/checkpoints/modernbert-useful/mb-clf-base-10M-c8192/hf
gcloud storage cp \
  gs://marin-us-east5/classifiers/useful_fasttext/.../model.bin \
  gs://marin-<region>/classifiers/useful_fasttext/.../model.bin
```

us-east5 is the default region (both models already live there — no copy needed).

## Re-thresholding ModernBERT (free; no TPU re-run)

`kept/` holds survivors with prob ≥ threshold; `tombstones/` holds the dropped ones *with* their
prob. To use a different threshold `t`, take `kept` rows with `modernbert_prob >= t` plus
`tombstone` rows with `modernbert_prob >= t` (those need their `text` rejoined from the survivor
parquet by `doc_id`). Bump the spec's `modernbert_threshold` (it is metadata, not in the hash).

## Downstream → curation benchmark

`dedup.py` consolidates `kept/` into a flat `{text}` tree and runs the general `lib/marin` dedup
engine; then `decon_apply.py` → `tokenize_deduped_extracted.py`; register `fastpipe_v1_10k` in
`scaling_law_sweeps/curation_plan.py` + `launch_10k_natural.py`. See `dedup.py` and the plan.
