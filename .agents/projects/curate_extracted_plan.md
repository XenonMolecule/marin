# curate_extracted: manifest-first dedup + tokenize for LLM-extracted specs

**Status**: planned 2026-05-10
**Goal**: For each (spec, N), produce a tokenized training dataset that has BFF-deduplicated the first N WARCs from `experiments/distill/baseline_warcs_3000.txt`.

## Architecture

```
Phase 1 (per-spec, idempotent-incremental, slow):
  inventory_region.py --spec X      → inventory_{region}_{spec}.jsonl.gz × regions
  resolve_duplicates.py --spec X    → resolved_{spec}.jsonl.gz
  transfer_region.py --spec X       → by_region/{region}/{spec}/data-{h}/batch_NNNN.jsonl.gz
                                       (rsync semantics; skip already-transferred)

Phase 2 (per (spec, N), fast, repeatable):
  curate_extracted.py --spec X --n N
    1. Load manifest, first-N hashes
    2. Strict gate: every hash present in resolved manifest with _done observed
    3. Filter resolved.jsonl.gz to those hashes, num_records > 0
    4. ExecutorStep bff_dedup(input_files=...)        ← NEW input_files kwarg on BffDedupConfig
    5. ExecutorStep default_tokenize(dedup output)
    6. Optional: replicate tokenized cache to chosen training regions
    7. Emit JSON for curation_plan.py paste
```

## Path scheme

```
gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/
  inventories/inventory_{region}_{spec}.jsonl.gz       NEW per-spec naming
  resolved/resolved_{spec}.jsonl.gz                    NEW per-spec
  by_region/{region}/{spec}/data-{h}/batch_NNNN.jsonl.gz   NEW spec subdir (legacy stays at by_region/{region}/data-...)

gs://marin-us-central1/curated/dedup/{spec}_{n}warcs-{exec_hash}/        ephemeral
gs://marin-us-central1/tokenized/{spec}_{n}warcs-{exec_hash}/            persistent
```

## Idempotency-incremental requirements

For Phase 1 re-run when more WARCs land:
- `inventory_region.py` rescans the regional bucket each time (cheap, listing-only).
- `resolve_duplicates.py` rebuilds resolved manifest from scratch each time from inventories (deterministic given the same input).
- `transfer_region.py` MUST skip files that already exist at destination. Use `gcloud storage rsync` semantics or a per-batch existence check.
- `curate_extracted.py` is N-specific so each (spec, N) is a fresh ExecutorStep tree; the executor's content-hash caching handles within-N idempotency.

## Per-spec, per-N decisions captured

| Decision | Choice |
|---|---|
| Staging strategy | us-central1, reuse existing `baseline_llm_extraction_consolidated` infrastructure |
| Multi-region duplicate priority | us-central1 > us-east5 > eu-west4 > us-east1 > us-west4 > us-central2 |
| Manifest-first vs dedup-first | **Manifest-first** (different from llm_curated_dclm_filtered pipeline) |
| Strict-done gate | Error out with offending hashes if missing |
| Dataset name in registry | `{spec}_{n}warcs` |
| Tokenization regions | Only regions we plan to train from (parameterizable, default: us-central1 only) |
| Cache hash by N | Yes — 100-WARC and 500-WARC datasets are independent BFF passes |
| Permanent artifacts | Tokenized cache only; dedup output is ephemeral |

## Code changes summary

| File | Action | Approx LoC |
|---|---|---:|
| `lib/marin/src/marin/transform/bff_dedup.py` | add `input_files: list[str] \| None` field to BffDedupConfig, prefer it over `input_path` glob | +10 |
| `experiments/baseline_collection/consolidate/inventory_region.py` | add `--spec` flag, switch source prefix | +20 |
| `experiments/baseline_collection/consolidate/resolve_duplicates.py` | add `--spec` flag, name outputs per-spec | +10 |
| `experiments/baseline_collection/consolidate/transfer_region.py` | add `--spec` flag, ensure rsync skips existing | +15 |
| `experiments/baseline_collection/consolidate/launch_*.py` | pass through `--spec` | +5 each |
| `experiments/baseline_collection/curate_extracted.py` | NEW manifest-first curation entrypoint | ~150 |

Total ~220 LoC; almost everything is glue around existing pieces.

## Execution order

1. Modify BffDedupConfig to accept input_files
2. Modify the 4 consolidation scripts + 3 launchers to accept --spec
3. Write curate_extracted.py
4. Validate against low_quality (whose Phase 1 is already done — no transfer, just re-run resolve and curate at N=100 to sanity-check)
5. Launch Phase 1 for high_quality (priority-100 is done; remaining will be incrementally added later)
6. Run Phase 2 for high_quality at N=100
7. Confirm with user, then run for med_quality, med_low_quality (once their priority-100 finishes)
8. User pastes generated registry lines, launches training via launch_curation_sweep.py

## Acceptance criteria

- For low_quality (already consolidated), `curate_extracted.py --spec low_quality --n 100` produces a sensible tokenized dataset and a registry line.
- For high_quality, Phase 1 transfer is incremental — first run moves only the priority-100 data; later runs add more.
- BFF dedup runs successfully against the manifest-filtered file list (no glob).
- Cache hashes differ between 100/500/3000 — different dedup outputs as expected.
