# Isoflop-Curation Checkpoint Cleanup — Savings Summary

**Date**: 2026-05-13
**Scope**: All `gs://marin-*/checkpoints/isoflop-curation/curation-*` runs with `.data_curation_DONE` marker, across 5 regions.
This includes the WARC scaling sweep (N={100,500,1k,2k}, tag `expWARC_natural`), the 3000-WARC fixed-model sweep (`expFM_natural`), and other isoflop curation experiments (`expA_natural`, `expB_T20T`, `expC_T33T`, etc.) — all under your `experiments/scaling_law_sweeps/` framework.

**Action**: Deleted inner `checkpoints/` (training state). Preserved `hf/` (inference-ready) and `.data_curation_DONE` + summary JSON.

## Headline

| | |
|---|---:|
| Runs cleaned | **1060** |
| **TB freed** | **18.96 TB** |
| In-progress runs left untouched | (full run dir intact, training can resume) |

## By region

| Region | DONE runs cleaned | Freed |
|---|---:|---:|
| marin-us-east5 | 501 | 10378.0 GB |
| marin-us-central1 | 269 | 5878.7 GB |
| marin-eu-west4 | 238 | 2314.2 GB |
| marin-us-central2 | 28 | 230.7 GB |
| marin-us-east1 | 24 | 159.5 GB |

## By experiment tag

| Experiment tag | Runs | Freed | Description |
|---|---:|---:|---|
| `expC_T33T` | 252 | 10041.3 GB | Curation isoflop experiment C (33T tokens, 10k WARCs) |
| `expFM_natural` | 199 | 4179.6 GB | Fixed-model sweep (3000 WARCs) |
| `expWARC_natural` | 505 | 3372.7 GB | WARC scaling sweep (N=100/500/1k/2k) |
| `expA_natural` | 62 | 874.1 GB | Curation isoflop experiment A |
| `expB_T20T` | 42 | 493.4 GB | Curation isoflop experiment B (20T tokens) |

## By WARC count (WARC + FM runs only)

| N WARCs | Runs cleaned | Freed |
|---:|---:|---:|
| 100 | 140 | 252.6 GB |
| 500 | 112 | 401.8 GB |
| 1000 | 140 | 851.1 GB |
| 2000 | 113 | 1867.1 GB |
| 3000 (FM) | 199 | 4179.6 GB |

## What was preserved

For each of the 1060 cleaned runs:
- `hf/` directory — inference-ready (loadable via `vllm` / `transformers`)
- `.data_curation_DONE` completion marker
- per-run summary JSON in `metadata/data_curation_*_results/`

For in-progress runs (no DONE marker): **entire run dir left intact**, including `checkpoints/`. Training can resume normally.

## Audit trail

- DONE path list: `/tmp/warc_done/done_paths.txt` (1060 entries)
- Pre-deletion sizes: `/tmp/warc_ckpt_sizes/*.out`
- Deletion logs (per-batch): `/tmp/warc_rm/batch_*.err`
- This summary: `.agents/scratchpad/warc_inventory/SAVINGS_SUMMARY.md`
