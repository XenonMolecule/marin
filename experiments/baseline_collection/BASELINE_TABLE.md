# 3000-WARC Baseline Extraction Stats

Pool: `gs://marin-us-central2/raw/commoncrawl/baseline_3000-265ff5` (3,000
WARCs, post-BOS-fix where applicable).

The BOS-fix only affected `nemotron_full` and `llm_curated` (caches built
between the Levanter regression on 2026-04-10 and its repair). DCLM,
Resiliparse, FineWeb-Edu, and plain Nemotron were built before that window
and did not need rebuilding.

## Simplified

| Pipeline | Documents | Tokens |
|---|---:|---:|
| Raw HTML | 156.4M | 3.63T |
| Resiliparse | 141.6M | 142.7B |
| DCLM | 2.05M | 2.66B |
| Nemotron-CC | 4.99M | 2.70B |
| FineWeb-Edu | 794K | 817M |
| LLM Extraction (Qwen3-8B) | 103.7M | 56.01B |


## Full

| Pipeline | Documents | Tokens | Source |
|---|---:|---:|---|
| Raw HTML | 156,425,213 (156.43M) | ~3.63T† | `gs://marin-us-central2/metadata/baseline_3000_html_byte_counts/_summary.json` |
| Resiliparse | 141,588,417 (141.59M) | 142,652,598,588 (142.65B) | `gs://marin-us-central2/tokenized/baseline_resiliparse-7278c1/train/.stats.json` |
| DCLM | 2,049,283 (2.05M) | 2,663,454,015 (2.66B) | `gs://marin-us-central2/tokenized/baseline_dclm-23e9be/train/.stats.json` |
| Nemotron-CC (no rephrase) | 2,917,704 (2.92M) | 1,919,401,016 (1.92B) | `gs://marin-us-central2/tokenized/baseline_nemotron-c67de9/train/.stats.json` |
| Nemotron-CC (BOS-fixed) | 4,994,055 (4.99M) | 2,700,501,906 (2.70B) | `gs://marin-us-central1/tokenized/baseline_nemotron_full_bos_fixed-4b1ce7/train/.stats.json` |
| FineWeb-Edu | 794,230 (794K) | 817,221,529 (817M) | `gs://marin-us-central2/tokenized/baseline_fineweb_edu-7a3bc5/train/.stats.json` |
| LLM Extraction Qwen3-8B (BOS-fixed) | 103,706,721 (103.71M) | 56,008,357,279 (56.01B) | `gs://marin-us-central1/tokenized/baseline_llm_curated_bos_fixed-d04ef8/train/.stats.json` |

† Raw HTML token figure is **not recorded anywhere in the repo or GCS** —
the `_summary.json` blob has records/bytes (156.43M records, 12.39 TB
plain UTF-8 HTML) but not tokens, and no `baseline_raw_html-*` tokenized
cache exists on GCS (the `tokenize_raw_html` step in
`experiments/baseline_collection/pipeline.py:260` is defined but has not
been run end-to-end). 3.63T comes from an external measurement logged in
the user's notes.

## LLM Extraction inference accounting (Qwen3-8B)

Computed 2026-05-04 by aggregating per-record `.tokens.gz` sidecars across
all 5 regions of the consolidated mirror and projecting per-batch means to
281,254 canonical resolved batches via bootstrap (Option A — MAR validated
by per-region consistency check, see Per-region table below).

| Slice | Projected | 95% CI |
|---|---:|---|
| **Total inference FLOPs** | **6.72 × 10²²** | [6.718e22, 6.727e22] |
| FLOPs on kept records | 5.10 × 10²² | [5.097e22, 5.104e22] |
| FLOPs on filtered (wasted) | 1.62 × 10²² | — |
| Input tokens, all records | 2.207 T | — |
| Input tokens, kept records | 1.687 T | — |
| Thinking tokens, kept records | 126.23 B | — |
| Response tokens, kept records | 59.65 B | — |

n = 145,648 sidecar-bearing canonical batches (51.8% of 281,254 resolved
unique batches); 46 transient GCS read errors (0.03%). Per-region
per-batch means agree to within ~1% across europe-west4 / us-central1 /
us-east1 / us-east5 / us-west4, confirming MAR. Per-batch CV: input
tokens 11.5%, FLOPs 12.7%, response tokens 18.2%.

Status distribution (observed records):
- kept 74.03%, filtered_short 18.86%,
- thinking_overflow_max_tokens 4.37%, thinking_overflow_context 2.71%,
- filtered_pattern 0.03%

Artifacts:
- `gs://marin-us-central2/scratch/llm_curated_flop_estimate/aggregate.jsonl.gz` (per-batch dump)
- `gs://marin-us-central2/scratch/llm_curated_flop_estimate/summary.{json,txt}`
- Aggregator: `experiments/baseline_collection/option_a_flop_report.py`
- Pre-stage: `experiments/baseline_collection/pre_stage_sidecars.py` (copies the small `.tokens.gz` sidecars to a us-central2 bucket so the aggregator runs intra-region; total egress paid: ~$0.012 for 604 MiB)
