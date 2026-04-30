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
