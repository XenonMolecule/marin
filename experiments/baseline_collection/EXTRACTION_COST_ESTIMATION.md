# LLM Extraction — Inference Token Counts & Cost Estimation

Consolidated reference for estimating the cost of the Qwen3-8B web-extraction
runs (e.g. "what would this have cost on Together AI instead of our TPUs").
Gathered 2026-05-25.

The relevant quantity for API pricing is **inference tokens** (prompt + output),
*not* the trained-token counts in `BASELINE_TABLE.md`. Output = thinking +
response (both are billed as completion tokens by API providers).

## Spec naming (important)

The three "quality runs" map to extraction specs in `extraction_specs.py`:

| User's term | Spec | GCS path |
|---|---|---|
| **low quality run** | `low_quality` (= legacy `llm_curated` baseline) | `documents/baseline_llm_extraction/` (no spec subdir) |
| **medium quality run** | `med_quality` (quality_extraction_v8) | `documents/baseline_llm_extraction/med_quality/` |
| **high quality run** | `high_quality` | `documents/baseline_llm_extraction/high_quality/` |
| (also exists) | `med_low_quality` | `documents/baseline_llm_extraction/med_low_quality/` |

The original 2026-05-04 inference accounting was run against the legacy
unprefixed path — that **is** the `low_quality` run.

## Token counts we have

### low_quality (= original llm_curated, computed 2026-05-04)

Projected from 51.8% sidecar coverage (145,648 of 281,254 resolved batches)
via bootstrap; per-region means agree to ~1% (MAR confirmed).

| Slice | Value |
|---|---:|
| Input tokens, all records | **2.207 T** |
| Thinking tokens, kept | 126.23 B |
| Response tokens, kept | 59.65 B |
| Input tokens, kept | 1.687 T |
| Total inference FLOPs | 6.72 × 10²² |
| Keep rate | 74.03% |

Artifacts: `gs://marin-us-central2/scratch/llm_curated_flop_estimate/{summary.json,aggregate.jsonl.gz}`
Aggregator: `option_a_flop_report.py` + `pre_stage_sidecars.py`

> Note: this projection reported **kept** thinking/response. For API billing we
> want **all-records** output tokens (you pay for filtered generations too).
> If a precise Together quote is needed, re-run `aggregate_flop_sidecars.py`
> against the legacy path to get `thinking_all` / `response_all` the same way
> high_quality has them below.

### high_quality (computed; full aggregation, no projection)

Direct sums over all 291,195 sidecar batches across 5 regions.
Source: `gs://marin-us-central1/scratch/high_quality_flop_estimate/summary.json`

| Slice | All records | Kept only |
|---|---:|---:|
| Input tokens | **2,444,315,993,098 (2.444 T)** | 195.65 B |
| Thinking tokens | 47,244,935,464 (47.24 B) | 17.50 B |
| Response tokens | 14,841,725,094 (14.84 B) | 12.44 B |
| **Output (think+resp)** | **62,086,660,558 (62.09 B)** | 29.95 B |
| Records | 144,843,486 | 10,835,778 |
| FLOPs | 6.96 × 10²² | 6.31 × 10²¹ |

Keep rate: **7.48%** (much stricter than low_quality's 74%). Status mix:
filtered_short 132.85M, kept 10.84M, thinking_overflow_max_tokens 602K,
thinking_overflow_context 513K, filtered_pattern 39.5K.

For API pricing the inputs are: **prompt = 2.444 T input tokens**,
**completion = 62.09 B output tokens**.

### med_quality — NOT YET COMPUTED ⚠️

No `med_quality_flop_estimate/` exists in any region's scratch. This is the
remaining run to count (see command below). `med_low_quality` is also
uncomputed if needed.

## How to count the tokens

Each extraction worker writes a per-record `batch_NNNN.tokens.gz` sidecar
(input/thinking/response token counts + status). Two paths:

### Spec-aware aggregator (preferred — what high_quality used)

`experiments/baseline_collection/aggregate_flop_sidecars.py` walks the
consolidated us-central1 mirror for one spec, sums exact Qwen3-8B inference
FLOPs and per-status token totals, and writes `summary.json` +
`aggregate.jsonl.gz`. Run as an in-region Iris CPU job (reads stay intra-region):

```bash
iris --config lib/iris/examples/marin.yaml job run \
    --cpu 4 --memory 8GB --region us-central1 \
    --job-name flop-agg-med_quality --no-wait \
    -- python -m experiments.baseline_collection.aggregate_flop_sidecars \
        --spec med_quality \
        --output-dir gs://marin-us-central1/scratch/med_quality_flop_estimate/
```

Then read the small `summary.json` for `grand_total_sums` (input_all /
thinking_all / response_all are the API-billable totals).

### Generic per-batch aggregate + projection (what low_quality used)

For runs whose sidecars are only partially present, aggregate per-region then
bootstrap-project to the full batch count:

```bash
# 1. Per-region aggregate (run once per region, in-region)
uv run python experiments/baseline_collection/aggregate_token_stats.py \
    --output gs://marin-us-central1/scratch/token_aggregate_us_central1.jsonl.gz \
    --buckets marin-us-central1

# 2. Project to full run with bootstrap CI
uv run python experiments/baseline_collection/project_tokens.py \
    --inputs /tmp/aggregates/{us_central1,us_east5,eu_west4}.jsonl.gz \
    --projected-batches 281254
```

FLOP formula (exact prefill+decode for Qwen3-8B) lives in
`aggregate_token_stats.py:inference_flops()` and is duplicated in
`aggregate_flop_sidecars.py` / `option_a_flop_report.py`.

## Together AI pricing — NOT in the repo

There is **no Together AI price table checked in**. The only references:

- `experiments/distill/web_extraction_reprocess.py` supports an `api` backend
  via litellm (`MODEL_NAME=together_ai/Qwen/Qwen3-8B`, `TOGETHER_API_KEY=...`,
  `INFERENCE_BACKEND=api`). This is the harness that *would* run extraction
  through Together rather than local vLLM/TPU.
- One cost hint in its docstring: *"first 10% of each spec (~82K rows) ≈ $1K
  with Kimi K2.5."* That's for a different (much larger) model than Qwen3-8B
  and is row-based, not token-based — not directly usable for these runs.

**To finish the cost estimate we need a current Together $/M-token rate** (input
and output priced separately) for the model in question — bring that to the
pricing session and multiply against the token totals above.

## Known API price points (per 1M tokens)

| Model | Input | Output |
|---|---:|---:|
| Qwen3-8B (the extraction model) | $0.10 | $0.15 |
| gpt-oss-120b | $0.15 | $0.60 |
| gpt-5.4-nano | $0.20 | $1.25 |

## Worked estimate

Cost = input_all × $in/M + output_all × $out/M. You pay for *all* records
(filtered prompts/generations included), so these use all-records totals.

### high_quality — exact

Direct token sums (not projected): input 2,444,315,993,098 / output
62,086,660,558 (thinking 47,244,935,464 + response 14,841,725,094).

| Model | Input $ | Output $ | **Total** |
|---|---:|---:|---:|
| Qwen3-8B | $244,431.5993 | $9,312.9991 | **$253,744.60** |
| gpt-oss-120b | $366,647.3990 | $37,251.9963 | **$403,899.40** |
| gpt-5.4-nano | $488,863.1986 | $77,608.3257 | **$566,471.52** |

### low_quality — derived from rounded projections†

input 2,207,000,000,000 / output 185,880,000,000 (kept only:
126.23B think + 59.65B resp). These tokens are *rounded* projections, so the
"exact" dollars below carry that rounding — not true cent-level precision.

| Model | Input $ | Output $ | **Total** |
|---|---:|---:|---:|
| Qwen3-8B | $220,700.00 | $27,882.00 | **$248,582.00** |
| gpt-oss-120b | $331,050.00 | $111,528.00 | **$442,578.00** |
| gpt-5.4-nano | $441,400.00 | $232,350.00 | **$673,750.00** |

med_quality: pending token count.

† low_quality only has **kept** output recorded (think 126.23B + resp 59.65B);
all-records output is higher, so its total is a lower bound until re-aggregated.
Input cost dominates because keep rates are low (7.5% high / 74% low) but every
prompt is billed.

