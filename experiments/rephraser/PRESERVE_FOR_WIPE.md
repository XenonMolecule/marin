# GCS Artifacts to Preserve (us-central1 Cluster Wipe)

Generated 2026-03-20. Auto-preserve threshold: files accessed in last 2 weeks (since ~March 6).

**Almost everything below was last modified before March 6 and is at risk of deletion.**

**Total: ~500 GB** (final checkpoints only, plus data)

## Rephraser SFT Model Checkpoints — Final Steps Only (~93 GB)

Only the final HF checkpoint for each model. The 4b-v5-72fd0f is a failed run (step 8 only) and is excluded.

| Path | Created | Size | Description |
|------|---------|------|-------------|
| `.../qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318/` | 2026-02-25 | 30.5 GB | **Main production rephraser (Qwen3 8B, v4)** |
| `.../qwen3-4b-thinking-rephraser-sft-32k-v5-e17272/hf/step-750/` | 2026-02-23 | 15.0 GB | Qwen3 4B thinking rephraser (v5) |
| `.../qwen3-1.7b-rephraser-sft-v4-4c525e/hf/step-1318/` | 2026-02-20 | 6.4 GB | Qwen3 1.7B rephraser v4 |
| `.../qwen3-0.6b-rephraser-sft-v6-5a8b19/hf/step-1318/` | 2026-02-21 | 2.2 GB | Qwen3 0.6B rephraser v6 |
| `.../qwen3-8b-rephraser-mid-sft-v1-5a71ec/hf/step-1250/` | 2026-02-23 | 30.5 GB | 8B mid-training SFT |
| `.../qwen3-1.7b-rephraser-mid-sft-v1-bab2de/hf/step-9398/` | 2026-03-05 | 6.4 GB | 1.7B mid-training SFT |
| `.../qwen3-0.6b-rephraser-mid-sft-v1-d39527/hf/step-9398/` | 2026-03-04 | 2.2 GB | 0.6B mid-training SFT |

All paths are under `gs://marin-us-central1/checkpoints/`.

## Scaling Ladder Base Checkpoint (165.1 GB)

Everything depends on this. **Created Jan 24 -- oldest artifact and most at risk.**

| Path | Created | Size |
|------|---------|------|
| `gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0/` | **2026-01-24** | 165.1 GB |

## Cooldown Training Outputs (~124 GB)

All created late Feb / early March. All at risk.

| Path | Created | Size | Description |
|------|---------|------|-------------|
| `gs://marin-us-central1/cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b/` | 2026-03-04 | 20.7 GB | **Main 150-WARC rephraser cooldown (v2)** |
| `gs://marin-us-central1/cooldown-rephraser-d7d976d3-v2-1cdc5a/` | 2026-02-28 | 20.7 GB | Rephraser cooldown v2 (alt) |
| `gs://marin-us-central1/cooldown-dclm-filtered-v1-54cdb7/` | 2026-03-03 | 5.2 GB | DCLM filtered cooldown |
| `gs://marin-us-central1/cooldown-dclm-100m-v2-32bcbc/` | 2026-02-27 | 31.4 GB | DCLM 100M baseline cooldown |
| `gs://marin-us-central1/short-cooldown-rephraser-d7d976d3-v2-eebec0/` | 2026-03-04 | 5.2 GB | **Short rephraser cooldown** |
| `gs://marin-us-central1/short-cooldown-dclm-49e19a/` | 2026-03-03 | 20.7 GB | Short DCLM cooldown |
| `gs://marin-us-central1/short-cooldown-nemotron-only-391eeb/` | 2026-03-03 | 20.7 GB | Short nemotron-only baseline |

Note: `cooldown-rephraser-d7d976d3-v2-8383c8` is a stub (0.2 KB), excluded.

## Tokenized Data (~15 GB)

| Path | Created | Size | Description |
|------|---------|------|-------------|
| `gs://marin-us-central1/tokenized/rephraser_spec_d7d976d3_cooldown-02c17e/` | 2026-03-02 | 1.2 GB | Main rephraser tokenized data (362M tokens) |
| `gs://marin-us-central1/tokenized/nemotron_cooldown_1e20-666089/` | 2026-03-02 | 10.1 GB | Nemotron cooldown tokens |
| `gs://marin-us-central1/tokenized/nemotron_cooldown_1e20_short_1b-413400/` | 2026-03-03 | 2.0 GB | Short nemotron tokens (1B) |
| `gs://marin-us-central1/tokenized/nemotron_cooldown_1e20_short_extra_300m-3102d5/` | 2026-03-03 | 0.6 GB | Extra 300M for baseline |
| `gs://marin-us-central1/tokenized/dclm_baseline_100m_llama3-f42a23/` | 2026-02-27 | 1.2 GB | DCLM 100M baseline tokenized |

## Raw WARC Data (~30 GB)

Slow to re-download from CommonCrawl.

| Path | Created | Size | Description |
|------|---------|------|-------------|
| `gs://marin-us-central1/raw/commoncrawl/rephraser_sweep_batch0-231d96/` | 2026-03-18 | ? | WARC batch (likely safe -- recent) |
| `gs://marin-us-central1/raw/commoncrawl/rephraser_sweep_batch0-690f70/` | 2026-02-28 | 17.0 GB | WARC batch (AT RISK) |
| `gs://marin-us-central1/raw/commoncrawl/rephraser_sweep_batch0-7ffa78/` | 2026-02-26 | 1.4 GB | WARC batch (AT RISK) |
| `gs://marin-us-central1/raw/commoncrawl/rephraser_sweep_batch0-7fff7e/` | 2026-02-26 | 11.5 GB | WARC batch (AT RISK) |

## CDX Indices (12.8 GB)

| Path | Size | Description |
|------|------|-------------|
| `gs://marin-us-central1/cdx/` | 12.8 GB | All CDX indices (code, math, medical) |

## Domain HTML Downloads (~166 GB + mathhelpforum)

| Path | Size | Description |
|------|------|-------------|
| `gs://marin-us-central1/downloaded/` | 165.6 GB | All domain HTML downloads (code, math, medical) |
| `gs://marin-us-central1/mathhelpforum/cc_download-5b0941/` | 2.8 GB | MathHelpForum WARC download (772k files) |

## Quick Copy-Paste List (just the paths)

```
gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318/
gs://marin-us-central1/checkpoints/qwen3-4b-thinking-rephraser-sft-32k-v5-e17272/hf/step-750/
gs://marin-us-central1/checkpoints/qwen3-1.7b-rephraser-sft-v4-4c525e/hf/step-1318/
gs://marin-us-central1/checkpoints/qwen3-0.6b-rephraser-sft-v6-5a8b19/hf/step-1318/
gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-mid-sft-v1-5a71ec/hf/step-1250/
gs://marin-us-central1/checkpoints/qwen3-1.7b-rephraser-mid-sft-v1-bab2de/hf/step-9398/
gs://marin-us-central1/checkpoints/qwen3-0.6b-rephraser-mid-sft-v1-d39527/hf/step-9398/
gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0/
gs://marin-us-central1/cooldown-rephraser-d7d976d3-150warc-v2-f8aa0b/
gs://marin-us-central1/cooldown-rephraser-d7d976d3-v2-1cdc5a/
gs://marin-us-central1/cooldown-dclm-filtered-v1-54cdb7/
gs://marin-us-central1/cooldown-dclm-100m-v2-32bcbc/
gs://marin-us-central1/short-cooldown-rephraser-d7d976d3-v2-eebec0/
gs://marin-us-central1/short-cooldown-dclm-49e19a/
gs://marin-us-central1/short-cooldown-nemotron-only-391eeb/
gs://marin-us-central1/tokenized/rephraser_spec_d7d976d3_cooldown-02c17e/
gs://marin-us-central1/tokenized/nemotron_cooldown_1e20-666089/
gs://marin-us-central1/tokenized/nemotron_cooldown_1e20_short_1b-413400/
gs://marin-us-central1/tokenized/nemotron_cooldown_1e20_short_extra_300m-3102d5/
gs://marin-us-central1/tokenized/dclm_baseline_100m_llama3-f42a23/
gs://marin-us-central1/raw/commoncrawl/rephraser_sweep_batch0-690f70/
gs://marin-us-central1/raw/commoncrawl/rephraser_sweep_batch0-7ffa78/
gs://marin-us-central1/raw/commoncrawl/rephraser_sweep_batch0-7fff7e/
gs://marin-us-central1/cdx/
gs://marin-us-central1/downloaded/
gs://marin-us-central1/mathhelpforum/cc_download-5b0941/
```
