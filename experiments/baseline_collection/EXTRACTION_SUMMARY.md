# LLM Web Extraction — Compute & Timeline

**Task**: 3,000 WARCs × ~54k records each → Qwen3-8B extraction on TPU
**Per WARC**: ~20h on 4-chip, ~18h on 8-chip (108 batches of 500 records)

## Available Compute

| TPU | Chips | tok/s per VM | Jobs/slice | tok/s per slice | Max slices | Regions |
|-----|-------|-------------|-----------|----------------|-----------|---------|
| v5p-8 | 4 | ~1,100 | 1 | ~1,100 | 4,096 | us-central1, us-east5 |
| v5p-16 | 4+4 | ~630 | **2** | **~1,260** | 2,048 | us-central1, us-east5 |
| v5p-32 | 4×4 | ~630 | **4** | **~2,520** | 1,024 | us-central1, us-east5 |
| v5litepod-4 | 4 | ~450 | 1 | ~450 | 2,048 | eu-west4, us-west4 |
| v5litepod-8 | 8 | ~690 | 1 | ~690 | 1,024 | eu-west4, us-west4 |
| v6e-4 | 4 | ~450 | 1 | ~450 | 3,072 | eu-west4, us-east1, us-east5 |
| v6e-8 | 8 | ~690 | 1 | ~690 | 1,536 | eu-west4, us-east1, us-east5 |

All tok/s numbers are **output tokens** (decode), measured on Qwen3-8B extraction workload.
Multi-host VMs share pod memory bandwidth, so per-VM throughput is ~630 vs ~1,100 standalone.
But 2 VMs × 630 = 1,260 total — still more per-slice than single-host.

Max slices = quota ceiling. Actual availability depends on autoscaler + cluster load.
Currently ~27 nodes ready at idle. Scales up with demand.

## Timeline (at 75% uptime)

| Sustained effective jobs | WARCs/day | **Days to finish** |
|-------------------------|-----------|-------------------|
| 50 | ~45 | **67** |
| 100 | ~90 | **33** |
| 150 | ~135 | **22** |
| 200 | ~180 | **17** |
| 250 | ~225 | **13** |
| 325 | ~290 | **10** |

**To hit 7 days**: either 475+ effective jobs, or ~200 jobs + pre-filter junk HTML (drops 35% of records, cuts per-WARC time to ~13h).

## What "effective jobs" means

- 1 v5p-8 slice = 1 effective job
- 1 v5p-16 slice = **2** effective jobs (both VMs run independently)
- 1 v5p-32 slice = **4** effective jobs
- Multi-host slices use `TPU_PROCESS_BOUNDS=1,1,1` for per-VM isolation

All TPU types confirmed working. Per-batch checkpointing survives preemption. Cross-region resume. Claim-based deduplication. Launch as many jobs as autoscaler will serve.
