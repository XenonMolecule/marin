# LLM Extraction Fleet Status

**Task**: Extract 3,000 CommonCrawl WARCs (~54k HTML records each) using Qwen3-8B rephraser
**Last updated**: 2026-04-07 22:05 UTC

## Per-WARC Stats

- Records per WARC (after char filter): **~54,000**
- Batch size: **500 records**
- Batches per WARC: **~108**
- Average output tokens per record: **~1,800** (includes thinking traces)

## Confirmed Working TPU Types

| TPU | Chips | TP | Batch Time | WARC Time | tok/s (output) | Regions | Max Slices | Status |
|-----|-------|-----|-----------|-----------|---------------|---------|-----------|--------|
| v5p-8 | 4 | 4 | **~11 min** | **~20h** | ~1,100 | us-central1, us-east5 | 4,096 | **production ready** |
| v6e-8 | 8 | 8 | ~10 min | ~18h | ~688 | eu-west4, us-east1, us-east5 | 1,536 | **production ready** |
| v6e-4 | 4 | 4 | ~14 min | ~25h | ~447 | eu-west4, us-east1, us-east5 | 3,072 | **production ready** |
| v5litepod-4 | 4 | 4 | ~14 min | ~25h | ~447 | eu-west4, us-west4 | 2,048 | **production ready** |
| v5litepod-8 | 8 | 8 | ~10 min | ~18h | ~688 | eu-west4, us-west4 | 1,024 | **production ready** |

**Measured timing** (from 20-batch sustained run on v5p-8, XLA cache warm):
- Batch 1 (cold XLA): ~13 min
- Batches 2-20 (warm): **~11 min average**
- Per WARC (108 batches): **~20h** on 4-chip, **~18h** on 8-chip

**Total max slices across single-host types: 11,776**

## Multi-Host TPUs (BREAKTHROUGH: zero-waste data parallel)

Multi-host slices run **one independent extraction job per VM** using
`TPU_PROCESS_BOUNDS=1,1,1` to restrict each VM's JAX to its local 4 chips.
The claim mechanism prevents duplicate work. **No wasted chips.**

| TPU | VMs/slice | Jobs/slice | tok/s per slice | Effective jobs (max slices) | Regions | Status |
|-----|-----------|-----------|----------------|---------------------------|---------|--------|
| v5p-16 | 2 | **2** | **~1,260** (2×630) | 4,096 | us-central1, us-east5 | **confirmed working** |
| v5p-32 | 4 | **4** | **~2,520** (4×630) | 4,096 | us-central1, us-east5 | **confirmed working** |
| v5p-64 | 8 | **8** | ~5,040 (est) | 4,096 | us-central1, us-east5 | untested (should work) |

**Launch command for multi-host** (same as single-host + process bounds):
```bash
iris job run --tpu v5p-16 --memory 128GB --max-retries 10 \
    --extra marin:vllm --extra marin:tpu --no-wait \
    -e TPU_PROCESS_BOUNDS "1,1,1" -e TPU_CHIPS_PER_PROCESS_BOUNDS "2,2,1" \
    --job-name extract-prod-v5p16-<N> \
    -- python run_extract_standalone.py --manifest ... --shuffle-seed <N>
```

## Not Yet Working

| TPU | Issue | Fix Needed |
|-----|-------|-----------|
| v4-8 | No workers provisioned (autoscaler not matching) | May need different runtime version or scale group config |

## Production Time Estimates

**Key formula**: `days = 3000 WARCs × avg_warc_hours / (effective_jobs × 24 × uptime_fraction)`

Each effective job = 1 VM with 4 local TPU chips processing 1 WARC at a time (~23h/WARC).
Assuming **75% uptime** (preemption + restart overhead + engine reload):

### Effective jobs by slice type

| Slice type | Slices requested | Effective jobs | Notes |
|-----------|-----------------|---------------|-------|
| 20 × v5p-8 | 20 | 20 | 1 job per slice |
| 10 × v5p-16 | 10 | **20** | 2 jobs per slice |
| 5 × v5p-32 | 5 | **20** | 4 jobs per slice |
| 20 × v5litepod-4 | 20 | 20 | 1 job per slice |
| 10 × v5litepod-8 | 10 | 10 | 1 job per slice (8 chips) |
| 10 × v6e-4 | 10 | 10 | 1 job per slice |
| 5 × v6e-8 | 5 | 5 | 1 job per slice (8 chips) |

### Scenarios

Using measured **~20h/WARC** on 4-chip nodes, **~18h/WARC** on 8-chip, **75% uptime** (preemption + restarts):

| Scenario | Slices | Effective jobs | WARCs/day | Days for 3000 |
|----------|--------|---------------|-----------|---------------|
| Conservative | 30 | 40 | ~36 | **~83 days** |
| Moderate | 60 | 80 | ~72 | **~42 days** |
| Aggressive | 100 | 150 | ~135 | **~22 days** |
| **Max push** | **150** | **250** | ~225 | **~13 days** |
| **All-out** | **200+** | **350+** | ~315 | **~10 days** |

### The path to 1 week

To hit **7 days**, we need ~430 WARCs/day sustained → ~475 effective jobs at 75% uptime.

**Realistic max-push plan:**
| Type | Slices | Jobs/slice | Effective jobs |
|------|--------|-----------|---------------|
| v5p-16 | 50 | 2 | 100 |
| v5p-32 | 20 | 4 | 80 |
| v5p-8 | 40 | 1 | 40 |
| v5litepod-4 | 40 | 1 | 40 |
| v5litepod-8 | 20 | 1 | 20 |
| v6e-4 | 30 | 1 | 30 |
| v6e-8 | 15 | 1 | 15 |
| **Total** | **215 slices** | | **325 effective jobs** |

At 325 jobs × 75% uptime × ~1.1 WARCs/day = **~268 WARCs/day → ~11 days**

**To reach 7 days** additionally requires one of:
- Pre-filter HTML < 500 chars (-35% records → ~13h/WARC → **7 days at 200 jobs**)
- More slices from autoscaler (475+ effective jobs)
- Reduce max_tokens to 2048 (-30% generation time)

## Potential Speedups (not yet implemented)

| Optimization | Impact | Effort |
|-------------|--------|--------|
| Pre-filter HTML < 500 chars | -35% records → -35% time | Low |
| Pre-filter with resiliparse (>200 words) | -70% records → -70% time | Medium |
| Reduce max_tokens from 4096 to 2048 | ~30% faster generation | Trivial (config change) |
| Smaller model (Qwen3-4B) | ~2x faster per token | Needs quality validation |
| Unlock v5p-16 (device isolation) | +2,048 slices | Medium |
| Unlock v4-8 | +2,048 slices | Unknown |

## Infrastructure

- **Checkpointing**: Per-batch (500 records). Max 13 min lost on preemption.
- **Claim mechanism**: `_claimed` file with 3h stale timeout. Cross-region resume.
- **Output format**: `data-{warc_hash}/batch_NNNN.jsonl.gz` + `_done` marker
- **Orchestration**: Each job reads full 3000-WARC manifest with `--shuffle-seed N`. Claims prevent duplicate work.
- **Launch command**:
  ```bash
  iris job run --tpu <TYPE> --memory 128GB --max-retries 10 \
      --extra marin:vllm --extra marin:tpu --no-wait \
      --job-name extract-prod-<N> \
      -- python experiments/baseline_collection/run_extract_standalone.py \
      --manifest experiments/distill/baseline_warcs_3000.txt \
      --output-subdir documents/baseline_llm_extraction \
      --shuffle-seed <N>
  ```

## Confirmed TPU Compatibility

All tested on 2026-04-07. Each produces correct extraction output.

| TPU | Chips | TP | Engine Load | Batch Time | Result |
|-----|-------|-----|-----------|-----------|--------|
| v5p-8 | 4 | 4 | 166s | ~13 min | **PASS** — production workhorse |
| v6e-8 | 8 | 8 | 156s | ~10 min | **PASS** — fastest per-job |
| v6e-4 | 4 | 4 | ~170s | ~14 min | **PASS** |
| v5litepod-4 | 4 | 4 | 196s | ~14 min | **PASS** |
| v5litepod-8 | 8 | 8 | 196s | ~10 min | **PASS** |
| v5p-16 (2-host) | 4/VM | 4 | 169s | ~13 min/VM | **PASS** — 2 parallel jobs per slice |
| v5p-32 (4-host) | 4/VM | 4 | 167s | ~13 min/VM | **PASS** — 4 parallel jobs per slice |
| v4-8 | 4 | 4 | — | — | pending (no workers provisioned) |

## Active Test Jobs

| Job | TPU | Batches Done | Status |
|-----|-----|-------------|--------|
| extract-checkpoint-test | v5p-8 | 15/108 | running |
| extract-v5p16-test-v2 | v5p-16 | 1/108 per VM (2 VMs) | running |
| extract-v5p32-test | v5p-32 | — | just launched |
