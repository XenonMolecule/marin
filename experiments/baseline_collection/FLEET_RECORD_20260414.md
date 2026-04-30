# Extraction Fleet Record — 2026-04-14 ~09:15 UTC

**Total active workers**: ~128 (107 running, 24 pending)
**Configs recovered from Iris gRPC `GetJobStatusResponse.request.entrypoint`**

## Parent Jobs & Exact Configs

All parents launched with:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
    --memory 2GB --cpu 2 --no-wait --job-name extract-{NAME} \
    -- python experiments/baseline_collection/launch_adaptive.py \
    --tpu-type {TYPE} --max-count {MAX} --initial-batch {INIT} --chunk-size {CHUNK} \
    --check-interval {INTERVAL} --child-priority {PRIORITY}
```

Note: ALL parents use `--child-priority batch` (even the "boost" ones). The "boost" naming is historical.

### v5p-8 (8 running, 0 pending)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v5p8-batch-b` | 12 | 2 | 2 | 300 | 4 | 0 |
| `v5p8-batch-e` | 12 | 2 | 2 | 300 | 4 | 0 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p8-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-8 --max-count 12 --initial-batch 2 --chunk-size 2 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p8-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-8 --max-count 12 --initial-batch 2 --chunk-size 2 --child-priority batch
```

### v5p-16 (16 running, 0 pending)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v5p16-batch-d` | 16 | 4 | 2 | 60 | 2 | 0 |
| `v5p16-batch-e` | 16 | 4 | 2 | 60 | 8 | 0 |
| `v5p16-boost` | 8 | 1 | 1 | 60 | 6 | 0 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p16-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-16 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p16-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-16 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p16-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-16 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch
```

### v5p-32 (23 running, 0 pending)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v5p32-batch-c` | 8 | 2 | 2 | 60 | 6 | 0 |
| `v5p32-batch-d` | 12 | 2 | 2 | 60 | 11 | 0 |
| `v5p32-boost` | 8 | 1 | 1 | 60 | 6 | 0 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p32-batch-e -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-32 --max-count 8 --initial-batch 2 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p32-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-32 --max-count 12 --initial-batch 2 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p32-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-32 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch
```

### v5p-64 (15 running, 1 pending)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v5p64-batch-d` | 12 | 4 | 2 | 60 | 6 | 0 |
| `v5p64-batch-e` | 8 | 2 | 2 | 60 | 5 | 0 |
| `v5p64-boost` | 8 | 1 | 1 | 60 | 4 | 1 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p64-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-64 --max-count 12 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p64-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-64 --max-count 8 --initial-batch 2 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p64-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-64 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch
```

### v5litepod-4 (0 running, 12 pending — no TPUs available)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v5litepod4-batch-a` | 48 | 8 | 4 | 60 | 0 | 4 |
| `v5litepod4-batch-b` | 48 | 8 | 4 | 60 | 0 | 8 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod4-batch-c -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-4 --max-count 48 --initial-batch 8 --chunk-size 4 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod4-batch-d -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-4 --max-count 48 --initial-batch 8 --chunk-size 4 --check-interval 60 --child-priority batch
```

### v5litepod-8 (0 running, 1 pending — no TPUs available)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v5litepod8-boost` | 8 | 1 | 1 | 60 | 0 | 1 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod8-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-8 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch
```

### v5litepod-16 (7 running, 2 pending)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v5litepod16-batch-b` | 16 | 4 | 2 | 60 | 2 | 0 |
| `v5litepod16-batch-e` | 16 | 4 | 2 | 60 | 4 | 2 |
| `v5litepod16-boost-v2` | 8 | 1 | 1 | 60 | 1 | 0 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod16-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-16 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod16-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-16 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod16-boost-v3 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-16 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch
```

### v5litepod-32 (3 running, 0 pending)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v5litepod32-boost` | 8 | 1 | 1 | 60 | 3 | 0 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod32-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-32 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch
```

### v6e-4 (30 running, 4 pending)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v6e4-batch-a` | 16 | 4 | 2 | 60 | 12 | 2 |
| `v6e4-batch-b` | 16 | 4 | 2 | 60 | 12 | 2 |
| `v6e4-batch-e` | 16 | 4 | 2 | 60 | 6 | 0 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e4-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-4 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e4-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-4 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e4-batch-h -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-4 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
```

### v6e-16 (0 running, 1 pending — no TPUs available)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v6e16-boost` | 8 | 1 | 1 | 60 | 0 | 1 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e16-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-16 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch
```

### v6e-32 (3 running, 3 pending)

| Parent | Max | Init | Chunk | Interval | Running | Pending |
|--------|-----|------|-------|----------|---------|---------|
| `v6e32-batch-a` | 12 | 2 | 2 | 60 | 3 | 2 |
| `v6e32-boost` | 8 | 1 | 1 | 60 | 0 | 1 |

Relaunch:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e32-batch-b -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-32 --max-count 12 --initial-batch 2 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e32-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-32 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch
```

---

## Quick Relaunch Script (all types)

Copy-paste this block to relaunch the full fleet with the same configs.
Suffixes incremented to avoid Iris name collisions. Bump the letter if taken.

```bash
# v5p-8 (2 parents, max=12, init=2, chunk=2, no explicit check-interval = default 300)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p8-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-8 --max-count 12 --initial-batch 2 --chunk-size 2 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p8-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-8 --max-count 12 --initial-batch 2 --chunk-size 2 --child-priority batch

# v5p-16 (2 batch + 1 boost)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p16-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-16 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p16-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-16 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p16-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-16 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch

# v5p-32 (2 batch + 1 boost)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p32-batch-e -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-32 --max-count 8 --initial-batch 2 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p32-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-32 --max-count 12 --initial-batch 2 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p32-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-32 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch

# v5p-64 (2 batch + 1 boost)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p64-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-64 --max-count 12 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p64-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-64 --max-count 8 --initial-batch 2 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5p64-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5p-64 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch

# v5litepod-4 (2 parents, max=48 — currently 0 TPUs available)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod4-batch-c -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-4 --max-count 48 --initial-batch 8 --chunk-size 4 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod4-batch-d -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-4 --max-count 48 --initial-batch 8 --chunk-size 4 --check-interval 60 --child-priority batch

# v5litepod-8 (1 parent — currently 0 TPUs available)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod8-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-8 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch

# v5litepod-16 (2 batch + 1 boost)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod16-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-16 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod16-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-16 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod16-boost-v3 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-16 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch

# v5litepod-32 (1 parent)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v5litepod32-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v5litepod-32 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch

# v6e-4 (3 parents)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e4-batch-f -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-4 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e4-batch-g -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-4 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e4-batch-h -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-4 --max-count 16 --initial-batch 4 --chunk-size 2 --check-interval 60 --child-priority batch

# v6e-16 (1 parent — currently 0 TPUs available)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e16-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-16 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch

# v6e-32 (1 batch + 1 boost)
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e32-batch-b -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-32 --max-count 12 --initial-batch 2 --chunk-size 2 --check-interval 60 --child-priority batch
uv run iris --config lib/iris/examples/marin.yaml job run --memory 2GB --cpu 2 --no-wait --job-name extract-v6e32-boost-v2 -- python experiments/baseline_collection/launch_adaptive.py --tpu-type v6e-32 --max-count 8 --initial-batch 1 --chunk-size 1 --check-interval 60 --child-priority batch
```

## Notes

- **Name collisions**: Iris remembers killed job names. Relaunch commands use incremented suffixes. If a suffix is taken, bump the letter.
- **All parents use `--child-priority batch`**. The "boost" naming is historical — they just have smaller max-count (8) and chunk-size (1) for more cautious scaling on expensive TPU types.
- **v5p-8 is the only type without `--check-interval 60`** — it uses the default 300s.
- **Config pattern**: Larger/more expensive TPU types use smaller max-count and chunk-size. v5litepod-4 is the exception with max=48 since those are small cheap chips.
- **Dashboard**: `uv run python experiments/baseline_collection/dashboard.py` on port 8090.
