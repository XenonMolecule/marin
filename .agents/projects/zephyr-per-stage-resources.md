# Zephyr: Per-Stage Resource Configuration

## Problem

Zephyr pipelines use a single worker pool with uniform resources for all stages.
In inference pipelines (`inference_v2`), the stages are:

- **stage0-Map**: Read/parse input files from GCS (`flat_map`). Pure I/O — no model, no TPU needed.
- **stage1-Reshard**: Redistribute shards. No compute.
- **stage2-Map → Write**: Run vLLM inference on TPU + write results. Needs TPU.

Because stage0 uses the same TPU workers as stage2, data loading is blocked when
TPU nodes are unavailable. On resource-starved clusters, jobs sit at
`stage0-Map 0/150, 0 workers` for hours even though the work is trivially CPU-bound.

## Proposed Solution

Allow each stage to specify its own `ResourceConfig`, so stage0 can use cheap
CPU-only workers while stage2 uses TPU workers.

## Key Files

| File | Role |
|------|------|
| `lib/zephyr/src/zephyr/execution.py` | `ZephyrCoordinator`, `ZephyrWorker`, `ZephyrContext` |
| `lib/zephyr/src/zephyr/plan.py` | `PhysicalStage`, `StageType`, plan construction |
| `lib/marin/src/marin/generation/inference_v2.py` | `InferenceV2Config`, pipeline construction |

## Changes

### 1. `PhysicalStage` — add optional resource config (`plan.py`)

```python
@dataclass
class PhysicalStage:
    operations: list[PhysicalOp]
    output_shards: int | None = None
    stage_type: StageType = StageType.WORKER
    resources: ResourceConfig | None = None  # NEW: per-stage override
```

When `None`, falls back to the context-level default (current behavior).

### 2. `ZephyrContext` — accept per-stage resources (`execution.py` ~L885)

Option A (simple): Add a `stage0_resources` field for pre-reshard stages:

```python
@dataclass
class ZephyrContext:
    ...
    resources: ResourceConfig = ...          # default (used for TPU stages)
    io_resources: ResourceConfig | None = None  # if set, used for stages before first reshard
```

Option B (general): Accept a dict mapping stage index to resources:

```python
    stage_resources: dict[int, ResourceConfig] | None = None
```

Option A is simpler and covers the common case. Recommend starting there.

### 3. `ZephyrCoordinator.run_pipeline` — swap worker pools at stage boundaries (`execution.py` ~L525-568)

In the stage loop, before executing each stage:

```python
for stage_idx, stage in enumerate(plan.stages):
    # Determine resources for this stage
    stage_res = stage.resources or self._default_resources

    if stage_res != self._current_resources:
        # Tear down old worker group
        self._shutdown_workers()
        # Spin up new worker group with stage-appropriate resources
        self._create_workers(stage_res, count=self._num_workers_for(stage_res))
        self._current_resources = stage_res

    # ... rest of stage execution unchanged
```

Key considerations:
- **Worker count may differ**: CPU stages might want more workers (cheap), TPU stages fewer.
- **Worker setup**: Stage0 CPU workers should skip `worker_fn` (no model loading).
  The `shared_data["worker_fn"]` callback is only relevant for inference stages.
- **Coordinator re-initialization**: `self.num_workers` tracking needs to handle
  the count changing between stages. The `initialize()` call sets expected worker
  count — either re-initialize or make it dynamic.

### 4. `ZephyrWorker` — conditionally skip model loading (`execution.py` ~L670)

Workers currently call `worker_fn()` during first task execution to load the model.
For CPU-only stage0 workers, this should be skipped. Options:

- Pass a flag in `shared_data` indicating whether to run `worker_fn`
- Have the coordinator set a stage-specific `shared_data` before each stage
- Simplest: stage0 workers never receive tasks that trigger `worker_fn` because
  the `flat_map` op doesn't use it — **verify this is already the case**

### 5. `InferenceV2Config` — expose the option (`inference_v2.py`)

```python
@dataclass
class InferenceV2Config:
    ...
    io_resources: ResourceConfig | None = None  # Resources for data-loading stages

    def _default_io_resources(self) -> ResourceConfig:
        return ResourceConfig(cpu=4, ram="16g")
```

In pipeline construction, attach `io_resources` to pre-reshard stages.

## Sequence of Operations (Runtime)

1. Pipeline starts → coordinator creates **CPU worker pool** (e.g., 32 workers)
2. stage0-Map runs: workers read GCS files, parse JSONL. Fast, no TPU.
3. stage0 completes → coordinator **shuts down CPU workers**
4. stage1-Reshard: no workers needed
5. Coordinator creates **TPU worker pool** (e.g., 16 workers). Workers load vLLM model.
6. stage2-Map→Write runs: inference + GCS writes. Slow, needs TPU.
7. stage2 completes → coordinator shuts down TPU workers

## Estimated Effort

- ~30 lines in `plan.py` (add field, propagate through plan construction)
- ~50 lines in `execution.py` (worker pool swap logic, resource tracking)
- ~10 lines in `inference_v2.py` (expose config, attach to stages)
- Tests: add a test with heterogeneous stage resources

Total: ~100 lines of production code + tests.

## Risks / Considerations

- **Worker pool swap latency**: Tearing down CPU workers and waiting for TPU workers
  adds a pause between stages. Acceptable since TPU compilation already takes ~10min.
- **Coordinator state**: Must cleanly handle worker count changes. The coordinator
  tracks workers by name — need to ensure old worker names don't collide with new ones.
- **Backwards compatibility**: All existing pipelines pass a single `resources` and
  would continue working (per-stage resources default to `None` → use context default).
- **Skip_existing interaction**: stage2's `skip_existing` check happens inside the
  Write op. Unaffected by worker pool changes.
