# Iris Usage Guide

## What is Iris?

Iris is a **controller-worker job orchestrator** that replaced Ray. The design is deliberately simpler:

```
Controller (single GCE VM):
├── gRPC service — job dispatch, worker registration
├── Scheduler — matches tasks to workers by constraints
├── Autoscaler — spins up/down TPU/GPU/CPU VMs as needed
└── Dashboard — web UI for monitoring

Workers (one per VM, auto-managed):
├── Task executor — runs your code in Docker containers
└── Heartbeat — reports status to controller every ~5s
```

Key difference from Ray: no distributed object store. Jobs communicate via JAX collectives (training) or GCS (data pipelines). The controller just schedules tasks onto VMs and manages their lifecycle.

---

## Day-to-Day Commands

All commands use this pattern — `--config` is a **global flag** that goes right after `iris`:

```bash
uv run iris --config lib/iris/examples/marin.yaml <subcommand>
```

### Submitting Jobs

```bash
# Basic TPU training job
uv run iris --config lib/iris/examples/marin.yaml job run \
  --extra marin:tpu \
  --tpu v5litepod-16 \
  -e WANDB_API_KEY $WANDB_API_KEY \
  -e HF_TOKEN $HF_TOKEN \
  -- python experiments/my_experiment.py

# Submit and detach (don't wait for completion)
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --extra marin:tpu --tpu v5litepod-16 \
  -- python experiments/my_experiment.py

# CPU-only data processing job
uv run iris --config lib/iris/examples/marin.yaml job run \
  -- python experiments/data_pipeline.py

# Pin to a specific zone
uv run iris --config lib/iris/examples/marin.yaml job run \
  --zone us-central2-b --tpu v5litepod-16 \
  -- python experiments/my_experiment.py
```

**Key flags:**

| Flag | What it does |
|---|---|
| `--tpu v5litepod-16` | Request a specific TPU type |
| `--gpu H100x8` | Request GPUs |
| `--extra marin:tpu` | Install Marin's TPU dependencies in the container |
| `--extra marin:cpu` | Install Marin's CPU dependencies |
| `-e KEY VALUE` | Pass environment variable (repeatable) |
| `--no-wait` | Detach after submission |
| `--zone us-central2-b` | Pin to a zone |
| `--memory 5GB` | Set memory requirement |
| `--region us-central2` | Pin to a region |

**Environment variables:** `HF_TOKEN`, `WANDB_API_KEY`, `HF_DATASETS_TRUST_REMOTE_CODE`, and `TOKENIZERS_PARALLELISM` are **auto-injected** from your shell environment. You only need `-e` if they're not already set in your shell.

### Monitoring Jobs

```bash
# List jobs (optionally filter by prefix)
uv run iris --config lib/iris/examples/marin.yaml job list
uv run iris --config lib/iris/examples/marin.yaml job list --prefix /michael

# Stream logs (follow mode)
uv run iris --config lib/iris/examples/marin.yaml job logs /michael/my-job --follow

# Recent logs only
uv run iris --config lib/iris/examples/marin.yaml job logs /michael/my-job --since-seconds 300

# Include child job logs (for multi-step pipelines)
uv run iris --config lib/iris/examples/marin.yaml job logs /michael/my-job --include-children
```

### Stopping Jobs

```bash
# Stop a job (includes children by default)
uv run iris --config lib/iris/examples/marin.yaml job stop /michael/my-job

# Stop just the parent, not children
uv run iris --config lib/iris/examples/marin.yaml job stop /michael/my-job --no-include-children
```

### Dashboard

```bash
# Opens SSH tunnel + prints dashboard URL
uv run iris --config lib/iris/examples/marin.yaml cluster dashboard
```

### Cluster Status

```bash
uv run iris --config lib/iris/examples/marin.yaml cluster status

# VM-level status (via controller)
uv run iris --controller-url http://localhost:10000 cluster vm status
```

---

## Job Lifecycle

Jobs go through these states:

```
PENDING → BUILDING → RUNNING → SUCCEEDED
                            ↘ FAILED
                            ↘ KILLED (you stopped it)
                            ↘ WORKER_FAILED (TPU preempted, etc.)
                            ↘ UNSCHEDULABLE (no matching hardware)
```

- **Preemption retries** default to 100 — if a TPU gets preempted, Iris automatically reschedules.
- **Failure retries** default to 0 — your code crashes = job fails. Override if needed.
- Job names follow the pattern `/<user>/<job-name>` (user defaults to your OS username).

---

## Dev TPU (Interactive Debugging)

For quick interactive TPU access:

```bash
# Allocate a dev TPU
uv run scripts/iris/dev_tpu.py --config lib/iris/examples/marin.yaml allocate --tpu-type v5p-8

# Run a command on it
uv run scripts/iris/dev_tpu.py --config lib/iris/examples/marin.yaml execute -- python train.py

# Run and stream logs (like job run but on your reserved TPU)
uv run scripts/iris/dev_tpu.py --config lib/iris/examples/marin.yaml watch -- python train.py
```

---

## Ray → Iris Cheat Sheet

| Ray (old) | Iris (new) |
|---|---|
| `uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait -- python exp.py` | `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --extra marin:tpu --tpu v5litepod-16 -- python exp.py` |
| `uv run scripts/ray/cluster.py dashboard` | `uv run iris --config lib/iris/examples/marin.yaml cluster dashboard` |
| `uv run scripts/ray/cluster.py list-jobs` | `uv run iris --config lib/iris/examples/marin.yaml job list` |
| `uv run scripts/ray/cluster.py stop-job <id>` | `uv run iris --config lib/iris/examples/marin.yaml job stop <id>` |
| `--env_vars WANDB_API_KEY=...` | `-e WANDB_API_KEY ...` (or auto-injected from shell) |

The biggest practical difference: you now **specify hardware explicitly** (`--tpu`, `--gpu`) instead of Ray picking it up from the cluster config. Iris autoscales the right VM type for your request.

---

## Programmatic Submission (Python SDK)

```python
from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec, EnvironmentSpec, tpu_device
from pathlib import Path

client = IrisClient.remote("http://controller:10000", workspace=Path("."))

job = client.submit(
    name="my-training-job",
    entrypoint=Entrypoint.from_command("python", "train.py"),
    resources=ResourceSpec(
        cpu=112,
        memory="192GB",
        device=tpu_device("v5litepod-16"),
    ),
    environment=EnvironmentSpec(
        env_vars={"WANDB_API_KEY": "..."},
        extras=["marin:tpu"],
    ),
)

job.wait()
```

---

## Available TPU Types

From the Marin production config (`lib/iris/examples/marin.yaml`):

- **v5e (v5litepod):** 4, 8, 16, 32, 64, 128, 256 chips — zones: `europe-west4-b`, `us-west4-a`
- **v6e:** 4, 8, 16, 32, 64, 128, 256 chips
- **v5p:** 8, 16, 32, 64, 128, 256, 512, 1024, 2048 chips
- **v4:** 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 chips

---

## Architecture Details

### Controller

Central coordinator running on a single GCE VM (`e2-standard-4` in `us-central1-a`). Manages:
- Job scheduling (constraint-based matching of tasks to workers)
- Autoscaling (spins up/down VMs based on pending demand)
- Worker registry (tracks all worker VMs via heartbeats)
- Dashboard (web UI served over SSH tunnel)

### Workers

One per VM. On startup, workers:
1. Register with the controller via gRPC
2. Wipe all previous `iris.managed=true` Docker containers (clean slate)
3. Enter heartbeat loop — controller sends task assignments and kill requests

For TPU jobs, workers auto-configure Docker containers with the right device passthrough (`/dev/vfio`), shared memory, JAX env vars (`JAX_PLATFORMS=tpu,cpu`, `PJRT_DEVICE=TPU`), and multi-host coordination (`JAX_COORDINATOR_ADDRESS`, etc.).

### Scale Groups

Define pools of hardware the autoscaler manages. Each group specifies:
- Zones, VM count per slice, device type/variant
- Min/max slices (autoscaler bounds)
- Priority (lower = preferred when multiple groups match)
- Preemptibility

### Job Names

Canonical format: `/<user>/<job-name>` (e.g., `/michael/train-llama`). Child jobs nest: `/michael/train-llama/eval`. User defaults to `getpass.getuser()`.
