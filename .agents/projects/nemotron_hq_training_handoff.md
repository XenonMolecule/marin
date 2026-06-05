# nemotron_hq WARC-scaling training: launch + monitor

You are launching training for the **nemotron_hq** method (internal name
`nemotron_qhigh`) at multiple WARC subsample scales for the data-curation
scaling-law sweep. Data is already prepped — tokenized caches exist and are
registered. You only need to launch the training coordinators and monitor them.

## What you're launching

Four training coordinators, one per WARC scale:

| Scale  | Method key       | Plans per N | Tokenized cache (already mirrored)                |
|--------|------------------|-------------|---------------------------------------------------|
| 100    | `nemotron_qhigh` | 4           | `baseline_nemotron_qhigh_100warcs`                |
| 500    | `nemotron_qhigh` | 9           | `baseline_nemotron_qhigh_500warcs`                |
| 1000   | `nemotron_qhigh` | 12          | `baseline_nemotron_qhigh_1000warcs`               |
| 2000   | `nemotron_qhigh` | 9           | `baseline_nemotron_qhigh_2000warcs`               |

Plan counts above are approximate (cartesian product of model sizes × budgets
defined for each N in `warc_scaling_plan.py`). The launcher will print the
exact count when it submits.

**Note:** N=3000 is **not yet supported** by the WARC-scaling sweep — adding it
requires registry changes (`WARC_COUNTS`, `_HIDDEN_SIZES_PER_N`,
`_BUDGETS_PER_N` in `warc_scaling_plan.py`, plus a `nemotron_qhigh_3000` entry
in `curation_plan.py`). Don't launch 3k until the user adds those.

## Prerequisites (verify before launching)

1. You're in the `marin` repo root: `/path/to/marin`
2. `uv` is installed and `.venv/` is bootstrapped:
   ```bash
   uv run python -c "import marin"  # should succeed silently
   ```
3. The `nemotron_qhigh_<N>` method keys exist in `curation_plan.py`:
   ```bash
   grep -E '"nemotron_qhigh_(100|500|1000|2000)":' experiments/scaling_law_sweeps/curation_plan.py
   ```
   You should see four `_method(...)` lines, one per N. If any is missing, stop
   and escalate — registry work needs the user.
4. The tokenized caches are present in us-central1:
   ```bash
   for n in 100 500 1000 2000; do
     gcloud storage ls "gs://marin-us-central1/tokenized/baseline_nemotron_qhigh_${n}warcs*/" | head -3
   done
   ```
   Each should print at least one path.

## Launch command (per N)

The launcher is `experiments/scaling_law_sweeps/launch_warc_scaling_sweep.py`.
It must run **inside an Iris CPU-only parent** in us-central1, which submits
TPU children for the actual training. The wrapper is `ray_run.py`.

Run these four commands, one per WARC scale. Use a unique `--job-name`
including a timestamp so coords don't collide:

```bash
# N=100
TS=$(date +%s)
uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \
    -e WANDB_API_KEY ${WANDB_API_KEY} \
    -e HF_TOKEN ${HF_TOKEN} \
    -- iris --cluster marin job run --priority production --no-wait \
        --memory 4GB --cpu 4 \
        --job-name nemotron_hq-100-coord-${TS} \
        -- python experiments/scaling_law_sweeps/launch_warc_scaling_sweep.py \
            --methods nemotron_qhigh --n-warcs 100 \
            --child-priority interactive

# N=500
TS=$(date +%s)
uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \
    -e WANDB_API_KEY ${WANDB_API_KEY} \
    -e HF_TOKEN ${HF_TOKEN} \
    -- iris --cluster marin job run --priority production --no-wait \
        --memory 4GB --cpu 4 \
        --job-name nemotron_hq-500-coord-${TS} \
        -- python experiments/scaling_law_sweeps/launch_warc_scaling_sweep.py \
            --methods nemotron_qhigh --n-warcs 500 \
            --child-priority interactive

# N=1000
TS=$(date +%s)
uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \
    -e WANDB_API_KEY ${WANDB_API_KEY} \
    -e HF_TOKEN ${HF_TOKEN} \
    -- iris --cluster marin job run --priority production --no-wait \
        --memory 4GB --cpu 4 \
        --job-name nemotron_hq-1000-coord-${TS} \
        -- python experiments/scaling_law_sweeps/launch_warc_scaling_sweep.py \
            --methods nemotron_qhigh --n-warcs 1000 \
            --child-priority interactive

# N=2000
TS=$(date +%s)
uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \
    -e WANDB_API_KEY ${WANDB_API_KEY} \
    -e HF_TOKEN ${HF_TOKEN} \
    -- iris --cluster marin job run --priority production --no-wait \
        --memory 4GB --cpu 4 \
        --job-name nemotron_hq-2000-coord-${TS} \
        -- python experiments/scaling_law_sweeps/launch_warc_scaling_sweep.py \
            --methods nemotron_qhigh --n-warcs 2000 \
            --child-priority interactive
```

### What the flags mean

- `--cluster us-central1`: the ray_run wrapper picks this cluster's config for
  the SSH tunnel.
- `--no_wait`: ray_run returns immediately after submission instead of
  attaching.
- `-e WANDB_API_KEY ... -e HF_TOKEN ...`: env vars the children need. Pass
  them as **separate `-e KEY VALUE` pairs**, not `--env_vars KEY=VAL`.
- `--priority production` on the parent: the coord itself must not get
  preempted (it babysits all children).
- `--memory 4GB --cpu 4`: minimal coord footprint.
- `--child-priority interactive`: children run at interactive priority. The
  parent must stay at `production`; children at `interactive` is correct.

### CRITICAL: never pass `MARIN_PREFIX`

Do **not** add `-e MARIN_PREFIX gs://...` to either the ray_run wrapper or the
iris job. That env var propagates to TPU children, makes them mis-identify
their region, and breaks the region-lock logic. The launcher resolves regions
correctly on its own.

## After launching

The launcher prints lines like:

```
Submitting plan curation-nemotron_qhigh_100-expWARC_natural-9e+18-d512-L8-B8 ...
```

and the iris parent stays alive holding a long sleep while children run. A
healthy launch will list the parent in `iris job list`:

```bash
uv run iris --cluster marin job list --state RUNNING | grep nemotron_hq
```

You should see one row per N that you launched.

## Monitoring

### Per-N progress (results JSONs)

Each completed training plan writes a result JSON to
`gs://marin-us-central1/metadata/data_curation_warc_scaling_results/`. To see
how many of the N=<N> plans for nemotron_qhigh have landed:

```bash
gcloud storage ls "gs://marin-us-central1/metadata/data_curation_warc_scaling_results/" \
  | grep -c "curation-nemotron_qhigh_<N>-expWARC"
```

Replace `<N>` with the scale you're checking. The launcher's startup log shows
the total plan count for each N; you're done when these match.

### Active child jobs

```bash
uv run iris --cluster marin job list --state RUNNING --prefix /<your-user>/nemotron_hq-<N>-coord-
```

(Use `--prefix` for accurate per-coord listing — the bare `list` is
truncated/paginated and can lie.)

### When a child fails

Children inherit zephyr's retry behaviour. If a single plan keeps failing,
download the full logs:

```bash
uv run iris --cluster marin job logs <full-child-job-id> > /tmp/child.log
```

Common failure modes worth flagging up rather than fixing yourself:

- `Insufficient memory ... need <X>GB, available <Y>GB`: cluster contention,
  not a real failure. The scheduler will retry. Leave it alone unless a child
  has been pending > 6 hours.
- `Preempted by /<other-user>/...`: someone else's higher-priority job evicted
  ours. Same — leave it; zephyr will retry.
- vLLM/TPU lockfile / `safetensors` errors during eval: known recurring
  issue. Escalate with the failing child's log.

## Rough timing expectations

- A coord parent stays alive for the entire duration of its TPU children.
  Expect N=100 to complete in a few hours, N=2000 in 1–2 days depending on
  cluster contention.
- Don't kill a coord just because a single child failed and retried — that's
  normal. Only escalate parents that haven't had any child progress in 12+
  hours.

## What to escalate

- All four coords need to be relaunched (e.g. you killed the wrong one).
- A coord parent in iris is `FAILED` or `KILLED` (not `RUNNING`).
- Many children failing with the same non-transient error (e.g. import
  errors, schema mismatches).
- Anyone asks you to launch N=3000 — that needs registry work first.

## What NOT to do

- Don't run `--methods all` — that would launch every method, not just
  nemotron_qhigh.
- Don't pass `--child-priority production` — children at production preempt
  other users' work and will be flagged.
- Don't set `MARIN_PREFIX` in the env (see above).
- Don't touch the shared iris cluster (no `iris cluster stop` etc.).
- Don't kill `/runner/...`, `/wmoss/...`, etc. — only your own coords.
