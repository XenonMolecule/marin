# Run Registry

A living registry of **natural-epoching** curation training runs — what has run,
where its results/checkpoints landed, and who is currently watching it.

- **`registry.yaml`** — source of truth (git-tracked, hand-editable).
- **`registry.py`** — the CLI. Mutations auto-regenerate the Markdown view.
- **`REGISTRY.md`** — generated human view. **Do not hand-edit.**

Run the CLI with the repo venv:

```bash
cd experiments/scaling_law_sweeps/run_registry
python registry.py list                 # or: ../../../.venv/bin/python registry.py list
```

## The identity rule (read this first)

A run is keyed on **(method, sample)**, never on WARC count alone. A `sample` is
one specific WARC *draw*:

- **`canon-{100,500,1000,2000,3000}w`** — the canonical nested ladder. These are
  deterministic `head -N` prefixes of `experiments/distill/baseline_warcs_3000.txt`,
  so `100 ⊂ 500 ⊂ 1k ⊂ 2k ⊂ 3k`, and `canon-3000w` is ~the prefix of the 10k pool.
- **`indepA-3000w`** — a *separate, independent* 3000-WARC draw. Same N as
  `canon-3000w` but a different lineage, so it is tracked as a distinct sample.

When you launch an independent draw, **register a new sample** (`add-sample`) and
give its runs a distinct results path/run-name at launch so result JSONs don't
collide with the canon ladder's.

## Supervision (who's watching what)

`claim` / `check` stamp `last_checked` (UTC, now) plus a `supervisor` so we can
both see, at a glance, "claude checked it 12m ago" vs "⚠ 26h ago — go look."

The supervisor label auto-derives from `CLAUDE_CODE_SESSION_ID`, so **parallel
Claude observer sessions self-identify distinctly** (e.g. `claude-6329ae0a`).
Pass `--by <name>` to override (humans should — e.g. `--by michael`).

```bash
python registry.py whoami                              # shows this session's label
python registry.py claim resiliparse_dedup__indepA-3000w   # auto: claude-<id8>
python registry.py check resiliparse_dedup__indepA-3000w --status running --note "cells 12/40, healthy"
python registry.py list --supervisor claude-6329ae0a       # what this session watches
python registry.py list --status running --stale-hours 2   # running but not checked in 2h
python registry.py release resiliparse_dedup__indepA-3000w
```

## Recording a new run

```bash
# 1. (once per draw) register the WARC sample
python registry.py add-sample --id indepA-3000w --lineage indepA --n 3000 \
    --manifest experiments/distill/independent/baseline_warcs_3000_indepA.txt \
    --desc "independent random 3000-WARC draw"

# 2. add the run, then fill in details as they land
python registry.py add --method resiliparse_dedup --sample indepA-3000w --status launching \
    --iris-job /michaelryan/... --results-glob 'gs://.../curation-...-*.json'
python registry.py set resiliparse_dedup__indepA-3000w --status complete \
    --wandb https://wandb.ai/... --checkpoint gs://marin-us-east5/checkpoints/...
```

## Keeping completion counts honest

`cells_done` is the number of completed result JSONs. `sync` recomputes it live
from `results_glob`:

```bash
python registry.py sync --all          # or: sync <run-id ...>
```

Seeded entries (`status=complete`, from the 2026-05-29 audit) mean "ran and
produced results" — they were **not** re-verified against planner totals
(`cells_total` is unknown). Run `sync` and spot-check before fully trusting.

## Training progress % (optional, occasional)

`progress` queries **wandb** for the mean per-cell training completion across the
planned grid (`min(1, _step/num_train_steps)` averaged over `cells_total`, so an
unstarted cell counts as 0 and one giant aspirational cell can't dominate). It
writes `pct_complete` + `last_progress_synced`, shown as the `train%` column.

```bash
WANDB_API_KEY=... python registry.py progress <run-id ...>   # target specific rows
```

This hits wandb (one query + per-run summary loads), so it's **heavier than
`sync`** — run it occasionally, **not** on every check, and prefer naming the
active rows over `--all` (which would query every historical row too). The
`wandb` field on each row links straight to that sweep's dashboard.

## Other commands

```bash
python registry.py show <run-id>       # full record for one run
python registry.py validate            # schema integrity (FK, statuses, dup ids)
python registry.py render              # regenerate REGISTRY.md
```
