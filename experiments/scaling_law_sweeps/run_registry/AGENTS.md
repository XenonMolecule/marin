# Run Registry — Agent Notes

Start with `/AGENTS.md` and `experiments/AGENTS.md`; this file is the operational
contract for any agent (Claude session) touching the run registry.

`README.md` documents the commands. **This file is about *when* you must use them.**
The registry only works as a shared source of truth if every session keeps it
current — a stale registry is worse than none, because the human trusts it.

## What this tracks

Natural-epoching curation training runs only. One row per **(method, sample)**.
A `sample` is a specific WARC draw identified by lineage, **not** by WARC count
— see "The identity rule" below. Run the CLI with the repo venv:
`.venv/bin/python registry.py <cmd>` (pyyaml is the only dependency).

## Your contract when working with runs

- **Asked to monitor / babysit a run?** `claim` it first. This stamps your
  session label + the current time so the human can see *you* have it. Then
  `check <id> --status ... --note ...` **every time you actually look at it** —
  the whole point is that `last_checked` reflects reality. If you stop watching,
  `release` it (don't leave a stale claim implying coverage you've dropped).
- **Launched a run?** `add` it immediately with `--status launching/running`,
  `--iris-job`, and `--results-glob`. Don't wait until it finishes.
- **A run finished?** `set <id> --status complete --wandb <url> --checkpoint
  <gs://...>`. The checkpoint location is the thing future-you will hunt for —
  record it.
- **Failed / abandoned?** Set the status honestly (`failed` / `abandoned`) with a
  `--note` saying why. Don't silently leave it `running`.
- **Recount completion truth** with `sync <id>` / `sync --all` (counts result
  JSONs from `results_glob`). Prefer this over hand-editing `cells_done`.

Do these proactively — updating the registry is part of "launch a run" and
"monitor a run", not a separate chore to ask permission for.

## The identity rule (do not violate)

WARC count alone never identifies a run. The `canon-*` samples are one nested
lineage (`head -N` of `baseline_warcs_3000.txt`; `canon-3000w` ≈ the 10k prefix).
An **independent fresh draw at the same N is a different sample**.

- New WARC draw → `add-sample --id <lineage>-<N>w --lineage <lineage> --n <N>
  --manifest <path>` *before* adding its runs.
- At launch, give the new draw's training runs a **distinct results path /
  run-name** so their result JSONs don't overwrite the canon ladder's. Then point
  the registry row's `results_glob` at that distinct path.

## Parallel observer sessions

The `supervisor` label auto-derives from `CLAUDE_CODE_SESSION_ID`
(`claude-<first8>`), so multiple parallel Claude sessions self-identify without
collision. `registry.py whoami` prints yours. Pass `--by` only to impersonate a
human (e.g. `--by michael`). To see only your own runs:
`list --supervisor $(… whoami)`. To find neglected jobs:
`list --status running --stale-hours 2`.

## Hands off the generated file

`REGISTRY.md` is regenerated on every mutation — **never hand-edit it**. Edit
`registry.yaml` (or use the CLI, which is safer). After any manual YAML edit, run
`validate` then `render`.

## Trusting seeded data

Rows seeded from the 2026-05-29 audit carry `status=complete` meaning "ran and
produced results", with `cells_total` unknown — they were **not** verified
against planner totals. Run `sync` and spot-check before treating a seeded
`complete` as ground truth.
