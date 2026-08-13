# 998M three-way system-prompt ablation @ 9e19 FLOPs

## RESULT (all three complete 2026-08-05 ~02:00 UTC)

**Prepending these system prompts HURT.** Arm A is worse on every eval set — not
just the aggregate — and the controls make that credible.

| | A sysprompt | B token-matched | C doc-matched |
|---|---|---|---|
| step | 56,496 | 56,515 | 55,563 |
| eval/loss | 2.9419 | **2.8368** | 2.8397 |
| eval/bpb | 1.0102 | **0.9705** | 0.9716 |
| eval/macro_bpb | 1.1211 | **1.0441** | 1.0451 |
| lima/bpb | 0.8718 | **0.8395** | 0.8404 |
| paloma/bpb | 1.0198 | **0.9778** | 0.9789 |
| uncheatable/bpb | 0.9448 | **0.9222** | 0.9234 |

- A is **+0.0397 bpb** vs the best baseline; loses on paloma (+0.042), lima
  (+0.032) and uncheatable (+0.023). No eval set where conditioning wins.
- **B vs C differ by 0.0011 bpb** despite 188k extra docs — the pipeline's noise
  floor. A's gap is ~36x that, so it is not measurement noise.
- C used 1.65% fewer tokens than B for 0.0011 bpb, so the token deficit costs
  almost nothing. A's deficit is about what the tokens ARE, not how many.

Reading: ~1.7% of A's budget went to short formulaic prefixes ("The following
document informs about…", 16 tokens) that carry little signal and never appear at
eval time. Note the earlier [S][D] project used 457-char prompts (+6.8% tokens);
these are 101-char. Whether richer prompts behave differently is untested.

Caveats: all three ran at ~8.9e19 not exactly 9e19 (uniformly — see below), and
this is one seed at one scale. The 1.8e20 run would show whether the gap widens.

Evals are directly comparable — the training cache contributes NO validation
split (verified: no eval key mentions the training set), so all arms are scored
on identical held-out data with no [S] tokens.


Three 998M runs at the frozen `d1536 @ 9e19` grid cell, differing only in whether
generated system prompts are prepended and how the control is matched.

## The cell (frozen — do not recompute)

`PLAN_GRID.md:76` → `| 1536 | 16 | 12 | 6144 | 64 | 57,022 | 1.49e+10 | 998M | 1 | v4-16 | v5p-16 |`

| field | value |
|---|---|
| params | 998,032,896 (hidden 1536, layers 16, heads 12, kv_heads 12, intermediate 6144) |
| arch | Qwen3 + `Llama3RotaryEmbeddingsConfig`, tokenizer `meta-llama/Meta-Llama-3.1-8B` (vocab 128256) |
| seq_len | 4096 |
| batch | 64 |
| steps | 57,022 → 14.948B tokens |
| lr | 0.0036842469 |
| adam_lr | 0.00026827743 |
| epsilon | 4.5236753e-08 |
| beta1 / beta2 | 0.9 / 0.9999 |
| z_loss | 1e-07 |
| grad clip | 0.1 |
| schedule | linear, warmup 0.1, decay 0.2, min_lr_ratio 0.0 |
| TPU | **v5p-16** (2 VMs, multi-host → coscheduling `group_by=tpu-name`, replicas=1) |
| TP | 1 |

Hyperparameters are the frozen `completed_adamh` ("delphi") values for this cell —
copied verbatim, never recomputed, per `curation_plan.py:903-908`.

## The three runs (user decisions 2026-08-04)

Corpus for all three: the **DCLM-30B corpus we just labeled**
(`gs://marin-us-central1/sysprompt_pretrain/dclm30b/corpus/`, 23,906,067 docs /
30.000B llama3 tokens). Extra docs for the token-matched arm come from the SAME
corpus (unlabeled blocks), so distribution is identical across arms.

| run | docs | [S] | cache tokens | steps | epochs | FLOPs |
|---|---|---|---|---|---|---|
| **A** sysprompt | N | yes | ~14.948B | 57,022 | 1.00 | 9.0e19 |
| **B** token-matched | N + ~196k | no | ~14.948B | 57,022 | 1.00 | 9.0e19 |
| **C** doc-matched | N (same as A) | no | ~14.696B | ~56,065 | 1.00 | 8.85e19 |

**User chose equal EPOCHS (1.00 each), not equal steps** — C trains ~1.7% fewer
steps rather than repeating data. C is therefore not strictly isoFLOP; that is
intentional, so no arm ever sees a document twice.

Steps are set per-arm as `round(D_obs / (64 * 4096))` — exactly one epoch of that
arm's own cache, mirroring the 447M [S][D] precedent
([[project_sysprompt_dclm_training]]).

## Measured constants

- system prompt = **16.0 llama3 tokens** mean; wrapped `[S]` block = **21.0**
  (`<|start_header_id|>system<|end_header_id|>\n\n{S}<|eot_id|>`), i.e. **+1.68%**
  over a 1,254.9-token average doc. Much shorter than the earlier [S][D] project's
  457-char prompts (+6.8%) — these are one-line descriptions (13.4 words).
- **The corpus `n_tokens` field IS the llama3 count** — verified on 300 docs, sum
  matched exactly (ratio 1.0000). So doc selection can hit token budgets precisely
  from metadata alone, with no pre-tokenization pass.

## Data availability (the gating constraint)

Labeled blocks: 6,528 (SC 0-3522 + marin 4000-11953, both fragmented) = 13.06M
docs. Need ~11.68M → ~12% headroom, so the fragmentation is affordable.

Corpus parts in GCS at planning time: us-central1 119-239, us-east1 40-119,
us-east5 119-181, us-central2 40-79. **Parts 0-118 were missing from us-central1**,
the training region (v5p lives in us-central1-a / us-east5-a). Relayed 0-118 from
SC (25.3 GB, free direction) so the whole corpus is region-local — training and
tokenization must never read cross-region.

## Build steps

1. **Relay** parts 0-118 → us-central1. (free; SC→GCS)
2. **Select docs**: walk labeled blocks in block order, accumulate `n_tokens`
   until arm A's `[S][D]` total reaches 14.948B. Freeze that doc-id list — arms A
   and C share it exactly.
3. **Materialize JSONL** (in-region job, us-central1):
   - A: `conditioned_text` = wrapped [S] + D
   - C: `text` = D, same doc ids
   - B: `text` = D, A's docs + extra unlabeled blocks to match A's token total
4. **Tokenize** three Levanter caches (llama3).
5. **Register** three methods in `curation_plan.METHODS` + `_D_OBS_DEFAULTS`
   (purely additive; cannot affect running sweeps).
6. **Launch** three children via `run_curation_train_standalone.py`,
   `--experiment-tag expFM_natural` (no slicing), pinned us-central1, v5p-16.

## Tooling (all scale-generic — a new budget is a flag, not an edit)

| script | role |
|---|---|
| `build_sysprompt30b_dataset.py index` | one reusable pass → per-block doc/token counts |
| `build_sysprompt30b_dataset.py emit --budget --hidden-dim --tag` | picks blocks, writes that scale's shards |
| `tokenize_sysprompt30b.py --tag` | three llama3 caches (A/B/C) |
| `launch_sysprompt30b_ablation.py --tag [--budget --hidden-dim]` | submits all three arms in parallel |

The token target and the TPU shape both come from `completed_adamh` /
`_candidate_for_fixed_model` — the same source the trainer reads — so dataset,
hyperparameters and hardware cannot drift from each other. TPU scales on its own:
9e19 → v5p-16, 1.8e20 → v5p-32, 3e20 → v5p-64.

`emit` HARD-FAILS rather than building an undersized dataset, and distinguishes
the two causes because they have different fixes:

- arm A short → not enough LABELED docs → generate more system prompts
- arm B short → not enough DOCS at all → add documents to the corpus

`--dry-run` answers "can I do this scale yet?" without writing anything.

## Remaining steps (after tokenization)

1. Read each cache's `train/.stats.json:total_tokens` — authoritative, accounts
   for BOS/EOS. Do NOT use the build-time estimates.
2. Add each cache hash to `curation_plan._D_OBS_DEFAULTS`, then register
   `sysprompt30b_998m_9e19_{A,B,C}` via `_method(..., pin_region="us-central1")`.
   Purely additive — cannot affect running sweeps.
3. `launch_sysprompt30b_ablation.py --tag 998m_9e19` (interactive priority,
   granted by the user 2026-08-04).

Steps per arm are `floor(d_obs / (batch * seq_len))` — exactly one epoch, so no
arm ever repeats a document.

## Verified before launch

- **Positional doc→block mapping**: part 150 line 0 == gen block 7500 idx
  15,000,000, same `doc_id`; 239*100,000 + 6,067 == 23,906,067. The builder
  asserts `doc_id` per record, so drift fails loudly instead of silently pairing
  a system prompt with the wrong document.
- **`conditioned_text` == wrapper + S + text** exactly (two-part test build).
- **Multi-host jax init fix is present AND committed** (`jax_init.py:142`).
  v5p-16 is 2 VMs; without it every worker dies on `multihost_broadcast_sync`.
- **All 240 corpus parts byte-verified** against `part_sizes.txt`. One part (118)
  silently failed to copy while the command reported success — presence is not
  validity, so verify sizes after every copy.
- Validation caches (paloma / uncheatable_eval / LIMA) present in us-central1.

## The malformed gen record (emit failure, 2026-08-04)

`emit` died with `JSONDecodeError: Unterminated string at line 1 column 72`. The
job then sat in `running` with a flat shard count for ~40 minutes, which reads
exactly like stragglers finishing — the monitor watched artifacts but not job
state, so a failed job and a slow one were indistinguishable. **Watch job state
alongside artifacts.**

Cause, after ruling out the obvious: **not corruption**. All three copies of
`block-000073.jsonl.zst` (SC, `sc_gen`, `gen_all`) are byte-identical with
matching md5 `07ee18b9…`. SC's generator writes an error record when a doc fails
to parse and embeds the raw document text — literal newlines included — in the
`error` field. That breaks JSONL: one record spans 4 lines, none of which parse.
2,000 records occupy 2,003 lines. Exactly 1 block in 6,528.

Joining the fragments cannot repair it — a JSON string may not contain a raw
newline — so the reader now SKIPS unparseable lines, bounded by
`MAX_BAD_LINE_FRACTION = 0.01` so genuine corruption still fails loudly. Nothing
real is lost: such records carry `error` and no `system_prompt`.

**The general lesson: size and md5 verification cannot see a malformed payload.**
This morning's failure was truncated parts (size caught it); this one was a
byte-perfect file with invalid contents. Different failure class, so
`build_sysprompt30b_dataset.py verify` parses every record rather than trusting
sizes. Run it after any change to the gen set.

Note `verify` flags `block-011953` as having 67 rows — that is CORRECT, it is the
corpus's final block (23,906,067 == 11953*2000 + 67), not a defect.

## LAUNCHED 2026-08-04 07:40 UTC — all three arms training

Coordinator `/michaelryan/sp30b-998m-9e19-coord` (keep-alive). Caches, real
`total_tokens` from each `.stats.json`, and the resulting one-epoch steps:

| arm | cache | d_obs | steps | vs A |
|---|---|---:|---:|---|
| A sysprompt | `sysprompt30b_998m_9e19_A-94f799` | 14,810,496,434 | 56,497 | — |
| B token-matched | `sysprompt30b_998m_9e19_B-64d3a5` | 14,815,377,151 | 56,516 | **100.03%** |
| C doc-matched | `sysprompt30b_998m_9e19_C-c94a3d` | 14,565,835,827 | 55,564 | **98.35%** |

The ratios were not imposed — they fell out of independent tokenization, and
C/A = 98.35% is exactly the measured `[S]` overhead. Good end-to-end check that
the token accounting is sound.

Caches came in **0.93% under** the corpus's own `n_tokens` estimate, uniformly
across arms. With the equal-epochs choice that puts all three at ~8.92e19 rather
than exactly 9e19 — uniform, so the comparison is unaffected, but the runs are
not literally 9e19.

### Monitoring these runs — do NOT grep for `workers-a0`

Training children are **direct iris TPU jobs**: their tasks are `.../0` and
`.../1` (the two v5p-16 VMs). They have no zephyr `workers-a0` actor group, so a
monitor looking for one reports `arms_with_tpu=0` while all three arms train
happily. Compounding it, an arm can sit in task state **1 (queued)** at the top
level while training — a `--state running` filter drops it and the arm looks
*gone*.

Both mistakes were made here, and produced a confident report that no arm had
ever acquired a TPU and one had disappeared. The truth was in
`checkpoints/<run>/checkpoints/eval_metrics.jsonl` the whole time.

**Watch this instead** — step and loss from the run's own eval log:

```
gs://marin-us-central1/checkpoints/isoflop-curation/\
curation-sysprompt30b_998m_9e19_{A,B,C}-expFM_natural-9e+19-d1536-L16-B64/\
checkpoints/eval_metrics.jsonl
```

### Reading the results

Arm A's `eval/loss` is NOT directly comparable to B and C: A's sequences contain
the `[S]` tokens, so it is scored on a different distribution and some gap is
expected whether or not conditioning helps. Use the shared eval sets
(`eval/lima/bpb`, paloma) for the cross-arm comparison. B vs C is the clean
internal check — they differ only by 188k extra documents.

## The real constraint

The account is **1110% over its Iris budget**, so the effective priority band is
BATCH regardless of what is requested, and us-central1's v5p pools are saturated
(v5p-16: 2/2 busy; v5p-8: 26/26). Three arms need 6 v5p VMs. Runs are
preemption-safe (100 retries, 15-min durable checkpoints,
`LEVANTER_PORTABLE_TPU_CACHE=1`), so a late start still makes progress — but the
only levers that change acquisition odds are raising the budget or shrinking the
extraction fleet. See [[feedback-iris-budget-demotes-priority-band]].

## Gotchas carried in from prior work

- `expFM_natural` must be the tag — `needs_slicing` only triggers on expB/expC, so
  Levanter slicing stays off. Setting a target budget here caused a documented
  memorization bug (`run_curation_train_standalone.py:898-901`).
- The child resolves the cache path **region-locally** from the method name;
  `_assert_all_components_local` hard-fails on a non-local component. Validation
  caches (paloma / uncheatable_eval / LIMA) must exist in us-central1.
- Multi-host v5p-16 needs the iris `jax_init` fix (already applied) or every worker
  dies with `multihost_broadcast_sync requires jax distributed client`.
- Set `LEVANTER_PORTABLE_TPU_CACHE=1` so a preempted run warm-hits the compile
  cache on a new slice instead of a ~76 min cold recompile.
