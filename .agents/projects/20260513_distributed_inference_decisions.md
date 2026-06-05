# Distributed Inference — Decisions to Make

*For the meeting — Michael Ryan, 2026-05-13.*

These are concrete forks. Each has a small number of options, real stakes,
and is small enough to resolve in five minutes. Walk down the list, force a
choice on each one. No philosophical "what is a workload" questions.

---

## Engine API

### 1. Does `engine.generate(prompts)` return strings or structured results?
- **A.** `list[str]` — minimal, matches vLLM's basic API.
- **B.** `list[GenerationResult]` with `input_tokens`, `thinking_tokens`,
        `response_tokens`, `finish_reason`, `raw_text`.

**At stake:** FLOP accounting and reasoning-token splitting are application
concerns under A and engine concerns under B. Today we reinvent it per app.

### 2. Who owns chat-template formatting?
- **A.** Engine — caller passes `messages: list[dict]`; engine applies the
        tokenizer's chat template and pre-tokenizes.
- **B.** Application — engine takes already-tokenized prompts
        (`TokensPrompt`), application does templating.

**At stake:** A standardizes prompt construction across applications; B
keeps the engine smaller and lets weird applications skip templating.

### 3. Is `<think>` token splitting an engine concern?
- **A.** Yes — engine knows the tokenizer, so it natively returns
        `thinking_tokens` / `response_tokens`. Non-thinking models return 0.
- **B.** No — engine returns raw token counts; applications split if they
        care.

**At stake:** A makes Q1's `GenerationResult` cleaner. B keeps the engine
agnostic to model family.

---

## Claim primitive

### 4. Shard-level claims, or shard + sub-shard from day one?
- **A.** Just shard (WARC). Steal mode is a bolted-on second mode like
        today.
- **B.** Native `(shard_id, sub_shard_id)` from the start. Steal mode
        becomes "claim sub-shards in reverse order."

**At stake:** A is simpler to ship and matches what most workloads need.
B has fewer special cases later, and the worker-flow walkthrough is half
as complicated.

### 5. On `_done` write, do we delete `_claimed`?
- **A.** Leave it (status quo).
- **B.** Delete it as the final act of `_done`.

**At stake:** B avoids the cost of every worker reading a defunct
`_claimed` for every done WARC during cross-region scans. A keeps an audit
trail of who finished what.

---

## Cross-region

### 6. Do outputs live in the writer's region forever, or mirror to one canonical region on `_done`?
- **A.** Stay in writer's region (status quo). Readers (training, dedup,
        eval) pay cross-region read costs.
- **B.** On `_done`, copy outputs to one canonical region.

**At stake:** Writer pays once on B; readers pay every time on A. Depends
on read frequency. We already mirror checkpoints — same playbook.

### 7. Does the global `_completed/` registry stay in one bucket, or get mirrored?
- **A.** One bucket (status quo, `gs://marin-us-central1/...`). Simple,
        cheap, single point of failure.
- **B.** Per-region copies with periodic merge.

**At stake:** A bus-factor-of-one for the fast-path skip. Has not bitten
us; would only matter during a us-central1 outage.

---

## Steal mode

### 8. Keep work-stealing, or drop it for oversubscription?
- **A.** Keep — flattens the endgame long tail; we already have it.
- **B.** Drop — run 1.5× workers and accept some duplicate compute.

**At stake:** Real cost. Steal mode is the most complex code path in the
system. *Before deciding: measure how much wall time it actually saves on
the current fleet.* If <10%, drop it.

### 9. If we keep it, do steal-mode patience tiers belong in user config?
- **A.** Hard-coded — today's `500 / 1000` WARC thresholds, `10 / 50 / 100`
        patience values.
- **B.** Config knobs on the library.

**At stake:** B is more flexible; A is one fewer thing to tune wrong.

---

## Workload + IO

### 10. Manifest format — plaintext URLs, or structured records?
- **A.** Plaintext, one URL per line (status quo).
- **B.** Structured (e.g., JSONL with `id`, optional `priority`, optional
        per-shard `hints`).

**At stake:** B enables priority queues, per-shard skip lists, and
metadata-carrying retries. A is two lines of code.

### 11. Does the IO contract carry `record_id` end-to-end?
- **A.** No — record order within a batch is enough (status quo).
- **B.** Yes — every record has a stable id; output records carry it;
        partial-batch retries are idempotent.

**At stake:** B makes per-record retry possible. A means a corrupted batch
needs full reprocessing.

---

## Launcher

### 12. Adaptive controller — Python parent process or Iris-native autoscaler?
- **A.** Python parent (status quo). Explicit, debuggable, one more thing
        to babysit.
- **B.** Iris-native autoscaler config. Declarative, but requires Iris
        features that may not exist (priority-aware backoff, preempted-child
        signal).

**At stake:** A is what we have and it works. B is right long-term but is
an Iris ask, not an inference-library ask.

### 13. Should the launcher cap `max_count` from workload size?
- **A.** No — user passes `--max-count`.
- **B.** Yes — library computes a sane cap from `(work_remaining,
        target_wall_time, observed_throughput)`.

**At stake:** B prevents launching 200 workers for a 50-WARC test job and
auto-shrinks when work is nearly done.

---

## Spec / prompt registry

### 14. What does a "spec" contain?
- **A.** Just the prompt (status quo).
- **B.** Prompt + model checkpoint + sampling params (a full reproducibility
        bundle).

**At stake:** B is the right unit for "re-run spec X six months from now."
A bakes the checkpoint into worker code.

### 15. `spec_id` — human-named or content-hashed?
- **A.** Human-named, with "bump the id if you edit the prompt"
        discipline (status quo, `low_quality` / `med_quality` / etc.).
- **B.** Content-hashed — editing the prompt without renaming is
        impossible because the path changes.

**At stake:** A is human-friendly and supports namespaces like
`low_quality`. B is bulletproof against accidental prompt edits silently
contaminating output.

---

## Scope of v0

### 16. What ships in the first library release?
Pick one as the v0 commitment:
- **A.** Engine + Coordinator. Workload, IO, launcher stay as user code.
- **B.** Engine + Coordinator + Workload (with plaintext-manifest default).
- **C.** Everything including the adaptive launcher.

**At stake:** A is shippable in two weeks. C is shippable in two months
and locks us into our current launcher's quirks. B is the middle.

---

## Empirically open — measure before deciding

These came up above and need data, not opinion:

- **Steal mode value.** What fraction of total wall time has steal mode
  saved vs. running the same compute without it? (Need a re-run with the
  feature flagged off, or a retrospective analysis of `_stealing/` claims
  vs. fleet wall time.)
- **Cross-region scan cost.** How many GCS list calls per WARC, and is
  that the bottleneck or just background noise? (Probably noise, but worth
  confirming before optimizing it away.)
- **Scaling ceiling.** At what fleet size does GCS-as-coordinator stop
  working? 200 workers is fine. 500? 2000?
