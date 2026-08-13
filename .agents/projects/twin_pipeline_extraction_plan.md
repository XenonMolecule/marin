# Twin-pipeline extraction port (two-stage v5.2 + one-call oc3) into marin

**Status**: planned 2026-07-20
**Branch**: `multi-spec-extraction` (atop `qwen3-useful-classifier`; snapshot tag `pre-multi-spec-extraction-20260720`)
**Source of truth**: `../small-rephraser/HANDOFF_final_pipelines.md` at tag `project-final` / `freeze-final` / `shippable-v5.2`.

## Goal

Port the two frozen, certified extraction pipelines into marin's WARC-extraction infra:
- **System 1 — two-stage v5.2** (primary): filter call → chunked extraction. Specs: `filter`, `extract_head`, `extract_cont`.
- **System 2 — one-call oc3** (fallback): merged judge+extract voting head → continuation chunks. Specs: `main`, `extract_cont`.

Each "system" is a pipeline with its own set of role-named multi-specs. This *replaces* the current single-prompt `ExtractionSpec` model (one system_message + one template, one greedy call/record) with a multi-call, data-dependent per-document pipeline.

## Decision record (settled with user)

| Decision | Choice |
|---|---|
| Execution engine | **Offline vLLM batch-map**, restructured as staged batch passes with a per-document scheduler. NOT an in-worker OpenAI server. |
| Tokenizer | **Qwen3 HF tokenizer only** (Option B): view 24,000 · chunks 18,000. No tiktoken/o200k. One tokenizer everywhere. |
| Faithfulness bar | Port pure machinery verbatim; re-express only the driver as batches. Verify against frozen cert numbers within the cert's own ±0.3 acc / ±0.5 macro / ±2 topQ t=0 batching band. |
| Outer infra | Unchanged: WARC download, one-WARC-=-one-shard, GCS layout `documents/baseline_llm_extraction/{id}/data-{hash}/`, consolidate → dedup → tokenize. |

## Why this slots in cleanly

The marin extraction stack has two layers, and only the inner one changes:

- **Outer (untouched)**: `multi_region_extraction.py` (Iris orchestration across regions/TPU) → `download_and_extract.py` / `run_extract_standalone.py` Zephyr `map_shard` over WARCs → per-region gzipped-JSONL output → `consolidate/` → `dedup_extracted.py` → `tokenize_deduped_extracted.py`. All of this is pipeline-agnostic; it just needs the spec key to become a pipeline id.
- **Inner (replaced)**: today `_process_warc_shard` does `format prompts → llm.generate(batch of 500) → clean → write`. We replace that inner call with a **per-document pipeline scheduler** driven by staged offline batches.

The shared machinery from small-rephraser (`preprocessing.py`, `chunking.py`, `loop_guard.py`, `prompt.py`) is **pure stdlib** (`re`, `bisect`, `collections`) and drops in verbatim. marin already ships `resiliparse` (needed for the two-stage filter view) and the Qwen tokenizer via `transformers`.

## The one genuine mismatch, and how offline handles it

The reference driver is a thread-per-document client (blocking litellm → vLLM OpenAI server) with continuous batching. marin's offline `LLM` gets throughput only from **passing a list to `generate`/`chat`**, and `chat_template_kwargs` (thinking on/off) applies to a whole `chat()` call. So we re-express each pipeline as **cross-document staged batches**, where a stage = one `chat()` call with uniform sampling params:

- thinking-ON calls (filter / one-call voting head) and thinking-OFF calls (extraction chunks) naturally fall into separate batches — matching the enable_thinking-per-call constraint for free.
- The verdict-gates and "extract past chunk 0?" decisions become **partition steps between batches** (exactly the per-doc decisions the user wants control over).
- At temp=0 the model output is a function of (messages, sampling params) only; execution order does not change it. Cross-document batching therefore reproduces the certified outputs up to vLLM's own t=0 batching nondeterminism — which the cert explicitly budgets for (±0.3 acc / ±2 topQ). So batched-offline is faithful *by construction*, not by luck.

### Offline chat adapter (`offline_chat.py`)

Thin wrapper over `llm.chat(list_of_message_lists, sampling_params, chat_template_kwargs={"enable_thinking": bool})` returning per item a small `InferenceResult`-shaped record mirroring the reference's `worker.run_inference`:
- `output_text` = `parse_output_markers(content)` — strip stray `<think>`, pull text between `[[ ## text ## ]]` and `[[ ## completed ## ]]`.
- `reasoning_content` — from vLLM's qwen3 reasoning parser (same parser code offline as server; **PREREQUISITE: confirm vllm-tpu build exposes `reasoning_parser="qwen3"` offline**).
- `finish_reason` — from `RequestOutput.outputs[0].finish_reason`; `"length"` = truncation (reasoning-runaway detection).
- `completion_tokens` — for empty-completion detection + FLOP accounting.

This gives us the reference's return contract without a server.

## New data model (replaces `ExtractionSpec`)

`experiments/baseline_collection/pipelines/pipeline_specs.py`:

- `PipelineType(StrEnum)`: `TWO_STAGE`, `ONE_CALL`.
- Separate classes per type (AGENTS.md: separate classes over boolean flags): `TwoStagePipeline`, `OneCallPipeline`. Each carries its role→prompt-file mapping:
  - two-stage: `filter`, `extract_head`, `extract_cont`
  - one-call: `main`, `extract_cont`
- Shared frozen budget config (one place): `tokenizer="qwen"`, `filter_view_tokens=24000`, `chunk_tokens=18000`, `filter_max_tokens=4096`, `extract_max_tokens=12288`, `loop_threshold=20`, `temperature=0.0`, loop-guard rung-2 `temperature=0.3`, `sentinels=("[NO_USEFUL_CONTENT]","[FILTERED_BY_PIPELINE]","[DOCUMENT_FILTERED]")`, drop marker `[NO_USEFUL_CONTENT]`.
- `PIPELINES` registry keyed by `pipeline_id` (`two_stage_v52`, `one_call_oc3`). **`pipeline_id` is the GCS namespace key** (replaces `spec_id`). No legacy-unprefixed special case — these are new ids.
- Vendored prompt files under `pipelines/prompts/` with a **sha256 manifest** checked at import (handoff §6.5 fidelity gate). Files: `system.txt`, `user.txt`, `two_stage/{filter,extract_head,extract_cont}.txt`, `one_call/{main,extract_cont}.txt`. Note `extract_cont.txt` is byte-identical across both (md5 `c346858990068e686e4ea99264e1d5b2`) — vendor once, reference twice.

## Module layout (new package)

```
experiments/baseline_collection/pipelines/
  preprocessing.py     # port verbatim (aggressive: strip scripts → body slice → strip style/svg/comments → NBSP → strip noisy attrs)
  chunking.py          # port verbatim (natural-breakpoint, headroom=0.98, tier1/tier2/fallback, protected spans)
  loop_guard.py        # port verbatim (THRESHOLD=20, worst_line_amplification, truncate_loops)
  prompt.py            # PromptFormatter (spec_as_input DSPy contract)
  offline_chat.py      # NEW: batched llm.chat → InferenceResult; reasoning parser + finish_reason
  pipeline_specs.py    # NEW: the pipeline registry / data model above
  two_stage.py         # NEW: staged-batch scheduler for TWO_STAGE
  one_call.py          # NEW: staged-batch scheduler for ONE_CALL
  token_budget.py      # qwen codec + view-cap (head+tail keep, "[... middle of page omitted ...]")
  prompts/             # vendored frozen .txt + sha256 manifest
```

## Staged-batch schedulers

### two_stage.py (over one WARC shard of N HTML records)
1. **Filter view**: per record, resiliparse `extract_plain_text(main_content=True, alt_texts=False)`; if <200 chars re-render `main_content=False`; cap at 24,000 qwen tokens (head 12k + `"\n\n[... middle of page omitted ...]\n\n"` + tail 12k).
2. **Filter batch**: one `chat()`, enable_thinking=True, max_tokens=4096, temp=0. Apply `_call` semantics (empty/premature-EOS → per-item retry list; reasoning-runaway `finish_reason=="length"` + empty → error, not drop).
3. **Partition**: `_is_drop` (substring test over sentinels ∪ empty) → dropped docs emit `[NO_USEFUL_CONTENT]` + filter reasoning, done. Kept docs continue.
4. **Chunk**: preprocess(aggressive) + `chunk_html(18000, qwen_count)`. Flatten all kept docs' chunks into one tagged list; chunk 0 → `extract_head`, chunks ≥1 → `extract_cont`.
5. **Extraction batch**: one `chat()`, enable_thinking=False, max_tokens=12288, temp=0.
6. **Loop-guard**: compute amplification per chunk output; amplified minority → rung-1 batch (think-ON, t=0) → still-amplified → rung-2 batch (think-OFF, t=0.3), keep least-amplifying → still-amplified → pure-Python truncation. (No verdict-bearing calls in stage 2 → full ladder allowed everywhere.)
7. **Join** per doc: non-empty, non-sentinel chunk outputs in order, `"\n\n".join`. Emit record.

### one_call.py
1. Preprocess + chunk all docs (no text view, no resiliparse).
2. **Voting head batch**: all docs' chunk 0 → `main` spec, enable_thinking=True, 12288, t=0. Head calls get loop **truncation only**, never retry (verdict-bearing).
3. **Partition**: chunk-0 keep → kept, chunk-0 payload = first segment. chunk-0 drop AND ≥2 chunks → second-voter batch (chunk 1, `main`, think-ON). All-drop → dropped (≤2 calls on dropped multi-chunk docs).
4. **Continuation batch**: kept docs' chunks after the keeping voter → `extract_cont`, think-OFF, 12288, t=0. Full loop-guard ladder (transcription, not verdict).
5. **Join**: keeping-chunk payload + subsequent non-drop payloads, `"\n\n".join`.

## Output record (per doc) — extends current schema

Keep marin's downstream-join fields, add pipeline provenance + the filter verdict (bonus: a clean keep/drop + reasoning signal that feeds the `qwen3-useful-classifier` work on this same branch):
```
text                # final joined extraction ("" or absent when dropped)
pipeline_id         # two_stage_v52 | one_call_oc3
decision            # KEEP | DROP | CONTEXT
drop_marker         # [NO_USEFUL_CONTENT] when dropped
filter_reasoning    # two-stage filter reasoning_content (or one-call head reasoning)
num_chunks
tokens_think / tokens_output   # per-doc sums, for FLOP accounting (as run_extract_standalone does)
url, warc_record_id, warc_file, snapshot   # unchanged, for DCLM joins
```
GCS layout unchanged: `gs://{regional_bucket}/documents/baseline_llm_extraction/{pipeline_id}/data-{warc_hash}/...`. Consolidate/dedup/tokenize unchanged (they operate on `text` + these fields).

## Wiring into the runner

Add a `--pipeline {two_stage_v52,one_call_oc3}` path to `download_and_extract.py` (and `run_extract_standalone.py` for the checkpointed/steal variant): when set, `_process_warc_shard` delegates to the scheduler instead of the flat 500-batch generate. `multi_region_extraction.py` gains `--pipeline` (parallel to today's `--spec`) and threads it into `_build_config_json` + the GCS subdir. Engine kwargs change: `max_model_len=32768` stays; extraction batches size themselves from the chunk list (no fixed 500). Model path → the certified checkpoint (see prerequisites).

## Prerequisites (do before coding the schedulers)

1. **Import the certified checkpoint** `qwen3-8b-rephraser-sft-small-final` into per-region marin buckets. It is NOT in `gs://marin-us-central1/checkpoints/` today (only 0.6B/1.7B rephrasers + the older 8B `v4-193d7b`). Need: locate the source (small-rephraser references a local `models/` path, not gs://), stage into `gs://marin-us-central1/checkpoints/...`, replicate in-region to other extraction regions (NO cross-region egress). **OPEN: does an 8B `small-final` export already exist in any marin bucket / where is the source?**
2. **Confirm vllm-tpu offline reasoning parser**: verify the marin vllm-tpu build exposes `reasoning_parser="qwen3"` (or equivalent) and `chat_template_kwargs={"enable_thinking": ...}` through `llm.chat` offline. Small spike on a v5e/v6e node. If not, fall back to parsing raw `<think>…</think>` from content (reference `parse_output_markers` already does this defensively).
3. **Vendor + sha-pin the 5 frozen prompt files** and the 2 templates; add the sha256 manifest gate.

## Verification protocol (port must pass — handoff §6)

1. **Prompt bytes**: for 3 sample docs, diff marin `PromptFormatter.format(...)` vs reference — byte-identical.
2. **Preprocess/chunk parity**: run both impls over `static/devset_subsets/fast_dev.txt`; chunk counts + boundaries match exactly.
3. **End-to-end devset** (1934 docs, the marin devset already lives in small-rephraser `static/warcs/marin_devset_1934_html/`): score with `scripts/analyze_devset_run.py`. Targets: two-stage ≈ 93.4–93.7 / 90.4–90.8, topQ ≥ 133/134; one-call ≈ 90.4–91.3 / 86.5–87.3.
4. **Degeneration lint** (release blocker): `scripts/lint_degeneration.py <run_dir>` — nothing beyond a few docs at ~×20, no doc ×100+.
5. **Spec integrity**: `sha256sum` vendored prompts vs small-rephraser at tag `project-final`.

## Open scope questions (for user)

1. **Both pipelines now, or two-stage first?** Two-stage is the primary/higher-quality system (~3 macro better); one-call is the cheaper fallback. Suggest: build the shared machinery + two-stage first, verify against certs, then one-call is a small increment (shares all machinery).
2. **Quality bands.** The old marin specs were `high/med/med_low/low_quality`. The frozen systems bake the quality bar into the `filter`/`main` reader-test. Do we (a) ship the two systems as single quality points for now, or (b) also want swappable filter specs to recreate high/med/low bands per pipeline? (b) is a later, cheap extension — filter spec is one swappable field.
3. **Checkpoint source** (prereq 1): is the 8B `small-final` already staged anywhere in GCS, or do I plan a fresh export/import (and from which in-region source)?

## Non-goals / inherited limits (accepted)

Script-only pages (~0.25%), bilingual foreign-chrome false-drops, Wikipedia talk pages (all documented in the reference final reports). No change to consolidation/dedup/tokenize. Not porting the 121-file `prompts/devset/` iteration archive — only the 5 frozen files.
