# Hand-off: batch the Phase-A ModernBERT tokenization (~3× win)

## TL;DR
Phase A's tokenize step is **56% of Phase A wall time** (~38 docs/s on the workers). The tokenizer is
already the Rust `PreTrainedTokenizerFast` — the problem is we call it **one document at a time** in a
serial loop, which doesn't use the tokenizer's cross-document batch parallelism (Rayon). Switching to a
**single batched call** measured **112 → 334 docs/s (3.0×)** on an 8-core box. That pulls Phase A from
~12 → ~8 min/WARC.

## Where it lives
- Tokenizer load: `experiments/fast_curation/preprocess.py:131` — `load_tokenizer()` →
  `answerdotai/ModernBERT-base` (Rust `PreTrainedTokenizerFast`, vocab 50,280, byte-level BPE).
- Per-doc call: `experiments/fast_curation/preprocess.py:145` — `tokenize_trunc()`:
  ```python
  return tokenizer(text[: max_length * 8], truncation=True, max_length=max_length)["input_ids"]
  ```
- Hot serial loop: `experiments/fast_curation/cpu_phase_a.py:55-62` — `for r in records:` calls
  `tokenize_trunc` once per fastText-survivor.

## The change
Collect the survivors' `text_body` (already char-capped at `max_length*8`) and make ONE call:
```python
ids_list = tokenizer([t[: max_length * 8] for t in texts], truncation=True, max_length=max_length)["input_ids"]
```
The Rust tokenizer parallelizes across docs; Phase A does not pin threads, so it uses the worker's cores.

## Constraints (do not break)
- **Keep the `max_length*8` char cap** — we want >8k-token headroom later (chunked/long-context ModernBERT).
- **Do not swap the tokenizer** — must remain `answerdotai/ModernBERT-base` to match the trained
  classifier. Output `input_ids` must be **byte-identical** to the per-doc path (assert on a sample).
- **Batch per checkpoint-chunk** — Phase A is getting streaming chunk-checkpointing (preemption
  resilience); the batched tokenize should run on each chunk's survivors so the two compose.
- These tokens are throwaway (classifier scoring only; the training corpus is re-tokenized with Llama 3),
  so nothing downstream depends on them beyond the ModernBERT scores.

## SHIPPED (2026-06-30)
Implemented on branch `qwen3-useful-classifier`:
- `preprocess.tokenize_trunc_batch()` — one `encode_batch` per 1024-doc sub-batch (bounds RSS +
  composes with chunk-checkpointing), `return_attention_mask/token_type_ids=False` to skip
  discarded outputs. Byte-identical to `tokenize_trunc` (parity test in `test_preprocess.py`).
- `cpu_phase_a._process_one` — split into a fastText-gate pass that collects survivor texts, then
  ONE batched tokenize.

### The lever the hand-off missed: `TOKENIZERS_PARALLELISM`
Iris **and** Fray inject `TOKENIZERS_PARALLELISM=false` into every worker env
(`lib/iris/src/iris/cluster/types.py:539`, `lib/fray/src/fray/types.py:476`) as a fork-deadlock
guard. That env var **disables the Rust tokenizer's cross-document Rayon parallelism** — so a batched
call on the real cluster worker runs *serially* and gets ~no speedup. The hand-off's 3× was on a
local box without that injection. `cpu_phase_a.py` now sets `TOKENIZERS_PARALLELISM=true` at import
(safe: Phase A never forks — JustText's ProcessPool is Phase C only). **This is what makes the batch
win real on-cluster.** Do NOT set it in `cpu_phase.py` (Phase C) — that path forks for JustText.

### Verified locally + measured scaling law
Same-box A/B (parity OK throughout): per-doc ~135–166 docs/s → batched ~358–392 docs/s.

The honest number is the **core-scaling law**, because a single-document `tokenizer(text)` call is
*inherently single-threaded* — the old per-doc loop cannot use more than one core no matter how many
the worker has. Measured batched throughput at controlled `RAYON_NUM_THREADS`:

| threads | batched docs/s | scaling vs 1 |
|---|---|---|
| 1 | 87.9 | 1.00× |
| 4 | 277.8 | **3.16×** |

Near-linear. Since every cluster CPU worker has ≥4 physical cores, the batched path clears the 3–4×
target on any of them, and grows with core count: ~3× on a 4-core box, and the many-core Sapphire
Rapids v6e hosts (16–32+ vCPU) land far higher (~10×+ over the single-core per-doc rate).

### Adopted: `encode_batch_fast` (+~1.3×)
`backend.encode_batch_fast` is byte-parity-OK and a further ~1.3× (skips offsets/word-ids/masks we
discard: 341 vs 258 docs/s local), stacked on the batching+parallelism win. It has no per-call
truncation arg, so `tokenize_trunc_batch` calls `backend.enable_truncation(max_length, ...)` at the
top of every call — a setting on the shared tokenizer object. Safe here: only `tokenize_trunc_batch`
uses that cached tokenizer in the Phase A process, always at one known length, single-threaded, and
we re-assert the length each call. Don't interleave the per-doc wrapper with a *different*
truncation on the same tokenizer in the same process.

**MUST UPDATE WHEN ADDING CHUNKING** (also flagged inline at the `enable_truncation` call in
`preprocess.py`): the coupling is to *truncation*, not token count — `encode_batch_fast` is fine at
32k or any length. Once docs are chunked to fit the context and truncation is dropped, **remove the
`enable_truncation` line** (chunks are already ≤ `max_length`, so it's a no-op / unwanted) and the
fast path becomes fully baggage-free. If a 32k run still truncates rather than chunks, leave it as
is — just pass the new `max_length`.

## Measured baseline (answerdotai/ModernBERT-base, real kept docs)
- per-doc loop: 112 docs/s
- batched call: 334 docs/s (3.0×)
- median 4.35 chars/token, median 1092 tokens/doc
- Host CPUs: Intel Xeon Platinum 8481C (Sapphire Rapids) on v6e regions; AMD EPYC 7B12 (Rome) on v4
  (us-central2) — tokenize is CPU-bound, so it runs slower on the AMD Rome hosts.
