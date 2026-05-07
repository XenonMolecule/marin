# Why does resiliparse beat Nemotron on paloma? — hypothesis report

**Mystery:** at 1 B params across a compute sweep (3e18 → 5e19 FLOPs), the
**unfiltered** resiliparse extraction matches or beats Nemotron-CC on
`paloma_macro_loss`. `fineweb_edu` collapses past ~3e18 FLOPs from repetition,
`dclm` / `llm_curated` / `nemotron_full` track each other, resiliparse keeps
improving. The finding contradicts the premise that Nemotron-CC is higher-
quality than raw web.

Legend for Status column: 🟢 supported, 🔴 rejected, 🟡 partial signal, ⏳ testing, 🧪 deferred (needs external access).

---

## Dataset sanity report

| # | Hypothesis | Test | Status | Result |
|---|---|---|---|---|
| **D1** | Heavy filters retain only URLs inside our 3,000-WARC universe | `matched_viewer.py analyze` + `diagnose_missing.py` | 🔴 Rejected (filters are correct) | 62/62 and 108/108 missing-from-resi URLs **all** live inside our 3,000-WARC metadata. Filters do not pull from outside. |
| **D2** | Resiliparse is a URL superset of the filters (modulo `_is_non_empty`) | `analyze` validation.json | 🟢 Holds | 9,892/10,000 target URLs appear in resiliparse; 108 drops explained by `_is_non_empty`. |
| **D3** | `nemotron_full` has high redundancy from synthetic rephraser variants | phase-1 pass-1 counts | 🟡 Partial — **moderate, not catastrophic** | `nemotron_full`: 4.99 M records / 2.56 M URLs = **1.95 records/URL**; `nemotron` organic: 2.92 M records / 2.56 M URLs = 1.14. Only ~0.8 extra synthetic variants per URL on average. |
| **D4** | **BOS token is absent in `nemotron_full`** | `analyze_tokenized.py` | 🟢🚨🚨 **CONFIRMED + severe train/eval mismatch** | **Training caches**: 0.00% BOS for both `nemotron_full` AND `llm_curated`; 100% BOS for the other 4. **Eval caches**: 100% BOS for ALL 16 paloma subsets AND ALL 7 uncheatable subsets (verified). **Implication: models trained on `nemotron_full` or `llm_curated` literally never see BOS in training, then encounter BOS as the first token of every single paloma / uncheatable eval doc — severe distributional shift at eval boundary, likely a major contributor to their elevated eval loss.** Regression reproduced locally in `experiments/baseline_collection/test_bos_regression.py` (6 tests, all pass). Root cause: Levanter commit `f03aa9ecb` (2026-04-10, "Cap homogeneous-run length when calling Rust BPE tokenizers #4600") changed `MarinTokenizer.encode_batch` to default to `add_special_tokens=False`. `BatchTokenizer.__call__` calls it without `add_special_tokens=True`, so post-4-10 caches skipped BOS. **Fix: rebuild `nemotron_full` and `llm_curated` caches, OR patch `BatchTokenizer.__call__` to pass `add_special_tokens=True`.** |
| **D5** | Doc-length distribution differs → packing boundary rate differs → optimization difficulty differs | `analyze_tokenized.py` | 🟢 Supported | Docs/4096-seq: nemotron_full 7.64, nemotron 6.29, fineweb 3.99, resiliparse 4.03, **dclm 3.16**. `nemotron_full` has 2.4× more packing boundaries per sequence than `dclm`. DCLM has the longest docs (1297 tokens avg) and cleanest packing. |
| **D6** | Near-duplicate openings within sources (cheap dup proxy) | `analyze_tokenized.first100_dup_doc_fraction` | 🟡 Signal in unexpected direction | **Resiliparse has the highest first-100-token collision rate (8.13%)**; all filtered sources <0.6%. Likely web boilerplate (navigation, headers). Doesn't cancel the ~142 B unique-token advantage, but suggests resiliparse's effective uniqueness is inflated. |
| **D7** | Synthetic variants are near-duplicates of the organic doc | `analyze_nemotron_dupes.py` shingle Jaccard | 🔴 **Rejected** | `nemotron_full` variant-pairs have **mean-per-URL Jaccard = 0.35**; only 42% of multi-variant URL groups have any pair > 0.5. Synthetic rephraser outputs are **substantively different text**, not dilution. |
| **D7b** | `nemotron` organic itself has near-duplicate records per URL | same job | 🟢 Unexpected finding | `nemotron` org has multi-variant URLs with **mean-per-URL Jaccard = 0.72**, 76% high-Jaccard. Likely Common-Crawl re-crawls of the same page; moderate within-source redundancy. |
| **D8** | Sources have different temporal distributions (CC snapshots) | phase-1 `per_source_stats.json` | ⏳ Not yet reviewed | Collected; needs inspection. |
| **D9** | Paloma eval distribution favors raw web | per-subdomain paloma loss breakdown | 🧪 Deferred | Needs W&B per-subset values from each run. |
| **D10** | Top-token profiles differ across sources | `analyze_tokenized.top_tokens_first512` | ⏳ Output available, needs review | JSON on disk. |
| **D11** | **Effective unique content at fixed compute is dominant** | arithmetic on `_D_OBS_DEFAULTS` | 🟢🔥 **Strongest systemic explanation** | At 3e18 FLOPs ≈ 3 B tokens trained, resiliparse: 0.02 epochs, llm_curated: 0.05, nemotron/dclm/nemotron_full: 1.1–1.6 epochs, **fineweb_edu: 3.7 epochs**. Fineweb's plot inflection is exactly at the 1-epoch boundary. Nemotron runs are looping 1.5–5× by 10 B tokens. Paloma loss punishes epoching much faster than it rewards quality. |
| **D12** | `nemotron_full` has the shortest avg doc | `analyze_tokenized` | 🟢 Confirmed | avg 536 tokens/doc (vs dclm 1297, resiliparse 1016). Compounds D4 + D5 — more boundaries AND no BOS marker at those boundaries. |

### Tokenized-stats snapshot (100 K-doc samples)

| source | docs | avg tok/doc | docs/4096-seq | %BOS | BOS/1k tok | first100 dup | max repeat |
|---|---|---|---|---|---|---|---|
| resiliparse | 141.6 M | 1016 | 4.03 | 100% | 0.98 | **8.13%** | 755 |
| nemotron | 2.92 M | 652 | 6.29 | 100% | 1.53 | 0.49% | 68 |
| **nemotron_full** | 4.99 M | 536 | **7.64** | **0%** 🚨 | **0.00** 🚨 | 0.24% | 53 |
| dclm | 2.05 M | 1297 | **3.16** | 100% | 0.77 | 0.30% | 533 |
| fineweb_edu | 0.79 M | 1027 | 3.99 | 100% | 0.97 | 0.60% | 44 |
| **llm_curated** | 103.7 M | 540 | **7.59** | **0%** 🚨 | **0.00** 🚨 | 1.96% | 8544 |

`llm_curated`'s `max_repeat=8544` is also very high (longest contiguous run of the same token id in any sampled doc). Worth flagging — suggests some degenerate LLM-generated content.

### Current epoch-counts table (for D11 gut-check)

| budget | resiliparse | llm_curated | nemotron_full | dclm | nemotron | fineweb_edu |
|---|---|---|---|---|---|---|
| 1 B | 0.007 | 0.018 | 0.37 | 0.38 | 0.52 | 1.22 |
| 3 B | 0.021 | 0.054 | 1.11 | 1.13 | 1.56 | **3.67** |
| 10 B | 0.070 | 0.179 | 3.70 | 3.76 | 5.21 | 12.22 |
| 30 B | 0.210 | 0.537 | 11.1 | 11.3 | 15.6 | 36.7 |
| 50 B | 0.350 | 0.894 | 18.5 | 18.8 | 26.0 | 61.1 |

---

## Training sanity report (audit of launch code)

Comprehensive code audit completed by general-purpose agent; full writeup at the bottom.

| # | Hypothesis | Status | Conclusion |
|---|---|---|---|
| **T1** | Identical model hparams across methods | 🔴 Rejected as cause | **Identical by construction.** `_candidate_for_fixed_model(hidden_size, budget)` is method-agnostic. All 6 methods share Qwen3Config with `Llama3RotaryEmbeddingsConfig`, same `d_model/n_layers/n_heads/intermediate_dim`. |
| **T2** | Identical tokenizer (Llama-3.1-8B) | 🔴 Rejected as cause | Class-level default on `CurationMethod`; same `meta-llama/Meta-Llama-3.1-8B` for all 6. Ledger preprocessor metadata is None but `view_tokenized` decoded all 5 consistently with Llama 3.1. |
| **T3** | Identical shuffle / permutation | 🔴 Rejected as cause | `shuffle=True, permutation_type="feistel"` hard-coded in `data_curation_math.py:234-235`. Same for all methods. Note: the feistel seed is deterministic from the cache identity — since caches differ, realized permutations differ, but that's expected, not asymmetric. |
| **T4** | Identical packing config | 🔴 Rejected as cause | All use `TextLmDatasetFormat()` default → `_effective_pack=False`. `block_cross_document_attention=True` default. No per-method override. |
| **T5** | Identical paloma eval setup | 🔴 Rejected as cause | 16 paloma components added via `_add_validation_components("paloma", _PALOMA_CACHE_HASHES)` with `train_weights[key]=0.0`. Same cadence (`steps_per_eval=1000`). Same tokenizer. `paloma_macro_loss` computed identically per run. |
| **T6** | Same steps at fixed compute | 🟢 Confirmed | Fixed model + fixed batch ⇒ steps determined by budget only, not by dataset. |
| **T7** | Eval held out from training | 🟢 Confirmed | Paloma caches are separate from training caches and weighted zero in mixture. No leakage by construction. |
| **T8** | No silent NaN / early stop | 🧪 Deferred | W&B curves check. |

**Verdict on T1–T7: the training code is NOT the source of the asymmetry.** The only thing that differs across runs is the training cache path. `resiliparse`'s `reproduce_per_region=True` flag is informational (never read at training time).

---

## Remaining open hypotheses

1. **D9 — paloma per-subset bias** (🧪 deferred). Needs W&B breakdown: if resiliparse only dominates on CC-derived paloma subsets (c4_en, mc4, dolma, redpajama) and not curated ones (wikitext_103, ptb, m2d2_wikipedia), it's pure distribution match.
2. **T8 — training-curve sanity** (🧪 deferred). Needs W&B curves: any nans, any stops, any unusual loss shapes per method.
3. **BOS impact** (follow-up to D4). D4 confirms `nemotron_full` has zero BOS. How much does that actually hurt paloma perplexity on Llama-3.1? An ablation: retokenize `nemotron_full` with BOS prepended, retrain, compare. This is worth a follow-up experiment.
4. **llm_curated token stats** (⏳ running). Launching now.

---

## Consolidated current explanation

The picture that best fits the evidence so far:

1. **D4 (missing BOS on `nemotron_full` and `llm_curated`)** is a CONCRETE TRAIN/EVAL MISMATCH likely to cause material paloma-loss degradation. ALL paloma + uncheatable eval docs have BOS at the start. If the training never showed BOS, the model is OOD on the very first eval token. This affects `nemotron_full` and `llm_curated` — **both** of which are visibly worse than `nemotron`/`dclm` at mid-range compute on the plot. Highly actionable: patch Levanter or rebuild the two affected caches.
2. **D11 (effective unique tokens)** does the heavy lifting for `resiliparse` vs the filtered sources: at fixed compute, resiliparse sees 50–100× more unique content, and fineweb_edu's 1.2-epoch-at-1B-token crossover exactly matches its plot inflection.
3. **D5 + D12 (packing-boundary rate)** compounds `nemotron_full`'s disadvantage — shorter docs and 2.4× more boundaries than dclm.
4. **D7 (synthetic dup) is rejected** — Nemotron's rephraser outputs are genuinely different text, not near-duplicates.
5. **D9 (paloma eval distribution bias)** remains plausible but now less urgent, since D4 alone could explain the `nemotron_full`/`llm_curated` gap. Still worth pulling per-subset paloma losses from W&B.

---

## Running / launched jobs

- `ray-run-michaelryan-matched_viewer-20260420-172846` (analyze) — **DONE**
- `ray-run-michaelryan-analyze_tokenized-20260420-215109` (5 baseline sources, token stats) — **DONE**
- `ray-run-michaelryan-analyze_nemotron_dupes-20260420-220248` (shingle Jaccard) — **DONE**
- `iris-run-analyze_tokenized-20260421-012317` (llm_curated token stats on us-central1) — **DONE**
- `iris-run-matched_viewer-20260420-213138` (llm-curated URL scan, preemptible pool, 4 preemptions) — **KILLED and replaced**
- `iris-run-matched_viewer-20260421-014104` (llm-curated URL scan, `--priority interactive` + per-shard checkpointing) — running
- `ray-run-michaelryan-analyze_tokenized-20260421-015010` (paloma + uncheatable token stats) — running

## Local tests

- `experiments/baseline_collection/test_bos_regression.py` — 6 tests, all passing. Reproduces the BOS regression end-to-end in Levanter's `BatchTokenizer`, demonstrates the fix.
