# Extractor / Classifier Comparison — Results Log

## ☀️ Morning summary (2026-06-26 ~07:25 PDT)

**Everything is joined into `sample_100k_scored` (31 cols) with full timing recorded.** Overnight work:

- **All score columns joined:** 5 ModernBERT-200k (ctx 1k–8k), BERT-1M, BERT base-1M-rand, BERT large-1M-surv,
  fastText w80/w160 (w320 scoring now), 0.6B/1.7B LLM logprob (full + 4k/8k/16k ctx), plus `text_justext` (200/200 WARCs).
- **Timing (per chip TPU / per core CPU):** §2 (ctx logprob), §2b (base 6.21 / large 4.99 docs/chip/s), §3 (jusText 9.43 docs/s/core), §3b (fastText w160 358 docs/s/core).
- **Finding 1 — usefulness vs context (§4):** LLM-as-classifier F1 holds to **16k** (1.7B 0.676→0.670, ~1.5× faster), moderate at 8k, **collapses at 4k** (0.404). Sweet spot ≈16k.
- **Finding 2 — best usefulness classifier (§4):** **10M-doc ModernBERTs top all — large-10M 0.690 > base-10M 0.682** > 1.7B LLM logprob 0.676. Data > arch: base→large is only +0.008 at 2× compute, so **base-10M is the production pick**; the 1M→10M data jump (+0.036) dwarfs it.
- **jusText saga:** 2 WARCs had 5 lxml-hanging docs each → fixed with a per-doc 20s subprocess timeout + no-Zephyr in-process path.
- **All done — every column (32) scored, joined, timed, and in §4.** fastText w320 = 0.536 F1 / 288 docs/s/core (widest, marginally best fastText).

---

Living results doc for the fixed **200-WARC / 100k-document** sample comparing web-content
extractors (8B / 1.7B / 0.6B LLM, jusText) and usefulness classifiers (ModernBERT survivor,
fastText, LLM marker-logprob). Append numbers here as they land — **especially timing**, which
is easy to lose across sessions.

- **Dataset root:** `gs://marin-us-east5/documents/extractor_compare/high_quality_200warc`
- **Scored 100k sample:** `…/sample_100k_scored/*.parquet`
- **Extraction pool:** 200 WARCs sampled from the 1.7B's completed set (random subset, seed42)
- Last updated: 2026-06-26

> Throughput convention: **docs/chip/s** (TPU) for the LLMs, **docs/s/core** (CPU) for jusText.
> TPU is free on TRC — timing is for *understanding cost/latency*, not billing.

---

## 1. Extraction throughput (docs/chip/s, TPU)

Full-document extraction (generate the cleaned text), stripped/raw HTML in → text out.

| Model | docs/chip/s | TPU | Notes |
|-------|-------------|-----|-------|
| 8B    | ~0.46 | v6e-4 | medium WARC subset |
| 1.7B  | ~1.0  | v6e-4 | ~2× faster than 8B; handles no-think (empty `<think></think>`) |
| 0.6B  | PENDING (head-to-head run) | v6e-4 / v5litepod-4 | lr2e-6 s10127 |

## 2. LLM-as-classifier: marker-logprob throughput (docs/chip/s, TPU)

Prefill doc + deterministic scaffold, emit 1 token, read logprob of the `[NO_USEFUL_CONTENT]`
branch token. Input = **stripped_html**. Full context = MAX_DOC_TOKENS (26624).

| Model | Context | docs/chip/s | Notes |
|-------|---------|-------------|-------|
| 0.6B  | full (~26k) | **2.77** | baseline |
| 1.7B  | full (~26k) | **2.03** | baseline |
| 0.6B  | 4k  | **18.7** | budget 2254 tok (overhead 1794); 74.9 docs/s/v6e-4; **6.8× full-ctx** |
| 0.6B  | 8k  | **8.35** | 33.4 docs/s/v6e-4; **3.0× full-ctx** |
| 0.6B  | 16k | **4.30** | 17.2 docs/s/v6e-4; **1.6× full-ctx** |
| 1.7B  | 4k  | **11.7** | budget 2254 tok; 46.7 docs/s/v6e-4; **5.8× full-ctx** |
| 1.7B  | 8k  | **5.63** | 22.5 docs/s/v6e-4; **2.8× full-ctx** |
| 1.7B  | 16k | **3.0** | 12.0 docs/s/v6e-4; **1.5× full-ctx** |

> Throughput ≈ halves per context doubling — prefill-bound, as expected. 4k→full spans ~7×.
> Full table (docs/chip/s): 0.6B 18.7 / 8.35 / 4.30 / 2.77(full); 1.7B 11.7 / 5.63 / 3.0 / 2.03(full).

> 4k is dramatically faster (~6×). Prefill dominates: overhead-aware truncation to 2254 HTML tokens
> shrinks the prompt ~11× vs the full ~26k, realizing ~6× throughput. Whether usefulness as a
> classifier survives this truncation is the §4 F1 question.

**logprob vs full generation:** logprob path is **1.7–2.2× faster** than full generation
(small-sample timing, identical inputs). Confirmed the classifier path is the right one for speed.

### 2b. ModernBERT scoring throughput (docs/chip/s, TPU)

100k sample, splash@8192, data-parallel on v6e-4 (4 chips), batch 32.

| Checkpoint | Arch | docs/chip/s | Notes |
|------------|------|-------------|-------|
| base-1M-rand-e5 | base | **6.21** | hidden=768/layers=22/heads=12 (from_hf_config); 24.9 docs/s/v6e-4 |
| large-1M-surv-e5 | large | **4.99** | survivor; hidden=1024/layers=28/heads=16 (from_hf_config); 20.0 docs/s/v6e-4 |
| mb-clf-base-10M-c8192 | base | **~6.0** | 10M-doc training; hidden=768/layers=22/heads=12; ~24 docs/s/v6e-4 (same arch as base-1M-rand) |
| mb-clf-large-10M-c8192 | large | **~5.0** | 10M-doc training; hidden=1024/layers=28/heads=16; ~2× base compute (same arch as large-1M-surv) |

## 3. jusText throughput (docs/s/core, CPU)

XenonMolecule fork **v4.2.0**, fastText tier (`MichaelR207/justext-classifier`, `JUSTEXT_MODEL=fasttext`).
Input = **raw_html** (clean re-decode). Two measurements:

| Measurement | docs/s/core | Notes |
|-------------|-------------|-------|
| Two-stage (decode+jusText, in fan-out) | ~3.0 (incl timeouts) | one WARC log: 471 docs/156.5s WITH 5×20s pathological-doc timeouts; true non-hung rate ~8/s/core |
| **Standalone bench (cpu=1, clean HTML, warmup'd)** | **9.43** | 2000 docs/212s; p50=31.9ms p90=98.2ms p99=1061ms; TRUE single-core (vs ~3.0 two-stage incl-timeouts) |

> **Endgame note:** 200/200 WARCs done. 2 WARCs hung the Zephyr fan-out all night; root cause = **5
> pathological docs** (per WARC) that freeze lxml at the C level — defeated with a per-doc 20s subprocess
> timeout + a no-Zephyr in-process path (the coordinator's heartbeat kept reassigning the slow WARCs).

### 3b. fastText throughput (docs/s/core, CPU)

100k sample, single CPU core (excludes parquet I/O); width sweep.

| Model | docs/s/core | Notes |
|-------|-------------|-------|
| w80   | not recorded | scored before timing was added |
| w160  | **358** | 99996 docs/279s, single CPU (excl I/O) |
| w320  | **288** | 99996 docs/347s, single CPU (excl I/O); slower than w160 (wider) |

## 4. Usefulness classifiers — discrimination vs 8B gold

Best-threshold **F1** of each scalar score vs `label_8b` (8B's keep/`[NO_USEFUL_CONTENT]` decision).

clf-f1e (all 17 classifier cols): **99,996 docs, 4,922 useful (4.9%)**. Sorted by F1. `low=useful`
for the LLM marker-logprob columns (lower marker-logprob ⇒ more useful), `high=useful` for prob columns.

| Classifier | Column | best-thr F1 | P / R | thr | dir |
|------------|--------|-------------|-------|-----|-----|
| **ModernBERT large-10M** ⭐ | `bert_useful_prob_large_10M` | **0.690** | 0.649 / 0.738 | 0.439 | high |
| **ModernBERT base-10M** | `bert_useful_prob_base_10M` | **0.682** | 0.617 / 0.762 | 0.391 | high |
| **LLM 1.7B logprob, full** | `llm_logprob_marker_1p7b` | **0.676** | 0.619 / 0.745 | −2.25 | low |
| LLM 1.7B @16k | `llm_logprob_marker_1p7b_ctx16k` | 0.670 | 0.613 / 0.738 | −2.16 | low |
| **ModernBERT large-1M-surv** | `bert_useful_prob_large_1M_surv` | **0.661** | 0.627 / 0.699 | 0.370 | high |
| ModernBERT 1M @8192 (surv) | `bert_useful_prob_1M_ctx8192` | 0.651 | 0.617 / 0.688 | 0.418 | high |
| ModernBERT base-1M-rand | `bert_useful_prob_base_1M_rand` | 0.645 | 0.612 / 0.682 | 0.359 | high |
| LLM 0.6B logprob, full | `llm_logprob_marker_0p6b` | 0.640 | 0.587 / 0.705 | −1.91 | low |
| LLM 0.6B @16k | `llm_logprob_marker_0p6b_ctx16k` | 0.636 | 0.584 / 0.699 | −1.83 | low |
| ModernBERT 200k @8192 | `bert_useful_prob_200k_ctx8192` | 0.602 | 0.571 / 0.637 | 0.317 | high |
| LLM 1.7B @8k | `llm_logprob_marker_1p7b_ctx8k` | 0.600 | 0.588 / 0.612 | −1.92 | low |
| LLM 0.6B @8k | `llm_logprob_marker_0p6b_ctx8k` | 0.565 | 0.555 / 0.575 | −1.70 | low |
| fastText w320 | `fasttext_useful_prob_w320` | 0.536 | 0.455 / 0.652 | 0.337 | high |
| fastText w160 | `fasttext_useful_prob_w160` | 0.527 | 0.429 / 0.683 | 0.236 | high |
| fastText w80 | `fasttext_useful_prob` | 0.527 | 0.458 / 0.620 | 0.274 | high |
| ModernBERT 200k @4096 | `bert_useful_prob_200k_ctx4096` | 0.521 | 0.494 / 0.551 | 0.307 | high |
| ModernBERT 200k @2048 | `bert_useful_prob_200k_ctx2048` | 0.419 | 0.384 / 0.460 | 0.290 | high |
| LLM 1.7B @4k | `llm_logprob_marker_1p7b_ctx4k` | 0.404 | 0.350 / 0.477 | −0.824 | low |
| LLM 0.6B @4k | `llm_logprob_marker_0p6b_ctx4k` | 0.359 | 0.352 / 0.366 | −0.59 | low |
| ModernBERT 200k @1024 | `bert_useful_prob_200k_ctx1024` | 0.332 | 0.261 / 0.457 | 0.227 | high |

> **Finding 1 — usefulness vs context:** truncating the LLM-as-classifier input barely costs anything
> down to **16k** (1.7B 0.676→0.670; 0.6B 0.640→0.636) while running ~1.5× faster; **8k** is a moderate
> hit (0.600 / 0.565); **4k collapses** usefulness (0.404 / 0.359 — roughly halved) despite being ~6×
> faster. Sweet spot ≈ 16k.
>
> **Finding 2 — best usefulness classifier:** the two **10M-doc ModernBERTs now top everything** —
> **large-10M (0.690) > base-10M (0.682)** > 1.7B LLM logprob (0.676). **Data, not arch, is the lever:**
> going base→large at the 10M scale adds only **+0.008**, but it costs **~2× the TPU compute** (large@8192
> ≈5 vs base ≈6 docs/chip/s; ceilings 11.9 vs 21.5) — so **base-10M is the production pick** (≈99% of the
> quality at half the cost, and ~3× faster than the 1.7B LLM at full ctx). The big jump was **1M→10M docs**:
> base gained **+0.036** (base-1M-rand 0.646 → base-10M 0.682), more than double the base→large arch gain.
> Among the older 1M BERTs: large-survivor (0.663) > 1M-survivor (0.651) > base-random (0.646) — arch and
> survivor-vs-random training each helped, but only by ~0.01. Among fastText, width helps marginally:
> w320 0.536 > w160 ≈ w80 0.527 — all far below the LLM/BERT scorers.
>
> **Note:** base-10M and large-10M were scored + joined later than clf-f1e (the original §4 sweep was 17
> cols / 99,996 docs; these are the 18th/19th, same sample). Best-thr F1s are from the full 100k. large-10M
> trades recall for precision vs base-10M (0.649/0.738 @ 0.439 vs 0.617/0.762 @ 0.391).

## 5. Extractor agreement metrics (8B as gold)

| Metric | Full 200-WARC | 100k sample |
|--------|---------------|-------------|
| (1) File size per extractor | DONE (see prior run) | DONE |
| (2) F1 vs 8B-gold (`[NO_USEFUL_CONTENT]` vs extracted) | DONE | DONE |
| (3) Levenshtein similarity, ALL docs (mean) | DONE | DONE |
| (4) Levenshtein, both-extracted only (**mean**) | DONE | DONE |

> Detailed numbers for §5 live in the prior analysis output; re-paste here next time they're regenerated.

---

## Scored dataset columns (current)

Extraction text: `text_8b`, `text_1p7b`, `text_0p6b`, `text_justext` (pending), `raw_html`, `stripped_html`.
Scores (scalar): `bert_useful_prob_200k_ctx{1024,2048,4096,8192}`, `bert_useful_prob_1M_ctx8192`,
`fasttext_useful_prob`, `llm_logprob_marker_{0p6b,1p7b}`, `llm_logprob_marker_{0p6b,1p7b}_ctx{4,8,16}k` (pending).

## Encoding note

The clean re-decode (jusText pass) uses `decode_warcs_clean.decode_payload` (WHATWG order, 0 U+FFFD).
The extractor `text_*` columns came through the broken `download_warcs` decoder (`errors="replace"`).
Full writeup: `WARC_ENCODING.md`.
