# High-Quality 3000-WARC: Distillation Dataset + Useful-vs-NoUseful Classifier

State reference as of 2026-06-02. Two artifacts built from the **high_quality**
extraction over the first 3000 WARCs of a Common Crawl draw:

1. **A distillation dataset** — `raw_html → reasoning_trace + final_output` (for
   training a generative extractor, e.g. distilling into Qwen3.5-800M).
2. **A useful-vs-`[NO_USEFUL_CONTENT]` fastText classifier** — a cheap pre-filter
   trained on the same pages.

Everything lives in **us-central2** (no HuggingFace push). Region note: the
upstream extraction is in us-central1; the raw HTML and all derived artifacts are
in us-central2.

---

## 1. The distillation dataset

**`gs://marin-us-central2/datasets/high_quality_3000_distill/`**

| path | what |
|---|---|
| `data/` | **10,444,035** useful docs, **3000 parquet shards** (~260 GiB). The distillation set. |
| `data_no_useful/` | **129,311,595** abstentions, 3000 parquet shards (~2.0 TB). Classifier negatives. |
| `README.md` | HuggingFace dataset card (schema, provenance, caveats, stats). |
| `_staging_metadata/` | internal intermediate (per-WARC metadata, no HTML); safe to delete. |

**Columns (both `data/` and `data_no_useful/`, 7 string columns):**

| column | `data/` (useful) | `data_no_useful/` (abstention) |
|---|---|---|
| `raw_html` | original page HTML (teacher input) | same |
| `reasoning_trace` | teacher `<think>…</think>` chain-of-thought | **empty** (never persisted for abstentions) |
| `final_output` | cleaned extracted document | literal `"[NO_USEFUL_CONTENT]"` |
| `url`, `warc_file`, `warc_record_id`, `snapshot` | provenance | provenance |

One parquet shard = one WARC; `data/` shard *i* and `data_no_useful/` shard *i*
are the **same WARC** (shards ordered identically by WARC hash).

**Provenance & how it was built:** the high_quality extraction (an LLM under an
"extreme quality" spec) was joined back to the raw page HTML.
- Useful docs: `data/` = the kept extractions joined to raw HTML.
- Negatives: pages that were **fed to the teacher** (`len(html) ≤ 172,032` chars,
  the `max_doc_tokens·6` pre-model filter) but **not kept**. Built by set-difference.
- Within-WARC exact dedup (xxh3_128 of text) on the useful side.

**Caveats:**
- **Abstention reasoning is gone** — the extraction pipeline discarded
  `[NO_USEFUL_CONTENT]` before persisting, so negatives have HTML + marker but no
  reasoning. (Recovering it would require re-running the teacher LLM.)
- **Exact dedup only** — fuzzy near-dup removal for N=3000 never completed
  upstream; ~3% cross-WARC exact dupes may remain.
- Pages > 172,032 chars were never shown to the teacher and are **not** negatives.

**Upstream source paths** (inputs to the build):
- Raw HTML (2.03 TiB, keyed `data-{warc_hash}.jsonl.gz`, schema `{id, html, url, metadata.warc_file}`):
  `gs://marin-us-central2/raw/commoncrawl/baseline_3000-265ff5/`
- High_quality extraction (has `generated_text` w/ `<think>`, `text`, `warc_record_id`, …):
  `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/{region}/high_quality/data-{hash}/batch_*.jsonl.gz`
  (manifest: `…/resolved/resolved_high_quality.jsonl.gz`)
- WARC manifest (3000 s3 paths, encodes `CC-MAIN-YYYY-WW` snapshot):
  `experiments/distill/baseline_warcs_3000.txt`

**Builder code:** `experiments/baseline_collection/build_hq_distill_dataset.py`
(+ `test_build_hq_distill_dataset.py`). Stages (each map-only, per-WARC, durable
`skip_existing`; run as Iris jobs):
- `assemble` (us-central1): extraction batches → per-WARC metadata (reasoning+final+ids), within-WARC dedup → central2 staging.
- `htmljoin` (us-central2): join staged metadata to raw HTML → `data/` parquet.
- `negatives` (us-central2): fed-but-not-kept HTML → `data_no_useful/` parquet.
- `card` (us-central2): README from parquet footers.

---

## 2. The frozen held-out split (used by the classifier; reuse for the generator)

35 snapshots span **CC-MAIN-2013-20 → 2017-26** (~4 years). Held-out is
**snapshot-stratified and frozen**: for each snapshot, the lowest-index WARC →
**val**, next → **test**. So **35 val + 35 test WARCs**, both covering every time
period; train = the other 2930. Deterministic, identical across runs → comparable
metrics as train size scales. Implemented in `held_out_indices()`.

Per-split line counts (natural ratio, from the full prep):

| split | WARCs | useful | no_useful | total |
|---|---:|---:|---:|---:|
| train | 2930 | 10,179,046 | 126,182,613 | 136,361,659 |
| val | 35 | 130,523 | 1,545,029 | 1,675,552 |
| test | 35 | 134,466 | 1,583,953 | 1,718,419 |

---

## 3. The classifier

**Goal:** binary `useful` vs `[NO_USEFUL_CONTENT]`, optimizing **recall of useful**
(keep all good pages), threshold-swept. fastText (linear bag-of-ngrams).

**Code:** `experiments/baseline_collection/fasttext_useful_classifier.py`
(+ `test_fasttext_useful_classifier.py`, 12 tests). Runs as Iris jobs with
`--extra cpu --extra dclm` (fastText = `fasttext-wheel`, in the `dclm` extra; NOT
in the local venv). Commands:

- **`pilot`** — prep (sample, balanced 1:1 within WARC) + train + threshold-sweep eval, one job. Key flags: `--representation {body_strip,raw_full,resiliparse}`, `--train-warcs`, `--max-per-class`, `--k-holdout 1`, `--eval-per-class`, and either autotune (`--autotune-duration`, `--autotune-metric`) **or** `--fixed-config` (DCLM recipe: epoch 5, lr 0.1, dim 100, wordNgrams 2, softmax) **or** `--reuse-config <metrics.json>`.
- **`prep`** — parallel Zephyr full-corpus prep: ALL data, **no balancing** (natural ratio), per-WARC gzipped fastText text shards, split-tagged. Flags: `--representation`, `--only-split {all,train,val,test}`, `--limit-warcs`.
- **`eval`** — score a saved `model.bin` on an external fastText-text test set. Flags: `--model`, `--test-glob` (default = full-prep natural-ratio test), `--quality-label` (`__label__useful` for ours, `__label__eli5` for DCLM), `--out`.

**Representations** (all whitespace-collapsed + lowercased, **no length cap**):
- `body_strip` — strip `<script>`, keep `<body>` (the user's `small-rephraser`
  `preprocess_html_for_extraction`, at
  `/Users/michaelryan/Documents/School/Stanford/Research/small-rephraser/scripts/run_spec_bench.py`). **The chosen representation.**
- `raw_full` — whole HTML.
- `resiliparse` — main-content plain text (DCLM-style), `extract_plain_text(main_content=True)`.

**Important eval caveat:** the `pilot` command's val/test are **balanced 1:1**, so
its precision is optimistic. The honest deployment curve comes from `eval` on the
**natural-ratio** full-prep test (~12:1). Recall is ratio-invariant (valid either
way); precision is not.

**Throughput reality:** body_strip keeps HTML markup → huge vocab → fastText runs
slow (~14.5k words/sec/thread). One epoch-5 train: ~24 min @ 200k, ~2 h @ 1M,
**~5–6 days @ full 136M**. So the full 136M is impractical on speed alone
(independent of the ~2.5 TB uncompressed disk need); **1M is the practical ceiling**
— which matches DCLM (~200k) and Dolma (~25 GB).

**Artifacts:** `gs://marin-us-central2/classifiers/useful_fasttext/`
- `body_strip_dclm200k/` — DCLM-recipe 200k: `model.bin`, `metrics.json` (balanced test), `eval_natural.json` (natural-ratio), `train/val/test.txt.gz`.
- `body_strip_200k-12h/`, `body_strip_1m/` — autotune runs (in flight).
- `full_prep_body_strip/{train,val,test}/` — full natural-ratio fastText text (~782 GiB gz).
- `full_prep_resiliparse/test/` — resiliparse test set (for the DCLM-model comparison).
- `*_smoke/` — early smoke runs (safe to delete).
- DCLM reference model (copied): `gs://marin-us-central2/resources/dclm/fasttext_oh_eli5.bin` (labels `__label__eli5` / `__label__cc`; P(eli5)=quality).

**Results.** DCLM-recipe body_strip 200k:
- *Balanced* test (1:1): best F1 **0.868** @ t=0.22 (P 0.835 / R 0.904).
- *Natural-ratio* test (11.8:1, deployment-real): best F1 **0.472** @ t=0.92; **P≥0.90 unreachable** (imbalance). DCLM off-the-shelf oh-hq classifier (labels `__label__cc`/`__label__hq`; quality = 1−P(cc)): best F1 **0.210** (barely above trivial 0.145) — **ours wins clearly.**
- **Recall-first (the real use):** at t=0.50 keep **93% of useful** while dropping **~70% of junk**; t=0.30 → 97% kept / ~56% junk dropped; t=0.70 → 86% kept / ~80% dropped. Usable as a cheap pre-filter.

**Key caveat / next experiment:** every trained model so far is **balanced**-trained (pilot balances 1:1), so probabilities are mis-calibrated for the 12:1 deployment prior. **Train on the natural ratio** (the full-prep data is already natural-ratio) for a better deployment PR curve — this is the clear next classifier experiment. Stored per-example `(score, useful)` preds (`eval` writes `*.preds.parquet`) let you re-sweep thresholds offline instantly.

---

## 4. HANDOFF — train Qwen3-0.6B to distill the extractor

Goal: SFT **Qwen3-0.6B** to reproduce the teacher: **input = `raw_html`** (with the
extraction-spec scaffold), **target = `reasoning_trace` + `final_output`** (or just
`final_output` for a no-CoT variant). The repo already has the machinery; the only
new code is a parquet→chat conversion.

### Inputs (all exist)
- **Training data:** `gs://marin-us-central2/datasets/high_quality_3000_distill/data/` — 10.44M rows, columns `raw_html`, `reasoning_trace`, `final_output`, `url`, `warc_file`, `warc_record_id`, `snapshot`.
- **Frozen eval split (reuse for zero leakage with the classifier):** the 35 val + 35 test WARCs from `held_out_indices()` in `experiments/baseline_collection/fasttext_useful_classifier.py` (snapshot-stratified; train = the other 2930 WARCs ≈ 10.18M rows). Or just train on all of `data/` and eval on those test WARCs' pages.
- **Prompt scaffold (what the teacher saw):** `experiments/baseline_collection/extraction_specs.py` → `SPECS["high_quality"].extraction_template` (wraps `[[ ## html ## ]]\n{example}` with the `_HIGH_QUALITY_RULES`). Put the rules in the **system** turn; `{example}` = the page HTML in the **user** turn.
- **Model + tokenizer:** `experiments/qwen3.py:qwen3_0_6b_hd128`, tokenizer `Qwen/Qwen3-0.6B`.

### Template to clone
`experiments/exp_qwen3_0_6b_rephraser_sft.py` — a working **Qwen3-0.6B SFT, 32k
context, chat-format** run. It uses: `experiments.defaults.default_sft`,
`experiments.simple_sft_config.SimpleSFTConfig`, `levanter…ChatLmDatasetFormat`
(+ `QWEN_3_CHAT_TEMPLATE`), `marin.transform.filter_by_context_length`,
`marin.processing.tokenize.tokenize`, launched via `executor_main`. It SFTs the
rephraser dataset (84k chat rows) on **v5p-8**, 1 epoch. Our task is the same shape
(HTML→output); just swap the dataset.

### Steps for the coworker
1. **Convert `data/` parquet → chat JSONL** (the one new transform). Per row emit
   `{"messages":[{"role":"system","content":<high_quality rules>},{"role":"user","content":<extraction_template with {example}=raw_html>},{"role":"assistant","content":<assistant>}]}` where `<assistant>` =
   `"<think>\n"+reasoning_trace+"\n</think>\n"+final_output` for the **with-CoT** variant, or just `final_output` for **no-CoT**. Write train (2930 WARCs) + val/test (the frozen 35+35) to GCS, e.g. `gs://marin-us-central2/datasets/high_quality_3000_distill_chat/{train,val,test}/`. (Mirror the per-WARC Zephyr pattern in `fasttext_useful_classifier.py:run_prep`.)
2. **filter_by_context_length** (as in the template) to drop rows whose prompt leaves
   <64 assistant tokens in `MAX_SEQ_LEN`. ⚠️ **Big effect here:** `raw_html` averages
   ~20k tokens and runs up to ~40k, so at 32k context a real fraction is dropped.
   Decide: **truncate the HTML input** to fit (faithful cap = the teacher's 28,672
   tokens) so you don't *lose* long pages, vs. filter them out. Recommend truncate.
3. **tokenize** with `Qwen/Qwen3-0.6B` → Levanter cache (use `window_size_bytes=1` so each shard is its own file, per the template's note).
4. **`default_sft(... SimpleSFTConfig(...))`** → train. Set `NUM_TRAIN_EXAMPLES`,
   `TARGET_EPOCHS`, `TRAIN_BATCH_SIZE`, `MAX_SEQ_LEN` (32k like the template), and a
   TPU `ResourceConfig` (template uses v5p-8; scale up for more data).
5. **Launch** with `uv run iris --config lib/iris/examples/marin.yaml job run … -- python experiments/<new>.py` (CPU coordinator; SFT runs on the requested TPU).

### Decisions to make first (these set cost/feasibility)
- **Input length cap** — biggest cost lever. Full-length × all 10.2M @ 32k is a
  multi-day pod run (see throughput note in §3). Recommend: truncate input (~8k–28,672 tokens) and/or subsample to ~1–2M examples and/or <1 epoch for a first run.
- **CoT vs no-CoT** target (`reasoning_trace`+`final` vs `final` only) — make both variants; the columns are already separated.
- **TPU size** — v5p-8 (template) for a subsampled first run; bigger for full.

### Eval
Generate on the **frozen 35 test WARCs'** `raw_html` and compare to `final_output`
(the classifier already reserved these, so reusing them keeps train/test disjoint
across both models).

---

## §5 Overnight optimization (started 2026-06-01 23:xx) — finding the best classifier

**Goal:** the strongest deployment classifier on the frozen natural-ratio (~12:1) test.

**Shared leaderboard:** `gs://marin-us-central2/classifiers/useful_fasttext/LEADERBOARD.md`
(+ `leaderboard.json`). Rebuild any time: `pilot ... ` then
`python fasttext_useful_classifier.py leaderboard` (run in-region via Iris — local gcsfs has SSL issues).
Ranks by natural-ratio best F1; degenerate evals (best-F1 at threshold 0) are flagged ⚠ and sunk.

**Results so far (natural-ratio best F1):**
- balanced front 200k = **0.472** · balanced front 1M = **0.425** (more balanced data HURT — calibration, not data)
- DCLM off-the-shelf (oh-eli5, neg-label cc) = **0.210**
- balanced pilot-test (1:1): front 0.868 < stratified 0.873 ≈ random 0.874 (sampling helps generalization)

**Two independent levers, never yet combined:**
1. **Ratio** — train at the true 12:1 (`--neg-per-pos 12`) instead of balanced. Fixes calibration.
2. **Sampling** — `--train-sample stratified` instead of front-first (front index ↔ snapshot ⇒ biased;
   front-80 covers only 24/35 snapshots, stratified covers 35/35).

**In flight:**
- Front natratio: `natratio200kpos` (12:1 fixed), `natratio200kpos-12h` (12:1 12h autotune), `natratio1Mpos` (1M×12M longshot).
- Sampling A/B (balanced 200k): `balrandom200k` (eval running), `balstratified200k` (lost model — superseded by `sweep_strat_n1`).
- **NEW composition — stratified × ratio sweep @200k fixed:** `sweep_strat_n{1,4,8,12}`.

**Orchestrator** (`/tmp/ft_overnight_orchestrator.sh`, bg): auto-launches each model's natural eval when
`model.bin` lands, refreshes the board on each new eval, wakes the agent when the fast 200k stratified
batch is fully evaluated.

**Decision tree (agent, as results land):**
1. Fast 200k stratified sweep done → read board → pick best ratio under stratified; compare stratified vs
   front at matched 12:1 (`sweep_strat_n12` vs `natratio200kpos`).
2. Compose & scale: launch 1M with (stratified + best ratio + best recipe). If the 12h autotune beats the
   DCLM recipe, autotune the winning stratified+ratio combo too.
3. As the 1M longshot + autotune land, eval, compare, push the winner further (more data / refine ratio
   around the sweet spot). Repeat.

**Guardrails:** CPU-only us-central2 (no cross-region, no egress); never kill running jobs; model.bin now
uploads before metrics.json so it's the reliable "done" signal.
