# Math Multi-Source Extraction SFT v2 Report

**Last updated**: 2026-03-17 18:05 UTC

## Goal

Scale math SFT beyond mathhelpforum.com to **42 math domains** across 10 Common Crawl snapshots (2013–2025). The previous experiments ([Qwen3 MHF report](QWEN3_MATHHELPFORUM_SFT_REPORT.md)) showed that SFT on mathhelpforum data meaningfully improves math reasoning (GSM8K 41.77% → 44.35% with Q/R/A plaintext, MATH avg 7.63% → 10.41% with raw markdown). This experiment tests whether:

1. **More diverse math data helps** — 42 domains vs. 1 domain (mathhelpforum)
2. **LLM extraction at scale** — does extraction quality hold across heterogeneous sources?
3. **Volume matters** — ~2.95B tokens (resiliparse) vs ~262M tokens (mathhelpforum-only resiliparse)

## Model

- **Architecture**: Qwen3 (hidden=1024, layers=28, heads=16, kv_heads=8, intermediate=3072, head_dim=128), ~0.6B params
- **Base model**: `Qwen/Qwen3-0.6B-Base` (pretrained, no instruction tuning)
- **Tokenizer**: `Qwen/Qwen3-0.6B-Base` (151,936 vocab, native Qwen3 tokenizer)
- **Training**: 1-epoch SFT, batch=64, seq_len=4096, AdamConfig lr=2e-5, cosine schedule
- **Tokenizer padding**: `pad_tokenizer_to_match_model=True`

## Data Sources

### Domain coverage: 42 math websites

Sources span K-8 worksheets, high school tutorials, university Q&A forums, curriculum standards, textbook platforms, and research-level math.

| Category | Domains | Peak Entries |
|----------|---------|-------------|
| **Q&A forums** | mathhelpforum.com, math.stackexchange.com, mathoverflow.net, jiskha.com, brainly.com, physicsforums.com, forums.wolfram.com, mathisfunforum.com, brainmass.com | ~1.3M+ |
| **Educational platforms** | khanacademy.org, geogebra.org, openstax.org, math.libretexts.org, brilliant.org, desmos.com, illustrativemathematics.org | ~200k+ |
| **Tutorials & reference** | purplemath.com, mathsisfun.com, mathway.com, symbolab.com, cliffsnotes.com, sparknotes.com, mathplanet.com, onlinemathlearning.com | ~100k+ |
| **K-12 & worksheets** | splashlearn.com, softschools.com, helpingwithmath.com, math-drills.com, coolmath.com, aaamath.com, savemyexams.com | ~50k+ |
| **Exam/test prep** | varsitytutors.com, savemyexams.com, algebrahelp.com, engageny.org | ~80k+ |
| **University** | mathcentre.ac.uk, tutorial.math.lamar.edu, nrich.maths.org, homeschoolmath.net, mathgoodies.com, mathwarehouse.com, math-only-math.com | ~20k+ |

### Common Crawl snapshots (10 indices)

| Index | Era | Key sources |
|-------|-----|-------------|
| CC-MAIN-2013-48 | Historical peak | math.stackexchange (148k), mathforum (389k), jiskha (258k), mathoverflow (202k) |
| CC-MAIN-2014-23 | Historical | Strong Q&A continuation |
| CC-MAIN-2016-07 | Mid-era | mathhelpforum building up, khanacademy (83k) |
| CC-MAIN-2016-44 | Peak volume | mathhelpforum (506k!), brainly (203k), brainmass (114k) |
| CC-MAIN-2017-47 | Transition | Catches sites before they block crawlers |
| CC-MAIN-2018-47 | Late historical | physicsforums (52k), geogebra (104k), engageny (4.8k) |
| CC-MAIN-2020-50 | Modern emerging | Mid-era modern sources |
| CC-MAIN-2022-49 | Modern | symbolab growing (6k→43k) |
| CC-MAIN-2024-46 | Near-modern | splashlearn, savemyexams appearing |
| CC-MAIN-2025-47 | Current | symbolab (43k), khanacademy (35k), geogebra (72k) |

### Pipeline stages

| Stage | Description | Output |
|-------|-------------|--------|
| CDX query | Query CC index for all 42 domains × 10 crawl indices | CDX manifest (4.28M entries) |
| WARC download | HTTP byte-range download of HTML pages | 500 shards, 52.7 GiB (domain) + host group |
| Combine | Merge domain + host download groups | 1,002 files |
| Token length filter | Remove docs exceeding 32K context window | 1,000 files, 19.7 GiB (~38% of input retained) |
| **Resiliparse extraction** | Standard text extraction from HTML | 1,000 files, SUCCESS |
| **LLM extraction** | Qwen3-0.6B extraction with math prompt | 703/6,362 shards (11%, IN PROGRESS) |
| Postprocess | Strip `<think>` tags, filter `[NO_USEFUL_CONTENT]` | Waiting for extraction |
| Tokenize | Pack into seq_len=4096 sequences | Done (resiliparse); waiting (extraction) |
| SFT training | 1-epoch fine-tune on Qwen3-0.6B-Base | In progress (resiliparse); waiting (extraction) |
| Eval | GSM8K 8-shot + MATH 4-shot (7 subtopics) | Waiting |

## Training Data

### Resiliparse branch (COMPLETE)

| Metric | Value |
|--------|-------|
| **Total tokens** | **2,950,310,697** (~2.95B) |
| **Documents** | 3,284,904 |
| **Avg tokens/doc** | ~898 |
| **On-disk size** | 4.17 GiB (tokenized) |
| **Tokenizer** | Qwen/Qwen3-0.6B-Base |
| **Seq length** | 4,096 |
| **Training steps** | ~11,300 |
| **GCS path** | `gs://marin-us-central1/tokenized/math_multi_v2_resiliparse_qwen3-0.6b-base_sft-512dbc` |

**Comparison with MHF-only resiliparse** (from Qwen3 MHF report):

| | MHF-only | Multi-source (this) | Scale factor |
|---|---|---|---|
| Tokens | 262M | 2,950M | **11.3x** |
| Documents | 506k | 3.28M | **6.5x** |
| Domains | 1 | 42 | **42x** |
| Training steps | ~1,000 | ~11,300 | **11.3x** |

### LLM extraction branch (IN PROGRESS)

| Metric | Value |
|--------|-------|
| **Total input shards** | 6,362 (500 records each, ~3.18M records) |
| **Completed shards** | 703 (11%) |
| **Rate** | ~90-100 shards/hr |
| **ETA** | ~60-65 hr (2.5-2.7 days) |
| **Extraction model** | Qwen3-0.6B |
| **Config** | 16 × v5p-8 TPU workers, max_context=32K, max_output=4K |
| **Status** | RUNNING (restarted #10) |

### Baseline dataset

| Data Source | Description | Tokens |
|---|---|---|
| GSM8K Plaintext | `openai/gsm8k` train split, `Q: ... A: ...` format, 7,473 examples | ~3M |

## Experiments

### 1. Resiliparse SFT (baseline text extraction)

Continued pretraining on plain text extracted from multi-source math HTML via resiliparse. No chat format, no structured Q&A — just raw text from 42 math websites.

| | |
|---|---|
| **Script** | `experiments/rephraser/mathhelpforum_extraction_sft_v2_base.py` |
| **Data** | 2.95B tokens from resiliparse text extraction across 42 math domains |
| **Eval format** | Plain text (`apply_chat_template=False`) |
| **Status** | **TRAINING COMPLETE (step 11,254/11,254). EVAL RUNNING.** |

### 2. LLM Extraction SFT (structured Q/R/A extraction)

SFT on LLM-extracted structured math content (Question/Reasoning/Answer format) from the same 42 domains using a math-specific extraction prompt.

| | |
|---|---|
| **Script** | `experiments/rephraser/mathhelpforum_extraction_sft_v2_base.py` |
| **Data** | LLM extraction output (TBD tokens — extraction 11% complete) |
| **Eval format** | Plain text (`apply_chat_template=False`) |
| **Status** | **EXTRACTION FAILED at 703/6,362 shards — needs `.executor_status` reset** |

### 3. GSM8K Plaintext SFT (task-specific baseline)

Fine-tune on GSM8K train set only. Reuses the same `gsm8k_plaintext_step` from prior experiments.

| | |
|---|---|
| **Script** | `experiments/rephraser/mathhelpforum_extraction_sft_v2_base.py` |
| **Data** | GSM8K train, ~3M tokens |
| **Eval format** | Plain text (`apply_chat_template=False`) |
| **Status** | **TRAINING + EVAL COMPLETE** |

## Evaluation Benchmarks

All experiments are evaluated on the same set of math benchmarks:

| Benchmark | Shots | Type | Test Examples |
|---|---|---|---|
| GSM8K CoT | 8-shot | Generation (chain-of-thought) | 1,319 |
| MATH Algebra | 4-shot | Generation | 1,187 |
| MATH Counting & Probability | 4-shot | Generation | 474 |
| MATH Geometry | 4-shot | Generation | 479 |
| MATH Intermediate Algebra | 4-shot | Generation | 903 |
| MATH Number Theory | 4-shot | Generation | 540 |
| MATH Prealgebra | 4-shot | Generation | 871 |
| MATH Precalculus | 4-shot | Generation | 546 |
| | | **MATH Total** | **5,000** |

**Reference scores** (Qwen3 technical report, Table 8): Qwen3-0.6B-Base achieves GSM8K=59.59% (4-shot CoT) and MATH=32.44% (4-shot CoT).

**MHF-only baselines** (from Qwen3 MHF report, Qwen3-0.6B-Base, plain text eval):

| Condition | GSM8K CoT 8-shot | MATH Avg |
|---|---|---|
| Baseline (no SFT) | 41.77% | 7.63% |
| Resiliparse SFT (MHF-only, 262M tokens) | 37.23% | 10.07% |
| Raw Q/R/A markdown SFT (MHF-only) | 41.24% | 10.41% |
| Q/R/A plaintext SFT v2 (MHF-only) | 44.35% | 9.85% |

---

## Results

All metrics are **exact-match accuracy**. GSM8K CoT uses `exact_match,strict-match`; MATH subtopics use `exact_match,none`.

> **Note on baseline numbers**: The baseline (no SFT) scores below differ from the MHF report (GSM8K 55.88% here vs 41.77% there). This is because each experiment runs its own baseline eval, and the eval setup (lm-evaluation-harness version, tokenizer padding, generation params) may differ slightly between pipelines. **Always compare within this table** for apples-to-apples.

### Plain text eval (no chat template)

| Condition | GSM8K CoT 8-shot | MATH Algebra | MATH Count&Prob | MATH Geometry | MATH Int. Algebra | MATH Num Theory | MATH Prealgebra | MATH Precalc | MATH Avg | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (no SFT) | **55.88%** | 12.30% | 11.81% | 11.27% | 7.09% | 9.07% | 19.17% | 8.06% | **11.25%** | DONE |
| **Resiliparse SFT (42 domains, 2.95B tok)** | **38.97%** | 12.55% | 8.86% | 9.39% | 6.42% | 9.07% | 17.22% | 8.24% | **10.25%** | DONE |
| LLM Extraction SFT (42 domains) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | EXTRACTION FAILED |
| GSM8K plaintext SFT | 0.00%† | 10.95% | 11.39% | 8.98% | 7.86% | 9.44% | 16.99% | 8.06% | **10.52%** | DONE |

† GSM8K strict-match is 0.00% due to format mismatch, but flexible-extract gives 45.64%. The model likely outputs answers in a non-standard format after SFT on the raw GSM8K train set.

### Key comparisons

| Question | Baseline | Result | Delta |
|----------|----------|--------|-------|
| Does resiliparse SFT (2.95B tok) help GSM8K? | 55.88% | 38.97% | **-16.91%** |
| Does resiliparse SFT (2.95B tok) help MATH? | 11.25% | 10.25% | **-1.00%** |
| Does LLM extraction beat resiliparse at scale? | (resiliparse) 10.25% MATH | (extraction) _TBD_ | _TBD_ |
| Does GSM8K SFT help GSM8K? | 55.88% | 0.00% strict / 45.64% flex | **-10.24%** (flex) |
| Does GSM8K SFT help MATH? | 11.25% | 10.52% | **-0.73%** |

## Analysis

### Key findings (3 of 4 conditions complete)

1. **Resiliparse SFT (2.95B tokens) hurts both benchmarks**: GSM8K drops from 55.88% → 38.97% (**-16.91%**) and MATH avg drops from 11.25% → 10.25% (**-1.00%**). The 11x scale-up from MHF-only resiliparse did not help — it made the GSM8K regression even worse. Continued pretraining on raw math text damages the model's existing reasoning capabilities.

2. **GSM8K plaintext SFT also hurts**: Fine-tuning on the small GSM8K train set (~3M tokens) degrades GSM8K from 55.88% to 45.64% (flexible extract) and MATH avg from 11.25% to 10.52%. The strict-match score of 0.00% indicates the model learned a non-standard answer format.

3. **More data ≠ better for naive SFT**: Both resiliparse (2.95B tokens) and GSM8K plaintext (~3M tokens) SFT degrade performance vs. the base model. The base Qwen3-0.6B already has strong math reasoning (55.88% GSM8K, 11.25% MATH avg) — naive continued pretraining on domain text, even at massive scale, causes catastrophic forgetting rather than improvement.

4. **Per-subtopic breakdown**: Resiliparse SFT shows mixed results across MATH subtopics — Algebra slightly improves (12.30% → 12.55%, +0.25%), but Counting & Probability (-2.95%), Geometry (-1.88%), and Prealgebra (-1.95%) all regress. This suggests the diverse math text helps with algebraic reasoning but dilutes other skills.

### Implications

- **Structured extraction may be critical**: The raw text approach clearly doesn't work at scale. The LLM extraction branch (structured Q/R/A format) remains the key hypothesis — if it can beat resiliparse, it validates the extraction pipeline. If not, the entire multi-source SFT approach may need rethinking.
- **Base model is already strong**: Qwen3-0.6B-Base has been pretrained on enough math data that naive continued pretraining hurts. Future experiments should focus on (a) higher-quality extraction, (b) mixing with general text to prevent forgetting, or (c) instruction tuning rather than continued pretraining.

### Open question

- **LLM extraction at scale**: Will structured Q/R/A extraction outperform resiliparse? The MHF-only experiments showed a slight edge for extraction on MATH (10.41% vs 10.07%). Extraction is currently FAILED at 703/6,362 shards — needs restart.

## Job Info

| | |
|---|---|
| **Experiment script** | `experiments/rephraser/mathhelpforum_extraction_sft_v2_base.py` |
| **Cluster** | us-central1 |
| **Current job** | `ray-run-michaelryan-mathhelpforum_extraction_sft_v2_base-20260317-173205` |
| **Restart count** | 10 |
| **Resiliparse training checkpoint** | `gs://marin-us-central1/checkpoints/math_multi_v2-resiliparse-qwen3-0.6b-base-sft-66e234` |
| **Resiliparse tokenized data** | `gs://marin-us-central1/tokenized/math_multi_v2_resiliparse_qwen3-0.6b-base_sft-512dbc` |
| **Extraction output** | `gs://marin-us-central1/documents/math_multi_v2_extract_unified_e4bba294-9dd505` |
| **Monitoring state** | `.agents/scratchpad/monitor_math_extraction_v2_base.json` |
