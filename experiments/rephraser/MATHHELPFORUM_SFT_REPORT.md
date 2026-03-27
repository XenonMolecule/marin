# Mathhelpforum SFT Experiment Report

**Last updated**: 2026-03-09

## Goal

Fine-tune the exp2166 pretrained checkpoint (~1.385B Qwen3, Llama3 tokenizer) on math content from mathhelpforum.com, then evaluate on GSM8K 8-shot CoT. Compare different data extraction methods to see if rephraser-extracted structured data improves math reasoning over raw text.

## Model

- **Architecture**: Qwen3 (hidden=1792, layers=18, heads=14, kv_heads=14, intermediate=7168), ~1.385B params
- **Tokenizer**: Llama3 (`meta-llama/Meta-Llama-3.1-8B`, 128,256 vocab)
- **Source checkpoint**: `exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0`, step 44660
- **Training**: 1-epoch SFT, batch=64, seq_len=4096, AdamConfig lr=2e-5, cosine schedule

## Experiments

### 1. GSM8K Baseline Eval (no SFT)

Evaluate the pretrained checkpoint directly on GSM8K -- no fine-tuning at all. This is the floor.

| | |
|---|---|
| **Script** | `experiments/rephraser/gsm8k_baseline_eval.py` |
| **What it does** | Runs GSM8K 8-shot CoT eval on the pretrained HF checkpoint (step 44758) |
| **Data** | None (eval only) |
| **Eval result** | exact_match strict=0.0%, flexible=0.0% |
| **Status** | COMPLETE |

### 2. GSM8K SFT

Fine-tune on the GSM8K train set itself (7,473 Q&A pairs with chain-of-thought). This is the ceiling -- if SFT on the actual task data doesn't help, nothing will.

| | |
|---|---|
| **Script** | `experiments/rephraser/gsm8k_sft.py` |
| **What it does** | Chat SFT on GSM8K train -> GSM8K eval |
| **Data** | `openai/gsm8k` train split (HuggingFace), chat format, 7,473 examples |
| **Tokens** | 3,180,237 (3.2M) |
| **Training steps** | 13 (1 epoch — only 13 gradient updates) |
| **Checkpoint** | `gs://marin-us-central1/checkpoints/gsm8k-sft-1e20-qwen3-e89008/` |
| **Eval result (no chat template)** | exact_match strict=0.0%, flexible=0.076% (false positive — see sample outputs) |
| **Eval result (with chat template)** | exact_match strict=0.0%, flexible=0.0% |
| **Status** | COMPLETE |

### 3. Mathhelpforum Resiliparse SFT

Continued pretraining on plain text extracted from mathhelpforum HTML via resiliparse (standard DCLM extraction). No chat format, no structured Q&A -- just raw text. Tests whether exposure to math text alone helps.

| | |
|---|---|
| **Script** | `experiments/rephraser/mathhelpforum_resiliparse_sft.py` |
| **What it does** | Resiliparse text extraction -> tokenize (plain text) -> continued pretraining -> GSM8K eval |
| **Data** | Plain text from mathhelpforum HTML |
| **Tokens** | 250,085,438 (250.1M) |
| **Training steps** | 954 (1 epoch) |
| **Checkpoint** | `gs://marin-us-central1/checkpoints/mathhelpforum-resiliparse-sft-1e20-qwen3-56a96f/` |
| **Eval result** | exact_match strict=0.0%, flexible=0.0% |
| **Status** | COMPLETE |

### 4. Mathhelpforum Rephraser SFT (wrong prompt)

Chat SFT on rephraser-extracted markdown. **Used the wrong extraction prompt** -- the generic "extract main content from HTML" prompt instead of a math-specific Q/R/A prompt. The rephraser output is raw forum threads (student questions + expert replies) without structured reasoning sections.

| | |
|---|---|
| **Script** | `experiments/rephraser/mathhelpforum_sft.py` |
| **Extraction prompt** | Generic content extraction (spec `71938abf`) |
| **What it does** | Parse forum posts into user/assistant chat turns -> tokenize -> SFT -> GSM8K eval |
| **Parse rate** | 80.7% (v2 parser detects forum post boundaries: bold usernames, `## Post by` headings, etc.) |
| **Data** | Forum Q&A, NOT structured Question/Reasoning/Answer |
| **Tokens** | 102,691,746 (102.7M) |
| **v1 eval result** | exact_match strict=0.0%, flexible=0.0% (only 5 training steps) |
| **v2 eval result (with chat template)** | exact_match strict=0.0%, flexible=0.0% |
| **Status** | COMPLETE |

**Note on v1 vs v2**: The v1 parser required `## Question` headings which only 1.2% of records had. The v2 parser detects diverse forum post boundaries (bold usernames with dates, `## Post by` headings, etc.) and achieves 80.7% parse rate. Step names were bumped to `*_v2` to force re-execution.

### 5. Mathhelpforum Q/R/A SFT (correct prompt)

Chat SFT on rephraser-extracted markdown using the **correct** math-specific Q/R/A prompt. The extraction prompt asks the rephraser to restructure forum threads into `## Question / ## Reasoning / ## Answer` sections, synthesizing expert replies into coherent explanations.

| | |
|---|---|
| **Extract script** | `experiments/rephraser/mathhelpforum_extract_qra.py` |
| **SFT script** | `experiments/rephraser/mathhelpforum_qra_sft.py` |
| **Extraction prompt** | Math Q/R/A restructuring (spec `4e6fd004`) |
| **What it does** | Q/R/A extraction -> parse into chat turns -> tokenize -> SFT -> GSM8K eval |
| **Parse rate** | 95.1% (from 1,536-record sample across 5 representative shards) |
| **Extraction output** | `gs://marin-us-central1/documents/mathhelpforum_qra_4e6fd004-44f6e5/` (1005 shards, COMPLETE) |
| **Postprocessed** | `gs://marin-us-central1/processed/mathhelpforum_qra_4e6fd004-200a38/` (COMPLETE) |
| **Tokens** | 117,354,706 (117.4M) |
| **Training steps** | 447 (1 epoch) |
| **Status** | COMPLETE |

### 6. GSM8K Plain-Text SFT (no chat template)

Same as experiment 2 but uses plain text `Q: ... A: ...` format for both training and evaluation. Tests whether avoiding chat template tokens fixes the empty-generation problem.

| | |
|---|---|
| **Script** | `experiments/rephraser/gsm8k_sft_plaintext.py` |
| **What it does** | Convert GSM8K to plain text -> tokenize (no chat template) -> SFT -> GSM8K eval (plain text) |
| **Data** | `openai/gsm8k` train split, formatted as `Q: {question}\nA: {answer}`, 7,473 examples |
| **Tokens** | 2,742,259 (2.7M) |
| **Training steps** | 11 (1 epoch) |
| **Checkpoint** | `gs://marin-us-central1/checkpoints/gsm8k-plaintext-sft-1e20-qwen3-a40a32/` |
| **Eval result** | exact_match strict=0.0%, flexible=0.076% |
| **Non-empty generations** | 0/20 |
| **Status** | COMPLETE |

### 7. Mathhelpforum Q/R/A Plain-Text SFT (no chat template)

Same as experiment 5 but uses plain text format. Q/R/A content is formatted as `Q: {question}\nA: {reasoning}. The answer is {answer}.` instead of chat messages.

| | |
|---|---|
| **Script** | `experiments/rephraser/mathhelpforum_qra_sft_plaintext.py` |
| **What it does** | Q/R/A extraction -> parse to plain text Q/A -> tokenize (no chat template) -> SFT -> GSM8K eval (plain text) |
| **Data** | Same Q/R/A extraction as experiment 5, reformatted as plain text |
| **Tokens** | 110,433,895 (110.4M) |
| **Training steps** | 421 (1 epoch) |
| **Checkpoint** | `gs://marin-us-central1/checkpoints/mathhelpforum-qra-plaintext-sft-1e20-qwen3-8b0030/` |
| **Eval result** | exact_match strict=0.0%, flexible=0.0% |
| **Non-empty generations** | 0/20 |
| **Status** | COMPLETE |

---

## Data Extraction Parse Rates

| Extraction Method | Parse Rate | Parser | Failure Modes |
|---|---|---|---|
| Generic prompt (spec `71938abf`) v1 parser | 1.2% | Required `## Question` headings | Rephraser output uses diverse forum formats, not structured headings |
| Generic prompt (spec `71938abf`) v2 parser | 80.7% | Detects bold usernames, `## Post by`, blockquote boundaries | 19.3% have no detectable post boundaries |
| Q/R/A prompt (spec `4e6fd004`) | **95.1%** | Matches `## Question [N]` / `## Reasoning [N]` / `## Answer [N]` headings (with optional number) | See breakdown below |

### Q/R/A Parse Failure Breakdown (from 76 failures in 1,536-record sample)

| Failure Type | Count | Description |
|---|---|---|
| Question only | 62 | `## Question` present but no `## Reasoning` or `## Answer` — rephraser extracted the question but not the answer |
| No headings | 3 | No `## Question` heading at all — rephraser produced unstructured output |
| Other | 11 | Has headings but content doesn't parse cleanly |

The Q/R/A prompt achieves 14.4pp higher parse rate than the generic prompt's v2 parser (95.1% vs 80.7%). The structured `## Question / ## Reasoning / ## Answer` format is more amenable to systematic parsing. The parser accepts both plain (`## Question`) and numbered (`## Question 1`) heading variants.

---

## Results

### Levanter Eval Results (INVALID — bug)

These were the original eval results from the Levanter inference engine. **All are invalid** due to only processing 2/1319 test samples (see "Inference Engine Bug" section below).

| Condition | strict | flexible | Training Steps | Eval Format | Status |
|---|---|---|---|---|---|
| Baseline (no SFT) | 0.0% | 0.0% | 0 | plain text | INVALID |
| GSM8K SFT (chat template eval) | 0.0% | 0.0% | 13 | chat template | INVALID |
| Resiliparse SFT | 0.0% | 0.0% | 954 | plain text | INVALID |
| GSM8K plain-text SFT | 0.0% | 0.076% | 11 | plain text | INVALID |
| Q/R/A plain-text SFT | 0.0% | 0.0% | 421 | plain text | INVALID |

### vLLM Re-evaluation Results (CORRECTED — full 1319 test examples)

Re-evaluated with vLLM Docker sidecar on TPU v5p-8 (see "Re-evaluation with vLLM" section for fix details).

#### Plain text eval (no chat template)

| Condition | strict-match | flexible-extract | Tokens | Steps | Non-empty | Status |
|---|---|---|---|---|---|---|
| Baseline (no SFT) | **2.12%** (28/1319) | **2.88%** (38/1319) | 0 | 0 | 1319/1319 | COMPLETE |
| GSM8K plaintext SFT | 1.90% (25/1319) | 2.58% (34/1319) | 2.7M | 11 | 1319/1319 | COMPLETE |
| Q/R/A plaintext SFT | 1.74% (23/1319) | 2.27% (30/1319) | 110.4M | 421 | 1319/1319 | COMPLETE |
| Resiliparse SFT | 2.12% (28/1319) | 2.65% (35/1319) | 250.1M | 954 | 1319/1319 | COMPLETE |

#### Chat template eval

| Condition | strict-match | flexible-extract | Tokens | Steps | Non-empty | Status |
|---|---|---|---|---|---|---|
| GSM8K chat SFT | 0.0% (0/1319) | 0.0% (0/1319) | 3.2M | 12 | **0/1319** | COMPLETE |
| Q/R/A chat SFT | 0.076% (1/1319) | 0.91% (12/1319) | 117.4M | 447 | **1319/1319** | COMPLETE |

**Key findings**:
- All plain text models generate non-empty text for every test example (1319/1319), not just 2/20 as the buggy Levanter eval suggested
- All plain text models score ~2% — the model (~1.4B params) is too small for multi-step math reasoning
- None of the SFT variants improve over the baseline; all differences are within noise
- The overlap between which questions each model gets right is surprisingly low (only 1 question correct across all 4 models)
- GSM8K chat SFT produces **all empty generations** (0/1319) — chat template completely breaks generation with only 12 SFT steps
- Q/R/A chat SFT generates non-empty text (1319/1319) but produces **forum thread summaries** instead of math solutions — most "correct" answers are coincidental (see sample outputs below)

### Sample Outputs (vLLM, full evaluation)

#### Correct examples

Most "correct" answers arrive at the right number through **wrong reasoning** — the model gets lucky with its arithmetic rather than solving the problem correctly.

**All 4 models correct — doc 1314** (target: 8)
> Q: John had a son James when he was 19. James is now twice as old as his sister Dora, who will turn 12 in 3 years. How old will John's youngest son, who was born when John was 32, be in 3 years?

| Model | Response | Correct? |
|---|---|---|
| Baseline | "John's oldest son, who was born when he was 32, will be 12 in 3 years. 12 - 32 = 8. The answer is 8." | Yes (wrong math: 12-32=-20, not 8) |
| QRA plaintext | "He is now twice as old as his sister Dora, who will turn 12 in 3 years. So he is 19 - 12 = 8. The answer is 8." | Yes (wrong math: 19-12=7, not 8) |

All models output "8" but through incorrect reasoning chains. The correct solution is: Dora is 12-3=9, James is 9*2=18, John is 18+19=37, youngest son is 37-32=5, in 3 years = 5+3=8.

**Simple arithmetic — doc 776** (target: 800)
> Q: The price of a laptop is $1000. If you get a 20% discount, how much do you have to pay?

> Baseline: "The price of a laptop is $1000. If you get a 20% discount, then you have to pay $1000 - 20% = $800. The answer is $800."

One of the rare examples where the model's reasoning is approximately correct.

**Lucky correct — doc 1101** (target: 16)
> Q: Amber, Micah, and Ahito ran 52 miles in total. Amber ran 8 miles. Micah ran 3.5 times what Amber ran. How many miles did Ahito run?

> Baseline: "Ahito ran 52 miles. He ran 8 miles. 52 x 8 = 16 miles. The answer is 16."

Gets the right answer (16) but through nonsensical arithmetic (52*8=416, not 16). The correct solution is 52 - 8 - (3.5*8) = 52 - 8 - 28 = 16.

#### Failure modes

**1. Wrong arithmetic (most common)**

The model mimics CoT format and produces "The answer is X" but gets the math wrong. 77% of responses (1014/1319) contain the phrase "The answer is" — the model learned the format but not the computation.

> Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder at $2 per egg. How much does she make? (target: 18)

| Model | Answer | Reasoning |
|---|---|---|
| Baseline | $4 | "$2 * 16 = $4" (wrong: 2*16=32) |
| GSM8K plaintext | $16 | "$2 * 16 = $16" (wrong: 2*16=32) |
| QRA plaintext | $8 | "$16 - 2 = 8" (wrong formula) |
| Resiliparse | $8 | "16 - 2 = 8" (wrong formula) |

All models skip the subtraction step (16-3-4=9 eggs remaining) and directly combine numbers incorrectly.

> Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total? (target: 3)

| Model | Answer | Reasoning |
|---|---|---|
| Baseline | 8 | "2 blue + 2 white = 4, then 4 blue + 4 white = 8" (doubles everything) |
| GSM8K plaintext | 4 | "2 * 2 = 4" |
| QRA plaintext | 2 | "The answer is 2 bolts of blue and half that much white fiber" (restates question) |
| Resiliparse | 4 | "2 + 1 = 4" (wrong: 2+1=3) |

**2. Repetition loops**

~1% of responses enter a degenerate loop, repeating the same phrase until hitting the token limit.

> Q: Charlie wants to sell beeswax candles... (target: 20)

> Baseline: "Charlie started with 10 candles. After selling 20, he made $10.00. So he made $10.00 - 20 = $2.00. So he made $2.00 - 20 = $1.00. So he made $1.00 - 20 = $0.50. So he made $0.50 - 20 = $0.25. So he made $0.25 - 20 = $0.25. So he made $0.25 - 20 = $0.25..."

> GSM8K plaintext (doc 20): "2/3 = 1/3 = 1/3 = 1/3 = 1/3 = 1/3 = 1/3 = 1/3..."

**3. SFT-specific behaviors**

QRA plaintext SFT sometimes produces structured-answer language reflecting its training data:

> Q: A robe takes 2 bolts of blue fiber... (target: 3)

> QRA plaintext: "**The original question asks for** the number of bolts of blue and white fiber. The answer is 2 bolts of blue and half that much white fiber. The answer is 2."

**4. Chat template: complete generation failure**

GSM8K chat SFT produces **empty responses for all 1319 test examples**. The chat template tokens cause the model to immediately emit an EOS token. This model was pretrained on plain text only — it has never seen chat template special tokens during pretraining, so the chat-formatted prompts are out-of-distribution.

#### SFT occasionally helps (but randomly)

Where SFT models got the right answer but baseline didn't:

> Q: Tracy used a piece of wire 4 feet long. The wire was cut into pieces 6 inches long. How many pieces? (target: 8)

| Model | Answer | Correct? |
|---|---|---|
| Baseline | 12 | No ("4 + 6 = 12") |
| GSM8K plaintext | **8** | Yes ("4 * 6 = 24, 24 - 4 = 8" — wrong math, lucky answer) |
| QRA plaintext | **8** | Yes ("4 + 6 = 8" — wrong math, lucky answer) |
| Resiliparse | 12 | No ("6 + 4 = 12") |

> Q: Cody eats 3x as many cookies as Amir. Amir eats 5. How many together? (target: 20)

| Model | Answer | Correct? |
|---|---|---|
| Baseline | 10 | No ("5 + 3 = 10") |
| GSM8K plaintext | **20** | Yes ("5 + 3 = 10, 10 + 3 = 20" — wrong reasoning, lucky answer) |
| QRA plaintext | 3 | No (loops: "Amir eats 5, Cody eats 3...") |
| Resiliparse | 15 | No ("5 + 3 = 10, 5 + 3 = 15") |

**The overlap between correct answers is remarkably low**: only **1 question** (doc 1314) is correct across all 4 models. Each model gets a different random subset of ~25-38 questions right, mostly through lucky arithmetic.

#### Chat template eval: two very different failure modes

**GSM8K chat SFT (12 steps): complete generation failure**

All 1319 responses are empty. The model immediately emits EOS. With only 12 gradient steps of chat-formatted SFT, the model didn't learn enough about the chat template to generate anything — the special tokens are completely out-of-distribution for a model pretrained on plain text.

**Q/R/A chat SFT (447 steps): forum thread summaries instead of math**

All 1319 responses are non-empty, but the model generates **mathhelpforum-style thread summaries** instead of solving the math problem. 12/1319 answers are coincidentally correct by flexible-extract (0.91%), but 8 of those 12 have the same answer: **6**. The model learned a "default answer" from the training data.

Typical response format (doc 0, target=18):
> "The original poster presented a series of problems involving counting, addition, subtraction, and percentages. The thread contains only the original question and a brief reply indicating that the thread was closed. No further discussion or solution is provided in the excerpt."

The model learned to describe forum threads rather than solve the math within them. Examples of "correct by coincidence":

| Doc | Question | Target | Model's actual response |
|---|---|---|---|
| 433 | 15 gallons ÷ 5 containers, Josey needs 1/4. How many pints? | 6 | "No further discussion...No additional discussion...**Answer:** 6" |
| 486 | James runs 12mi/day, 5 days/week at 10mph. Hours/week? | 6 | "The correct answer is **6**" (wrong — answer should be 6, but reasoning is about trees/parking) |
| 488 | 27 puppies, 2/3 Pomeranians, 1/3 girls. How many girl Poms? | 6 | "No further discussion...**Answer:** 6" |
| 543 | Jairus gets $0.8, Jenny $0.5 per task. Difference after 20? | 6 | "**Answer:** 6" (target is 6, but correct calc is 20*(0.8-0.5)=$6) |

Only doc 543 has a target that makes mathematical sense given the model's "default 6" behavior. The other correct answers are pure coincidence.

The comparison questions from the plain text analysis all fail under chat eval:

| Doc | Question | Plain text baseline | QRA chat SFT |
|---|---|---|---|
| 0 | Duck eggs ($18) | $4 (wrong math) | "The original poster presented..." (no answer) |
| 1 | Robe fiber (3) | 8 (wrong math) | "The original poster presented..." (no answer) |
| 1314 | John's son (8) | 8 (lucky) | 2 (wrong) |

### Takeaway

The ~1.4B model consistently generates CoT-formatted text in plain text mode but cannot perform multi-step arithmetic. At ~2% accuracy, performance is indistinguishable from random guessing. SFT on math data (GSM8K train, mathhelpforum Q/R/A, or resiliparse text) does not meaningfully improve GSM8K performance at this model scale.

Chat template evaluation reveals two failure modes depending on SFT duration:
- **Too few steps** (12 steps, GSM8K chat SFT): model emits EOS immediately (0/1319 non-empty)
- **Enough steps** (447 steps, Q/R/A chat SFT): model generates non-empty text but in the wrong domain — it produces forum thread summaries instead of math solutions, reflecting its mathhelpforum training data

### Other Notes

**Very few gradient steps for GSM8K SFT**: The GSM8K train set is small (~7,473 examples, ~3.18M tokens). At batch_size=64, seq_len=4096, that's only **13 gradient updates** for 1 epoch. The model barely sees the data. This is by design (1-epoch SFT) but worth noting as a limiting factor.

---

## File Reference

| File | Purpose |
|---|---|
| `gsm8k_baseline_eval.py` | Eval pretrained checkpoint on GSM8K (no SFT) |
| `gsm8k_sft.py` | SFT on GSM8K train set + eval |
| `mathhelpforum_resiliparse_sft.py` | SFT on resiliparse plain text + eval |
| `mathhelpforum_extract.py` | Extraction pipeline (wrong generic prompt, spec `71938abf`) |
| `mathhelpforum_sft.py` | SFT on generic-prompt extraction + eval |
| `mathhelpforum_extract_qra.py` | Extraction pipeline (correct Q/R/A prompt, spec `4e6fd004`) |
| `mathhelpforum_qra_sft.py` | SFT on Q/R/A extraction + eval (chat template) |
| `gsm8k_sft_plaintext.py` | SFT on GSM8K train set + eval (plain text, no chat template) |
| `mathhelpforum_qra_sft_plaintext.py` | SFT on Q/R/A extraction + eval (plain text, no chat template) |
| `debug_generation.py` | Debug script: load checkpoints directly and generate on test examples |

### GCS Paths

| What | Path |
|---|---|
| Pretrained checkpoint | `gs://marin-us-central1/exp2166-.../checkpoints/step-44660` |
| GSM8K SFT checkpoint | `gs://marin-us-central1/checkpoints/gsm8k-sft-1e20-qwen3-e89008/` |
| Resiliparse SFT checkpoint | `gs://marin-us-central1/checkpoints/mathhelpforum-resiliparse-sft-1e20-qwen3-56a96f/` |
| Generic extraction output | `gs://marin-us-central1/documents/mathhelpforum_chat_v2-9ca368/` |
| Q/R/A extraction output | `gs://marin-us-central1/documents/mathhelpforum_qra_4e6fd004-44f6e5/` |
| Q/R/A postprocessed | `gs://marin-us-central1/processed/mathhelpforum_qra_4e6fd004-200a38/` |
| GSM8K plaintext SFT checkpoint | `gs://marin-us-central1/checkpoints/gsm8k-plaintext-sft-1e20-qwen3-a40a32/` |
| Q/R/A plaintext SFT checkpoint | `gs://marin-us-central1/checkpoints/mathhelpforum-qra-plaintext-sft-1e20-qwen3-8b0030/` |

---

## Inference Engine Bug (CRITICAL)

**Discovered**: 2026-03-05

### Root cause

The Levanter inference engine's `generate()` method (`lib/levanter/src/levanter/inference/engine.py:1016`) calls `_prefill_batch()` only **once** before entering the decode loop. The prefill buffer is sized to `max_prefill_size` which defaults to `page_table.max_len_per_seq` = `max_seq_len` = **2048 tokens**.

With GSM8K 8-shot CoT prompts at ~700 tokens each, only **2 prompts fit** in the 2048-token prefill buffer:
- Request 0 (721 tokens): offset=0 → 721 < 2048 → admitted
- Request 1 (684 tokens): offset=721 → 1405 < 2048 → admitted
- Request 2 (707 tokens): offset=1405 → 2112 > 2048 → **BREAK** (line 908)

The remaining 1317 requests are **never processed**. After the first 2 finish generating, the decode loop detects stagnation (0 new tokens for 2 iterations) and exits. The 1317 unprocessed requests get empty token lists `[]`, which decode to empty strings.

### Evidence

- **All experiments** (baseline, GSM8K SFT, QRA SFT, resiliparse SFT) show exactly 2/1319 non-empty: always doc_id=0 and doc_id=1 (the first two requests submitted)
- Pattern is 100% consistent across chat template and plain text evals
- Non-empty samples are always the same two test questions regardless of SFT condition

### Fix applied

In `lib/levanter/src/levanter/eval_harness.py`:

1. Set `max_prefill_size = max_seqs * max_length` (32 * 2048 = 65536) so a full batch of prompts fits in the prefill buffer
2. Batch generation requests by `max_seqs_in_prefill` (16), calling `engine.generate()` in a loop so all 1319 requests get processed

### Impact

**All eval results in this report are invalid.** The reported 0.0% scores reflect only 2 test examples being evaluated, not the full 1319. Re-evaluation with the fix is needed.

---

## Re-evaluation with vLLM (2026-03-09)

### Why vLLM instead of Levanter

After discovering the inference engine bug, initial re-evaluation attempts (v1-v5) tried to fix the Levanter inference path. All failed:

- **v1-v4**: Various OOM, batch-sizing, and prefill buffer fixes. Each revealed new issues.
- **v5**: Applied `max_prefill_size=16384`, `hbm_utilization=0.3`, and reference attention fallback. The XLA Pallas ragged paged attention kernel requires `num_combined_kv_heads` to factor into `{1,2,4,8}` or be divisible by 8. Our model has 14 KV heads → `2*14=28` (for bf16 packing), and `28/2=14` is NOT in `{1,2,4,8}` and `14%8≠0`. This forced a reference attention fallback that would take ~14 hours per checkpoint. Jobs ran but got stuck in an infinite lm-eval config override loop and ultimately FAILED.

**Root insight**: Levanter's eval_harness.py literally says *"Levanter does not support generation (use VLLM or something)"*. GSM8K CoT is a `generate_until` task. The standard Marin eval pipelines (`default_base_eval`, `default_sft_eval`) already route generation tasks through vLLM. The reeval scripts simply used the wrong evaluator function.

**Fix**: Switched from `evaluate_levanter_lm_evaluation_harness()` to `evaluate_lm_evaluation_harness()` in both `reeval_gsm8k.py` and `reeval_gsm8k_chat.py`. The vLLM evaluator runs an OpenAI-compatible vLLM server as a Docker sidecar, and lm-eval connects via HTTP (`local-completions` or `local-chat-completions`).

### Fixes required for vLLM eval path

Three additional issues had to be fixed before vLLM eval worked:

#### Fix 1: TPU detection on Ray workers (v6 failure)

**Error**: `RuntimeError: Cannot determine default docker image for unknown resource type 'unknown'`

**Root cause**: `_detect_tpu_environment()` in `lib/marin/src/marin/inference/vllm_server.py` only checked the `TPU_NAME` environment variable, which is not set on Ray-managed TPU workers.

**Fix**: Updated `_detect_tpu_environment()` to also check for TPU device files:
- `/dev/vfio/[0-9]*` (v4/v5p TPUs use VFIO passthrough)
- `/dev/accel[0-9]*` (v5e/v6e TPUs use accel devices)

Also added forwarding of `MARIN_VLLM_MODE` env var in `lm_evaluation_harness_evaluator.py` so the Fray job inherits the vLLM backend mode from the outer environment.

#### Fix 2: Model architecture tag (v7 failure)

**Error**: `KeyError: 'layers.6.self_attn.k_norm.weight'`

**Root cause**: All checkpoint `config.json` files had `"architectures": ["LlamaForCausalLM"]`, but the model is actually Qwen3 with QK-normalization (`k_norm.weight`, `q_norm.weight`). vLLM dispatches the model loader based on the `architectures` field, so it used Llama's loader which doesn't support QK-norm weights.

**Fix**: Updated `config.json` in all 6 GCS checkpoints, changing `"architectures": ["LlamaForCausalLM"]` to `"architectures": ["Qwen3ForCausalLM"]`:

| Checkpoint | config.json path |
|---|---|
| Baseline (exp2166) | `.../exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0/hf/step-44758/config.json` |
| GSM8K plaintext SFT | `.../checkpoints/gsm8k-plaintext-sft-1e20-qwen3-a40a32/hf/step-10/config.json` |
| Q/R/A plaintext SFT | `.../checkpoints/mathhelpforum-qra-plaintext-sft-1e20-qwen3-8b0030/hf/step-421/config.json` |
| Resiliparse SFT | `.../checkpoints/mathhelpforum-resiliparse-sft-1e20-qwen3-56a96f/hf/step-954/config.json` |
| GSM8K chat SFT | `.../checkpoints/gsm8k-sft-1e20-qwen3-e89008/hf/step-12/config.json` |
| Q/R/A chat SFT | `.../checkpoints/mathhelpforum-qra-sft-1e20-qwen3-c06b54/hf/step-447/config.json` |

#### Fix 3: MARIN_VLLM_MODE forwarding

**File**: `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`

Added forwarding of `MARIN_VLLM_MODE` env var from the outer environment to the Fray job, so the TPU worker uses the same vLLM backend (Docker vs native) as intended.

### Re-evaluation scripts (current state)

Both scripts now use `evaluate_lm_evaluation_harness` (vLLM) instead of `evaluate_levanter_lm_evaluation_harness` (Levanter):

**`reeval_gsm8k.py`** — 4 plain-text checkpoints, `apply_chat_template=False`:
- `reeval8-baseline` (exp2166 pretrained)
- `reeval8-gsm8k-plaintext-sft`
- `reeval8-qra-plaintext-sft`
- `reeval8-resiliparse-sft`

**`reeval_gsm8k_chat.py`** — 2 chat-template checkpoints, `apply_chat_template=True`:
- `reeval8-gsm8k-chat-sft`
- `reeval8-qra-chat-sft`

### Re-evaluation attempt history

| Version | Evaluator | Outcome |
|---|---|---|
| v1-v4 | Levanter | Various OOM / batch / prefill failures |
| v5 | Levanter (ref attention) | Stuck in lm-eval config override loop, FAILED |
| v6 | vLLM (Docker) | FAILED: TPU_NAME not set on Ray workers → unknown resource type |
| v7 | vLLM (Docker, TPU fix) | FAILED: LlamaForCausalLM loader can't load Qwen3 k_norm/q_norm weights |
| v8 | vLLM (Docker, Qwen3 arch) | FAILED: `max_gen_toks=4096` + 8-shot prompt (~700 tok) exceeds `max_model_len=4096` → HTTP 400 from vLLM `/completions`. Also: chat checkpoints missing `chat_template` in tokenizer → crash on `/chat/completions`. |
| v9 | vLLM (Docker, max_gen_toks=1024) | **SUCCESS** — plain text evals complete (4/4 checkpoints). Chat evals: added Llama 3.1 chat template to tokenizer_config.json in GCS, RUNNING. |

### Fixes applied in v8→v9

#### Fix 4: max_gen_toks overflow (v8 plain text failure)

**Error**: `requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: http://127.0.0.1:.../v1/completions`

**Root cause**: `DEFAULT_LM_EVAL_MODEL_KWARGS` has `max_gen_toks=4096`. With 8-shot GSM8K prompts at ~700 tokens, lm-eval requests `max_tokens=4096`, but `700 + 4096 = 4796 > max_model_len(4096)`. vLLM rejects requests where `max_tokens + prompt_tokens > max_model_len`.

**Fix**: Created custom engine kwargs in both reeval scripts: `GSM8K_ENGINE_KWARGS = {"max_model_len": 4096, "max_gen_toks": 1024}`. 1024 tokens is more than enough for GSM8K CoT answers (typically < 500 tokens).

#### Fix 5: Missing chat_template in tokenizer (v8 chat failure)

**Error**: `"As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if the tokenizer does not define one. None"`

**Root cause**: The Llama3 base tokenizer doesn't have a `chat_template` field in `tokenizer_config.json`. vLLM's `/chat/completions` endpoint (used when `apply_chat_template=True`) requires one.

**Fix**: Added the Llama 3.1 chat template (from `experiments/chat_templates/llama3pt1_chat_template.py`, with `{% generation %}`/`{% endgeneration %}` Levanter-specific tags stripped) to `tokenizer_config.json` in both chat checkpoint directories:
- `gs://marin-us-central1/checkpoints/gsm8k-sft-1e20-qwen3-e89008/hf/step-12/tokenizer_config.json`
- `gs://marin-us-central1/checkpoints/mathhelpforum-qra-sft-1e20-qwen3-c06b54/hf/step-447/tokenizer_config.json`

### Files changed

| File | Change |
|---|---|
| `experiments/rephraser/reeval_gsm8k.py` | Switched to `evaluate_lm_evaluation_harness` (vLLM), added `engine_kwargs` with `max_gen_toks=1024` |
| `experiments/rephraser/reeval_gsm8k_chat.py` | Same switch, plus `apply_chat_template=True` |
| `lib/marin/src/marin/inference/vllm_server.py` | `_detect_tpu_environment()` checks device files, not just `TPU_NAME` |
| `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py` | Forward `MARIN_VLLM_MODE` env var to Fray job |
| 6x GCS `config.json` | `architectures` changed from `LlamaForCausalLM` to `Qwen3ForCausalLM` |
| 2x GCS `tokenizer_config.json` | Added `chat_template` field (Llama 3.1 template) for chat checkpoints |

### Re-evaluation results (vLLM, full 1319 test examples)

#### Plain text eval (no chat template) — COMPLETE

| Checkpoint | GSM8K CoT strict-match | GSM8K CoT flexible-extract |
|---|---|---|
| Baseline (no SFT) | **2.12%** | **2.88%** |
| GSM8K plaintext SFT (11 steps) | 1.90% | 2.58% |
| Q/R/A plaintext SFT (421 steps) | 1.74% | 2.27% |
| Resiliparse SFT (954 steps) | 2.12% | 2.65% |

All scores are ~2%. The model is too small (~1.4B params) for meaningful 8-shot CoT math reasoning. None of the SFT variants improve over the baseline — all differences are within noise at this accuracy level.

#### Chat template eval — RUNNING

| Checkpoint | GSM8K CoT strict-match | GSM8K CoT flexible-extract |
|---|---|---|
| GSM8K chat SFT (12 steps) | *pending* | *pending* |
| Q/R/A chat SFT (447 steps) | *pending* | *pending* |

---

## Known Issues

1. **Wrong extraction prompt**: The initial mathhelpforum extraction used a generic "extract HTML to markdown" prompt instead of a math-specific Q/R/A prompt. The output preserves raw forum threads rather than restructuring into Question/Reasoning/Answer format.

2. **Parser v1 failure**: The original `_parse_markdown_to_messages` required `## Question` headings, but the rephraser output uses diverse forum post formats. Only 1.2% of records parsed. Fixed in v2 (80.7% parse rate).

3. **Tokenization failure**: The v2 tokenization step crashed during the first shard (`part-00000.tmp`). Root cause unclear (logs truncated). Cleaned up stale output and relaunched.

4. **Eval OOM**: The initial eval setup used f32 params and large inference buffer, causing OOM on v5p-8 TPU. Fixed by switching to bf16 params, reducing max_length 4096->2048, reducing InferenceEngine max_seqs 256->32 and hbm_utilization 0.5->0.3.

5. **0% GSM8K scores**: All completed SFT experiments scored 0% on GSM8K 8-shot CoT. **Now known to be caused by the inference engine bug** — only 2/1319 samples were actually evaluated. Re-evaluation in progress with vLLM (see "Re-evaluation with vLLM" section).

6. **Wrong architecture tag in checkpoints**: All HF checkpoints were exported with `"architectures": ["LlamaForCausalLM"]` despite being Qwen3 models with QK-normalization. This prevented vLLM from loading them. Fixed by updating all 6 `config.json` files to `"architectures": ["Qwen3ForCausalLM"]`.

7. **TPU detection on Ray workers**: The vLLM server code only checked `TPU_NAME` env var to detect TPU hardware, which is not set on Ray-managed TPU workers. Fixed by also checking for device files (`/dev/vfio/*`, `/dev/accel*`).

8. **max_gen_toks overflow**: `DEFAULT_LM_EVAL_MODEL_KWARGS` sets `max_gen_toks=4096`, which combined with 8-shot GSM8K prompts (~700 tokens) exceeds `max_model_len=4096`. vLLM's `/completions` endpoint returns HTTP 400. Fixed by using custom `max_gen_toks=1024` in reeval scripts.

9. **Missing chat_template in tokenizer**: Llama3 base tokenizer lacks a `chat_template` field. As of transformers v4.44, vLLM's `/chat/completions` endpoint requires one. Fixed by adding Llama 3.1 chat template to the two chat checkpoint `tokenizer_config.json` files in GCS.
