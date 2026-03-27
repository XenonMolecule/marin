# Math Extraction SFT Regression Analysis

**Date**: 2026-03-23 (updated with lr5e-7_bs64 comparison)
**Models compared**: Qwen3-0.6B-Base (baseline) vs lr5e-7_bs16 vs **lr5e-7_bs64** (new best)
**Training data**: 2.95B tokens from 42 math web domains, LLM-extracted Q/R/A format

## Executive Summary

Math extraction SFT can **improve** algebra performance over baseline when using the right hyperparameters (lr5e-7_bs64: +3.3pp math_verify, +2.1pp exact_match on algebra), but training data quality issues still cap the gains and cause regressions on simpler tasks.

After analyzing 3,267 eval problems across 3 benchmarks, ~900 training documents, and comparing two SFT configs (lr5e-7_bs16 vs lr5e-7_bs64) against baseline:

| Benchmark | Baseline | SFT (bs16) | Delta | SFT (bs64) | Delta | bs64 vs bs16 |
|-----------|----------|------------|-------|------------|-------|--------------|
| Algebra (EM) | 37.2% | 37.8% | +0.6pp | **39.3%** | **+2.1pp** | +1.5pp better |
| PreAlgebra (EM) | 46.7% | 41.6% | -5.2pp | **44.2%** | **-2.5pp** | 2.7pp better |
| GSM8K (flex) | 62.8% | 58.8% | -4.0pp | 58.9% | -3.9pp | ~same |

**Key finding**: Increasing batch size from 16→64 (same LR=5e-7) halves the PreAlgebra regression while doubling the Algebra gain, without changing the GSM8K gap at all. This suggests **two distinct mechanisms**: (1) a hyperparameter-sensitive regression that larger batches fix (algebra/prealgebra), and (2) a persistent data-quality-driven gap that no hyperparameter can fix (GSM8K).

**Root causes, ranked by impact:**

1. **Hyperparameter sensitivity** — the biggest factor; lr5e-7_bs64 recovers most of the regression seen in initial configs
2. **Domain mismatch** (45% physicsforums, 42% GeoGebra — neither teaches elementary math problem-solving)
3. **Data quality** (27% minimal docs <200 chars, 46% have no LaTeX, only 24% have complete Q/R/A structure)
4. **Wrong computation** (90%+ of remaining regressions are computational errors, not format issues)
5. **Repetition loops** (reduced 34% by larger batch: 19 vs 29 occurrences)
6. **Format mismatch** (training uses `## Question`/`## Reasoning`/`## Answer`; eval uses `Problem:`/`Solution:`/`\boxed{}`)

---

## 1. Failure Mode Breakdown

### 1.1 Regression Categories

| Category | Algebra | PreAlgebra | GSM8K | Total | % |
|----------|---------|------------|-------|-------|---|
| **wrong_computation** | 105 | 114 | 130 | 349 | 90.2% |
| **repetition_loop** | 20 | 5 | 4 | 29 | 7.5% |
| **truncation** | 5 | 2 | 2 | 9 | 2.3% |
| **Total regressions** | 130 | 121 | 136 | 387 | 100% |

The dominant failure mode is **wrong computation** — the SFT model arrives at a different (wrong) numerical answer. This is not a format or extraction issue; the model is genuinely computing incorrectly on problems the baseline solved correctly.

**Repetition loops** are a secondary issue, concentrated in algebra (15.4% of algebra regressions). The SFT model enters degenerate loops where it repeats the same mathematical expression or phrase until it runs out of tokens.

### 1.2 Per-Difficulty Breakdown

**Algebra** — SFT gains on medium problems but loses on easy and hard:

| Level | Baseline | SFT | Delta |
|-------|----------|-----|-------|
| Level 1 (easiest) | 76.3% | 71.1% | **-5.2pp** |
| Level 2 | 53.7% | 59.7% | +6.0pp |
| Level 3 | 44.4% | 49.4% | +5.0pp |
| Level 4 | 29.7% | 25.8% | -3.9pp |
| Level 5 (hardest) | 10.1% | 10.1% | 0.0pp |

**PreAlgebra** — SFT loses at every level except Level 5:

| Level | Baseline | SFT | Delta |
|-------|----------|-----|-------|
| Level 1 | 68.6% | 64.0% | -4.7pp |
| Level 2 | 66.1% | 59.9% | -6.2pp |
| Level 3 | 53.1% | 42.4% | **-10.7pp** |
| Level 4 | 40.8% | 37.7% | -3.1pp |
| Level 5 | 17.6% | 17.6% | 0.0pp |

**Key insight**: PreAlgebra Level 3 loses 10.7pp — the largest single regression. The SFT data likely contains very little content at this difficulty level (basic word problems, fraction arithmetic, order of operations) since the training data skews toward physics forums and advanced topics.

---

## 2. Training Data Quality

### 2.1 Structure Classification (n=868 postprocessed docs)

| Type | Count | % | Description |
|------|-------|---|-------------|
| tutorial | 237 | 27.3% | No Q/R/A structure, freeform text |
| minimal | 231 | 26.6% | <200 characters, barely any content |
| qa_complete | 210 | 24.2% | Full Q/R/A with all three sections |
| qa_no_answer | 176 | 20.3% | Has Q and R but missing/empty Answer |
| exercise_list | 11 | 1.3% | Lists of exercises without solutions |
| qa_no_reasoning | 2 | 0.2% | Q and A but no reasoning |
| garbled | 1 | 0.1% | HTML artifacts |

**Only 24.2% of training data has the intended complete Q/R/A structure.** The rest is either too short, missing key sections, or in freeform tutorial format. This means 3/4 of the training signal is teaching the model to produce content that doesn't match the extraction prompt's intended output format.

### 2.2 Quality Metrics

| Metric | Count | % |
|--------|-------|---|
| Has any LaTeX (`$...$`) | 465 | 53.6% |
| Has display math (`$$...$$`) | 289 | 33.3% |
| Has HTML artifacts | 3 | 0.3% |
| Has `<think>` tags (leaked) | 13 | 1.5% |
| **No LaTeX at all** | **403** | **46.4%** |

**46% of training documents contain zero LaTeX.** For a math SFT dataset, this is a critical quality issue. These docs are likely GeoGebra tool pages, non-English content, or minimal descriptions that passed the 50-character minimum filter but contain no actual mathematical content.

### 2.3 Document Length Distribution

| Stat | Value |
|------|-------|
| Mean | 1,298 chars |
| Median | 783 chars |
| p10 | 82 chars |
| p90 | 2,759 chars |
| Min | 51 chars (the postprocessing minimum) |

The median document is only 783 characters (~195 tokens). Many of these are too short to contain a meaningful math problem with worked solution.

### 2.4 Token Budget (Q/R/A docs only, n=388)

For docs that have Q/R/A structure, how are tokens distributed?

| Component | Avg % of tokens |
|-----------|----------------|
| Headers (`## Question`, etc.) | 4.7% |
| Question text | 31.9% |
| **Reasoning** | **55.8%** |
| Answer | 5.1% |
| Other | 2.7% |

The token budget is actually reasonable for Q/R/A docs — 56% goes to reasoning. But since only 24% of docs have this structure, the effective reasoning-token fraction across all training data is much lower (~13%).

### 2.5 Domain Distribution — The Smoking Gun

| Domain | Count | % |
|--------|-------|---|
| **physicsforums.com** | 392 | **45.2%** |
| **geogebra.org** (all subdomains) | 391 | **45.0%** |
| aaamath.com | 16 | 1.8% |
| symbolab.com | 16 | 1.8% |
| khanacademy.org | 11 | 1.3% |
| mathgoodies.com | 9 | 1.0% |
| mathwarehouse.com | 9 | 1.0% |
| homeschoolmath.net | 8 | 0.9% |
| engageny.org | 6 | 0.7% |
| Other (9 domains) | 10 | 1.2% |

**90% of the training data comes from just two sources: physicsforums.com and geogebra.org.** Neither is well-suited for elementary math problem-solving:

- **PhysicsForums** (45%): Contains physics-heavy discussions, often at university level (covariant derivatives, oscillation mechanics, electromagnetic fields). The math is correct but targets a completely different distribution than PreAlgebra/GSM8K.

- **GeoGebra** (45%): Mostly tool documentation pages, interactive applet descriptions, and non-English content. Many pages contain no actual math — just descriptions of GeoGebra commands or activities.

The domains that would actually help with eval benchmarks (aaamath, khanacademy, mathgoodies, homeschoolmath) collectively represent only **5%** of the sample.

---

## 3. Training Data Examples

### 3.1 Good Training Data

**Example: PhysicsForums — Hoop Oscillation** (physicsforums.com, 1,647 chars)

Well-structured Q/R/A with parallel axis theorem derivation:
```
## Question
We want to support a thin hoop by a horizontal nail and have the hoop make one
complete small-angle oscillation each 2.0 s. What must the hoop's radius be?

## Reasoning
The moment of inertia must be adjusted using the parallel-axis theorem:
$$I = I_{\text{center}} + md^2 = MR^2 + MR^2 = 2MR^2$$
Thus the period becomes:
$$T = 2\pi\sqrt{\frac{2R}{g}}$$
Solving for R: ... R ≈ 0.497 m

## Answer
R ≈ 0.497 m
```

This is correct, well-formatted, and has complete reasoning. But it teaches university-level physics, not the elementary algebra/arithmetic tested in evals.

### 3.2 Problematic Training Data

**Bad #1: GeoGebra Command Reference** (wiki.geogebra.org, 1,205 chars)
```
# AreCongruent Command
This article is about **GeoGebra command**.
## Command Categories
- 3D_Commands, Algebra Commands, Chart Commands...
## Syntax: AreCongruent( <Object>, <Object> )
```
No math, no reasoning, no problem-solving. Just software documentation.

**Bad #2: Minimal Description** (geogebra.org, 261 chars)
```
The Gradient Formula
Shows the derivation of the gradient formula, starting from
the definition of gradient as "rise over run".
```
Too short. No actual derivation shown despite claiming one.

**Bad #3: Non-English Content** (geogebra.org, 503 chars)
```
# Speiling av figur om en linje
Her skal du speile trekanten om linja ved hjelp av verktøyene...
```
Norwegian-language GeoGebra activity. The extraction prompt says to reject non-English content (`[NO_USEFUL_CONTENT]`), but this passed through.

**Bad #4: Empty Shells** (geogebra.org, 53-79 chars each)
```
# LAST ONE FOR SURE CALC PROJECT
Author: sydneyh
```
```
# fasci di parabole
Author: profcantone
esempio 23 pag.327
```
These are GeoGebra project stubs with no content. They passed the 50-character minimum but contain zero educational value.

---

## 4. Format Mismatch Analysis

### 4.1 Eval Format vs Training Format

**Eval prompt (minerva_math, 4-shot CoT):**
```
Problem:
Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.

Solution: The expressions inside each square root must be non-negative...
Therefore, the domain is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
[test problem here]

Solution:
```

**Training data format:**
```
# Could Someone Check This Answer?

## Question
We want to support a thin hoop by a horizontal nail...

## Reasoning
The replies explain that the moment of inertia must be adjusted...
$$I = I_{\text{center}} + md^2 = MR^2 + MR^2 = 2MR^2$$

## Answer
R ≈ 0.497 m
```

| Aspect | Eval | Training |
|--------|------|----------|
| Problem marker | `Problem:` | `## Question` |
| Solution marker | `Solution:` | `## Reasoning` |
| Answer format | `\boxed{...}` then `Final Answer: The final answer is X.` | `## Answer` section |
| Context | Self-contained single problem | Forum thread with meta-discussion |
| Difficulty | Elementary to intermediate | University-level physics/math |
| Loss masking | N/A | **None** — all tokens weighted equally |

The formats are completely different. The model never sees `Problem:`/`Solution:` pairs during training, and never sees `\boxed{}` answer formatting. The eval's 4-shot examples are supposed to teach the model these patterns at inference time, but the SFT has shifted the model's generation distribution away from this format.

---

## 5. Generation Behavior Shift

### 5.1 Response Length

| Task | Model | Mean | Median | p95 |
|------|-------|------|--------|-----|
| Algebra | Baseline | 399 | 374 | 691 |
| Algebra | **SFT** | **349** | **319** | **627** |
| PreAlgebra | Baseline | 363 | 317 | 701 |
| PreAlgebra | **SFT** | **304** | **271** | **609** |
| GSM8K | Baseline | 346 | 317 | 617 |
| GSM8K | **SFT** | **254** | **236** | **441** |

**SFT responses are systematically shorter** — 12-27% shorter across all tasks. On GSM8K, the median response drops from 317 to 236 characters (26% shorter). This suggests the model is producing less detailed reasoning chains, which would directly reduce accuracy on problems that require multi-step computation.

### 5.2 Answer Formatting

| Task | Model | Uses `\boxed{}` | Extraction artifacts |
|------|-------|-----------------|---------------------|
| Algebra | Baseline | 77.6% | 0.0% |
| Algebra | **SFT** | **66.5%** | 0.0% |
| PreAlgebra | Baseline | 83.7% | 0.0% |
| PreAlgebra | **SFT** | **72.4%** | 0.0% |
| GSM8K | Baseline | 0.3% | 0.0% |
| GSM8K | SFT | 0.0% | 0.0% |

**SFT reduces `\boxed{}` usage by 11pp.** Since minerva_math uses `\boxed{}` for answer extraction, fewer boxed answers means more extraction failures. However, the `math_verify` metric (which doesn't require `\boxed{}`) shows similar regressions, so this isn't the primary cause.

No training-format artifacts (`## Question`, `## Reasoning`, etc.) were found in eval outputs. The format contamination is more subtle — the model's generation style changes without introducing explicit training-format markers.

### 5.3 Repetition and Diversity

| Task | Model | Mean 4-gram diversity | Low diversity (<0.5) |
|------|-------|-----------------------|---------------------|
| Algebra | Baseline | 0.970 | 7 (0.6%) |
| Algebra | **SFT** | **0.960** | **32 (2.7%)** |
| PreAlgebra | Baseline | 0.947 | 20 (2.3%) |
| PreAlgebra | SFT | 0.949 | 23 (2.6%) |
| GSM8K | Baseline | 0.973 | 0 (0.0%) |
| GSM8K | **SFT** | 0.981 | **7 (0.6%)** |

SFT increases degenerate repetition in algebra (4.5x more low-diversity outputs). This accounts for the 20 repetition-loop regressions in algebra.

---

## 6. Side-by-Side Generation Comparisons

### 6.1 Regression: Wrong Computation (most common failure)

**Problem** (Algebra Level 3, doc_id=47): *Simplify $(2-2i)(5+5i)$, where $i^2 = -1$.*
- **Gold**: 20
- **Baseline**: Correctly expands: $10+10i-10i-10i^2 = 10+10i-10i+10 = \boxed{20}$ ✓
- **SFT**: Makes arithmetic error: $10+10i-10i-10i^2 = 10-10i-10(-1) = 10-10i+10 = \boxed{20-10i}$ ✗

The SFT model incorrectly introduces a $-10i$ term that should cancel. This is a basic algebraic simplification error that the baseline handles correctly.

**Problem** (GSM8K, doc_id=4): *Kylar buys 16 glasses at $5, every second glass 60% off. Total cost?*
- **Gold**: 64
- **Baseline**: Correctly computes: 8 × $5 + 8 × $3 = $40 + $24 = $64 ✓
- **SFT**: Misinterprets "every second glass" as geometric decay (each subsequent glass = 60% of previous), sums a geometric series, gets wrong answer ✗

The SFT model over-complicates a simple word problem, applying an unnecessarily sophisticated mathematical framework that leads to a wrong answer.

**Problem** (PreAlgebra Level 1, doc_id=11): *What is $8-4 \div 2-1$?*
- **Gold**: 5
- **Baseline**: Gets $8-2-1=7$ ✗ (makes order-of-operations error, subtracted wrong)
- **SFT**: Gets $8-2-1=\boxed{5}$ ✓ (correct!)

This is one case where SFT actually helps — the model gets a basic order-of-operations problem right that the baseline got wrong. But these improvements are outnumbered by regressions.

### 6.2 Regression: Repetition Loop

**Problem** (Algebra Level 2, doc_id=22): *$(17^6-17^5)\div16=17^x$. Find $x$.*
- **Gold**: 5
- **Baseline**: Factors correctly: $\frac{17^5(17-1)}{16} = \frac{17^5 \cdot 16}{16} = 17^5$, so $x=\boxed{5}$ ✓
- **SFT**: Starts correctly ($17^5 \cdot 16$) then enters loop: `=2^{16}\cdot(17^4)^2=2^{16}\cdot(17^4)^2=2^{16}\cdot(17^4)^2=...` ✗

The SFT model takes a correct intermediate step but then enters a degenerate algebraic manipulation loop, repeating the same expression until token limit.

**Problem** (Algebra Level 1, doc_id=258): *Two numbers sum to 40, differ by 12. Product?*
- **Gold**: 364
- **Baseline**: Sets up system, solves correctly: $x=26, y=14$, product = $\boxed{364}$ ✓
- **SFT**: Sets up system correctly, then enters LaTeX spacing loop: `\qquad\qquad\qquad\qquad...` ✗

### 6.3 Regression: GSM8K Computation Error

**Problem** (GSM8K, doc_id=119): *60 elves; 1/3 quit, then 10 more quit. How many left?*
- **Gold**: 30
- **Baseline**: 60/3=20 quit → 40 left → 40-10=30 ✓
- **SFT**: Incorrectly computes 60/3=20, then divides again: 20/3=6.666... and enters repeating decimal loop ✗

The SFT model applies division twice when the problem only calls for one subtraction.

### 6.4 Improvement: GSM8K Arithmetic

**Problem** (GSM8K, doc_id=0): *Janet's ducks lay 16 eggs/day. She eats 3, bakes 4. Sells rest at $2. Revenue?*
- **Gold**: 18
- **Baseline**: Incorrectly computes 16-3=13 eggs left, then separately 16-4=12, sells 12 → $24 ✗
- **SFT**: Correctly chains: 16-3=13, 13-4=9, 9×2=$18 ✓

This is a genuine improvement — the SFT model correctly sequences the subtraction operations.

---

## 7. Root Cause Assessment

### Contributing Factor 1: Domain Distribution Mismatch (HIGH IMPACT)

**Evidence**: 90% of training data comes from physicsforums.com (45%) and geogebra.org (45%). PhysicsForums content is university-level physics; GeoGebra content is software documentation. Neither teaches the elementary problem-solving patterns tested by PreAlgebra, GSM8K, or even most MATH Algebra problems.

**Mechanism**: The model learns the mathematical "style" of these domains — verbose physics derivations and tool descriptions — which overwrites the baseline's ability to do concise step-by-step arithmetic. This explains why PreAlgebra Level 3 (-10.7pp) is the hardest hit: these are practical word problems that require simple arithmetic, which is the exact opposite of what physicsforums teaches.

**Comparison with coding/medical domains**: The user noted that resiliparse SFT *improved* performance in coding and medical domains. This is consistent — code and medical web content is more likely to contain correct, structured information in formats that align with downstream eval tasks. Math forums contain correct content but in formats and difficulty levels that don't match eval benchmarks.

### Contributing Factor 2: Low-Quality Training Documents (HIGH IMPACT)

**Evidence**: 27% of docs are minimal (<200 chars), 46% have no LaTeX, 20% have Q/R/A but no Answer section, and 1.5% still contain leaked `<think>` tags. Only 24% of documents have complete Q/R/A structure.

**Mechanism**: The model spends most of its training capacity on documents that contain little to no mathematical reasoning signal. These documents dilute the useful signal and teach the model to produce short, low-information responses.

### Contributing Factor 3: Wrong Computation from Noisy Signal (MEDIUM IMPACT)

**Evidence**: 90% of regressions are wrong_computation, not format or extraction issues. The SFT model makes genuine arithmetic errors on problems the baseline solved correctly.

**Mechanism**: When the training data contains forum discussions where intermediate steps are wrong (a common pattern in "check my answer" forum threads) or where the extraction model synthesized reasoning incorrectly, the SFT model learns these error patterns. PhysicsForums threads often contain initial wrong answers that get corrected in replies — the extraction prompt tries to synthesize the correct answer, but may incorporate incorrect intermediate steps.

### Contributing Factor 4: Shorter Responses (MEDIUM IMPACT)

**Evidence**: SFT responses are 12-27% shorter than baseline across all tasks. `\boxed{}` usage drops by 11pp.

**Mechanism**: The training data's median length is 783 chars. Many documents are very short. This teaches the model that brief responses are appropriate, reducing the detailed step-by-step reasoning that is critical for multi-step math problems.

### Contributing Factor 5: Repetition Loops (LOW-MEDIUM IMPACT)

**Evidence**: 7.5% of regressions (29/387) involve degenerate repetition. Concentrated in algebra (20/130 = 15.4%).

**Mechanism**: The model learns to produce LaTeX-heavy mathematical expressions from the training data but sometimes gets stuck in repetitive generation loops. This is a known failure mode of continued pretraining on formatted content.

### Contributing Factor 6: Format Mismatch (LOW IMPACT)

**Evidence**: Eval uses `Problem:`/`Solution:`/`\boxed{}` format; training uses `## Question`/`## Reasoning`/`## Answer`. However, no explicit training-format artifacts appear in eval outputs.

**Mechanism**: The format difference is real but the 4-shot prompt at eval time successfully overrides the training format. The impact is indirect — the model's generation distribution shifts, producing less `\boxed{}` and shorter responses, but doesn't produce `## Question` headers in eval output.

---

## 8. Recommendations

### Priority 1: Fix Domain Distribution

**Action**: Filter or reweight training data to emphasize domains that match eval task difficulty.

- **Remove GeoGebra** entirely — 45% of training data with almost no mathematical problem-solving content
- **Down-weight PhysicsForums** — useful content but wrong difficulty level
- **Up-weight** elementary math sources: khanacademy, aaamath, mathgoodies, homeschoolmath, mathwarehouse
- **Add** sources with explicit worked solutions at appropriate difficulty: artofproblemsolving.com, purplemath, mathbits.com

**Expected impact**: HIGH — this addresses the root cause

### Priority 2: Raise Minimum Quality Threshold

**Action**: Tighten postprocessing filters.

- **Increase minimum length** from 50 to 500 characters — eliminates the 27% minimal docs
- **Require LaTeX presence** — eliminates the 46% no-math docs
- **Language filter** — reject non-English content that passes extraction
- **Require complete Q/R/A structure** — only train on docs with all three sections populated

**Expected impact**: HIGH — reduces noise in training signal by ~50%

### Priority 3: Mix with General Data

**Action**: Blend math SFT data with general pretraining data (e.g., 50/50 or 70/30 mix).

- Prevents catastrophic forgetting of baseline capabilities
- Ensures the model maintains its existing arithmetic skills while gaining new ones
- Standard approach in instruction tuning literature

**Expected impact**: MEDIUM — addresses the regression floor

### Priority 4: Use Loss Masking

**Action**: Switch from `TextLmDatasetFormat` to `ChatLmDatasetFormat` with `mask_user_turns=True`.

- Currently training on ALL tokens including question text and headers (32% of tokens in Q/R/A docs)
- With masking, loss only computed on reasoning+answer (~61% of tokens)
- Concentrates training signal on the actual mathematical reasoning

**Expected impact**: MEDIUM — improves training efficiency

### Priority 5: Eval-Aligned Format

**Action**: Reformulate training data to match eval prompt format.

- Use `Problem:` / `Solution:` / `\boxed{}` structure instead of `## Question` / `## Reasoning` / `## Answer`
- Include "Final Answer: The final answer is X." pattern
- This could be done via a second LLM pass or template-based reformatting

**Expected impact**: LOW-MEDIUM — addresses format gap but not the content quality issues

---

## 9. Updated Analysis: lr5e-7_bs64 (New Best Config)

The completed HP sweep revealed lr5e-7_bs64 as the best config — the first to beat baseline on minerva math. This section compares it against both baseline and lr5e-7_bs16 to understand what larger batch size changes.

### 9.1 Regression/Improvement Counts

| Task | bs16 Regressions | bs16 Improvements | bs16 Net | bs64 Regressions | bs64 Improvements | bs64 Net |
|------|-----------------|-------------------|----------|-----------------|-------------------|----------|
| **Algebra** | 130 | 137 | +7 | 130 | **155** | **+25** |
| **PreAlgebra** | 121 | 76 | -45 | **108** | **86** | **-22** |
| **GSM8K** | 136 | 88 | -48 | 136 | 89 | **-47** |

**Key insight**: On algebra, bs64 has the *same* number of regressions (130) but 18 more improvements (+13%). On prealgebra, bs64 has 13 fewer regressions (-11%) and 10 more improvements. GSM8K is essentially unchanged — the batch size effect doesn't help there.

### 9.2 Failure Mode Shift

| Category | bs16 Total | bs64 Total | Change |
|----------|-----------|-----------|--------|
| wrong_computation | 349 (90.2%) | 345 (92.2%) | -4 |
| **repetition_loop** | **29 (7.5%)** | **19 (5.1%)** | **-10 (34% reduction)** |
| truncation | 9 (2.3%) | 10 (2.7%) | +1 |

The biggest qualitative improvement is **repetition loops drop by 34%**. In algebra specifically, repetition loops drop from 20→11 (45% reduction). This is a direct effect of larger batch sizes providing more stable gradient updates — the model is less likely to enter degenerate generation loops.

### 9.3 Per-Difficulty: Where bs64 Helps and Hurts

**Algebra** — bs64 dramatically recovers easy and hard problems:

| Level | Baseline | bs16 | bs16 Δ | bs64 | bs64 Δ | bs64 vs bs16 |
|-------|----------|------|--------|------|--------|--------------|
| Level 1 (easiest) | 76.3% | 71.1% | -5.2pp | **74.8%** | **-1.5pp** | +3.7pp better |
| Level 2 | 53.7% | 59.7% | +6.0pp | 59.7% | +6.0pp | same |
| Level 3 | 44.4% | 49.4% | +5.0pp | 47.9% | +3.4pp | -1.6pp worse |
| Level 4 | 29.7% | 25.8% | -3.9pp | **29.0%** | **-0.7pp** | +3.2pp better |
| Level 5 (hardest) | 10.1% | 10.1% | +0.0pp | **12.7%** | **+2.6pp** | +2.6pp better |

bs64 nearly eliminates the Level 1 regression (-1.5pp vs -5.2pp) and Level 4 regression (-0.7pp vs -3.9pp), and is the only config to improve Level 5. The tradeoff is a slightly smaller Level 3 gain.

**PreAlgebra** — bs64 recovers Levels 2-5 but introduces a new Level 1 regression:

| Level | Baseline | bs16 | bs16 Δ | bs64 | bs64 Δ | bs64 vs bs16 |
|-------|----------|------|--------|------|--------|--------------|
| Level 1 (easiest) | 68.6% | 64.0% | -4.7pp | 57.0% | **-11.6pp** | -6.9pp worse |
| Level 2 | 66.1% | 59.9% | -6.2pp | **63.3%** | **-2.8pp** | +3.4pp better |
| Level 3 | 53.1% | 42.4% | -10.7pp | **50.9%** | **-2.2pp** | +8.5pp better |
| Level 4 | 40.8% | 37.7% | -3.1pp | **38.7%** | **-2.1pp** | +1.0pp better |
| Level 5 (hardest) | 17.6% | 17.6% | +0.0pp | **18.7%** | **+1.0pp** | +1.0pp better |

**The PreAlgebra Level 1 regression (-11.6pp) is the most concerning finding.** bs64 is dramatically worse than bs16 on the easiest prealgebra problems, while being dramatically better on Levels 2-5. Examining the examples (see 9.5), some of these "regressions" appear to be answer extraction false negatives where bs64's slightly different formatting confuses exact_match — but many are genuine computational errors on trivial problems like division and square roots.

### 9.4 Generation Behavior Comparison

| Task | Model | Avg Length | Repetition % | \boxed{} % | Extraction Artifacts % |
|------|-------|-----------|-------------|------------|----------------------|
| Algebra | baseline | 399 | 3.1% | 77.6% | 0.0% |
| Algebra | bs16 | 349 (-13%) | 7.7% | 66.5% | 0.0% |
| Algebra | **bs64** | **361 (-10%)** | **5.6%** | 61.8% | 0.0% |
| PreAlg | baseline | 363 | 4.0% | 83.7% | 0.0% |
| PreAlg | bs16 | 304 (-16%) | 5.3% | 72.4% | 0.0% |
| PreAlg | **bs64** | **314 (-13%)** | **4.9%** | 68.8% | 0.0% |
| GSM8K | baseline | 346 | 0.3% | 0.3% | 0.0% |
| GSM8K | bs16 | 254 (-27%) | 0.9% | 0.0% | 0.0% |
| GSM8K | **bs64** | **257 (-26%)** | **0.6%** | 0.0% | 0.0% |

**Observations:**
- **bs64 produces slightly longer responses than bs16** (closer to baseline), suggesting it retains more of the baseline's reasoning verbosity
- **Repetition rate is lower for bs64** across all tasks (5.6% vs 7.7% algebra, 0.6% vs 0.9% GSM8K)
- **\boxed{} usage continues to drop** — bs64 uses \boxed{} in 61.8% of algebra responses vs 66.5% for bs16 vs 77.6% baseline. Both SFT configs shift the model away from the \boxed{} format
- **No extraction artifacts** in any config — the training format headers (## Question/etc.) do not leak into eval generations
- **GSM8K "the answer is" format** is preserved by both SFT configs (~98% vs 97% baseline), so the GSM8K gap is not a format issue

### 9.5 Three-Way Comparison: What bs64 Fixes and Breaks

**Problem churn is high** — bs64 doesn't simply fix all of bs16's errors. It fixes some and breaks others:

| Task | Fixed by bs64 | New bs64 regressions | Unique bs64 improvements |
|------|--------------|---------------------|-------------------------|
| | (base=Y, bs16=N, bs64=Y) | (base=Y, bs16=Y, bs64=N) | (base=N, bs16=N, bs64=Y) |
| **Algebra** | 44 | 44 | **55** |
| **PreAlgebra** | 39 | 26 | **32** |
| **GSM8K** | 34 | 34 | **23** |

The net improvement comes primarily from **unique improvements** — problems that *neither* baseline nor bs16 solved, but bs64 does. This suggests bs64 isn't just recovering lost baseline knowledge; it's genuinely learning new mathematical reasoning from the training data.

### 9.6 Example: Repetition Loop Fixed by bs64

**Problem** (Algebra Level 2, doc_id=22): *What is the value of $x$ in $(17^6-17^5)\div16=17^x$?*
**Gold**: 5

| Model | Response | Correct? |
|---|---|---|
| **Baseline** | Factors out $17^5$: $\frac{17^5 \cdot 16}{16} = 17^5$ → $x=5$ | Yes |
| **bs16** | "$17^5 \cdot 16 = 2^{16} \cdot 17^5 = 2^{16} \cdot (17^4)^2 = 2^{16} \cdot (17^4)^2 = ...$" *(enters repetition loop)* | No |
| **bs64** | Factors out $17^5$: $\frac{17^5 \cdot 16}{16} = 17^5$ → $x=5$ | **Yes** |

> bs16 enters a degenerate loop trying to decompose 16 as $2^4$. bs64 takes the clean, direct approach — identical to baseline. This is exactly the kind of regression that larger batch size eliminates by providing more stable gradients.

### 9.7 Example: New bs64 Regression on Trivial Problem

**Problem** (Algebra Level 1, doc_id=4): *If $2^8=4^x$, what is the value of $x$?*
**Gold**: 4

| Model | Response | Correct? |
|---|---|---|
| **Baseline** | Rewrites $4^x = (2^2)^x = 2^{2x}$, so $8=2x$, $x=4$ | Yes |
| **bs16** | Same approach → $x=\boxed{4}$ | Yes |
| **bs64** | "We can rewrite $2^8$ as $2^4 \cdot 2^4$, so $4^x = 2^4 \cdot 2^4 = 2^8$... $x = 8$" | **No** |

> bs64 takes an incorrect approach, confusing the exponent with the value. This is a genuine reasoning error on a Level 1 problem — the model "knows" more advanced techniques but occasionally fumbles basics.

### 9.8 Example: bs64 Uniquely Solves a Problem Both Others Miss

**Problem** (Algebra Level 3, doc_id=35): *If $9^{18n}=27^{24}$, find $n$.*
**Gold**: 2

| Model | Response | Correct? |
|---|---|---|
| **Baseline** | Takes log of both sides but gets cut off before finishing | No |
| **bs16** | Incorrectly converts: $27^{24} = 3^{120}$, then $3^{18n} = 3^{120}$, gets $n = 20/3$ | No |
| **bs64** | "$27^{24} = (3^3)^{24} = 3^{72}$... $n = \frac{72 \log 3}{18 \log 9} = \frac{4}{2} = \boxed{2}$" | **Yes** |

> bs16 makes a conversion error ($27^{24} \ne 3^{120}$); bs64 correctly computes $3^{72}$ and solves cleanly. This demonstrates that bs64 is learning genuine mathematical reasoning from the training data, not just memorizing.

### 9.9 Example: GSM8K — Where bs64 Still Fails

**Problem** (GSM8K, doc_id=15): *Claire makes a 3 egg omelet every morning. How many dozens of eggs will she eat in 4 weeks?*
**Gold**: 7

| Model | Response | Correct? |
|---|---|---|
| **Baseline** | 4×7=28 days, 28×3=84 eggs, 84/12=7 dozens | Yes |
| **bs16** | Same correct reasoning → 7 dozens | Yes |
| **bs64** | "3 eggs per omelet means 3 × 12 = **36 eggs per day**" *(confuses eggs with dozens)* → 84 dozens | **No** |

> bs64 makes a unit confusion error — multiplying eggs by 12 (dozen) instead of computing eggs per day directly. This type of word problem reasoning error is representative of the persistent GSM8K gap that batch size alone cannot fix.

### 9.10 Revised Root Cause Assessment

With the bs64 data, the root cause ranking shifts:

| # | Cause | Evidence | Fixable by HP? |
|---|-------|----------|----------------|
| 1 | **Hyperparameter sensitivity** | bs64 recovers 2.7pp on prealgebra, 1.5pp on algebra vs bs16 | **Yes** (already fixed) |
| 2 | **Gradient instability at small batch** | 34% fewer repetition loops at bs64, Level 1/4 recovery | **Yes** (already fixed) |
| 3 | **Training data domain mismatch** | 90% physicsforums+geogebra; GSM8K gap unchanged by HP | **No** — need better data |
| 4 | **Training data quality** | 27% minimal, 46% no LaTeX; wrong computations persist | **No** — need better filtering |
| 5 | **Easy-problem forgetting** | PreAlgebra Level 1 -11.6pp for bs64 despite overall improvement | **Partially** — mixing with general data may help |
| 6 | **\boxed{} format erosion** | Drops from 78%→62% with SFT; some exact_match false negatives | **No** — need format-aligned training |

**The GSM8K gap (~4pp) and PreAlgebra Level 1 regression (-11.6pp) are now the primary remaining challenges.** Neither responds to hyperparameter tuning. They likely require:
- Better training data (more word problems, more elementary-level content)
- Data mixing with general pretraining data to prevent simple-task forgetting
- Curriculum or difficulty-aware training to preserve easy-problem performance
