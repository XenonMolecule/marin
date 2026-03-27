# Math Data Deep Dive — V2 LLM Quality Assessment

## Executive Summary

After automated analysis of **4.7M documents** (V1) and manual LLM review of **770 sampled documents** across 25 domains (V2), we identify **five compounding root causes** explaining why math SFT consistently hurts benchmark performance across all 24 HP configs tested on Qwen3-0.6B-Base.

**The core finding:** Only ~15% of the extraction data and ~5-10% of the resiliparse data is directly relevant to the target benchmarks (minerva_math algebra/prealgebra, GSM8K). The remaining 85-95% is noise, off-topic content, or actively harmful signal that teaches the model wrong behaviors.

| Dataset | Total Docs | Useful % | Benchmark-Relevant % | Verdict |
|---------|-----------|----------|---------------------|---------|
| Extraction | 1,429,456 | ~25% | ~15% | Fixable with aggressive filtering |
| Resiliparse | 3,284,904 | ~5-10% | ~2-3% | **Do not use** |

---

## The Five Root Causes

### 1. Massive Off-Topic Contamination (~40-50% of extraction, ~55% of resiliparse)

The training data contains enormous amounts of non-math content:

| Source | Issue | Volume |
|--------|-------|--------|
| physicsforums.com | Physics, not math | 228K extraction + 410K resiliparse |
| forums.wolfram.com | Mathematica software support | 93K extraction + 103K resiliparse |
| brainmass.com | Business/finance/economics homework | 50K extraction + 133K resiliparse |
| sparknotes.com | Literature summaries | 1.8K extraction + 59K resiliparse |
| cliffsnotes.com | Literature/test prep | 0.2K extraction + 61K resiliparse |
| geogebra.org + subdomains | Interactive applet descriptions, login pages | 62K extraction + 25K resiliparse |
| mathoverflow.net | Graduate-level research math (too advanced) | 138K extraction + 308K resiliparse |

**Impact:** The model learns a diffuse distribution over physics, business, software, and philosophy instead of concentrating on algebra and arithmetic problem-solving.

**Example (accounts.geogebra.org):** ALL 20 extraction samples from this domain are literal login pages:
```
Sign in
GeoGebra Account
Email or username
Password
```

**Example (brainmass.com, classified as "math"):**
> Calculate the weighted average cost of capital (WACC) for a firm with 40% debt...

### 2. Data Quality Floor Problems (30% missing answers, 23% stubs)

The extraction pipeline's biggest structural flaw is training on incomplete content:

| Structure Type | Extraction Count | % | Impact |
|----------------|-----------------|---|--------|
| qa_complete | 590,623 | 41.3% | Useful (if correct) |
| **qa_missing_answer** | **428,628** | **30.0%** | **Trains model to not conclude** |
| tutorial | 191,708 | 13.4% | Mixed — no Q/A structure |
| minimal (<200 chars) | 93,070 | 6.5% | No learning signal |
| question_only | 91,050 | 6.4% | Teaches question-posing, not solving |
| exercise_list | 25,503 | 1.8% | Lists without solutions |
| garbled | 2,694 | 0.2% | Pure noise |

**The 30% `qa_missing_answer` problem is critical.** These documents have questions and partial reasoning but NO final answer. The model learns to produce open-ended rambling without reaching a conclusion — the exact failure mode we observe in the sweep regressions.

**Example (mathhelpforum.com, qa_missing_answer):**
> ## Question
> How do I find the Pareto Distribution for...
>
> ## Reasoning
> Hi, would be thankful if anyone can solve this.
>
> *(82 chars total, no answer)*

### 3. LLM Extraction Artifacts (5.1% of samples, ~70K estimated docs)

The extraction pipeline leaks the extraction LLM's chain-of-thought reasoning into the training data:

- **`<think>` tags**: Full reasoning traces about how to extract content from HTML ("We need to extract main content text from this HTML...")
- **Meta-commentary**: "Let's produce that.", "Thus we output:", "The HTML is a login page for GeoGebra..."
- **Extraction instructions**: `[[ ## extraction_spec ## ]]`
- **Raw HTML fragments**: `</div></div></body>` repeated hundreds of times

**Example (mathoverflow.net/questions/18848):** 18,068 characters of `<think>` tag content where the extraction LLM reasons about how to parse the page. This entire document gets tokenized and trained on.

**Impact:** The model learns to generate extraction boilerplate instead of math solutions. Even at 5%, this contaminates the learned distribution significantly.

### 4. Format Mismatch with Evaluation (SEVERE)

Training and eval use incompatible formats:

| Aspect | Training Data | minerva_math Eval | GSM8K Eval |
|--------|--------------|-------------------|------------|
| Problem marker | `## Question` | `Problem:` | Conversational |
| Solution marker | `## Reasoning` | `Solution:` | Step-by-step |
| Answer format | `## Answer` section | `\boxed{answer}` | `#### number` |
| Tone | Forum conversational | Academic | Textbook |
| Loss masking | **None** (all tokens trained equally) | N/A | N/A |

The model learns to:
1. Generate markdown headers (`## Question`, `## Reasoning`) before each section
2. Write in forum conversational tone ("Hi everyone", "I'm stuck on this")
3. Produce the `## Answer` label instead of extracting a boxed answer

Without loss masking, ~60% of training tokens are spent on headers, questions, and structural markers rather than mathematical reasoning.

### 5. Mathematical Errors in Source Data

Forum data inherently contains wrong answers. Confirmed errors found in 350-sample review:

| Error | Source | Correct Answer |
|-------|--------|---------------|
| `-∛((7y)³) = 1/(7y)` | brainly.com/question/225971 | `-7y` |
| `100,203 = 100,000 - 200 - 3` | brainly.com/question/107812 | `100,000 + 200 + 3` |
| `-3k·k³² = -1.853e+15kk` | brainly.com/question/248256 | `-3k³³` |
| Quotient rule with wrong denominator | mathhelpforum.com/13369 | Fixed |
| Brownian motion "proof" = 1 sentence | physicsforums.com/279442 | Full proof needed |

Extrapolating from the sample error rate across 97K brainly docs alone suggests **thousands of documents with mathematical errors**. Training on wrong math is worse than training on no math.

---

## Resiliparse: Why It's Strictly Worse

The resiliparse data has ALL five problems above PLUS three additional ones:

### 6. Math Notation Destroyed

LaTeX is preserved in only ~15% of the resiliparse corpus. In 85%+ of documents, mathematical expressions are either:
- Stripped entirely (LaTeX images become empty space)
- Rendered as crude plaintext (`x^2 + 3x + 1` instead of `$x^2 + 3x + 1$`)
- Converted to garbage (`C^2`, `(1-p)^x`)

The model literally learns to do math without notation — the opposite of what benchmarks test.

### 7. Forum Boilerplate Dominates (40-70% of tokens)

Every resiliparse document includes massive navigation chrome:

| Domain | Boilerplate Content | Est. % of Tokens |
|--------|-------------------|-----------------|
| mathhelpforum.com | LinkBack URL, Thread Tools, user profiles, Similar Discussions | 40-60% |
| physicsforums.com | Chegg Tutors ads, share buttons, user badges, Loading... placeholders | 35-50% |
| mathforum.org | Full sidebar taxonomy (repeated verbatim, ~500 chars per page) | 50-70% |
| khanacademy.org | "If you're seeing this message..." JavaScript error (76K docs!) | 90-100% |
| brainly.com | "Certified Answer" badges, "Free help with homework" promos | 60-80% |

**Example (khanacademy.org, 90.2% of their 84K docs):**
```
If you're seeing this message, it means we're having trouble loading
external resources for Khan Academy.
```
This 221-character JavaScript-disabled error message appears in ~76K documents. The model is trained to output web error messages.

### 8. Automated Quality Metrics Are Unreliable

Two resiliparse domains flagged as "high quality" are almost entirely **tutor advertisement pages**:

| Domain | Automated "High Quality" % | Actual Content |
|--------|---------------------------|----------------|
| purplemath.com | 66.9% | Tutor directory listings ("Karsten C. — I've put together three...") |
| algebrahelp.com | 83.7% | Tutor profile pages (Chester PA, Lynn MA statistics tutors) |

**~110K documents** are mislabeled as high quality. These pages match quality heuristics (long text, math keywords in URLs, domain name contains "math") but contain zero mathematical content.

---

## Per-Domain Quality Summary

### Extraction Data — Domain Ratings

| Domain | Docs | Rating | Useful % | Benchmark-Relevant | Keep? |
|--------|-----:|:------:|:--------:|:------------------:|:-----:|
| mathhelpforum.com | 231K | 3/5 | 35-45% | ~25% | Yes (filter) |
| physicsforums.com | 228K | 2.5/5 | 15-20% | ~5% | Mostly remove |
| jiskha.com | 186K | 3/5 | 35-40% | ~30% (GSM8K) | Yes (filter) |
| mathoverflow.net | 138K | 2/5 | 5-10% | ~2% | Remove (too advanced) |
| mathforum.org | 107K | 2/5 | 15-20% | ~8% | Keep Dr. Math only |
| brainly.com | 97K | 3.5/5 (high) | 40-50% | ~35% | Yes (high tier only) |
| forums.wolfram.com | 93K | 1.5/5 | 10-15% | ~3% | Remove |
| mathisfunforum.com | 78K | 3/5 | 30-35% | ~15% | Yes (filter) |
| brilliant.org | 65K | 2.5/5 | 20-25% | ~10% | Filter (remove stubs) |
| brainmass.com | 50K | 2/5 | 15-20% | ~8% | Remove (business) |
| geogebra.org + subs | 62K | 0.5/5 | 2-5% | ~1% | **Remove entirely** |

### Resiliparse Data — Domain Ratings

| Domain | Docs | Rating | Useful % | Keep? |
|--------|-----:|:------:|:--------:|:-----:|
| math.libretexts.org | 13K | 3.5/5 | 50-60% | **Only keeper** |
| All others | 3.27M | 0.5-2.5/5 | 2-15% | Remove |

**Resiliparse verdict: Do not use.** The extraction pipeline is strictly better for math SFT.

---

## Recommendations (Prioritized)

### Tier 1: Immediate Filters (no reprocessing needed)

1. **Remove entire domains**: geogebra.org (all subdomains), accounts.geogebra.org, forums.wolfram.com, brainmass.com, mathoverflow.net, sparknotes.com, cliffsnotes.com
2. **Remove `qa_missing_answer` documents** (30% of extraction data) — these teach incomplete reasoning
3. **Remove documents < 300 chars** — stubs with no learning signal
4. **Remove documents containing `<think>` or `</think>` tags** — LLM extraction artifacts
5. **Remove documents with raw HTML** (`</div>`, `</body>`, `&nbsp;`)
6. **Remove `minimal` and `garbled` structure types**
7. **Keep only `qa_complete` structure type** for initial experiment

**Estimated data after Tier 1 filters: ~350K docs (from 1.43M)**

### Tier 2: Quality Improvement (requires some reprocessing)

8. **Domain-aware quality filtering**: Keep only "high" tier from brainly.com, physicsforums.com. Keep all tiers from mathhelpforum.com, jiskha.com (after Tier 1 filters)
9. **Strip extraction format headers**: Remove `# Title`, `## Question`, `## Reasoning`, `## Answer` headers. Present content in a format closer to eval: `Problem: ... Solution: ... The answer is X`
10. **Answer verification**: For simple algebra/arithmetic, verify final answers computationally. Flag and remove documents with wrong math.
11. **Topic filtering**: Use keyword/classifier to remove physics, chemistry, biology content from jiskha.com and physicsforums.com
12. **Language filter**: Remove non-English content

**Estimated data after Tier 2: ~150K high-quality, benchmark-relevant docs**

### Tier 3: Architecture Changes

13. **Loss masking**: Only compute loss on reasoning + answer tokens, not question/header tokens. This alone could 2-3x the effective training signal.
14. **Format augmentation**: During training, randomly present each document in one of several formats (eval-style `Problem/Solution/\boxed{}`, chat-style, plain-text). This prevents format contamination.
15. **Curriculum ordering**: Train on easier content (GSM8K-level) first, then harder (algebra), not random shuffling.
16. **Synthetic augmentation**: Use a strong model (Qwen3-72B or Claude) to generate correct solutions for problems where the forum answer is missing or wrong.

### The Curated Subset Experiment

Based on this analysis, the highest-impact experiment would be training on a **curated ~150K subset**:

| Source | Selection Criteria | Est. Docs |
|--------|-------------------|-----------|
| brainly.com | high-tier, qa_complete only | ~40K |
| jiskha.com | high-tier, qa_complete, math-topic only | ~30K |
| mathhelpforum.com | high-tier, qa_complete only | ~50K |
| mathforum.org (Dr. Math) | Dr. Math archive pages only | ~13K |
| mathisfunforum.com | high-tier only | ~15K |
| **Total** | | **~148K** |

**Hypothesis:** This 148K-doc curated subset will outperform the full 1.43M-doc dataset because it eliminates the 85% noise that dilutes and contradicts the math signal.

---

## Appendix: Data Volume Summary

### V1 Automated Analysis

| Metric | Extraction | Resiliparse |
|--------|-----------|-------------|
| Total documents | 1,429,456 | 3,284,904 |
| Total characters | 2.47B | N/A |
| Estimated tokens | ~618M | N/A |
| Domains | 125 | 233 |
| High quality % | 36.6% | 8.8% |
| Low quality % | 9.2% | 59.7% |
| Non-math % | N/A (structured) | 44.0% |

### Top 10 Extraction Domains by Token Volume

| Rank | Domain | Docs | Chars | % of Total |
|-----:|--------|-----:|------:|-----------:|
| 1 | physicsforums.com | 227,902 | 490M | 19.8% |
| 2 | mathoverflow.net | 138,127 | 431M | 17.4% |
| 3 | mathhelpforum.com | 231,244 | 393M | 15.9% |
| 4 | mathforum.org | 107,024 | 226M | 9.1% |
| 5 | jiskha.com | 185,907 | 224M | 9.1% |
| 6 | mathisfunforum.com | 78,355 | 186M | 7.5% |
| 7 | forums.wolfram.com | 93,030 | 151M | 6.1% |
| 8 | brainmass.com | 49,704 | 82M | 3.3% |
| 9 | brainly.com | 97,461 | 73M | 2.9% |
| 10 | brilliant.org | 65,269 | 40M | 1.6% |

Note: The top 2 domains by token volume (physicsforums, mathoverflow) together represent **37.2% of all training tokens** but have near-zero benchmark relevance. This alone explains much of the regression.

---

*V2 report generated by LLM quality assessment of 770 sampled documents. V1 automated analysis processed 4,714,360 total documents.*
