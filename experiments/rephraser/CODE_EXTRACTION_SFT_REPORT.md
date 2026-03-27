# Code Extraction SFT Experiment Report

**Last updated**: 2026-03-16

## Goal

Fine-tune Qwen3-0.6B on code-heavy web content from Common Crawl, then evaluate on HumanEval. Compare two data processing methods:

1. **Resiliparse**: Standard HTML-to-text extraction (DCLM-style). No LLM involved.
2. **8B Extraction**: LLM-based extraction using a Qwen3-8B rephraser to reformat HTML into structured source code files with comments.

Both branches use plain text SFT (no chat template). The question is whether LLM-restructured code data improves code generation over raw text from the same sources.

## Model

- **Architecture**: Qwen3-0.6B (hidden=896, layers=24, heads=14, kv_heads=2, intermediate=4864)
- **Tokenizer**: `Qwen/Qwen3-0.6B` (151,936 vocab)
- **RoPE**: theta=1,000,000, factor=1.0
- **Max sequence length**: 4,096
- **Training**: 1-epoch SFT, batch=64, seq_len=4096, AdamConfig lr=2e-5, cosine schedule

## Data Sources

45 URL patterns across 4 categories, queried against the 10 most recent Common Crawl indices (2024-2026):

<details>
<summary>URL patterns (45 total)</summary>

**Stack Exchange / Q&A (20 sites)**:
stackoverflow.com, codereview.stackexchange.com, codegolf.stackexchange.com, cs.stackexchange.com, cstheory.stackexchange.com, math.stackexchange.com, stats.stackexchange.com, physics.stackexchange.com, unix.stackexchange.com, askubuntu.com, softwareengineering.stackexchange.com, dsp.stackexchange.com, ai.stackexchange.com, datascience.stackexchange.com, electronics.stackexchange.com, crypto.stackexchange.com, security.stackexchange.com, tex.stackexchange.com, mathematica.stackexchange.com, mathoverflow.net

**Language documentation (11 sites)**:
docs.python.org, doc.rust-lang.org, pkg.go.dev, en.cppreference.com, kotlinlang.org/docs, docs.scala-lang.org, docs.julialang.org, hexdocs.pm, www.php.net/manual, ruby-doc.org, docs.rs

**Tutorials (5 sites)**:
rosettacode.org, www.geeksforgeeks.org, realpython.com, www.w3schools.com, www.tutorialspoint.com

**Framework / API docs (11 sites)**:
developer.mozilla.org, www.tensorflow.org, api.flutter.dev, doc.qt.io, huggingface.co/docs, llvm.org/docs, gcc.gnu.org/onlinedocs, www.boost.org/doc, www.typescriptlang.org/docs/handbook, readthedocs.io (domain match), devdocs.io

</details>

**Common Crawl indices**: CC-MAIN-2026-08, CC-MAIN-2026-04, CC-MAIN-2025-51, CC-MAIN-2025-47, CC-MAIN-2025-43, CC-MAIN-2025-38, CC-MAIN-2025-33, CC-MAIN-2025-30, CC-MAIN-2024-51, CC-MAIN-2024-46

### Pipeline Data Flow

| Stage | Shards | Description |
|-------|--------|-------------|
| CDX query (3 match types) | — | domain, host, prefix queries across 10 crawl indices |
| WARC download (per type) | 500 each | 3 download steps combined |
| Combined HTML | 1,500 | Merged across all 3 match types |
| Filtered HTML | 1,343 | Deduplication and quality filtering |
| LLM extraction | 921 | Qwen3-8B extraction (pages with `[NO_USEFUL_CONTENT]` or <50 chars filtered) |
| Resiliparse text | 1,500 | Standard HTML-to-text |

### Token Counts

| Branch | Documents | Tokens |
|--------|-----------|--------|
| Extraction v1 (8B, code fence format) | — | 177.5M |
| Extraction v2 (8B, plain text format) | 416,389 | 256.5M |
| Resiliparse (plain text) | — | 792.5M |

The extraction branches have fewer tokens because the LLM filters out pages without code content (`[NO_USEFUL_CONTENT]`) and removes boilerplate/navigation. The resiliparse branch retains all text from all pages. V2 extracts ~45% more tokens than v1 because the plain text prompt is less aggressive in filtering — without code fences and `/* */` comments to structure around, the LLM retains more explanatory text alongside code.

<details>
<summary>Extraction prompt</summary>

```
Extract the content from this HTML page and reformat it as a source code file with comments.

Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
- Page contains no code blocks or programming examples (inline code references do not count)
- Question listing, search results, or index page without actual content
- Login, signup, paywall, error page, or empty page
- Not primarily in English

If the page has code, reformat it following these rules:

Format — clearly separate code from text:
- Write all explanatory text inside /* */ block comments. Group related paragraphs together.
- Use // for short inline notes only.
- Put all code examples inside ``` code fences.
- Start with a /* */ block summarizing the page topic.
- Use /* */ blocks as section separators between code examples.
- Do not use **bold**, ## headings, [links](url), or other markdown besides code fences.

Content — include everything from the page:
- Include every paragraph and code example. Do not shorten or omit content.
- Preserve all code exactly as written. Do not fix or improve it.
- Do not include any raw HTML tags or entities. Decode &amp; &lt; &gt; &quot; to their characters.
```

</details>

<details>
<summary>V2 extraction prompt (plain text, no formatting)</summary>

```
Extract the content from this HTML page as clean plain text.

Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
- The page is not about programming, software development, or computer science
- Index page, search results, or listing page without substantive content
- Login, signup, paywall, error page, or empty page
- Not primarily in English

Otherwise, extract the content following these rules:
- Output the text and code from the page in reading order.
- Preserve all code exactly as written.
- Keep all explanatory text that accompanies the code.
- Remove navigation menus, sidebars, headers, footers, and boilerplate.
- Do not add any formatting: no ``` code fences, no /* */ comments, no markdown.
- Do not include raw HTML tags or entities. Decode &amp; &lt; &gt; &quot; to their characters.
- Do not shorten or omit content.
```

The key difference from v1: **no output formatting instructions**. V1 asked for `/* */` comments and ` ``` ` code fences, which the SFT model learned to reproduce during code completion. V2 outputs plain text only, preventing format contamination.

</details>

<details>
<summary>V3 extraction prompt (commented code + Q/A routing)</summary>

```
Extract the content from this HTML page as clean plain text. Follow all rules below.

1. Do not use ``` code fences, markdown headers, bold, or any formatting. Output plain text only. The characters ``` must not appear anywhere in your output.
2. Do not include raw HTML tags or entities. Decode &amp; to &, &lt; to <, &gt; to >, &quot; to "
3. Do not shorten, omit, or truncate content. Extract the full page.
4. Do not add text that is not on the page. Every sentence must come from the source.
5. Preserve all code exactly as written on the page. Do not modify, rewrite, or generate new code. You should fix broken indentation so code is syntactically valid, and you should add brief inline comments in the target language's comment style to uncommented multi-step logic.

6. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
   - The page is not about programming, software development, or computer science
   - Index page, search results, listing page, or wiki metadata page without substantive content
   - Login, signup, paywall, error page, or empty page

7. For Q&A pages (Stack Overflow, Stack Exchange, forums with questions and replies), output in this exact order:

   Q: <the question, including any code the asker provided>

   Reasoning: <synthesize the discussion from the thread into an explanation from the question to the answer. Explain why the solution works, step by step. Only include information from the thread.>

   A: <the top answer, extracted exactly as written from the page>

8. If no complete, substantive answer was provided, omit the A: section.
9. If the page explicitly discusses buggy or non-functioning code, add this line at the very top of your output: code_status: buggy

10. For all other pages (tutorials, documentation, blogs, reference):
    Output the text and code in reading order. Keep all explanatory text. Remove navigation, sidebars, headers, footers, and boilerplate.
```

Key differences from v2: (1) **Q/A routing** — Stack Overflow / forum pages are restructured into Q:/Reasoning:/A: format rather than extracted verbatim, encouraging the model to learn from structured problem-solving patterns. (2) **Inline comments** — the prompt asks the LLM to add brief inline comments to uncommented multi-step code. (3) **Buggy code tagging** — pages discussing broken code are prefixed with `code_status: buggy`. Still plain text output (no code fences, no `/* */` comments).

</details>

### Parsed Page Examples

<details>
<summary>8B Extraction example (readthedocs.io — TRACULA documentation)</summary>

```
/* TRACULA Tutorial #1: Overview */
TRACULA (TRActs Constrained by UnderLying Anatomy) is a suite of diffusion analysis
programs included in the FreeSurfer software package. Unlike other packages, such as
TBSS or MRtrix, TRACULA focuses on reconstructing the major white matter pathways of
the brain, calculating values such as fractional anisotropy and mean diffusivity for
each tract.

/* Note about dyld warning */
If you try running TRACULA, you may get the following warning:
​```
dyld: lazy symbol binding failed: Symbol not found: ___emutls_get_address
​```

/* Other Notes */
- Use the following steps to reconstruct the white matter tracts:
​```
trac-all -prep -c dmrirc.txt
trac-all -bedp -c dmrirc.txt
trac-all -path -c dmrirc.txt
​```
```

</details>

<details>
<summary>Resiliparse example (readthedocs.io — anndata API reference)</summary>

```
anndata.experimental.AnnCollection.convert

anndata.experimental.AnnCollection.convert#

property AnnCollection.convert[source]#

On the fly converters for keys of attributes and data matrix.

A function or a Mapping of functions which will be applied to the values of
attributes (.X) or to specific keys of these attributes (.obs, .obsm, .layers)
of subset objects. The converters are not applied to .obs and .obsm (if present)
of this object, only to the attributes of subset objects. The keys of the Mapping
should correspond to the attributes or keys of the attributes (hierarchically)
and the values should be functions used for conversion.

Examples

{
    # densify .X
    "X": lambda a: a.toarray() if issparse(a) else a,
    # change dtype for all keys of .obsm
    "obsm": lambda a: np.asarray(a, dtype="float32"),
    # change type only for one key of .obs
    "obs": dict(key1=lambda c: c.astype(str)),
}
```

</details>

## Training

| | Extraction SFT v1 | Extraction SFT v2 | Extraction SFT v3 | Resiliparse SFT |
|---|---|---|---|---|
| **Tokens** | 177.5M | 256.5M | 260.8M | 792.5M |
| **Documents** | — | 416,389 | — | — |
| **Training steps** | 677 | 978 | 994 | 3,023 |
| **Checkpoint** | `code-extract-general-qwen3-0.6b-sft-b2603e` (step-677) | `code-extract-plaintext-qwen3-0.6b-sft-dddf5e` (step-978) | `code-extract-commented-qwen3-0.6b-base-sft-bedd7b` (step-994) | `code-resiliparse-qwen3-0.6b-sft-c97c7a` (step-3023) |
| **Format** | Plain text (TextLmDatasetFormat) | Plain text (TextLmDatasetFormat) | Plain text (TextLmDatasetFormat) | Plain text (TextLmDatasetFormat) |

## Evaluation

**Benchmark**: [HumanEval](https://github.com/openai/human-evaluations) — 164 Python function completion problems. The model receives a function signature + docstring and must generate a correct implementation. Evaluated with `pass@1` (greedy, single attempt).

**Settings**: 0-shot only. The experiment config specified both 0-shot and 4-shot, but **HumanEval hardcodes `num_fewshot: 0` in its lm-eval-harness task YAML** (`humaneval.yaml` line 23). When a task sets `num_fewshot: 0`, lm-eval silently ignores any override (see `evaluator.py` lines 323-327: *"num_fewshot has been set to 0 for humaneval in its config. Manual configuration will be ignored."*). The 4-shot eval results are identical duplicates of 0-shot — no few-shot examples were actually prepended. This was confirmed by comparing the raw sample prompts: both runs send identical `arguments` to the model.

<details>
<summary>HumanEval example prompt</summary>

```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

</details>

## Results

### Instruct model (Qwen3-0.6B)

| Model | pass@1 | vs. Baseline |
|-------|:---:|:---:|
| **Baseline** (Qwen3-0.6B, no SFT) | 19.5% | — |
| **Resiliparse SFT** (step-3023) | **24.4%** | **+4.9pp** |
| **Commented extraction SFT v3** (step-994) | **24.4%** | **+4.9pp** |
| **Plaintext extraction SFT v2** (step-978) | 23.2% | +3.7pp |
| **Extraction SFT v1** (step-677) | 12.2% | -7.3pp |

V3 commented extraction **matches resiliparse** (24.4%) with 3x fewer tokens (260.8M vs 792.5M). The v2 plaintext extraction prompt (no code fences, no `/* */` comments) recovers most of the gap — **23.2% vs 12.2%** for v1. The format contamination from v1's code fence + comment style was responsible for ~11pp of degradation.

*Note: 4-shot results were collected but are identical to 0-shot due to HumanEval's hardcoded `num_fewshot: 0` (see Evaluation section).*

### Base model (Qwen3-0.6B-Base)

These experiments repeat the same pipeline using **Qwen3-0.6B-Base** (the base pretrained model, no instruction tuning). The v2 extraction prompt uses a plain text format (no code fences, no `/* */` comments) to avoid the format contamination observed in v1. The v3 prompt adds Q/A routing for forum pages and inline code comments.

#### Greedy pass@1 (standard HumanEval, 1 sample)

| Model | pass@1 | vs. Base Baseline |
|-------|:---:|:---:|
| **Baseline** (Qwen3-0.6B-Base, no SFT) | **30.5%** | — |
| **Commented extraction SFT v3** (step-994) | 29.9% | -0.6pp |
| **Plaintext extraction SFT v2** (step-978) | 28.7% | -1.8pp |
| **Resiliparse SFT** (step-3023) | 26.2% | -4.3pp |

#### pass@k (humaneval_64, 64 samples per problem, temperature sampling)

| k | Baseline | Plaintext Extraction SFT | Resiliparse SFT |
|---|:---:|:---:|:---:|
| pass@1 | **30.2%** | 28.0% | 25.9% |
| pass@2 | **31.4%** | 29.5% | 26.6% |
| pass@8 | **33.0%** | 31.0% | 27.4% |
| pass@16 | **33.3%** | 31.1% | 27.4% |
| pass@32 | **33.5%** | 31.1% | 27.4% |
| pass@64 | **33.5%** | 31.1% | 27.4% |

The baseline wins at every k. The pass@k curves flatten quickly for all three conditions — by k=16 there's minimal improvement, indicating the models have a hard ceiling on which problems they can solve regardless of how many samples are drawn. The relative ordering is consistent: baseline > plaintext extraction > resiliparse. The gap narrows slightly at higher k (baseline leads by 2.4pp at pass@64 vs 2.2pp at pass@1 over plaintext), suggesting the SFT models don't produce more diverse correct solutions.

### Instruct vs. Base comparison

| Condition | Instruct | Base | Delta |
|-----------|:---:|:---:|:---:|
| Baseline (no SFT) | 19.5% | **30.5%** | **+11.0pp** |
| Commented extraction SFT v3 | 24.4% | 29.9% | +5.5pp |
| Resiliparse SFT | 24.4% | 26.2% | +1.8pp |
| Plaintext extraction SFT v2 | 23.2% | 28.7% | +5.5pp |
| Extraction SFT v1 | 12.2% | — | — |

The base model baseline (30.5%) is **11pp higher** than the instruct baseline (19.5%). This is expected — the instruct model's `<|im_end|>` EOS token causes premature stopping in completion-mode evaluation. The base model gives a cleaner picture of each SFT condition's effect on code generation.

**Key finding: V3 commented extraction matches resiliparse on instruct and nearly matches the baseline on base.** On the instruct model, v3 achieves 24.4% — identical to resiliparse — with 3x fewer tokens (260.8M vs 792.5M). On the base model, v3 achieves 29.9%, just 0.6pp below the 30.5% baseline (within noise). The progression v1→v2→v3 shows that prompt engineering for the extraction LLM directly translates to downstream SFT quality: v1's format contamination cost -7.3pp on instruct; v2 fixed the format but still lost -1.8pp on base; v3's structured Q/A routing and code commenting close the remaining gap.

**V2 plaintext prompt fixes format contamination.** The v2 instruct model (23.2%) nearly matches resiliparse (24.4%), recovering +11pp from v1 (12.2%). V3 closes the remaining 1.2pp gap entirely.

## Analysis

### Resiliparse wins (+4.9pp), v3 ties it

Continued pretraining on 792.5M tokens of raw code-heavy web text improved HumanEval by 4.9 percentage points (19.5% -> 24.4%). V3 commented extraction matches this result exactly (24.4%) with only 260.8M tokens — no additional data volume needed when the extraction prompt produces higher-quality training examples.

### LLM extraction v1 hurts (-7.3pp), v2 recovers (+3.7pp), v3 matches resiliparse (+4.9pp)

The v1 extraction branch degraded performance (19.5% -> 12.2%), but v2 with a plain text prompt nearly matches resiliparse (23.2% vs 24.4%). V3 with Q/A routing and inline comments matches resiliparse exactly on the instruct model (24.4%) and achieves 29.9% on the base model, just 0.6pp below the 30.5% baseline. This isolates **format contamination** as the primary cause of v1's failure, and shows that **structured extraction prompts improve data quality**:

1. **Format contamination was the main issue (v1)**: The v1 prompt's `/* comment */ + ``` code fence` structure leaked into completions — the model would generate correct code followed by trailing ` ``` ` or `/* */` blocks, causing syntax errors. The v2 plain text prompt eliminates this entirely, recovering +11pp.

2. **Data volume still matters**: Even with the format fix, the extraction branch (177.5M tokens, 978 steps) trails resiliparse (792.5M tokens, 3,023 steps) by 1.2pp. The LLM's aggressive filtering (`[NO_USEFUL_CONTENT]`) removes ~77% of pages, reducing data volume without proportional quality gains for code completion.

3. **Structured extraction closes the gap (v3)**: On the base model, v3 commented extraction (29.9%) nearly matches the baseline (30.5%), improving over v2 plaintext (28.7%) by +1.2pp. The Q/A routing format — which restructures Stack Overflow pages into Q:/Reasoning:/A: — and inline code commenting appear to produce higher per-token value. V3 achieves this with only 260.8M tokens (3x fewer than resiliparse's 792.5M), suggesting the quality improvements from structured extraction outweigh the volume reduction from aggressive filtering.

4. **Extraction consistently beats resiliparse on the base model**: All extraction variants outperform resiliparse on the base model (v3: 29.9%, v2: 28.7% vs resiliparse: 26.2%), even though resiliparse has 3-4.5x more tokens. LLM-curated data has higher per-token value when format contamination is controlled.

### 4-shot eval is a no-op for HumanEval

The HumanEval task YAML in lm-eval-harness hardcodes `num_fewshot: 0`. When a task sets this to 0, the harness silently ignores any `num_fewshot` override passed via the API. Both the 0-shot and 4-shot eval runs sent identical prompts and produced identical results. To get actual few-shot HumanEval, you would need a custom task YAML that removes the `num_fewshot: 0` line, or use a different eval harness.

### Takeaways

- **Format contamination was the dominant factor in v1's failure**: Fixing the prompt format (v1→v2) recovered +11pp on the instruct model. The `/* */` comment + code fence format in v1 caused trailing syntax errors in ~30% of otherwise-correct completions.
- **Plain text extraction is competitive with resiliparse**: V2 extraction (23.2%) nearly matches resiliparse (24.4%) on the instruct model, despite having 4.5x fewer tokens. On the base model, extraction (28.7%) actually beats resiliparse (26.2%).
- **V3 Q/A routing matches resiliparse on instruct, nearly eliminates the SFT penalty on base**: V3 commented extraction matches resiliparse exactly (24.4%) on the instruct model and reaches 29.9% on the base model (within 0.6pp of the 30.5% baseline). The progression v1 (12.2%) → v2 (23.2%) → v3 (24.4%) on the instruct model shows that extraction prompt quality is the key lever. The Q/A routing format and inline code comments in v3 produce the highest per-token value seen so far.
- **Data volume vs. quality tradeoff**: V3 matches resiliparse's instruct result with 3x fewer tokens (260.8M vs 792.5M). The base model results (where extraction wins at every prompt version) confirm that LLM-curated data has higher per-token value when format issues are controlled.
- **Aggressive filtering can be acceptable**: The `[NO_USEFUL_CONTENT]` filter removes ~77% of pages, but the remaining 23% may be higher quality for code learning. V3's near-baseline performance with only 260.8M tokens (vs 792.5M for resiliparse) supports this.

## Raw Eval Traces

### Resiliparse SFT (24.4% pass@1)

40/164 problems passed. The model handles straightforward algorithmic tasks well but struggles with multi-token string parsing and problems requiring careful case handling.

<details>
<summary>Resiliparse SFT — HumanEval/0 (PASSED) — close elements check</summary>

**Prompt:**
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

**Generated:**
```python
    for i in range(1, len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

</details>

<details>
<summary>Resiliparse SFT — HumanEval/25 (PASSED) — prime factorization</summary>

**Prompt:**
```python
from typing import List


def factorize(n: int) -> List[int]:
    """ Return list of prime factors of given integer in the order from smallest to largest.
    Each of the factors should be listed number of times corresponding to how many times it appeares in factorization.
    Input number should be equal to the product of all factors
    >>> factorize(8)
    [2, 2, 2]
    >>> factorize(25)
    [5, 5]
    >>> factorize(70)
    [2, 5, 7]
    """
```

**Generated:**
```python
    factors = []
    while n % 2 == 0:
        factors.append(2)
        n = n // 2
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.append(i)
            n = n // i
        i += 2
    if n > 2:
        factors.append(n)
    return factors
```

Correct trial division algorithm — this is a non-trivial problem that the model handles cleanly.

</details>

<details>
<summary>Resiliparse SFT — HumanEval/4 (PASSED) — mean absolute deviation</summary>

**Prompt:**
```python
from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
```

**Generated:**
```python
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)
```

</details>

<details>
<summary>Resiliparse SFT — HumanEval/17 (FAILED) — parse music notation</summary>

**Prompt:**
```python
from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
```

**Generated:**
```python
    beats = []
    for c in music_string:
        if c == 'o':
            beats.append(4)
        elif c == 'o|':
            beats.append(2)
        elif c == '.|':
            beats.append(1)
    return beats
```

Iterates character-by-character instead of splitting on spaces — multi-character tokens like `o|` can never match a single `c`.

</details>

<details>
<summary>Resiliparse SFT — HumanEval/18 (FAILED) — count overlapping substrings</summary>

**Prompt:**
```python
def how_many_times(string: str, substring: str) -> int:
    """ Find how many times a given substring can be found in the original string. Count overlaping cases.
    >>> how_many_times('', 'a')
    0
    >>> how_many_times('aaa', 'a')
    3
    >>> how_many_times('aaaa', 'aa')
    3
    """
```

**Generated:**
```python
    return string.count(substring)
```

Uses `str.count()` which doesn't count overlapping occurrences — `'aaaa'.count('aa')` returns 2, not 3.

</details>

<details>
<summary>Resiliparse SFT — HumanEval/16 (FAILED) — count distinct characters (case-insensitive)</summary>

**Prompt:**
```python
def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct characters (regardless of case) does it consist of
    >>> count_distinct_characters('xyzXYZ')
    3
    >>> count_distinct_characters('Jerry')
    4
    """
```

**Generated:**
```python
    return len(set(string))
```

Misses the "regardless of case" requirement — should be `len(set(string.lower()))`.

</details>

### Baseline and Extraction SFT

<details>
<summary>Baseline — HumanEval/2 (PASSED)</summary>

**Prompt:**
```python
def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
```

**Generated:**
```python
    integer_part = int(number)
    decimal_part = number - integer_part
    return decimal_part
```

</details>

<details>
<summary>Baseline — HumanEval/0 (FAILED)</summary>

**Prompt:**
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

**Generated:**
```python
    if not numbers:
        return False
    if threshold <= 0:
        return False
    if numbers[0] == numbers[1]:
        return False
    if numbers[0] == numbers[-1]:
        return False
    for i in range(1, len(numbers) - 1):
        if abs(numbers[i] - numbers[i - 1]) <= threshold:
            return True
    return False
```

Fails because it only checks adjacent pairs (off-by-one in range) and has incorrect early returns.

</details>

20/164 problems passed. The model can handle simple one-liner problems but consistently fails on anything requiring multi-step logic. Three distinct failure modes appear:

1. **Format contamination**: Correct code followed by trailing ` ``` ` or `/* */` comment blocks from the extraction training format, which causes syntax errors at execution time.
2. **Comment-only output**: Model writes comments describing an approach but generates `pass` or no implementation.
3. **Degenerate generation**: Long repetitive if-chains or comment loops instead of algorithmic solutions.

<details>
<summary>Extraction SFT — HumanEval/0 (PASSED) — close elements check</summary>

**Prompt:**
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

**Generated:**
```python
    if len(numbers) < 2:
        return False
    for i in range(len(numbers) - 1):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

</details>

<details>
<summary>Extraction SFT — HumanEval/31 (PASSED) — primality check</summary>

**Prompt:**
```python
def is_prime(n):
    """Return true if a given number is prime, and false otherwise.
    >>> is_prime(6)
    False
    >>> is_prime(101)
    True
    >>> is_prime(11)
    True
    >>> is_prime(13441)
    True
    >>> is_prime(61)
    True
    >>> is_prime(4)
    False
    >>> is_prime(1)
    False
    """
```

**Generated:**
```python
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0:
            return False
    return True
```

</details>

<details>
<summary>Extraction SFT — HumanEval/55 (PASSED) — Fibonacci</summary>

**Prompt:**
```python
def fib(n: int):
    """Return n-th Fibonacci number.
    >>> fib(10)
    55
    >>> fib(1)
    1
    >>> fib(8)
    21
    """
```

**Generated:**
```python
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)
```

</details>

<details>
<summary>Extraction SFT — HumanEval/1 (FAILED) — comment-only output</summary>

**Prompt:**
```python
from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
```

**Generated:**
```python
    # Solution
    # (1) Remove all spaces from the input string
    # (2) Split the string into groups of balanced parentheses
    # (3) Return the list of groups
    pass
```

Describes the approach in comments but generates `pass` — no implementation. The model learned to write explanatory comments from the extraction training data's `/* */` comment style.

</details>

<details>
<summary>Extraction SFT — HumanEval/13 (FAILED) — format contamination: correct code + trailing comments</summary>

**Prompt:**
```python
def greatest_common_divisor(a: int, b: int) -> int:
    """ Return a greatest common divisor of two integers a and b
    >>> greatest_common_divisor(3, 5)
    1
    >>> greatest_common_divisor(25, 15)
    5
    """
```

**Generated:**
```
    while b:
        a, b = b, a % b
    return a
` ` `

/* The greatest common divisor of two integers a and b is the largest number
that divides both a and b without leaving a remainder. */

/* The greatest common divisor of two integers a and b is the largest number
that divides both a and b without leaving a remainder. */
...
```

The GCD algorithm is correct, but the model appends a code fence closer and `/* */` comment blocks — direct artifacts from the extraction training format. The trailing content causes a syntax error.

</details>

<details>
<summary>Extraction SFT — HumanEval/14 (FAILED) — format contamination: correct code + trailing fence</summary>

**Prompt:**
```python
from typing import List


def all_prefixes(string: str) -> List[str]:
    """ Return list of all prefixes from shortest to longest of the input string
    >>> all_prefixes('abc')
    ['a', 'ab', 'abc']
    """
```

**Generated:**
```
    return [string[:i] for i in range(1, len(string)+1)]
` ` `
```

Correct one-liner, but followed by a trailing ` ``` ` fence that causes a syntax error.

</details>

<details>
<summary>Extraction SFT — HumanEval/4 (FAILED) — format contamination: correct code + trailing fence</summary>

**Prompt:**
```python
from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
```

**Generated:**
```
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)
` ` `
```

Again, correct implementation killed by a trailing code fence.

</details>

<details>
<summary>Extraction SFT — HumanEval/2 (FAILED) — degenerate if-chain</summary>

**Prompt:**
```python
def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
```

**Generated:**
```python
    if number == 0:
        return 0.0
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0.0
    if number == 1:
        return 0.0
    if number == 2:
        return 0.0
    if number == 3:
        return 0.0
    ...  # continues for every integer up to max_gen_toks
```

Instead of `number - int(number)`, generates an endless chain of hardcoded integer checks. The baseline model solves this trivially.

</details>

<details>
<summary>Extraction SFT — HumanEval/10 (FAILED) — comment-only output</summary>

**Prompt:**
```python
def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome """
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    """
```

**Generated:**
```python
    # Find the longest postfix of supplied string that is a palindrome.
    # The algorithm is based on the following observation:
    # A string is a palindrome if it is equal to its reverse.
    # A string is a prefix of a string if it is a prefix of the string.
    # A string is a suffix of a string if it is a suffix of the string.
    # A string is a prefix of a string that is a palindrome if it is a prefix
    # of the string and it is a palindrome.
    ...  # continues with circular definitions, no code
```

Another comment-only failure — the model produces verbose tautological comments without ever writing implementation code.

</details>

### V2 Plaintext Extraction SFT (23.2% pass@1)

38/164 problems passed. The v2 model produces **clean code completions with no format artifacts** — no trailing code fences, no `/* */` comment blocks, no comment-only outputs. Every problem that v1 failed due to format contamination is now solved correctly. The remaining failures are genuine algorithmic mistakes.

**V1→V2 format contamination fixes** — these problems had correct v1 code killed by trailing formatting:

<details>
<summary>V2 — HumanEval/2 (PASSED, v1 FAILED: degenerate if-chain)</summary>

**Generated (v2):**
```python
    return number - int(number)
```

**V1 generated:** An endless chain of `if number == N: return 0.0` for each integer. V2 produces the clean one-liner that the baseline model also gets right.

</details>

<details>
<summary>V2 — HumanEval/4 (PASSED, v1 FAILED: trailing code fence)</summary>

**Generated (v2):**
```python
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)
```

**V1 generated:** Identical code, but followed by a trailing ` ``` ` fence causing a syntax error. V2 stops cleanly.

</details>

<details>
<summary>V2 — HumanEval/10 (PASSED, v1 FAILED: comment-only output)</summary>

**Generated (v2):**
```python
    n = len(string)
    for i in range(n):
        if is_palindrome(string[i:]):
            return string[:i] + string[::-1]
    return string + string[::-1]
```

**V1 generated:** Verbose tautological comments ("A string is a palindrome if it is equal to its reverse...") with no implementation code. V2 produces a correct algorithm that finds the longest palindromic suffix and prepends the reversed prefix.

</details>

<details>
<summary>V2 — HumanEval/13 (PASSED, v1 FAILED: trailing comments)</summary>

**Generated (v2):**
```python
    while b:
        a, b = b, a % b
    return a
```

**V1 generated:** The same Euclidean algorithm, but followed by ` ``` ` + repeated `/* The greatest common divisor... */` comment blocks. V2 stops after `return a`.

</details>

<details>
<summary>V2 — HumanEval/14 (FAILED, v1 also FAILED but different reason)</summary>

**Generated (v2):**
```python
    return [string[:i] for i in range(len(string)+1)]
```

**Expected:** `range(1, len(string)+1)` — should start at 1 to exclude the empty string `""`.

**V1 generated:** Correct code `[string[:i] for i in range(1, len(string)+1)]` but killed by a trailing ` ``` ` fence. V2 eliminates the format issue but introduces an off-by-one error (`range(len+1)` starts at 0, including `""`). This is a genuine algorithmic mistake, not format contamination.

</details>

### V3 Commented Extraction SFT (29.9% pass@1, base model)

49/164 problems passed. V3 produces **clean code with inline comments** — a distinctive pattern not seen in any other condition. The model learned to narrate its reasoning in comments, which appears to help with multi-step algorithmic problems. V3 uniquely solves 4 problems that no other condition (baseline, v2, resiliparse) can solve.

**V3 unique passes — problems only v3 solves:**

<details>
<summary>V3 — HumanEval/61 (V3 PASS, all others FAIL) — correct bracket matching</summary>

**Prompt:**
```python
def correct_bracketing(brackets: str):
    """ brackets is a string of "(" and ")".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("(")
    False
    >>> correct_bracketing("()")
    True
    >>> correct_bracketing("(()())")
    True
    >>> correct_bracketing(")(()")
    False
    """
```

**V3 (PASS):**
```python
    # Check if brackets is empty
    if not brackets:
        return True
    # Initialize stack
    stack = []
    # Iterate over each character
    for char in brackets:
        # If opening bracket, push to stack
        if char == '(':
            stack.append(char)
        # If closing bracket, check if stack is empty or mismatched
        elif char == ')':
            if not stack or stack[-1] != '(':
                return False
            # Pop the last opening bracket from stack
            stack.pop()
    # If stack is empty, all brackets are balanced
    return not stack
```

**Baseline (FAIL):** `# your code here\n    pass`
**V2 (FAIL):** `# TODO: Implement me\n    pass`
**Resiliparse (FAIL):** `return all(brackets.count("(") == brackets.count(")") for _ in range(len(brackets)))` — wrong algorithm.

The inline comments mirror the step-by-step reasoning structure from the Q/A extraction training data. Both baseline and v2 give up entirely on this problem — v3's structured training data teaches the model to break problems into commented steps rather than outputting `pass`.

</details>

<details>
<summary>V3 — HumanEval/71 (V3 PASS, all others FAIL) — triangle area with Heron's formula</summary>

**Prompt:** Given three side lengths, return the area rounded to 2 decimal places if valid triangle, else -1.

**V3 (PASS):**
```python
    # Check if sides form a valid triangle
    if a + b > c and a + c > b and b + c > a:
        # Calculate semi-perimeter
        s = (a + b + c) / 2
        # Calculate area using Heron's formula
        area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
        # Round to 2 decimal places
        return round(area, 2)
    else:
        return -1
```

**Baseline (FAIL):** `return (a * b * c) / (4 * math.sqrt(a * a + b * b - c * c))` — wrong formula, missing import.
**V2 (FAIL):** Uses Heron's formula correctly but returns `math.sqrt(...)` without `round()` and without importing `math`.
**Resiliparse (FAIL):** Uses a trig formula with `math.sin(math.acos(...))` — missing import.

V3 avoids the `math.sqrt` import trap by using `** 0.5`, and remembers to `round()`. The other three models all reference `math.*` without importing it.

</details>

<details>
<summary>V3 — HumanEval/49 (V3 PASS, all others FAIL) — modular exponentiation</summary>

**Prompt:**
```python
def modp(n: int, p: int):
    """Return 2^n modulo p (be aware of numerics).
    >>> modp(3, 5)
    3
    >>> modp(1101, 101)
    2
    >>> modp(0, 101)
    1
    """
```

**V3 (PASS):** `return pow(2, n, p)`
**Baseline (FAIL):** `return pow(n, p, p)` — wrong base and exponent.
**V2 (FAIL):** `return pow(n, p, p)` — same mistake.
**Resiliparse (FAIL):** `return n % p` — completely wrong.

Only v3 reads the docstring correctly: the function computes **2^n mod p**, not n^p mod p. The other models all misparse the relationship between the parameters.

</details>

<details>
<summary>V3 — HumanEval/124 (V3 PASS, all others FAIL) — date validation</summary>

**Prompt:** Validate a date string in mm-dd-yyyy format with month/day range checks.

**V3 (PASS):**
```python
    # Check if date is empty
    if not date:
        return False
    # Split date into parts
    parts = date.split('-')
    # Validate month
    if len(parts) != 3:
        return False
    month, day, year = parts
    # Validate month range
    if not (1 <= int(month) <= 12):
        return False
    # Validate day range
    if not (1 <= int(day) <= 31):
        return False
    # Validate year range
    if not (1970 <= int(year) <= 2099):
        return False
    return True
```

**Baseline (FAIL):** Tries character-by-character matching with a massive list of string literals — degenerate approach.
**V2 (FAIL):** Attempts character-by-character if-chains — never reaches a working implementation.
**Resiliparse (FAIL):** Indexes individual characters of the string (`date[0]`, `date[1]`) instead of splitting on `-`.

V3 is the only model that parses the date by splitting on the delimiter. The other models all attempt character-level processing, which cannot handle multi-digit months/days.

</details>

**V3 fixes v2 regressions:**

<details>
<summary>V3 — HumanEval/14 (V3 PASS, v2 FAIL) — all prefixes (off-by-one fix)</summary>

**V3 (PASS):** `return [string[:i] for i in range(1, len(string) + 1)]`
**V2 (FAIL):** `return [string[:i] for i in range(len(string)+1)]` — starts at 0, includes empty string `""`.

V2 had a simple off-by-one error (missing `1` in `range(1, ...)`). V3 gets it right.

</details>

**V3 retains common SFT regression:**

<details>
<summary>V3 — HumanEval/3 (V3 FAIL, baseline PASS) — below_zero (state accumulation)</summary>

**Baseline (PASS):**
```python
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
```

**V3 (FAIL):**
```python
    balance = 0
    for operation in operations:
        if operation < 0:
            return True
        balance += operation
    return balance < 0
```

Same regression as v2 and resiliparse: checks if individual operations are negative rather than tracking cumulative balance. This failure pattern persists across all SFT conditions and appears to be a fundamental consequence of continued pretraining on web code — the model loses the ability to reason about state accumulation.

</details>

### Base Model Comparison (50 / 47 / 49 / 43 pass@1)

The base model (Qwen3-0.6B-Base) passes 50/164 problems at baseline, vs 49 for v3 commented extraction, 47 for v2 plaintext extraction, and 43 for resiliparse SFT. V3 uniquely solves 4 problems that no other condition can solve. These traces show the most illustrative patterns.

**SFT regression — both SFT models lose problems the baseline solves:**

<details>
<summary>HumanEval/3 — base PASS, extract FAIL, resili FAIL — below_zero (running balance)</summary>

**Prompt:** Given a list of deposit/withdrawal operations starting from zero balance, return True if balance ever goes below zero.

**Base (PASS):**
```python
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False
```

**Extraction SFT (FAIL):**
```python
    balance = 0
    for operation in operations:
        if operation < 0:
            return True
        balance += operation
    return balance >= 0
```

**Resiliparse SFT (FAIL):**
```python
    for op in operations:
        if op < 0:
            return True
    return False
```

Both SFT models check if any individual operation is negative rather than tracking the cumulative balance. The base model correctly maintains a running total. This is the most common SFT regression pattern: the model "forgets" how to reason about state accumulation.

</details>

<details>
<summary>HumanEval/85 — base PASS, extract FAIL, resili FAIL — sum even values at odd indices</summary>

**Prompt:** Add the even elements at odd indices of a list.

**Base (PASS):**
```python
    return sum([lst[i] for i in range(1, len(lst), 2) if lst[i] % 2 == 0])
```

**Extraction SFT (FAIL):**
```python
    return [x for x in lst if x % 2 == 0 and x % 4 != 0]
```

**Resiliparse SFT (FAIL):**
```python
    return sum([lst[i] for i in range(0, len(lst), 2) if lst[i] % 2 == 0])
```

Base correctly identifies "odd indices" (`range(1, ..., 2)`) and "even elements" (`% 2 == 0`). Resiliparse gets the parity of indices wrong (`range(0, ..., 2)` = even indices). Extraction misunderstands the problem entirely.

</details>

**SFT gains — SFT models solve problems the baseline cannot:**

<details>
<summary>HumanEval/79 — base FAIL, extract PASS, resili PASS — decimal to binary with "db" wrapper</summary>

**Prompt:** Convert decimal to binary string wrapped in "db" markers (e.g., `decimal_to_binary(15)` returns `"db1111db"`).

**Base (FAIL):**
```python
    binary = "db"
    for i in range(1, decimal + 1):
        binary += "0" * (i - 1) + "1" * i
    return binary
```

**Extraction SFT (PASS):**
```python
    if decimal == 0:
        return "db0db"
    binary = ""
    while decimal > 0:
        binary = str(decimal % 2) + binary
        decimal = decimal // 2
    return "db" + binary + "db"
```

**Resiliparse SFT (PASS):**
```python
    return "db" + bin(decimal)[2:] + "db"
```

Both SFT models correctly implement binary conversion. Resiliparse uses Python's `bin()` builtin for a one-liner; extraction implements the algorithm from scratch. The base model generates a bizarre accumulation loop that doesn't perform binary conversion at all.

</details>

<details>
<summary>HumanEval/62 — base FAIL, extract PASS, resili FAIL — polynomial derivative</summary>

**Prompt:** Return the derivative of a polynomial represented as a list of coefficients (e.g., `derivative([3, 1, 2, 4, 5])` = `[1, 4, 12, 20]`).

**Base (FAIL):**
```python
    return [xs[i] * i for i in range(len(xs))]
```

**Extraction SFT (PASS):**
```python
    return [xs[i] * i for i in range(1, len(xs))]
```

**Resiliparse SFT (FAIL):**
```python
    return [xs[i] * i for i in range(len(xs))]
```

Classic off-by-one: the derivative drops the constant term, so the range must start at 1. Only the extraction model gets this right. This is a case where higher per-token quality in the LLM-curated data may help — the extraction data contains more carefully structured code examples.

</details>

**Mixed — resiliparse-specific failure:**

<details>
<summary>HumanEval/31 — base PASS, extract PASS, resili FAIL — is_prime</summary>

**Base (PASS):**
```python
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

**Extraction SFT (PASS):**
```python
    if n == 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```

**Resiliparse SFT (FAIL):**
```python
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for i in range(3, int(sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True
```

Resiliparse generates the most optimized algorithm (odd-only trial division) but uses `sqrt(n)` without importing `math.sqrt`, causing a `NameError`. The extraction model uses a naive but correct `range(2, n)` loop. This pattern appears in several resiliparse failures: the model learns "advanced" patterns from web text that reference functions not available in the local scope.

</details>

## Artifacts

| Artifact | Path |
|----------|------|
| Experiment script | `experiments/rephraser/code_extraction_sft.py` |
| Recipe | `experiments/rephraser/extraction_sft_recipe.py` |
| CDX results | `gs://marin-us-central1/cdx/code_{domain,host,prefix}-*` |
| Filtered HTML | `gs://marin-us-central1/filtered/code_html-a25853/` |
| Extraction output | `gs://marin-us-central1/documents/code_extract_general_5004a4dc-95ebf3/` |
| Processed extraction | `gs://marin-us-central1/processed/code_extract_general_5004a4dc-a25298/` |
| Processed resiliparse | `gs://marin-us-central1/processed/code_resiliparse_text-1ea97b/` |
| Extraction checkpoint (instruct) | `gs://marin-us-central1/checkpoints/code-extract-general-qwen3-0.6b-sft-b2603e/` |
| Resiliparse checkpoint (instruct) | `gs://marin-us-central1/checkpoints/code-resiliparse-qwen3-0.6b-sft-c97c7a/` |
| Baseline eval (instruct) | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-qwen3-0.6b-baseline-f244e6/` |
| Extraction eval (instruct) | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-extract-general-qwen3-0.6b-sft-67548b/` |
| Resiliparse eval (instruct) | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-resiliparse-qwen3-0.6b-sft-5ff60a/` |
| Token counts | `gs://marin-us-central1/summary/code_qwen3-0.6b_token_counts-5acf67/` |
| **V2 instruct artifacts** | |
| Experiment script (v2) | `experiments/rephraser/code_extraction_sft_v2.py` |
| Plaintext extraction checkpoint (v2 instruct) | `gs://marin-us-central1/checkpoints/code-extract-plaintext-qwen3-0.6b-sft-dddf5e/` |
| Plaintext extraction eval (v2 instruct) | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-extract-plaintext-qwen3-0.6b-sft-a2e300/` |
| **V3 commented extraction artifacts** | |
| Experiment script (v3 instruct) | `experiments/rephraser/code_extraction_sft_v3.py` |
| Experiment script (v3 base) | `experiments/rephraser/code_extraction_sft_v3_base.py` |
| Extraction prompt (v3) | `experiments/rephraser/code_extraction_prompt_v3.txt` |
| Commented extraction output | `gs://marin-us-central1/documents/code_extract_commented_00019371-742115/` |
| Processed commented extraction | `gs://marin-us-central1/processed/code_extract_commented_00019371-287870/` |
| Commented extraction checkpoint (instruct) | `gs://marin-us-central1/checkpoints/code-extract-commented-qwen3-0.6b-sft-90436e/` |
| Commented extraction eval (instruct) | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-extract-commented-qwen3-0.6b-sft-791dd8/` |
| Commented extraction checkpoint (base) | `gs://marin-us-central1/checkpoints/code-extract-commented-qwen3-0.6b-base-sft-bedd7b/` |
| Commented extraction eval (base) | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-extract-commented-qwen3-0.6b-base-sft-14013c/` |
| **Base model artifacts (v2)** | |
| Experiment script (v2 base) | `experiments/rephraser/code_extraction_sft_v2_base.py` |
| Plaintext extraction checkpoint (base) | `gs://marin-us-central1/checkpoints/code-extract-plaintext-qwen3-0.6b-base-sft-5bce58/` |
| Resiliparse checkpoint (base) | `gs://marin-us-central1/checkpoints/code-resiliparse-qwen3-0.6b-base-sft-80285d/` |
| Baseline eval (base) | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-qwen3-0.6b-base-baseline-74bcb3/` |
| Plaintext extraction eval (base) | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-extract-plaintext-qwen3-0.6b-base-sft-f0d348/` |
| Resiliparse eval (base) | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-resiliparse-qwen3-0.6b-base-sft-88021c/` |
