# Code Extraction SFT: Results

**Experiment**: Fine-tune Qwen3-0.6B on code-heavy web content (45 URL patterns across Stack Exchange, language docs, tutorials, and framework docs; 10 CC crawls) and evaluate on HumanEval pass@1. Compare standard HTML extraction (resiliparse) vs LLM-based extraction (Qwen3-8B reformats HTML into structured code files with `/* */` comments and code fences).

| Model | HumanEval pass@1 | vs. Baseline | Training tokens |
|-------|:---:|:---:|:---:|
| **Baseline** (Qwen3-0.6B, no SFT) | 19.5% | — | — |
| **Resiliparse SFT** | **24.4%** | **+4.9pp** | 792.5M |
| **Extraction SFT** | 12.2% | -7.3pp | 177.5M |

**Resiliparse wins.** Plain text extraction with no LLM involvement improves HumanEval by +4.9pp. LLM extraction *hurts* by -7.3pp.

## Why extraction hurts

Two problems, both visible in the eval traces:

1. **Format contamination** — The extraction prompt trains the model on `/* */` comment blocks and ` ``` ` code fences. At eval time, the model generates correct code but appends these artifacts, causing syntax errors. Several problems (HumanEval/4, /13, /14) have *correct implementations* that fail solely due to trailing ` ``` ` markers.

2. **4.5x less data** — The extraction prompt aggressively filters pages without code blocks (`[NO_USEFUL_CONTENT]`), leaving only 177.5M tokens vs 792.5M for resiliparse. For code, volume of exposure matters more than reformatting.

<details>
<summary>Example: format contamination (HumanEval/13 — GCD)</summary>

The extraction-trained model writes a correct GCD algorithm, then appends code fences and `/* */` comments from the training format:

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
```

/* The greatest common divisor of two integers a and b is the largest number
that divides both a and b without leaving a remainder. */

/* The greatest common divisor of two integers a and b is the largest number
that divides both a and b without leaving a remainder. */
...

The algorithm is correct. The trailing `/* */` blocks cause a syntax error.

</details>

<details>
<summary>Example: comment-only output (HumanEval/10 — make palindrome)</summary>

**Prompt:**
```python
def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
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
    ...  # continues with circular definitions, no code
```

The model writes verbose comments but never produces an implementation.

</details>

## Note on eval

The experiment config included both 0-shot and 4-shot HumanEval, but HumanEval hardcodes `num_fewshot: 0` in its lm-eval-harness task YAML, silently ignoring any override. Both runs produced identical results. The 4-shot config has been removed.

---

<details>
<summary>Full report (data sources, pipeline, training details, all eval traces)</summary>

# Code Extraction SFT Experiment Report

**Last updated**: 2026-03-12

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

**Stack Exchange / Q&A (20 sites)**:
stackoverflow.com, codereview.stackexchange.com, codegolf.stackexchange.com, cs.stackexchange.com, cstheory.stackexchange.com, math.stackexchange.com, stats.stackexchange.com, physics.stackexchange.com, unix.stackexchange.com, askubuntu.com, softwareengineering.stackexchange.com, dsp.stackexchange.com, ai.stackexchange.com, datascience.stackexchange.com, electronics.stackexchange.com, crypto.stackexchange.com, security.stackexchange.com, tex.stackexchange.com, mathematica.stackexchange.com, mathoverflow.net

**Language documentation (11 sites)**:
docs.python.org, doc.rust-lang.org, pkg.go.dev, en.cppreference.com, kotlinlang.org/docs, docs.scala-lang.org, docs.julialang.org, hexdocs.pm, www.php.net/manual, ruby-doc.org, docs.rs

**Tutorials (5 sites)**:
rosettacode.org, www.geeksforgeeks.org, realpython.com, www.w3schools.com, www.tutorialspoint.com

**Framework / API docs (11 sites)**:
developer.mozilla.org, www.tensorflow.org, api.flutter.dev, doc.qt.io, huggingface.co/docs, llvm.org/docs, gcc.gnu.org/onlinedocs, www.boost.org/doc, www.typescriptlang.org/docs/handbook, readthedocs.io (domain match), devdocs.io

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

| Branch | Tokens |
|--------|--------|
| Extraction (8B-extracted) | 177.5M |
| Resiliparse (plain text) | 792.5M |

The extraction branch has ~4.5x fewer tokens because the LLM filters out pages without code content (`[NO_USEFUL_CONTENT]`) and removes boilerplate/navigation. The resiliparse branch retains all text from all pages.

### Extraction prompt

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
- Put all code examples inside code fences.
- Start with a /* */ block summarizing the page topic.
- Use /* */ blocks as section separators between code examples.
- Do not use **bold**, ## headings, [links](url), or other markdown besides code fences.

Content — include everything from the page:
- Include every paragraph and code example. Do not shorten or omit content.
- Preserve all code exactly as written. Do not fix or improve it.
- Do not include any raw HTML tags or entities. Decode &amp; &lt; &gt; &quot; to their characters.
```

## Training

| | Extraction SFT | Resiliparse SFT |
|---|---|---|
| **Tokens** | 177.5M | 792.5M |
| **Training steps** | 677 | 3,023 |
| **Checkpoint** | `code-extract-general-qwen3-0.6b-sft-b2603e` (step-677) | `code-resiliparse-qwen3-0.6b-sft-c97c7a` (step-3023) |
| **Format** | Plain text (TextLmDatasetFormat) | Plain text (TextLmDatasetFormat) |

## Evaluation

**Benchmark**: HumanEval — 164 Python function completion problems. The model receives a function signature + docstring and must generate a correct implementation. Evaluated with `pass@1` (greedy, single attempt).

**Settings**: 0-shot only. HumanEval hardcodes `num_fewshot: 0` in its lm-eval-harness task YAML. When a task sets `num_fewshot: 0`, lm-eval silently ignores any override. The 4-shot eval results are identical duplicates of 0-shot.

## Results

| Model | pass@1 | vs. Baseline |
|-------|:---:|:---:|
| **Baseline** (Qwen3-0.6B, no SFT) | 19.5% | — |
| **Resiliparse SFT** (step-3023) | **24.4%** | **+4.9pp** |
| **Extraction SFT** (step-677) | 12.2% | -7.3pp |

## Analysis

### Resiliparse wins (+4.9pp)

Continued pretraining on 792.5M tokens of raw code-heavy web text improved HumanEval by 4.9 percentage points (19.5% -> 24.4%). This is a meaningful gain from a simple pipeline: standard HTML extraction, no LLM involvement, just exposure to more code.

### LLM extraction hurts (-7.3pp)

The 8B extraction branch significantly degraded performance (19.5% -> 12.2%). Several factors likely contribute:

1. **Much less data**: The extraction branch has only 177.5M tokens vs 792.5M for resiliparse (4.5x less). The LLM filters aggressively — pages without code blocks are mapped to `[NO_USEFUL_CONTENT]` and discarded. This may filter too much useful context.

2. **Format mismatch**: The extraction prompt reformats code into a `/* comment */ + code fence` structure. HumanEval expects raw Python function completions. Training on commented/fenced code may teach the model a completion style that doesn't match the eval format.

3. **Fewer training steps**: Only 677 steps vs 3,023 for resiliparse. With a single epoch, the smaller dataset means fewer gradient updates.

### Takeaways

- **Raw text wins for code**: Unlike the math domain where structured extraction might help with reasoning chains, code generation benefits more from volume of raw code exposure.
- **Aggressive filtering can backfire**: The extraction prompt's `[NO_USEFUL_CONTENT]` filter removed ~77% of the data. For a domain where the model needs broad code exposure, this is too aggressive.
- **Format matters**: Training data format should match evaluation format. Reformatting code into `/* */` comment blocks doesn't align with HumanEval's raw function completion format.

## Raw Eval Traces

### Resiliparse SFT (24.4% pass@1)

40/164 problems passed. The model handles straightforward algorithmic tasks well but struggles with multi-token string parsing and problems requiring careful case handling.

**Resiliparse SFT — HumanEval/0 (PASSED) — close elements check**

Prompt:
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

Generated:
```python
    for i in range(1, len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**Resiliparse SFT — HumanEval/25 (PASSED) — prime factorization**

Prompt:
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

Generated:
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

**Resiliparse SFT — HumanEval/4 (PASSED) — mean absolute deviation**

Generated: `sum(abs(x - mean) for x in numbers) / len(numbers)` — correct one-liner.

**Resiliparse SFT — HumanEval/17 (FAILED) — parse music notation**

Generated iterates character-by-character instead of splitting on spaces — multi-character tokens like `o|` can never match a single `c`.

**Resiliparse SFT — HumanEval/18 (FAILED) — count overlapping substrings**

Generated `string.count(substring)` which doesn't count overlapping occurrences.

**Resiliparse SFT — HumanEval/16 (FAILED) — count distinct characters (case-insensitive)**

Generated `len(set(string))` — misses the "regardless of case" requirement.

### Extraction SFT (12.2% pass@1)

20/164 problems passed. Three distinct failure modes:

1. **Format contamination**: Correct code followed by trailing code fences or `/* */` comment blocks from the extraction training format, causing syntax errors at execution time.
2. **Comment-only output**: Model writes comments describing an approach but generates `pass` or no implementation.
3. **Degenerate generation**: Long repetitive if-chains or comment loops instead of algorithmic solutions.

**Extraction SFT — HumanEval/0 (PASSED) — close elements check**

Generated correct O(n^2) pairwise comparison.

**Extraction SFT — HumanEval/31 (PASSED) — primality check**

Generated correct trial division with sqrt optimization.

**Extraction SFT — HumanEval/55 (PASSED) — Fibonacci**

Generated correct recursive implementation.

**Extraction SFT — HumanEval/13 (FAILED) — format contamination**

Correct GCD algorithm but appends code fences and `/* */` comment blocks — direct artifacts from the extraction training format.

**Extraction SFT — HumanEval/14, /4 (FAILED) — format contamination**

Correct one-liners killed by trailing code fences.

**Extraction SFT — HumanEval/1 (FAILED) — comment-only output**

Outputs `# (1) Remove all spaces ... # (2) Split ... # (3) Return ... pass`

**Extraction SFT — HumanEval/10 (FAILED) — comment-only output**

Produces verbose tautological comments without ever writing implementation code.

**Extraction SFT — HumanEval/2 (FAILED) — degenerate if-chain**

Instead of `number - int(number)`, generates an endless chain of hardcoded `if number == N: return 0.0` checks.

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
| Extraction checkpoint | `gs://marin-us-central1/checkpoints/code-extract-general-qwen3-0.6b-sft-b2603e/` |
| Resiliparse checkpoint | `gs://marin-us-central1/checkpoints/code-resiliparse-qwen3-0.6b-sft-c97c7a/` |
| Baseline eval | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-qwen3-0.6b-baseline-f244e6/` |
| Extraction eval | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-extract-general-qwen3-0.6b-sft-67548b/` |
| Resiliparse eval | `gs://marin-us-central1/evaluation/lm_evaluation_harness/code-resiliparse-qwen3-0.6b-sft-5ff60a/` |
| Token counts | `gs://marin-us-central1/summary/code_qwen3-0.6b_token_counts-5acf67/` |

</details>
