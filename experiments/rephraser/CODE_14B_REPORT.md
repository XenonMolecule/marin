# Code Extraction SFT: 14B Hyperparameter Sweep Results

## Results

**Data:** 260M tokens from 45 URL patterns (Stack Overflow, Stack Exchange sites, language docs, tutorials, framework docs) across 10 Common Crawl indices, extracted with Qwen3-8B rephraser using V3 commented extraction prompt. Resiliparse comparison uses 793M tokens of plain text from the same HTML.

| Model | Extraction | MBPP 3-shot | HumanEval 0-shot | MBPP Δ | HumanEval Δ |
|---|---|---|---|---|---|
| **Qwen3-14B-Base** | none (baseline) | **74.0%** | 54.3% | — | — |
| Qwen3-14B-Base | V3 extraction | 73.8% | **77.4%** | -0.2pp | **+23.1pp** |
| Qwen3-14B-Base | Resiliparse | 73.2% | 59.1% | -0.8pp | +4.8pp |
| **Qwen3-0.6B-Base** | none (baseline) | **39.8%** | 30.5% | — | — |
| Qwen3-0.6B-Base | V3 extraction | 39.4% | **33.5%** | -0.4pp | **+3.0pp** |
| Qwen3-0.6B-Base | Resiliparse | 38.2% | 31.1% | -1.6pp | +0.6pp |

V3 extraction SFT is MBPP-neutral while dramatically improving HumanEval. The effect scales with model size (+23pp for 14B vs +3pp for 0.6B). V3 consistently outperforms resiliparse at both scales. All hyperparameter decisions (LR, BS, WD, warmup) were made using MBPP 3-shot as the primary metric; HumanEval was evaluated but not used for selection.

## What the SFT Actually Teaches

Analysis of all 164 HumanEval problems for 14B:

| Category | Count | % of fixes |
|---|---|---|
| Format/stopping fixes (correct code + trailing markdown junk) | 35 | 78% |
| Genuine algorithm improvements (wrong code → correct code) | 10 | 22% |
| Regressions (baseline passes, SFT fails) | 7 | — |

The base model already solves most problems but appends markdown fences that break the test harness. Stripping the format effect, real coding improvement is ~54% → ~60%. The remaining +17pp is output formatting — still useful since a model that can't stop talking is useless in a pipeline.

MBPP isn't affected because 3-shot prompting already teaches the base model correct output format, eliminating the SFT's formatting advantage.

<details>
<summary>14B HumanEval Output Samples</summary>

#### Example 1: Format Fix (HumanEval/18)

**Task:** *Find how many times a given substring can be found in the original string, counting overlapping cases.* `how_many_times('aaaa', 'aa')` → `3`

Both models generate identical, correct code. The baseline fails only because it appends a trailing markdown fence that the test harness interprets as part of the output.

**Baseline (FAIL):**
```
    count = 0
    for i in range(len(string) - len(substring) + 1):
        if string[i:i+len(substring)] == substring:
            count += 1
    return count
~~~
```
(The `~~~` above represents a trailing triple-backtick fence the baseline appends after the code.)

**V3 SFT (PASS):** Identical code, stops cleanly after `return count`.

---

#### Example 2: Genuine Algorithm Improvement (HumanEval/1)

**Task:** *Separate a string of nested parentheses into balanced groups.* `'( ) (( )) (( )( ))'` → `['()', '(())', '(()())']`

**Baseline (FAIL):** Uses a stack but doesn't track the actual characters — returns empty strings:
```python
    for char in paren_string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if len(stack) == 0:
                result.append('')     # always appends empty string
            else:
                stack.pop()
                if len(stack) == 0:
                    result.append('')  # never captures the group content
    return result
```

**V3 SFT (PASS):** Tracks `current_group` to build up each parenthesis group:
```python
    for char in paren_string:
        if char == "(":
            stack.append(char)
            current_group += char
        elif char == ")":
            if stack:
                stack.pop()
                current_group += char
                if not stack:
                    result.append(current_group)
                    current_group = ""
    return result
```

---

#### Example 3: Prose Contamination (HumanEval/17)

**Task:** *Parse a music notation string into beat durations.* `'o'`=4 beats, `'o|'`=2, `'.|'`=1.

**Baseline (FAIL):** Outputs a natural language explanation instead of code:
```
To solve this problem, we need to parse a string of musical notes represented
in a special ASCII format and convert it into a list of integers indicating
the duration of each note in beats. Here's a step-by-step approach...
```

**V3 SFT (PASS):** Just writes the function body:
```python
    notes = music_string.split()
    beats = []
    for note in notes:
        if note == 'o':
            beats.append(4)
        elif note == 'o|':
            beats.append(2)
        elif note == '.|':
            beats.append(1)
    return beats
```

</details>

<details>
<summary>Hyperparameter Sweep Details</summary>

All HP decisions used **MBPP 3-shot pass@1** as the sole selection metric. HumanEval was evaluated post-hoc but never influenced HP choices.

### Qwen3-14B-Base

**Phase 1** (5 LRs × 3 batch sizes on v5p-32):

| Config | MBPP 3-shot | | Config | MBPP 3-shot |
|---|---|---|---|---|
| lr1e-5_bs16 | **73.8%** | | lr1e-5_bs64 | 72.6% |
| lr2e-6_bs16 | 73.4% | | lr1e-6_bs64 | 72.4% |
| lr1e-6_bs16 | 73.2% | | lr1e-5_bs32 | 72.2% |
| lr1e-6_bs32 | 73.2% | | lr5e-5_bs64 | 71.0% |
| lr2e-6_bs32 | 72.8% | | lr5e-5_bs32 | 67.4% |

bs=16 consistently best. lr=5e-5 catastrophic.

**Phase 2** (4 WDs × 3 warmups at lr=2e-6, bs=32):

| Config | MBPP 3-shot | HumanEval | | Config | MBPP 3-shot | HumanEval |
|---|---|---|---|---|---|---|
| wd0.05_wu0.0 | **73.8%** | **77.4%** | | wd0.01_wu0.03 | 72.8% | — |
| wd0.05_wu0.1 | **73.8%** | — | | wd0.01_wu0.1 | 72.8% | 75.6% |
| wd0.1_wu0.03 | 73.2% | 74.4% | | wd0.001_wu0.0 | 72.4% | 75.6% |
| wd0.001_wu0.03 | 73.0% | 77.4% | | wd0.1_wu0.1 | 72.4% | 76.8% |
| wd0.1_wu0.0 | 73.0% | 77.4% | | | | |

wd=0.05 clearly optimal. Multiple configs tied at 77.4% HumanEval.

### Qwen3-0.6B-Base

**Phase 1** winner: lr=2e-6, bs=32 (39.0% MBPP). **Phase 2** winner: wd=0.001, wu=0.0 (39.4% MBPP, 33.5% HumanEval).

### How 14B Differs from 0.6B
- Much less HP-sensitive (1.6pp spread vs wider)
- Higher optimal LR (1e-5 vs 2e-6), higher optimal WD (0.05 vs 0.001)
- Smaller batch size preferred (16 vs 32)

</details>

<details>
<summary>V3 Code Extraction Prompt</summary>

```
Extract the content from this HTML page as clean plain text. Follow all rules below.

1. Do not use ``` code fences, markdown headers, bold, or any formatting. Output plain text only.
2. Do not include raw HTML tags or entities.
3. Do not shorten, omit, or truncate content. Extract the full page.
4. Do not add text that is not on the page.
5. Preserve all code exactly as written. Fix broken indentation so code is syntactically valid,
   and add brief inline comments to uncommented multi-step logic.

6. Output exactly [NO_USEFUL_CONTENT] if the page is not about programming/CS,
   is an index/login/error page, or lacks substantive content.

7. For Q&A pages (Stack Overflow, forums), output in this order:
   Q: <question + code>
   Reasoning: <synthesized explanation from the thread>
   A: <top answer, extracted exactly>

8. If no complete answer was provided, omit the A: section.
9. If the page discusses buggy code, prepend: code_status: buggy
10. For tutorials/docs/blogs: output text and code in reading order,
    removing navigation and boilerplate.
```

</details>

<details>
<summary>URL Sources (45 patterns)</summary>

**Stack Exchange (20):** stackoverflow.com, codereview/codegolf/cs/cstheory/math/stats/physics/unix/softwareengineering/dsp/ai/datascience/electronics/crypto/security/tex/mathematica.stackexchange.com, askubuntu.com, mathoverflow.net

**Language Docs (11):** docs.python.org, doc.rust-lang.org, pkg.go.dev, en.cppreference.com, kotlinlang.org/docs, docs.scala-lang.org, docs.julialang.org, hexdocs.pm, php.net/manual, ruby-doc.org, docs.rs

**Tutorials (5):** rosettacode.org, geeksforgeeks.org, realpython.com, w3schools.com, tutorialspoint.com

**Framework Docs (9):** developer.mozilla.org, tensorflow.org, api.flutter.dev, doc.qt.io, huggingface.co/docs, llvm.org/docs, gcc.gnu.org/onlinedocs, boost.org/doc, typescriptlang.org/docs, readthedocs.io, devdocs.io

</details>
