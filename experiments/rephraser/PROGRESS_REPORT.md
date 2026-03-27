# Progress this week (3/16-3/20)

<details>
<summary>Data Filtering Exploration</summary>

### Token Count Analysis: 150 WARC Files → Rephraser Cooldown Sweep

| Stage                  | Tokens           | % of Raw HTML | % of Resiliparse Text | Compression vs Raw |
|------------------------|------------------|---------------|------------------------|--------------------|
| Raw HTML               | ~144,000,000,000 | 100%          | —                      | 1x                 |
| Resiliparse (text extraction) | 3,630,539,832  | 2.52%         | 100%                   | ~40x               |
| Rephraser (LLM extraction)    | 362,242,311    | 0.25%         | 9.98%                  | ~400x              |
| DCLM-filtered          | 5,671,922        | 0.004%        | 0.16%                  | ~25,000x           |

 **Key takeaways**:
  - 97.5% of raw HTML tokens are boilerplate (tags, scripts, CSS, navigation, ads) — resiliparse strips 144B tokens down to 3.6B
  - The rephraser keeps 10% of the extracted text (362M of 3.6B), selecting only high-quality, informative content
  - DCLM's quality filtering is 64x more aggressive than the rephraser, keeping just 5.7M tokens (0.16% of resiliparse text)
  - End-to-end, the rephraser compresses raw HTML by 400x — from 144B tokens to 362M

<details>
<summary>Prompt that got us 90% reduction (can probably be even less picky)</summary>

```
Extract the main content from the provided HTML into clean Markdown.

First, check if the page should be rejected. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
- Not primarily in English
- Login, signup, account, checkout, paywall, or subscribe page
- Error page, captcha, cookie wall, bot check, or "session expired"
- Empty or near-empty page, directory index, or navigation-only page
- User profile, member page, or "who posted" page
- Image gallery or photo album listing without articles
- Search results page with no actual results
- Page where the main content is behind a login wall or paywall
- Product listing, gift card, or e-commerce page with prices/availability
- Social media post that is just an image or a single short caption
- Blog tag page, category page, or archive page that only lists post titles and teasers
- After removing boilerplate, the remaining useful text would be under ~100 words

If the page passes, extract with these rules:
- Output Markdown only. No commentary or analysis.
- Preserve original wording. Do not summarize or rewrite.
- Remove boilerplate: navbars, footers, sidebars, ads, share buttons, related links, breadcrumbs.
- Preserve all technical content exactly: code blocks verbatim with language tags, math/LaTeX using $$ delimiters, chemical formulas, tables.
- Do not truncate or simplify content due to length.
- Include comments/replies only if they add real information (answers, corrections).
- Start with the page title as a top-level heading if available.
```

</details>

</details>

---

<details>
<summary>Code Experiment Findings</summary>

# Code Extraction SFT: 14B Hyperparameter Sweep Results

## Results

**Data:** 260M tokens from 45 URL patterns (Stack Overflow, Stack Exchange sites, language docs, tutorials, framework docs) across 10 Common Crawl indices, extracted with Qwen3-8B rephraser using V3 commented extraction prompt. Resiliparse comparison uses 793M tokens of plain text from the same HTML.

All hyperparameter decisions used **MBPP 3-shot** as the validation metric. **HumanEval 0-shot** is the held-out test set — never used for HP selection.

**Best configs selected by MBPP 3-shot (validation), evaluated on HumanEval (test):**

| Model | Extraction | Best Config | MBPP 3-shot (val) | HumanEval 0-shot (test) |
|---|---|---|---|---|
| **Qwen3-14B-Base** | none (baseline) | — | **74.0%** | 54.3% / 72.6%* |
| Qwen3-14B-Base | V3 extraction | lr=2e-6, bs=32, wd=0.05, wu=0.0 | 73.8% (-0.2pp) | **77.4%** (+4.8pp*) |
| Qwen3-14B-Base | Resiliparse | lr=1e-6, bs=16, wd=0.05, wu=0.03 | 73.4% (-0.6pp) | 69.5% (-3.1pp*) |
| **Qwen3-0.6B-Base** | none (baseline) | — | **39.8%** | 30.5% |
| Qwen3-0.6B-Base | V3 extraction | lr=2e-6, bs=32, wd=0.001, wu=0.0 | 39.4% (-0.4pp) | **33.5%** (+3.0pp) |
| Qwen3-0.6B-Base | Resiliparse | lr=2e-6, bs=64, wd=0.1, wu=0.03 | 38.2% (-1.6pp) | 31.1% (+0.6pp) |

*\*HumanEval deltas vs fence-stripped baseline (72.6%). Raw baseline is 54.3%.*

**Key findings:**
- SFT is MBPP-neutral (slight degradation vs 74.0% baseline) — 3-shot prompting already teaches the base model correct output format
- V3 extraction delivers a genuine **+4.8pp HumanEval improvement** over fence-stripped baseline on the held-out test set
- Resiliparse's MBPP-optimal config **hurts HumanEval by 3.1pp** — V3 extraction is teaching real coding skill that plain text SFT doesn't
- V3 consistently outperforms resiliparse at both model scales

## What the SFT Actually Teaches

We stripped trailing markdown fences from the base model's outputs and re-ran evaluation locally (verified to match the official harness on unmodified outputs):

| Model | HumanEval 0-shot |
|---|---|
| Baseline (raw output) | 54.3% |
| **Baseline (fences stripped)** | **72.6%** |
| Best resiliparse SFT | 72.0% |
| Best V3 SFT | **77.4%** |

The base model already solves 72.6% of HumanEval problems — it just appends markdown fences that break the test harness. Stripping fences recovers +18.3pp. On top of that, V3 extraction SFT adds a genuine **+4.8pp** beyond what the base model can do (77.4% - 72.6% = 8 additional problems solved). Resiliparse SFT roughly matches the fence-stripped baseline.

MBPP isn't affected because 3-shot prompting already teaches the base model correct output format, eliminating the SFT's formatting advantage.

---

<details>
<summary>14B HumanEval Output Samples</summary>

#### Example 1: Format Fix (HumanEval/18)

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

```python
from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
```

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

---

<details>
<summary>Hyperparameter Sweep Details</summary>

All HP decisions used **MBPP 3-shot pass@1** as the sole selection metric. HumanEval was evaluated post-hoc but never influenced HP choices.

### Qwen3-14B-Base

**Phase 1** (5 LRs × 3 batch sizes on v5p-32):

| Config | MBPP 3-shot | | Config | MBPP 3-shot |
|---|---|---|---|---|
| lr1e-5_bs16 | **73.8%** | | lr5e-6_bs32 | 72.8% |
| lr2e-6_bs16 | 73.4% | | lr1e-5_bs64 | 72.6% |
| lr1e-6_bs16 | 73.2% | | lr1e-6_bs64 | 72.4% |
| lr1e-6_bs32 | 73.2% | | lr5e-6_bs64 | 72.2% |
| lr2e-6_bs64 | 73.2% | | lr1e-5_bs32 | 72.2% |
| lr2e-6_bs32 | 72.8% | | lr5e-5_bs64 | 71.0% |
| | | | lr5e-5_bs16 | 69.0% |
| | | | lr5e-5_bs32 | 67.4% |

bs=16 consistently best across LRs. lr=5e-5 catastrophic (67-71%).

**Phase 2** (4 WDs × 3 warmups at lr=2e-6, bs=32):

| Config | MBPP 3-shot | HumanEval | | Config | MBPP 3-shot | HumanEval |
|---|---|---|---|---|---|---|
| wd0.05_wu0.0 | **73.8%** | **77.4%** | | wd0.001_wu0.1 | 72.8% | 77.4% |
| wd0.05_wu0.1 | **73.8%** | — | | wd0.01_wu0.03 | 72.8% | — |
| wd0.01_wu0.0 | 73.4% | — | | wd0.01_wu0.1 | 72.8% | 75.6% |
| wd0.1_wu0.03 | 73.2% | 74.4% | | wd0.001_wu0.0 | 72.4% | 75.6% |
| wd0.001_wu0.03 | 73.0% | 77.4% | | wd0.1_wu0.1 | 72.4% | 76.8% |
| wd0.05_wu0.03 | 73.0% | — | | | | |
| wd0.1_wu0.0 | 73.0% | 77.4% | | | | |

wd=0.05 clearly optimal. Multiple configs tied at 77.4% HumanEval.

### Qwen3-0.6B-Base

**Phase 1** winner: lr=2e-6, bs=32 (39.0% MBPP). **Phase 2** winner: wd=0.001, wu=0.0 (39.4% MBPP, 33.5% HumanEval).

### Resiliparse 14B Phase 2 (4 WDs × 3 warmups at lr=1e-6, bs=16)

| Config | MBPP 3-shot | HumanEval | | Config | MBPP 3-shot | HumanEval |
|---|---|---|---|---|---|---|
| wd0.05_wu0.03 | **73.4%** | 69.5% | | wd0.01_wu0.0 | 73.2% | — |
| wd0.05_wu0.1 | **73.4%** | 70.7% | | wd0.05_wu0.0 | 73.2% | — |
| wd0.001_wu0.1 | **73.4%** | 69.5% | | wd0.1_wu0.0 | 72.6% | 71.3% |
| wd0.01_wu0.03 | 73.2% | 70.1% | | wd0.1_wu0.03 | 72.6% | — |
| wd0.001_wu0.0 | 72.6% | 71.3% | | wd0.001_wu0.03 | 72.4% | 69.5% |
| wd0.01_wu0.1 | 72.6% | 72.0% | | wd0.1_wu0.1 | 72.2% | 72.0% |

wd=0.05 optimal for resiliparse too. MBPP and HumanEval optimize at different configs.

### How 14B Differs from 0.6B
- Much less HP-sensitive (1.6pp spread vs wider)
- Higher optimal LR (1e-5 vs 2e-6 for V3; 1e-6 vs 2e-6 for resiliparse)
- Higher optimal WD (0.05 vs 0.001)
- Smaller batch size preferred (16 vs 32)
- wd=0.05 optimal for both V3 and resiliparse at 14B scale

</details>

---

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

---

<details>
<summary>URL Sources (45 patterns)</summary>

**Stack Exchange (20):** stackoverflow.com, codereview/codegolf/cs/cstheory/math/stats/physics/unix/softwareengineering/dsp/ai/datascience/electronics/crypto/security/tex/mathematica.stackexchange.com, askubuntu.com, mathoverflow.net

**Language Docs (11):** docs.python.org, doc.rust-lang.org, pkg.go.dev, en.cppreference.com, kotlinlang.org/docs, docs.scala-lang.org, docs.julialang.org, hexdocs.pm, php.net/manual, ruby-doc.org, docs.rs

**Tutorials (5):** rosettacode.org, geeksforgeeks.org, realpython.com, w3schools.com, tutorialspoint.com

**Framework Docs (9):** developer.mozilla.org, tensorflow.org, api.flutter.dev, doc.qt.io, huggingface.co/docs, llvm.org/docs, gcc.gnu.org/onlinedocs, boost.org/doc, typescriptlang.org/docs, readthedocs.io, devdocs.io

</details>

---

</details>

---

<details>
<summary>Math + Medical Experiments Ongoing</summary>

Same extraction pipeline as code, applied to two new domains. Extraction is in progress; SFT experiments have not started yet.

### Math (Redoing with more sources)

**Status:** Extraction ~60% complete.

**Evals:** GSM8K (CoT 8-shot), MATH suite (4-shot / 0-shot).

<details>
<summary>Math Sources (~70 sites)</summary>

**Tier 1 — Large modern (>5k pages in 2025):** geogebra.org (72k), symbolab.com (43k), khanacademy.org (35k), splashlearn.com (24k), varsitytutors.com (21k), dummies.com (9.4k), mathcentre.ac.uk (8.1k), openstax.org (7.6k), math.libretexts.org (7.4k), purplemath.com (7.4k), mathway.com (7.3k), sparknotes.com (7.2k)

**Tier 2 — Medium modern (1k–5k):** math-drills.com, forums.wolfram.com, gradesaver.com, onlinemathlearning.com, mathisfunforum.com, savemyexams.com, softschools.com, illustrativemathematics.org, nrich.maths.org, desmos.com, math-only-math.com, brilliant.org

**Tier 3 — Small modern (<1k):** helpingwithmath.com, tutorial.math.lamar.edu, intmath.com, gmatclub.com, coolmath.com, virtualnerd.com, basic-mathematics.com, analyzemath.com, mathwords.com, mathplanet.com, mathsisfun.com, freemathhelp.com, math.com, betterexplained.com, kutasoftware.com, mathbitsnotebook.com, themathpage.com, chilimath.com, betterlesson.com, emathinstruction.com, statlect.com, mathportal.org

**Historical Q&A goldmines (2013/2016 crawls):** mathhelpforum.com (506k in 2016 — largest math forum), chegg.com (435k in 2013), mathforum.org (389k in 2013), jiskha.com (258k in 2013), wyzant.com (252k in 2013), brainly.com (203k in 2016), mathoverflow.net (202k in 2013), coursehero.com (190k in 2016), math.stackexchange.com (148k in 2013), brainmass.com (114k in 2016), geogebra.org (104k in 2018), algebrahelp.com (58k in 2016), physicsforums.com (53k in 2013), cliffsnotes.com (43k in 2013), openstudy.com (42k in 2013)

</details>

<details>
<summary>Math Extraction Prompt</summary>

```
Extract the main content from the provided HTML into clean Markdown.

First, check if the page should be rejected. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
- Not primarily in English
- Login, signup, account, checkout, paywall, or subscribe page
- Error page, captcha, cookie wall, bot check, or "session expired"
- Empty or near-empty page, directory index, or navigation-only page
- User profile, member page, or "who posted" page
- Image gallery or photo album listing without articles
- Search results page with no actual results
- Page where the main content is behind a login wall or paywall
- Product listing, gift card, or e-commerce page with prices/availability
- Social media post that is just an image or a single short caption
- Blog tag page, category page, or archive page that only lists post titles and teasers
- After removing boilerplate, the remaining useful text would be under ~100 words

If the page passes, extract with these rules:
- Output Markdown only. No commentary or analysis.
- Preserve original wording. Do not summarize or rewrite.
- Remove boilerplate: navbars, footers, sidebars, ads, share buttons, related links, breadcrumbs.
- Preserve all technical content exactly: code blocks verbatim with language tags, math/LaTeX using $ delimiters, chemical formulas, tables.
- Do not truncate or simplify content due to length.
- Include comments/replies only if they add real information (answers, corrections).
- Start with the page title as a top-level heading if available.
```

</details>

### Medical

**Status:** Domain sources identified, extraction not yet started.

<details>
<summary>Medical Sources (~50 sites across 4 tiers)</summary>

**Tier 1 — Q&A Forums:** forums.studentdoctor.net (23M+ posts), healthboards.com (5M+ messages), healthunlocked.com, allnurses.com (1M+ members), patient.info, medhelp.org

**Tier 1 — Reference:** mayoclinic.org, webmd.com, drugs.com, clevelandclinic.org, ncbi.nlm.nih.gov, medlineplus.gov, merckmanuals.com

**Tier 2 — Education:** radiopaedia.org, kenhub.com, osmosis.org, teachmeanatomy.info, teachmesurgery.com, teachmephysiology.com, geekymedics.com, pathologystudent.com

**Tier 2 — Specialty Forums:** psychforums.com, crazyboards.org, healingwell.com, inspire.com, health.stackexchange.com, dentaltown.com

**Tier 2 — Reference:** rxlist.com, emedicine.medscape.com, wikem.org, fpnotebook.com, openanesthesia.org, lifeinthefastlane.com, rebelem.com, emcrit.org

**Tier 3 — Niche:** innerbody.com, eyewiki.org, orthobullets.com, cancerresearchuk.org, breastcancer.org, diabetes.org, heart.org, cdc.gov, who.int, nhs.uk

**Tier 3 — Pharmacology/Nursing:** pharmacytimes.com, nursingtimes.net, registerednursern.com, nurseslabs.com, safemedication.com

</details>

<details>
<summary>Medical Extraction Prompt</summary>

```
Extract the medical content from this HTML page as clean Markdown. Follow all rules below.

1. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
   - The page is not about medicine, health, nursing, pharmacology, or clinical science
   - Index page, search results, listing page, category page, user profile, or navigation page without substantive content
   - Product page, supplement advertisement, or commercial health product promotion
   - Login, signup, paywall, error page, or empty page
   - Not primarily in English

2. Remove all boilerplate: navigation, footers, sidebars, ads, user signatures, join dates, post counts, reaction buttons, related thread links, and cookie/privacy banners. Do not include any raw HTML tags or undecoded HTML entities in your output. Do not output [[ ## text ## ]] or similar framework markers.

3. Preserve all medical terminology, drug names, dosages, lab values, and anatomical terms exactly as written. Use standard medical abbreviations where they appear in the source (e.g., BP, HR, CBC, BMP). Do not expand or alter abbreviations.

4. For Q&A pages (health forums, medical Q&A sites, patient communities, clinical discussion boards), output in this format:
   - Start with the thread title as a top-level heading.
   - Use up to three sections: ## Question, ## Discussion, ## Answer.
   - **Question**: The original poster's question or clinical scenario, preserving medical details faithfully (symptoms, medications, lab values, history).
   - **Discussion**: Synthesize the useful replies into a coherent medical explanation written in an impersonal clinical voice — do not address users by name, do not narrate the conversation, and do not attribute information to specific posters. Reproduce the clinical reasoning, differential diagnoses, treatment recommendations, and explanations from the replies. When a reply explains a mechanism, treatment rationale, or diagnostic approach, include the full explanation. If the thread ends without resolving the question, state only what was established.
   - **Answer**: The final answer or clinical recommendation, only if present in the thread. If no clear answer was reached, omit the ## Answer section entirely.
   - If the thread contains multiple distinct questions, use separate ## Question, ## Discussion, ## Answer blocks for each.

5. For reference pages (medical encyclopedias, drug databases, clinical guidelines, educational content):
   Output the text in reading order. Keep all explanatory content, definitions, mechanisms of action, indications, contraindications, dosages, side effects, and clinical pearls. For drug pages, preserve the structured format (indications, dosage, interactions, etc.). For disease pages, preserve etiology, pathophysiology, symptoms, diagnosis, and treatment sections.

6. Do not add information that is not on the page. Every claim, drug interaction, dosage, or clinical recommendation in your output must come from the source HTML.
```

</details>

### Law (Exploratory)

Domain sources identified but not prioritized. ~25 legal Q&A sites (avvo.com, justia.com, law.cornell.edu, etc.) with domain-adapted extraction prompt preserving case citations and legal terminology.

</details>

---

<details>
<summary>Potential Regimes</summary>

</details>

---

<details>
<summary>Yejin’s Feedback</summary>

</details>

---