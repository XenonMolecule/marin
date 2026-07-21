# resiliparse silently drops ~7% of pages (main-content failure, NOT JS) — 2026-07-09

**Observed**: dev-set docs where the resiliparse extraction is empty but hq/nemo (LLM extractors)
keep real content (e.g. `biorxiv.org/content/10.1101/150078v2`). First-guess "JS-rendered body" was
WRONG — refuted below.

**Test** (`experiments/baseline_collection/resiliparse_diag.py`, Zephyr on Iris, us-east5 over
`documents/bert_pipeline/decoded_10k` raw HTML): per doc compute `resiliparse_main` (main_content=True,
the pipeline baseline), `resiliparse_full` (main_content=False, take-everything), and static
`text_body` lengths.

**Result** (104,150 docs):
- **7.0%** (8,077) have `resiliparse_main` EMPTY (<50 chars).
- Of those empty docs: **92%** have static body text ≥200 chars (content IS in the static HTML);
  **76%** are recovered by resiliparse's own `full` mode.
- Examples (main=0 but huge body): economist.com, faqs.org, hollywood.com, hotels.com, biorxiv-type.

**Conclusion**: NOT JavaScript. resiliparse's `main_content=True` boilerplate/main-region heuristic
fails on certain layouts and discards the entire body even though the text is present. LLM extractors
(hq/nemo) don't use that heuristic, so they keep the ~7% resiliparse drops. This is a real
extraction-quality gap the LLM pipeline closes — relevant when arguing the hq spec's value and when
scoring resiliparse as a baseline. Companion to [[project_devset_hard_label_build]].
