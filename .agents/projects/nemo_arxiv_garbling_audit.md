# Does nemotron butcher arxiv? (2026-07-09)

**Question.** A nemo arxiv extraction looked catastrophically garbled (broken inline LaTeX,
`$z$ $<$ $10$ …`). Is this systemic, or a rare tail?

**Method.** `build_devset.py arxiv-audit --method {nemo,dclm,resiliparse}` (Zephyr, in-region
us-central2): scan each pipeline's *whole* corpus, keep docs whose URL contains `arxiv.org`, and
per doc record `n_dollar` (`$` count = LaTeX-density proxy), `arxiv_refs`, `is_listing`
(`"authors and titles"` present or ≥5 `arXiv:` refs), `heavy_latex` (`n_dollar>=15`).
Output: `gs://marin-us-central2/scratch/provenance_10k/devset/arxiv_audit/{method}/`.

## Result — nemo is the CLEANEST of the three on arxiv, not the worst

| pipeline | n arxiv | heavy-LaTeX ≥15$ | listing% | avg $ | avg chars |
|---|---:|---:|---:|---:|---:|
| resiliparse (universe) | 3,827 | 7.1% | 11.4% | 4.1 | 3,185 |
| dclm | 307 | 10.7% | 9.8% | 8.2 | 4,275 |
| **nemo** | 1,195 | **4.5%** | **0.1%** | 2.2 | 1,953 |

`$`-density distribution (share of each method's arxiv docs):

| $ count | resiliparse | dclm | nemo |
|---|---:|---:|---:|
| 0 | 73.4 | 66.8 | 80.1 |
| 1–14 | 19.5 | 22.5 | 15.4 |
| 15–49 | 6.0 | 9.1 | 4.4 |
| 50–149 | 0.8 | 1.0 | 0.2 |
| 150+ | 0.3 | 0.7 | 0.0 |

## Takeaways

1. **Heavy garbling is rare (~4.5% of nemo arxiv) and nemo beats dclm (10.7%) and raw
   resiliparse (7.1%).** Severe (≥50 $) is ~0.2%; *zero* nemo arxiv docs have 150+ $. 80% have no
   `$` at all. The alarming doc was a genuine artifact but a rare tail case — not a load bug, not
   systemic corruption.
2. **Nemo almost never emits arxiv listing pages** (0.1% vs ~10%). Its single worst doc
   (`export.arxiv.org/list/hep-ph/new`, $=100) is a listing, not a paper.
3. **Mechanism.** Nemo keeps fewer + shorter arxiv docs (1,195 vs 3,827; 1,953 vs 3,185 avg chars)
   — mostly **abstract/landing pages** it then rephrases. A math-heavy abstract rephrased with the
   raw inline `$…$` tokens retained produces the garbled look. Confined to the math-heavy tail.
4. LaTeX in arxiv is source content (affects all extractors); it is NOT a nemotron-specific defect.
   Companion to [[devset_pipeline_coverage]] (arxiv coverage: nemo dominates at 87% of agreed-good).

---
# Per-register hard-negative mining (2026-07-09)

Dev set was ~100% keep in most content registers (selection filtered out negatives) -> un-tunable.
Mined negatives: `warc_junk_scan.py` over decoded_10k raw HTML (thin/linkfarm/error/spam) +
`find-negatives` workflow (sonnet, register + drop-worthiness) + `build_devset.py hq-lookup`
(join membership for kept_hq = hq false-positives).

Result: of 218 judged drops, only **33 are content-register** (rest = generic junk/nav_index).
Injected **41 negatives** (33 content + 9 hq-FPs) into the tool as a `neg_sample`/`neg_text` pool
(gs://marin-us-central2/scratch/provenance_10k/devset/hard/). Registers still at 0 negatives:
history, poetry, philosophy_religion, social_science, lifestyle — the DCLM-filtered pool genuinely
lacks bad-but-topical content for them (their junk is all frames-stubs/parked-domains -> junk register).
Finding: **hq/the pipeline already drop most junk upstream** — only 15/218 bad pages were in the pool,
hq kept 9. Companion to [[project_devset_hard_label_build]].

---
# DCLM-vs-hq disagreement search (2026-07-09)

Searched `kept_dclm AND NOT kept_hq` ranked by DCLM's own `dclm_ft` (membership has dclm_ft /
nemo_quality / fineweb_score). `build_devset.py dclm-wins`. dclm_ft is SATURATED at 1.0 for the top;
468/500 top wins are in `other` (non-curated domains). LLM-judged 497 for quality+register:

**KEY FINDING: at top DCLM confidence, 78% (392) are DCLM FALSE POSITIVES** — content-empty
question-list / FAQ-title / unanswered-quiz pages (nav_index 72, test_prep-shaped 63, qa/product FAQs).
DCLM's fastText loves them (look educational) but they have no answers; **hq correctly drops them, so
hq's PRECISION beats DCLM here**. Only 21% (105) are genuine hq false-negatives (real content hq
wrongly dropped). Biggest genuine win = **test_prep** (exam material WITH answers) — a register hq
drops wholesale that we didn't have. Added 65 wins + `test_prep` register to the tool (1934 docs).
Implication: DCLM is not beating hq in hidden ways; the recoverable gap is test_prep/study-material
with real content. Companion to [[project_devset_hard_label_build]].
