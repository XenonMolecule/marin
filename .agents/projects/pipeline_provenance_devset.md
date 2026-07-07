# Pipeline Provenance → Dev Set for Extraction-Spec Iteration

**Status:** Phase 0 in progress (2026-07-04). Owner: Michael Ryan.
**Branch:** qwen3-useful-classifier (working).

## Goal

`high_quality` (our LLM `html→text` extraction, `fastpipe_v3`) is Pareto-optimal
on uncheatable-eval loss and strong on code, but **slips on specific domains**:
BBC news + AO3 fiction (per-domain bpb) and broad-knowledge / commonsense
benchmarks (LAMBADA, Jeopardy, ARC, COPA, Winograd, CSQA — DCLM CoreV2).

Build a **dev set of "pure-signal" documents** we can run the extraction spec
against, to iterate: keep what helps, drop what harms. Two directions, both:
1. **Recover weak domains** — what DCLM/Nemotron keep that we drop that would
   help news/fiction/knowledge.
2. **Protect code strength** — what we keep that DCLM/Nemotron drop that makes us
   strong on code/technical.

## Dev-set principle: OPINIONATED and WINNER-ANCHORED (per Michael, 2026-07-04)

The goal is **Pareto optimality** — high_quality should absorb every pipeline's
per-domain edge. So the dev set is not symmetric hq-vs-dclm diffing. For each
eval domain, first find the **winning pipeline on that domain**, then mine THAT
pipeline's filtering decisions there:

- If DCLM wins fiction (ao3) → the signal docs are the ones **DCLM keeps** in
  fiction (esp. DCLM-keeps-but-hq-drops). Preserve those.
- If Nemotron wins news (bbc) → look to **Nemotron's** news keep/drop decisions.
- Where **high_quality wins** (code, overall loss) → mine OUR keep-set that
  DCLM/Nemotron drop, and treat the loser's extra docs there as candidate garbage.

So the mining direction per domain is set by a **per-domain winner map**, not a
fixed reference method. Requires: per-method × per-domain eval numbers for the
10k sweep (uncheatable per-domain bpb + CoreV2 accuracy). Winner defined at the
scale that matters (default: largest / crossover scale, ~2.9B–8B, since some
edges only appear at scale). Dev set = union over domains of
(domain-winner's signal docs), each labeled (domain, winning_pipeline,
eval_similarity, who-kept/dropped).

## The key enabling fact (why this is tractable without influence functions)

All pipelines extract the **same 10,364 WARCs** and every doc carries `url`:
- hq (CORRECTED source — the full trained corpus, us-central1, NOT the 516-WARC
  partial `fastpipe_v3-da3893385e`): the decon+dedup HF export with url re-attached:
  `gs://marin-us-central1/documents/baseline_high_quality_hf_export/10364warcs/joined/*.parquet`
  → cols `text,url,warc_record_id,warc_file,snapshot`; 19,968,996 docs / 10,364 WARCs.
  (No fasttext/modernbert scores here — those only exist in the fastpipe partial.)
  **Region wrinkle:** hq is us-central1, the others us-central2 → extract hq's tiny
  per-method parquet in-region on central1, copy (~1GB, one-time) to central2, join there.
- dclm: `filtered/dclm_400m_1x_10k_dclm_resharded-1fe977/*.jsonl.gz`
  → `text,url,warc_record_id,dclm_fasttext_score,dclm_language_score`
- nemotron_full: `filtered/dclm_400m_1x_10k_nemotron_full-96bad9/*.jsonl.gz`
  → `text,url,nemotron_quality,nemotron_kind,nemotron_id`
- fineweb_edu: `filtered/dclm_400m_1x_10k_fineweb_edu-0d49e9/**/*.jsonl.gz`
  → `text,url,file_path,dump,fineweb_score,fineweb_int_score`
- resiliparse (unfiltered superset / denominator):
  `extracted/dclm_400m_1x_10k_resiliparse-f0887f/data-NNNNN-of-10364.jsonl.gz`
  → `text,url` (1:1 with WARCs)
- fineweb_cc: pre-tokenization text largely deleted (only tokenized cache); skip.

**`url` is the universal join key.** Each pipeline = a different keep/drop-and-
rewrite decision over the same source URLs → a natural experiment. Document-level
set differences are exact; small matched-token trainings causally confirm which
docs move the evals. Influence functions are the fallback, not the lead — the
counterfactual corpora already exist.

## Established prior findings (do not redo)

- **Contamination ruled out** (`scratch/decontamination_analysis.md`): benchmark
  overlap on gap-driving tasks is negligible in every method and uncorrelated
  with score; nemotron beats hq with *less* overlap. Gap is genuine coverage.
- Benchmark-gap reports: `scratch/plots/benchmark_examples/corev2_*.html` — gaps
  cluster on broad world-knowledge + commonsense-narrative.
- Loss subcomponents: `scratch/plots/subcomponents_10k/` (uncheatable per-domain
  bpb incl. bbc_news, ao3_english, github_*, arxiv_*, wikipedia).

## Hypotheses

- **H1 genre-drop (coverage):** spec discards whole registers (news, fiction,
  forum/conversational) DCLM keeps. Predict hq retention on
  bbc.co.uk/archiveofourown.org/reddit ≪ dclm.
- **H2 code strength:** hq wins code by preserving clean code/tech-doc content
  and stripping web chrome; predict hq retention on github/stackoverflow/docs ≫
  dclm/nemo, and hq∖dclm clusters on code/technical.
- **H3 filtering vs mangling:** for URLs kept by BOTH hq and dclm in a weak
  domain, is hq's text shorter/lossy? If yes the fix is spec *instructions*, not
  a keep-list. (Unique lever: same-URL text under every pipeline.)
- **H4 causal:** re-adding identified news/fiction docs to hq closes BBC/AO3 loss
  at 447M matched tokens; removing hq code docs collapses code.

## Plan

- **Phase 0** `experiments/baseline_collection/provenance_audit_10k.py`
  - `extract`: map each pipeline's shards → per-method parquet
    (`url,domain,text_len,scores`) under
    `gs://marin-us-central2/scratch/provenance_10k/per_method/{method}/`.
    Checkpointed (skip-if-exists). In-region us-central2 CPU iris job.
  - `join`: duckdb full-outer-join (union-then-left-join) → durable membership
    table `.../membership/` + summaries `.../summary/` (domain_retention,
    named_domain_retention, membership_patterns).
  - Launch: `uv run iris --cluster=marin job run --region us-central2
    --enable-extra-resources --cpu=32 --memory=128GB --extra=cpu -- python
    experiments/baseline_collection/provenance_audit_10k.py extract --method all`
- **Phase 0b** per-domain retention table + characterize hq∖dclm and dclm∖hq
  (top domains, length, dclm fastText dist, language, topical clusters).
- **Phase 0c** same-URL extraction diff (H3) — pass-2 targeted text read for
  URLs kept by both hq+dclm in weak domains.
- **Phase 1** retrieval attribution: embed eval-domain text (BBC/AO3/code/
  knowledge), score set-difference docs by similarity → **union-anchored dev
  set**, each doc labeled by (eval-similarity, spec-decision).
- **Phase 2** 10+ causal ablation trainings (447M/157M matched tokens):
  hq+readd(news/fiction), hq−code, etc. Validate the dev set / prove the hero run.
- **Phase 3** rewrite extraction spec; verify new keep/drop against dev-set labels
  using the existing useful-classifier harness.

## PHASE 0 RESULT (2026-07-04) — the gap is THREE different stories

Per-domain URL retention (% of union URLs each method keeps), @8B winner:

| eval domain | winner | HQ% | DCLM% | NEMO% | FWEDU% | verdict |
|---|---|---|---|---|---|---|
| ao3_english (fiction) | DCLM | **0.5** | 61.0 | 52.9 | 0.0 | COVERAGE gap — mine DCLM/NEMO |
| arxiv | DCLM | **8.5** | 28.4 | 74.7 | 0.1 | COVERAGE gap — mine NEMO/DCLM |
| reddit (social) | — | **7.5** | 39.4 | 80.2 | 0.4 | COVERAGE gap — mine NEMO |
| bbc_news | FW_CC | **62.0** | 34.4 | 15.9 | 14.0 | MANGLING — HQ keeps MOST, still loses |
| github | HQ | 71.0 | 26.4 | 26.5 | 3.4 | HQ edge — PROTECT |
| wikipedia | HQ | 82.8 | 24.9 | 9.0 | 27.7 | HQ edge — PROTECT |

Membership: union 32.49M URLs; kept hq 19.94M / dclm 5.88M / nemo 9.92M (unique;
17.76M incl. 5 synthetic rephrase variants) / fwedu 2.33M. **HQ keeps the MOST URLs
overall (15.8M unique to HQ)** — it is a *selective register filter*, not a blunt
over-filter.

Three actionable stories:
1. **COVERAGE gap (fiction/arxiv/reddit/forum/social):** the spec's "useful content"
   definition throws these registers away wholesale (fiction ~99% dropped). → spec
   keep-list fix; dev set mines DCLM/NEMO kept docs there.
2. **MANGLING (news):** HQ keeps 62% of bbc.co.uk (more than anyone) yet loses the
   bbc_news loss eval → NOT a filtering gap. The loss must come from extraction quality
   or which-BBC-docs. → Phase 0c same-URL diff; spec extraction-instructions fix.
3. **HQ edge (code/technical/wiki/reference):** HQ retains far more than everyone
   (github 71% vs fwedu 3.4%). → PROTECT; dev set includes these as positive anchors.

Artifacts: membership table `gs://marin-us-central2/scratch/provenance_10k/membership/`;
summaries `.../summary/{domain_retention,named_domain_retention,membership_patterns}.json`;
report `experiments/baseline_collection/provenance_report.py`.

## PHASE 0c RESULT (2026-07-04) — news is NOT mangling; extractor is faithful

Chased the news gap (HQ keeps 62% of bbc.co.uk yet loses bbc_news bpb). Findings:
- **Length:** on same news URLs, hq_len ≈ dclm_len (bbc 0.91x, nyt 1.02x), slightly
  longer than nemo. NOT truncation.
- **Rewriting:** same-URL word-5-gram overlap HQ vs DCLM (verbatim) = **median 0.98,
  92% of pairs verbatim (>=0.8), 1% heavily reworded**. HQ does NOT paraphrase news.
- **Tail (8%):** where they differ, HQ is usually CLEANER (strips nav/boilerplate
  DCLM keeps); a minority HQ trims to a stub.
- **Winner map:** bbc_news @8B FW_CC 0.748 > NEMO 0.767 > HQ 0.770 ≈ DCLM 0.772. HQ
  TIES DCLM; both trail FW_CC/NEMO by ~0.02 bpb.

**Conclusion:** the news gap is NOT an extraction defect. HQ's news extraction is
faithful and full-length (this is WHY it wins code/wiki). The small (~0.02) news gap
to FW_CC/NEMO is compositional (mixture proportion / other outlets), not per-doc
quality. **Priority returns to the COVERAGE gaps** (fiction 0.11 bpb, arxiv, reddit) —
5x larger and a clear filtering problem. Tooling built (reusable):
`provenance_audit_10k.py {lengthcompare,fetchtext}`, `scratch/build_newsdiff_report.py`,
viewer `scratch/newsdiff_viewer.html`. The same-URL faithfulness check is a good
**guardrail** for the dev set (ensure a new spec doesn't start mangling), but coverage
is the primary signal.

## COMPREHENSIVE COVERAGE (2026-07-04) — HQ is broadly permissive; one surgical hole

Retention by content category over ALL 32.5M union URLs (top-5000 domains = only
39% of corpus, so category/TLD cuts are the comprehensive view):

| category | %corpus | HQ% | DCLM% | NEMO% | FWEDU% |
|---|---|---|---|---|---|
| other | 85.2 | 61.8 | 16.6 | 31.2 | 7.4 |
| blog | 7.7 | 55.8 | 30.5 | 26.0 | 4.9 |
| news | 3.6 | 60.5 | 19.5 | 30.4 | 6.8 |
| forum_social | 1.4 | 71.6 | 22.7 | 22.7 | 2.9 |
| reference_wiki | 0.7 | 59.7 | 39.3 | 15.5 | 18.6 |
| code_tech | 0.6 | **84.1** | 19.3 | 15.2 | 2.3 |
| academic | 0.3 | 54.9 | 16.0 | 37.7 | 11.4 |
| qa_help | 0.3 | 67.6 | 19.7 | 23.6 | 7.7 |
| ecommerce | 0.2 | 32.4 | 10.5 | 63.4 | 4.1 |
| **fiction** | 0.1 | **2.2** | 49.1 | 57.9 | 0.4 |

- **HQ is the MOST comprehensive extractor** — highest retention in nearly every
  category. Not an over-filterer; broadly permissive (keeps 15.8M URLs no one else does).
- **Fiction = the one catastrophic content blind spot.** HQ keeps 2.2% vs DCLM 49% /
  NEMO 58%. Every fiction site: ao3 0.9, fanfiction 0.4, wattpad 2.1, fictionpress 2.0,
  literotica 2.9, deviantart 1.3, royalroad 1.3. Systematic categorical rejection of
  creative writing.
- **arxiv-specific hole:** arxiv.org HQ 8.5 (NEMO 75) — but HQ is FINE on other academic
  (ieee 67, biomedcentral 61, plos 53). So it's preprint/equation-page-specific.
- **Under-keeps that are CORRECT (garbage NEMO over-keeps):** ecommerce/booking/directory
  (airbnb/agoda/bizrate/ticketsinventory — NEMO 63%, HQ 32%). HQ rightly filters these.
- **HQ unique strengths (100/0 over the others):** technical/hobbyist forums (murga-linux,
  daemonforums, perlguru, vectorlinux, electro-music, hobiecat) + non-English (aif.ru,
  taz.de, denik.cz, estadao). TLD: edu 82, gov 83.
- **News/FW_CC:** HQ keeps most news (60.5%), extracts faithfully (Phase 0c). FW_CC's
  ~0.02 bpb news edge is compositional (lightest filter → broadest natural web prose);
  FW_CC docs unrecoverable → deprioritize vs fiction/arxiv.

**Spec implication:** the fix is NARROW — teach the spec that narrative/fiction and
arxiv preprints ARE useful content; keep filtering ecommerce/directory junk. Tools:
`provenance_audit_10k domaindiff`; summaries `.../summary/{category,tld,domain_retention_full}.json`.

## KNOWLEDGE-GAP DECOMPOSITION (2026-07-05) — it's NOT just fiction

Coverage (retention) != benchmark performance: training is isoflop, so per-token
DENSITY/composition matters, not whether a URL is kept. HQ keeps the most URLs but
is the most DILUTED corpus (85.7% generic "other" vs DCLM 72%). The knowledge gap
splits into 3 sub-gaps with DIFFERENT winners (`per_benchmark_bpb_8B.json`):

1. **Fact recall -> DCLM.** jeopardy +0.184 (HQ 0.679 vs DCLM 0.495!), naturalqs +0.058,
   squad +0.013, coqa +0.020. DCLM concentrates knowledge-dense expository (reference_wiki
   2.4x HQ char-share, blog 2.6x, academic 1.7x); HQ diluted. COMPOSITION/DENSITY fix.
2. **Narrative/discourse -> DCLM/FWCC.** lambada +0.133. Ties to the fiction hole (HQ
   fiction char-share 90x lower). COVERAGE fix (stop dropping fiction/long narrative).
3. **Science MC -> NEMO.** arc_easy +0.037, arc_challenge +0.043, sciq, winogrande +0.053.
   NEMO wins with LESS knowledge char-share than HQ (1.05 vs 1.69%) -> its edge is the
   SYNTHETIC rephrasing (5 QA-style variants/doc), not natural content. SYNTHETIC fix
   (needs a training ablation to confirm; not visible in URL data).

HQ WINS csqa/piqa/drop (everyday commonsense + reading-comp) -> protect.

Composition (char-weighted category share, `composition_by_method.json`): knowledge
char-share HQ 1.69 / DCLM 2.77 / NEMO 1.05 / FWED 4.18. HQ "other" 85.7% (most diluted).

Multi-agent workflow `knowledge-gap-investigation` (wf_694387d9-a3b) verifying each
sub-gap adversarially + producing ranked dev-set doc priorities. Data + brief in
`scratch/gap_investigation/`.

## Cost discipline

- Everything touching corpora runs in-region us-central2 (reuse matched_viewer
  cross-region guards). Laptop only pulls the small summary JSONs.
- TPU (training) is free on TRC; GCS egress is the only real cost — avoid it.
