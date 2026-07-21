# Dev-set domain priorities — WORK IN PROGRESS

**Status: WIP synthesis (2026-07-07).** Consolidates every discovery so far about which
domains/registers hq is missing, across all phases, with numbers + an example doc each.
This is a *prep artifact for choosing dev-set targets*, not a final spec. Example docs are
illustrative (drawn from the pool), not a keep-list.

---

## TL;DR — priority order for the dev set

hq is a **broadly permissive** extractor (keeps the most URLs of any pipeline, 15.8M unique
to it) with a few **surgical holes** and a **density/quality** problem. The knowledge gap is
not one thing — it decomposes into coverage holes, a composition/density dilution, and a
synthetic-data effect. Priorities, highest first:

1. **Fiction / long-form narrative** — catastrophic coverage hole (hq keeps **2.2%**). COVERAGE fix.
2. **Knowledge-dense expository** (reference/wiki, academic, expository blogs) — hq is *diluted*, not absent. DENSITY / per-doc CONTENT-QUALITY fix.
3. **Science explainers + MC-style science** — arc/sciq gap; partly Nemotron's synthetic edge. COVERAGE + SYNTHETIC.
4. **How-to / procedural blogs** (cooking, health, everyday) — biggest Phase-C missing register; platform-hosted so needs content-level selection. CONTENT-QUALITY fix.
5. **arxiv / preprints** — narrow specific hole (hq 8.5% on arxiv.org; fine elsewhere academic). COVERAGE fix.
6. **Q&A / forum / social** (reddit, stackexchange) — coverage gap vs Nemotron. COVERAGE fix.
7. *(watch, not add)* **News** — hq already keeps the most (62%) and extracts faithfully; the residual edge is compositional/unrecoverable. GUARDRAIL only.

**PROTECT (hq already wins — don't regress):** code/tech (hq 84% retention, wins bpb),
everyday commonsense + reading-comp (csqa/piqa/drop), technical/hobbyist forums, non-English.

---

## Evidence base (which phase produced what)

| phase | what | artifact |
|---|---|---|
| 0 / 0b | per-URL membership + per-domain retention (who keeps what) | `.../provenance_10k/membership/` + `summary/` |
| 0c | same-URL extraction diff (filtering vs mangling) | project doc |
| winner map | per-domain @8B winner (uncheatable bpb + CoreV2) | project doc |
| comprehensive | retention by content category over all 32.5M URLs | `summary/{category,tld,...}.json` |
| knowledge-gap | per-benchmark bpb decomposition (3 sub-gaps) | `per_benchmark_bpb_8B.json` |
| Phase 2 ablation | reweight/epoch causal runs (density & epoch REFUTED) | separate report (`scratch/ablation_report/`) |
| Phase A/B/C | eval-anchored keyword scan → missing domains + verifier tiers | `.../keyword_agg/`, this session |

---

## The prioritized registers (numbers + example doc)

### 1. Fiction / narrative — the one catastrophic coverage hole
**Numbers.** Retention over all fiction URLs: **hq 2.2%** vs DCLM 49.1% / NEMO 57.9% / FWEDU 0.4%.
Per site: ao3 0.9, fanfiction 0.4, wattpad 2.1, fictionpress 2.0, literotica 2.9, deviantart 1.3,
royalroad 1.3 — systematic categorical rejection of creative writing. Benchmark tie-in: **lambada
+0.133 bpb** (hq worse) — narrative/discourse. Phase C: fanfiction/AO3/literotica appear as
*pure coverage gaps* (quality-gap ≈ 0 → hq simply never has them).
**Fix type.** COVERAGE — teach the spec that narrative/fiction is useful content.
**Example doc.** `fanfiction.net` — *"Home V: Obsession. By Nan Smith. Rated: PG. Disclaimer: The
familiar characters and settings in this story are not mine; they belong to DC Comics, Warner Bros…"*
(also `archiveofourown.org` — *"Vapors… Kushina had heard that giving birth was a noisy, messy, and
painful affair…"*). **Honest note on the link:** fiction shows up in the keyword scan only
*incidentally* (a story that happens to mention baking/grandma), so its eval tie is NOT a per-doc
subject — it's the **aggregate lambada +0.133 bpb** narrative/discourse signal. Fiction is a
coverage decision justified by the benchmark decomposition, not by keyword matches.

### 2. Knowledge-dense expository — hq is diluted, not absent
**Numbers.** hq keeps reference/wiki (59.7%), academic (54.9%), blog (55.8%) — it HAS them — but
the *char-weighted knowledge share* is low: knowledge char-share **hq 1.69%** vs DCLM 2.77 / FWEDU
4.18 (NEMO 1.05). DCLM concentrates knowledge-dense expository (reference_wiki **2.4×** hq char-share,
blog **2.6×**, academic **1.7×**); hq's corpus is 85.7% generic "other" (most diluted). Benchmark
tie-in: **fact recall → DCLM**: jeopardy **+0.184** (hq 0.679 vs DCLM 0.495!), naturalqs +0.058,
squad +0.013, coqa +0.020. Phase C 3/3-verifier agreement (all filters kept, hq alone dropped) is
led by reference/answers domains: ipl.org, wikipedia, wikimili, answers.com, slideshare, slideplayer.
**Fix type.** DENSITY / per-doc CONTENT QUALITY. NB the Phase-2 ablation showed a blunt density
reweight is *not* the fix (see ablation report) → it's about keeping the *right, fact-dense* pages.
**Example doc** (3/3 verifiers kept, hq alone dropped): `wisc.edu` — *"Immunotherapy is the process
of using a component, or components, of the immune system to eliminate cancer. One of the earliest
observations that immunotherapy could be beneficial in pediatric…"* → eval link: **medicine: cancer
treatment / cure**. A clean expository medical page all three quality filters kept and hq threw away.

### 3. Science explainers + MC science
**Numbers.** **Science MC → NEMO**: arc_easy +0.037, arc_challenge +0.043, sciq, winogrande +0.053
(hq worse). Nemo wins with LESS knowledge char-share than hq (1.05 vs 1.69%) → its edge is SYNTHETIC
rephrasing (5 QA-style variants/doc), not natural content. Phase C surfaces science-explainer domains
hq drops: biomedcentral, sciencedaily, phys.org, plos, nih.gov. (hq is fine on some academic: ieee 67,
biomedcentral 61, plos 53 — the hole is preprint/equation-specific + explainer density.)
**Fix type.** COVERAGE of science explainers + (separately) a SYNTHETIC-data question Nemo's edge raises.
**Example doc.** `biomedcentral.com` — *"Review · Open Access · Pharmaceuticals and personal care
products in water and wastewater: a review of treatment processes and use of photocatalyst immobilized
on functionalized carbon…"* → eval link: **biochemistry / environmental chemistry**. Also `plos.org`
— *"Fish Predation by Semi-Aquatic Spiders: A Global Pattern"* → **biology: amphibian metamorphosis /
predation**. Clean, knowledge-dense science reviews the quality filters vary on and hq drops.

### 4. How-to / procedural blogs (cooking, health, everyday)
**Numbers.** Phase C's single biggest missing register: **blog = 815k missing docs** (of which
27.7k quality-gap). Top eval subjects carried by the missing docs are overwhelmingly procedural/
everyday: cooking (pasta), baking (cookies), health (diabetes & carbs, infection), everyday (disposing
items, clothing, recycling), nutrition (calories). Platform-hosted (blogspot 498k / wordpress 215k
coverage-gap) → the domain doesn't tell you if it's knowledge-rich, so this needs **content-level**
selection (cross-pipeline / independent-judge), not a domain rule.
**Fix type.** CONTENT-QUALITY keep-decision on platform blogs.
**Example doc** (3/3 verifiers kept, hq dropped): `blogspot.com` — *"10 Most Toxic Places on Earth.
Any substance, no matter how seemingly benign it may be, can cause…"* → eval link: **environmental
science: activities that add pollutants**. Illustrates the platform problem: the domain is just
"blogspot"; the *content* is a knowledge-rich environmental explainer the filters kept and hq dropped.

### 5. arxiv / preprints — narrow specific hole
**Numbers.** arxiv.org **hq 8.5%** vs NEMO 75% / DCLM 28% — but hq is fine on other academic
(ieee 67, biomedcentral 61, plos 53). So it's preprint/equation-page-specific, not academic-wide.
**Fix type.** COVERAGE — handle equation/preprint pages.
**Example doc.** `arxiv.org` — *"Physics. Authors and titles for Nov 2015 [ total of 1122 entries ]…
arXiv:1511.00023 [pdf]…"* → eval link: **astronomy / chemistry** (via listed titles). NB this is a
*listing* page — the actual value is the preprints/equation pages, which resiliparse also extracts
poorly; the arxiv hole is as much an extraction problem as a keep-decision.

### 6. Q&A / forum / social
**Numbers.** reddit (social) **hq 7.5%** vs NEMO 80.2% / DCLM 39.4%; forum_social category hq 71.6%
overall but the *eval-relevant* Q&A (stackexchange, answers.com) shows up in Phase C coverage gaps.
**Fix type.** COVERAGE of Q&A/how-to threads (protect the technical forums hq already wins).
**Example doc.** `stackexchange.com` — *"Assume a planet that always presents one side to the sun.
No moons. The orbit of the planet around the star is essentially spherical…"* → eval link:
**astronomy: Earth's rotation / tidal locking**. Also `answers.com` — *"chemistry (kĕm'ĭ-strē) n. The
science of the composition, structure, properties, and reactions of matter…"* → **biochemistry**. Clean
factual Q&A/reference hq drops (protect the technical forums like murga-linux/perlguru where hq wins).

### 7. News — WATCH, do not prioritize
**Numbers.** hq keeps the MOST news (bbc.co.uk **62%** vs DCLM 34 / NEMO 16 / FWEDU 14) and extracts
it faithfully (Phase 0c: 92% of same-URL pairs verbatim, hq does NOT paraphrase/mangle). The residual
bbc_news bpb edge (FW_CC ~0.02) is *compositional* (lightest filter → broadest natural prose) and
FW_CC's docs are unrecoverable. → **guardrail** (don't start mangling news), not a target.

---

## The 0-vote residual (this session, exploratory)
Using resiliparse (raw unfiltered pool) as the *universe*: **3.73M** coverage-gap docs exist in the
pool that hq dropped; graded by how many independent quality filters (dclm+nemo+fwedu) kept them:
3/3 = 10.5k (hq alone wrong), down to **0-vote = 7.77M** (no filter kept — mostly junk by construction,
since it's the reject pile of every classifier). Interesting seam inside the residual: practical-
expertise domains (cooking/recipes, health/fitness, hobby) the academic-leaning filters systematically
discard. Independent-LLM-judge pass in flight to estimate the real needle-rate. Treat as a haystack to
mine, not a keep-list.

**Residual seam example** (0-vote — no filter kept it): `cookinglight.com` — *"…switch it up with our
favorite Cooking Light Diet side dishes under 200 calories…"* → eval link: **cooking: preparing pasta**.
Plausibly useful practical content that every academic-leaning filter dropped. **Residual junk example**
(also 0-vote): `winecellarkw.com` — *"Call (519) 578-0505 … Store Hours Sunday: CLOSED Monday: CLOSED…"*
→ matched **everyday life: wine storage** on keywords but is a store-hours page. The seam and the junk sit
side by side in the residual — hence the judge pass.

### Judge pass — needle-rate in the 0-vote residual (2026-07-07)
Independent judge (Claude, beholden to none of the pipeline filters) read 132 snippets — 22 per
category — from the 0-vote residual and labeled each knowledge-rich / borderline / junk:

| category | clear-useful | borderline | junk | read |
|---|---|---|---|---|
| **cooking** | **~41%** (9/22) | 2 | 11 | real recipes w/ ingredients+steps — the standout seam |
| blog_platform | ~14% (3/22) | 10 | 9 | recipes/reviews/technical posts buried in personal diary |
| other | ~14% (3/22) | 9 | 10 | occasional Perl docs / science / how-to among store+nav junk |
| health_fitness | ~14% (3/22) | 8 | 11 | **borderline = calorie-DB entries** (factual but thin/templated); + gym promos |
| reference | ~9% (2/22) | 9 | 11 | 0-vote reference is thin .gov/.edu **data/nav** (obituary search, specimen records); the RICH reference is in the 3/3 tier, not here |
| hobby_sport | ~5% (1/22) | 2 | 19 | mostly **e-commerce** (the regex caught guitarcenter/soccergarage/marinedepot) — a category artifact, little real hobby expertise |

**Verdict: ~16% clear-useful overall — junk-dominated as expected, but structured.** The usable
0-vote seam is **narrower than the top-domain view suggested**: it's concentrated in **recipes /
cooking how-to** (~40%), which is genuinely knowledge-rich procedural content mapping to the piqa/
hellaswag cooking-and-everyday eval items. The other "practical" categories are weaker than they
looked — health is mostly repetitive nutrition-fact tables, and "hobby" was inflated by retail domains.
So the 0-vote edge is real but should be mined **selectively (recipe/how-to procedural text)**, not
taken as "add practical domains wholesale." Caveats: judge on 400-char snippets, N=132, coarse
category regex (understates hobby by retail contamination).

### Judge pass — the 3/3 tier (same protocol, 132 snippets)
Ran the identical judge on the 3/3 tier (all of dclm+nemo+fwedu kept it, hq alone dropped):

| category | 3/3 clear-useful | 0-vote clear-useful |
|---|---|---|
| reference | **~86%** | ~9% |
| other | ~82% | ~14% |
| health_fitness | ~77% | ~14% |
| blog_platform | ~77% | ~14% |
| hobby_sport | ~64% | ~5% |
| cooking | ~50% | ~41% |
| **overall** | **~73%** | **~16%** |

**The verifier score works — ~4.5× enrichment (73% vs 16%).** Docs all three independent filters
kept but hq dropped are knowledge-rich ~73% of the time, and they're dominated by **health / science /
reference explainers** (scleroderma, immunotherapy, ocean acidification, dinosaur taxonomy, waste
management) — exactly the knowledge-dense expository that maps to registers #2/#3 and the fact-recall
gap. So the **3/3 tier (~10,537 docs) is the ready-to-use dev-set positive set** (light cleaning only),
while the 0-vote residual is a haystack to mine selectively. Next: scale the judge on larger,
cleaner-categorized samples + fuller text; consider a 3/3-anchored first dev-set cut.

---

## Length analysis — the 3/3-dropped docs are a SPEC rejection, not a context-limit artifact
hq's extractor is a Qwen3 LLM with a 32k context; tested whether the 3/3-dropped docs were dropped
mechanically for exceeding it. Two probes (Qwen3 tokens):
- **Extracted-text length** (resiliparse) of the 3/3-dropped docs: median **1,481**, p90 8,298, p99 27,888;
  only **0.3% over 32k**. Compared to hq-kept (median 999): dropped docs are ~1.5–3× longer — a *soft
  length bias* below the cap, not a wall.
- **HTML:text token ratio** on a 24-WARC / 42.5k-doc sample: median **1.36×**, p90 2.46× — the raw HTML
  barely inflates over content. Applying it: **est. ~2.4% of the 3/3-dropped docs have HTML >32k**.

**Verdict: ~97.6% were dropped by the "useful content" SPEC (they fit the context); only a ~2–3% long
tail is genuinely context-limited.** So the primary fix is the **keep-decision spec** (it rejects good
normal-length pages); long-doc/chunking handling is a small ~2–3% bonus. There is also a soft length
bias to watch (hq under-keeps longer docs even well below 32k). Data: `length_probe/`, `html_ratio/`.

## What the ablation rules out (see `scratch/ablation_report/`)
The Phase-2 causal runs tested whether hq's gap is fixable by (a) reweighting toward knowledge density
or (b) matching DCLM's token count via epoching. Both were **refuted** → the lever is per-document
CONTENT QUALITY (what hq keeps/extracts), which is exactly what the dev-set work targets. Full numbers
+ plots in the separate ablation report.

---

## Open questions / next
- Fill example docs (from the residual sample + targeted pulls) — IN PROGRESS.
- Needle-rate of the practical-expertise seam in the 0-vote residual (LLM judge).
- Split each register's fix into coverage (keep-decision) vs quality (extraction-completeness).
- Decide dev-set sampling per register: content-typed domains → sample by domain; platform domains → cross-pipeline / judge.
