# Symmetric per-pipeline coverage by register (2026-07-09)

**What this measures.** For each register (domain-category), take the docs that ≥2 of the **5**
pipelines (hq, dclm, nemo, fwedu, fwcc) agree are good — a *symmetric* consensus counting all
pipelines equally (unlike `verifier_score`, which excludes hq because hq is the audit target).
Then report **what % of those agreed-good docs each pipeline included**. This is the fair,
hq-inclusive coverage comparison.

**How "register" is defined.** Per URL → registered domain (eTLD+1), then a top-down curated
`domain LIKE '%substring%'` classifier (`build_devset.py::_case_sql`, first-match-wins). It's a
cheap domain proxy — platform domains (blogspot/wordpress) bin by platform, not content — so read
the low-count / platform rows with care. Source: `build_devset.py coverage --threshold 2`
(membership + fwcc). Data: `gs://marin-us-central2/scratch/provenance_10k/devset/coverage.json`.

## Coverage of ≥2-of-5 agreed-good docs (% each pipeline included)

| register | n_good | **hq %** | dclm % | nemo % | fwedu % | fwcc % |
|---|---:|---:|---:|---:|---:|---:|
| **fiction** | 20,575 | **2.2** | 78.5 | 38.0 | 0.5 | 97.0 |
| **educational** | 18,641 | **42.5** | 40.0 | 29.2 | 50.4 | 84.9 |
| **arxiv** | 211 | **52.6** | 51.7 | 87.2 | 2.8 | 9.5 |
| **howto** | 1,126,099 | **55.5** | 39.1 | 30.8 | 9.8 | 85.9 |
| **reference_expository** | 157,313 | **57.3** | 45.7 | 19.3 | 35.4 | 87.7 |
| other | 14,795,109 | 60.2 | 23.0 | 35.3 | 14.0 | 91.2 |
| news | 156,431 | 64.3 | 33.4 | 22.8 | 10.9 | 91.6 |
| tabular | 2,585 | 65.5 | 16.2 | 20.6 | 40.2 | 89.4 |
| science | 64,700 | 65.1 | 20.7 | 30.6 | 51.2 | 82.3 |
| math | 6,706 | 71.5 | 68.7 | 24.9 | 15.5 | 37.2 |
| qa_forum | 179,855 | 71.8 | 41.8 | 31.3 | 8.0 | 70.4 |
| legal | 53,028 | 76.3 | 23.1 | 25.2 | 6.4 | 92.6 |
| code | 37,678 | **80.1** | 54.0 | 32.9 | 7.2 | 62.0 |

## Key takeaways

1. **hq's gaps are concentrated, not broad.** On the agreed-good docs hq is a *strong* includer
   (60–80%) in most registers. Its real holes, ranked: **fiction (2.2%, catastrophic) ≫
   educational (42.5%) > arxiv (52.6%) ≈ howto (55.5%) ≈ reference (57.3%)**. Elsewhere hq is fine.
2. **Quality-weighting corrects the raw-retention view.** Raw retention said "arxiv hq 8.5%", but
   among *agreed-good* arxiv docs hq includes 52.6% — hq was correctly dropping junk arxiv. Fiction
   stays catastrophic under both metrics. So dev-set priority by *real* gap = fiction ≫ educational
   > arxiv ≈ howto ≈ reference.
3. **The other pipelines' fingerprints:** **fwcc** (light filter) tops coverage nearly everywhere
   (85–97%) *except* math (37%) and arxiv (9.5%) — it drops symbol-heavy content. **fwedu** peaks on
   science (51%) / educational (50%) / tabular (40%) — its education bias. **nemo** dominates arxiv
   (87%); **dclm** is strongest on fiction (78%), math (69%), code (54%).

Caveats: ≥2-of-5 consensus partly includes the pipeline being measured (mild circularity); domain
classifier is a proxy (see above); arxiv n=211 is tiny. Companion to
[[devset_domain_priorities_WIP]].
