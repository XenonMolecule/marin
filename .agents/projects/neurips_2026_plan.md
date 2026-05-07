# NeurIPS 2026 Submission Plan

**Submission deadline:** 2026-05-06
**Total working window:** 20 days

## Summary

The author has 20 days to complete four remaining experiment tracks and a
co-author-reviewed paper draft. The plan concentrates all experiment risk into
the first 10 days and reserves the second 10 days for analysis, writing, and
polish. A hard cutoff on non-optional experimental work is set at **2026-04-26
(day 10)**. All experiments launch before the author's 2026-04-20 departure
for ICLR so that the 2026-04-20 through 2026-04-29 travel window requires
monitoring only, not new launches or heavy debugging.

## Experiment inventory

| Track | Est. duration | Status | Criticality |
|---|---|---|---|
| Extraction fleet | ~3 days | In progress (launched prior to 2026-04-16) | Required |
| Medical extraction rerun | 2 days | Not started | Required |
| Data curation scaling laws (IsoFLOP, ExpA + ExpB) | 5 days | Not started; set size TBD after meeting with Percy | Required (scope reduction expected) |
| Qwen3 8B + Marin 8B training | 5 days | Not started | Required |
| Rephraser experiments | 4 days | Not started | **Optional / stretch** |

The medical rerun, scaling laws, and 8B training all depend on extraction
output and therefore cannot start until extraction lands (~2026-04-19). Once
extraction is complete, these three tracks run in parallel across separate
TPU regions to avoid capacity contention.

## Scope reductions and open decisions

1. **Scaling-law candidate set.** The full plan grid contains 78 model
   candidates × 4 curation methods × 2 experiment variants (ExpA natural,
   ExpB simulated-epoching). A reduced core subset is expected out of the
   2026-04-16 meeting with Percy; the remainder moves to the appendix or is
   dropped.
2. **Rephraser track.** Treated as nice-to-have. Go/no-go is made on
   2026-04-29 upon return from ICLR, contingent on the non-optional tracks
   being on rails. The track trades directly against writing time and is
   unlikely to be launched if any required track slips.
3. **8B ship-at-step-N checkpoint.** The step count at which training is
   declared complete (or cut short) must be agreed in advance, before
   2026-04-19 launch, to remove decision pressure during travel.
4. **Medical rerun input source.** Must be confirmed whether the rerun
   consumes the currently-running extraction fleet's output or a prior dump.
   This determines whether medical can begin as early as 2026-04-17 (prior
   dump) or must wait until 2026-04-19 (fresh extraction).

## Day-by-day schedule

| Date | Day | Location | Experiment activity | Writing activity |
|---|---|---|---|---|
| Thu 2026-04-16 | 1 | Home | Monitor extraction fleet; meet with Percy to finalize scaling set | Outline; paper skeleton |
| Fri 2026-04-17 | 2 | Home | Extraction continues; launch medical rerun if input source allows | Related work |
| Sat 2026-04-18 | 3 | Home | Extraction finishing; medical running; stage scaling + 8B launch scripts | Methods draft |
| Sun 2026-04-19 | 4 | Home | Extraction complete; medical finishes; **launch scaling laws + 8B training**; pre-travel monitoring setup (logscan, babysit-job loops, dashboards) | Finalize methods; intro draft |
| Mon 2026-04-20 | 5 | Travel → ICLR | Scaling + 8B running autonomous; monitor 30 min | Light — related work polish |
| Tue 2026-04-21 | 6 | ICLR | Monitor; restart failed child jobs only | Method prose |
| Wed 2026-04-22 | 7 | ICLR | Monitor | Experimental setup section |
| Thu 2026-04-23 | 8 | ICLR | Monitor; early scaling fits begin to land | Draft results skeleton |
| Fri 2026-04-24 | 9 | ICLR | Scaling + 8B expected to complete; initial plots | Draft results from available data |
| Sat 2026-04-25 | 10 | ICLR | Buffer day for slipped runs; generate first full plot set | Results section |
| **Sun 2026-04-26** | **11** | **ICLR** | **HARD CUTOFF for non-optional experiments.** Kill anything unfinished at its latest checkpoint | Results + discussion |
| Mon 2026-04-27 | 12 | ICLR | Analysis only — scaling fits, 8B eval tables, medical metrics | Results + discussion |
| Tue 2026-04-28 | 13 | ICLR | Analysis; figure iteration | Discussion, limitations |
| Wed 2026-04-29 | 14 | Travel → Home | **Rephraser go/no-go decision.** If go, launch before end of day (finishes ~2026-05-03) | Tighten results |
| Thu 2026-04-30 | 15 | Home | Analysis sprint; final tables and figures | Full draft coalesces |
| Fri 2026-05-01 | 16 | Home | Final experimental numbers locked | **Target: complete first draft** |
| Sat 2026-05-02 | 17 | Home | Rephraser results folded in (if launched) | Co-author review cycle |
| Sun 2026-05-03 | 18 | Home | — | Revision; appendix |
| Mon 2026-05-04 | 19 | Home | — | Polish; second co-author pass |
| Tue 2026-05-05 | 20 | Home | — | Proofread; submission dry run; **submit by evening** |
| Wed 2026-05-06 | 21 | Home | — | **Buffer day.** Not a work day in the plan |

## Capacity plan (non-optional parallel block, 2026-04-19 through 2026-04-26)

To avoid v5p contention during the seven-day parallel block, tracks are
segregated by region:

| Track | TPU footprint | Target region |
|---|---|---|
| Scaling laws (reduced set) | v5p-8 through v5p-64, many small slices | us-central1 + us-east5-a (via `mirror://` data paths; region-agnostic via Iris) |
| Qwen3 8B + Marin 8B | Single large v5p slice | us-east1-d (training) |
| Medical rerun | SFT slice + eval | Whatever has free capacity; window is short (≤ 2 days) |

## Risk register

| Risk | Mitigation |
|---|---|
| 8B training runs past 5 days | "Ship at step N" checkpoint agreed pre-launch; hard-cut at 2026-04-26 |
| Scaling sweep failures during travel | Pre-wired logscan + babysit-job loops; reduced candidate set lowers surface area |
| Medical rerun blocked on extraction | Confirm input source 2026-04-16; if blocked, accept medical starts 2026-04-19 and compresses to 2 days during travel |
| Rephraser consumes writing time | Decision gated on 2026-04-29 state; default is to skip |
| Co-author review bottleneck | First draft locked 2026-05-01 to give reviewers four business days |
| Cross-region contention | Tracks pre-assigned to distinct regions; see capacity plan |

## Milestones

- **2026-04-19:** Extraction complete; medical complete; scaling + 8B launched.
- **2026-04-26:** Hard cutoff for required experimental work.
- **2026-05-01:** First complete draft.
- **2026-05-05:** Submission.
