# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase C report: the domains/website-types hq is missing (feeds the real dev set).

Reads the aggregate outputs (domains.parquet, heldout_missing_docs.parquet) and produces:
  - a ranked table of domains carrying hq-worse-eval content that dclm/nemo keep but hq
    drops, split into COVERAGE gap (hq never had the url) vs QUALITY gap (hq had it but
    its extraction didn't carry the content — the ablation's content-quality story);
  - a rollup by website-TYPE (blog / wiki / qa / news / science / fiction / …) via the
    shared URL categorizer, so we see which registers hq systematically misses;
  - a figure.

Run locally after the aggregate completes (reads the small aggregate outputs from GCS).
"""

from __future__ import annotations

import argparse
import collections
import subprocess
import sys

import pyarrow.parquet as pq

from experiments.baseline_collection.provenance_audit_10k import categorize

AGG = "gs://marin-us-central2/scratch/provenance_10k/keyword_agg"
LOCAL = "/tmp/keyword_agg"


def _pull() -> None:
    subprocess.run(["mkdir", "-p", LOCAL], check=False)
    for f in ("domains.parquet", "heldout_missing_docs.parquet"):
        subprocess.run(["gcloud", "storage", "cp", f"{AGG}/{f}", f"{LOCAL}/{f}"], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()
    _pull()

    dom = pq.read_table(f"{LOCAL}/domains.parquet").to_pylist()
    # heldout can be large (raw-pool universe) — read only the columns the report needs.
    heldout = pq.read_table(f"{LOCAL}/heldout_missing_docs.parquet", columns=["verifier_score", "subjects"]).to_pylist()
    tot_cov = sum(r["n_coverage_gap"] for r in dom)
    tot_qual = sum(r["n_quality_gap"] for r in dom)
    tot_all3 = sum(r.get("n_all3_agree", 0) for r in dom)
    tot_front = sum(r.get("n_frontier", 0) for r in dom)
    print(f"=== {len(dom)} domains carry hq-worse-eval content that the pool has but hq drops ===")
    print(f"    coverage-gap={tot_cov:,} (hq never had it)  quality-gap={tot_qual:,} (hq extracted it worse)")
    print(
        f"    3/3 verifiers agree (dclm+nemo+fwedu kept, hq alone wrong)={tot_all3:,}  |  "
        f"0/3 FRONTIER (in raw pool, hq dropped, NO filter kept — the potential edge)={tot_front:,}\n"
    )

    # Verifier-tier distribution across all missing docs (independent quality votes, hq excluded).
    tier = collections.Counter(r.get("verifier_score", 0) for r in heldout)
    print("--- MISSING DOCS BY # INDEPENDENT VERIFIERS (dclm+nemo+fwedu) THAT KEPT IT ---")
    for v in (3, 2, 1, 0):
        tag = {3: "hq alone wrong", 2: "strong", 1: "weak", 0: "FRONTIER — no filter kept it"}[v]
        print(f"  {v}/3 verifiers: {tier.get(v, 0):>8,}   {tag}")
    print()

    print(f"--- TOP {args.top} DOMAINS by missing eval-content docs ---")
    print(f'{"domain":30} {"cover":>7} {"qual":>6} {"avgV":>5} {"all3":>5} {"front":>6} {"type":>12}')
    ranked = sorted(dom, key=lambda r: -(r["n_coverage_gap"] + r["n_quality_gap"]))
    for r in ranked[: args.top]:
        typ = categorize(f"http://{r['domain']}/")
        print(
            f'{r["domain"][:30]:30} {r["n_coverage_gap"]:>7} {r["n_quality_gap"]:>6} '
            f'{r.get("avg_verifiers", 0)!s:>5} {r.get("n_all3_agree", 0):>5} {r.get("n_frontier", 0):>6} {typ:>12}'
        )

    # Domains ranked by the 0/3 FRONTIER — eval-content in the raw pool that EVERY filter dropped.
    print("\n--- TOP DOMAINS by 0/3 FRONTIER (raw-pool eval-content no quality filter kept) ---")
    for r in sorted(dom, key=lambda r: -r.get("n_frontier", 0))[:25]:
        typ = categorize(f"http://{r['domain']}/")
        print(f'  {r["domain"][:32]:32} frontier={r.get("n_frontier", 0):>6}  ({typ})')

    # Rollup by website type.
    by_type = collections.Counter()
    by_type_qual = collections.Counter()
    for r in dom:
        t = categorize(f"http://{r['domain']}/")
        by_type[t] += r["n_coverage_gap"] + r["n_quality_gap"]
        by_type_qual[t] += r["n_quality_gap"]
    print("\n--- MISSING CONTENT BY WEBSITE TYPE (register hq systematically drops) ---")
    print(f'{"type":16} {"missing_docs":>13} {"of which quality-gap":>22}')
    for t, n in by_type.most_common():
        print(f"{t:16} {n:>13,} {by_type_qual[t]:>22,}")

    # Which eval subjects the missing docs cover (top).
    subj = collections.Counter()
    for r in heldout:
        for s in (r.get("subjects") or "").split("|"):
            if s:
                subj[s] += 1
    print("\n--- TOP EVAL SUBJECTS carried by the missing docs (what to add) ---")
    for s, n in subj.most_common(30):
        print(f"  {n:>6}  {s}")

    # Figure.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = ranked[:25]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 8))
        names = [r["domain"][:28] for r in top][::-1]
        cov = [r["n_coverage_gap"] for r in top][::-1]
        qual = [r["n_quality_gap"] for r in top][::-1]
        a1.barh(names, cov, color="#e05a5a", label="coverage gap (hq missing url)")
        a1.barh(names, qual, left=cov, color="#f0b429", label="quality gap (hq extracted worse)")
        a1.set_title("Top domains: eval-content hq is missing")
        a1.legend(fontsize=8)
        a1.set_xlabel("missing docs")
        types = by_type.most_common(12)
        a2.barh([t for t, _ in types][::-1], [n for _, n in types][::-1], color="#7aa2f7")
        a2.set_title("Missing eval-content by website type")
        a2.set_xlabel("missing docs")
        fig.suptitle(
            "Domains/website-types hq is missing (carry the eval content hq loses on, kept by dclm/nemo)", fontsize=13
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig("scratch/missing_domains.png", dpi=120, bbox_inches="tight")
        print("\nwrote scratch/missing_domains.png")
    except ImportError:
        print("\n(matplotlib unavailable — text report only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
