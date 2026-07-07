# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross the provenance retention table (Phase 0) with the per-domain winner map.

Tests the core mechanism behind the eval gaps: does high_quality UNDER-RETAIN
exactly in the domains where it LOSES on loss, and OVER-RETAIN where it wins?
If retention rank tracks eval rank, the gap is a filtering-coverage story and the
dev set should mine the winner's kept docs in that domain.

Inputs (small; downloaded from us-central2):
  gs://marin-us-central2/scratch/provenance_10k/summary/named_domain_retention.json
  gs://marin-us-central2/scratch/provenance_10k/summary/domain_retention.json
  scratch/winner_map.json  (from scratch/build_winner_map.py)

Usage:
  python experiments/baseline_collection/provenance_report.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

SUMMARY_GS = "gs://marin-us-central2/scratch/provenance_10k/summary"

# eval domain (winner_map key, sans "uncheatable:" prefix) → source registered
# domains that carry that content. Retention on these domains is the mechanism.
EVAL_TO_SOURCE: dict[str, list[str]] = {
    "ao3_english": ["archiveofourown.org", "fanfiction.net"],
    "bbc_news": ["bbc.co.uk", "bbc.com"],
    "github_python": ["github.com", "github.io"],
    "github_cpp": ["github.com", "github.io"],
    "wikipedia_english": ["wikipedia.org"],
    "arxiv_computer_science": ["arxiv.org"],
    "arxiv_physics": ["arxiv.org"],
}
METHOD_COL = {"HQ": "n_hq", "DCLM": "n_dclm", "NEMO": "n_nemo", "FW_EDU": "n_fwedu"}


def _download(name: str) -> dict | list:
    tmp = Path(tempfile.mkdtemp()) / name
    subprocess.run(["gcloud", "storage", "cp", f"{SUMMARY_GS}/{name}", str(tmp)], check=True, capture_output=True)
    return json.loads(tmp.read_text())


def _retention_for(domain_rows: dict[str, dict], sources: list[str]) -> tuple[dict[str, float], int]:
    """Aggregate retention % per method across the given source domains.

    domain_retention.json stores counts (n_hq...) as strings (duckdb bigint →
    JSON via default=str), so cast. Returns (pct_by_method, total_n_urls).
    """
    tot = {"n_urls": 0, "n_hq": 0, "n_dclm": 0, "n_nemo": 0, "n_fwedu": 0}
    for src in sources:
        row = domain_rows.get(src)
        if not row:
            continue
        for k in tot:
            tot[k] += int(row.get(k, 0))
    if tot["n_urls"] == 0:
        return {}, 0
    return {m: 100.0 * tot[col] / tot["n_urls"] for m, col in METHOD_COL.items()}, tot["n_urls"]


def main() -> None:
    # domain_retention.json holds per-domain COUNTS (n_hq, n_dclm, ...); the named
    # table only has percentages, so the count table is the source of truth here.
    domain_rows = {r["domain"]: r for r in _download("domain_retention.json")}
    winner_map = json.loads(Path("scratch/winner_map.json").read_text())
    scale_key = next(k for k in winner_map if k.startswith("8B"))
    wm = winner_map[scale_key]

    print(f"=== Retention (% of source-domain URLs kept) vs eval winner @ {scale_key} ===\n")
    print(f'{"eval domain":24} {"winner":7} {"HQ%":6} {"DCLM%":6} {"NEMO%":6} {"FWEDU%":7} {"n_urls":8} verdict')
    for eval_dom, sources in EVAL_TO_SOURCE.items():
        wkey = "uncheatable:" + eval_dom
        if wkey not in wm:
            continue
        winner = wm[wkey]["winner"]
        ret, n_urls = _retention_for(domain_rows, sources)
        if not ret:
            print(f"{eval_dom:24} {winner:7} (no source-domain rows: {sources})")
            continue
        hq = ret.get("HQ", float("nan"))
        # Extracted-method proxy for the winner (FW_CC/RESIL are not in the table).
        proxy = (
            winner if winner in METHOD_COL else max((m for m in METHOD_COL if m != "HQ"), key=lambda m: ret.get(m, -1))
        )
        proxy_ret = ret.get(proxy, float("nan"))
        if winner == "HQ":
            verdict = "HQ wins → PROTECT"
        elif proxy_ret > hq + 1:
            verdict = f"CONFIRMED (mine {proxy}{' proxy' if proxy != winner else ''})"
        else:
            verdict = "not-by-retention → Phase 0c (mangling)"
        print(
            f"{eval_dom:24} {winner:7} {hq:5.1f} {ret.get('DCLM', float('nan')):5.1f} "
            f"{ret.get('NEMO', float('nan')):5.1f} {ret.get('FW_EDU', float('nan')):6.1f} {n_urls:8} {verdict}"
        )
    print(
        "\nReading: 'CONFIRMED' = an extractable method retains more of the domain than "
        "HQ → filtering/coverage gap; mine that method's kept docs there. "
        "'not-by-retention' = HQ keeps as many URLs but still loses → extraction-quality "
        "(mangling) story (Phase 0c). FW_CC/RESIL are not extracted (FW_CC text deleted; "
        "RESIL is the expensive superset), so the news winner FW_CC is proxied by NEMO."
    )


if __name__ == "__main__":
    main()
