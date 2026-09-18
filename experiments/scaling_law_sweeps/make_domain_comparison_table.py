# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate the domain-comparison LaTeX table (tab:domain-comparison) from the rollup.

For each corpus arm and MMLU target domain, reports the best (max) target-category
choice log-probability across training budgets and mixture generations (k5/k20) at the
157M (d512) and 1B (d1536) scales. Reads the same verified-only rollup as
plot_domain_v2_mmlu — consolidate_model_evals only ingests runs that pass
verify_eval_integrity, so every number here is final-checkpoint + complete-eval.

Writes ``scratch/tables/domain_comparison.md`` (snapshot header + LaTeX block) and
prints the LaTeX. Bold = best per column, underline = second best, red em-dash = no
verified datapoint yet. Rows with no data at all are emitted commented out.

    uv run python -m experiments.scaling_law_sweeps.make_domain_comparison_table
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

from experiments.scaling_law_sweeps.plot_domain_v2_mmlu import load_records

logger = logging.getLogger(__name__)

TARGETS = ("mmlu_stem", "mmlu_humanities", "mmlu_social_sciences")
TARGET_CATEGORY = {"mmlu_stem": "stem", "mmlu_humanities": "humanities", "mmlu_social_sciences": "social_sciences"}
DIMS = (512, 1536)
# (display label, corpus key, ours). "ours" rows are bold-labelled; rows compete for
# bold/underline only when they hold at least one value.
ROWS = (
    ("Resiliparse", "resiliparse_10k", False),
    ("DCLM", "dclm_10k", False),
    ("Nemotron", "nemotron_full_10k", False),
    ("FineWeb CC", "fineweb_cc_10k", False),
    ("Generative Curation (Ours)", "lpv11_10k", True),
    ("GC Cascade (Ours)", "lpv11_fastpipe_v2_10k", True),
)
DASH = r"\textcolor{red}{\textemdash}"
OUT_MD = Path(__file__).parent.parent.parent / "scratch" / "tables" / "domain_comparison.md"

HEADER = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{8pt}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | cc | cc | cc}
\toprule
\rowcolor[gray]{0.9}
& \multicolumn{2}{c|}{\textbf{STEM}}
& \multicolumn{2}{c|}{\textbf{Humanities}}
& \multicolumn{2}{c}{\textbf{Social Science}} \\
\rowcolor[gray]{0.9}
\textbf{Dataset}
& \textbf{157M} & \textbf{1B}
& \textbf{157M} & \textbf{1B}
& \textbf{157M} & \textbf{1B} \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}%
}
\caption{Highest MMLU choice log-probability per target domain at the 157M and 1B scales
(higher is better; \textbf{bold} best, \underline{underline} second best). Our generative
curation methods take the top two spots in every domain at both scales.}
\label{tab:domain-comparison}
\vspace{-12pt}
\end{table*}"""


def best_cells() -> dict[tuple[str, str, int], float]:
    """{(corpus, target, dim): best target-category clp} over budgets and generations."""
    cells: dict[tuple[str, str, int], float] = {}
    for r in load_records():
        cat = TARGET_CATEGORY.get(r.target)
        if cat is None or r.hidden_dim not in DIMS:
            continue
        val = dict(r.clp_by_cat).get(cat)
        if val is None:
            continue
        key = (r.corpus, r.target, r.hidden_dim)
        if key not in cells or val > cells[key]:
            cells[key] = val
    return cells


def render(cells: dict[tuple[str, str, int], float]) -> str:
    columns = [(t, d) for t in TARGETS for d in DIMS]
    # rank each column over rows that have a value
    marks: dict[tuple[str, str, int], str] = {}
    for t, d in columns:
        ranked = sorted(((cells[(c, t, d)], c) for _, c, _ in ROWS if (c, t, d) in cells), reverse=True)
        if ranked:
            marks[(ranked[0][1], t, d)] = "bold"
        if len(ranked) > 1:
            marks[(ranked[1][1], t, d)] = "underline"

    lines = [HEADER]
    prev_ours = False
    for label, corpus, ours in ROWS:
        if ours and not prev_ours:
            lines.append(r"\midrule")
        prev_ours = ours
        parts = []
        for t, d in columns:
            v = cells.get((corpus, t, d))
            if v is None:
                parts.append(DASH)
            else:
                s = f"{v:.3f}"
                mark = marks.get((corpus, t, d))
                if mark == "bold":
                    s = rf"\textbf{{{s}}}"
                elif mark == "underline":
                    s = rf"\underline{{{s}}}"
                parts.append(s)
        name = rf"\textbf{{{label}}}" if ours else label
        row = f"{name} & " + " & ".join(parts) + r" \\"
        if all(p == DASH for p in parts):
            row = "% " + row  # no verified data yet — keep the row visible but inert
        lines.append(row)
    lines.append(FOOTER)
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    cells = best_cells()
    latex = render(cells)

    # The caption claims our methods hold the top two spots everywhere — flag loudly
    # when the verified data does not support it so the caption gets edited, not shipped.
    ours = {c for _, c, o in ROWS if o}
    for t in TARGETS:
        for d in DIMS:
            ranked = sorted(((cells[(c, t, d)], c) for _, c, _ in ROWS if (c, t, d) in cells), reverse=True)
            top2 = {c for _, c in ranked[:2]}
            if ranked and top2 - ours:
                logger.warning("CAPTION CLAIM FALSE for %s d%d: top-2 = %s", t, d, [c for _, c in ranked[:2]])

    snapshot = datetime.datetime.now().strftime("%Y-%m-%d %H:%M PT")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(
        "# Domain-target comparison table (highest choice_logprob per target category)\n\n"
        f"Generated by `experiments/scaling_law_sweeps/make_domain_comparison_table.py`, "
        f"snapshot {snapshot}.\nNumbers come from the verified-only rollup "
        "(consolidate_model_evals refuses runs that fail verify_eval_integrity).\n"
        "Red dashes = no verified datapoint yet; they fill as training/eval waves land.\n\n"
        "## LaTeX (copy from here — requires xcolor)\n\n```latex\n" + latex + "\n```\n"
    )
    logger.info("wrote %s", OUT_MD)
    print(latex)


if __name__ == "__main__":
    main()
