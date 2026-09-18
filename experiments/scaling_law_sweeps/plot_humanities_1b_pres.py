# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Presentation figure: MMLU-humanities clp vs tokens for the 1B humanities-target arms.

Single panel, external-document styling (matches the cultural presentation figure):
plain-English title, large legend inside, tokens x-axis. Reads the mmlu_target_models
rollup; strict complete-category rule.

    uv run python -m experiments.scaling_law_sweeps.plot_humanities_1b_pres
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from experiments.scaling_law_sweeps.olmo_bpb.mmlu_olmes import MMLU_CATEGORIES
from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS

logger = logging.getLogger(__name__)
OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "cultural_target"
ROLLUP = "marin-us-central1/metadata/mmlu_target_models/rollup.csv"
DISPLAY = {
    "dclm_10k": ("DCLM-baseline", "#1f77b4", 3.0),
    "nemotron_full_10k": ("Nemotron-CC", "#ff7f0e", 3.0),
    "resiliparse_10k": ("Resiliparse", "#9467bd", 3.0),
    "fineweb_edu_10k": ("FineWeb-Edu", "#17becf", 3.0),
    "lpv11_fastpipe_v2_10k": ("Ours", "#006d2c", 4.2),
}
RX = re.compile(
    r"^curation-(?P<corpus>.+?)_mix_mmlu_humanities_clp_(?P<gen>r100k\d+)_lambda0p05"
    r"-expFM_natural-(?P<budget>[\d.e+]+)-d(?P<dim>\d+)-"
)
# fwedu competes as its natural mix (no targeted arm); seed variants excluded so one point per budget.
NAT_RX = re.compile(r"^curation-fineweb_edu_10k-expFM_natural-(?P<budget>[\d.e+]+)-d(?P<dim>\d+)-L\d+-B\d+$")
SIZES = {512: ("157M models", 157e6, "humanities_157m_pres"), 1536: ("1B models", 998e6, "humanities_1b_pres")}


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1536, choices=sorted(SIZES))
    args = ap.parse_args()
    size_label, params, out_name = SIZES[args.dim]
    fs = fsspec.filesystem("gcs", use_listings_cache=False)
    by_run: dict[str, dict[str, float]] = {}
    with fs.open(ROLLUP, "r") as fh:
        for row in csv.DictReader(fh):
            t = row["task"]
            if not t.startswith("mmlu_sl_verb_") or t == "mmlu_sl_verb_5shot":
                continue
            by_run.setdefault(row["run"], {})[t.removeprefix("mmlu_sl_verb_").removesuffix("_5shot")] = float(row["clp"])

    subs = MMLU_CATEGORIES["humanities"]
    pts: dict[str, list[tuple[float, float]]] = {}
    for run, tasks in by_run.items():
        m = RX.match(run)
        corpus = m.group("corpus") if m else None
        if not m:
            m = NAT_RX.match(run)
            corpus = "fineweb_edu_10k" if m else None
        if not m or int(m.group("dim")) != args.dim:
            continue
        vals = [tasks[s] for s in subs if s in tasks]
        if len(vals) != len(subs):
            logger.warning("partial humanities eval dropped: %s", run)
            continue
        tokens = float(m.group("budget")) / (6.0 * params)
        # best variant per (corpus, budget): keep the max across k5/k20
        pts.setdefault(corpus, []).append((tokens, sum(vals) / len(vals)))

    plt.rcParams.update(PAPER_RCPARAMS)
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    for corpus, (label, color, lw) in DISPLAY.items():
        if corpus not in pts:
            continue
        best: dict[float, float] = {}
        for x, y in pts[corpus]:
            best[x] = max(best.get(x, -1e9), y)
        xs, ys = zip(*sorted(best.items()))
        ax.plot(xs, ys, "-", marker="o", color=color, linewidth=lw, label=label)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x/1e9:g}B" if x >= 1e9 else f"{x/1e6:g}M"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Mean choice log-probability\n(higher is better)")
    ax.set_title(f"MMLU Humanities ({size_label})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=18, frameon=True)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    logger.info("wrote %s.{png,pdf}", out)


if __name__ == "__main__":
    main()
