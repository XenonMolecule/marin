# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stacked olmixexact figure: uncheatable macro-bpb (top row) over olmix51 bpb (bottom row).

One column per model size, both rows sharing arm colors/labels and ONE legend. Reuses the
uncheatable plot's W&B cache and the BPB plot's strict (complete-task-set) loader.

    uv run python -m experiments.scaling_law_sweeps.plot_olmixexact_stacked \
        --methods ... --relabel "lpv11_10k_mix_olmixexact_lambda0p01=Generative Curation" \
        --dims 512 1536 2432 --suffix _5arm
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiments.scaling_law_sweeps.plot_core_v2_by_size_10k import OLMIXEXACT_COMPARE_METHODS
from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS
from experiments.scaling_law_sweeps.plot_olmixexact_bpb_by_size import (
    METHOD_DISPLAY,
    METHOD_LINEWIDTH,
    _legend_label,
    _tokens_label,
)
from experiments.scaling_law_sweeps.plot_olmixexact_uncheatable_by_size import CACHE, PARAM_LABEL
from experiments.scaling_law_sweeps.plot_olmo_bpb_by_size_10k import EXCLUDED_RUN_STEMS, _load_records

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "olmo_bpb_10k" / "paper"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[512, 1536, 2432])
    ap.add_argument("--relabel", nargs="+", default=[], metavar="METHOD=LABEL")
    ap.add_argument("--legend-size", type=float, default=20.0)
    ap.add_argument("--suffix", default="_stacked")
    args = ap.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.relabel)
    display = {m: (overrides.get(m, METHOD_DISPLAY[m][0]), METHOD_DISPLAY[m][1]) for m in args.methods}

    # Top row: uncheatable macro bpb from the cached W&B pull (run --refresh on the
    # single-row plot first if the cache is stale).
    unch = [
        r
        for r in json.load(open(CACHE))
        if r["method"] in display and r["dim"] in set(args.dims) and r["stem"] not in EXCLUDED_RUN_STEMS
    ]
    # Bottom row: olmix51 via the strict complete-task-set loader.
    bpb = [
        r
        for r in _load_records(OLMIXEXACT_COMPARE_METHODS)
        if "olmix51_bpb" in r.bpb and r.method in display and r.hidden_dim in set(args.dims)
    ]

    plt.rcParams.update(PAPER_RCPARAMS)
    dims = sorted(set(args.dims))
    fig, axes = plt.subplots(
        2, len(dims), figsize=(7.2 * len(dims), 12.0), squeeze=False, gridspec_kw={"hspace": 0.14, "wspace": 0.13}
    )

    for col, dim in enumerate(dims):
        ax_u, ax_b = axes[0][col], axes[1][col]
        for method, (label, color) in display.items():
            pts = sorted((r["tokens"], r["macro_bpb"]) for r in unch if r["method"] == method and r["dim"] == dim)
            if pts:
                x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                ax_u.plot(
                    x,
                    y,
                    "-",
                    color=color,
                    linewidth=METHOD_LINEWIDTH.get(method, 3.0),
                    alpha=0.95,
                    label=_legend_label(method, label),
                    zorder=3,
                )
                ax_u.scatter(x, y, color=color, s=70, edgecolors="white", linewidths=0.8, zorder=4)
            pts = sorted((r.tokens, r.bpb["olmix51_bpb"]) for r in bpb if r.method == method and r.hidden_dim == dim)
            if pts:
                x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                ax_b.plot(x, y, "-", color=color, linewidth=METHOD_LINEWIDTH.get(method, 3.0), alpha=0.95, zorder=3)
                ax_b.scatter(x, y, color=color, s=70, edgecolors="white", linewidths=0.8, zorder=4)
        # Clamp both rows to the bottom row's data range so the uncheatable row cannot
        # extend past budgets that have olmix evals yet.
        col_tokens = [r.tokens for r in bpb if r.hidden_dim == dim]
        for ax in (ax_u, ax_b):
            ax.set_xscale("log")
            if col_tokens:
                ax.set_xlim(min(col_tokens) * 0.8, max(col_tokens) * 1.25)
            lo, hi = ax.get_xlim()
            subs = (1.0,) if np.log10(hi / lo) > 1.6 else (1.0, 2.0, 5.0)
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=subs))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_label))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
        ax_u.set_title(PARAM_LABEL.get(dim, f"d{dim}"))
        ax_b.set_xlabel("Tokens trained")
    axes[0][0].set_ylabel("BPB (Uncheatable Eval)")
    axes[1][0].set_ylabel("BPB (olmix 51-task suite)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in labels:
                handles.append(h)
                labels.append(ll)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        bbox_to_anchor=(0.5, -0.025),
        fontsize=args.legend_size,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"olmixexact_stacked{args.suffix}"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    logger.info("wrote %s.{png,pdf}", stem)


if __name__ == "__main__":
    main()
