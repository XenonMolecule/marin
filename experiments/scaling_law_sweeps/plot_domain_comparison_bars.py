# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bar-chart versions of the domain-comparison table (best MMLU choice log-prob per target).

Same cells as ``make_domain_comparison_table`` (verified rollup, best over budgets), one bar
per dataset in the line plots' colour scheme. Higher is better; bars grow upward from a floor
just below the panel's worst value so differences stay readable. Three layouts:

* ``grid``  — rows = model size, columns = target domain (``domain_comparison_bars``).
* ``row``   — one row of domain panels, 157M and 1B bars side by side per dataset
              (``domain_comparison_bars_row``).
* ``split`` — one single-row figure per model size (``domain_comparison_bars_157m`` / ``_1b``).

    uv run python -m experiments.scaling_law_sweeps.plot_domain_comparison_bars --layout grid row split
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.scaling_law_sweeps.make_domain_comparison_table import DIMS, ROWS, TARGET_CATEGORY, TARGETS, best_cells
from experiments.scaling_law_sweeps.plot_domain_v2_mmlu import CORPUS_DISPLAY, load_records
from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS, crop_pdf_to_ink

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "cultural_target"
TARGET_TITLE = {
    "mmlu_stem": "MMLU STEM",
    "mmlu_humanities": "MMLU Humanities",
    "mmlu_social_sciences": "MMLU Social Science",
}
DIM_TITLE = {512: "157M", 1536: "1B"}
MAIN_TEXT_DIM = 1536
YLABEL = "Choice log-prob (↑ better)"
# Short bar labels; "(Ours)" rows are drawn bold.
BAR_LABEL = {"GC LLM (Ours)": "GC LLM", "GC Cascade (Ours)": "GC Cascade"}
# Corpora with no domain-tuned OLMix solve, drawn from their natural (unmixed) runs and
# styled like every other bar; the caption carries the caveat.
NATURAL_ROWS = (("FineWeb-Edu", "fineweb_edu_10k"),)
# Bars are ordered by this cell, lowest to highest, in every panel.
SORT_KEY = ("mmlu_stem", 1536)
FLOOR_PAD = 0.15
# Paper figure: everything scaled up from PAPER_RCPARAMS so it stays legible at column width.
FIGURE_RCPARAMS = {"axes.labelsize": 24, "axes.titlesize": 32, "xtick.labelsize": 22, "ytick.labelsize": 24}
X_TICK_FONTSIZE = 21
X_TICK_ROTATION = 40
VALUE_FONTSIZE = 16
PAIR_VALUE_FONTSIZE = 12
LEGEND_FONTSIZE = 22
PANEL_W, PANEL_H = 7.4, 5.8
# Paired layout: the smaller size is drawn lighter so the two bars per dataset read as a pair.
PAIR_ALPHA = {512: 0.45, 1536: 1.0}
PAIR_OFFSET = 0.2
PAIR_WIDTH = 0.38


# (label, corpus, ours, hatched) — tuned rows plus the natural-only references, sorted by SORT_KEY.
Row = tuple[str, str, bool, bool]


def natural_cells() -> dict[tuple[str, str, int], float]:
    """{(corpus, target, dim): best category clp} for NATURAL_ROWS, from their unmixed runs."""
    wanted = {c for _, c in NATURAL_ROWS}
    cells: dict[tuple[str, str, int], float] = {}
    for r in load_records():
        if r.corpus not in wanted or r.hidden_dim not in DIMS:
            continue
        for target, cat in TARGET_CATEGORY.items():
            val = dict(r.clp_by_cat).get(cat)
            if val is None:
                continue
            key = (r.corpus, target, r.hidden_dim)
            if key not in cells or val > cells[key]:
                cells[key] = val
    return cells


def sorted_rows(cells: dict) -> list[Row]:
    rows: list[Row] = [(lbl, c, ours, False) for lbl, c, ours in ROWS] + [
        (lbl, c, False, True) for lbl, c in NATURAL_ROWS
    ]
    target, dim = SORT_KEY
    return sorted(rows, key=lambda r: cells.get((r[1], target, dim), float("-inf")))


def _value_range(cells: dict, rows: list[Row], dims: tuple[int, ...]) -> tuple[float, float]:
    vals = [cells[(c, t, d)] for d in dims for t in TARGETS for _, c, _, _ in rows if (c, t, d) in cells]
    return min(vals) - FLOOR_PAD, max(vals) + FLOOR_PAD


def _style_x(ax, rows: list[Row], label_size: float = X_TICK_FONTSIZE) -> None:
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [BAR_LABEL.get(lbl, lbl) for lbl, _, _, _ in rows], rotation=X_TICK_ROTATION, ha="right", fontsize=label_size
    )
    for tick, (_, _, ours, _) in zip(ax.get_xticklabels(), rows, strict=True):
        if ours:
            tick.set_fontweight("bold")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)


def _bar(ax, x, val, floor, color, width, alpha, ours: bool, labels: bool, hatched: bool = False):
    ax.bar(
        x,
        val - floor,
        bottom=floor,
        color=color,
        width=width,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.8,
    )
    if labels:
        ax.text(
            x,
            val + 0.015,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_FONTSIZE if width > 0.5 else PAIR_VALUE_FONTSIZE,
            fontweight="bold" if ours else "normal",
            color="#222222",
        )


def draw_panel(ax, cells: dict, rows: list[Row], target: str, dims: tuple[int, ...], floor, top, labels: bool) -> None:
    paired = len(dims) > 1
    for k, (_, corpus, ours, hatched) in enumerate(rows):
        color = CORPUS_DISPLAY[corpus][1]
        for dim in dims:
            val = cells.get((corpus, target, dim))
            if val is None:
                continue
            if paired:
                x = k + (PAIR_OFFSET if dim == max(dims) else -PAIR_OFFSET)
                _bar(ax, x, val, floor, color, PAIR_WIDTH, PAIR_ALPHA[dim], ours, labels, hatched)
            else:
                _bar(ax, k, val, floor, color, 0.72, 1.0, ours, labels, hatched)
    ax.set_ylim(floor, top)
    ax.set_title(TARGET_TITLE[target])
    _style_x(ax, rows)


def render_grid(cells: dict, rows: list[Row], labels: bool) -> Path:
    fig, axes = plt.subplots(
        len(DIMS), len(TARGETS), figsize=(PANEL_W * len(TARGETS), PANEL_H * len(DIMS)), squeeze=False
    )
    for i, dim in enumerate(DIMS):
        floor, top = _value_range(cells, rows, (dim,))
        for j, target in enumerate(TARGETS):
            draw_panel(axes[i][j], cells, rows, target, (dim,), floor, top, labels)
            if i > 0:
                axes[i][j].set_title("")
        axes[i][0].set_ylabel(f"{DIM_TITLE[dim]} models\n{YLABEL}")
    fig.tight_layout()
    return _save(fig, "domain_comparison_bars")


def render_row(cells: dict, rows: list[Row], labels: bool) -> Path:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=((PANEL_W + 0.6) * len(TARGETS), PANEL_H), squeeze=False)
    floor, top = _value_range(cells, rows, DIMS)
    for j, target in enumerate(TARGETS):
        draw_panel(axes[0][j], cells, rows, target, DIMS, floor, top, labels)
    axes[0][0].set_ylabel(YLABEL)
    handles = [Patch(facecolor="#555555", alpha=PAIR_ALPHA[d], label=f"{DIM_TITLE[d]} models") for d in DIMS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(DIMS),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    return _save(fig, "domain_comparison_bars_row")


def render_split(cells: dict, rows: list[Row], labels: bool) -> list[Path]:
    out = []
    for dim in DIMS:
        fig, axes = plt.subplots(1, len(TARGETS), figsize=(PANEL_W * len(TARGETS), PANEL_H), squeeze=False)
        floor, top = _value_range(cells, rows, (dim,))
        for j, target in enumerate(TARGETS):
            draw_panel(axes[0][j], cells, rows, target, (dim,), floor, top, labels)
        # The 1B figure is the main-text one (size stated in the caption); the 157M one goes to
        # the appendix and keeps its size tag so it can't be mistaken for the headline result.
        axes[0][0].set_ylabel(YLABEL if dim == MAIN_TEXT_DIM else f"{DIM_TITLE[dim]} models\n{YLABEL}")
        fig.tight_layout()
        out.append(_save(fig, f"domain_comparison_bars_{DIM_TITLE[dim].lower()}"))
    return out


def _save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / name
    # The axes' tight bbox omits a rotated y-label taller than the plot area.
    extra = [ax.yaxis.label for ax in fig.axes] + fig.legends
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.2, bbox_extra_artists=extra)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, bbox_extra_artists=extra)
    crop_pdf_to_ink(stem.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("wrote %s.{png,pdf}", stem)
    return stem


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", nargs="+", choices=["grid", "row", "split"], default=["grid", "row", "split"])
    ap.add_argument("--no-values", action="store_true", help="Omit the printed value above each bar.")
    args = ap.parse_args()
    cells = {**best_cells(), **natural_cells()}
    rows = sorted_rows(cells)
    logger.info("bar order (by %s, low→high): %s", SORT_KEY, [r[0] for r in rows])
    plt.rcParams.update(PAPER_RCPARAMS)
    plt.rcParams.update(FIGURE_RCPARAMS)
    labels = not args.no_values
    if "grid" in args.layout:
        render_grid(cells, rows, labels)
    if "row" in args.layout:
        render_row(cells, rows, labels)
    if "split" in args.layout:
        render_split(cells, rows, labels)


if __name__ == "__main__":
    main()
