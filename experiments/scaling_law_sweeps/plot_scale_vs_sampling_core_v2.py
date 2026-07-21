# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale-vs-sampling decomposition of DCLM Core_v2.

Puts three training regimes of the SAME curation methods on one canvas so the
two effects that separate a 3000-WARC sweep from the 10k-WARC sweep can be read
directly off the page:

  * RANDOM 3k  -- an independent uniform 3000-WARC draw.
  * BIASED 3k  -- the canonical head-of-pool 3000-WARC anchor.
  * 10k        -- the full 10,364-WARC pool.

At a fixed (method, width, compute budget) the vertical gaps decompose cleanly:

    sampling effect = BIASED 3k  - RANDOM 3k     (same corpus size, different draw)
    scale effect    = 10k        - BIASED 3k     (same draw style, more data)

This is the Core_v2 companion to ``plot_biased_vs_random_3k.py`` (which plots
loss and has no 10k arm) and reuses the method palette from
``plot_core_v2_by_size_10k.py`` -- the script that produced the shared 10k
figure. It reads the small local score caches under
``experiments/scaling_law_sweeps/dclm_core/analysis_3k_core_v2/`` (no GCS), so
the x-axis is the compute budget (FLOPs) parsed from each run's budget tag --
the axis that matches compute exactly across regimes, which is what makes the
gaps a fair read.

Two views (``--view``):

  * ``grid``    -- rows = method (DCLM / Nemotron / Resiliparse), cols = width;
                   each panel overlays the three regimes (color = regime). The
                   clean analytical cut: read the two gaps per (method, width).
  * ``overlay`` -- faceted by width like the shared 10k figure, color = method,
                   line style = regime (10k solid, biased-3k dash, random-3k
                   dot). Restricted to the three decomposable methods so it
                   stays legible while matching the shared aesthetic.

Usage::

    .venv/bin/python -m experiments.scaling_law_sweeps.plot_scale_vs_sampling_core_v2
    # -> scratch/plots/scale_vs_sampling/{grid,overlay}_core_v2.html
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from experiments.scaling_law_sweeps.plot_core_v2_by_size_10k import METHOD_COLORS

logger = logging.getLogger(__name__)

_ANALYSIS_DIR = Path(__file__).parent / "dclm_core" / "analysis_3k_core_v2"
DEFAULT_3K_CACHE = _ANALYSIS_DIR / "all_scores_3k.json"
DEFAULT_10K_CACHE = _ANALYSIS_DIR / "all_scores_10k.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "scale_vs_sampling"

# Human labels per width (exact trainable-param counts from the sweep configs).
PARAM_LABEL: dict[int, str] = {
    512: "157M",
    1024: "447M",
    1536: "998M",
    2432: "2.90B",
    3328: "6.70B",
    3584: "8.11B",
}

# Raw run-method name (10k suffix already stripped) -> canonical method key.
CANONICAL_METHOD: dict[str, str] = {
    # already-canonical (10k + biased-3k share these two names)
    "dclm": "dclm",
    "nemotron": "nemotron",
    "resiliparse": "resiliparse",
    "high_quality": "high_quality",
    "fineweb_edu": "fineweb_edu",
    "fineweb_cc": "fineweb_cc",
    # biased-3k run names
    "nemotron_full_bos_fixed": "nemotron",
    "resiliparse_dedup": "resiliparse",
    "high_quality_3000": "high_quality",
    "med_quality_3000": "med_quality",
    # random-3k run names
    "dclm_random_3000": "dclm",
    "nemotron_full_random_3000": "nemotron",
    "resiliparse_random_dedup_3000": "resiliparse",
}

# The three methods present in ALL three regimes -> the decomposable set.
DECOMPOSABLE_METHODS: tuple[str, ...] = ("dclm", "nemotron", "resiliparse")
# Widths where the 3k arms exist (10k also covers 1024/3584, dropped in `grid`).
COMPARE_WIDTHS: tuple[int, ...] = (512, 1536, 2432)

# Regime -> (display label, color, dash, plot order). In the grid view the row
# already encodes method, so color is free to encode the regime.
REGIME_STYLE: dict[str, tuple[str, str, str, int]] = {
    "random-3k": ("random 3k", "#d62728", "dash", 0),  # red
    "biased-3k": ("biased 3k", "#1f77b4", "solid", 1),  # blue
    "10k": ("10k", "#2ca02c", "solid", 2),  # green
}


@dataclass(frozen=True)
class Point:
    """One trained run's Core_v2 at a (method, width, compute) in one regime."""

    method: str  # canonical
    width: int
    flops: float
    core_v2: float  # 0-1 fraction (caches store x100; divided on load)
    regime: str


# Methods that legitimately have no place in the 3-regime decomposition (extra
# 10k-only baselines and the fastpipe overlay); skipped silently, not warned.
_KNOWN_SKIP = {"fineweb_cc", "med_quality"}


def _canon(raw_method: str) -> str | None:
    key = CANONICAL_METHOD.get(raw_method)
    if key is None and not (raw_method in _KNOWN_SKIP or raw_method.startswith("fastpipe")):
        logger.warning("Unmapped method name: %s", raw_method)
    return key


def _load_points(cache_3k: Path, cache_10k: Path) -> list[Point]:
    """Read both local caches into regime-tagged points (Core_v2 as a fraction).

    Regime is inferred from provenance, never from the shared method name:
    10k rows come from the 10k cache; 3k rows are random iff the run name carries
    the ``_random_`` marker, else biased.
    """
    points: list[Point] = []

    rows_10k = json.loads(cache_10k.read_text())
    for r in rows_10k:
        raw = r["method"].removesuffix("_10k")
        method = _canon(raw)
        if method is None:
            continue
        points.append(
            Point(method, int(r["width"]), float(r["budget"]), float(r["core_v2"]) / 100.0, "10k")
        )

    rows_3k = json.loads(cache_3k.read_text())
    for r in rows_3k:
        raw = r["method"]
        method = _canon(raw)
        if method is None:
            continue
        regime = "random-3k" if "_random_" in raw else "biased-3k"
        points.append(
            Point(method, int(r["width"]), float(r["budget"]), float(r["core_v2"]) / 100.0, regime)
        )

    logger.info(
        "Loaded %d points (%s)",
        len(points),
        ", ".join(f"{k}={sum(1 for p in points if p.regime == k)}" for k in REGIME_STYLE),
    )
    return points


def _panel_label(width: int) -> str:
    return f"d{width} ({PARAM_LABEL.get(width, '?')})"


def _shared_y_range(points: list[Point]) -> list[float]:
    ys = [p.core_v2 for p in points]
    lo, hi = min(ys), max(ys)
    pad = 0.04 * (hi - lo) if hi > lo else 0.01
    return [lo - pad, hi + pad]


def _build_grid(points: list[Point], output_dir: Path) -> None:
    """rows = method, cols = width; each panel overlays the three regimes.

    The direct scale-vs-sampling read: within a panel the green(10k)-blue(biased)
    gap is the scale effect and the blue(biased)-red(random) gap is the sampling
    effect, at matched compute.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    methods = [m for m in DECOMPOSABLE_METHODS]
    widths = list(COMPARE_WIDTHS)
    grid_pts = [p for p in points if p.method in methods and p.width in widths]
    y_range = _shared_y_range(grid_pts)

    # (method, width, regime) -> sorted [(flops, core_v2)]
    cells: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
    for p in grid_pts:
        cells[(p.method, p.width, p.regime)].append((p.flops, p.core_v2))
    for v in cells.values():
        v.sort()

    titles = [f"{m.upper()} | {_panel_label(w)}" for m in methods for w in widths]
    fig = make_subplots(
        rows=len(methods),
        cols=len(widths),
        subplot_titles=titles,
        horizontal_spacing=0.055,
        vertical_spacing=0.09,
    )

    legend_seen: set[str] = set()
    for ri, method in enumerate(methods):
        for ci, width in enumerate(widths):
            for regime in sorted(REGIME_STYLE, key=lambda k: REGIME_STYLE[k][3]):
                pts = cells.get((method, width, regime), [])
                if not pts:
                    continue
                label, color, dash, _ = REGIME_STYLE[regime]
                show = label not in legend_seen
                legend_seen.add(label)
                fig.add_trace(
                    go.Scatter(
                        x=[f for f, _ in pts],
                        y=[c for _, c in pts],
                        mode="lines+markers",
                        name=label,
                        legendgroup=label,
                        showlegend=show,
                        marker=dict(size=6, color=color),
                        line=dict(color=color, width=2, dash=dash),
                        hovertemplate=(
                            f"{label} | {method} d{width}"
                            "<br>flops=%{x:.2e}<br>Core_v2=%{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=ri + 1,
                    col=ci + 1,
                )
            fig.update_xaxes(type="log", title_text="compute (FLOPs)" if ri == len(methods) - 1 else None,
                             row=ri + 1, col=ci + 1)
            fig.update_yaxes(range=y_range, title_text="DCLM Core_v2" if ci == 0 else None,
                             row=ri + 1, col=ci + 1)

    fig.update_layout(
        template="plotly_white",
        title=(
            "DCLM Core_v2: scale vs sampling (rows=method, cols=width)"
            "<br><sub>at matched compute: green-blue gap = SCALE (3k->10k), "
            "blue-red gap = SAMPLING (random->biased 3k)</sub>"
        ),
        width=340 * len(widths) + 140,
        height=300 * len(methods) + 120,
        margin=dict(l=70, r=40, t=100, b=60),
        legend=dict(title=dict(text="regime"), orientation="v", x=1.01, y=0.5),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "grid_core_v2.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    logger.info("Wrote %s", out)


def _build_overlay(points: list[Point], output_dir: Path) -> None:
    """Shared-figure aesthetic: facet by width, color = method, dash = regime.

    Restricted to the three decomposable methods so up to 3 regimes x 3 methods
    stays legible. Solid = 10k, dash = biased-3k, dot = random-3k.
    """
    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    methods = [m for m in DECOMPOSABLE_METHODS]
    over_pts = [p for p in points if p.method in methods]
    y_range = _shared_y_range(over_pts)

    regime_dash = {"10k": "solid", "biased-3k": "dash", "random-3k": "dot"}
    widths = sorted({p.width for p in over_pts})
    n_cols = 3
    n_rows = math.ceil(len(widths) / n_cols)

    # (width, method, regime) -> sorted [(flops, core_v2)]
    cells: dict[tuple[int, str, str], list[tuple[float, float]]] = defaultdict(list)
    for p in over_pts:
        cells[(p.width, p.method, p.regime)].append((p.flops, p.core_v2))
    for v in cells.values():
        v.sort()

    titles = [_panel_label(w) for w in widths] + [""] * (n_rows * n_cols - len(widths))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.12)

    method_seen: set[str] = set()
    regime_seen: set[str] = set()
    for idx, width in enumerate(widths):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        for method in methods:
            color = METHOD_COLORS.get(method, "#333333")
            for regime, dash in regime_dash.items():
                pts = cells.get((width, method, regime), [])
                if not pts:
                    continue
                # One legend entry per method (color) and one per regime (a neutral
                # dash swatch) -- kept separate so the two encodings read cleanly.
                show = method not in method_seen
                method_seen.add(method)
                fig.add_trace(
                    go.Scatter(
                        x=[f for f, _ in pts],
                        y=[c for _, c in pts],
                        mode="lines+markers",
                        name=method,
                        legendgroup=method,
                        showlegend=show,
                        marker=dict(size=5, color=color),
                        line=dict(color=color, width=2, dash=dash),
                        hovertemplate=(
                            f"{method} | {regime} | d{width}"
                            "<br>flops=%{x:.2e}<br>Core_v2=%{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )
        fig.update_xaxes(type="log", title_text="compute (FLOPs)", row=row, col=col)
        fig.update_yaxes(range=y_range, title_text="DCLM Core_v2" if col == 1 else None, row=row, col=col)

    # Neutral dash-style swatches so the regime<->dash mapping is documented.
    for regime, dash in regime_dash.items():
        if regime in regime_seen:
            continue
        regime_seen.add(regime)
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines", name=REGIME_STYLE[regime][0],
                       legendgroup="regime", legendgrouptitle=dict(text="line style = regime"),
                       line=dict(color="#666666", width=2, dash=dash), showlegend=True),
            row=1, col=1,
        )

    fig.update_layout(
        template="plotly_white",
        title=(
            "DCLM Core_v2 vs compute by model scale -- three regimes overlaid"
            "<br><sub>color = method, line style = regime (solid 10k, dash biased-3k, dot random-3k)</sub>"
        ),
        width=360 * n_cols + 120,
        height=360 * n_rows + 120,
        margin=dict(l=70, r=40, t=100, b=60),
        legend=dict(orientation="v", x=0.72, y=0.24),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "overlay_core_v2.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    logger.info("Wrote %s", out)


def _build_delta(points: list[Point], output_dir: Path) -> None:
    """Sampling effect in isolation: (biased-3k - random-3k) Core_v2.

    For every (method, width) computes the biased-minus-random gap at each
    compute budget both regimes share, then plots that delta vs compute -- one
    panel per width, one colored line per method, with a dashed zero reference.
    A curve hugging zero means the head-vs-uniform WARC draw did not matter at
    that scale; excursions above/below zero are where the draw helped/hurt.
    """
    import math

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # (method, width) -> {flops: {regime: core}}; keep only matched budgets.
    by_cell: dict[tuple[str, int], dict[float, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for p in points:
        if p.method in DECOMPOSABLE_METHODS and p.regime in ("biased-3k", "random-3k"):
            by_cell[(p.method, p.width)][p.flops][p.regime] = p.core_v2

    # (width) -> method -> sorted [(flops, delta)]. Delta is in raw Core_v2 units
    # (a fraction), matching the y-axis of the main per-regime figures so the two
    # read on the same scale -- no x100 rescaling.
    deltas: dict[int, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for (method, width), by_flops in by_cell.items():
        for flops, regimes in by_flops.items():
            if "biased-3k" in regimes and "random-3k" in regimes:
                deltas[width][method].append((flops, regimes["biased-3k"] - regimes["random-3k"]))
    for w in deltas:
        for method in deltas[w]:
            deltas[w][method].sort()

    widths = sorted(deltas)
    if not widths:
        logger.warning("No matched biased/random budgets; skipping delta plot.")
        return
    n_cols = min(3, len(widths))
    n_rows = math.ceil(len(widths) / n_cols)

    all_deltas = [d for w in deltas for m in deltas[w] for _, d in deltas[w][m]]
    bound = max(0.02, 1.15 * max(abs(min(all_deltas)), abs(max(all_deltas))))

    titles = [_panel_label(w) for w in widths] + [""] * (n_rows * n_cols - len(widths))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.14)

    seen: set[str] = set()
    for idx, width in enumerate(widths):
        row, col = idx // n_cols + 1, idx % n_cols + 1
        fig.add_hline(y=0.0, line=dict(color="#999999", width=1, dash="dash"), row=row, col=col)
        for method in DECOMPOSABLE_METHODS:
            pts = deltas[width].get(method, [])
            if not pts:
                continue
            show = method not in seen
            seen.add(method)
            fig.add_trace(
                go.Scatter(
                    x=[f for f, _ in pts],
                    y=[d for _, d in pts],
                    mode="lines+markers",
                    name=method,
                    legendgroup=method,
                    showlegend=show,
                    marker=dict(size=6, color=METHOD_COLORS.get(method, "#333333")),
                    line=dict(color=METHOD_COLORS.get(method, "#333333"), width=2),
                    hovertemplate=(
                        f"{method} | d{width}"
                        "<br>flops=%{x:.2e}<br>biased−random=%{y:+.4f}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
        fig.update_xaxes(type="log", title_text="compute (FLOPs)", row=row, col=col)
        fig.update_yaxes(range=[-bound, bound], tickformat="+.3f", zeroline=False,
                         title_text="Δ Core_v2 (biased − random)" if col == 1 else None,
                         row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        title=(
            "Sampling effect: BIASED 3k − RANDOM 3k (Core_v2, matched compute)"
            "<br><sub>same Core_v2 units as the per-regime figures; above 0 = head-of-pool "
            "draw helped, near 0 = draw didn't matter; one panel per width, one line per method</sub>"
        ),
        width=360 * n_cols + 160,
        height=340 * n_rows + 120,
        margin=dict(l=80, r=60, t=100, b=60),
        legend=dict(title=dict(text="method"), orientation="v", x=1.01, y=0.5),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "delta_biased_minus_random_core_v2.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    logger.info("Wrote %s", out)


def _print_effect_table(points: list[Point]) -> None:
    """Console decomposition at each (method, width) with all three regimes."""
    by_key: dict[tuple[str, int], dict[str, dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
    for p in points:
        by_key[(p.method, p.width)][p.regime][p.flops] = p.core_v2

    logger.info("=== scale-vs-sampling at the top shared compute budget (Core_v2 x100) ===")
    logger.info("%-12s %-6s %8s %8s %8s | %8s %8s", "method", "width", "random", "biased", "10k", "sampling", "scale")
    for method in DECOMPOSABLE_METHODS:
        for width in COMPARE_WIDTHS:
            regimes = by_key.get((method, width), {})
            if not all(k in regimes for k in ("random-3k", "biased-3k", "10k")):
                continue
            # Highest FLOPs budget shared by all three regimes.
            shared = set(regimes["random-3k"]) & set(regimes["biased-3k"]) & set(regimes["10k"])
            if not shared:
                continue
            f = max(shared)
            r, b, t = regimes["random-3k"][f], regimes["biased-3k"][f], regimes["10k"][f]
            logger.info(
                "%-12s %-6d %8.2f %8.2f %8.2f | %8.2f %8.2f",
                method, width, r * 100, b * 100, t * 100, (b - r) * 100, (t - b) * 100,
            )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-3k", type=Path, default=DEFAULT_3K_CACHE)
    parser.add_argument("--cache-10k", type=Path, default=DEFAULT_10K_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view", choices=["grid", "overlay", "delta", "both", "all"], default="all")
    args = parser.parse_args(argv)

    points = _load_points(args.cache_3k, args.cache_10k)
    if not points:
        raise SystemExit("No points loaded; nothing to plot.")

    _print_effect_table(points)
    if args.view in ("grid", "both", "all"):
        _build_grid(points, args.output_dir)
    if args.view in ("overlay", "both", "all"):
        _build_overlay(points, args.output_dir)
    if args.view in ("delta", "all"):
        _build_delta(points, args.output_dir)


if __name__ == "__main__":
    main()
