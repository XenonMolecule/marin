# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Method comparison at a single WARC-scaling cell (N, hidden_size), paper quality.

For each metric, draws one figure that fixes (sampled_warcs, hidden_dim) and
plots one loss-vs-tokens curve per curation method. A dashed vertical line per
method marks one full epoch over that method's curated token pool. Optionally
shades the background by which method currently has the lowest loss
(`--shade-best`).

Reads per-run summary.json written by `run_curation_train_standalone.py`,
filtered to `experiment_tag == "expWARC_natural"`. The summary's
`plan.method_name` has the form `<base>_<N>` (e.g. `dclm_100`); we split on the
underscore to recover the base and N.

Usage::

    # Fast inner loop (no pull): re-uses the local summaries on disk.
    uv run --with matplotlib --with numpy python \\
        experiments/scaling_law_sweeps/plot_warc_scaling_method_comparison.py \\
        --n-warcs 100 --hidden-size 512

    # Refresh data, then plot.
    uv run --with matplotlib --with numpy python \\
        experiments/scaling_law_sweeps/plot_warc_scaling_method_comparison.py \\
        --n-warcs 100 --hidden-size 512 --pull

    # With the green/yellow/red best-method background shading.
    uv run --with matplotlib --with numpy python \\
        experiments/scaling_law_sweeps/plot_warc_scaling_method_comparison.py \\
        --n-warcs 100 --hidden-size 512 --shade-best

    # Just refresh data, skip plotting.
    uv run --with matplotlib --with numpy python \\
        experiments/scaling_law_sweeps/plot_warc_scaling_method_comparison.py \\
        --pull-only

The canonical method set (DCLM + Resiliparse-dedup + three spec-driven quality
tiers) and the green→yellow→red shading palette are the defaults.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from experiments.scaling_law_sweeps.plot_curation_isoflop import (
    _compute_paloma_macro_loss,
    _compute_uncheatable_macro_loss,
    load_summaries,
)

logger = logging.getLogger(__name__)

EXPERIMENT_TAG = "expWARC_natural"

DEFAULT_RESULTS_PREFIX = "scratch/warc_summaries/"
DEFAULT_OUTPUT_DIR = "scratch/plots/warc_scaling_paper"
DEFAULT_RESULTS_GS = "gs://marin-us-central1/metadata/data_curation_warc_scaling_results/"

DEFAULT_METHODS: tuple[str, ...] = (
    "dclm",
    "resiliparse_dedup",
    "low_quality",
    "med_quality",
    "high_quality",
)

DEFAULT_METRICS: tuple[str, ...] = ("uncheatable_macro_loss",)


# (display_label, line color, shade color). Quality tiers carry an "(ours)"
# suffix per user preference. Shade colors realize a green→yellow→red gradient
# across the quality tiers; DCLM and Resiliparse get pastel matches of their
# line color so every method has a shade if it ever wins.
@dataclass(frozen=True)
class MethodStyle:
    label: str
    line: str
    shade: str


METHOD_STYLES: dict[str, MethodStyle] = {
    "dclm": MethodStyle("DCLM", "#1f77b4", "#cce5ff"),
    "resiliparse_dedup": MethodStyle("Resiliparse", "#5d3a8a", "#e1d4f0"),
    "high_quality": MethodStyle("HQ (ours)", "#2ca02c", "#c8e6c9"),
    "med_quality": MethodStyle("MQ (ours)", "#ff7f0e", "#fff59d"),
    "low_quality": MethodStyle("LQ (ours)", "#e91e63", "#ffcdd2"),
}


METRIC_ALIASES: dict[str, str] = {
    "lima": "eval/lima/loss",
    "lima_loss": "eval/lima/loss",
    "lima_bpb": "eval/lima/bpb",
    "paloma": "paloma_macro_loss",
    "paloma_macro_loss": "paloma_macro_loss",
    "uncheatable": "uncheatable_macro_loss",
    "uncheatable_macro_loss": "uncheatable_macro_loss",
}

METRIC_LABELS: dict[str, str] = {
    "eval/lima/loss": "LIMA Loss",
    "eval/lima/bpb": "LIMA bpb",
    "paloma_macro_loss": "Paloma Macro Loss",
    "uncheatable_macro_loss": "Uncheatable Eval Macro Loss",
}


def resolve_metric_key(raw: str) -> str:
    return METRIC_ALIASES.get(raw, raw)


def metric_label(metric_key: str) -> str:
    return METRIC_LABELS.get(metric_key, metric_key)


def metric_short(metric_key: str) -> str:
    if metric_key.startswith("eval/lima/"):
        return "lima_" + metric_key.split("/")[-1]
    if metric_key in ("paloma_macro_loss", "uncheatable_macro_loss"):
        return metric_key
    return metric_key.replace("/", "_")


def extract_loss(eval_metrics: dict, metric_key: str) -> float | None:
    if metric_key == "paloma_macro_loss":
        return _compute_paloma_macro_loss(eval_metrics)
    if metric_key == "uncheatable_macro_loss":
        return _compute_uncheatable_macro_loss(eval_metrics)
    val = eval_metrics.get(metric_key)
    return float(val) if val is not None else None


def pull_gs_to_local(gs_prefix: str, local_dir: Path) -> None:
    """Mirror gs://.../*.json into a local directory using `gcloud storage cp`."""
    if shutil.which("gcloud") is None:
        raise RuntimeError(
            "`gcloud` not found on PATH; cannot --pull. Install Google Cloud CLI "
            "or pre-populate the local --results-prefix directory manually."
        )
    local_dir.mkdir(parents=True, exist_ok=True)
    src = gs_prefix.rstrip("/") + "/*.json"
    logger.info("pulling %s -> %s", src, local_dir)
    res = subprocess.run(
        ["gcloud", "storage", "cp", src, str(local_dir) + "/"],
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"gcloud storage cp failed (rc={res.returncode}) for {src}")


@dataclass(frozen=True)
class Point:
    tokens: float
    loss: float
    epochs: float
    flops: float
    pool_tokens: float
    params: int


def collect_points(
    summaries: list[dict],
    method_base: str,
    sampled_warcs: int,
    hidden_dim: int,
    metric_key: str,
) -> list[Point]:
    """Pull every run matching (method_base, N, hidden_dim) from the summary set."""
    full_method = f"{method_base}_{sampled_warcs}"
    pts: list[Point] = []
    for s in summaries:
        plan = s.get("plan", {})
        if plan.get("experiment_tag") != EXPERIMENT_TAG:
            continue
        if plan.get("method_name") != full_method:
            continue
        if int(plan.get("hidden_dim", 0)) != hidden_dim:
            continue
        eval_d = s.get("eval") or {}
        loss = extract_loss(eval_d, metric_key)
        if loss is None:
            continue
        toks = (s.get("tokens") or {}).get("tokens_trained")
        eps = (s.get("tokens") or {}).get("effective_epochs")
        flops = plan.get("budget_flops")
        pool = (s.get("method") or {}).get("d_obs_tokens")
        params = (s.get("model") or {}).get("total_trainable_params")
        if toks is None or eps is None or pool is None or params is None:
            continue
        pts.append(
            Point(
                tokens=float(toks),
                loss=float(loss),
                epochs=float(eps),
                flops=float(flops or 0.0),
                pool_tokens=float(pool),
                params=int(params),
            )
        )
    pts.sort(key=lambda p: p.tokens)
    return pts


def fmt_tokens(x: float, _pos=None) -> str:
    if x <= 0:
        return ""
    if x >= 1e12:
        v = x / 1e12
        return (f"{v:.1f} T" if v < 10 else f"{v:.0f} T").replace(".0 ", " ")
    if x >= 1e9:
        v = x / 1e9
        return (f"{v:.1f} B" if v < 10 else f"{v:.0f} B").replace(".0 ", " ")
    if x >= 1e6:
        v = x / 1e6
        return (f"{v:.1f} M" if v < 10 else f"{v:.0f} M").replace(".0 ", " ")
    return f"{x:.0f}"


def fmt_tokens_compact(x: float) -> str:
    """Like fmt_tokens but no space between number and unit — e.g. '98M', '1.7B'."""
    return fmt_tokens(x).replace(" ", "")


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


PAPER_RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "legend.frameon": False,
    "figure.dpi": 120,
}


def shade_by_winner(
    ax,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    palette: dict[str, str],
    n_grid: int = 400,
    alpha: float = 0.55,
) -> None:
    """Shade the axes background by which method has the lowest loss at each x.

    Operates in log-x space. A method comes into play at its first sampled token
    count and then stays in play to the right edge by holding its last observed
    loss (flat right-extrapolation). This keeps a method's winning band going
    past its last data point as long as its final loss stays below every other
    method's — otherwise a method that merely ran out of points would visually
    cede the lead to a strictly worse one. No left-extrapolation: a method is
    not assumed good before its first point. Contiguous winner-runs collapse to
    one axvspan so the result is a small number of bands.
    """
    if not curves:
        return
    x_lo = min(arr[0].min() for arr in curves.values())
    x_hi = max(arr[0].max() for arr in curves.values())
    grid = np.logspace(np.log10(x_lo), np.log10(x_hi), n_grid)

    methods = list(curves.keys())
    interp = np.full((len(methods), len(grid)), np.nan, dtype=float)
    for i, m in enumerate(methods):
        xs, ys = curves[m]
        # In play from the first sampled point onward; np.interp clamps to ys[-1]
        # beyond xs.max(), giving the flat right-extrapolation we want.
        in_range = grid >= xs.min()
        if in_range.any():
            interp[i, in_range] = np.interp(
                np.log10(grid[in_range]),
                np.log10(xs),
                ys,
            )

    winners: list[str | None] = []
    for j in range(len(grid)):
        col = interp[:, j]
        if np.all(np.isnan(col)):
            winners.append(None)
        else:
            winners.append(methods[int(np.nanargmin(col))])

    seg_start = grid[0]
    prev = winners[0]
    for j in range(1, len(grid)):
        if winners[j] != prev:
            if prev in palette:
                ax.axvspan(seg_start, grid[j], color=palette[prev], alpha=alpha, zorder=0, linewidth=0)
            seg_start = grid[j]
            prev = winners[j]
    if prev in palette:
        ax.axvspan(seg_start, grid[-1], color=palette[prev], alpha=alpha, zorder=0, linewidth=0)


def plot_method(ax, pts: list[Point], style: MethodStyle) -> float | None:
    if not pts:
        return None
    x = np.array([p.tokens for p in pts])
    y = np.array([p.loss for p in pts])

    ax.plot(x, y, "-", color=style.line, linewidth=3.0, alpha=0.95, label=style.label, zorder=3)
    ax.scatter(x, y, color=style.line, s=85, edgecolors="white", linewidths=0.8, zorder=4)
    pool = pts[0].pool_tokens
    ax.axvline(pool, color=style.line, linestyle="--", linewidth=1.8, alpha=0.55, zorder=1)
    return pool


def render_figure(
    summaries: list[dict],
    methods: list[str],
    sampled_warcs: int,
    hidden_dim: int,
    metric_key: str,
    shade_best: bool,
    out_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(PAPER_RCPARAMS)

    fig, ax = plt.subplots(figsize=(10.0, 7.0))

    gathered: dict[str, list[Point]] = {}
    for method in methods:
        pts = collect_points(summaries, method, sampled_warcs, hidden_dim, metric_key)
        if pts:
            gathered[method] = pts

    if shade_best and gathered:
        curves = {m: (np.array([p.tokens for p in pts]), np.array([p.loss for p in pts])) for m, pts in gathered.items()}
        palette = {m: METHOD_STYLES[m].shade for m in gathered if m in METHOD_STYLES}
        shade_by_winner(ax, curves, palette)

    method_pools: dict[str, float] = {}
    for method in methods:
        if method not in gathered:
            continue
        style = METHOD_STYLES.get(method)
        if style is None:
            logger.warning("No METHOD_STYLES entry for %s; using fallback color.", method)
            style = MethodStyle(label=method, line="#444444", shade="#dddddd")
        pool = plot_method(ax, gathered[method], style)
        if pool is not None:
            method_pools[style.label] = pool

    ax.set_xscale("log")
    ax.set_xlabel("Tokens trained")
    ax.set_ylabel(metric_label(metric_key))
    # Title intentionally omitted — model size + N is described in the caption.
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, p: ""))

    # Legend inside the axes, upper-right (after DCLM's spike comes back down).
    # Frame on with semi-opaque white so background shading and any curve that
    # passes through still reads cleanly.
    handles, labels = ax.get_legend_handles_labels()
    # Bake the per-method 1-epoch pool size into the label in a compact form
    # (e.g. "HQ (ours), 414M"). Quality tiers already carry the "(ours)" tag,
    # so the token count gets appended after a comma to keep it readable.
    enriched: list[str] = []
    for lab in labels:
        if lab in method_pools:
            tok = fmt_tokens_compact(method_pools[lab])
            if lab.endswith(")"):
                enriched.append(f"{lab[:-1]}, {tok})")
            else:
                enriched.append(f"{lab} ({tok})")
        else:
            enriched.append(lab)
    legend = ax.legend(
        handles,
        enriched,
        loc="upper right",
        ncol=1,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.5,
        labelspacing=0.3,
        borderaxespad=0.6,
        frameon=True,
        fontsize=15,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.92)
    legend.get_frame().set_edgecolor("#bbbbbb")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.98))

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_shaded" if shade_best else ""
    stem = f"warc_scaling__N{sampled_warcs}_d{hidden_dim}" f"__{metric_short(metric_key)}{suffix}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.3)
    # pad_inches=0 emits the tight crop directly — no separate pdfcrop pass.
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return png, pdf


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--n-warcs",
        type=int,
        default=100,
        help="WARC subsample size (e.g. 100, 500, 1000, 2000). Default: 100.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="Model hidden dim (e.g. 512, 1024, 1536). Default: 512.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Method base names (without the _N suffix). Order is preserved in the legend.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help=(
            "Metric keys or aliases. Aliases: lima, paloma_macro_loss, "
            "uncheatable_macro_loss. Generates one figure per metric."
        ),
    )
    parser.add_argument(
        "--shade-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Shade the plot background by which method currently has the lowest loss "
            "(default: on). Pass --no-shade-best to disable."
        ),
    )
    parser.add_argument(
        "--results-prefix",
        default=DEFAULT_RESULTS_PREFIX,
        help="Local directory of summary.json files. Default is local for fast iteration.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Before plotting, refresh local --results-prefix from gs:// (see --results-gs).",
    )
    parser.add_argument(
        "--pull-only",
        action="store_true",
        help="Refresh local data from gs:// and exit without plotting.",
    )
    parser.add_argument(
        "--results-gs", default=DEFAULT_RESULTS_GS, help="gs:// source for summary.json pulls when --pull is set."
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Run-name suffix filter for load_summaries (matches plot_curation_isoflop.py).",
    )
    args = parser.parse_args(argv)

    if args.pull or args.pull_only:
        pull_gs_to_local(args.results_gs, Path(args.results_prefix))
        if args.pull_only:
            logger.info("--pull-only requested; skipping plot generation.")
            return

    # load_summaries filters on `curation-{method}-`, so pass `<base>_<N>`.
    load_methods = [f"{m}_{args.n_warcs}" for m in args.methods]
    summaries = load_summaries(args.results_prefix, load_methods, args.suffix)

    out_dir = Path(args.output_dir)
    for raw_metric in args.metrics:
        metric_key = resolve_metric_key(raw_metric)
        png, pdf = render_figure(
            summaries=summaries,
            methods=list(args.methods),
            sampled_warcs=args.n_warcs,
            hidden_dim=args.hidden_size,
            metric_key=metric_key,
            shade_best=args.shade_best,
            out_dir=out_dir,
        )
        logger.info("wrote %s", png)
        logger.info("wrote %s", pdf)


if __name__ == "__main__":
    main()
