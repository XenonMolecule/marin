# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loss-vs-tokens-trained curves for the fixed-model curation sweep, one
panel per model size.

For each curation method, plots a single line per (method, hidden_size)
showing the chosen validation loss against tokens trained. A dashed vertical
line per method marks one full epoch over that method's curated token pool.
One method may be marked as "ours" — it is drawn with a thicker line and
larger markers and is labelled "(ours)" in the legend.

Reads per-run summary.json written by `run_curation_train_standalone.py`,
filtered to `experiment_tag == "expFM_natural"`. LIMA losses are merged in
from a sidecar directory (older runs computed LIMA post-hoc).

Usage:

    # Fast inner loop (no pull): re-uses the local summaries / sidecars on disk.
    uv run --with matplotlib --with numpy python \
        experiments/scaling_law_sweeps/plot_loss_vs_tokens_by_size.py

    # Refresh data, then plot. The --pull flag rsyncs gs:// summaries + LIMA
    # sidecars into the local scratch dirs before plotting.
    uv run --with matplotlib --with numpy python \
        experiments/scaling_law_sweeps/plot_loss_vs_tokens_by_size.py --pull

    # Just refresh data, skip plotting (useful when you want fresh data for
    # several follow-up invocations).
    uv run --with matplotlib --with numpy python \
        experiments/scaling_law_sweeps/plot_loss_vs_tokens_by_size.py --pull-only

    # Single metric, custom method set (e.g. include resiliparse_dedup once
    # its coverage is sufficient):
    uv run --with matplotlib --with numpy python \
        experiments/scaling_law_sweeps/plot_loss_vs_tokens_by_size.py \
        --metrics uncheatable_macro_loss \
        --methods dclm nemotron_full_bos_fixed resiliparse_dedup high_quality_3000 \
        --ours-method high_quality_3000

The four canonical methods are:
  - dclm                      (DCLM-baseline)
  - nemotron_full_bos_fixed   (Nemotron-CC)
  - resiliparse               (Resiliparse — switch to resiliparse_dedup once we have full coverage)
  - high_quality_3000         (LLM-Curated, ours)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from experiments.scaling_law_sweeps.plot_curation_isoflop import (
    _compute_paloma_macro_loss,
    _compute_uncheatable_macro_loss,
    load_summaries,
)

logger = logging.getLogger(__name__)

EXPERIMENT_TAG = "expFM_natural"

# Default to local paths for fast iteration; `--pull` refreshes from gs://.
DEFAULT_RESULTS_PREFIX = "scratch/fm_summaries/"
DEFAULT_LIMA_SIDECAR_PREFIX = "scratch/fm_lima_sidecar/"
DEFAULT_OUTPUT_DIR = "scratch/plots/loss_vs_tokens"

DEFAULT_RESULTS_GS = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"
DEFAULT_LIMA_SIDECAR_GS = "gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results/"

DEFAULT_METHODS: tuple[str, ...] = (
    "dclm",
    "nemotron_full_bos_fixed",
    "resiliparse_dedup",
    "high_quality_3000",
)
DEFAULT_OURS_METHOD = "high_quality_3000"
DEFAULT_HIDDEN_SIZES: tuple[int, ...] = (512, 1536, 2432)
DEFAULT_METRICS: tuple[str, ...] = ("lima", "paloma_macro_loss", "uncheatable_macro_loss")

# Stable display label + color per method. Keep colors paper-friendly.
# "ours" (high_quality_3000) is green; Nemotron is red; DCLM blue; Resiliparse purple.
METHOD_DISPLAY: dict[str, tuple[str, str]] = {
    "dclm": ("DCLM-baseline", "#1f77b4"),
    "nemotron_full": ("Nemotron-CC", "#d62728"),
    "nemotron_full_bos_fixed": ("Nemotron-CC", "#d62728"),
    "resiliparse": ("Resiliparse (no dedup)", "#9467bd"),
    # Canonical "Resiliparse" line for the paper is the deduped variant.
    "resiliparse_dedup": ("Resiliparse", "#9467bd"),
    "fineweb_edu": ("FineWeb-Edu", "#ff7f0e"),
    "fineweb_cc": ("FineWeb-CC", "#e377c2"),
    "llm_curated": ("LLM-Curated", "#8c564b"),
    "llm_curated_bos_fixed": ("LLM-Curated", "#8c564b"),
    "high_quality_3000": ("Spec-Driven Extraction (ours)", "#2ca02c"),
    "high_quality_2000": ("Spec-Driven Extraction (ours, 2k)", "#2ca02c"),
    # 10k-natural sweep: method names are renamed to these base names before
    # plotting (see TENK_METHOD_MAP), so the base name carries the display entry.
    "high_quality": ("Spec-Driven Extraction (ours)", "#2ca02c"),
}

# 10k-natural method names (launch_10k_natural.METHOD_NAMES) → base method name.
# The 10k runs share the expFM_natural tag with the 3k sweep; the `_10k` suffix
# disambiguates them in GCS. Renaming to base lets them reuse METHOD_DISPLAY.
# Mirrors plot_fixed_model_sweep.TENK_METHOD_MAP (nemotron_10k → nemotron_full).
TENK_METHOD_MAP: dict[str, str] = {
    "dclm_10k": "dclm",
    "nemotron_10k": "nemotron_full",
    "high_quality_10k": "high_quality",
    "fineweb_cc_10k": "fineweb_cc",
    "fineweb_edu_10k": "fineweb_edu",
    # resiliparse_10k's cache (resiliparse_decon_10364warcs) is extract → fuzzy
    # dedup (dedup_resiliparse_warc_scaling --n 10364) → decontaminate, i.e. the
    # 10k analog of the 3k paper's resiliparse_dedup. Map to that base so it gets
    # the deduped "Resiliparse" label (not "Resiliparse (no dedup)").
    "resiliparse_10k": "resiliparse_dedup",
}


# Resolves CLI-friendly metric aliases to the canonical key used in summaries.
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
    "uncheatable_macro_loss": "Uncheatable-Eval Macro Loss",
}


def resolve_metric_key(raw: str) -> str:
    return METRIC_ALIASES.get(raw, raw)


def metric_label(metric_key: str) -> str:
    if metric_key in METRIC_LABELS:
        return METRIC_LABELS[metric_key]
    # Generic fallback for arbitrary eval/.../loss keys.
    return metric_key


def metric_short(metric_key: str) -> str:
    """Filename-friendly short name for a metric key."""
    if metric_key.startswith("eval/lima/"):
        return "lima_" + metric_key.split("/")[-1]
    if metric_key == "paloma_macro_loss":
        return "paloma_macro_loss"
    if metric_key == "uncheatable_macro_loss":
        return "uncheatable_macro_loss"
    return metric_key.replace("/", "_")


def extract_loss(eval_metrics: dict, metric_key: str) -> float | None:
    if metric_key == "paloma_macro_loss":
        return _compute_paloma_macro_loss(eval_metrics)
    if metric_key == "uncheatable_macro_loss":
        return _compute_uncheatable_macro_loss(eval_metrics)
    val = eval_metrics.get(metric_key)
    return float(val) if val is not None else None


def pull_gs_to_local(gs_prefix: str, local_dir: Path) -> None:
    """Mirror gs://.../*.json into a local directory using `gcloud storage cp`.

    Only the JSON files at the prefix are copied (no recursive walk).
    Local files that no longer exist remotely are NOT deleted — we just
    overwrite. Fast-iteration users can `rm scratch/fm_summaries/*.json`
    manually if they want a clean refresh.
    """
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


def merge_lima_sidecars(summaries: list[dict], lima_dir: Path) -> None:
    """Merge eval/lima/* sidecar values into each matching summary's eval dict."""
    if not lima_dir.exists():
        logger.warning("LIMA sidecar dir does not exist: %s (LIMA metrics will be skipped)", lima_dir)
        return
    by_run: dict[str, dict] = {}
    for p in lima_dir.glob("*.json"):
        try:
            with p.open() as f:
                sc = json.load(f)
            run_name = sc.get("run_name") or p.stem
            by_run[run_name] = sc
        except Exception as e:
            logger.warning("Skipping malformed LIMA sidecar %s: %s", p, e)
    merged = 0
    for s in summaries:
        run_name = s.get("plan", {}).get("run_name")
        sc = by_run.get(run_name)
        if sc is None:
            continue
        s.setdefault("eval", {})
        for k, v in sc.items():
            if k.startswith("eval/lima/") and v is not None:
                s["eval"][k] = v
        merged += 1
    logger.info("Merged LIMA sidecars into %d/%d summaries", merged, len(summaries))


def apply_bos_fixed_preference(summaries: list[dict]) -> list[dict]:
    """For nemotron_full / llm_curated: drop pre-BOS-fix rows, rename `*_bos_fixed`
    to the base method name. Matches `plot_fixed_model_sweep.py --prefer-bos-fixed`.
    """
    out: list[dict] = []
    renames = 0
    for s in summaries:
        mn = s.get("plan", {}).get("method_name")
        if mn in ("nemotron_full", "llm_curated"):
            continue  # pre-BOS-fix; drop
        out.append(s)
        if mn in ("nemotron_full_bos_fixed", "llm_curated_bos_fixed"):
            s["plan"]["method_name"] = mn.removesuffix("_bos_fixed")
            renames += 1
    logger.info(
        "prefer-bos-fixed: hid %d pre-fix summaries, renamed %d _bos_fixed → base",
        len(summaries) - len(out),
        renames,
    )
    return out


def apply_tenk_rename(summaries: list[dict]) -> None:
    """Rename 10k-natural method names (dclm_10k, ...) to their base method in place.

    No-op for the 3k sweep (no `_10k` names). Runs after apply_bos_fixed_preference
    so nemotron_10k → nemotron_full isn't caught by that step's pre-fix drop.
    """
    renamed = 0
    for s in summaries:
        mn = s.get("plan", {}).get("method_name")
        base = TENK_METHOD_MAP.get(mn)
        if base is not None:
            s["plan"]["method_name"] = base
            renamed += 1
    if renamed:
        logger.info("renamed %d 10k-natural method names to their base method", renamed)


@dataclass(frozen=True)
class Point:
    tokens: float
    loss: float
    epochs: float
    flops: float
    pool_tokens: float  # d_obs_tokens for the curated pool (one-epoch size)
    params: int  # total_trainable_params at this hidden_dim


def collect_points(
    summaries: list[dict],
    method: str,
    hidden_dim: int,
    metric_key: str,
) -> list[Point]:
    pts: list[Point] = []
    for s in summaries:
        plan = s.get("plan", {})
        if plan.get("experiment_tag") != EXPERIMENT_TAG:
            continue
        if plan.get("method_name") != method:
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
    """Format token counts as e.g. '300 M', '1 B', '10 B', '1 T'."""
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


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


# Chinchilla compute-optimal is ~20 tokens per parameter; overtraining multiples
# mark how far past compute-optimal each panel's runs reach. These are per-panel
# (each panel is a fixed model size, so the token budget scales with params).
CHINCHILLA_TOKENS_PER_PARAM = 20.0
# Multiplication sign for the "10x/100x" labels, built via chr() so the source
# stays pure ASCII (ruff RUF001 flags a literal multiplication glyph as ambiguous).
_MULT = chr(0x00D7)
# (multiplier over Chinchilla-optimal, short top-of-line label)
CHINCHILLA_MARKERS: tuple[tuple[float, str], ...] = (
    (1.0, f"1{_MULT}"),
    (10.0, f"10{_MULT}"),
    (100.0, f"100{_MULT}"),
)
CHINCHILLA_MARKER_COLOR = "0.30"
CHINCHILLA_LEGEND_LABEL = f"Chinchilla-optimal (20 tok/param) & 10{_MULT}/100{_MULT} overtrained"


def draw_chinchilla_markers(ax, params: int) -> None:
    """Draw vertical guides at 1x/10x/100x the Chinchilla-optimal token budget.

    Compute-optimal is ~20 tokens/param, so the guides sit at 20*params,
    200*params, 2000*params. The x-limits are frozen to the data range first,
    and any marker falling outside it is skipped, so the guides never stretch
    the axis. Short "1x/10x/100x" labels are drawn at the top of each line.
    """
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)  # freeze so axvline never rescales the axis
    ymin, ymax = ax.get_ylim()
    for mult, label in CHINCHILLA_MARKERS:
        xt = mult * CHINCHILLA_TOKENS_PER_PARAM * params
        if not (xmin <= xt <= xmax):
            continue
        ax.axvline(xt, color=CHINCHILLA_MARKER_COLOR, linestyle=(0, (1, 1)), linewidth=1.6, alpha=0.7, zorder=2)
        ax.text(
            xt,
            ymax,
            label,
            rotation=90,
            ha="right",
            va="top",
            fontsize=14,
            color=CHINCHILLA_MARKER_COLOR,
            alpha=0.9,
            zorder=2,
            clip_on=True,
        )
    ax.set_ylim(ymin, ymax)


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


def plot_method(ax, pts: list[Point], color: str, label: str, is_ours: bool) -> float | None:
    """Draw one method on `ax`. Returns the method's 1-epoch pool size."""
    if not pts:
        return None
    x = np.array([p.tokens for p in pts])
    y = np.array([p.loss for p in pts])

    lw = 4.5 if is_ours else 3.0
    ms = 140 if is_ours else 85
    z_line = 5 if is_ours else 3
    z_pt = 6 if is_ours else 4
    alpha_line = 1.0 if is_ours else 0.95

    ax.plot(x, y, "-", color=color, linewidth=lw, alpha=alpha_line, label=label, zorder=z_line)
    ax.scatter(x, y, color=color, s=ms, edgecolors="white", linewidths=1.0 if is_ours else 0.8, zorder=z_pt)

    pool = pts[0].pool_tokens
    ax.axvline(
        pool, color=color, linestyle="--", linewidth=2.6 if is_ours else 1.8, alpha=0.75 if is_ours else 0.55, zorder=1
    )
    return pool


def draw_panel(
    ax,
    summaries: list[dict],
    methods: list[str],
    hidden_dim: int,
    metric_key: str,
    ours_method: str | None,
    method_pools: dict[str, float],
) -> int | None:
    """Draw one (method-set, hidden_dim, metric) panel on `ax`.

    Plots each method's curve + one-epoch marker, styles the log-x axis, and
    overlays the Chinchilla-optimal guides. Accumulates one-epoch pool sizes
    into `method_pools` (keyed by display label) for the shared legend.
    Returns the panel's trainable-param count (None if no method had data).
    """
    panel_params: int | None = None
    for method in methods:
        pts = collect_points(summaries, method, hidden_dim, metric_key)
        label, color = METHOD_DISPLAY.get(method, (method, "#444444"))
        pool = plot_method(ax, pts, color, label, is_ours=(method == ours_method))
        if pool is not None:
            method_pools.setdefault(label, pool)
        if panel_params is None and pts:
            panel_params = pts[0].params
    ax.set_xscale("log")
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, p: ""))
    if panel_params is not None:
        draw_chinchilla_markers(ax, panel_params)
    return panel_params


def add_shared_legend(fig, axes_flat, method_pools: dict[str, float], y_anchor: float) -> None:
    """Build one figure-level legend from all panels, ordered by pool size.

    De-duplicates method labels across panels, enriches each with its
    tok/epoch pool size, and appends the Chinchilla-guide proxy entry.
    """
    handles, labels = [], []
    for ax in axes_flat:
        h, ll = ax.get_legend_handles_labels()
        for hh, lab in zip(h, ll, strict=True):
            if lab not in labels:
                handles.append(hh)
                labels.append(lab)
    pairs = list(zip(handles, labels, strict=True))
    pairs.sort(key=lambda hl: method_pools.get(hl[1], float("inf")))
    handles = [h for h, _ in pairs]
    enriched_labels = [
        f"{lab}  ({fmt_tokens(method_pools[lab])} tok/epoch)" if lab in method_pools else lab for _, lab in pairs
    ]

    # Explain the gray dotted vertical guides drawn by draw_chinchilla_markers.
    handles.append(Line2D([0], [0], color=CHINCHILLA_MARKER_COLOR, linestyle=(0, (1, 1)), linewidth=1.6))
    enriched_labels.append(CHINCHILLA_LEGEND_LABEL)

    fig.legend(
        handles,
        enriched_labels,
        loc="lower center",
        ncol=len(enriched_labels),
        bbox_to_anchor=(0.5, y_anchor),
        handlelength=2.4,
        columnspacing=1.8,
    )


def render_figure(
    summaries: list[dict],
    methods: list[str],
    hidden_sizes: list[int],
    metric_key: str,
    ours_method: str | None,
    out_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update(PAPER_RCPARAMS)

    n_panels = len(hidden_sizes)
    fig_w = max(8.0, 7.5 * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 7.0), sharey=False)
    if n_panels == 1:
        axes = [axes]

    method_pools: dict[str, float] = {}
    for ax, hidden_dim in zip(axes, hidden_sizes, strict=True):
        panel_params = draw_panel(ax, summaries, methods, hidden_dim, metric_key, ours_method, method_pools)
        panel_title = (
            f"{fmt_params(panel_params)} params  (d={hidden_dim})" if panel_params is not None else f"d={hidden_dim}"
        )
        ax.set_xlabel("Tokens trained")
        ax.set_title(panel_title, pad=10, fontweight="semibold")

    axes[0].set_ylabel(metric_label(metric_key))

    add_shared_legend(fig, axes, method_pools, y_anchor=0.04)

    fig.tight_layout(rect=(0.04, 0.10, 0.99, 0.98))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"loss_vs_tokens__{metric_short(metric_key)}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.3)
    # pad_inches=0 emits the tight crop directly — no separate pdfcrop pass.
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return png, pdf


def render_grid_figure(
    summaries: list[dict],
    methods: list[str],
    hidden_sizes: list[int],
    metric_keys: list[str],
    ours_method: str | None,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Render a single figure with one row per metric and one column per size.

    Columns share model size (title + Chinchilla guides on the top row, x-axis
    label on the bottom row); each row is a metric (y-axis label on the left
    column). One shared legend spans the bottom.
    """
    plt.rcParams.update(PAPER_RCPARAMS)

    n_rows = len(metric_keys)
    n_cols = len(hidden_sizes)
    fig_w = max(8.0, 7.5 * n_cols)
    fig_h = 6.2 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=False, squeeze=False)

    method_pools: dict[str, float] = {}
    panel_params_by_col: dict[int, int] = {}
    for r, metric_key in enumerate(metric_keys):
        for c, hidden_dim in enumerate(hidden_sizes):
            ax = axes[r][c]
            panel_params = draw_panel(ax, summaries, methods, hidden_dim, metric_key, ours_method, method_pools)
            if panel_params is not None:
                panel_params_by_col[c] = panel_params
            if c == 0:
                ax.set_ylabel(metric_label(metric_key))
            if r == n_rows - 1:
                ax.set_xlabel("Tokens trained")

    for c, hidden_dim in enumerate(hidden_sizes):
        pp = panel_params_by_col.get(c)
        title = f"{fmt_params(pp)} params  (d={hidden_dim})" if pp is not None else f"d={hidden_dim}"
        axes[0][c].set_title(title, pad=10, fontweight="semibold")

    # Legend sits below all rows; anchor scales inversely with row count so it
    # clears the bottom panels regardless of figure height.
    legend_y = 0.06 / n_rows
    add_shared_legend(fig, [ax for row in axes for ax in row], method_pools, y_anchor=legend_y)

    fig.tight_layout(rect=(0.04, 0.06 + 0.02 * n_rows, 0.99, 0.99))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "loss_vs_tokens__grid__" + "_".join(metric_short(m) for m in metric_keys)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return png, pdf


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Method names (as written in summary plan.method_name). Order is preserved.",
    )
    parser.add_argument(
        "--ours-method",
        default=DEFAULT_OURS_METHOD,
        help="Which method to highlight as 'ours' (thicker line / larger markers / legend tag).",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_HIDDEN_SIZES),
        help="Hidden dims to render as side-by-side panels.",
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
        "--results-prefix",
        default=DEFAULT_RESULTS_PREFIX,
        help="Where to find summary.json files. Default is local for fast iteration.",
    )
    parser.add_argument("--lima-sidecar-prefix", default=DEFAULT_LIMA_SIDECAR_PREFIX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pull",
        action="store_true",
        help=(
            "Before plotting, refresh local --results-prefix and "
            "--lima-sidecar-prefix from gs:// (see --results-gs / --lima-sidecar-gs). "
            "Default is no pull — re-uses whatever is on disk for fast iteration."
        ),
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
        "--lima-sidecar-gs",
        default=DEFAULT_LIMA_SIDECAR_GS,
        help="gs:// source for LIMA sidecar pulls when --pull is set.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Run-name suffix filter (matches plot_fixed_model_sweep.py --suffix).",
    )
    parser.add_argument(
        "--prefer-bos-fixed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set (default), drop pre-BOS-fix nemotron_full/llm_curated rows and "
            "rename `*_bos_fixed` variants to the base method name."
        ),
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help=(
            "Render a single combined figure with one row per --metric and one "
            "column per --hidden-size, instead of one figure per metric."
        ),
    )
    args = parser.parse_args(argv)

    if args.pull or args.pull_only:
        pull_gs_to_local(args.results_gs, Path(args.results_prefix))
        pull_gs_to_local(args.lima_sidecar_gs, Path(args.lima_sidecar_prefix))
        if args.pull_only:
            logger.info("--pull-only requested; skipping plot generation.")
            return

    # When the user explicitly asks for a non-default method set that includes
    # a `*_bos_fixed` variant, don't silently rewrite it back to the base name.
    rewrite_methods = args.methods
    if args.prefer_bos_fixed:
        # We need to load summaries for both the base name and the _bos_fixed
        # name so the rewrite step can do its job before filtering.
        load_methods = set(rewrite_methods)
        for m in list(load_methods):
            if m in ("nemotron_full", "llm_curated"):
                load_methods.add(m + "_bos_fixed")
        load_methods = sorted(load_methods)
    else:
        load_methods = list(rewrite_methods)

    summaries = load_summaries(args.results_prefix, load_methods, args.suffix)
    methods_for_plot = list(args.methods)
    if args.prefer_bos_fixed:
        summaries = apply_bos_fixed_preference(summaries)
        # In-summary method_name has been rewritten from `*_bos_fixed` to the
        # base; mirror that on the plot-side method list so the filter matches.
        methods_for_plot = [m.removesuffix("_bos_fixed") if m.endswith("_bos_fixed") else m for m in methods_for_plot]

    # 10k-natural runs are loaded by their `_10k` names (so load_summaries finds
    # the files) then renamed to base; mirror that on the plot-side method list.
    apply_tenk_rename(summaries)
    methods_for_plot = [TENK_METHOD_MAP.get(m, m) for m in methods_for_plot]

    lima_dir = Path(args.lima_sidecar_prefix)
    merge_lima_sidecars(summaries, lima_dir)

    ours_for_plot = (
        args.ours_method.removesuffix("_bos_fixed")
        if args.prefer_bos_fixed and args.ours_method.endswith("_bos_fixed")
        else args.ours_method
    )
    ours_for_plot = TENK_METHOD_MAP.get(ours_for_plot, ours_for_plot)

    out_dir = Path(args.output_dir)
    metric_keys = [resolve_metric_key(m) for m in args.metrics]

    if args.grid:
        png, pdf = render_grid_figure(
            summaries=summaries,
            methods=methods_for_plot,
            hidden_sizes=args.hidden_sizes,
            metric_keys=metric_keys,
            ours_method=ours_for_plot,
            out_dir=out_dir,
        )
        logger.info("wrote %s", png)
        logger.info("wrote %s", pdf)
        return

    for metric_key in metric_keys:
        png, pdf = render_figure(
            summaries=summaries,
            methods=methods_for_plot,
            hidden_sizes=args.hidden_sizes,
            metric_key=metric_key,
            ours_method=ours_for_plot,
            out_dir=out_dir,
        )
        logger.info("wrote %s", png)
        logger.info("wrote %s", pdf)


if __name__ == "__main__":
    main()
