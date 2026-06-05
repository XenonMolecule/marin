# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot L(C)-per-model for the fixed-model data-curation sweep.

Primary plot: for each curation method, three curves (one per model size --
157M / 998M / 8.11B), sweeping seven compute budgets. Each point annotated
with its effective epoch count over D_obs so the epoching regime is visible
alongside the loss.

Reads per-run summary.json written by `run_curation_train_standalone.py`,
filters to `experiment_tag == "expFM_natural"` (the fixed-model sweep's tag),
and fits a power law L(C) = A * C^(-beta) per (method, hidden_size).

Usage:
    # Pull latest summaries locally (avoids SSL/gcsfs issues) then plot:
    gcloud storage cp "gs://marin-us-central1/metadata/data_curation_fixed_model_results/*.json" \\
        scratch/fm_summaries/
    uv run python experiments/scaling_law_sweeps/plot_fixed_model_sweep.py \\
        --methods all --results-prefix scratch/fm_summaries/ \\
        --output-dir scratch/plots/fixed_model

    # Single method:
    uv run python experiments/scaling_law_sweeps/plot_fixed_model_sweep.py \\
        --methods dclm --results-prefix scratch/fm_summaries/ \\
        --output-dir scratch/plots/fixed_model
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from experiments.scaling_law_sweeps.fixed_model_plan import (
    EXPERIMENT_TAG,
    TARGET_HIDDEN_SIZES,
)
from experiments.scaling_law_sweeps.plot_curation_isoflop import (
    COMPARE_COLORS,
    DEFAULT_METRIC,
    _compute_paloma_macro_loss,
    _fit_loss_vs_x,
    load_summaries,
)

# Separate results prefix for the fixed-model sweep so summaries don't mingle
# with the older ExpA/B sweep's output directory.
DEFAULT_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FixedModelRecord:
    method: str
    hidden_dim: int
    params: int
    flops: float
    tokens: float
    loss: float
    epochs_over_d_obs: float


def _extract_metric(eval_metrics: dict, metric_key: str) -> float | None:
    if metric_key == "paloma_macro_loss":
        return _compute_paloma_macro_loss(eval_metrics)
    val = eval_metrics.get(metric_key)
    return float(val) if val is not None else None


def summary_to_record(summary: dict, metric_key: str) -> FixedModelRecord | None:
    plan = summary.get("plan", {})
    if plan.get("experiment_tag") != EXPERIMENT_TAG:
        return None
    eval_metrics = summary.get("eval") or {}
    loss = _extract_metric(eval_metrics, metric_key)
    if loss is None:
        return None
    tokens_info = summary.get("tokens", {})
    model = summary.get("model", {})
    return FixedModelRecord(
        method=plan.get("method_name", "?"),
        hidden_dim=int(plan.get("hidden_dim", 0)),
        params=int(model.get("total_trainable_params", 0)),
        flops=float(plan.get("budget_flops", 0)),
        tokens=float(tokens_info.get("tokens_trained", 0)),
        loss=loss,
        epochs_over_d_obs=float(tokens_info.get("effective_epochs", 0)),
    )


def _params_label(hidden_dim: int, params: int) -> str:
    """e.g. 'd512 (157M)' -- short label for legend entries."""
    if params >= 1e9:
        pstr = f"{params / 1e9:.2f}B"
    elif params >= 1e6:
        pstr = f"{params / 1e6:.0f}M"
    else:
        pstr = f"{params}"
    return f"d{hidden_dim} ({pstr})"


def _fmt_count(val: float) -> str:
    """Format a token/element count with a B/T/M suffix (more readable than 2.66e+09)."""
    if val >= 1e12:
        return f"{val / 1e12:.2f}T"
    if val >= 1e9:
        return f"{val / 1e9:.2f}B"
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    return f"{val:.0f}"


def _fmt_x(val: float, x_axis: str) -> str:
    """Format an x-axis value. Tokens get B/T suffix; FLOPs stay scientific."""
    return _fmt_count(val) if x_axis == "tokens" else f"{val:.2e}"


METRIC_DISPLAY_LABELS: dict[str, str] = {
    "eval/lima/loss": "LIMA Loss",
    "eval/lima/bpb": "LIMA BPB",
    "paloma_macro_loss": "Paloma Macro Loss",
    "uncheatable_macro_loss": "Uncheatable Macro Loss",
    "eval/paloma/bpb": "Paloma BPB",
    "eval/paloma/macro_bpb": "Paloma Macro BPB",
    "eval/loss": "Eval Loss",
    "eval/bpb": "Eval BPB",
    "eval/macro_loss": "Eval Macro Loss",
    "eval/macro_bpb": "Eval Macro BPB",
}


def _clean_metric_label(key: str) -> str:
    return METRIC_DISPLAY_LABELS.get(key, key)


def _clean_params_label(params: int) -> str:
    """Round-to-1-sig-fig param label: 157M, 1B, 3B, 8B. (998M rounds to 1B.)"""
    rounded_b = round(params / 1e9)
    if rounded_b >= 1:
        return f"{rounded_b}B"
    return f"{round(params / 1e6)}M"


# Line style per hidden_dim so the three model sizes are visually distinct.
HIDDEN_DIM_DASH = {512: "solid", 1536: "dash", 3584: "dot"}


def plot_lc_per_model(
    records: list[FixedModelRecord],
    metric_key: str,
    output_dir: Path,
    mode: str = "projections",
    clean_titles: bool = False,
) -> None:
    """One figure per method: L(C) curves for each hidden_size.

    mode="projections": overlay the power-law fit trace (dashed, faint).
    mode="one_epoch": overlay a dashed vertical line per model size at the
        compute budget where tokens_trained equals D_obs (one epoch).
    """
    import plotly.graph_objects as go

    by_method: dict[str, list[FixedModelRecord]] = {}
    for r in records:
        by_method.setdefault(r.method, []).append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    for method, method_records in sorted(by_method.items()):
        color = COMPARE_COLORS.get(method, "#333")
        fig = go.Figure()
        by_hidden: dict[int, list[FixedModelRecord]] = {}
        for r in method_records:
            by_hidden.setdefault(r.hidden_dim, []).append(r)
        for hidden_dim in TARGET_HIDDEN_SIZES:
            pts = sorted(by_hidden.get(hidden_dim, []), key=lambda r: r.flops)
            if not pts:
                continue
            flops_list = [p.flops for p in pts]
            loss_list = [p.loss for p in pts]
            epochs_list = [p.epochs_over_d_obs for p in pts]
            params = pts[0].params
            label = _params_label(hidden_dim, params)
            dash = HIDDEN_DIM_DASH.get(hidden_dim, "solid")
            hover = [
                f"C={f:.1e} FLOPs<br>loss={l:.4f}<br>epochs={e:.2g}x"
                for f, l, e in zip(flops_list, loss_list, epochs_list, strict=True)
            ]
            fig.add_trace(
                go.Scatter(
                    x=flops_list,
                    y=loss_list,
                    mode="lines+markers+text",
                    marker=dict(size=10, color=color, symbol="circle"),
                    line=dict(color=color, dash=dash, width=2),
                    text=[f"{e:.1g}x" for e in epochs_list],
                    textposition="top center",
                    textfont=dict(size=8),
                    name=label,
                    hovertemplate="%{hovertext}<extra>" + label + "</extra>",
                    hovertext=hover,
                )
            )
            if mode == "projections":
                # Power-law fit L(C) = A * C^(-beta). Needs at least 2 points.
                fit = _fit_loss_vs_x(flops_list, loss_list)
                if fit is not None:
                    import numpy as np

                    A, beta = fit
                    c_range = np.logspace(
                        np.log10(min(flops_list)) - 0.1,
                        np.log10(max(flops_list)) + 0.1,
                        80,
                    )
                    loss_proj = A * c_range ** (-beta)
                    fig.add_trace(
                        go.Scatter(
                            x=c_range,
                            y=loss_proj,
                            mode="lines",
                            line=dict(color=color, dash=dash, width=1),
                            opacity=0.4,
                            name=f"{label} fit (beta={beta:.3f})",
                            hoverinfo="skip",
                            showlegend=True,
                        )
                    )
            elif mode in ("one_epoch", "all_epochs"):
                # Vertical dashed line(s) where this model hits k epochs over
                # D_obs: C_ke = k * (flops / epochs_over_d_obs) is invariant
                # across budget for a given (method, model).
                c_one_epoch_candidates = [p.flops / p.epochs_over_d_obs for p in pts if p.epochs_over_d_obs > 0]
                if not c_one_epoch_candidates:
                    continue
                c_one_epoch = sum(c_one_epoch_candidates) / len(c_one_epoch_candidates)
                if mode == "one_epoch":
                    ks = [1]
                else:
                    max_eps = max(p.epochs_over_d_obs for p in pts)
                    ks = list(range(1, int(max_eps) + 2))
                for k in ks:
                    x_k = k * c_one_epoch
                    opac = 0.7 if k == 1 else max(0.12, 0.5 / (k**0.5))
                    fig.add_trace(
                        go.Scatter(
                            x=[x_k, x_k],
                            y=[min(loss_list) * 0.95, max(loss_list) * 1.05],
                            mode="lines",
                            line=dict(color=color, dash=dash, width=1.5 if k == 1 else 1),
                            opacity=opac,
                            name=(f"{label} 1 epoch (C={c_one_epoch:.2e} FLOPs)" if k == 1 else f"{label} {k} ep"),
                            hoverinfo="skip",
                            showlegend=(k == 1),
                        )
                    )

        mode_tag = {
            "projections": " (projections)",
            "one_epoch": " (1-epoch markers)",
            "all_epochs": " (all-epoch markers)",
        }.get(mode, "")
        if clean_titles:
            title = f"L({_clean_metric_label(metric_key)}) per model size -- {method}"
            yaxis_title = _clean_metric_label(metric_key)
        else:
            title = (
                f"L(C) per model size -- {method}{mode_tag} (expFM_natural)"
                "<br><sub>Annotations: epochs_over_d_obs. Line style: solid=157M, dash=998M, dot=8.11B."
                + (" Fit label 'beta' refers to L = A*C^-beta (power law)." if mode == "projections" else "")
                + "</sub>"
            )
            yaxis_title = metric_key
        fig.update_layout(
            template="plotly_white",
            xaxis_type="log",
            xaxis_title="Compute budget C (FLOPs, log)",
            yaxis_title=yaxis_title,
            title=title,
            width=1000,
            height=620,
        )
        suffix = {"one_epoch": "_one_epoch", "all_epochs": "_all_epochs"}.get(mode, "")
        out = output_dir / f"lc_per_model_{method}{suffix}.html"
        fig.write_html(str(out))
        logger.info("Wrote %s", out)


_X_AXIS_SPECS = {
    "flops": {
        "attr": "flops",
        "title": "Compute budget C (FLOPs, log)",
        "hover_label": "C",
        # How far to project beyond the largest observed x. 3e20 (our max
        # budget) * 30 = ~1e22, which matches the 8.11B Chinchilla-optimal
        # range and is a natural "where would this sweep take us next?" probe.
        "projection_mult": 30.0,
    },
    "tokens": {
        "attr": "tokens",
        "title": "Tokens trained (log)",
        "hover_label": "tokens",
        "projection_mult": 30.0,
    },
}

# Minimum observed points required to draw a projection. With 2 points the
# fit is just a line through those 2; not useful as a forecast. 3+ gives a
# real least-squares fit whose beta estimate is meaningful.
_MIN_POINTS_FOR_PROJECTION = 3

# Minimum observed points required for the 3-parameter Chinchilla/Hoffmann
# fit `L = E + A * x^(-alpha)`. With N=3 the fit is exact (3 params, 3 points)
# and the curvature estimate is arbitrary. N>=4 gives real degrees of freedom
# and a meaningful irreducible-loss asymptote.
_MIN_POINTS_FOR_CHINCHILLA = 4


def _fit_loss_vs_x_chinchilla(x_vals: list[float], loss_vals: list[float]) -> tuple[float, float, float] | None:
    """Fit `L = E + A * x^(-alpha)` via non-linear least squares in log-x space.

    This is the Hoffmann/Chinchilla functional form used by
    `Training Compute-Optimal Large Language Models` and discussed in
    Muennighoff et al.'s data-constrained scaling work. Unlike the pure
    power law `L = A * x^(-alpha)` which curves to zero in log-log, this
    form has an irreducible-loss asymptote E that bends the forecast flatter
    as x grows -- which matches the observed curvature in the fixed-model
    sweep's data-epoching regime.

    Returns (E, A, alpha) or None if the fit fails or is ill-conditioned.

    Strategy:
      - Initialize with the 2-parameter power-law fit as a starting point.
      - Constrain E in [0, min(loss) - 1e-3] so `L - E > 0` and log is well-defined.
      - Constrain A, alpha > 0.
      - If scipy fails (e.g. Jacobian singular), return None.
    """
    if len(x_vals) < _MIN_POINTS_FOR_CHINCHILLA:
        return None
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except Exception:
        return None

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(loss_vals, dtype=float)

    # Warm-start from the power-law fit.
    pl = _fit_loss_vs_x(x_vals, loss_vals)
    if pl is None:
        return None
    A_init, alpha_init = pl
    E_init = max(0.0, float(y_arr.min()) - 0.5)

    def model(x, E, A, alpha):
        return E + A * np.power(x, -alpha)

    # Bounds: E in [0, min(loss) - eps], A>0, alpha>0 with reasonable caps.
    y_min = float(y_arr.min())
    eps = 1e-3
    lower = (0.0, 1e-6, 1e-3)
    upper = (max(0.0, y_min - eps) if y_min > eps else 1e-6, 1e6, 5.0)
    # If y_min is too close to 0, the Chinchilla E-asymptote isn't meaningful;
    # bail out.
    if upper[0] <= lower[0]:
        return None

    try:
        popt, _ = curve_fit(
            model,
            x_arr,
            y_arr,
            p0=(min(E_init, upper[0] - eps), A_init, alpha_init),
            bounds=(lower, upper),
            maxfev=5000,
        )
    except Exception:
        return None

    E_fit, A_fit, alpha_fit = float(popt[0]), float(popt[1]), float(popt[2])
    # Sanity: residuals must be finite.
    if not np.all(np.isfinite(model(x_arr, *popt))):
        return None
    return E_fit, A_fit, alpha_fit


def _add_method_comparison_traces(
    fig,
    hidden_records: list[FixedModelRecord],
    x_axis: str,
    mode: str,
    *,
    row: int | None = None,
    col: int | None = None,
    show_legend: bool = True,
) -> None:
    """Add per-method traces for one model size to ``fig``.

    Shared by `plot_method_comparison_per_model` (single figure) and
    `plot_method_comparison_side_by_side` (one subplot per model size).

    ``show_legend=False`` suppresses legend entries -- used by the side-by-side
    variant on every subplot after the first so the legend isn't tripled.
    """
    import plotly.graph_objects as go

    spec = _X_AXIS_SPECS[x_axis]
    x_attr = spec["attr"]
    add_kwargs = {"row": row, "col": col} if (row is not None and col is not None) else {}

    by_method: dict[str, list[FixedModelRecord]] = {}
    for r in hidden_records:
        by_method.setdefault(r.method, []).append(r)

    for method, method_records in sorted(by_method.items()):
        pts = sorted(method_records, key=lambda r: getattr(r, x_attr))
        color = COMPARE_COLORS.get(method, "#333")
        x_list = [getattr(p, x_attr) for p in pts]
        loss_list = [p.loss for p in pts]
        epochs_list = [p.epochs_over_d_obs for p in pts]
        flops_list = [p.flops for p in pts]
        tokens_list = [p.tokens for p in pts]
        # customdata carries both flops and tokens + epochs so hover has
        # all context regardless of which axis is on X.
        custom = list(zip(flops_list, tokens_list, epochs_list, strict=True))
        # Try the 3-parameter Chinchilla/Hoffmann form first
        # (L = E + A*x^-alpha) when we have >=4 points. It bends toward an
        # irreducible-loss asymptote E, which is what data-epoching
        # regimes actually produce (see Muennighoff et al. 2023). If fewer
        # points or fit fails, fall back to the 2-parameter power law.
        chinch = _fit_loss_vs_x_chinchilla(x_list, loss_list)
        pl_fit = _fit_loss_vs_x(x_list, loss_list) if len(x_list) >= 2 else None
        # Only show fit exponent in legend when we're actually drawing a
        # projection based on the fit. On epoch-marker plots the exponent
        # is disconnected from anything visible, so it just confuses.
        if mode == "projections" and chinch is not None:
            _, _, alpha_used = chinch  # E unused in legend, shown in forecast trace
            fit_suffix = f" (alpha={alpha_used:.3f})"
        elif mode == "projections" and pl_fit is not None:
            fit_suffix = f" (beta={pl_fit[1]:.3f})"
        else:
            fit_suffix = ""
        # legendgroup ties data + forecast/epoch overlays for a method together
        # so a single legend click toggles them as a unit. Cheap quality-of-life
        # win that also makes side-by-side legend dedup work uniformly.
        fig.add_trace(
            go.Scatter(
                x=x_list,
                y=loss_list,
                mode="lines+markers",
                marker=dict(size=9, color=color),
                line=dict(color=color, width=2),
                name=method + fit_suffix,
                legendgroup=method,
                showlegend=show_legend,
                customdata=custom,
                hovertemplate=(
                    "method="
                    + method
                    + "<br>FLOPs=%{customdata[0]:.1e}"
                    + "<br>tokens=%{customdata[1]:.2e}"
                    + "<br>loss=%{y:.4f}"
                    + "<br>epochs=%{customdata[2]:.2g}x<extra></extra>"
                ),
            ),
            **add_kwargs,
        )

        if mode == "projections":
            # Projection line. Prefer the Chinchilla fit when available
            # (>=4 points); fall back to pure power-law at N=3; skip at N<3.
            projection_form = None
            if chinch is not None:
                E_fit, A_fit, alpha_fit = chinch
                projection_form = "chinchilla"
                forecast_label = f"{method} forecast (E={E_fit:.3f}, A={A_fit:.2e}, alpha={alpha_fit:.3f})"
            elif pl_fit is not None and len(x_list) >= _MIN_POINTS_FOR_PROJECTION:
                A_fit, beta_fit = pl_fit
                E_fit = 0.0
                alpha_fit = beta_fit
                projection_form = "power_law"
                forecast_label = f"{method} forecast (A={A_fit:.2e}, beta={beta_fit:.3f})"

            if projection_form is not None:
                import numpy as np

                x_max = max(x_list)
                x_proj_end = x_max * spec["projection_mult"]
                x_proj = np.logspace(np.log10(x_max), np.log10(x_proj_end), 80)
                loss_proj = E_fit + A_fit * x_proj ** (-alpha_fit)
                fig.add_trace(
                    go.Scatter(
                        x=x_proj,
                        y=loss_proj,
                        mode="lines",
                        line=dict(color=color, dash="dash", width=1.5),
                        opacity=0.6,
                        name=forecast_label,
                        legendgroup=method,
                        showlegend=show_legend,
                        hovertemplate=(
                            "forecast "
                            + method
                            + " ("
                            + projection_form
                            + ")<br>"
                            + spec["hover_label"]
                            + "=%{x:.2e}<br>loss=%{y:.4f}<extra></extra>"
                        ),
                    ),
                    **add_kwargs,
                )
        elif mode in ("one_epoch", "all_epochs"):
            # Vertical dashed line(s) where this method hits k epochs over
            # D_obs (for this fixed model size). Invariant:
            # tokens_1epoch = D_obs; flops_1epoch = flops / epochs_over_d_obs.
            ones = [getattr(p, x_attr) / p.epochs_over_d_obs for p in pts if p.epochs_over_d_obs > 0]
            if ones:
                x_one_epoch = sum(ones) / len(ones)
                if mode == "one_epoch":
                    ks = [1]
                else:
                    max_eps = max(p.epochs_over_d_obs for p in pts)
                    ks = list(range(1, int(max_eps) + 2))
                for k in ks:
                    x_k = k * x_one_epoch
                    opac = 0.7 if k == 1 else max(0.12, 0.5 / (k**0.5))
                    fig.add_trace(
                        go.Scatter(
                            x=[x_k, x_k],
                            y=[min(loss_list) * 0.95, max(loss_list) * 1.05],
                            mode="lines",
                            line=dict(color=color, dash="dash", width=1.5 if k == 1 else 1),
                            opacity=opac,
                            name=(
                                f"{method} 1 epoch ({spec['hover_label']}={_fmt_x(x_one_epoch, x_axis)})"
                                if k == 1
                                else f"{method} {k} ep"
                            ),
                            legendgroup=method,
                            hoverinfo="skip",
                            showlegend=(show_legend and k == 1),
                        ),
                        **add_kwargs,
                    )


def plot_method_comparison_per_model(
    records: list[FixedModelRecord],
    metric_key: str,
    output_dir: Path,
    x_axis: str = "flops",
    mode: str = "projections",
    clean_titles: bool = False,
) -> None:
    """One figure per model size: all methods compared along x_axis (flops or tokens).

    Color-coded by curation method via `plot_curation_isoflop.COMPARE_COLORS`.
    Hover exposes epochs_over_d_obs so the epoching regime is visible.

    mode="projections": overlay dashed forecast extending past the last data point.
    mode="one_epoch": overlay a dashed vertical line per method at the x where
        tokens_trained equals D_obs (1 epoch).
    """
    import plotly.graph_objects as go

    spec = _X_AXIS_SPECS[x_axis]
    by_hidden: dict[int, list[FixedModelRecord]] = {}
    for r in records:
        by_hidden.setdefault(r.hidden_dim, []).append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    for hidden_dim in TARGET_HIDDEN_SIZES:
        hidden_records = by_hidden.get(hidden_dim, [])
        if not hidden_records:
            continue
        fig = go.Figure()
        params = hidden_records[0].params
        label = _params_label(hidden_dim, params)

        _add_method_comparison_traces(fig, hidden_records, x_axis, mode)

        mode_tag = {
            "projections": " (projections)",
            "one_epoch": " (1-epoch markers)",
            "all_epochs": " (all-epoch markers)",
        }.get(mode, "")
        fit_subtitle = (
            "<br><sub>alpha: Chinchilla fit exponent (L = E + A*x^-alpha, N>=4 pts); "
            "beta: power-law fit exponent (L = A*x^-beta, N=2-3 pts)</sub>"
            if mode == "projections"
            else ""
        )
        if clean_titles:
            title = f"Training {_clean_params_label(params)} model on data from 3000 WARC files"
            yaxis_title = _clean_metric_label(metric_key)
        else:
            title = f"Method comparison at {label} vs {x_axis}{mode_tag} -- expFM_natural{fit_subtitle}"
            yaxis_title = metric_key
        fig.update_layout(
            template="plotly_white",
            xaxis_type="log",
            xaxis_title=spec["title"],
            yaxis_title=yaxis_title,
            title=title,
            width=1000,
            height=600,
        )
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        suffix = {"one_epoch": "_one_epoch", "all_epochs": "_all_epochs"}.get(mode, "")
        out = output_dir / f"method_comparison_{safe_label}_x_{x_axis}{suffix}.html"
        fig.write_html(str(out))
        logger.info("Wrote %s", out)


# Default panels for the side-by-side variant: matches the d512 / d1536 / d2432
# triplet shared in the request. Override via `--side-by-side-hidden-sizes`.
DEFAULT_SIDE_BY_SIDE_HIDDEN_SIZES: tuple[int, ...] = (512, 1536, 2432)


def plot_method_comparison_side_by_side(
    records: list[FixedModelRecord],
    metric_key: str,
    output_dir: Path,
    x_axis: str = "tokens",
    mode: str = "all_epochs",
    clean_titles: bool = False,
    hidden_sizes: tuple[int, ...] = DEFAULT_SIDE_BY_SIDE_HIDDEN_SIZES,
) -> None:
    """Combined figure: one method-comparison panel per model size, left to right.

    Same trace logic as `plot_method_comparison_per_model` but lays out the
    requested model sizes as adjacent subplots so they can be eyeballed
    together. Legend is rendered once (on the first panel) since methods
    repeat across panels.
    """
    from plotly.subplots import make_subplots

    spec = _X_AXIS_SPECS[x_axis]
    by_hidden: dict[int, list[FixedModelRecord]] = {}
    for r in records:
        by_hidden.setdefault(r.hidden_dim, []).append(r)

    panels = [(hd, by_hidden.get(hd, [])) for hd in hidden_sizes]
    panels = [(hd, recs) for hd, recs in panels if recs]
    if not panels:
        logger.warning(
            "side-by-side: no records for hidden_sizes=%s metric=%s; skipping",
            hidden_sizes,
            metric_key,
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    subplot_titles = [
        (
            f"Training {_clean_params_label(recs[0].params)} model on data from 3000 WARC files"
            if clean_titles
            else _params_label(hd, recs[0].params)
        )
        for hd, recs in panels
    ]
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_yaxes=False,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
    )

    for idx, (_hd, hidden_records) in enumerate(panels):
        _add_method_comparison_traces(
            fig,
            hidden_records,
            x_axis,
            mode,
            row=1,
            col=idx + 1,
            show_legend=(idx == 0),
        )
        # Each panel gets its own log x-axis + axis title; y-axis title only on
        # the leftmost panel since all three share the same metric.
        fig.update_xaxes(type="log", title_text=spec["title"], row=1, col=idx + 1)
        if idx == 0:
            yaxis_title = _clean_metric_label(metric_key) if clean_titles else metric_key
            fig.update_yaxes(title_text=yaxis_title, row=1, col=idx + 1)

    mode_tag = {
        "projections": " (projections)",
        "one_epoch": " (1-epoch markers)",
        "all_epochs": " (all-epoch markers)",
    }.get(mode, "")
    if clean_titles:
        title = ""
    else:
        title = f"Method comparison side-by-side vs {x_axis}{mode_tag} -- expFM_natural"
    fig.update_layout(
        template="plotly_white",
        title=title,
        width=360 * len(panels) + 240,  # ~2/3 of the per-panel figure + legend room
        height=372,
    )

    suffix = {"one_epoch": "_one_epoch", "all_epochs": "_all_epochs"}.get(mode, "")
    sizes_tag = "_".join(f"d{hd}" for hd, _ in panels)
    out = output_dir / f"method_comparison_side_by_side_{sizes_tag}_x_{x_axis}{suffix}.html"
    fig.write_html(str(out))
    logger.info("Wrote %s", out)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "dclm",
            "nemotron_org",
            "nemotron_full",
            "fineweb_edu",
            "resiliparse",
            "llm_curated",
            "nemotron_full_bos_fixed",
            "llm_curated_bos_fixed",
            "llm_curated_dedup",
            "llm_curated_dclm_filtered",
            "high_quality_3000",
        ],
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Run-name suffix filter (matches what was passed via --run-suffix at launch). Empty = no filter.",
    )
    parser.add_argument("--metrics", nargs="+", default=[DEFAULT_METRIC, "paloma_macro_loss", "eval/lima/loss"])
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument(
        "--lima-sidecar-prefix",
        default="scratch/fm_lima_results/",
        help=(
            "Directory of post-hoc LIMA eval sidecar JSONs (one per run_name). "
            "The plotter merges each sidecar's eval/lima/* keys into the "
            "corresponding summary's eval dict so LIMA becomes a first-class metric."
        ),
    )
    parser.add_argument(
        "--prefer-bos-fixed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set (default), drop pre-BOS-fix training runs for nemotron_full/llm_curated "
            "and rename the `*_bos_fixed` variants to the base method name in the plot. "
            "Disable with --no-prefer-bos-fixed to see both. "
            "The underlying summary JSONs stay on disk either way — this is a display-only filter."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="scratch/plots/fixed_model",
        help=(
            "Where to write HTML plots + records.csv. Default lives under "
            "scratch/ (gitignored, persistent across sessions)."
        ),
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Skip plots, just dump CSVs.",
    )
    parser.add_argument(
        "--cut-after-2-loss-increases",
        action="store_true",
        help=(
            "Per-(method, hidden_dim) curve: starting after the global-min "
            "point, drop all points starting at the first index where loss "
            "increases for two consecutive steps. Catches over-epoching tails."
        ),
    )
    parser.add_argument(
        "--clean-titles",
        action="store_true",
        help=(
            "Cleaner per-figure titles for supplementary/external use: "
            "'Training Xparam model on data from 3000 WARC files' for the "
            "method-comparison plots, simplified L(C) titles for per-method "
            "plots, and human metric names like 'LIMA Loss' on the y-axis."
        ),
    )
    parser.add_argument(
        "--side-by-side-hidden-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIDE_BY_SIDE_HIDDEN_SIZES),
        help=(
            "Model hidden sizes (left-to-right) for the side-by-side combined "
            "method-comparison plots. Default: 512 1536 2432."
        ),
    )
    args = parser.parse_args(argv)

    methods = args.methods
    if "all" in methods:
        methods = [
            "dclm",
            "nemotron_org",
            "nemotron_full",
            "fineweb_edu",
            "resiliparse",
            "llm_curated",
            "nemotron_full_bos_fixed",
            "llm_curated_bos_fixed",
            "llm_curated_dclm_filtered",
            "high_quality_3000",
        ]

    summaries = load_summaries(args.results_prefix, methods, args.suffix)
    if not summaries:
        logger.error("No summaries found under %s for methods=%s suffix=%s", args.results_prefix, methods, args.suffix)
        return

    # Display-only filter: the nemotron_full/llm_curated runs trained before the
    # MarinTokenizer BOS-regression fix (2026-04-10). Their replacements carry
    # a `_bos_fixed` method name. Default: drop the old ones and rename the
    # fixed ones to the base name so the plot shows a single coherent curve.
    if args.prefer_bos_fixed:
        before = len(summaries)
        affected = {"nemotron_full", "llm_curated"}
        summaries = [s for s in summaries if s.get("plan", {}).get("method_name") not in affected]
        renames = 0
        for s in summaries:
            mn = s.get("plan", {}).get("method_name", "")
            if mn.endswith("_bos_fixed") and mn.removesuffix("_bos_fixed") in affected:
                s["plan"]["method_name"] = mn.removesuffix("_bos_fixed")
                renames += 1
        logger.info(
            "prefer-bos-fixed: hid %d pre-fix summaries, renamed %d _bos_fixed → base",
            before - len(summaries),
            renames,
        )

    # Merge LIMA sidecar data into each summary's `eval` dict so LIMA metrics
    # become plottable through the standard summary_to_record pipeline.
    lima_dir = Path(args.lima_sidecar_prefix)
    if lima_dir.exists():
        by_run = {}
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
    else:
        logger.warning("LIMA sidecar dir does not exist: %s (LIMA metrics will be skipped)", lima_dir)

    base_dir = Path(args.output_dir)
    for metric_key in args.metrics:
        # Folder-name scheme: preserve historical short names (bpb,
        # paloma_macro_loss) but disambiguate LIMA so it doesn't collide with
        # a generic "loss" folder. e.g. "eval/lima/loss" → "lima_loss".
        if metric_key.startswith("eval/lima/"):
            metric_short = "lima_" + metric_key.split("/")[-1]
        else:
            metric_short = metric_key.split("/")[-1]
        metric_dir = base_dir / metric_short
        records = [r for s in summaries if (r := summary_to_record(s, metric_key)) is not None]
        if not records:
            logger.warning("No records for metric %s", metric_key)
            continue

        if args.cut_after_2_loss_increases:
            from collections import defaultdict

            by_curve: dict[tuple[str, int], list[FixedModelRecord]] = defaultdict(list)
            for r in records:
                by_curve[(r.method, r.hidden_dim)].append(r)
            kept: list[FixedModelRecord] = []
            dropped_count = 0
            for key, curve in by_curve.items():
                curve.sort(key=lambda r: r.flops)
                # Anchor at the global-min point: tracking for 2-consec increases
                # only begins AFTER the minimum, so early-curve oscillations
                # don't accidentally trigger the cut.
                if len(curve) < 3:
                    kept.extend(curve)
                    continue
                min_idx = min(range(len(curve)), key=lambda i: curve[i].loss)
                cut_idx = len(curve)
                consec_up = 0
                for i in range(min_idx + 1, len(curve)):
                    if curve[i].loss > curve[i - 1].loss:
                        consec_up += 1
                        if consec_up >= 2:
                            # Drop starting at i (the 2nd consecutive increase);
                            # keep up to and including i-1 (the 1st increase).
                            cut_idx = i
                            break
                    else:
                        consec_up = 0
                kept.extend(curve[:cut_idx])
                dropped = len(curve) - cut_idx
                if dropped:
                    dropped_count += dropped
                    logger.info(
                        "cut-after-2-loss-increases: %s d%d -> kept %d/%d points (min at idx %d)",
                        key[0],
                        key[1],
                        cut_idx,
                        len(curve),
                        min_idx,
                    )
            logger.info(
                "cut-after-2-loss-increases: dropped %d total points across %d curves (metric=%s)",
                dropped_count,
                len(by_curve),
                metric_key,
            )
            records = kept

        logger.info(
            "Metric %s: %d records across %d methods x %d sizes",
            metric_key,
            len(records),
            len({r.method for r in records}),
            len({r.hidden_dim for r in records}),
        )

        # CSV dump is cheap and always useful.
        metric_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metric_dir / "records.csv"
        with csv_path.open("w") as f:
            f.write("method,hidden_dim,params,flops,tokens,loss,epochs_over_d_obs\n")
            for r in sorted(records, key=lambda r: (r.method, r.hidden_dim, r.flops)):
                f.write(
                    f"{r.method},{r.hidden_dim},{r.params},"
                    f"{r.flops:.6e},{r.tokens:.6e},{r.loss:.6f},{r.epochs_over_d_obs:.4f}\n"
                )
        logger.info("Wrote %s", csv_path)

        if args.csv_only:
            continue

        for mode in ("projections", "one_epoch", "all_epochs"):
            plot_lc_per_model(records, metric_key, metric_dir, mode=mode, clean_titles=args.clean_titles)
            plot_method_comparison_per_model(
                records, metric_key, metric_dir, x_axis="flops", mode=mode, clean_titles=args.clean_titles
            )
            plot_method_comparison_per_model(
                records, metric_key, metric_dir, x_axis="tokens", mode=mode, clean_titles=args.clean_titles
            )
            for x_axis in ("flops", "tokens"):
                plot_method_comparison_side_by_side(
                    records,
                    metric_key,
                    metric_dir,
                    x_axis=x_axis,
                    mode=mode,
                    clean_titles=args.clean_titles,
                    hidden_sizes=tuple(args.side_by_side_hidden_sizes),
                )


if __name__ == "__main__":
    main()
