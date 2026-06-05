# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot per-(hidden_size, N) method comparisons for the WARC-scaling sweep.

Reuses `plot_fixed_model_sweep`'s `plot_method_comparison_per_model` shape but
indexes by (hidden_dim, sampled_warcs) instead of (hidden_dim) alone, so each
plot fixes both the architecture AND the input WARC count and compares the
4 curation methods at that cell.

Reads per-run summary.json filtered to `experiment_tag == "expWARC_natural"`.

Usage:
    gcloud storage cp "gs://marin-us-central1/metadata/data_curation_warc_scaling_results/*.json" \\
        scratch/warc_summaries/
    uv run python experiments/scaling_law_sweeps/plot_warc_scaling_sweep.py \\
        --results-prefix scratch/warc_summaries/ \\
        --output-dir scratch/plots/warc_scaling
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from experiments.scaling_law_sweeps.plot_curation_isoflop import (
    COMPARE_COLORS,
    DEFAULT_METRIC,
    _compute_paloma_macro_loss,
    _compute_uncheatable_macro_loss,
    _fit_loss_vs_x,
    load_summaries,
)
from experiments.scaling_law_sweeps.plot_fixed_model_sweep import (
    _MIN_POINTS_FOR_PROJECTION,
    _X_AXIS_SPECS,
    _clean_metric_label,
    _clean_params_label,
    _fit_loss_vs_x_chinchilla,
    _fmt_x,
    _params_label,
)
from experiments.scaling_law_sweeps.warc_scaling_plan import (
    EXPERIMENT_TAG,
    WARC_COUNTS,
    WARC_METHOD_BASE_NAMES,
    hidden_sizes_for,
)


def _all_method_keys() -> list[str]:
    """Full method-name list (`{base}_{N}`) for `load_summaries` filtering."""
    return [f"{base}_{n}" for base in WARC_METHOD_BASE_NAMES for n in WARC_COUNTS]


def _load_fm_lima_sidecar(prefix: str) -> dict[str, dict]:
    """Load LIMA sidecar JSONs keyed by run_name.

    Each sidecar carries `eval/lima/loss` and `eval/lima/bpb` for older
    fixed-model runs that were evaluated before LIMA was in the validation set.
    Returns {run_name: {eval/lima/loss: ..., eval/lima/bpb: ...}}.
    """
    import fsspec

    sidecars: dict[str, dict] = {}
    if not prefix:
        return sidecars

    if prefix.startswith("gs://"):
        fs, base = fsspec.core.url_to_fs(prefix.rstrip("/"))
        files = [p for p in fs.ls(base) if p.endswith(".json")]
        for fp in files:
            try:
                with fs.open(fp, "r") as f:
                    sd = json.load(f)
            except Exception as e:
                logger.warning("sidecar read failed %s: %s", fp, e)
                continue
            run_name = sd.get("run_name")
            if not run_name:
                continue
            metrics = {k: sd[k] for k in ("eval/lima/loss", "eval/lima/bpb") if sd.get(k) is not None}
            if metrics:
                sidecars[run_name] = metrics
    else:
        local = Path(prefix)
        for fp in sorted(local.glob("*.json")):
            try:
                sd = json.loads(fp.read_text())
            except Exception as e:
                logger.warning("sidecar read failed %s: %s", fp, e)
                continue
            run_name = sd.get("run_name")
            if not run_name:
                continue
            metrics = {k: sd[k] for k in ("eval/lima/loss", "eval/lima/bpb") if sd.get(k) is not None}
            if metrics:
                sidecars[run_name] = metrics
    logger.info("Loaded %d FM LIMA sidecars from %s", len(sidecars), prefix)
    return sidecars


def _merge_sidecar_into_summary(summary: dict, sidecars: dict[str, dict]) -> None:
    """Merge sidecar lima metrics into the summary's eval dict in place.

    Only fills missing keys — never overwrites an existing eval value (so
    runs that DID log lima inline keep their original numbers).
    """
    plan = summary.get("plan", {})
    run_name = plan.get("run_name") or plan.get("run_name_core")
    if not run_name or run_name not in sidecars:
        return
    eval_d = summary.setdefault("eval", {}) or {}
    if eval_d is None:
        eval_d = {}
        summary["eval"] = eval_d
    for k, v in sidecars[run_name].items():
        eval_d.setdefault(k, v)


DEFAULT_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_warc_scaling_results/"
# Fixed-model sweep results = the 3000-WARC anchor experiment. Merging these
# in as N=3000 records lets the grid show the full N axis (100 → 3000).
FM_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"
FM_EXPERIMENT_TAG = "expFM_natural"
# LIMA sidecar: many older fixed-model runs were evaluated before LIMA was
# added to the validation suite. The sidecar holds eval/lima/{loss,bpb} keyed
# by run_name and is merged into the main summary's eval dict at load time
# so cross-N grid plots show LIMA at N=3000 too.
FM_LIMA_SIDECAR_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results/"

# 10k natural-epoch sweep (launch_10k_natural.py). These runs share the
# expFM_natural tag and write to their own results prefix; their methods carry
# sampled_warcs=10364 (curation_plan.EXPC_SAMPLED_WARCS — the definitive count;
# the manifest has 10363 lines but 10364 records due to a missing trailing
# newline). They route through `fm_summary_to_record` (same as the N=3000 FM
# anchor) into an N=10364 grid row once their summaries are loaded.
TENK_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_10k_natural_results/"
TENK_N = 10364
TENK_WIDTHS = (512, 1024, 1536, 2432, 3584)  # = launch_10k_natural.WIDTHS
TENK_METHOD_KEYS = ("dclm_10k", "nemotron_10k")

# Map fixed-model method_name → warc-scaling base name. BOS-fixed versions are
# preferred since the WARC subsamples use the BOS-fixed caches; the older
# non-BOS-fixed runs are dropped from the grid.
_FM_METHOD_MAP: dict[str, str] = {
    "dclm": "dclm",
    "resiliparse": "resiliparse",
    "nemotron_full_bos_fixed": "nemotron_full",
    # Nemotron-CC-HQ canonical 3k. Same name in FM and WARC sweeps.
    "nemotron_qhigh": "nemotron_qhigh",
    "llm_curated_bos_fixed": "llm_curated",
    # Fuzzy doc-deduped llm_curated. This is the SAME corpus as low_quality
    # at N=3000 (low_quality is the legacy llm_curated extraction spec), so we
    # alias it onto the "low_quality" base name to serve as the N=3000 anchor
    # in the WARC-scaling cross-N plot. The existing llm_curated_dedup FM runs
    # plot at N=3000 on the low_quality curve alongside the per-N WARC
    # subsamples (low_quality_500/1000/2000).
    "llm_curated_dedup": "low_quality",
    # Fuzzy doc-deduped resiliparse N=3000 anchor — anchors the
    # resiliparse_dedup curve in the WARC-scaling cross-N plot. Same base
    # name as the per-N entries (resiliparse_dedup_500/1000/2000).
    "resiliparse_dedup": "resiliparse_dedup",
    # DCLM-faithful curation pipeline on llm_curated extraction (full corpus).
    "llm_curated_dclm_filtered": "llm_curated_dclm_filtered",
    # high_quality at N=3000 (full corpus). Method name carries the _3000 suffix
    # in the FM sweep registration but maps to the same base as the per-N WARC
    # subsamples (high_quality_500/1000/2000) so cross-N plots show one curve.
    "high_quality_3000": "high_quality",
    # WARC-scaling methods launched via the FM sweep (when we need a custom
    # budget below the per-N _BUDGETS_PER_N floor, e.g. 3e16 at N=2000 for
    # an extended low-token comparison). sampled_warcs is read from the
    # method entry in curation_plan.METHODS, so these still route to the
    # correct (base, N) cell on the warc-scaling grid.
    "dclm_2000": "dclm",
    "high_quality_2000": "high_quality",
    # 10k natural-epoch methods (launch_10k_natural.py). sampled_warcs=10364 in
    # their summaries routes them to the N=10364 row via fm_summary_to_record.
    # nemotron_10k is the full Nemotron-CC corpus -> the "nemotron_full" base.
    "dclm_10k": "dclm",
    "nemotron_10k": "nemotron_full",
    # Intentionally excluded: "fineweb_edu" (dropped from sweep),
    # "nemotron_full" / "llm_curated" (non-BOS-fixed, superseded).
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WarcScalingRecord:
    method: str  # base name without _N suffix (e.g., "dclm")
    sampled_warcs: int  # 100, 500, 1000, or 2000
    hidden_dim: int
    params: int
    flops: float
    tokens: float
    loss: float
    epochs_over_d_obs: float


def _extract_metric(eval_metrics: dict, metric_key: str) -> float | None:
    if metric_key == "paloma_macro_loss":
        return _compute_paloma_macro_loss(eval_metrics)
    if metric_key == "uncheatable_macro_loss":
        return _compute_uncheatable_macro_loss(eval_metrics)
    val = eval_metrics.get(metric_key)
    return float(val) if val is not None else None


def fm_summary_to_record(summary: dict, metric_key: str) -> WarcScalingRecord | None:
    """Map a fixed-model (3000-WARC) summary into a WarcScalingRecord.

    Filters to BOS-fixed methods only (nemotron/llm_curated) so the merged grid
    compares like-for-like with the WARC-subsample sweep. sampled_warcs comes
    from `method.sampled_warcs` in the summary (3000 by construction).
    """
    plan = summary.get("plan", {})
    if plan.get("experiment_tag") != FM_EXPERIMENT_TAG:
        return None
    fm_method = plan.get("method_name", "")
    base = _FM_METHOD_MAP.get(fm_method)
    if base is None:
        return None
    eval_metrics = summary.get("eval") or {}
    loss = _extract_metric(eval_metrics, metric_key)
    if loss is None:
        return None
    method_info = summary.get("method", {})
    model_info = summary.get("model", {})
    tokens_info = summary.get("tokens", {})
    return WarcScalingRecord(
        method=base,
        sampled_warcs=int(method_info.get("sampled_warcs", 3000)),
        hidden_dim=int(plan.get("hidden_dim", 0)),
        params=int(model_info.get("total_trainable_params", 0)),
        flops=float(plan.get("budget_flops", 0.0)),
        tokens=float(tokens_info.get("tokens_trained", 0.0)),
        loss=loss,
        epochs_over_d_obs=float(tokens_info.get("effective_epochs", 0.0)),
    )


def summary_to_record(summary: dict, metric_key: str) -> WarcScalingRecord | None:
    """Parse one run summary into a WarcScalingRecord (or skip).

    Filters out non-WARC-scaling experiments and any run with the metric missing.
    The method_name in the summary is `<base>_<N>` (e.g. "dclm_500"); we split
    to recover the base name + N.
    """
    plan = summary.get("plan", {})
    if plan.get("experiment_tag") != EXPERIMENT_TAG:
        return None
    eval_metrics = summary.get("eval") or {}
    loss = _extract_metric(eval_metrics, metric_key)
    if loss is None:
        return None
    full_method = plan.get("method_name", "")
    if "_" not in full_method:
        return None
    base, n_str = full_method.rsplit("_", 1)
    try:
        sampled_warcs = int(n_str)
    except ValueError:
        return None
    model_info = summary.get("model", {})
    tokens_info = summary.get("tokens", {})
    return WarcScalingRecord(
        method=base,
        sampled_warcs=sampled_warcs,
        hidden_dim=int(plan.get("hidden_dim", 0)),
        params=int(model_info.get("total_trainable_params", 0)),
        flops=float(plan.get("budget_flops", 0.0)),
        tokens=float(tokens_info.get("tokens_trained", 0.0)),
        loss=loss,
        epochs_over_d_obs=float(tokens_info.get("effective_epochs", 0.0)),
    )


def plot_method_comparison_per_size_and_n(
    records: list[WarcScalingRecord],
    metric_key: str,
    output_dir: Path,
    x_axis: str = "tokens",
    mode: str = "all_epochs",
    clean_titles: bool = False,
) -> None:
    """One figure per (hidden_dim, N): all 4 methods compared along x_axis.

    Mirrors `plot_fixed_model_sweep.plot_method_comparison_per_model` but adds
    the sampled_warcs dimension. mode in {"projections", "one_epoch", "all_epochs"}.
    """
    import plotly.graph_objects as go

    spec = _X_AXIS_SPECS[x_axis]
    x_attr = spec["attr"]

    by_cell: dict[tuple[int, int], list[WarcScalingRecord]] = {}
    for r in records:
        by_cell.setdefault((r.hidden_dim, r.sampled_warcs), []).append(r)

    for (hidden_dim, n_warcs), cell_records in sorted(by_cell.items()):
        # One subfolder per N for easier navigation
        n_dir = output_dir / f"N{n_warcs}"
        n_dir.mkdir(parents=True, exist_ok=True)
        if not cell_records:
            continue
        params = cell_records[0].params
        label = _params_label(hidden_dim, params)

        by_method: dict[str, list[WarcScalingRecord]] = {}
        for r in cell_records:
            by_method.setdefault(r.method, []).append(r)

        fig = go.Figure()
        for method, method_records in sorted(by_method.items()):
            pts = sorted(method_records, key=lambda r: getattr(r, x_attr))
            color = COMPARE_COLORS.get(method, "#333")
            x_list = [getattr(p, x_attr) for p in pts]
            loss_list = [p.loss for p in pts]
            epochs_list = [p.epochs_over_d_obs for p in pts]
            flops_list = [p.flops for p in pts]
            tokens_list = [p.tokens for p in pts]
            custom = list(zip(flops_list, tokens_list, epochs_list, strict=True))
            chinch = _fit_loss_vs_x_chinchilla(x_list, loss_list)
            pl_fit = _fit_loss_vs_x(x_list, loss_list) if len(x_list) >= 2 else None
            if mode == "projections" and chinch is not None:
                _, _, alpha_used = chinch
                fit_suffix = f" (alpha={alpha_used:.3f})"
            elif mode == "projections" and pl_fit is not None:
                fit_suffix = f" (beta={pl_fit[1]:.3f})"
            else:
                fit_suffix = ""
            fig.add_trace(
                go.Scatter(
                    x=x_list,
                    y=loss_list,
                    mode="lines+markers",
                    marker=dict(size=9, color=color),
                    line=dict(color=color, width=2),
                    name=method + fit_suffix,
                    customdata=custom,
                    hovertemplate=(
                        "method="
                        + method
                        + "<br>FLOPs=%{customdata[0]:.1e}"
                        + "<br>tokens=%{customdata[1]:.2e}"
                        + "<br>loss=%{y:.4f}"
                        + "<br>epochs=%{customdata[2]:.2g}x<extra></extra>"
                    ),
                )
            )

            if mode == "projections":
                projection_form = None
                E_fit = A_fit = alpha_fit = 0.0  # type: ignore[assignment]
                forecast_label = ""
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
                            hovertemplate=(
                                "forecast "
                                + method
                                + " ("
                                + projection_form
                                + ")<br>"
                                + spec["hover_label"]
                                + "=%{x:.2e}<br>loss=%{y:.4f}<extra></extra>"
                            ),
                            showlegend=True,
                        )
                    )
            elif mode in ("one_epoch", "all_epochs"):
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
            title = f"Training {_clean_params_label(params)} model on data from {n_warcs} WARC files"
            yaxis_title = _clean_metric_label(metric_key)
        else:
            title = f"Method comparison at {label}, N={n_warcs} WARCs vs {x_axis}{mode_tag} -- {EXPERIMENT_TAG}"
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
        out = n_dir / f"method_comparison_{safe_label}_x_{x_axis}{suffix}.html"
        fig.write_html(str(out))
        logger.info("Wrote %s", out)


def plot_grid_overview(
    records: list[WarcScalingRecord],
    metric_key: str,
    output_dir: Path,
    x_axis: str = "tokens",
    mode: str = "all_epochs",
) -> None:
    """One big subplot grid per metric: rows = N, cols = model size.

    157M (h=512) is guaranteed to be a column even if no records exist for it
    in some N row — it's the cross-N anchor we care about visually.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    spec = _X_AXIS_SPECS[x_axis]
    x_attr = spec["attr"]

    by_cell: dict[tuple[int, int], list[WarcScalingRecord]] = {}
    for r in records:
        by_cell.setdefault((r.hidden_dim, r.sampled_warcs), []).append(r)

    # Columns: union of PLANNED hidden_dims for the WARC sweep (so empty-but-
    # expected cells are visible placeholders, not silently dropped) PLUS any
    # observed dim from the FM sweep (which used a subset of TARGET_HIDDEN_SIZES).
    # 157M (h=512) is always present by virtue of being in every N's plan.
    planned_cols: set[int] = set()
    for n in WARC_COUNTS:
        planned_cols.update(hidden_sizes_for(n))
    observed_cols = {h for (h, _) in by_cell}
    cols = sorted(planned_cols | observed_cols)
    # Rows: planned N values (subsamples + 3000 anchor) plus any observed.
    planned_rows = set(WARC_COUNTS) | {3000}
    rows = sorted(planned_rows | {n for (_, n) in by_cell})
    if not rows or not cols:
        logger.info("Grid skipped (no records) for metric=%s mode=%s x=%s", metric_key, mode, x_axis)
        return

    # Subplot titles per (row, col).
    n_cols = len(cols)
    n_rows = len(rows)
    titles: list[str] = []
    for n_warcs in rows:
        for hidden_dim in cols:
            cell = by_cell.get((hidden_dim, n_warcs), [])
            params = cell[0].params if cell else 0
            label = _params_label(hidden_dim, params) if params else f"d={hidden_dim}"
            titles.append(f"N={n_warcs} | {label}")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
    )

    seen_methods: set[str] = set()
    for ri, n_warcs in enumerate(rows):
        for ci, hidden_dim in enumerate(cols):
            cell = by_cell.get((hidden_dim, n_warcs), [])
            if not cell:
                continue
            by_method: dict[str, list[WarcScalingRecord]] = {}
            for r in cell:
                by_method.setdefault(r.method, []).append(r)
            for method, method_records in sorted(by_method.items()):
                pts = sorted(method_records, key=lambda r: getattr(r, x_attr))
                color = COMPARE_COLORS.get(method, "#333")
                x_list = [getattr(p, x_attr) for p in pts]
                loss_list = [p.loss for p in pts]
                epochs_list = [p.epochs_over_d_obs for p in pts]
                show_legend = method not in seen_methods
                seen_methods.add(method)
                fig.add_trace(
                    go.Scatter(
                        x=x_list,
                        y=loss_list,
                        mode="lines+markers",
                        marker=dict(size=6, color=color),
                        line=dict(color=color, width=1.5),
                        name=method,
                        legendgroup=method,
                        showlegend=show_legend,
                        customdata=list(zip(x_list, loss_list, epochs_list, strict=True)),
                        hovertemplate=(
                            f"method={method}"
                            + f"<br>N={n_warcs}, hidden={hidden_dim}"
                            + "<br>x=%{x:.2e}<br>loss=%{y:.4f}"
                            + "<br>epochs=%{customdata[2]:.2g}x<extra></extra>"
                        ),
                    ),
                    row=ri + 1,
                    col=ci + 1,
                )
                if mode == "all_epochs":
                    ones = [getattr(p, x_attr) / p.epochs_over_d_obs for p in pts if p.epochs_over_d_obs > 0]
                    if ones:
                        x_one_epoch = sum(ones) / len(ones)
                        max_eps = max(p.epochs_over_d_obs for p in pts)
                        for k in range(1, int(max_eps) + 2):
                            x_k = k * x_one_epoch
                            opac = 0.7 if k == 1 else max(0.12, 0.5 / (k**0.5))
                            fig.add_trace(
                                go.Scatter(
                                    x=[x_k, x_k],
                                    y=[min(loss_list) * 0.95, max(loss_list) * 1.05],
                                    mode="lines",
                                    line=dict(color=color, dash="dash", width=1.0 if k == 1 else 0.6),
                                    opacity=opac,
                                    showlegend=False,
                                    hoverinfo="skip",
                                    legendgroup=method,
                                ),
                                row=ri + 1,
                                col=ci + 1,
                            )

    # Log x-axis on every subplot
    for ri in range(1, n_rows + 1):
        for ci in range(1, n_cols + 1):
            fig.update_xaxes(type="log", row=ri, col=ci)

    suffix_tag = {
        "all_epochs": " (all-epoch markers)",
        "one_epoch": " (1-epoch markers)",
        "projections": " (projections)",
    }.get(mode, "")
    fig.update_layout(
        template="plotly_white",
        title=f"WARC-scaling grid -- {metric_key} vs {x_axis}{suffix_tag}",
        width=300 * n_cols + 200,
        height=260 * n_rows + 120,
        showlegend=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"one_epoch": "_one_epoch", "all_epochs": "_all_epochs"}.get(mode, "")
    out = output_dir / f"grid_overview_x_{x_axis}{suffix}.html"
    fig.write_html(str(out))
    logger.info("Wrote %s", out)


def _per_n_size_lineup(n_warcs: int) -> list[int]:
    """Return the sorted list of planned hidden_sizes for this N (3000 = FM trio)."""
    if n_warcs == 3000:
        # Fixed-model sweep used 512/1536/2432 (subset of TARGET_HIDDEN_SIZES).
        return [512, 1536, 2432]
    if n_warcs == TENK_N:
        # 10k natural-epoch sweep widths (launch_10k_natural.WIDTHS). Explicit
        # branch: hidden_sizes_for() only knows the WARC_COUNTS subsamples and
        # would raise for 10364.
        return list(TENK_WIDTHS)
    return sorted(hidden_sizes_for(n_warcs))


def plot_grid_compressed(
    records: list[WarcScalingRecord],
    metric_key: str,
    output_dir: Path,
    x_axis: str = "tokens",
    mode: str = "all_epochs",
    log_y: bool = False,
) -> None:
    """Compressed grid: rows = N, cols = position relative to 157M anchor.

    Unlike `plot_grid_overview` which aligns by absolute hidden_dim, this view
    packs each row tightly: 157M is always in the center column, smaller models
    (e.g. 69M, 105M) line up to its left in size order, bigger models (e.g.
    273M, 1B, 1.93B, 2.9B) to its right. Different N rows may show different
    absolute sizes in the same column — that's the point: scan a row to see all
    that N's data, scan the 157M column to see fixed-arch comparison across N.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    spec = _X_AXIS_SPECS[x_axis]
    x_attr = spec["attr"]

    by_cell: dict[tuple[int, int], list[WarcScalingRecord]] = {}
    for r in records:
        by_cell.setdefault((r.hidden_dim, r.sampled_warcs), []).append(r)

    rows = sorted(set(WARC_COUNTS) | {3000} | {n for (_, n) in by_cell})
    if not rows:
        return

    # Per-row planned hidden_dims (sorted small-to-big, with 157M anchor ensured).
    per_row_planned: dict[int, list[int]] = {}
    all_sizes: set[int] = set()
    for n in rows:
        sizes = sorted(set(_per_n_size_lineup(n)) | {512})
        per_row_planned[n] = sizes
        all_sizes.update(sizes)

    # Greedy compact assignment: each unique hidden_dim placed at the smallest
    # column index >= every participating row's next-free col. Same hidden_dim
    # always shares one column across rows (so 1B-class d=1536 aligns across
    # N=100, N=1000, N=3000). No bubble pass — earlier versions had one but it
    # cascaded and pushed d=1536 out of alignment, so we keep just the greedy
    # output, which happens to align same-size cells by construction.
    next_col_per_row: dict[int, int] = {n: 0 for n in rows}
    cell_size: dict[tuple[int, int], int] = {}
    row_pos: dict[int, dict[int, int]] = {n: {} for n in rows}
    for h in sorted(all_sizes):
        rows_with_h = [n for n in rows if h in per_row_planned[n]]
        chosen = max(next_col_per_row[n] for n in rows_with_h)
        for n in rows_with_h:
            cell_size[(n, chosen)] = h
            row_pos[n][h] = chosen
            next_col_per_row[n] = chosen + 1

    n_cols = (max(c for (_, c) in cell_size) + 1) if cell_size else 1
    n_rows = len(rows)

    titles: list[str] = []
    for n in rows:
        for ci in range(n_cols):
            h = cell_size.get((n, ci))
            if h is None:
                titles.append("")
            else:
                example = next((r for r in records if r.hidden_dim == h), None)
                params = example.params if example else 0
                if params:
                    titles.append(f"N={n} | {_params_label(h, params)}")
                else:
                    titles.append(f"N={n} | d={h}")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
    )

    seen_methods: set[str] = set()
    for ri, n in enumerate(rows):
        for ci in range(n_cols):
            h = cell_size.get((n, ci))
            if h is None:
                fig.update_xaxes(visible=False, row=ri + 1, col=ci + 1)
                fig.update_yaxes(visible=False, row=ri + 1, col=ci + 1)
                continue
            cell = by_cell.get((h, n), [])
            if not cell:
                continue
            by_method: dict[str, list[WarcScalingRecord]] = {}
            for r in cell:
                by_method.setdefault(r.method, []).append(r)
            for method, method_records in sorted(by_method.items()):
                pts = sorted(method_records, key=lambda r: getattr(r, x_attr))
                color = COMPARE_COLORS.get(method, "#333")
                x_list = [getattr(p, x_attr) for p in pts]
                loss_list = [p.loss for p in pts]
                epochs_list = [p.epochs_over_d_obs for p in pts]
                show_legend = method not in seen_methods
                seen_methods.add(method)
                fig.add_trace(
                    go.Scatter(
                        x=x_list,
                        y=loss_list,
                        mode="lines+markers",
                        marker=dict(size=6, color=color),
                        line=dict(color=color, width=1.5),
                        name=method,
                        legendgroup=method,
                        showlegend=show_legend,
                        customdata=list(zip(x_list, loss_list, epochs_list, strict=True)),
                        hovertemplate=(
                            f"method={method}"
                            + f"<br>N={n}, hidden={h}"
                            + "<br>x=%{x:.2e}<br>loss=%{y:.4f}"
                            + "<br>epochs=%{customdata[2]:.2g}x<extra></extra>"
                        ),
                    ),
                    row=ri + 1,
                    col=ci + 1,
                )
                if mode == "all_epochs":
                    ones = [getattr(p, x_attr) / p.epochs_over_d_obs for p in pts if p.epochs_over_d_obs > 0]
                    if ones:
                        x_one_epoch = sum(ones) / len(ones)
                        max_eps = max(p.epochs_over_d_obs for p in pts)
                        for k in range(1, int(max_eps) + 2):
                            x_k = k * x_one_epoch
                            opac = 0.7 if k == 1 else max(0.12, 0.5 / (k**0.5))
                            fig.add_trace(
                                go.Scatter(
                                    x=[x_k, x_k],
                                    y=[min(loss_list) * 0.95, max(loss_list) * 1.05],
                                    mode="lines",
                                    line=dict(color=color, dash="dash", width=1.0 if k == 1 else 0.6),
                                    opacity=opac,
                                    showlegend=False,
                                    hoverinfo="skip",
                                    legendgroup=method,
                                ),
                                row=ri + 1,
                                col=ci + 1,
                            )

    for ri, n in enumerate(rows, start=1):
        for ci in range(1, n_cols + 1):
            if cell_size.get((n, ci - 1)) is not None:
                fig.update_xaxes(type="log", row=ri, col=ci)
                if log_y:
                    fig.update_yaxes(type="log", row=ri, col=ci)

    suffix_tag = {
        "all_epochs": " (all-epoch markers)",
        "one_epoch": " (1-epoch markers)",
        "projections": " (projections)",
    }.get(mode, "")
    log_y_tag = " [log y]" if log_y else ""
    fig.update_layout(
        template="plotly_white",
        title=(
            f"WARC-scaling COMPRESSED grid (planned cells, hidden_dims aligned) "
            f"-- {metric_key} vs {x_axis}{suffix_tag}{log_y_tag}"
        ),
        width=220 * n_cols + 140,
        # 50% compressed + 15% breathing room added back
        height=125 * n_rows + 90,
        showlegend=True,
        margin=dict(l=40, r=40, t=70, b=30),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"one_epoch": "_one_epoch", "all_epochs": "_all_epochs"}.get(mode, "")
    log_y_suffix = "_logy" if log_y else ""
    out = output_dir / f"grid_compressed_x_{x_axis}{suffix}{log_y_suffix}.html"
    fig.write_html(str(out))
    logger.info("Wrote %s", out)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument(
        "--fm-results-prefix",
        default=FM_RESULTS_PREFIX,
        help="Fixed-model (3000-WARC) sweep results to merge in as the N=3000 row. " "Pass empty string to disable.",
    )
    parser.add_argument(
        "--fm-lima-sidecar-prefix",
        default=FM_LIMA_SIDECAR_PREFIX,
        help="LIMA sidecar for older FM runs missing eval/lima/* in the main summary. "
        "Merged in by run_name. Pass empty string to disable.",
    )
    parser.add_argument(
        "--tenk-results-prefix",
        default=TENK_RESULTS_PREFIX,
        help="10k natural-epoch (launch_10k_natural.py) results to merge in as the "
        f"N={TENK_N} row. Routed through the FM path (shared expFM_natural tag). "
        "Pass empty string to disable.",
    )
    parser.add_argument("--output-dir", default="scratch/plots/warc_scaling")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[DEFAULT_METRIC, "eval/lima/loss", "eval/macro_bpb", "uncheatable_macro_loss"],
        help="Eval metrics to plot. One subdir per metric.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["all_epochs", "projections"],
        choices=["projections", "one_epoch", "all_epochs"],
    )
    parser.add_argument("--clean-titles", action="store_true")
    args = parser.parse_args(argv)

    summaries = load_summaries(args.results_prefix, _all_method_keys(), suffix="")
    fm_summaries: list[dict] = []
    if args.fm_results_prefix:
        fm_summaries = load_summaries(
            args.fm_results_prefix,
            list(_FM_METHOD_MAP.keys()),
            suffix="",
        )
        logger.info("Loaded %d fixed-model summaries (3000-WARC anchor)", len(fm_summaries))
        # Merge LIMA sidecar (older FM runs were missing eval/lima/* inline)
        sidecar_metrics = _load_fm_lima_sidecar(args.fm_lima_sidecar_prefix)
        for s in fm_summaries:
            _merge_sidecar_into_summary(s, sidecar_metrics)
    if args.tenk_results_prefix:
        # 10k natural-epoch summaries: same expFM_natural tag + _FM_METHOD_MAP
        # entries (dclm_10k / nemotron_10k), so they go through fm_summary_to_record
        # into the N=10364 row. New runs -> eval/lima is inline, no sidecar needed.
        tenk_summaries = load_summaries(args.tenk_results_prefix, list(TENK_METHOD_KEYS), suffix="")
        logger.info("Loaded %d 10k natural-epoch summaries (N=%d)", len(tenk_summaries), TENK_N)
        fm_summaries.extend(tenk_summaries)
    output_root = Path(args.output_dir)

    for metric_key in args.metrics:
        records: list[WarcScalingRecord] = []
        for s in summaries:
            r = summary_to_record(s, metric_key)
            if r is not None:
                records.append(r)
        for s in fm_summaries:
            r = fm_summary_to_record(s, metric_key)
            if r is not None:
                records.append(r)
        logger.info("Metric %s: %d records", metric_key, len(records))
        if not records:
            continue
        # Sub-dir per metric, sanitized.
        metric_dir = output_root / metric_key.replace("/", "_")
        for mode in args.modes:
            for x_axis in ("flops", "tokens"):
                plot_method_comparison_per_size_and_n(
                    records, metric_key, metric_dir, x_axis=x_axis, mode=mode, clean_titles=args.clean_titles
                )
                plot_grid_overview(records, metric_key, metric_dir, x_axis=x_axis, mode=mode)
                plot_grid_compressed(records, metric_key, metric_dir, x_axis=x_axis, mode=mode)
                plot_grid_compressed(records, metric_key, metric_dir, x_axis=x_axis, mode=mode, log_y=True)


if __name__ == "__main__":
    main()
