# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot IsoFLOP curves from the data-curation sweep results.

Reads per-run summary.json files written by run_curation_train_standalone,
transforms them into IsoFlopRecords, fits scaling laws, and produces
interactive Plotly HTML plots — one set per experiment (A, B, or C).

Auto-detects experiment_tags present in the loaded summaries (so ExpC
summaries are picked up automatically once they land alongside ExpA/B).

Usage:
    # Plot all completed DCLM runs (ExpA + ExpB + any ExpC):
    uv run python experiments/scaling_law_sweeps/plot_curation_isoflop.py \
        --methods dclm

    # Just ExpC for the data-rich methods (e.g. for overnight checkpointing):
    uv run python experiments/scaling_law_sweeps/plot_curation_isoflop.py \
        --methods llm_curated_bos_fixed resiliparse \
        --experiments expC_T33T

    # Local download for fast iteration (avoids re-reading from GCS):
    gcloud storage cp 'gs://marin-us-central1/metadata/data_curation_isoflop_results/curation-*expC_T33T*' /tmp/expc/
    uv run python experiments/scaling_law_sweeps/plot_curation_isoflop.py \
        --methods llm_curated_bos_fixed resiliparse \
        --experiments expC_T33T \
        --results-prefix /tmp/expc/ \
        --output-dir plots/curation_isoflop_expc

    # Just dump the records as CSV (for custom plotting):
    uv run python experiments/scaling_law_sweeps/plot_curation_isoflop.py \
        --methods dclm --csv-only

Note on ExpC fits: fitting a quadratic-in-log-tokens IsoFLOP minimum requires
≥3 distinct hidden_dim values at each compute budget. The pre-copied
fixed-model summaries only cover hidden_dim ∈ {512, 1536} for many budgets,
so per-method IsoFLOP fits won't converge until the live ExpC runs land
(which fill in the remaining hidden_dim points). Scatter/CSV outputs are
produced regardless; fit JSONs will be empty until enough points land.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import fsspec
import pandas as pd
from marin.scaling_laws.isoflop_analysis import IsoFlopRecord, fit_scaling_laws
from marin.scaling_laws.scaling_plots import create_isoflop_plot, create_scaling_plot

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_isoflop_results/"
DEFAULT_METRIC = "eval/paloma/bpb"

# Display labels for known experiment_tags. New tags fall back to the raw value
# in titles / filenames (still works, just less pretty).
EXPERIMENT_TAG_LABELS: dict[str, str] = {
    "expA_natural": "ExpA_natural",
    "expB_T20T": "ExpB_T20T",
    "expC_T33T": "ExpC_T33T",
}


def _label_for_tag(tag: str) -> str:
    return EXPERIMENT_TAG_LABELS.get(tag, tag)


def _budget_key(budget_flops: float) -> int:
    """Stable integer FLOPs key (units of 1e18), robust to JSON float round-trips."""
    return round(budget_flops / 1e18)


# Per-method batch selection at the d512/1.8e19 (157M) cell, which exists at both B64
# and B128 (same budget/tokens/epochs, different optimization). We DROP the non-chosen
# batch so each method contributes one point:
#   - shallow methods (dclm / high_quality / fineweb_cc / resiliparse): keep B64 -- the
#     smaller batch fixes the off-trend optimization artifact the natural B128 point had
#     (B128 sat at/above its 9e18 neighbor for high_quality/fineweb_cc).
#   - deep-epoch methods (nemotron, fineweb_edu): keep B128 -- in the overtrained regime
#     the larger batch is the better / more-stable point (B64 overfits slightly more).
# Keyed by (method_name, hidden_dim, budget//1e18, batch_size) -> DROP.
SUPERSEDED_CELLS: set[tuple[str, int, int, int]] = {
    ("dclm_10k", 512, 18, 128),
    ("high_quality_10k", 512, 18, 128),
    ("fineweb_cc_10k", 512, 18, 128),
    ("resiliparse_10k", 512, 18, 128),
    ("nemotron_10k", 512, 18, 64),
    ("fineweb_edu_10k", 512, 18, 64),
}


def _apply_result_overrides(summaries: list[dict]) -> list[dict]:
    """Drop summaries for cells listed in SUPERSEDED_CELLS (no-op for everything else).

    Plot consumers key cells by (method, hidden, budget) and never read batch, so a
    superseded cell and its replacement would otherwise both plot as points. Removing
    the superseded summary leaves only the replacement.
    """
    kept: list[dict] = []
    for s in summaries:
        plan = s.get("plan", {})
        budget = plan.get("budget_flops")
        key = (
            plan.get("method_name"),
            plan.get("hidden_dim"),
            _budget_key(budget) if budget is not None else None,
            plan.get("batch_size"),
        )
        if key in SUPERSEDED_CELLS:
            logger.info("Dropping superseded cell %s (run=%s)", key, plan.get("run_name"))
            continue
        kept.append(s)
    return kept


def load_summaries(
    results_prefix: str,
    methods: list[str],
    suffix: str,
) -> list[dict]:
    """Load all summary.json files matching the given methods + suffix.

    Supports both gs:// and local paths. For gs://, uses fsspec; for local
    paths (e.g. /tmp/curation_summaries/), uses pathlib directly.
    """
    summaries = []
    if results_prefix.startswith("gs://"):
        fs, prefix = fsspec.core.url_to_fs(results_prefix.rstrip("/"))
        all_files = fs.ls(prefix)
        for fpath in all_files:
            fname = fpath.split("/")[-1]
            if not fname.endswith(".json"):
                continue
            if suffix and f"-{suffix}.json" not in fname:
                continue
            method_match = any(f"curation-{m}-" in fname for m in methods)
            if not method_match:
                continue
            try:
                with fs.open(fpath, "r") as f:
                    summaries.append(json.load(f))
            except Exception as e:
                logger.warning("Failed to read %s: %s", fpath, e)
    else:
        local = Path(results_prefix)
        for fpath in sorted(local.glob("*.json")):
            fname = fpath.name
            if suffix and f"-{suffix}.json" not in fname:
                continue
            method_match = any(f"curation-{m}-" in fname for m in methods)
            if not method_match:
                continue
            try:
                summaries.append(json.loads(fpath.read_text()))
            except Exception as e:
                logger.warning("Failed to read %s: %s", fpath, e)
    summaries = _apply_result_overrides(summaries)
    logger.info("Loaded %d summaries for methods=%s suffix=%s", len(summaries), methods, suffix)
    return summaries


PALOMA_DATASETS = [
    "4chan",
    "c4_100_domains",
    "c4_en",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
]


def _compute_paloma_macro_loss(eval_metrics: dict) -> float | None:
    """Macro-average loss across the 16 paloma validation datasets."""
    losses = []
    for ds in PALOMA_DATASETS:
        val = eval_metrics.get(f"eval/paloma/{ds}/loss")
        if val is not None:
            losses.append(float(val))
    if not losses:
        return None
    return sum(losses) / len(losses)


# Uncheatable_eval validation datasets (delphi's canonical 7).
UNCHEATABLE_EVAL_DATASETS = [
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
]


def _compute_uncheatable_macro_loss(eval_metrics: dict) -> float | None:
    """Macro-average loss across the 7 uncheatable_eval validation datasets.

    Prefer the precomputed `eval/uncheatable_eval/macro_loss` if present;
    otherwise fall back to averaging per-dataset losses ourselves so older
    summaries (or any summary missing the precomputed key) still work.
    """
    precomputed = eval_metrics.get("eval/uncheatable_eval/macro_loss")
    if precomputed is not None:
        return float(precomputed)
    losses = []
    for ds in UNCHEATABLE_EVAL_DATASETS:
        val = eval_metrics.get(f"eval/uncheatable_eval/{ds}/loss")
        if val is not None:
            losses.append(float(val))
    if not losses:
        return None
    return sum(losses) / len(losses)


# --- Outlier filter (tail-trim + LOO + absolute floor) -----------------------
# Default abs-residual floor per metric, calibrated from inspection of ExpC
# T=33T data so that ~4-8 visibly-off-curve points get dropped without
# eating LC's flatter U-curves. User can override via CLI.
_DEFAULT_OUTLIER_FLOOR_BY_METRIC: dict[str, float] = {
    "eval/paloma/bpb": 0.04,
    "paloma_macro_loss": 0.07,
    "uncheatable_macro_loss": 0.07,
    "eval/lima/loss": 0.10,
    "eval/lima/bpb": 0.04,
}


def _huber_quad_logx(L_arr, y_arr, delta: float = 0.05) -> tuple[float, float, float]:
    """Pure-numpy Huber quadratic fit in (L, y) — minimal stand-alone version
    for outlier filtering. Returns (a, b, c) for y = a*L^2 + b*L + c.
    """
    import numpy as np
    from scipy.optimize import minimize

    L_arr = np.asarray(L_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)
    init = np.polyfit(L_arr, y_arr, 2)

    def obj(p):
        a, b, c = p
        r = y_arr - (a * L_arr**2 + b * L_arr + c)
        absr = np.abs(r)
        return np.sum(np.where(absr <= delta, 0.5 * r**2, delta * (absr - 0.5 * delta)))

    res = minimize(obj, init, method="BFGS")
    a, b, c = res.x
    return float(a), float(b), float(c)


def detect_outlier_runs(
    summaries: list[dict],
    metric_key: str,
    abs_floor: float | None = None,
    drop_tails_each: int = 2,
) -> set[str]:
    """Identify per-(method, FLOPs) outliers by run_name.

    Hybrid filter:
      - For groups with n>=8: drop 2 leftmost+rightmost by tokens, fit a Huber
        parabola to the interior, compute residuals of all points against that
        interior fit. A point is an outlier iff its residual exceeds `abs_floor`
        (above the curve — only upward outliers count).
      - For groups with 4<=n<8: leave-one-out residuals; drop points whose
        upward LOO residual exceeds `abs_floor`.
      - n<4: no filtering.

    Why "fit-to-interior": pure LOO can't detect coherent tail-clusters because
    the parabola happily accommodates an upward-bending tail. Trimming the tails
    BEFORE fitting establishes the smooth U-curve baseline so tail bends become
    detectable as residuals.

    Returns the set of `plan.run_name` values to exclude.
    """
    import numpy as np

    if abs_floor is None:
        abs_floor = _DEFAULT_OUTLIER_FLOOR_BY_METRIC.get(metric_key, 0.07)

    # Build records keyed by (method, FLOPs)
    records: list[tuple[str, str, float, float, float]] = []
    for s in summaries:
        plan = s.get("plan", {})
        run_name = plan.get("run_name")
        method = plan.get("method_name")
        flops = plan.get("budget_flops")
        tokens = (s.get("tokens") or {}).get("tokens_trained")
        e = s.get("eval") or {}
        if metric_key == "paloma_macro_loss":
            metric = _compute_paloma_macro_loss(e)
        elif metric_key == "uncheatable_macro_loss":
            metric = _compute_uncheatable_macro_loss(e)
        else:
            metric = e.get(metric_key)
        if metric is None or run_name is None or tokens is None or flops is None:
            continue
        records.append((run_name, method, float(flops), float(tokens), float(metric)))

    by_group: dict[tuple[str, float], list] = defaultdict(list)
    for rec in records:
        by_group[(rec[1], rec[2])].append(rec)

    drop_run_names: set[str] = set()
    for (lab, C), grp in by_group.items():
        # Iteratively drop and refit until no more outliers are detected.
        # Necessary because a coherent multi-point tail-cluster (e.g. DCLM 9e+19
        # has ~5 undertrained corners that all bend up together) survives the
        # one-shot fit-to-interior since the second-tier outliers move into
        # the "interior" once the worst is trimmed. Iterating peels off one
        # ring at a time and each pass exposes the next.
        current = list(grp)
        for _iteration in range(20):  # safety cap
            n = len(current)
            srt = sorted(current, key=lambda r: r[3])
            this_pass_drops = []
            if n >= 2 * drop_tails_each + 4:
                interior = srt[drop_tails_each : n - drop_tails_each]
                L_int = np.log10([r[3] for r in interior])
                y_int = np.array([r[4] for r in interior])
                a, b, c = _huber_quad_logx(L_int, y_int)
                for r in srt:
                    Li = np.log10(r[3])
                    pred = a * Li**2 + b * Li + c
                    resid = r[4] - pred
                    if resid > abs_floor:
                        this_pass_drops.append(r)
            elif n >= 4:
                # LOO fallback: leave each out, fit through others.
                L_all = np.log10([r[3] for r in srt])
                y_all = np.array([r[4] for r in srt])
                for i in range(n):
                    others_L = np.delete(L_all, i)
                    others_y = np.delete(y_all, i)
                    a, b, c = _huber_quad_logx(others_L, others_y)
                    pred = a * L_all[i] ** 2 + b * L_all[i] + c
                    resid = y_all[i] - pred
                    if resid > abs_floor:
                        this_pass_drops.append(srt[i])
            else:
                break
            if not this_pass_drops:
                break
            for r in this_pass_drops:
                drop_run_names.add(r[0])
            drop_set = {id(r) for r in this_pass_drops}
            current = [r for r in current if id(r) not in drop_set]
            if len(current) < 4:
                break
    return drop_run_names


def summary_to_record(summary: dict, metric_key: str = DEFAULT_METRIC) -> IsoFlopRecord | None:
    """Transform a summary.json dict into an IsoFlopRecord."""
    plan = summary.get("plan", {})
    model = summary.get("model", {})
    tokens_info = summary.get("tokens", {})
    eval_metrics = summary.get("eval", {})

    if metric_key == "paloma_macro_loss":
        metric_val = _compute_paloma_macro_loss(eval_metrics)
    elif metric_key == "uncheatable_macro_loss":
        metric_val = _compute_uncheatable_macro_loss(eval_metrics)
    else:
        metric_val = eval_metrics.get(metric_key)
    if metric_val is None:
        logger.warning("No metric %s in %s", metric_key, plan.get("run_name", "?"))
        return None

    return IsoFlopRecord(
        tokens=float(tokens_info.get("tokens_trained", 0)),
        metric=float(metric_val),
        flops=float(plan.get("budget_flops", 0)),
        params=float(model.get("total_trainable_params", 0)),
        label=plan.get("method_name", "unknown"),
    )


def build_dataframe(records: list[IsoFlopRecord]) -> pd.DataFrame:
    """Convert IsoFlopRecords to the DataFrame format expected by scaling_plots."""
    rows = []
    for r in records:
        rows.append(
            {
                "tokens": r.tokens,
                "loss": r.metric,
                "flops": r.flops,
                "params": r.params,
                "label": r.label,
                "name": f"{r.label}-{r.flops:.0e}-{r.params:.0e}",
            }
        )
    return pd.DataFrame(rows)


def plot_experiment(
    summaries: list[dict],
    experiment_tag: str,
    metric_key: str,
    output_dir: Path,
    title_suffix: str = "",
) -> pd.DataFrame | None:
    """Filter summaries for one experiment, fit scaling laws, produce plots."""
    filtered = [s for s in summaries if s.get("plan", {}).get("experiment_tag") == experiment_tag]
    if not filtered:
        logger.info("No summaries for experiment %s", experiment_tag)
        return None

    records = [r for s in filtered if (r := summary_to_record(s, metric_key)) is not None]
    if len(records) < 3:
        logger.warning("Only %d records for %s — need ≥3 for fitting", len(records), experiment_tag)
        return None

    logger.info(
        "Experiment %s: %d records across %d budgets", experiment_tag, len(records), len(set(r.flops for r in records))
    )

    df = build_dataframe(records)

    try:
        fit_result = fit_scaling_laws(records)
        fig_isoflop = create_isoflop_plot(df, fit_result.minima_records, fit_result.fit_curves)
        fig_scaling = create_scaling_plot(fit_result.minima_records, fit_result.scaling_fits)
    except Exception as e:
        logger.warning("Fitting failed for %s: %s — generating scatter-only plot", experiment_tag, e)
        fit_result = None
        fig_isoflop = create_isoflop_plot(df, [], {})
        fig_scaling = None

    exp_label = _label_for_tag(experiment_tag)
    methods_in_data = sorted(df.label.unique())
    methods_tag = "+".join(methods_in_data)
    tag = f"{exp_label} — {methods_tag}{title_suffix}"
    file_prefix = f"{methods_tag}_{experiment_tag}"

    fig_isoflop.update_layout(title=f"IsoFLOP — {tag}", xaxis_title="Tokens trained", yaxis_title=metric_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    isoflop_path = output_dir / f"isoflop_{file_prefix}.html"
    fig_isoflop.write_html(str(isoflop_path))
    logger.info("Wrote %s", isoflop_path)

    if fig_scaling is not None:
        scaling_path = output_dir / f"scaling_{file_prefix}.html"
        fig_scaling.update_layout(title=f"Scaling Law Fit — {tag}")
        fig_scaling.write_html(str(scaling_path))
        logger.info("Wrote %s", scaling_path)

    csv_path = output_dir / f"records_{file_prefix}.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("Wrote %s", csv_path)

    if fit_result is not None:
        fit_path = output_dir / f"fit_{file_prefix}.json"
        fit_path.write_text(
            json.dumps(
                {
                    "minima": [
                        {
                            "label": m.label,
                            "flops": m.flops,
                            "optimal_tokens": m.optimal_tokens,
                            "optimal_params": m.optimal_params,
                            "loss_at_optimal": m.loss_at_optimal,
                        }
                        for m in fit_result.minima_records
                    ],
                    "scaling_fits": {k: {"alpha": v.alpha, "A": v.A} for k, v in fit_result.scaling_fits.items()},
                },
                indent=2,
            )
        )
        logger.info("Wrote %s", fit_path)

    return df


COMPARE_COLORS = {
    "dclm": "#1f77b4",
    "nemotron_org": "#ff7f0e",
    # nemotron_full: was green #2ca02c — moved to black 2026-05-11 to free
    # green for high_quality (the natural "good data" color).
    "nemotron_full": "#000000",
    # FineWeb family (10k natural-epoch). fineweb_edu's old #d62728 red is now
    # owned by med_low_quality, so use a pink/salmon pair distinct from it and
    # from low_quality's #e91e63.
    "fineweb_edu": "#e377c2",  # magenta-pink (FineWeb-Edu)
    "fineweb_cc": "#ff9896",  # salmon (full FineWeb-CC)
    "resiliparse": "#9467bd",
    # Per-N fuzzy-deduped resiliparse — same hue, darker shade so it sits
    # next to the non-deduped curve on cross-method plots.
    "resiliparse_dedup": "#5d3a8a",
    "llm_curated": "#8c564b",
    # BOS-fixed rebuilds: same base hue as the broken twin, darker shade to
    # separate visually from the original on the legend.
    "nemotron_full_bos_fixed": "#14571c",
    "llm_curated_bos_fixed": "#4a2b22",
    # Nemotron-CC-HQ (quality=high subset, ~30% of full Nemotron-CC).
    # Goldenrod = "premium / high quality" + warm to stay in nemotron family,
    # distinct from nemotron_full's black on cross-method plots.
    "nemotron_qhigh": "#B8860B",
    # Deduped llm_curated — distinct teal so it pops next to the brown
    # bos_fixed twin on cross-method comparisons.
    "llm_curated_dedup": "#17becf",
    # DCLM-faithful curation on llm_curated extraction (full 200-shard corpus).
    "llm_curated_dclm_filtered": "#bcbd22",
    # ExpC 10k re-extractions: keep the base-method hue but use a saturated
    # primary so the 4 ExpC methods (these two + LC bos_fixed + resiliparse)
    # are all visually distinct from each other on the cross-method comparison.
    "dclm_10k": "#1f77b4",  # dclm blue
    "nemotron_10k": "#2ca02c",  # nemotron green
    # LLM-extracted quality bands — low→high quality gradient using freed
    # palette slots (fineweb_edu was dropped; nemotron_full moved to black
    # so its green could host high_quality).
    "low_quality": "#e91e63",  # pink
    "med_low_quality": "#d62728",  # red (was fineweb_edu)
    "med_quality": "#ff7f0e",  # orange (was nemotron_org, deprecated)
    "high_quality": "#2ca02c",  # green (was nemotron_full, now black)
}


def _fit_loss_vs_x(x_vals: list[float], loss_vals: list[float]) -> tuple[float, float] | None:
    """Fit loss = A * x^(-beta) via linear regression in log-log space.

    Returns (A, beta) or None if fit fails.
    """
    import numpy as np

    if len(x_vals) < 2:
        return None
    log_x = np.log(x_vals)
    log_l = np.log(loss_vals)
    # log(L) = log(A) - beta * log(x) → linear fit
    coeffs = np.polyfit(log_x, log_l, 1)
    beta = -coeffs[0]
    A = np.exp(coeffs[1])
    return A, beta


# How far to project the forecast lines
PROJECTION_MAX_TOKENS = 1e14
PROJECTION_MAX_FLOPS = 1e27


def plot_comparison(
    summaries: list[dict],
    experiment_tag: str,
    metric_key: str,
    output_dir: Path,
) -> None:
    """Cross-method comparison plots with projected forecast lines.

    Produces three plots, each with observed D* points + dashed forecast:
      1. D* scatter: x=optimal_tokens (log), y=loss, per method + forecast to 1e15 tokens
      2. Scaling fit: x=compute (log), y=D* (log), per method + forecast to 1e23 FLOPs
      3. Loss frontier: x=compute (log), y=loss, per method + forecast to 1e23 FLOPs
    """
    import numpy as np
    import plotly.graph_objects as go

    filtered = [s for s in summaries if s.get("plan", {}).get("experiment_tag") == experiment_tag]
    if not filtered:
        return

    by_method: dict[str, list[dict]] = {}
    for s in filtered:
        m = s["plan"]["method_name"]
        by_method.setdefault(m, []).append(s)

    all_minima: list[tuple[str, object]] = []
    all_fits: dict[str, object] = {}  # D* = A * C^alpha
    loss_vs_tokens_fits: dict[str, tuple[float, float]] = {}  # L = A * D*^(-beta)
    loss_vs_compute_fits: dict[str, tuple[float, float]] = {}  # L = A * C^(-beta)

    for method, method_sums in sorted(by_method.items()):
        records = [r for s in method_sums if (r := summary_to_record(s, metric_key)) is not None]
        if len(records) < 3:
            logger.info("Skipping %s for comparison (only %d records)", method, len(records))
            continue
        try:
            fit_result = fit_scaling_laws(records)
            method_minima = []
            for m in fit_result.minima_records:
                all_minima.append((method, m))
                method_minima.append(m)
            if method in fit_result.scaling_fits:
                all_fits[method] = fit_result.scaling_fits[method]

            # Fit loss power laws through the minima
            if len(method_minima) >= 2:
                tokens_list = [m.optimal_tokens for m in method_minima]
                flops_list = [m.flops for m in method_minima]
                loss_list = [m.loss_at_optimal for m in method_minima]
                lt_fit = _fit_loss_vs_x(tokens_list, loss_list)
                if lt_fit:
                    loss_vs_tokens_fits[method] = lt_fit
                lc_fit = _fit_loss_vs_x(flops_list, loss_list)
                if lc_fit:
                    loss_vs_compute_fits[method] = lc_fit
        except Exception as e:
            logger.warning("Fit failed for %s: %s", method, e)

    if not all_minima:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    exp_label = _label_for_tag(experiment_tag)
    methods_present = sorted({m for m, _ in all_minima})

    # --- Plot 1: D* vs Loss with forecast ---
    fig1 = go.Figure()
    for method in methods_present:
        pts = [(rec.optimal_tokens, rec.loss_at_optimal, rec.flops) for m, rec in all_minima if m == method]
        color = COMPARE_COLORS.get(method, "#333")
        fig1.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="markers+text",
                marker=dict(size=10, color=color),
                text=[f"{p[2]:.0e}" for p in pts],
                textposition="top center",
                textfont=dict(size=8),
                name=method,
                hovertemplate="%{text} FLOPs<br>tokens=%{x:.2e}<br>bpb=%{y:.4f}<extra>%{fullData.name}</extra>",
            )
        )
        if method in loss_vs_tokens_fits:
            A, beta = loss_vs_tokens_fits[method]
            t_range = np.logspace(
                np.log10(min(p[0] for p in pts)) - 0.2,
                np.log10(PROJECTION_MAX_TOKENS),
                100,
            )
            loss_proj = A * t_range ** (-beta)
            fig1.add_trace(
                go.Scatter(
                    x=t_range,
                    y=loss_proj,
                    mode="lines",
                    line=dict(color=color, dash="dash", width=1.5),
                    name=f"{method} forecast",
                    hovertemplate="tokens=%{x:.2e}<br>bpb=%{y:.4f}<extra>%{fullData.name}</extra>",
                )
            )
    fig1.update_layout(
        template="plotly_white",
        xaxis_type="log",
        xaxis_title="Optimal tokens D* (log)",
        yaxis_title=metric_key,
        title=f"Compute-Optimal Frontier (Tokens vs Loss) — {exp_label}",
        width=1000,
        height=600,
    )
    fig1.write_html(str(output_dir / f"compare_dstar_{experiment_tag}.html"))

    # --- Plot 2: Compute vs D* with forecast ---
    fig2 = go.Figure()
    for method in methods_present:
        pts = [(rec.flops, rec.optimal_tokens) for m, rec in all_minima if m == method]
        color = COMPARE_COLORS.get(method, "#333")
        fig2.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="markers",
                marker=dict(size=10, color=color),
                name=method,
                hovertemplate="C=%{x:.2e}<br>D*=%{y:.2e}<extra>%{fullData.name}</extra>",
            )
        )
        if method in all_fits:
            fit = all_fits[method]
            c_range = np.logspace(
                np.log10(min(p[0] for p in pts)) - 0.3,
                np.log10(PROJECTION_MAX_FLOPS),
                100,
            )
            d_star = fit.A * c_range**fit.alpha
            fig2.add_trace(
                go.Scatter(
                    x=c_range,
                    y=d_star,
                    mode="lines",
                    line=dict(color=color, dash="dash", width=1.5),
                    name=f"{method} forecast (α={fit.alpha:.3f})",
                    hovertemplate="C=%{x:.2e}<br>D*=%{y:.2e}<extra>%{fullData.name}</extra>",
                )
            )
    fig2.update_layout(
        template="plotly_white",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Compute budget (FLOPs, log)",
        yaxis_title="Optimal tokens D* (log)",
        title=f"Scaling Law D*(C) with Forecast — {exp_label}",
        width=1000,
        height=600,
    )
    fig2.write_html(str(output_dir / f"compare_scaling_{experiment_tag}.html"))

    # --- Plot 3: Loss frontier with forecast ---
    fig3 = go.Figure()
    for method in methods_present:
        pts = sorted([(rec.flops, rec.loss_at_optimal) for m, rec in all_minima if m == method])
        color = COMPARE_COLORS.get(method, "#333")
        fig3.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="markers",
                marker=dict(size=10, color=color),
                name=method,
                hovertemplate="C=%{x:.2e}<br>bpb=%{y:.4f}<extra>%{fullData.name}</extra>",
            )
        )
        if method in loss_vs_compute_fits:
            A, beta = loss_vs_compute_fits[method]
            c_range = np.logspace(
                np.log10(min(p[0] for p in pts)) - 0.3,
                np.log10(PROJECTION_MAX_FLOPS),
                100,
            )
            loss_proj = A * c_range ** (-beta)
            fig3.add_trace(
                go.Scatter(
                    x=c_range,
                    y=loss_proj,
                    mode="lines",
                    line=dict(color=color, dash="dash", width=1.5),
                    name=f"{method} forecast (β={beta:.4f})",
                    hovertemplate="C=%{x:.2e}<br>bpb=%{y:.4f}<extra>%{fullData.name}</extra>",
                )
            )
    fig3.update_layout(
        template="plotly_white",
        xaxis_type="log",
        xaxis_title="Compute budget (FLOPs, log)",
        yaxis_title=metric_key,
        title=f"Loss Frontier with Forecast — {exp_label}",
        width=1000,
        height=600,
    )
    fig3.write_html(str(output_dir / f"compare_frontier_{experiment_tag}.html"))

    methods_str = ", ".join(methods_present)
    logger.info("Comparison plots with forecasts for %s (%s) → %s/", experiment_tag, methods_str, output_dir)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--methods", nargs="+", default=["dclm"])
    parser.add_argument(
        "--suffix",
        default="",
        help="Filename suffix filter (e.g. 'v4'). Empty (default) matches all "
        "summaries in the prefix; ExpC and current ExpA/B summaries don't carry "
        "a suffix anymore.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Which experiment_tags to plot (e.g. 'expA_natural expB_T20T expC_T33T'). "
        "Default: auto-detect from the loaded summaries.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["eval/paloma/bpb", "paloma_macro_loss", "eval/lima/loss"],
        help="Eval metric keys from summary.json. Each gets its own subfolder. "
        "Use 'paloma_macro_loss' for macro-averaged loss across 16 paloma datasets. "
        "'eval/lima/loss' is the LIMA validation loss (added to ExpC by default).",
    )
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--output-dir", default="plots/curation_isoflop")
    parser.add_argument("--csv-only", action="store_true", help="Just dump CSVs, no plots")
    parser.add_argument(
        "--drop-tail-outliers",
        action="store_true",
        help="In addition to the standard plots, also produce a parallel plot tree "
        "under '<metric_short>_outliers_removed/' with visibly-off-curve points dropped. "
        "Uses fit-to-interior (n>=8) or LOO (4<=n<8) and an absolute-residual floor "
        "calibrated per-metric. See `detect_outlier_runs()` for the algorithm.",
    )
    parser.add_argument(
        "--outlier-floor",
        type=float,
        default=None,
        help="Absolute-residual floor for the outlier filter (in metric units, "
        "above the curve). If None (default), uses metric-specific defaults: "
        "0.04 for bpb, 0.07 for paloma/uncheatable macro loss, 0.10 for lima loss.",
    )
    args = parser.parse_args(argv)

    summaries = load_summaries(args.results_prefix, args.methods, args.suffix)
    if not summaries:
        logger.error("No summaries found. Check --methods and --suffix.")
        return

    # Auto-detect experiment tags if not user-specified, so ExpC summaries are
    # picked up automatically once they land alongside ExpA/ExpB.
    if args.experiments:
        exp_tags = list(args.experiments)
    else:
        exp_tags = sorted(
            {s.get("plan", {}).get("experiment_tag") for s in summaries if s.get("plan", {}).get("experiment_tag")}
        )
    logger.info("Plotting for experiment_tags: %s", exp_tags)

    base_dir = Path(args.output_dir)

    for metric_key in args.metrics:
        # e.g. "eval/paloma/bpb" -> "bpb", "eval/paloma/loss" -> "loss".
        # Disambiguate "eval/lima/loss" so it doesn't collide with the
        # paloma "loss" subfolder. (Same scheme as plot_fixed_model_sweep.)
        if metric_key.startswith("eval/lima/"):
            metric_short = "lima_" + metric_key.split("/")[-1]
        else:
            metric_short = metric_key.split("/")[-1]

        # Build the list of (label, summaries_subset, output_dir) we'll plot.
        # Always plot the full set; if --drop-tail-outliers is on, also plot a
        # filtered view to a parallel `_outliers_removed/` folder.
        passes: list[tuple[str, list[dict], Path]] = [("all", summaries, base_dir / metric_short)]
        if args.drop_tail_outliers:
            drop_run_names = detect_outlier_runs(summaries, metric_key, abs_floor=args.outlier_floor)
            filtered = [s for s in summaries if s.get("plan", {}).get("run_name") not in drop_run_names]
            passes.append(("outliers_removed", filtered, base_dir / f"{metric_short}_outliers_removed"))
            print(f"\n{'#'*60}")
            print(f"# Outlier filter for {metric_key}: dropped {len(drop_run_names)} of {len(summaries)} runs")
            print(
                f"#   floor = {args.outlier_floor if args.outlier_floor is not None else _DEFAULT_OUTLIER_FLOOR_BY_METRIC.get(metric_key, 0.07)}"
            )
            print("#   dropped run_names:")
            for rn in sorted(drop_run_names):
                print(f"#     - {rn}")
            print(f"{'#'*60}")

        for pass_label, pass_summaries, output_dir in passes:
            print(f"\n{'#'*60}")
            print(f"# Metric: {metric_key} ({pass_label}) -> {output_dir}/")
            print(f"{'#'*60}")

            for exp_tag in exp_tags:
                methods_in_summaries = sorted({s["plan"]["method_name"] for s in pass_summaries})
                for method in methods_in_summaries:
                    method_summaries = [s for s in pass_summaries if s["plan"]["method_name"] == method]
                    df = plot_experiment(
                        method_summaries,
                        experiment_tag=exp_tag,
                        metric_key=metric_key,
                        output_dir=output_dir / method,
                    )
                    if df is not None:
                        print(f"\n{'='*60}")
                        print(f"{method} / {exp_tag} ({pass_label}): {len(df)} runs")
                        print(f"  Budgets: {sorted(df.flops.unique())}")
                        print(f"  Loss range: {df.loss.min():.3f} — {df.loss.max():.3f}")
                        print(f"  Output: {output_dir / method}/")

                plot_comparison(pass_summaries, exp_tag, metric_key, output_dir / "_comparison")


if __name__ == "__main__":
    main()
