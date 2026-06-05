# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side comparison: BIASED 3k vs RANDOM 3k WARC draw.

A standalone diagnostic plot (separate from the warc-scaling dashboard) to
isolate the effect of the WARC-sample bias. Two 3000-WARC pools, both trained
under the fixed-model natural-epoch sweep (tag ``expFM_natural``, same widths x
budgets), so the comparison is apples-to-apples per (method, width, budget):

  - BIASED 3k: the canonical draw = the ~prefix of the 10k pool (the original
    3000-WARC anchor). Methods: ``dclm`` / ``nemotron_full_bos_fixed`` /
    ``resiliparse_dedup``.
  - RANDOM 3k: an independent random 3000-WARC draw. Methods:
    ``dclm_random_3000`` / ``nemotron_full_random_3000`` /
    ``resiliparse_random_dedup_3000``.

Layout: one row per method (DCLM / Nemotron / Resiliparse-Dedup), one column per
model width. Each cell overlays the biased curve (solid) and random curve
(dashed) of loss vs compute, so the vertical gap between the two lines is the
bias impact at that (method, width).

Usage::

    uv run python experiments/scaling_law_sweeps/plot_biased_vs_random_3k.py
    # -> scratch/plots/biased_vs_random/<metric>/grid_x_{flops,tokens}.html
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from experiments.scaling_law_sweeps.curation_plan import METHODS
from experiments.scaling_law_sweeps.plot_curation_isoflop import load_summaries
from experiments.scaling_law_sweeps.plot_warc_scaling_sweep import _extract_metric

logger = logging.getLogger(__name__)

FM_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"
FM_TAG = "expFM_natural"

# (display label, biased method_name, random method_name).
METHOD_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("DCLM", "dclm", "dclm_random_3000"),
    ("Nemotron", "nemotron_full_bos_fixed", "nemotron_full_random_3000"),
    ("Resiliparse-Dedup", "resiliparse_dedup", "resiliparse_random_dedup_3000"),
)

# Biased reference curve overlaid (green) on EVERY panel at the matching width,
# so each method's biased+random curves can be read against the best-curation
# baseline. High Quality has no random draw, so it's reference-only.
REFERENCE_METHOD = "high_quality_3000"
REFERENCE_LABEL = "High Quality (biased)"
REFERENCE_COLOR = "#2ca02c"  # green, dotted

BIASED_COLOR = "#1f77b4"  # blue, solid
RANDOM_COLOR = "#d62728"  # red, dashed
_X_ATTR = {"flops": "flops", "tokens": "tokens"}

# Only the widths the BIASED anchor actually trained -> the comparable set.
# Random-3k also ran 3328/3584 (now killed), but those have no biased twin, so
# we exclude them from the comparison (keeps it a clean 3x3: 3 methods x 3 widths).
COMPARE_WIDTHS: tuple[int, ...] = (512, 1536, 2432)


def _record(summary: dict, metric_key: str) -> dict | None:
    plan = summary.get("plan", {})
    if plan.get("experiment_tag") != FM_TAG:
        return None
    loss = _extract_metric(summary.get("eval") or {}, metric_key)
    if loss is None:
        return None
    return {
        "method": plan.get("method_name", ""),
        "width": int(plan.get("hidden_dim", 0)),
        "flops": float(plan.get("budget_flops", 0.0)),
        "tokens": float((summary.get("tokens") or {}).get("tokens_trained", 0.0)),
        "params": int((summary.get("model") or {}).get("total_trainable_params", 0)),
        "loss": loss,
    }


def _params_label(params: int) -> str:
    if params >= 1e9:
        return f"{params / 1e9:.2f}B"
    return f"{params / 1e6:.0f}M"


def _tok_label(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    return f"{n / 1e6:.0f}M"


def _corpus_tokens(method_name: str) -> str:
    """Available corpus size (post-filter d_obs) for a curation method.

    This is the natural-epoch token pool the method draws from -- the biased vs
    random 3k draws retain different amounts after filtering (Nemotron, DCLM,
    Resiliparse all differ), which sets how many epochs each budget point runs.
    """
    m = METHODS.get(method_name)
    d_obs = getattr(m, "d_obs_tokens", 0) if m is not None else 0
    return _tok_label(d_obs) if d_obs > 0 else "?"


def _plot_grid(records: list[dict], metric_key: str, output_dir: Path, x_axis: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x_attr = _X_ATTR[x_axis]
    # (method_name, width) -> sorted [(x, loss)]
    by_cell: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    params_for: dict[int, int] = {}
    for r in records:
        if r["width"] not in COMPARE_WIDTHS:
            continue  # exclude widths with no biased twin (e.g. random-only 3328/3584)
        by_cell[(r["method"], r["width"])].append((r[x_attr], r["loss"]))
        params_for.setdefault(r["width"], r["params"])
    for k in by_cell:
        by_cell[k].sort()

    widths = sorted({w for (_, w) in by_cell})
    if not widths:
        logger.warning("No data for metric %s; skipping.", metric_key)
        return
    n_rows, n_cols = len(METHOD_PAIRS), len(widths)

    # Each panel title carries the model size (per width) and the method's
    # biased/random corpus token counts (per row) so the token gap that drives
    # epoch count is always on-screen.
    titles = [
        f"{label} | d{w} ({_params_label(params_for.get(w, 0))})"
        f"<br><span style='font-size:11px;color:#666'>"
        f"corpus: biased {_corpus_tokens(bm)} · random {_corpus_tokens(rm)}</span>"
        for (label, bm, rm) in METHOD_PAIRS
        for w in widths
    ]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    legend_seen: set[str] = set()
    for ri, (_label, biased_m, random_m) in enumerate(METHOD_PAIRS):
        for ci, w in enumerate(widths):
            # This method's biased + random curves, plus the High Quality biased
            # reference (green dotted) overlaid on every panel for comparison.
            for pool, mname, color, dash in (
                ("biased 3k", biased_m, BIASED_COLOR, "solid"),
                ("random 3k", random_m, RANDOM_COLOR, "dash"),
                (REFERENCE_LABEL, REFERENCE_METHOD, REFERENCE_COLOR, "dot"),
            ):
                pts = by_cell.get((mname, w), [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                show = pool not in legend_seen
                legend_seen.add(pool)
                # HQ is a single method, so its corpus size goes in the legend;
                # biased/random vary by row, so those go in the panel titles.
                disp = f"{pool} · {_corpus_tokens(REFERENCE_METHOD)}" if pool == REFERENCE_LABEL else pool
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        name=disp,
                        legendgroup=pool,
                        showlegend=show,
                        marker=dict(size=5, color=color),
                        line=dict(color=color, width=2, dash=dash),
                        hovertemplate=(f"{pool} | d{w}<br>{x_axis}=%{{x:.2e}}<br>loss=%{{y:.4f}}<extra></extra>"),
                    ),
                    row=ri + 1,
                    col=ci + 1,
                )
            fig.update_xaxes(type="log", row=ri + 1, col=ci + 1)

    fig.update_layout(
        template="plotly_white",
        title=f"BIASED 3k vs RANDOM 3k -- {metric_key} vs {x_axis} (rows=method, cols=width)",
        # Compressed to fit a laptop viewport: the 3x3 grid (method x width, with
        # the High Quality reference overlaid green on every panel) sits ~one screen.
        width=280 * n_cols + 150,
        height=215 * n_rows + 90,
        showlegend=True,
        margin=dict(l=50, r=40, t=70, b=40),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"grid_x_{x_axis}.html"
    fig.write_html(str(out))
    logger.info("Wrote %s", out)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fm-results-prefix", default=FM_RESULTS_PREFIX)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["uncheatable_macro_loss", "eval/macro_bpb"],
        help="Eval metrics to plot (one grid per metric, per x-axis).",
    )
    parser.add_argument("--output-dir", default="scratch/plots/biased_vs_random")
    args = parser.parse_args(argv)

    all_methods = [m for (_, b, r) in METHOD_PAIRS for m in (b, r) if m] + [REFERENCE_METHOD]
    summaries = load_summaries(args.fm_results_prefix, all_methods, suffix="")
    logger.info("Loaded %d summaries across %d methods", len(summaries), len(all_methods))

    output_root = Path(args.output_dir)
    for metric_key in args.metrics:
        records = [r for s in summaries if (r := _record(s, metric_key)) is not None]
        biased_set = {b for (_, b, _) in METHOD_PAIRS}
        random_set = {r for (_, _, r) in METHOD_PAIRS if r}
        n_biased = sum(1 for r in records if r["method"] in biased_set)
        n_random = sum(1 for r in records if r["method"] in random_set)
        n_ref = sum(1 for r in records if r["method"] == REFERENCE_METHOD)
        logger.info(
            "Metric %s: %d records (%d biased, %d random, %d high-quality-ref)",
            metric_key,
            len(records),
            n_biased,
            n_random,
            n_ref,
        )
        if not records:
            continue
        metric_dir = output_root / metric_key.replace("/", "_")
        for x_axis in ("flops", "tokens"):
            _plot_grid(records, metric_key, metric_dir, x_axis)


if __name__ == "__main__":
    main()
