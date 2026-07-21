# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot DCLM Core_v2 vs training tokens, gridded by model scale (hidden_dim).

Companion to ``plot_warc_scaling_sweep.py``'s ``grid_compressed_x_tokens``
view, but for the 10k-WARC curation isoflop sweep and with DCLM Core_v2
(a downstream benchmark accuracy) on the y-axis instead of macro loss.

Layout mirrors the reference dashboard style:

  * one subplot per ``hidden_dim`` (model scale), sorted ascending, arranged on a
    2-row grid (3 panels top, 2 bottom) so each panel stays roughly square rather
    than squished wide-and-short — scan left-to-right / top-to-bottom from the
    smallest model (d512) to the largest (d3584);
  * x-axis = training tokens on a log ("compressed") scale;
  * y-axis = DCLM Core_v2;
  * one ``lines+markers`` trace per curation method, with colors held CONSISTENT
    across subplots (reusing ``plot_curation_isoflop.COMPARE_COLORS``) and a
    single shared legend.

Within a subplot, a method's points are its runs at that width across the
isoflop budgets, sorted by tokens and connected — so the curve shows Core_v2
rising with compute at a fixed model scale, per method.

Data sources (both in ``gs://marin-us-central1/metadata/``):

  * Core_v2 per run: ``data_curation_10k_core_results/*_summary.json`` — each
    file has ``dclm.Core_v2`` (float). The run name is parsed out of the
    filename ``curation-<method>_10k-expFM_natural-<budget>-d<H>-L<L>-B<B>``.
  * Tokens per run (x-axis): ``data_curation_10k_natural_results/<stem>.json``
    (same run-name stem, without ``_summary``) — the token count is read
    straight from ``tokens.tokens_trained``. Joined to Core_v2 by run stem.
    The subplot's model-size label is read from ``model.total_trainable_params``
    in this same file (identical across methods/budgets at a given width).

Usage::

    export SSL_CERT_FILE=$(.venv/bin/python -m certifi)
    .venv/bin/python -m experiments.scaling_law_sweeps.plot_core_v2_by_size_10k
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from experiments.scaling_law_sweeps.plot_curation_isoflop import COMPARE_COLORS, _compute_uncheatable_macro_loss

logger = logging.getLogger(__name__)

DEFAULT_CORE_PREFIX = "gs://marin-us-central1/metadata/data_curation_10k_core_results/"
DEFAULT_TOKENS_PREFIX = "gs://marin-us-central1/metadata/data_curation_10k_natural_results/"
# The 3000-WARC fixed-model sweep shares one core-results prefix (biased + random
# runs distinguished by method name) and reads tokens from the FM summaries.
THREEK_CORE_PREFIX = "gs://marin-us-central1/metadata/data_curation_3k_core_results/"
FM_TOKENS_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"
DEFAULT_OUTPUT = Path(__file__).parent.parent.parent / "scratch" / "plots" / "core_v2" / "core_v2_grid_x_tokens.html"

# Full planned grid for the 6 baseline methods: 6 methods x 5 hidden_dims x ~8
# budgets. Used only for the "N of TOTAL" partial-data annotation. The
# fast_pipe_v3 runs are a separate overlay (d1024/d1536 only) and are counted
# apart from this baseline total.
TOTAL_PLANNED_CELLS = 240

# Filename stem parser. Two run families share this layout:
#   * baselines:  curation-<method>_10k-expFM_natural-<budget>-d<H>-L<L>-B<B>
#   * fastpipe:   curation-fastpipe_v3_<T>-expFM_natural-<budget>-d<H>-L<L>-B<B>
# The ``_10k`` suffix is present only on the baselines, so it is optional here
# and stripped when present; the fastpipe method keeps its full ``fastpipe_v3_<T>``
# name (e.g. ``fastpipe_v3_20``) as its series key.
_STEM_RE = re.compile(
    r"^curation-(?P<method>.+?)(?:_10k)?-expFM_natural-(?P<budget>[0-9eE+.\-]+)-d(?P<hidden>\d+)-L(?P<layers>\d+)-B(?P<batch>\d+)$"
)

# Per-scale color for the 6 baseline methods present in the 10k core sweep.
# Reuse the canonical curation palette; ``nemotron`` (the 10k method name) has
# no direct key in COMPARE_COLORS (which splits it into _org/_full/_qhigh), so
# pin it to the classic nemotron orange, distinct from all other 5 methods here.
METHOD_COLORS = {
    "dclm": COMPARE_COLORS["dclm"],  # blue
    "nemotron": "#ff7f0e",  # orange
    "fineweb_cc": COMPARE_COLORS["fineweb_cc"],  # salmon
    "fineweb_edu": COMPARE_COLORS["fineweb_edu"],  # magenta-pink
    "high_quality": COMPARE_COLORS["high_quality"],  # green
    "resiliparse": COMPARE_COLORS["resiliparse"],  # purple
    "med_quality": "#8c564b",  # brown (3k sweeps only; absent from 10k)
}

# The 3000-WARC sweeps name their runs differently from the canonical 10k method
# names. These aliases fold each 3k run-method onto its canonical color/label so
# the biased-3k and random-3k figures reuse the exact same palette as the 10k
# figure. Identity for 10k names -> the default (10k) path is unchanged.
BIASED_3K_METHODS = {
    "dclm",
    "nemotron_full_bos_fixed",
    "resiliparse_dedup",
    "high_quality_3000",
    "fineweb_edu",
    "med_quality_3000",
}
RANDOM_3K_METHODS = {
    "dclm_random_3000",
    "nemotron_full_random_3000",
    "resiliparse_random_dedup_3000",
}
METHOD_ALIAS = {
    "nemotron_full_bos_fixed": "nemotron",
    "resiliparse_dedup": "resiliparse",
    "high_quality_3000": "high_quality",
    "med_quality_3000": "med_quality",
    "dclm_random_3000": "dclm",
    "nemotron_full_random_3000": "nemotron",
    "resiliparse_random_dedup_3000": "resiliparse",
}

# Loss-optimal checkpoint marker: a dashed vertical at the tokens_trained where a
# method reached its MINIMUM uncheatable-eval macro_loss, at each width. Computed
# from the plotted runs (see ``_loss_min_tokens``) rather than hardcoded, so it
# tracks whichever runs are shown and works for every regime -- for the 3k sweeps
# it can land at an interior budget where over-epoching bottomed the loss out
# before the top-compute run.

# fast_pipe_v3 filter-threshold overlay. Rendered dashed with diamond markers in
# a light-grey -> near-black sequential ramp (v3_20 -> v3_100) so the 5 thresholds
# read as one grey family — clearly NOT one of the colored solid baselines — while
# staying internally separable by their light->dark ordering.
_FASTPIPE_RE = re.compile(r"^fastpipe_v3_(?P<thr>\d+)$")
FASTPIPE_COLORS = {
    20: "#bbbbbb",
    40: "#909090",
    60: "#666666",
    80: "#3d3d3d",
    100: "#111111",
}


def _is_fastpipe(method: str) -> bool:
    return _FASTPIPE_RE.match(method) is not None


def _method_sort_key(method: str) -> tuple[int, float, str]:
    """Order baselines first (alphabetical), then fastpipe by numeric threshold."""
    m = _FASTPIPE_RE.match(method)
    if m is not None:
        return (1, float(m.group("thr")), method)
    return (0, 0.0, method)


def _method_label(method: str) -> str:
    """Legend label; fastpipe thresholds get a grouped, self-describing name."""
    m = _FASTPIPE_RE.match(method)
    if m is not None:
        return f"fastpipe v3_{m.group('thr')} (filter thr)"
    return method


@dataclass(frozen=True)
class CoreRecord:
    """One trained run: its scale, method, tokens, Core_v2, and val loss."""

    run_stem: str
    method: str
    hidden_dim: int
    num_layers: int
    budget: str
    tokens: float
    core_v2: float
    total_params: int
    macro_loss: float | None  # uncheatable-eval macro loss; None if absent


def _loss_min_tokens(records: list[CoreRecord]) -> dict[int, dict[str, float]]:
    """For each (hidden_dim, method), the tokens of its lowest-loss run.

    Drives the loss-optimal vline. Fastpipe series and runs missing a loss are
    ignored. Keyed ``[hidden_dim][method] -> tokens``.
    """
    best: dict[tuple[int, str], tuple[float, float]] = {}  # (dim, method) -> (loss, tokens)
    for r in records:
        if r.macro_loss is None or _is_fastpipe(r.method):
            continue
        key = (r.hidden_dim, r.method)
        if key not in best or r.macro_loss < best[key][0]:
            best[key] = (r.macro_loss, r.tokens)
    out: dict[int, dict[str, float]] = {}
    for (dim, method), (_, tokens) in best.items():
        out.setdefault(dim, {})[method] = tokens
    return out


def _params_label(hidden_dim: int, total_params: int) -> str:
    """Human label for a subplot: width + actual trainable param count.

    ``total_params`` is the exact ``model.total_trainable_params`` reported by
    the training run, formatted as M below 1B and B at/above 1B.
    """
    if total_params >= 1e9:
        approx = f"{total_params / 1e9:.2f}B"
    else:
        approx = f"{total_params / 1e6:.0f}M"
    return f"d{hidden_dim} ({approx})"


def _open_gcs(path: str):
    """Open a gs:// object for reading via rigging's cached GCS filesystem."""
    from rigging.filesystem import filesystem

    return filesystem("gcs").open(path)


def _load_records(
    core_prefix: str,
    tokens_prefix: str,
    include_methods: set[str] | None = None,
    alias: dict[str, str] | None = None,
) -> list[CoreRecord]:
    """Scan the core-results bucket, join tokens by run stem, return records.

    Reads are fanned out across a thread pool (they're pure network latency).
    Runs whose Core_v2 or joined token count is missing are dropped with a
    warning rather than silently skipped.

    ``include_methods`` (raw filename method names) restricts which runs load --
    used to select just the biased or just the random 3k family out of the shared
    3k core prefix. ``alias`` folds each raw method onto a canonical key so the 3k
    families reuse the 10k palette. Both default to the 10k behaviour (no filter,
    no aliasing).
    """
    from rigging.filesystem import filesystem

    alias = alias or {}
    fs = filesystem("gcs")
    core_paths = fs.glob(core_prefix.rstrip("/") + "/*_summary.json")
    logger.info("Found %d core summaries under %s", len(core_paths), core_prefix)

    def _load_one(core_path: str) -> CoreRecord | None:
        fname = core_path.rsplit("/", 1)[-1]
        stem = fname[: -len("_summary.json")]
        m = _STEM_RE.match(stem)
        if m is None:
            logger.warning("Unparseable core filename: %s", fname)
            return None
        raw_method = m.group("method")
        if include_methods is not None and raw_method not in include_methods:
            return None
        method = alias.get(raw_method, raw_method)
        with _open_gcs("gs://" + core_path if not core_path.startswith("gs://") else core_path) as fh:
            core_v2 = json.load(fh).get("dclm", {}).get("Core_v2")
        if core_v2 is None:
            logger.warning("No dclm.Core_v2 in %s", fname)
            return None
        tokens_path = tokens_prefix.rstrip("/") + "/" + stem + ".json"
        try:
            with _open_gcs(tokens_path) as fh:
                natural = json.load(fh)
        except FileNotFoundError:
            natural = None
        tokens = natural.get("tokens", {}).get("tokens_trained") if natural is not None else None
        total_params = natural.get("model", {}).get("total_trainable_params") if natural is not None else None
        macro_loss = _compute_uncheatable_macro_loss(natural.get("eval") or {}) if natural is not None else None
        if tokens is None:
            logger.warning("No tokens_trained joined for %s", stem)
            return None
        if total_params is None:
            logger.warning("No total_trainable_params joined for %s", stem)
            return None
        return CoreRecord(
            run_stem=stem,
            method=method,
            hidden_dim=int(m.group("hidden")),
            num_layers=int(m.group("layers")),
            budget=m.group("budget"),
            tokens=float(tokens),
            core_v2=float(core_v2),
            total_params=int(total_params),
            macro_loss=macro_loss,
        )

    records: list[CoreRecord] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_load_one, p) for p in core_paths]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is not None:
                records.append(rec)
    logger.info("Loaded %d joined records", len(records))
    return records


def _build_figure(
    records: list[CoreRecord],
    *,
    draw_loss_vlines: bool = True,
    title_main: str = "DCLM CoreV2 vs tokens by model scale (10k curation isoflop)",
    subtitle: str | None = None,
):
    """Grid of subplots (one per hidden_dim) of Core_v2 vs tokens, per method.

    Panels are laid out on a 2-row grid (3 top, 2 bottom) so each stays roughly
    square. All panels share one Core_v2 y-range so scales are directly
    comparable, the log token axis is labelled only at the decade ticks (1B/10B/
    100B) with a light minor grid, and the full legend (6 baselines + 5 fastpipe
    thresholds) sits horizontally below the plot so nothing is clipped.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dims = sorted({r.hidden_dim for r in records})
    params_for_dim = {r.hidden_dim: r.total_params for r in records}
    loss_min = _loss_min_tokens(records) if draw_loss_vlines else {}

    n_panels = len(dims)
    n_cols = 3
    n_rows = math.ceil(n_panels / n_cols)

    def _row_col(idx: int) -> tuple[int, int]:
        return idx // n_cols + 1, idx % n_cols + 1

    # Bottom-most occupied panel index per column, so we only stamp the "tokens"
    # x-title on the lowest panel in each column (col 3 has no second row here).
    bottom_panel: dict[int, int] = {}
    for idx in range(n_panels):
        _, col = _row_col(idx)
        bottom_panel[col] = idx

    # Shared Core_v2 range across every panel (small symmetric pad).
    y_vals = [r.core_v2 for r in records]
    y_lo, y_hi = min(y_vals), max(y_vals)
    y_pad = 0.04 * (y_hi - y_lo) if y_hi > y_lo else 0.01
    y_range = [y_lo - y_pad, y_hi + y_pad]

    titles = [_params_label(d, params_for_dim[d]) for d in dims]
    titles += [""] * (n_rows * n_cols - n_panels)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    seen_methods: set[str] = set()
    fastpipe_title_shown = False
    for idx, dim in enumerate(dims):
        row, col = _row_col(idx)
        cell = [r for r in records if r.hidden_dim == dim]
        by_method: dict[str, list[CoreRecord]] = {}
        for r in cell:
            by_method.setdefault(r.method, []).append(r)
        for method in sorted(by_method, key=_method_sort_key):
            pts = sorted(by_method[method], key=lambda r: r.tokens)
            is_fp = _is_fastpipe(method)
            if is_fp:
                thr = int(_FASTPIPE_RE.match(method).group("thr"))
                color = FASTPIPE_COLORS.get(thr, "#333333")
            else:
                color = METHOD_COLORS.get(method, "#333333")
            dash = "dash" if is_fp else "solid"
            marker_symbol = "diamond" if is_fp else "circle"
            label = _method_label(method)
            show_legend = method not in seen_methods
            seen_methods.add(method)
            # All fastpipe thresholds share one legendgroup so the family gets a
            # single group title and toggles together; baselines stay per-method.
            legendgroup = "fastpipe_v3" if is_fp else method
            group_title = None
            if is_fp and show_legend and not fastpipe_title_shown:
                group_title = dict(text="fastpipe v3 (filter threshold)")
                fastpipe_title_shown = True
            fig.add_trace(
                go.Scatter(
                    x=[p.tokens for p in pts],
                    y=[p.core_v2 for p in pts],
                    mode="lines+markers",
                    marker=dict(size=7, color=color, symbol=marker_symbol),
                    line=dict(color=color, width=2, dash=dash),
                    name=label,
                    legendgroup=legendgroup,
                    legendgrouptitle=group_title,
                    showlegend=show_legend,
                    customdata=[p.budget for p in pts],
                    hovertemplate=(
                        f"method={method}<br>d={dim}"
                        "<br>tokens=%{x:.3e}<br>Core_v2=%{y:.4f}"
                        "<br>budget=%{customdata}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            # Loss-optimal-checkpoint marker: a dashed vertical line at the token
            # count where this baseline method bottomed out its uncheatable-eval
            # macro_loss. Added as a scatter trace (not a layout shape) sharing the
            # method's legendgroup so it toggles with the method's legend entry.
            loss_min_tok = loss_min.get(dim, {}).get(method) if (draw_loss_vlines and not is_fp) else None
            if loss_min_tok is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[loss_min_tok, loss_min_tok],
                        y=y_range,
                        mode="lines",
                        line=dict(color=color, dash="dash", width=1.5),
                        name=f"{method} loss-min",
                        legendgroup=legendgroup,
                        showlegend=False,
                        hovertemplate=(f"{method} loss-min<br>d={dim}<br>tokens=%{{x:.3e}}<extra></extra>"),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(
            type="log",
            tickmode="array",
            tickvals=[1e9, 1e10, 1e11],
            ticktext=["1B", "10B", "100B"],
            minor=dict(showgrid=False),
            title_text="tokens" if idx == bottom_panel.get(col) else None,
            row=row,
            col=col,
        )
        # Every panel shows the (shared-range) Core_v2 tick values; only the
        # leftmost column carries the axis title to avoid repetition.
        fig.update_yaxes(
            range=y_range,
            title_text="DCLM Core_v2" if col == 1 else None,
            showticklabels=True,
            row=row,
            col=col,
        )

    n_fastpipe = sum(1 for r in records if _is_fastpipe(r.method))
    n_baseline = len(records) - n_fastpipe
    if subtitle is None:
        subtitle = (
            f"partial data: {n_baseline} of {TOTAL_PLANNED_CELLS} planned baseline cells "
            f"+ {n_fastpipe} fastpipe_v3 runs (d1024/d1536 only); "
            "x=tokens_trained (log), one line per curation method"
        )
    # Legend placement is layout-aware. When the grid has a trailing empty cell
    # (e.g. the 10k 5-panel case), park the legend there to fill otherwise-wasted
    # space. When every cell is occupied (e.g. the 3-panel biased/random-3k case),
    # a parked legend would sit ON a panel, so move it outside to the right and
    # widen the canvas so nothing is clipped.
    n_empty = n_rows * n_cols - n_panels
    extra_width = 0
    if n_empty >= 1:
        empty_idx = n_panels  # first trailing empty cell
        er, ec = empty_idx // n_cols, empty_idx % n_cols  # 0-indexed row/col
        col_w = (1.0 - 0.06 * (n_cols - 1)) / n_cols  # matches horizontal_spacing
        row_h = (1.0 - 0.12 * (n_rows - 1)) / n_rows  # matches vertical_spacing
        legend = dict(
            title=dict(text="curation method"),
            orientation="v",
            xanchor="left",
            x=ec * (col_w + 0.06),
            yanchor="middle",
            y=1.0 - er * (row_h + 0.12) - row_h / 2,
        )
        right_margin = 40
    else:
        legend = dict(
            title=dict(text="curation method"),
            orientation="v",
            xanchor="left",
            x=1.02,
            yanchor="middle",
            y=0.5,
        )
        right_margin = 60
        extra_width = 150  # room for the outside-right legend

    fig.update_layout(
        template="plotly_white",
        title=f"{title_main}<br><sub>{subtitle}</sub>",
        width=360 * n_cols + 120 + extra_width,
        height=360 * n_rows + 120,
        showlegend=True,
        margin=dict(l=70, r=right_margin, t=100, b=60),
        legend=legend,
    )
    return fig


def _write_csv(records: list[CoreRecord], path: Path) -> None:
    """Write the tidy underlying data next to the HTML."""
    import csv

    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_stem", "method", "hidden_dim", "num_layers", "budget", "tokens", "core_v2"])
        for r in sorted(records, key=lambda r: (r.hidden_dim, r.method, r.tokens)):
            writer.writerow([r.run_stem, r.method, r.hidden_dim, r.num_layers, r.budget, r.tokens, r.core_v2])
    logger.info("Wrote %s", path)


# Per-regime wiring: which prefixes, which raw methods to keep, the title, and
# whether the 10k loss-optimal vlines apply. Selected by --regime; --results-gs /
# --tokens-gs still override the prefixes for ad-hoc runs.
REGIME_PRESETS: dict[str, dict] = {
    "10k": dict(
        core=DEFAULT_CORE_PREFIX,
        tokens=DEFAULT_TOKENS_PREFIX,
        include=None,
        vlines=True,
        title="DCLM CoreV2 vs tokens by model scale (10k curation isoflop)",
        default_out="core_v2/core_v2_grid_x_tokens.html",
    ),
    "biased3k": dict(
        core=THREEK_CORE_PREFIX,
        tokens=FM_TOKENS_PREFIX,
        include=BIASED_3K_METHODS,
        vlines=True,
        title="DCLM CoreV2 vs tokens by model scale (BIASED 3k curation sweep)",
        default_out="core_v2/core_v2_grid_x_tokens_biased3k.html",
    ),
    "random3k": dict(
        core=THREEK_CORE_PREFIX,
        tokens=FM_TOKENS_PREFIX,
        include=RANDOM_3K_METHODS,
        vlines=True,
        title="DCLM CoreV2 vs tokens by model scale (RANDOM 3k curation sweep)",
        default_out="core_v2/core_v2_grid_x_tokens_random3k.html",
    ),
}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--regime", choices=list(REGIME_PRESETS), default="10k", help="Which sweep to plot.")
    parser.add_argument("--results-gs", default=None, help="Override GCS prefix of *_summary.json Core_v2 files.")
    parser.add_argument("--tokens-gs", default=None, help="Override GCS prefix of natural-run token summaries.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (defaults per regime).")
    args = parser.parse_args(argv)

    preset = REGIME_PRESETS[args.regime]
    core_gs = args.results_gs or preset["core"]
    tokens_gs = args.tokens_gs or preset["tokens"]
    output = args.output or (DEFAULT_OUTPUT.parent.parent / preset["default_out"])

    records = _load_records(core_gs, tokens_gs, include_methods=preset["include"], alias=METHOD_ALIAS)
    if not records:
        raise SystemExit("No records loaded; nothing to plot.")

    output.parent.mkdir(parents=True, exist_ok=True)
    subtitle = (
        None if args.regime == "10k" else f"{len(records)} runs; x=tokens_trained (log), one line per curation method"
    )
    fig = _build_figure(records, draw_loss_vlines=preset["vlines"], title_main=preset["title"], subtitle=subtitle)
    fig.write_html(str(output), include_plotlyjs="cdn")
    logger.info("Wrote %s", output)
    _write_csv(records, output.with_suffix(".csv"))

    # Console readout: per-scale method ranking, so the trend is visible without
    # opening the HTML.
    dims = sorted({r.hidden_dim for r in records})
    for dim in dims:
        cell = [r for r in records if r.hidden_dim == dim]
        # Best (max) Core_v2 per method at this scale.
        best: dict[str, float] = {}
        for r in cell:
            best[r.method] = max(best.get(r.method, -math.inf), r.core_v2)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        logger.info("d%d best-Core_v2 ranking: %s", dim, ", ".join(f"{m}={v:.4f}" for m, v in ranked))


if __name__ == "__main__":
    main()
