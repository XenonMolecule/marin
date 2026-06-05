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
    "llm_curated": ("LLM-Curated", "#8c564b"),
    "llm_curated_bos_fixed": ("LLM-Curated", "#8c564b"),
    "high_quality_3000": ("Spec-Driven Extraction (ours)", "#2ca02c"),
    "high_quality_2000": ("Spec-Driven Extraction (ours, 2k)", "#2ca02c"),
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
        panel_params: int | None = None
        for method in methods:
            pts = collect_points(summaries, method, hidden_dim, metric_key)
            label, color = METHOD_DISPLAY.get(method, (method, "#444444"))
            pool = plot_method(ax, pts, color, label, is_ours=(method == ours_method))
            if pool is not None:
                method_pools.setdefault(label, pool)
            if panel_params is None and pts:
                panel_params = pts[0].params
        panel_title = (
            f"{fmt_params(panel_params)} params  (d={hidden_dim})" if panel_params is not None else f"d={hidden_dim}"
        )
        ax.set_xscale("log")
        ax.set_xlabel("Tokens trained")
        ax.set_title(panel_title, pad=10, fontweight="semibold")
        ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.55)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_tokens))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, p: ""))

    axes[0].set_ylabel(metric_label(metric_key))

    handles, labels = [], []
    for ax in axes:
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

    fig.legend(
        handles,
        enriched_labels,
        loc="lower center",
        ncol=len(enriched_labels),
        bbox_to_anchor=(0.5, 0.04),
        handlelength=2.4,
        columnspacing=1.8,
    )

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

    lima_dir = Path(args.lima_sidecar_prefix)
    merge_lima_sidecars(summaries, lima_dir)

    out_dir = Path(args.output_dir)
    for raw_metric in args.metrics:
        metric_key = resolve_metric_key(raw_metric)
        ours_for_plot = (
            args.ours_method.removesuffix("_bos_fixed")
            if args.prefer_bos_fixed and args.ours_method.endswith("_bos_fixed")
            else args.ours_method
        )
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
