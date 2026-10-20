# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paper figure: the extractor-only ablation, per model size.

Two mixture arms over the SAME 10,364-WARC document set, the same topic/quality labels and the
same olmixexact weights. The only difference is the HTML-to-text extractor that produced the
training text:

* ``lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01`` — the resiliparse-rs fork (markdown)
* ``lpv11_fastpipe_v2_upstreamrs_10k_mix_olmixexact_lambda0p01`` — upstream resiliparse (plain)

Same visual language as :mod:`plot_olmixexact_bpb_by_size`: ``PAPER_RCPARAMS``, one panel per
model size, log tokens-trained x-axis, white-edged markers, a shared bottom legend carrying each
corpus's pool size, PNG + PDF, and no title.

Metric note: the upstream arm has no olmo_bpb eval wave, so this reads the bits-per-byte the
training runs logged themselves (Paloma + uncheatable), not the olmix 51-task suite.

The two curves sit within ~0.002 BPB of each other, so the overlay alone cannot show the effect:
``--difference`` writes the companion figure (upstream minus fork) where the systematic gap is
legible.

One fork cell (2e+19, d512) is dropped: the lpv11 thread wrote a ``.badmodel`` copy of it and the
re-run reproduced the same anomalous value (1.419 BPB against ~1.11 for its neighbours).

Usage::

    uv run python -m experiments.scaling_law_sweeps.plot_extractor_ablation_by_size
    uv run python -m experiments.scaling_law_sweeps.plot_extractor_ablation_by_size --dims 512 1536 2432
    uv run python -m experiments.scaling_law_sweeps.plot_extractor_ablation_by_size --difference
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import fsspec
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiments.scaling_law_sweeps.curation_plan import METHODS
from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS

logger = logging.getLogger(__name__)

RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_10k_natural_results/"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "extractor_ablation"

FORK = "lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01"
UPSTREAM = "lpv11_fastpipe_v2_upstreamrs_10k_mix_olmixexact_lambda0p01"

# Fork keeps its locked repo colour (#006d2c, the "GC Cascade" dark green). The upstream arm takes
# the locked resiliparse purple: this arm IS upstream resiliparse extraction, so the hue already
# means that in every other curation figure.
METHOD_DISPLAY: dict[str, tuple[str, str]] = {
    FORK: ("GC Cascade — fork extractor", "#006d2c"),
    UPSTREAM: ("GC Cascade — upstream resiliparse", "#9467bd"),
}
# The fork line is the locked-emphasis GC line; upstream draws thinner ON TOP so both read where
# the curves coincide, which is nearly everywhere.
METHOD_LINEWIDTH = {FORK: 4.2, UPSTREAM: 2.4}
METHOD_ZORDER = {FORK: 3, UPSTREAM: 5}

# Metrics the training runs log inline, so both arms have them without an eval wave.
METRIC_KEY = {
    "bpb": "eval/bpb",
    "paloma": "eval/paloma/macro_bpb",
    "uncheatable": "eval/uncheatable_eval/macro_bpb",
}
METRIC_LABEL = {
    "bpb": "BPB (Paloma + uncheatable)",
    "paloma": "BPB (Paloma macro)",
    "uncheatable": "BPB (uncheatable macro)",
}

# Fork run the lpv11 thread flagged as a bad model; the 09-17 re-run reproduced it exactly.
EXCLUDED_CELLS = {(FORK, "2e+19-d512-L6-B128")}


@dataclass(frozen=True)
class Record:
    method: str
    cell: str
    hidden_dim: int
    total_params: int
    tokens: int
    bpb: dict[str, float]


def load_records(methods: list[str], results_prefix: str = RESULTS_PREFIX) -> list[Record]:
    """Every completed cell of ``methods``, read from the sweep's result JSONs."""
    fs = fsspec.filesystem(results_prefix.split("://", 1)[0])
    root = results_prefix.split("://", 1)[1]
    names = [p for p in fs.ls(root, refresh=True) if p.endswith(".json") and ".badmodel" not in p]
    wanted = {m: f"curation-{m}-expFM_natural-" for m in methods}
    todo = [(m, p) for p in names for m, pre in wanted.items() if p.rsplit("/", 1)[-1].startswith(pre)]

    def read(item: tuple[str, str]) -> Record | None:
        method, path = item
        payload = json.loads(fs.cat(path))
        plan = payload["plan"]
        cell = plan["run_name_core"].split("expFM_natural-")[-1]
        if (method, cell) in EXCLUDED_CELLS:
            logger.info("dropping flagged cell %s / %s", method, cell)
            return None
        return Record(
            method=method,
            cell=cell,
            hidden_dim=plan["hidden_dim"],
            total_params=payload["model"]["total_trainable_params"],
            tokens=payload["tokens"]["tokens_trained"],
            bpb={k: payload["eval"][v] for k, v in METRIC_KEY.items() if v in payload["eval"]},
        )

    with ThreadPoolExecutor(32) as pool:
        return [r for r in pool.map(read, todo) if r is not None]


def _legend_label(method: str, base_label: str) -> str:
    return f"{base_label} — {METHODS[method].d_obs_tokens / 1e9:.1f}B tokens"


def _tokens_label(x: float, _pos=None) -> str:
    if x == 0:
        return "0"
    if x >= 1e9:
        return f"{x / 1e9:g}B"
    return f"{x / 1e6:g}M"


def _set_log_ticks(ax) -> None:
    """Token ticks that stay legible at any panel span.

    The 1-2-5 locator collides past ~1.6 decades, and below half a decade matplotlib falls back to
    auto subdivisions that overprint (the 2.9B difference panel spans 0.98B-1.6B and produced
    "1.4B1.5B1.6B"). Decades get one tick each, mid spans get 1-2-5, narrow spans get four
    explicit geometric ticks.
    """
    lo, hi = ax.get_xlim()
    decades = np.log10(hi / lo)
    if decades < 0.5:
        ticks = sorted({float(f"{t:.2g}") for t in np.geomspace(lo, hi, 4)})
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    else:
        subs = (1.0,) if decades > 1.6 else (1.0, 2.0, 5.0)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=subs))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_label))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _params_title(params: int) -> str:
    return f"{params / 1e9:.1f}B params" if params >= 1e9 else f"{params / 1e6:.0f}M params"


def _finish(fig, axes, ylabel: str, out_stem: Path, fig_height: float, legend_ncol: int | None) -> None:
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_ncol or len(labels),
        bbox_to_anchor=(0.5, -0.12 * 7.0 / fig_height),
    )
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    logger.info("Wrote %s.{png,pdf}", out_stem)


def plot_overlay(records: list[Record], dims: list[int], metric: str, args) -> None:
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2 * len(dims), args.fig_height), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, dim in zip(axes, dims, strict=True):
        params = next(r.total_params for r in records if r.hidden_dim == dim)
        for method, (label, color) in METHOD_DISPLAY.items():
            pts = sorted((r.tokens, r.bpb[metric]) for r in records if r.method == method and r.hidden_dim == dim)
            if not pts:
                continue
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(
                x,
                y,
                "-",
                color=color,
                linewidth=METHOD_LINEWIDTH[method],
                alpha=0.95,
                label=_legend_label(method, label),
                zorder=METHOD_ZORDER[method],
            )
            ax.scatter(x, y, color=color, s=85, edgecolors="white", linewidths=0.8, zorder=METHOD_ZORDER[method] + 1)
        ax.set_xscale("log")
        _set_log_ticks(ax)
        ax.set_title(_params_title(params))
        ax.set_xlabel("Tokens trained")
        ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
    _finish(
        fig,
        axes,
        METRIC_LABEL[metric],
        args.output_dir / f"extractor_ablation_{metric}_by_size{args.suffix}",
        args.fig_height,
        args.legend_ncol,
    )


def plot_difference(records: list[Record], dims: list[int], metric: str, args) -> None:
    """Companion figure: upstream minus fork on the matched cells, so a ~0.002 BPB gap is legible."""
    fork = {(r.hidden_dim, r.cell): r for r in records if r.method == FORK}
    up = {(r.hidden_dim, r.cell): r for r in records if r.method == UPSTREAM}
    matched = sorted(set(fork) & set(up), key=lambda k: (k[0], up[k].tokens))
    if not matched:
        raise SystemExit("no matched cells between the two arms")
    color = METHOD_DISPLAY[UPSTREAM][1]
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2 * len(dims), args.fig_height), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, dim in zip(axes, dims, strict=True):
        keys = [k for k in matched if k[0] == dim]
        if keys:
            x = np.array([up[k].tokens for k in keys])
            y = np.array([up[k].bpb[metric] - fork[k].bpb[metric] for k in keys])
            ax.plot(
                x,
                y,
                "-",
                color=color,
                linewidth=3.0,
                alpha=0.95,
                label="Upstream resiliparse minus fork extractor  (below zero = upstream better)",
                zorder=3,
            )
            ax.scatter(x, y, color=color, s=85, edgecolors="white", linewidths=0.8, zorder=4)
            ax.set_title(_params_title(fork[keys[0]].total_params))
        ax.axhline(0.0, color="#444444", linewidth=1.4, zorder=2)
        ax.set_xscale("log")
        _set_log_ticks(ax)
        ax.set_xlabel("Tokens trained")
        ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
    _finish(
        fig,
        axes,
        f"Δ {METRIC_LABEL[metric]}",
        args.output_dir / f"extractor_ablation_{metric}_difference{args.suffix}",
        args.fig_height,
        1,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="bpb", choices=sorted(METRIC_LABEL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dims", type=int, nargs="+", default=None, help="Widths to plot (default: all matched).")
    parser.add_argument("--fig-height", type=float, default=7.0)
    parser.add_argument("--legend-ncol", type=int, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--difference", action="store_true", help="Also write the upstream-minus-fork figure.")
    parser.add_argument(
        "--matched-only", action="store_true", help="Drop fork cells the upstream arm has not trained yet."
    )
    args = parser.parse_args(argv)

    records = [r for r in load_records([FORK, UPSTREAM]) if args.metric in r.bpb]
    if not records:
        raise SystemExit("No result records loaded.")
    if args.matched_only:
        shared = {(r.hidden_dim, r.cell) for r in records if r.method == UPSTREAM}
        records = [r for r in records if (r.hidden_dim, r.cell) in shared]
    dims = sorted({r.hidden_dim for r in records if r.method == UPSTREAM})
    if args.dims:
        dims = [d for d in dims if d in set(args.dims)]
    records = [r for r in records if r.hidden_dim in set(dims)]
    logger.info("%d records over widths %s", len(records), dims)

    plt.rcParams.update(PAPER_RCPARAMS)
    plot_overlay(records, dims, args.metric, args)
    if args.difference:
        plot_difference(records, dims, args.metric, args)


if __name__ == "__main__":
    main()
