# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paper figure: OlmixExact mixtures on the olmix 51-task BPB suite, per model size.

Same visual language as ``plot_loss_vs_tokens_by_size`` (the paper's loss figures):
``PAPER_RCPARAMS``, one panel per hidden size, tokens-trained log x-axis, thick
white-edged markers, a shared bottom legend, PNG + PDF output, and no title — model
sizes and the metric are stated in the caption. Data comes from
``plot_olmo_bpb_by_size_10k._load_records`` (all-region olmo_bpb results joined to
training tokens), restricted to the four ``*_mix_olmixexact_lambda0p01`` arms.

Usage::

    export SSL_CERT_FILE=$(.venv/bin/python -m certifi)
    uv run python -m experiments.scaling_law_sweeps.plot_olmixexact_bpb_by_size
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiments.scaling_law_sweeps.curation_plan import METHODS
from experiments.scaling_law_sweeps.plot_core_v2_by_size_10k import OLMIXEXACT_COMPARE_METHODS
from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS
from experiments.scaling_law_sweeps.plot_olmo_bpb_by_size_10k import _load_records

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "olmo_bpb_10k" / "paper"
DEFAULT_METRIC = "olmix51_bpb"

# Paper display names + canonical curation palette (colors locked repo-wide).
# Legend appends each corpus's pool size (d_obs) read from the method registry.
METHOD_DISPLAY: dict[str, tuple[str, str]] = {
    "dclm_10k_mix_olmixexact_lambda0p01": ("DCLM-baseline", "#1f77b4"),
    "high_quality_10k_mix_olmixexact_lambda0p01": ("Old Spec (HQ)", "#2ca02c"),
    "lpv11_fastpipe_v1_10k_mix_olmixexact_lambda0p01": ("New Spec (lpv11)", "#d62728"),
    # The TRUE llm_pipeline_v1_1 LLM extraction (not the fastpipe cascade); brown is the
    # only unclaimed tab10 hue that reads against the locked assignments above.
    "lpv11_10k_mix_olmixexact_lambda0p01": ("lpv11 (LLM extract)", "#74c476"),
    "lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01": ("fastpipe_v2", "#006d2c"),
    # Same corpus and weights as fastpipe_v2, upstream-resiliparse text instead of the fork's.
    # Locked resiliparse purple: this arm IS upstream resiliparse extraction.
    "lpv11_fastpipe_v2_upstreamrs_10k_mix_olmixexact_lambda0p01": ("fastpipe_v2 (upstream rslp)", "#9467bd"),
    # lpv11 documents, fork extraction. Burnt umber, not a third green: it is the only hue left
    # that clears the colorblind and normal-vision separation floors against every locked color
    # here, including lpv11's own #74c476 (deutan dE 30.8, normal dE 36.3).
    "lpv11_forkrs_10k_mix_olmixexact_lambda0p01": ("lpv11 (fork extract)", "#7f3b08"),
    "resiliparse_10k_mix_olmixexact_lambda0p01": ("Resiliparse", "#9467bd"),
    # Locked convention gives nemotron orange; its paper red would collide with lpv11 here.
    "nemotron_full_10k_mix_olmixexact_lambda0p01": ("Nemotron-CC", "#ff7f0e"),
    # Paper pink for fineweb_cc; cyan for fineweb_edu (its paper orange is taken by nemotron here).
    "fineweb_cc_10k_mix_olmixexact_lambda0p01": ("FineWeb", "#e377c2"),
    "fineweb_edu_10k_mix_olmixexact_lambda0p01": ("FineWeb-Edu", "#17becf"),
}


# GC/Distilled-GC draw thicker than the field (user 2026-09-15 palette revision:
# lpv11 true = light green, fastpipe_v2 = dark green, both emphasized).
METHOD_LINEWIDTH = {
    "lpv11_10k_mix_olmixexact_lambda0p01": 4.2,
    "lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01": 4.2,
}


def _legend_label(method: str, base_label: str) -> str:
    return f"{base_label} — {METHODS[method].d_obs_tokens / 1e9:.1f}B tokens"


def _tokens_label(x: float, _pos=None) -> str:
    """Human token counts for axis ticks: 0, 50M, 500M, 1B, 2B — never scientific notation."""
    if x == 0:
        return "0"
    if x >= 1e9:
        return f"{x / 1e9:g}B"
    return f"{x / 1e6:g}M"


METRIC_LABEL = {
    "olmix51_bpb": "BPB (olmix 51-task suite)",
    "macro_bpb": "BPB (OLMo Base-Easy macro)",
}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default=DEFAULT_METRIC, choices=sorted(METRIC_LABEL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of METHOD_DISPLAY keys (default: all).")
    parser.add_argument("--dims", type=int, nargs="+", default=None, help="Widths to plot (default: all present).")
    parser.add_argument("--legend-size", type=float, default=None, help="Legend font size override.")
    parser.add_argument("--legend-ncol", type=int, default=None, help="Legend columns (default: auto).")
    parser.add_argument("--fig-height", type=float, default=7.0, help="Figure height in inches (panels are 7.2in wide).")
    parser.add_argument("--suffix", default="", help="Filename suffix for filtered variants.")
    parser.add_argument(
        "--relabel",
        nargs="+",
        default=[],
        metavar="METHOD=LABEL",
        help="Override legend labels, e.g. lpv11_fastpipe_v1_10k_mix_olmixexact_lambda0p01='Generative Curation (ours)'.",
    )
    parser.add_argument(
        "--recolor",
        nargs="+",
        default=[],
        metavar="METHOD=COLOR",
        help="Override line colors, e.g. high_quality_10k_mix_olmixexact_lambda0p01='#7f7f7f'.",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="One line drawn above the panels saying what is held fixed, e.g. "
        "'Held fixed: LLM filter — varying: the extractor'. Keeps a controlled comparison self-describing.",
    )
    parser.add_argument(
        "--marker",
        nargs="+",
        default=[],
        metavar="METHOD=SHAPE",
        help="Override point markers (matplotlib marker codes, e.g. 's', '^', 'D'). Pairs with a shared "
        "hue so two runs in the same group read as a group while staying individually distinguishable.",
    )
    parser.add_argument(
        "--linestyle",
        nargs="+",
        default=[],
        metavar="METHOD=STYLE",
        help="Override line styles (matplotlib style strings, e.g. '--', ':'). Lets a 2x2 design encode "
        "one factor as hue and the other as dash, so both groupings read at once.",
    )
    args = parser.parse_args(argv)

    records = [r for r in _load_records(OLMIXEXACT_COMPARE_METHODS) if args.metric in r.bpb]
    if not records:
        raise SystemExit("No bpb records loaded.")

    plt.rcParams.update(PAPER_RCPARAMS)
    display = METHOD_DISPLAY
    overrides = dict(kv.split("=", 1) for kv in args.relabel)
    color_overrides = dict(kv.split("=", 1) for kv in args.recolor)
    style_overrides = dict(kv.split("=", 1) for kv in args.linestyle)
    marker_overrides = dict(kv.split("=", 1) for kv in args.marker)
    unknown_r = [
        m for m in {**overrides, **color_overrides, **style_overrides, **marker_overrides} if m not in METHOD_DISPLAY
    ]
    if unknown_r:
        raise SystemExit(f"unknown --relabel/--recolor methods: {unknown_r}")
    display = {m: (overrides.get(m, lbl), color_overrides.get(m, c)) for m, (lbl, c) in display.items()}
    if args.methods:
        unknown = [m for m in args.methods if m not in METHOD_DISPLAY]
        if unknown:
            raise SystemExit(f"unknown methods: {unknown}")
        display = {m: display[m] for m in args.methods}
        records = [r for r in records if r.method in display]
    dims = sorted({r.hidden_dim for r in records})
    if args.dims:
        dims = [d for d in dims if d in set(args.dims)]
        records = [r for r in records if r.hidden_dim in set(dims)]
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2 * len(dims), args.fig_height), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, dim in zip(axes, dims):
        params = next(r.total_params for r in records if r.hidden_dim == dim)
        for method, (label, color) in display.items():
            pts = sorted((r.tokens, r.bpb[args.metric]) for r in records if r.method == method and r.hidden_dim == dim)
            if not pts:
                continue
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            mk = marker_overrides.get(method, "o")
            # marker goes on the line too so the legend handle carries it; it lands exactly under the
            # scatter below, which is what gives every point its white rim.
            ax.plot(
                x,
                y,
                style_overrides.get(method, "-"),
                color=color,
                linewidth=METHOD_LINEWIDTH.get(method, 3.0),
                alpha=0.95,
                marker=mk,
                markersize=9.2,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=_legend_label(method, label),
                zorder=3,
            )
            ax.scatter(x, y, color=color, s=85, marker=mk, edgecolors="white", linewidths=0.8, zorder=4)
        ax.set_xscale("log")
        # 1-2-5 ticks collide once a panel spans more than ~1.6 decades; widen to decades.
        lo, hi = ax.get_xlim()
        subs = (1.0,) if np.log10(hi / lo) > 1.6 else (1.0, 2.0, 5.0)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=subs))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_label))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(f"{params / 1e9:.1f}B params" if params >= 1e9 else f"{params / 1e6:.0f}M params")
        ax.set_xlabel("Tokens trained")
        ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
    axes[0].set_ylabel(METRIC_LABEL[args.metric])
    if args.note:
        # Sits above the per-panel titles; bbox_inches="tight" grows the canvas to fit it.
        fig.text(
            0.5,
            1.0,
            args.note,
            ha="center",
            va="bottom",
            fontsize=plt.rcParams.get("axes.titlesize", 16),
            style="italic",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in labels:
                handles.append(h)
                labels.append(ll)
    ncol = args.legend_ncol or (len(labels) if args.legend_size is None else max(1, min(len(labels), 2)))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, (-0.12 if args.legend_size is None else -0.22) * 7.0 / args.fig_height),
        fontsize=args.legend_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"olmixexact_{args.metric}_by_size{args.suffix}"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    logger.info("Wrote %s.{png,pdf}", stem)


if __name__ == "__main__":
    main()
