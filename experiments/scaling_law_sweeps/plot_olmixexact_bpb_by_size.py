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
    "resiliparse_10k_mix_olmixexact_lambda0p01": ("Resiliparse", "#9467bd"),
}


def _legend_label(method: str, base_label: str) -> str:
    return f"{base_label} — {METHODS[method].d_obs_tokens / 1e9:.1f}B tokens"


def _tokens_label(x: float, _pos=None) -> str:
    """Human token counts for axis ticks: 50M, 500M, 1B, 2B — never scientific notation."""
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
    args = parser.parse_args(argv)

    records = [r for r in _load_records(OLMIXEXACT_COMPARE_METHODS) if args.metric in r.bpb]
    if not records:
        raise SystemExit("No bpb records loaded.")

    plt.rcParams.update(PAPER_RCPARAMS)
    dims = sorted({r.hidden_dim for r in records})
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2 * len(dims), 7.0), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, dim in zip(axes, dims):
        params = next(r.total_params for r in records if r.hidden_dim == dim)
        for method, (label, color) in METHOD_DISPLAY.items():
            pts = sorted((r.tokens, r.bpb[args.metric]) for r in records if r.method == method and r.hidden_dim == dim)
            if not pts:
                continue
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(x, y, "-", color=color, linewidth=3.0, alpha=0.95, label=_legend_label(method, label), zorder=3)
            ax.scatter(x, y, color=color, s=85, edgecolors="white", linewidths=0.8, zorder=4)
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

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in labels:
                handles.append(h)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.12))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"olmixexact_{args.metric}_by_size"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    logger.info("Wrote %s.{png,pdf}", stem)


if __name__ == "__main__":
    main()
