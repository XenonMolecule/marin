# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paper figure: OlmixExact mixtures vs their own natural baselines, per model size.

Companion to ``plot_olmixexact_bpb_by_size`` (the four mixtures head to head); this
figure asks the WITHIN-corpus question instead — does OLMIX re-weighting beat training
on the corpus's natural distribution? Solid lines are the mixture arms, dashed lines
the natural baselines, one hue per corpus (repo convention: a mix arm reads as "that
corpus, re-weighted").

Default metric is ``olmix47_bpb`` (the paper's 51-task suite minus its 4 MMLU category
tasks): the natural baselines were evaluated before MMLU joined the suite, so 47 tasks
is the largest subset on which the comparison needs no re-evaluation. Once the MMLU
top-up wave has merged into every baseline's results.json, pass ``--metric olmix51_bpb``
for the exact-suite version.

Usage::

    export SSL_CERT_FILE=$(.venv/bin/python -m certifi)
    uv run python -m experiments.scaling_law_sweeps.plot_olmixexact_vs_natural_by_size
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

from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS
from experiments.scaling_law_sweeps.plot_olmixexact_bpb_by_size import _tokens_label
from experiments.scaling_law_sweeps.plot_olmo_bpb_by_size_10k import _load_records

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "olmo_bpb_10k" / "paper"
DEFAULT_METRIC = "olmix47_bpb"

# (mix method, natural method as parsed by _STEM_RE, display name, corpus hue).
# The natural 10k baselines lose their `_10k` suffix in the stem regex.
PAIRS: list[tuple[str, str, str, str]] = [
    ("dclm_10k_mix_olmixexact_lambda0p01", "dclm", "DCLM-baseline", "#1f77b4"),
    ("high_quality_10k_mix_olmixexact_lambda0p01", "high_quality", "Old Spec (HQ)", "#2ca02c"),
    ("lpv11_fastpipe_v1_10k_mix_olmixexact_lambda0p01", "lpv11_fastpipe_v1", "New Spec (lpv11)", "#d62728"),
    ("resiliparse_10k_mix_olmixexact_lambda0p01", "resiliparse", "Resiliparse", "#9467bd"),
]

METRIC_LABEL = {
    "olmix47_bpb": "BPB (olmix suite, 47 of 51 tasks)",
    "olmix51_bpb": "BPB (olmix 51-task suite)",
}

# TEMPORARY exclusions (user request 2026-08-28): the lpv11 natural d512 bump at
# 2.6-4.3B tokens. Diagnosed, NOT a bug: held-out LM metrics (eval loss, lima,
# uncheatable macro) improve monotonically through these cells; the bump is
# +0.3..+1.0 bpb swings on the basic_skills_* synthetic probes, which are
# high-variance for 157M models. Excluded for visual clarity only — remove these
# entries to restore the honest full series.
EXCLUDED_RUNS: frozenset[str] = frozenset(
    {
        "curation-lpv11_fastpipe_v1-expFM_natural-2e+18-d512-L6-B16",
        "curation-lpv11_fastpipe_v1-expFM_natural-3e+18-d512-L6-B32",
    }
)


def _draw_figure(records: list, pairs: list, metric: str, out_stem: Path, legend_ncol: int) -> None:
    """One mix-vs-natural figure over `pairs`, paneled by model size, PNG + PDF."""
    plt.rcParams.update(PAPER_RCPARAMS)
    # Panels limited to sizes the MIX sweep trains (36-cell grid tops out at d3584);
    # the baselines' extra sizes would add panels with nothing to compare against.
    mix_methods = {pair[0] for pair in pairs}
    dims = sorted({r.hidden_dim for r in records if r.method in mix_methods})
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2 * len(dims), 7.0), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, dim in zip(axes, dims):
        params = next(r.total_params for r in records if r.hidden_dim == dim)
        for mix_m, nat_m, label, color in pairs:
            for method, style, suffix in ((mix_m, "-", "mix"), (nat_m, "--", "natural")):
                pts = sorted((r.tokens, r.bpb[metric]) for r in records if r.method == method and r.hidden_dim == dim)
                if not pts:
                    continue
                x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                ax.plot(x, y, style, color=color, linewidth=3.0, alpha=0.95, label=f"{label} ({suffix})", zorder=3)
                ax.scatter(x, y, color=color, s=70, edgecolors="white", linewidths=0.8, zorder=4)
        ax.set_xscale("log")
        # Baseline grids span up to ~3 decades of tokens; 1-2-5 ticks collide there,
        # so widen to decades-only when the panel is wide.
        lo, hi = ax.get_xlim()
        subs = (1.0,) if np.log10(hi / lo) > 1.6 else (1.0, 2.0, 5.0)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=subs))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_label))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(f"{params / 1e9:.1f}B params" if params >= 1e9 else f"{params / 1e6:.0f}M params")
        ax.set_xlabel("Tokens trained")
        ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
    axes[0].set_ylabel(METRIC_LABEL[metric])

    handles, labels = [], []
    for ax in axes:
        for h, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in labels:
                handles.append(h)
                labels.append(ll)
    order = [f"{label} ({suffix})" for _, _, label, _ in pairs for suffix in ("mix", "natural")]
    ranked = sorted(zip(labels, handles), key=lambda lh: order.index(lh[0]) if lh[0] in order else 99)
    fig.legend(
        [h for _, h in ranked],
        [ll for ll, _ in ranked],
        loc="lower center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    logger.info("Wrote %s.{png,pdf}", out_stem)


# Filename slug per corpus for the single-corpus figures.
_SLUG = {
    "dclm_10k_mix_olmixexact_lambda0p01": "dclm",
    "high_quality_10k_mix_olmixexact_lambda0p01": "hq",
    "lpv11_fastpipe_v1_10k_mix_olmixexact_lambda0p01": "lpv11",
    "resiliparse_10k_mix_olmixexact_lambda0p01": "resiliparse",
}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default=DEFAULT_METRIC, choices=sorted(METRIC_LABEL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    wanted = {m for pair in PAIRS for m in pair[:2]}
    records = [r for r in _load_records(wanted) if args.metric in r.bpb and r.run_stem not in EXCLUDED_RUNS]
    if not records:
        raise SystemExit("No bpb records loaded.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _draw_figure(
        records, PAIRS, args.metric, args.output_dir / f"olmixexact_vs_natural_{args.metric}_by_size", legend_ncol=4
    )
    for pair in PAIRS:
        _draw_figure(
            records,
            [pair],
            args.metric,
            args.output_dir / f"olmixexact_vs_natural_{_SLUG[pair[0]]}_{args.metric}_by_size",
            legend_ncol=2,
        )


if __name__ == "__main__":
    main()
