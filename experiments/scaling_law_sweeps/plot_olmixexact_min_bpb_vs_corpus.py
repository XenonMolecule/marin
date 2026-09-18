# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter of the lowest bpb each arm attains (any training budget, model sizes up to 2.9B by
default) against the arm's corpus size in tokens. One directly labelled point per arm, log-x.

olmix51 uses the strict complete-task-set loader from the by-size BPB plot. The uncheatable
bpb metric reads the W&B cache the stacked figure uses (refresh it via the uncheatable
plotter); uncheatable loss joins the same runs to their natural-results summaries on GCS.

    uv run python -m experiments.scaling_law_sweeps.plot_olmixexact_min_bpb_vs_corpus \
        --metric olmix51_bpb \
        --relabel "lpv11_10k_mix_olmixexact_lambda0p01=Generative Curation" \
                  "lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01=GC Cascade" --suffix _6arm
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from experiments.scaling_law_sweeps.curation_plan import METHODS
from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS
from experiments.scaling_law_sweeps.plot_olmixexact_bpb_by_size import METHOD_DISPLAY, _tokens_label
from experiments.scaling_law_sweeps.plot_olmixexact_uncheatable_by_size import CACHE
from experiments.scaling_law_sweeps.plot_olmo_bpb_by_size_10k import (
    DEFAULT_TOKENS_PREFIX,
    EXCLUDED_RUN_STEMS,
    _load_records,
    _open_gcs,
)

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "olmo_bpb_10k" / "paper"
DEFAULT_METHODS = [
    "dclm_10k_mix_olmixexact_lambda0p01",
    "nemotron_full_10k_mix_olmixexact_lambda0p01",
    "resiliparse_10k_mix_olmixexact_lambda0p01",
    "lpv11_10k_mix_olmixexact_lambda0p01",
    "lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01",
    "fineweb_cc_10k_mix_olmixexact_lambda0p01",
]
# The 8.1B (d3584) ladder only exists for some arms, so the default excludes it.
DEFAULT_DIMS = [512, 1536, 2432]
# Labels sit in line with their point, to the right by default. These arms prefer the left
# side, but only when the point actually lands in the right part of the axis (past
# LEFT_LABEL_MIN_FRACTION of the span), so a linear axis stretched by a bigger corpus does
# not push their label into the y axis.
PREFER_LEFT_LABEL = {
    "resiliparse_10k_mix_olmixexact_lambda0p01",
    "lpv11_10k_mix_olmixexact_lambda0p01",
}
LEFT_LABEL_MIN_FRACTION = 0.6
LABEL_GAP_POINTS = 14
# Released full-corpus token counts shown after the label (DCLM-baseline 4T, Nemotron-CC
# 6.3T, FineWeb 15T, FineWeb-Edu 1.3T). Unreleased arms get no count: a linear scale-up of
# the 10,364-WARC yield ignores how dedup survival falls with corpus size, so it is not shown.
RELEASED_TOKENS: dict[str, float] = {
    "dclm_10k_mix_olmixexact_lambda0p01": 4.0e12,
    "nemotron_full_10k_mix_olmixexact_lambda0p01": 6.3e12,
    "fineweb_cc_10k_mix_olmixexact_lambda0p01": 15e12,
    "fineweb_edu_10k_mix_olmixexact_lambda0p01": 1.3e12,
}


def _full_corpus_label(method: str, label: str) -> str:
    released = RELEASED_TOKENS.get(method)
    if released is None:
        return label
    return f"{label} ({released / 1e12:g}T)"


METRIC_YLABEL = {
    "olmix51_bpb": "Best BPB (olmix 51-task suite)",
    "uncheatable_macro_bpb": "Best BPB (Uncheatable Eval)",
    "uncheatable_macro_loss": "Lowest Val Loss (Uncheatable Eval)",
}
UNCHEATABLE_LOSS_KEY = "eval/uncheatable_eval/macro_loss"


def _points(metric: str, methods: set[str], dims: set[int]) -> list[tuple[str, float, str]]:
    """(method, bpb, run_stem) triples for every eligible run of `metric`."""
    if metric == "olmix51_bpb":
        return [
            (r.method, r.bpb[metric], r.run_stem)
            for r in _load_records(methods)
            if metric in r.bpb and r.hidden_dim in dims
        ]
    rows = [
        r
        for r in json.loads(CACHE.read_text())
        if r["method"] in methods and r["dim"] in dims and r["stem"] not in EXCLUDED_RUN_STEMS
    ]
    if metric == "uncheatable_macro_bpb":
        return [(r["method"], r["macro_bpb"], r["stem"]) for r in rows]

    def _loss(row: dict) -> tuple[str, float, str] | None:
        with _open_gcs(DEFAULT_TOKENS_PREFIX + row["stem"] + ".json") as fh:
            value = (json.load(fh).get("eval") or {}).get(UNCHEATABLE_LOSS_KEY)
        if value is None:
            logger.warning("no %s in natural summary for %s", UNCHEATABLE_LOSS_KEY, row["stem"])
            return None
        return (row["method"], float(value), row["stem"])

    with ThreadPoolExecutor(max_workers=16) as ex:
        return [pt for pt in ex.map(_loss, rows) if pt is not None]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric", default="olmix51_bpb", choices=sorted(METRIC_YLABEL))
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)
    ap.add_argument("--relabel", nargs="+", default=[], metavar="METHOD=LABEL")
    ap.add_argument("--linear-x", action="store_true", help="Linear corpus-size axis instead of log.")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.relabel)
    display = {m: (overrides.get(m, METHOD_DISPLAY[m][0]), METHOD_DISPLAY[m][1]) for m in args.methods}

    best: dict[str, tuple[float, str]] = {}
    for method, bpb, stem in _points(args.metric, set(args.methods), set(args.dims)):
        if method not in best or bpb < best[method][0]:
            best[method] = (bpb, stem)

    plt.rcParams.update(PAPER_RCPARAMS)
    fig, ax = plt.subplots(figsize=(11, 7.0))
    for method, (label, color) in display.items():
        if method not in best:
            logger.warning("no %s records for %s", args.metric, method)
            continue
        x, (y, stem) = METHODS[method].d_obs_tokens, best[method]
        logger.info("%-22s min %s=%.4f  corpus=%.1fB  (%s)", label, args.metric, y, x / 1e9, stem)
        ax.scatter([x], [y], color=color, s=260, edgecolors="white", linewidths=1.0, zorder=4)

    xs = [METHODS[m].d_obs_tokens for m in display]
    if args.linear_x:
        ax.set_xlim(0, max(xs) * 1.15)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
    else:
        ax.set_xscale("log")
        ax.set_xlim(min(xs) * 0.45, max(xs) * 1.8)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_label))

    for method, (label, _) in display.items():
        if method not in best:
            continue
        x, y = METHODS[method].d_obs_tokens, best[method][0]
        fraction = ax.transLimits.transform((x, y))[0]  # 0..1 across the axis after scaling
        put_left = method in PREFER_LEFT_LABEL and fraction > LEFT_LABEL_MIN_FRACTION
        ax.annotate(
            _full_corpus_label(method, label),
            (x, y),
            xytext=(-LABEL_GAP_POINTS if put_left else LABEL_GAP_POINTS, 0),
            textcoords="offset points",
            ha="right" if put_left else "left",
            va="center",
            fontsize=17,
            color="#222222",
            zorder=5,
        )
    ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
    ax.set_xlabel("Corpus size (tokens)")
    ax.set_ylabel(METRIC_YLABEL[args.metric])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"olmixexact_min_{args.metric}_vs_corpus{args.suffix}{'_linear' if args.linear_x else ''}"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    logger.info("wrote %s.{png,pdf}", stem)


if __name__ == "__main__":
    main()
