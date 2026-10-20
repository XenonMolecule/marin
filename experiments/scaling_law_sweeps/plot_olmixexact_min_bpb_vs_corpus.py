# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter of the lowest bpb each arm attains (any training budget, model sizes up to 2.9B by
default) against the arm's corpus size in tokens. One directly labelled point per arm, log-x.

olmix51 uses the strict complete-task-set loader from the by-size BPB plot and is always live.
The uncheatable metrics come from the W&B cache the stacked figure uses (bpb straight from the
cache, loss joined from the natural-results summaries on GCS). Pass ``--refresh-wandb`` before
regenerating paper figures: without it the plot silently reuses whatever runs the last pull
saw, and the script warns with the cache's age.

Every metric's resolved runs are also cached locally (``points_cache_<metric>.json`` next to the
figures) so styling iterations take seconds; ``--refresh-data`` re-scans GCS (and
``--refresh-wandb`` re-pulls W&B first). ``--paper`` renders every paper variant in one process
with the per-variant label placement baked in.

    uv run python -m experiments.scaling_law_sweeps.plot_olmixexact_min_bpb_vs_corpus \
        --metric olmix51_bpb \
        --relabel "lpv11_10k_mix_olmixexact_lambda0p01=Generative Curation" \
                  "lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01=GC Cascade" --suffix _6arm
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from experiments.scaling_law_sweeps.curation_plan import METHODS
from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS
from experiments.scaling_law_sweeps.plot_olmixexact_bpb_by_size import METHOD_DISPLAY, _tokens_label
from experiments.scaling_law_sweeps.plot_olmixexact_uncheatable_by_size import CACHE, _join_tokens, _pull_wandb
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
ABOVE_LABEL_RIGHT_ALIGN_FRACTION = 0.8
LABEL_GAP_POINTS = 24
# Paper figure: everything scaled up from PAPER_RCPARAMS so it stays legible at column width.
POINT_LABEL_FONTSIZE = 22
MARKER_AREA = 1100
STAR_AREA = 2600  # a star of equal area reads smaller than a disc; oversize it
FIGURE_RCPARAMS = {"axes.labelsize": 28, "xtick.labelsize": 24, "ytick.labelsize": 24}
OURS_TAG = "(Ours)"
# Released full-corpus token counts shown after the label (DCLM-baseline 4T, Nemotron-CC
# 6.3T, FineWeb 15T, FineWeb-Edu 1.3T).
RELEASED_TOKENS: dict[str, float] = {
    "dclm_10k_mix_olmixexact_lambda0p01": 4.0e12,
    "nemotron_full_10k_mix_olmixexact_lambda0p01": 6.3e12,
    "fineweb_cc_10k_mix_olmixexact_lambda0p01": 15e12,
    "fineweb_edu_10k_mix_olmixexact_lambda0p01": 1.3e12,
}
# Unreleased arms: full-Common-Crawl projections, shown with a tilde. These extrapolate the
# 10,364-WARC yield with the measured dedup-survival decay, not a linear scale-up.
PROJECTED_TOKENS: dict[str, float] = {
    "lpv11_10k_mix_olmixexact_lambda0p01": 25e12,
    "lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01": 20e12,
    "resiliparse_10k_mix_olmixexact_lambda0p01": 75e12,
}


def _full_corpus_label(method: str, label: str) -> str:
    """Append the corpus size to the label's first line: `(4T)` released, `(~25T)` projected."""
    if method in RELEASED_TOKENS:
        count = f"({RELEASED_TOKENS[method] / 1e12:g}T)"
    elif method in PROJECTED_TOKENS:
        count = f"(~{PROJECTED_TOKENS[method] / 1e12:g}T)"
    else:
        return label
    first, sep, rest = label.partition("\n")
    return f"{first} {count}{sep}{rest}"


METRIC_YLABEL = {
    "olmix51_bpb": "← Higher Quality (Min olmix BPB)",
    "uncheatable_macro_bpb": "← Higher Quality (Min Uncheatable BPB)",
    "uncheatable_macro_loss": "← Higher Quality (Min Val Loss)",
}
UNCHEATABLE_LOSS_KEY = "eval/uncheatable_eval/macro_loss"
STALE_CACHE_HOURS = 12.0


def _uncheatable_rows(refresh: bool) -> list[dict]:
    if refresh:
        rows = _join_tokens(_pull_wandb())
        CACHE.write_text(json.dumps(rows, indent=0))
        logger.info("refreshed W&B cache: %d runs", len(rows))
        return rows
    age_hours = (time.time() - CACHE.stat().st_mtime) / 3600
    if age_hours > STALE_CACHE_HOURS:
        logger.warning("W&B cache is %.0f h old; pass --refresh-wandb before reporting these numbers", age_hours)
    return json.loads(CACHE.read_text())


def _pareto_frontier(points: dict[str, tuple[float, float]]) -> set[str]:
    """Arms not dominated by any other on (more tokens, lower metric)."""
    return {
        m
        for m, (x, y) in points.items()
        if not any(ox >= x and oy <= y and (ox, oy) != (x, y) for om, (ox, oy) in points.items() if om != m)
    }


def _scan_points(metric: str, refresh_wandb: bool) -> list[dict]:
    """Every run of `metric` for every arm in METHOD_DISPLAY, straight from GCS / W&B."""
    if metric == "olmix51_bpb":
        return [
            {"method": r.method, "dim": r.hidden_dim, "value": r.bpb[metric], "stem": r.run_stem}
            for r in _load_records(set(METHOD_DISPLAY))
            if metric in r.bpb
        ]
    rows = [r for r in _uncheatable_rows(refresh_wandb) if r["stem"] not in EXCLUDED_RUN_STEMS]
    if metric == "uncheatable_macro_bpb":
        return [{"method": r["method"], "dim": r["dim"], "value": r["macro_bpb"], "stem": r["stem"]} for r in rows]

    def _loss(row: dict) -> dict | None:
        with _open_gcs(DEFAULT_TOKENS_PREFIX + row["stem"] + ".json") as fh:
            value = (json.load(fh).get("eval") or {}).get(UNCHEATABLE_LOSS_KEY)
        if value is None:
            logger.warning("no %s in natural summary for %s", UNCHEATABLE_LOSS_KEY, row["stem"])
            return None
        return {"method": row["method"], "dim": row["dim"], "value": float(value), "stem": row["stem"]}

    with ThreadPoolExecutor(max_workers=16) as ex:
        return [pt for pt in ex.map(_loss, rows) if pt is not None]


def _points(metric: str, refresh_data: bool, refresh_wandb: bool) -> list[dict]:
    """`_scan_points` behind a local JSON cache so styling iterations do not touch GCS."""
    cache = OUT_DIR / f"points_cache_{metric}.json"
    if not (refresh_data or refresh_wandb) and cache.exists():
        age_hours = (time.time() - cache.stat().st_mtime) / 3600
        if age_hours > STALE_CACHE_HOURS:
            logger.warning("%s cache is %.0f h old; pass --refresh-data before reporting", metric, age_hours)
        return json.loads(cache.read_text())
    points = _scan_points(metric, refresh_wandb)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(points, indent=0))
    logger.info("cached %d %s runs to %s", len(points), metric, cache.name)
    return points


@dataclass(frozen=True)
class Variant:
    """One figure: which metric/arms to draw and the per-arm label placement that keeps it clean."""

    metric: str
    suffix: str
    methods: tuple[str, ...] = tuple(DEFAULT_METHODS)
    linear_x: bool = False
    # Right edge of the x-axis in tokens; None pads past the largest corpus.
    xmax: float | None = None
    label_right: frozenset[str] = frozenset()
    label_above: frozenset[str] = frozenset()
    label_dy: dict[str, float] = field(default_factory=dict)
    # Pareto-frontier arms as stars instead of discs (off: every arm is a plain disc).
    stars: bool = False
    # Replaces METRIC_YLABEL, e.g. a two-line label naming a single-size cut.
    ylabel: str | None = None


_GC = "lpv11_10k_mix_olmixexact_lambda0p01"
_RESILIPARSE = "resiliparse_10k_mix_olmixexact_lambda0p01"
_FIVE_ARMS = tuple(m for m in DEFAULT_METHODS if m != _RESILIPARSE)
PAPER_RELABEL = {
    _GC: "GC LLM\n(Ours)",
    "lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01": "GC Cascade\n(Ours)",
    "dclm_10k_mix_olmixexact_lambda0p01": "DCLM",
}
LINEAR_XMAX = 4e11
PAPER_VARIANTS = (
    Variant("olmix51_bpb", "_6arm", label_above=frozenset({_RESILIPARSE})),
    Variant("uncheatable_macro_bpb", "_6arm", label_above=frozenset({_RESILIPARSE})),
    Variant("uncheatable_macro_loss", "_6arm", label_above=frozenset({_RESILIPARSE})),
    Variant("uncheatable_macro_loss", "_6arm", linear_x=True, xmax=LINEAR_XMAX, label_above=frozenset({_RESILIPARSE})),
    Variant("uncheatable_macro_loss", "_5arm_noresili", methods=_FIVE_ARMS),
    Variant("uncheatable_macro_loss", "_5arm_noresili", methods=_FIVE_ARMS, linear_x=True, xmax=LINEAR_XMAX),
)


def render(v: Variant, points: list[dict], dims: set[int], relabel: dict[str, str]) -> None:
    display = {m: (relabel.get(m, METHOD_DISPLAY[m][0]), METHOD_DISPLAY[m][1]) for m in v.methods}
    best: dict[str, tuple[float, str]] = {}
    for pt in points:
        if pt["method"] not in display or pt["dim"] not in dims:
            continue
        if pt["method"] not in best or pt["value"] < best[pt["method"]][0]:
            best[pt["method"]] = (pt["value"], pt["stem"])

    frontier = _pareto_frontier({m: (METHODS[m].d_obs_tokens, best[m][0]) for m in display if m in best})
    logger.info("pareto frontier: %s", ", ".join(display[m][0].replace("\n", " ") for m in frontier))

    plt.rcParams.update(PAPER_RCPARAMS)
    plt.rcParams.update(FIGURE_RCPARAMS)
    fig, ax = plt.subplots(figsize=(13, 10))
    for method, (label, color) in display.items():
        if method not in best:
            logger.warning("no %s records for %s", v.metric, method)
            continue
        x, (y, stem) = METHODS[method].d_obs_tokens, best[method]
        logger.info("%-22s min %s=%.4f  corpus=%.1fB  (%s)", label.replace("\n", " "), v.metric, y, x / 1e9, stem)
        on_frontier = method in frontier and v.stars
        ax.scatter(
            [x],
            [y],
            color=color,
            marker="*" if on_frontier else "o",
            s=STAR_AREA if on_frontier else MARKER_AREA,
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
        )

    xs = [METHODS[m].d_obs_tokens for m in display]
    if v.linear_x:
        ax.set_xlim(0, v.xmax or max(xs) * 1.32)
        # With a fixed right edge, step 100B so the last tick lands on the edge.
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1e11) if v.xmax else mticker.MaxNLocator(nbins=7))
    else:
        ax.set_xscale("log")
        ax.set_xlim(min(xs) * 0.45, v.xmax or max(xs) * 1.8)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_label))

    for method, (label, _) in display.items():
        if method not in best:
            continue
        x, y = METHODS[method].d_obs_tokens, best[method][0]
        fraction = ax.transLimits.transform((x, y))[0]  # 0..1 across the axis after scaling
        put_left = method in PREFER_LEFT_LABEL and fraction > LEFT_LABEL_MIN_FRACTION and method not in v.label_right
        dy = v.label_dy.get(method, 0.0)
        if method in v.label_above:
            # A long label centred over a far-right point overhangs the axes; hang it to the left.
            above_ha = "right" if fraction > ABOVE_LABEL_RIGHT_ALIGN_FRACTION else "center"
            above_dx = LABEL_GAP_POINTS if above_ha == "right" else 0
            xytext, ha, va = (above_dx, LABEL_GAP_POINTS + dy), above_ha, "bottom"
        else:
            xytext = (-LABEL_GAP_POINTS if put_left else LABEL_GAP_POINTS, dy)
            ha, va = ("right" if put_left else "left"), "center"
        ax.annotate(
            _full_corpus_label(method, label),
            (x, y),
            xytext=xytext,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=POINT_LABEL_FONTSIZE,
            fontweight="bold" if OURS_TAG in label else "normal",
            color="#222222",
            zorder=5,
        )
    ax.grid(True, which="major", linewidth=1.1, alpha=0.65)
    ax.set_xlabel("More Data (tokens from DCLM-400M-1x pool) \u2192")
    ax.set_ylabel(v.ylabel or METRIC_YLABEL[v.metric])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"olmixexact_min_{v.metric}_vs_corpus{v.suffix}{'_linear' if v.linear_x else ''}"
    # The axes' tight bbox omits the rotated y-label, which is taller than the plot area.
    extra = [ax.yaxis.label]
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.2, bbox_extra_artists=extra)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, bbox_extra_artists=extra)
    plt.close(fig)
    logger.info("wrote %s.{png,pdf}", stem)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", action="store_true", help="Render every PAPER_VARIANTS figure in one process.")
    ap.add_argument("--metric", default="olmix51_bpb", choices=sorted(METRIC_YLABEL))
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)
    ap.add_argument("--relabel", nargs="+", default=[], metavar="METHOD=LABEL")
    ap.add_argument("--linear-x", action="store_true", help="Linear corpus-size axis instead of log.")
    ap.add_argument("--refresh-data", action="store_true", help="Re-scan GCS instead of using the local points cache.")
    ap.add_argument(
        "--refresh-wandb", action="store_true", help="Re-pull the uncheatable W&B cache first; implies --refresh-data."
    )
    ap.add_argument(
        "--label-right",
        nargs="+",
        default=[],
        metavar="METHOD",
        help="Force these arms' labels to the right of the point, overriding the placement rule.",
    )
    ap.add_argument(
        "--label-above",
        nargs="+",
        default=[],
        metavar="METHOD",
        help="Centre these arms' labels above the point instead of beside it.",
    )
    ap.add_argument(
        "--label-dy",
        nargs="+",
        default=[],
        metavar="METHOD=POINTS",
        help="Vertical nudge for an arm's label in points (negative = down), for collisions.",
    )
    ap.add_argument("--suffix", default="")
    ap.add_argument("--xmax", type=float, default=None, help="Right edge of the x-axis in tokens, e.g. 4e11.")
    ap.add_argument("--stars", action="store_true", help="Draw Pareto-frontier arms as stars instead of discs.")
    ap.add_argument("--ylabel", default=None, help="Override the y-axis label (literal '\\n' starts a new line).")
    args = ap.parse_args()

    if args.paper:
        variants = PAPER_VARIANTS
        relabel = dict(PAPER_RELABEL)
    else:
        variants = (
            Variant(
                args.metric,
                args.suffix,
                methods=tuple(args.methods),
                linear_x=args.linear_x,
                xmax=args.xmax,
                label_right=frozenset(args.label_right),
                label_above=frozenset(args.label_above),
                label_dy={m: float(d) for m, d in (kv.split("=", 1) for kv in args.label_dy)},
                stars=args.stars,
                ylabel=None if args.ylabel is None else args.ylabel.replace("\\n", "\n"),
            ),
        )
        relabel = dict(kv.split("=", 1) for kv in args.relabel)

    points_by_metric: dict[str, list[dict]] = {}
    for v in variants:
        if v.metric not in points_by_metric:
            points_by_metric[v.metric] = _points(v.metric, args.refresh_data, args.refresh_wandb)
        render(v, points_by_metric[v.metric], set(args.dims), relabel)


if __name__ == "__main__":
    main()
