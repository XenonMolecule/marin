# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cultural-target sweep: cultural-suite choice_logprob vs training tokens, one panel per size.

The dedicated plot for the `*_mix_cultural_clp_lambda0p01` training runs (user 2026-09-13:
these arms stay OUT of the olmixexact head-to-head plots). X = training tokens (budget / 6N),
Y = mean raw choice_logprob over the 19-task cultural suite (BLEnD-MCQ, CulturalBench-Easy,
FORK, NormAd), read from `metadata/cultural_target_models/{shots}shot/<run>/results.json`.
Higher is better; the mix objective was exactly this quantity.

    uv run python -m experiments.scaling_law_sweeps.plot_cultural_target_by_size
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "cultural_target"
RESULTS_BUCKETS = (
    "marin-us-east5",
    "marin-us-east1",
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-west4",
    "marin-eu-west4",
)
RESULTS_SUBPATH = "metadata/cultural_target_models/5shot"

# Canonical curation palette (colors locked repo-wide; matches plot_olmixexact_bpb_by_size).
ARM_DISPLAY: dict[str, tuple[str, str]] = {
    "dclm_10k_mix_cultural_clp_lambda0p01": ("DCLM-baseline", "#1f77b4"),
    "lpv11_10k_mix_cultural_clp_lambda0p01": ("lpv11 (LLM extract)", "#74c476"),
    "lpv11_fastpipe_v2_10k_mix_cultural_clp_lambda0p01": ("fastpipe_v2", "#006d2c"),
    "resiliparse_10k_mix_cultural_clp_lambda0p01": ("Resiliparse", "#9467bd"),
    "nemotron_full_10k_mix_cultural_clp_lambda0p01": ("Nemotron-CC", "#ff7f0e"),
}

SIZE_LABEL = {512: "157M (d512)", 1536: "998M (d1536)"}
# Plain-English panel titles for external documents (grants, slides) — no internal codes.
SIZE_LABEL_CLEAN = {512: "Cultural Knowledge (157M models)", 1536: "Cultural Knowledge (1B models)"}
# Canonical parameter counts for the fixed-model widths; tokens = budget / (6 * params).
SIZE_PARAMS = {512: 157e6, 1024: 447e6, 1536: 998e6, 2432: 2.9e9, 3584: 8.1e9}


def _tokens_label(x: float, _pos=None) -> str:
    if x >= 1e9:
        return f"{x / 1e9:g}B"
    return f"{x / 1e6:g}M"


_RUN_RE = re.compile(r"^curation-(?P<arm>.+)-expFM_natural-(?P<budget>[\d.e+]+)-d(?P<dim>\d+)-L\d+-B\d+$")


METRICS = {
    "clp": ("choice_logprob,none", "Cultural suite mean choice_logprob\n(higher is better)"),
    "acc": ("acc,none", "Cultural suite mean accuracy"),
}


@dataclass(frozen=True)
class Record:
    arm: str
    budget: float
    hidden_dim: int
    value: float


def load_records(result_key: str) -> list[Record]:
    fs = fsspec.filesystem("gcs")
    records = []
    for bucket in RESULTS_BUCKETS:
        try:
            paths = [p for p in fs.find(f"{bucket}/{RESULTS_SUBPATH}", maxdepth=2) if p.endswith("/results.json")]
        except FileNotFoundError:
            continue
        for path in paths:
            m = _RUN_RE.match(path.rsplit("/", 2)[-2])
            if not m or m.group("arm") not in ARM_DISPLAY:
                continue
            doc = json.load(fs.open(path))
            vals = [
                task[result_key]
                for task in (doc.get("results") or {}).values()
                if isinstance(task, dict) and result_key in task
            ]
            if not vals:
                continue
            records.append(Record(m.group("arm"), float(m.group("budget")), int(m.group("dim")), sum(vals) / len(vals)))
    return records


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metric", default="clp", choices=sorted(METRICS))
    parser.add_argument("--dims", type=int, nargs="+", default=None, help="Widths to plot (default: all present).")
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="External-document styling: plain-English titles, large legend, no suptitle jargon.",
    )
    parser.add_argument("--legend-size", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    result_key, ylabel = METRICS[args.metric]
    records = load_records(result_key)
    if not records:
        raise SystemExit("No cultural-target eval results found.")
    logger.info("loaded %d records", len(records))

    plt.rcParams.update(PAPER_RCPARAMS)
    dims = sorted({r.hidden_dim for r in records})
    if args.dims:
        dims = [d for d in dims if d in set(args.dims)]
        records = [r for r in records if r.hidden_dim in set(dims)]
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2 * len(dims), 7.0), sharey=True, squeeze=False)
    for ax, dim in zip(axes[0], dims):
        for arm, (label, color) in ARM_DISPLAY.items():
            pts = sorted(
                (r.budget / (6.0 * SIZE_PARAMS[dim]), r.value) for r in records if r.arm == arm and r.hidden_dim == dim
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=color, label=label)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(_tokens_label))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Training tokens")
        labels_map = SIZE_LABEL_CLEAN if args.presentation else SIZE_LABEL
        ax.set_title(labels_map.get(dim, f"d{dim}"))
        ax.grid(True, alpha=0.3)
    if args.presentation:
        pres_ylabel = {"clp": "Mean choice log-probability\n(higher is better)", "acc": "Mean accuracy"}[args.metric]
        axes[0][0].set_ylabel(pres_ylabel)
        for ax, dim in zip(axes[0], dims):
            ax.set_title(SIZE_LABEL_CLEAN.get(dim, f"d{dim}"))
            ax.legend(loc="lower right", fontsize=args.legend_size or 18, frameon=True)
    else:
        axes[0][0].set_ylabel(ylabel)
        axes[0][0].legend(loc="lower right", fontsize=args.legend_size or 9)
        fig.suptitle("Cultural-target mixes — cultural suite (19 tasks), 5-shot", y=1.02)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(f"d{d}" for d in dims) if args.dims else "by_size"
    if args.presentation:
        suffix += "_pres"
    stem = args.output_dir / f"cultural_target_{args.metric}_{suffix}"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    logger.info("wrote %s.{png,pdf}", stem)


if __name__ == "__main__":
    main()
