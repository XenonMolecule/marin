# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Domain-target sweep v2: cultural clp vs tokens, k=5 vs k=20 head-to-head.

Two figures over `metadata/cultural_target_models/5shot`:
  * `domain_v2_k5_vs_k20`: one panel per big-pool corpus (lpv11, fastpipe_v2, resiliparse),
    k=5 solid vs k=20 dashed — the repetition-cap comparison the sweeps were split for.
  * `domain_v2_all`: every v2 cultural arm plus the fwedu natural baseline, one panel per size.

    uv run python -m experiments.scaling_law_sweeps.plot_domain_v2_cultural
"""

from __future__ import annotations

import argparse
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
BUCKETS = (
    "marin-us-east5",
    "marin-us-east1",
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-west4",
    "marin-eu-west4",
)
SUBPATH = "metadata/cultural_target_models/5shot"

CORPUS_DISPLAY = {
    "dclm_10k": ("DCLM-baseline", "#1f77b4"),
    "lpv11_10k": ("lpv11 (LLM extract)", "#74c476"),
    "lpv11_fastpipe_v2_10k": ("fastpipe_v2", "#006d2c"),
    "resiliparse_10k": ("Resiliparse", "#9467bd"),
    "nemotron_full_10k": ("Nemotron-CC", "#ff7f0e"),
    "fineweb_cc_10k": ("FineWeb", "#e377c2"),
    "fineweb_edu_10k": ("FineWeb-Edu (natural)", "#17becf"),
}
SIZE_PARAMS = {512: 157e6, 1536: 998e6}

_V2_RE = re.compile(
    r"^curation-(?P<corpus>.+?)_mix_cultural_clp_(?P<gen>r100k\d+)_lambda0p05"
    r"-expFM_natural-(?P<budget>[\d.e+]+)-d(?P<dim>\d+)-L\d+-B\d+$"
)
_NAT_RE = re.compile(
    r"^curation-(?P<corpus>fineweb_edu)_10k-expFM_natural-(?P<budget>[\d.e+]+)-d(?P<dim>\d+)-L\d+-B\d+$"
)


@dataclass(frozen=True)
class Record:
    corpus: str
    gen: str  # "k5" | "k20" | "natural"
    budget: float
    hidden_dim: int
    clp: float


# Reads the on-cluster rollup (see consolidate_model_evals) — never the raw results.json.
ROLLUP = "marin-us-central1/metadata/cultural_target_models/rollup.csv"


def load_records() -> list[Record]:
    import csv

    fs = fsspec.filesystem("gcs", use_listings_cache=False)
    by_run: dict[str, list[float]] = {}
    with fs.open(ROLLUP, "r") as fh:
        for row in csv.DictReader(fh):
            by_run.setdefault(row["run"], []).append(float(row["clp"]))
    records = []
    EXPECTED_ROWS = 20  # 19 cultural tasks + suite aggregate; NEVER average a partial eval
    for run, vals in by_run.items():
        if len(vals) != EXPECTED_ROWS:
            logger.warning("partial cultural eval for %s (%d/%d rows) — dropped", run, len(vals), EXPECTED_ROWS)
            continue
        m = _V2_RE.match(run)
        if m:
            corpus, gen = m.group("corpus"), m.group("gen").removeprefix("r100")
        else:
            m = _NAT_RE.match(run)
            if not m:
                continue
            corpus, gen = "fineweb_edu_10k", "natural"
        records.append(Record(corpus, gen, float(m.group("budget")), int(m.group("dim")), sum(vals) / len(vals)))
    return records


def _tokens_label(x, _pos=None):
    return f"{x / 1e9:g}B" if x >= 1e9 else f"{x / 1e6:g}M"


def _setup_ax(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_tokens_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training tokens")
    ax.grid(True, alpha=0.3)


def _tokens(r: Record) -> float:
    return r.budget / (6.0 * SIZE_PARAMS[r.hidden_dim])


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records()
    logger.info("loaded %d v2 records", len(records))
    plt.rcParams.update(PAPER_RCPARAMS)

    # Figure 1: k5 vs k20 per big-pool corpus (both sizes pooled per panel, split by style).
    big = [c for c in ("lpv11_10k", "lpv11_fastpipe_v2_10k", "resiliparse_10k") if any(r.corpus == c for r in records)]
    if big:
        fig, axes = plt.subplots(1, len(big), figsize=(6.4 * len(big), 6.4), sharey=True, squeeze=False)
        for ax, corpus in zip(axes[0], big):
            label, color = CORPUS_DISPLAY[corpus]
            for gen, style in (("k5", "-"), ("k20", "--")):
                for dim, alpha in ((512, 0.55), (1536, 1.0)):
                    pts = sorted(
                        (_tokens(r), r.clp)
                        for r in records
                        if r.corpus == corpus and r.gen == gen and r.hidden_dim == dim
                    )
                    if not pts:
                        continue
                    xs, ys = zip(*pts)
                    ax.plot(
                        xs,
                        ys,
                        style,
                        marker="o",
                        color=color,
                        alpha=alpha,
                        label=f"{gen} · {'1B' if dim == 1536 else '157M'}",
                    )
            _setup_ax(ax)
            ax.set_title(label)
            ax.legend(fontsize=12)
        axes[0][0].set_ylabel("Cultural suite mean choice_logprob\n(higher is better)")
        fig.suptitle("Repetition cap: k=5 vs k=20 (R=100B, λ=0.05)", y=1.02)
        fig.tight_layout()
        stem = args.output_dir / "domain_v2_k5_vs_k20"
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
        logger.info("wrote %s.{png,pdf}", stem)

    # Figure 2: all v2 cultural arms + fwedu natural, one panel per size.
    dims = sorted({r.hidden_dim for r in records})
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2 * len(dims), 7.0), sharey=True, squeeze=False)
    for ax, dim in zip(axes[0], dims):
        for corpus, (label, color) in CORPUS_DISPLAY.items():
            for gen, style in (("k20", "-"), ("k5", ":"), ("natural", "--")):
                pts = sorted(
                    (_tokens(r), r.clp) for r in records if r.corpus == corpus and r.gen == gen and r.hidden_dim == dim
                )
                if not pts:
                    continue
                xs, ys = zip(*pts)
                suffix = "" if gen == "k20" else f" ({gen})"
                ax.plot(xs, ys, style, marker="o", color=color, label=label + suffix)
        _setup_ax(ax)
        ax.set_title("1B models" if dim == 1536 else "157M models")
    axes[0][0].set_ylabel("Cultural suite mean choice_logprob\n(higher is better)")
    axes[0][0].legend(fontsize=10, loc="lower right")
    fig.suptitle("Domain-target v2 — cultural suite (λ=0.05 solves)", y=1.02)
    fig.tight_layout()
    stem = args.output_dir / "domain_v2_all"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    logger.info("wrote %s.{png,pdf}", stem)


if __name__ == "__main__":
    main()
