# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Domain-target sweep v2: MMLU category clp vs tokens, one panel per target category.

Reads `metadata/mmlu_target_models/5shot` (all 57 subjects per run) and scores each
MMLU-target arm on ITS OWN target category (mean choice_logprob over that category's
subjects). The fwedu natural baseline is scored on every category. Same visual
conventions as `plot_domain_v2_cultural`: corpus = color, k=5 dotted vs k=20 solid,
157M at reduced alpha vs 1B full.

    uv run python -m experiments.scaling_law_sweeps.plot_domain_v2_mmlu
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

from experiments.scaling_law_sweeps.olmo_bpb.mmlu_olmes import MMLU_CATEGORIES
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
SUBPATH = "metadata/mmlu_target_models/5shot"

CORPUS_DISPLAY = {
    "dclm_10k": ("DCLM-baseline", "#1f77b4"),
    "lpv11_10k": ("lpv11 (LLM extract)", "#74c476"),
    "lpv11_fastpipe_v2_10k": ("fastpipe_v2", "#006d2c"),
    "resiliparse_10k": ("Resiliparse", "#9467bd"),
    "nemotron_full_10k": ("Nemotron-CC", "#ff7f0e"),
    "fineweb_cc_10k": ("FineWeb-CC", "#e377c2"),
    "fineweb_edu_10k": ("FineWeb-Edu (natural)", "#17becf"),
}
SIZE_PARAMS = {512: 157e6, 1536: 998e6}
TARGETS = ("mmlu_stem", "mmlu_humanities", "mmlu_social_sciences")

_V2_RE = re.compile(
    r"^curation-(?P<corpus>.+?)_mix_(?P<target>mmlu_\w+?)_clp_(?P<gen>r100k\d+)_lambda0p05"
    r"-expFM_natural-(?P<budget>[\d.e+]+)-d(?P<dim>\d+)-L\d+-B\d+$"
)
_NAT_RE = re.compile(r"^curation-fineweb_edu_10k-expFM_natural-(?P<budget>[\d.e+]+)-d(?P<dim>\d+)-L\d+-B\d+$")


@dataclass(frozen=True)
class Record:
    corpus: str
    gen: str  # "k5" | "k20" | "natural"
    target: str  # which arm objective this run trained for ("all" for natural)
    budget: float
    hidden_dim: int
    clp_by_cat: tuple[tuple[str, float], ...]


def _category_means(doc: dict) -> dict[str, float]:
    res = doc.get("results") or {}
    out = {}
    for cat, subjects in MMLU_CATEGORIES.items():
        vals = [
            res[s]["choice_logprob,none"]
            for s in subjects
            if isinstance(res.get(s), dict) and "choice_logprob,none" in res[s]
        ]
        if vals:
            out[cat] = sum(vals) / len(vals)
    return out


# Raw results.json files embed per-sample logs (~228 MB each); NEVER read them from a
# laptop. `consolidate_model_evals` rolls them up on-cluster to this small CSV.
ROLLUP = "marin-us-central1/metadata/mmlu_target_models/rollup.csv"


def load_records(refresh: bool = True) -> list[Record]:
    import csv

    fs = fsspec.filesystem("gcs", use_listings_cache=False)
    by_run: dict[str, dict[str, float]] = {}
    with fs.open(ROLLUP, "r") as fh:
        for row in csv.DictReader(fh):
            # rollup task names are "mmlu_sl_verb_<subject>_5shot"; categories use bare subjects.
            task = row["task"]
            if not task.startswith("mmlu_sl_verb_") or task == "mmlu_sl_verb_5shot":
                continue
            subject = task.removeprefix("mmlu_sl_verb_").removesuffix("_5shot")
            by_run.setdefault(row["run"], {})[subject] = float(row["clp"])
    records = []
    for run, tasks in by_run.items():
        m = _V2_RE.match(run)
        if m:
            corpus, target, gen = m.group("corpus"), m.group("target"), m.group("gen").removeprefix("r100")
        else:
            m = _NAT_RE.match(run)
            if not m:
                continue
            corpus, target, gen = "fineweb_edu_10k", "all", "natural"
        cats = {}
        for cat, subjects in MMLU_CATEGORIES.items():
            vals = [tasks[s] for s in subjects if s in tasks]
            # NEVER average a partial category: a run missing any subject is dropped
            # from that category (see the 2026-09-15 partial-BPB incident).
            if len(vals) == len(subjects):
                cats[cat] = sum(vals) / len(vals)
            elif vals:
                logger.warning(
                    "partial MMLU eval for %s (%s: %d/%d) — category dropped", run, cat, len(vals), len(subjects)
                )
        if cats:
            records.append(
                Record(corpus, gen, target, float(m.group("budget")), int(m.group("dim")), tuple(sorted(cats.items())))
            )
    return records


def _tokens(r: Record) -> float:
    return r.budget / (6.0 * SIZE_PARAMS[r.hidden_dim])


def _tokens_label(x, _pos=None):
    return f"{x / 1e9:g}B" if x >= 1e9 else f"{x / 1e6:g}M"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cached", action="store_true", help="Reuse the last downloaded records.")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(refresh=not args.cached)
    logger.info("loaded %d MMLU-suite records", len(records))
    plt.rcParams.update(PAPER_RCPARAMS)

    for target in TARGETS:
        cat = target.removeprefix("mmlu_")
        dims = sorted({r.hidden_dim for r in records if r.target in (target, "all")})
        if not dims:
            continue
        fig, axes = plt.subplots(1, len(dims), figsize=(7.2 * len(dims), 7.0), sharey=True, squeeze=False)
        for ax, dim in zip(axes[0], dims):
            for corpus, (label, color) in CORPUS_DISPLAY.items():
                for gen, style in (("k20", "-"), ("k5", ":"), ("natural", "--")):
                    pts = sorted(
                        (_tokens(r), dict(r.clp_by_cat).get(cat))
                        for r in records
                        if r.corpus == corpus
                        and r.gen == gen
                        and r.hidden_dim == dim
                        and (r.target == target or r.target == "all")
                        and dict(r.clp_by_cat).get(cat) is not None
                    )
                    if not pts:
                        continue
                    xs, ys = zip(*pts)
                    suffix = "" if gen == "k20" else f" ({gen})"
                    ax.plot(xs, ys, style, marker="o", markersize=5, color=color, label=label + suffix)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(_tokens_label))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("Training tokens")
            ax.set_title("1B models" if dim == 1536 else "157M models")
            ax.grid(True, alpha=0.3)
        axes[0][0].set_ylabel(f"MMLU {cat.replace('_', ' ')} mean choice_logprob\n(higher is better)")
        axes[0][0].legend(fontsize=10, loc="lower right")
        fig.suptitle(f"{target}-target arms — scored on {cat.replace('_', ' ')}", y=1.02)
        fig.tight_layout()
        stem = args.output_dir / f"domain_v2_{target}"
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
        logger.info("wrote %s.{png,pdf}", stem)


if __name__ == "__main__":
    main()
