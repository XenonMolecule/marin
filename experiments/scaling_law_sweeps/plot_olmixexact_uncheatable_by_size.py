# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Uncheatable-eval companion to ``plot_olmixexact_bpb_by_size``.

Same head-to-head layout (one panel per model size, tokens-trained on x), but the y-axis
is the FROZEN 2025-09-01→09-14 Uncheatable Eval macro bpb, which every sweep cell logs to
W&B during training (``eval/uncheatable_eval/macro_bpb``; see
``plot_uncheatable_random_ladder`` for the metric's provenance). Contamination-proof
counterpart to the 51-task suite: the eval text postdates every corpus.

Tokens per run join from ``data_curation_10k_natural_results/<stem>.json`` exactly as the
bpb plotter does. The W&B pull is cached; pass ``--refresh`` to re-pull.

Usage::
    uv run python -m experiments.scaling_law_sweeps.plot_olmixexact_uncheatable_by_size [--refresh]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import PAPER_RCPARAMS
from experiments.scaling_law_sweeps.plot_olmixexact_bpb_by_size import (
    METHOD_DISPLAY,
    METHOD_LINEWIDTH,
    _legend_label,
    _tokens_label,
)
from experiments.scaling_law_sweeps.plot_olmo_bpb_by_size_10k import EXCLUDED_RUN_STEMS

logger = logging.getLogger(__name__)

WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin"
MACRO_KEY = "eval/uncheatable_eval/macro_bpb"
NAME_RE = re.compile(r"curation-(?P<method>.+)-expFM_natural-(?P<budget>[0-9eE+.\-]+)-d(?P<dim>\d+)-L\d+-B\d+$")
PARAM_LABEL = {512: "157M params", 1024: "447M params", 1536: "998M params", 2432: "2.9B params", 3584: "8.1B params"}

TOKENS_PREFIX = "marin-us-central1/metadata/data_curation_10k_natural_results"
OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "olmo_bpb_10k" / "paper"
CACHE = OUT_DIR / "wandb_uncheatable_olmixexact_cache.json"


def _pull_wandb() -> list[dict]:
    import wandb

    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"display_name": {"$regex": "curation-.*olmixexact.*-expFM_natural-"}},
        per_page=500,
    )
    rows: list[dict] = []
    for r in runs:
        m = NAME_RE.match(r.name)
        if not m or m.group("method") not in METHOD_DISPLAY:
            continue
        v = r.summary.get(MACRO_KEY)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        rows.append({"stem": r.name, "method": m.group("method"), "dim": int(m.group("dim")), "macro_bpb": float(v)})
    logger.info("pulled %d runs with uncheatable macro bpb", len(rows))
    return rows


def _join_tokens(rows: list[dict]) -> list[dict]:
    fs = fsspec.filesystem("gcs")
    kept = []
    for row in rows:
        try:
            with fs.open(f"{TOKENS_PREFIX}/{row['stem']}.json") as fh:
                doc = json.load(fh)
        except FileNotFoundError:
            continue
        tokens = (doc.get("tokens") or {}).get("tokens_trained")
        if tokens:
            kept.append({**row, "tokens": float(tokens)})
    logger.info("joined tokens for %d/%d runs", len(kept), len(rows))
    return kept


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--methods", nargs="+", default=None, help="Subset of METHOD_DISPLAY keys.")
    ap.add_argument("--dims", type=int, nargs="+", default=None)
    ap.add_argument("--legend-size", type=float, default=None)
    ap.add_argument("--relabel", nargs="+", default=[], metavar="METHOD=LABEL")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    if not args.refresh and CACHE.exists():
        records = json.loads(CACHE.read_text())
    else:
        records = _join_tokens(_pull_wandb())
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(records, indent=0))
    records = [r for r in records if r["stem"] not in EXCLUDED_RUN_STEMS]

    plt.rcParams.update(PAPER_RCPARAMS)
    display = dict(METHOD_DISPLAY)
    overrides = dict(kv.split("=", 1) for kv in args.relabel)
    display = {m: (overrides.get(m, lbl), c) for m, (lbl, c) in display.items()}
    if args.methods:
        display = {m: display[m] for m in args.methods}
        records = [r for r in records if r["method"] in display]
    if args.dims:
        records = [r for r in records if r["dim"] in set(args.dims)]
    dims = sorted({r["dim"] for r in records})
    fig, axes = plt.subplots(1, len(dims), figsize=(7.2 * len(dims), 7.0), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, dim in zip(axes, dims, strict=True):
        for method, (label, color) in display.items():
            pts = sorted((r["tokens"], r["macro_bpb"]) for r in records if r["method"] == method and r["dim"] == dim)
            if not pts:
                continue
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(
                x,
                y,
                "-",
                color=color,
                linewidth=METHOD_LINEWIDTH.get(method, 3.0),
                alpha=0.95,
                label=_legend_label(method, label),
                zorder=3,
            )
            ax.scatter(x, y, color=color, s=85, edgecolors="white", linewidths=0.8, zorder=4)
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        subs = (1.0,) if np.log10(hi / lo) > 1.6 else (1.0, 2.0, 5.0)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=subs))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tokens_label))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(PARAM_LABEL.get(dim, f"d{dim}"))
        ax.set_xlabel("Tokens trained")
        ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
    axes[0].set_ylabel("Uncheatable Eval macro bpb (frozen 2025-09 snapshot)")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, ll in zip(*ax.get_legend_handles_labels(), strict=True):
            if ll not in labels:
                handles.append(h)
                labels.append(ll)
    ncol = len(labels) if args.legend_size is None else max(1, min(len(labels), 3))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.12 if args.legend_size is None else -0.2),
        fontsize=args.legend_size,
    )
    stem = OUT_DIR / f"olmixexact_uncheatable_macro_bpb_by_size{args.suffix}"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    logger.info("Wrote %s.{png,pdf}", stem)


if __name__ == "__main__":
    main()
