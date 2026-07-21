# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Uncheatable-eval val-loss (bpb) companion to ``plot_core_v2_random_ladder``.

Same crossover question, different metric. The WARC-scaling runs validate every
step against the FROZEN 2025-09-01→09-14 Uncheatable Eval snapshot (7 datasets),
logging ``eval/uncheatable_eval/{dataset}/{loss,bpb}`` and ``macro_{loss,bpb}``
to W&B (entity ``marin-community`` / project ``marin``). Unlike Core v2 (GCS
summaries) this metric lives ONLY in W&B run.summary.

Builds the same 3-row (N=100/300/500) × model-scale grid as the Core plotter,
HQ (green) vs DCLM (blue), once for macro bpb and once per disaggregated dataset.
bpb is bits-per-byte: LOWER is better (annotated on each figure).

The W&B pull (174 runs) is cached to JSON so re-plots are instant; pass
``--refresh`` to re-pull.

Usage::
    export WANDB_API_KEY=...        # see memory
    export SSL_CERT_FILE=$(.venv/bin/python -m certifi)
    .venv/bin/python -m experiments.scaling_law_sweeps.plot_uncheatable_random_ladder [--refresh]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin"
NAME_RE = re.compile(
    r"curation-(?P<method>high_quality|dclm)_random_(?P<n>\d+)"
    r"-expWARC_natural-(?P<budget>[0-9eE+.\-]+)-d(?P<dim>\d+)-L\d+-B\d+$"
)

NS = [100, 300, 500, 1000]
PARAMS = {256: "69M", 512: "157M", 768: "273M", 1024: "~500M", 1536: "998M"}
DATASETS = [
    "wikipedia_english",
    "bbc_news",
    "arxiv_physics",
    "arxiv_computer_science",
    "github_python",
    "github_cpp",
    "ao3_english",
]
MACRO_KEY = "eval/uncheatable_eval/macro_bpb"

HQ_COLOR = "#2ca02c"
DCLM_COLOR = "#1f77b4"

OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "uncheatable"
CACHE = OUT_DIR / "wandb_uncheatable_cache.json"


def _pull_wandb() -> list[dict]:
    """One row per matching run: method, n, budget, dim, and every uncheatable bpb key."""
    import wandb

    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"display_name": {"$regex": "curation-(high_quality|dclm)_random_(100|300|500|1000)-expWARC"}},
        per_page=500,
    )
    keys = [MACRO_KEY] + [f"eval/uncheatable_eval/{d}/bpb" for d in DATASETS]
    rows: list[dict] = []
    for r in runs:
        m = NAME_RE.match(r.name)
        if not m or r.state != "finished":
            continue
        metrics = {k: v for k in keys if isinstance((v := r.summary.get(k)), (int, float)) and not isinstance(v, bool)}
        if MACRO_KEY not in metrics:
            continue
        rows.append(
            {
                "method": m.group("method"),
                "n": int(m.group("n")),
                "budget": float(m.group("budget")),
                "dim": int(m.group("dim")),
                "metrics": metrics,
            }
        )
    logger.info("pulled %d finished runs with uncheatable bpb", len(rows))
    return rows


def _load(refresh: bool) -> list[dict]:
    if not refresh and CACHE.exists():
        rows = json.loads(CACHE.read_text())
        logger.info("loaded %d rows from cache %s", len(rows), CACHE)
        return rows
    rows = _pull_wandb()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(rows, indent=0))
    return rows


def _cells(rows: list[dict], method: str, n: int, dim: int, key: str) -> list[tuple[float, float]]:
    return sorted(
        (r["budget"], r["metrics"][key])
        for r in rows
        if r["method"] == method and r["n"] == n and r["dim"] == dim and key in r["metrics"]
    )


def _grid_figure(rows: list[dict], key: str, label: str, out: Path) -> None:
    """3 rows (N) × model-scale cols; HQ vs DCLM bpb. LOWER is better."""
    dims = sorted({r["dim"] for r in rows})
    allv = [r["metrics"][key] for r in rows if key in r["metrics"]]
    if not allv:
        logger.warning("no data for %s; skipping", key)
        return
    ylo, yhi = min(allv), max(allv)
    ypad = 0.06 * (yhi - ylo or 0.1)
    nrow, ncol = len(NS), len(dims)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False, sharey=True)
    for ri, n in enumerate(NS):
        for ci, dim in enumerate(dims):
            ax = axes[ri][ci]
            for method, color, mk in (("high_quality", HQ_COLOR, "o"), ("dclm", DCLM_COLOR, "s")):
                pts = _cells(rows, method, n, dim, key)
                if pts:
                    ax.plot([b for b, _ in pts], [v for _, v in pts], marker=mk, ms=5, lw=2, color=color)
            ax.set_xscale("log")
            ax.set_ylim(ylo - ypad, yhi + ypad)
            ax.grid(True, which="major", alpha=0.2)
            if ri == 0:
                ax.set_title(f"d{dim} ({PARAMS.get(dim, '?')})", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"N={n} WARCs\n{label}", fontsize=10)
            if ri == nrow - 1:
                ax.set_xlabel("FLOPs", fontsize=9)
    handles = [
        Line2D([0], [0], color=HQ_COLOR, lw=2.5, marker="o", label="HQ (high_quality)"),
        Line2D([0], [0], color=DCLM_COLOR, lw=2.5, marker="s", label="DCLM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"DCLM vs HQ — {label} — grid of model scale × WARC count (N=100/300/500/1000 random)\n"
        "Uncheatable Eval (frozen 2025-09-01→09-14) · LOWER is better",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("wrote %s", out)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-pull from W&B instead of using the JSON cache")
    args = ap.parse_args()
    rows = _load(args.refresh)
    for n in NS:
        hq = sum(1 for r in rows if r["method"] == "high_quality" and r["n"] == n)
        dc = sum(1 for r in rows if r["method"] == "dclm" and r["n"] == n)
        logger.info("N=%d: HQ %d, DCLM %d", n, hq, dc)
    _grid_figure(rows, MACRO_KEY, "macro bpb", OUT_DIR / "uncheatable_random_ladder_macro.png")
    for d in DATASETS:
        _grid_figure(rows, f"eval/uncheatable_eval/{d}/bpb", f"{d} bpb", OUT_DIR / f"uncheatable_random_ladder_{d}.png")


if __name__ == "__main__":
    main()
