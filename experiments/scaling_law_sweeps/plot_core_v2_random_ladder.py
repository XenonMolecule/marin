# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the HQ - DCLM Core_v2 gap across the N=100 -> 300 -> 500 random-sample ladder.

The crossover question: at 10k WARCs HQ is WORSE than DCLM on Core; at N=100
(both biased and random) HQ is BETTER. Does the HQ-DCLM gap SHRINK / flip as we
grow the (nested, seed-0) random sample 100 -> 300 -> 500? If the gap-vs-FLOPs
line drops toward/below zero as N grows (especially at high budget / large model),
that's the crossover emerging at a small, cheap scale.

Two figures, gridded by model scale (hidden dim):
  * ``..._gap.png``      : y = HQ_Core - DCLM_Core, one line per N. Zero-line = crossover.
  * ``..._absolute.png`` : y = Core_v2, HQ (solid) vs DCLM (dashed), colored by N.

Core_v2 from ``<region>/metadata/data_curation_10k_core_results/<stem>_summary.json``
-> ``dclm.Core_v2``; scattered across regions (methods pinned/floated/re-homed), so
ALL regions are scanned. x = training FLOPs and model dim are parsed from the run
stem, so NO token/param join is needed (works even mid-fill).

Usage::
    export SSL_CERT_FILE=$(.venv/bin/python -m certifi)
    .venv/bin/python -m experiments.scaling_law_sweeps.plot_core_v2_random_ladder
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

REGIONS = ["us-east5", "us-central1", "us-east1", "eu-west4", "us-central2", "us-west4"]
CORE_SUB = "metadata/data_curation_10k_core_results"
OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "core_v2"
NS = [100, 300, 500, 1000, 2000]
N_COLOR = {100: "#1f77b4", 300: "#ff7f0e", 500: "#2ca02c", 1000: "#9467bd", 2000: "#8c564b"}
PARAMS = {256: "69M", 512: "157M", 768: "273M", 1024: "~500M", 1536: "998M", 2432: "2.9B", 3328: "~6B", 3584: "8.1B"}
_STEM_RE = re.compile(r"-expWARC_natural-(?P<budget>[0-9eE+.\-]+)-d(?P<hidden>\d+)-L\d+-B\d+$")


def _sh(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def _core_of(path: str) -> float | None:
    try:
        return json.loads(_sh(["gcloud", "storage", "cat", path]))["dclm"]["Core_v2"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _collect(method: str) -> dict[tuple[float, int], float]:
    """cell (budget, hidden) -> Core_v2, scanned across all regions."""
    files: list[str] = []
    for r in REGIONS:
        files += [
            ln
            for ln in _sh(
                ["gcloud", "storage", "ls", f"gs://marin-{r}/{CORE_SUB}/curation-{method}-expWARC_natural-*_summary.json"]
            ).splitlines()
            if ln.strip()
        ]
    out: dict[tuple[float, int], float] = {}
    with ThreadPoolExecutor(max_workers=24) as ex:
        vals = list(ex.map(_core_of, files))
    for f, v in zip(files, vals):
        if v is None:
            continue
        m = _STEM_RE.search(f.rsplit("/", 1)[-1].replace("_summary.json", ""))
        if m:
            out[(float(m.group("budget")), int(m.group("hidden")))] = v
    return out


def _load() -> dict[int, dict]:
    data: dict[int, dict] = {}
    for n in NS:
        hq = _collect(f"high_quality_random_{n}")
        dc = _collect(f"dclm_random_{n}")
        data[n] = {"hq": hq, "dclm": dc}
        logger.info("N=%d: HQ %d, DCLM %d, matched %d", n, len(hq), len(dc), len(set(hq) & set(dc)))
    return data


def _dims(data: dict[int, dict]) -> list[int]:
    dims: set[int] = set()
    for n in NS:
        for src in ("hq", "dclm"):
            dims |= {d for _, d in data[n][src]}
    return sorted(dims)


def _panels(dims: list[int]):
    ncol = min(len(dims), 2)
    nrow = math.ceil(len(dims) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow), squeeze=False)
    return fig, axes, ncol, nrow


def _gap_figure(data: dict[int, dict], out: Path) -> None:
    dims = _dims(data)
    fig, axes, ncol, nrow = _panels(dims)
    for idx, dim in enumerate(dims):
        ax = axes[idx // ncol][idx % ncol]
        for n in NS:
            hq, dc = data[n]["hq"], data[n]["dclm"]
            pts = sorted((b, hq[(b, dim)] - dc[(b, dim)]) for (b, d) in set(hq) & set(dc) if d == dim)
            if pts:
                ax.plot([b for b, _ in pts], [g for _, g in pts], marker="o", ms=6, lw=2, color=N_COLOR[n], label=f"N={n}")
        ax.axhline(0, color="#d62728", lw=1.2, ls="--", alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(f"d{dim} ({PARAMS.get(dim, '?')})", fontsize=11)
        ax.grid(True, which="major", alpha=0.25)
        if idx % ncol == 0:
            ax.set_ylabel("HQ − DCLM  (Core v2)")
        if idx // ncol == nrow - 1:
            ax.set_xlabel("training FLOPs")
    for j in range(len(dims), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [Line2D([0], [0], color=N_COLOR[n], lw=2.5, marker="o", label=f"N={n} WARCs") for n in NS]
    handles.append(Line2D([0], [0], color="#d62728", lw=1.2, ls="--", label="crossover (HQ=DCLM)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("HQ − DCLM Core v2 gap across the N=100→300→500 random ladder\n(does HQ's edge shrink/flip as WARCs grow?)", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("wrote %s", out)
    plt.close(fig)


def _absolute_figure(data: dict[int, dict], out: Path) -> None:
    dims = _dims(data)
    fig, axes, ncol, nrow = _panels(dims)
    for idx, dim in enumerate(dims):
        ax = axes[idx // ncol][idx % ncol]
        for n in NS:
            for src, ls, mk in (("hq", "solid", "o"), ("dclm", (0, (4, 3)), "s")):
                pts = sorted((b, v) for (b, d), v in data[n][src].items() if d == dim)
                if pts:
                    ax.plot([b for b, _ in pts], [v for _, v in pts], marker=mk, ms=4, lw=1.8, color=N_COLOR[n], linestyle=ls, alpha=0.9)
        ax.axhline(0, color="#bbb", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"d{dim} ({PARAMS.get(dim, '?')})", fontsize=11)
        ax.grid(True, which="major", alpha=0.25)
        if idx % ncol == 0:
            ax.set_ylabel("DCLM Core v2")
        if idx // ncol == nrow - 1:
            ax.set_xlabel("training FLOPs")
    for j in range(len(dims), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [Line2D([0], [0], color=N_COLOR[n], lw=2.5, label=f"N={n}") for n in NS]
    handles += [Line2D([0], [0], color="#444", lw=1.8, marker="o", label="HQ"), Line2D([0], [0], color="#444", lw=1.8, ls=(0, (4, 3)), marker="s", label="DCLM")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Core v2 vs FLOPs — HQ (solid) vs DCLM (dashed) across N=100/300/500", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("wrote %s", out)
    plt.close(fig)


HQ_COLOR = "#2ca02c"
DCLM_COLOR = "#1f77b4"


def _grid_figure(data: dict[int, dict], out: Path) -> None:
    """Big grid: rows = N (100/300/500), cols = model size; HQ vs DCLM Core per panel."""
    dims = _dims(data)
    allv = [v for n in NS for src in ("hq", "dclm") for v in data[n][src].values()]
    ylo, yhi = (min(allv), max(allv)) if allv else (0, 0.1)
    ypad = 0.06 * (yhi - ylo or 0.1)
    nrow, ncol = len(NS), len(dims)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False, sharey=True)
    for ri, n in enumerate(NS):
        for ci, dim in enumerate(dims):
            ax = axes[ri][ci]
            for src, color, mk in (("hq", HQ_COLOR, "o"), ("dclm", DCLM_COLOR, "s")):
                pts = sorted((b, v) for (b, d), v in data[n][src].items() if d == dim)
                if pts:
                    ax.plot([b for b, _ in pts], [v for _, v in pts], marker=mk, ms=5, lw=2, color=color)
            ax.axhline(0, color="#bbb", lw=0.7, ls=":")
            ax.set_xscale("log")
            ax.set_ylim(ylo - ypad, yhi + ypad)
            ax.grid(True, which="major", alpha=0.2)
            if ri == 0:
                ax.set_title(f"d{dim} ({PARAMS.get(dim, '?')})", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"N={n} WARCs\nCore v2", fontsize=10)
            if ri == nrow - 1:
                ax.set_xlabel("FLOPs", fontsize=9)
    handles = [
        Line2D([0], [0], color=HQ_COLOR, lw=2.5, marker="o", label="HQ (high_quality)"),
        Line2D([0], [0], color=DCLM_COLOR, lw=2.5, marker="s", label="DCLM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("DCLM vs HQ Core v2 — grid of model scale × WARC count (N=100/300/500/1000/2000 random)", fontsize=14, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("wrote %s", out)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = _load()
    _gap_figure(data, OUT_DIR / "core_v2_random_ladder_gap.png")
    _absolute_figure(data, OUT_DIR / "core_v2_random_ladder_absolute.png")
    _grid_figure(data, OUT_DIR / "core_v2_random_ladder_grid.png")


if __name__ == "__main__":
    main()
