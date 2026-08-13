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
                [
                    "gcloud",
                    "storage",
                    "ls",
                    f"gs://marin-{r}/{CORE_SUB}/curation-{method}-expWARC_natural-*_summary.json",
                ]
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
        data[n] = {m: _collect(f"{m}_random_{n}") for m in METHOD_STYLE}
        logger.info("N=%d: %s", n, {m: len(c) for m, c in data[n].items() if c})
    return data


def _dims(data: dict[int, dict]) -> list[int]:
    dims: set[int] = set()
    for n in NS:
        for cells in data[n].values():
            dims |= {d for _, d in cells}
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
            hq, dc = data[n]["high_quality"], data[n]["dclm"]
            pts = sorted((b, hq[(b, dim)] - dc[(b, dim)]) for (b, d) in set(hq) & set(dc) if d == dim)
            if pts:
                ax.plot(
                    [b for b, _ in pts], [g for _, g in pts], marker="o", ms=6, lw=2, color=N_COLOR[n], label=f"N={n}"
                )
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
    fig.suptitle(
        "HQ − DCLM Core v2 gap across the N=100→300→500 random ladder\n(does HQ's edge shrink/flip as WARCs grow?)",
        fontsize=13,
        y=1.0,
    )
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
            for src, ls, mk in (("high_quality", "solid", "o"), ("dclm", (0, (4, 3)), "s")):
                pts = sorted((b, v) for (b, d), v in data[n][src].items() if d == dim)
                if pts:
                    ax.plot(
                        [b for b, _ in pts],
                        [v for _, v in pts],
                        marker=mk,
                        ms=4,
                        lw=1.8,
                        color=N_COLOR[n],
                        linestyle=ls,
                        alpha=0.9,
                    )
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
    handles += [
        Line2D([0], [0], color="#444", lw=1.8, marker="o", label="HQ"),
        Line2D([0], [0], color="#444", lw=1.8, ls=(0, (4, 3)), marker="s", label="DCLM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Core v2 vs FLOPs — HQ (solid) vs DCLM (dashed) across N=100/300/500", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("wrote %s", out)
    plt.close(fig)


# LOCKED color convention (2026-07-22): dclm=blue, high_quality=green, nemotron=orange,
# resiliparse=purple, llm_pipeline_v1=red. fastpipe bands = brown ramp (one group).
METHOD_STYLE: dict[str, tuple[str, str, str]] = {
    "llm_pipeline_v1": ("#d62728", "o", "llm_pipeline_v1"),
    "llm_pipeline_v1_1": ("#8c564b", "X", "llm_pipeline_v1_1"),
    "llm_simple_v1": ("#17becf", "s", "llm_simple_v1"),
    "dclm": ("#1f77b4", "^", "dclm"),
    "high_quality": ("#2ca02c", "s", "high_quality"),
    "high_quality_v2": ("#e377c2", "D", "high_quality_v2"),
    "nemotron_full": ("#ff7f0e", "D", "nemotron"),
    "resiliparse": ("#9467bd", "v", "resiliparse"),
    "fastpipe_v3_100": ("#5c3d2e", "P", "fastpipe 100%"),
    "fastpipe_v3_80": ("#7d5540", "P", "fastpipe 80%"),
    "fastpipe_v3_60": ("#a06e52", "P", "fastpipe 60%"),
    "fastpipe_v3_40": ("#c08a6a", "P", "fastpipe 40%"),
    "fastpipe_v3_20": ("#dbb28f", "P", "fastpipe 20%"),
}


# Numeric param counts per hidden dim (for token->FLOPs epoch conversion: C ≈ 6·N·D).
_PARAMS_NUM = {256: 69e6, 512: 157e6, 768: 273e6, 1024: 500e6, 1536: 998e6, 2432: 2.9e9, 3328: 6e9, 3584: 8.1e9}


def _epoch_flops(method: str, n: int, dim: int) -> tuple[float | None, float | None]:
    """(1-epoch, 2-epoch) training FLOPs for a method's data at a given model size."""
    from experiments.scaling_law_sweeps.curation_plan import METHODS

    m = METHODS.get(f"{method}_random_{n}")
    p = _PARAMS_NUM.get(dim)
    if m is None or p is None:
        return None, None
    x1 = 6.0 * p * m.d_obs_tokens
    return x1, 2.0 * x1


def _draw_epochs(ax, data_n: dict, dim: int, n: int) -> None:
    """Vertical dashed (1 epoch) / dotted (2 epoch) marks per present method, without rescaling x."""
    xl = ax.get_xlim()
    for method, (color, _mk, _lab) in METHOD_STYLE.items():
        if not any(d == dim for (_b, d) in data_n.get(method, {})):
            continue
        x1, x2 = _epoch_flops(method, n, dim)
        if x1:
            ax.axvline(x1, color=color, ls=(0, (4, 2)), lw=0.9, alpha=0.45, zorder=0)
        if x2:
            ax.axvline(x2, color=color, ls=(0, (1, 2)), lw=1.0, alpha=0.45, zorder=0)
    ax.set_xlim(xl)


def _grid_figure(data: dict[int, dict], out: Path) -> None:
    """Big grid: rows = N, cols = model size; Core v2 per curation method per panel."""
    dims = _dims(data)
    allv = [v for n in NS for cells in data[n].values() for v in cells.values()]
    ylo, yhi = (min(allv), max(allv)) if allv else (0, 0.1)
    ypad = 0.06 * (yhi - ylo or 0.1)
    nrow, ncol = len(NS), len(dims)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False, sharey=True)
    for ri, n in enumerate(NS):
        for ci, dim in enumerate(dims):
            ax = axes[ri][ci]
            for method, (color, mk, _lab) in METHOD_STYLE.items():
                pts = sorted((b, v) for (b, d), v in data[n].get(method, {}).items() if d == dim)
                if pts:
                    ax.plot([b for b, _ in pts], [v for _, v in pts], marker=mk, ms=5, lw=2, color=color)
            ax.axhline(0, color="#bbb", lw=0.7, ls=":")
            ax.set_xscale("log")
            _draw_epochs(ax, data[n], dim, n)
            ax.set_ylim(ylo - ypad, yhi + ypad)
            ax.grid(True, which="major", alpha=0.2)
            if ri == 0:
                ax.set_title(f"d{dim} ({PARAMS.get(dim, '?')})", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"N={n} WARCs\nCore v2", fontsize=10)
            if ri == nrow - 1:
                ax.set_xlabel("FLOPs", fontsize=9)
    present = {m for n in NS for m, cells in data[n].items() if cells}
    handles = [
        Line2D([0], [0], color=c, lw=2.5, marker=mk, label=lab)
        for m, (c, mk, lab) in METHOD_STYLE.items()
        if m in present
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False, bbox_to_anchor=(0.5, -0.01)
    )
    fig.suptitle(
        "Core v2 by curation method — model scale × WARC count (N=100/300/500/1000/2000 random)\nHIGHER is better",
        fontsize=14,
        y=1.0,
    )
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
