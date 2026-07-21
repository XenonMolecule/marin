# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OLMo Base-Easy bpb crossover suite for the random-WARC ladder.

Companion to ``plot_core_v2_random_ladder`` / ``plot_uncheatable_random_ladder``: same
HQ-vs-DCLM, model-scale x WARC-count grid, but the metric is OLMo Base-Easy bits-per-byte
grouped into the categories the user cares about — **Code_bpb, Math_bpb, QA_bpb** (plus the
overall macro). LOWER bpb is better.

Reads per-checkpoint results from
``<region>/metadata/olmo_bpb_results/<run_name>/results.json`` (``tasks/<task>/<variant>/bpb``
+ ``averages/macro_bpb``), produced by ``run_olmo_bpb_eval``. Categories are the task groups
from ``olmo_bpb_tasks_set``. Results are scattered across regions, so ALL are scanned. Model
dim + budget are parsed from the run stem (no token join needed).

Usage::
    export SSL_CERT_FILE=$(.venv/bin/python -m certifi)
    .venv/bin/python -m experiments.scaling_law_sweeps.plot_olmo_bpb_random_ladder
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import (
    CODE_BPB,
    MATH_BPB,
    MT_MBPP_BPB,
    QA_LANG_BPB,
)

logger = logging.getLogger(__name__)

REGIONS = ["us-east5", "us-east1", "us-central1", "eu-west4"]
SUB = "metadata/olmo_bpb_results"
OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "olmo_bpb"
NS = [100, 300, 500, 1000, 2000]
PARAMS = {256: "69M", 512: "157M", 768: "273M", 1024: "~500M", 1536: "998M", 2432: "2.9B"}
NAME_RE = re.compile(
    r"curation-(?P<method>high_quality|dclm)_random_(?P<n>\d+)"
    r"-expWARC_natural-(?P<budget>[0-9eE+.\-]+)-d(?P<dim>\d+)-L\d+-B\d+$"
)
# metric -> the task keys ("<task>/<variant>") whose bpb we average
CATEGORIES = {
    "Code_bpb": list(CODE_BPB) + list(MT_MBPP_BPB),
    "Math_bpb": list(MATH_BPB),
    "QA_bpb": list(QA_LANG_BPB),
}
HQ_COLOR = "#2ca02c"
DCLM_COLOR = "#1f77b4"


def _sh(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def _metrics_of(path: str) -> dict | None:
    """Parse one results.json into {macro_bpb, Code_bpb, Math_bpb, QA_bpb}."""
    try:
        d = json.loads(_sh(["gcloud", "storage", "cat", path]))
    except (json.JSONDecodeError, TypeError):
        return None
    tasks = d.get("tasks", {})

    def bpb_of(task_key: str) -> float | None:
        # `tasks` is flat-keyed by the full "task/variant" string (not nested).
        b = tasks.get(task_key, {}).get("bpb")
        return b if isinstance(b, (int, float)) else None

    out: dict[str, float] = {}
    macro = d.get("averages", {}).get("macro_bpb")
    if isinstance(macro, (int, float)):
        out["macro_bpb"] = macro
    for cat, keys in CATEGORIES.items():
        vals = [b for k in keys if (b := bpb_of(k)) is not None]
        if vals:
            out[cat] = sum(vals) / len(vals)
    return out or None


def _load() -> list[dict]:
    files: list[str] = []
    for r in REGIONS:
        files += [
            ln
            for ln in _sh(["gcloud", "storage", "ls", f"gs://marin-{r}/{SUB}/curation-*_random_*/results.json"]).splitlines()
            if ln.strip().endswith("results.json")
        ]
    with ThreadPoolExecutor(max_workers=24) as ex:
        metrics = list(ex.map(_metrics_of, files))
    rows: list[dict] = []
    for f, m in zip(files, metrics):
        if m is None:
            continue
        stem = f.rsplit("/", 2)[-2]  # .../<run_name>/results.json
        mt = NAME_RE.match(stem)
        if not mt:
            continue
        rows.append(
            {"method": mt.group("method"), "n": int(mt.group("n")), "budget": float(mt.group("budget")), "dim": int(mt.group("dim")), **m}
        )
    logger.info("loaded %d HQ/DCLM olmo-bpb results", len(rows))
    return rows


def _cells(rows: list[dict], method: str, n: int, dim: int, key: str) -> list[tuple[float, float]]:
    return sorted((r["budget"], r[key]) for r in rows if r["method"] == method and r["n"] == n and r["dim"] == dim and key in r)


def _grid(rows: list[dict], key: str, out: Path) -> None:
    dims = sorted({r["dim"] for r in rows})
    allv = [r[key] for r in rows if key in r]
    if not allv:
        logger.warning("no data for %s", key)
        return
    ylo, yhi = min(allv), max(allv)
    ypad = 0.06 * (yhi - ylo or 0.1)
    nrow, ncol = len(NS), len(dims)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), squeeze=False, sharey=True)
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
                ax.set_ylabel(f"N={n} WARCs\n{key}", fontsize=10)
            if ri == nrow - 1:
                ax.set_xlabel("FLOPs", fontsize=9)
    handles = [
        Line2D([0], [0], color=HQ_COLOR, lw=2.5, marker="o", label="HQ (high_quality)"),
        Line2D([0], [0], color=DCLM_COLOR, lw=2.5, marker="s", label="DCLM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"DCLM vs HQ — {key} — grid of model scale x WARC count (N=100/300/500/1000/2000 random)\n"
        "OLMo Base-Easy bits-per-byte · LOWER is better",
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
    rows = _load()
    for n in NS:
        hq = sum(1 for r in rows if r["method"] == "high_quality" and r["n"] == n)
        dc = sum(1 for r in rows if r["method"] == "dclm" and r["n"] == n)
        logger.info("N=%d: HQ %d, DCLM %d", n, hq, dc)
    for key in ("macro_bpb", "Code_bpb", "Math_bpb", "QA_bpb"):
        _grid(rows, key, OUT_DIR / f"olmo_bpb_random_ladder_{key}.png")


if __name__ == "__main__":
    main()
