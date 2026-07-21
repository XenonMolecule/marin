# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot DCLM Core_v2 vs compute for the N=100 WARC-scaling models.

Replicates ``plot_core_v2_by_size_10k.py``'s style (one panel per model scale,
one colored line per curation method, Core_v2 on y) for the N=100 WARC sweep, and
overlays the two SAMPLING regimes so the biased-vs-random gap is directly visible:

  * BIASED  (dashed) : dclm_100 / nemotron_full_100 / nemotron_qhigh_100 /
                       high_quality_100 — head-100 of the date-SORTED 3k manifest
                       (2 crawls, both 2013).
  * RANDOM  (solid)  : {dclm,nemotron_full,high_quality}_random_100 — uniform draw
                       (seed 0) over the whole 10k pool (60 crawls, 2013-2022).

Same color per method across regimes; regime distinguished by linestyle. This
makes the headline finding legible at a glance: on N=100, HQ stays above DCLM in
BOTH regimes (it does NOT reproduce the HQ-worse crossover seen at 10k).

Two figures, x = FLOPs (the isoflop control variable) and x = tokens, each gridded
by model scale.

Data (join by run stem ``curation-<method>-expWARC_natural-<budget>-d<H>-L<L>-B<B>``):
  * Core_v2: ``<region>/metadata/data_curation_10k_core_results/<stem>_summary.json``
    -> ``dclm.Core_v2``. Summaries are scattered across regions, so ALL are scanned.
  * tokens + params: ``us-central1/metadata/data_curation_warc_scaling_results/<stem>.json``
    -> ``tokens.tokens_trained`` and ``model.total_trainable_params``.

Usage::
    export SSL_CERT_FILE=$(.venv/bin/python -m certifi)
    .venv/bin/python -m experiments.scaling_law_sweeps.plot_core_v2_warc100
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.scaling_law_sweeps.plot_curation_isoflop import COMPARE_COLORS

logger = logging.getLogger(__name__)

REGIONS = ["us-central1", "us-central2", "us-east1", "us-east5", "us-west4", "eu-west4"]
CORE_SUB = "metadata/data_curation_10k_core_results"
TOK_PREFIX = "gs://marin-us-central1/metadata/data_curation_warc_scaling_results"
OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "core_v2"

# base method -> (biased method name, random method name). qhigh has no random arm.
METHOD_ARMS: dict[str, tuple[str, str | None]] = {
    "dclm": ("dclm_100", "dclm_random_100"),
    "nemotron_full": ("nemotron_full_100", "nemotron_full_random_100"),
    "nemotron_qhigh": ("nemotron_qhigh_100", None),
    "high_quality": ("high_quality_100", "high_quality_random_100"),
}
COLORS = {
    "dclm": COMPARE_COLORS["dclm"],
    "nemotron_full": "#ff7f0e",
    "nemotron_qhigh": "#d62728",
    "high_quality": COMPARE_COLORS["high_quality"],
}
LABELS = {
    "dclm": "DCLM",
    "nemotron_full": "Nemotron-full",
    "nemotron_qhigh": "Nemotron-qhigh",
    "high_quality": "HQ (high_quality)",
}
# stem method token -> (base, regime)
_METHOD_TO_BASE_REGIME: dict[str, tuple[str, str]] = {}
for base, (bia, rnd) in METHOD_ARMS.items():
    _METHOD_TO_BASE_REGIME[bia] = (base, "biased")
    if rnd:
        _METHOD_TO_BASE_REGIME[rnd] = (base, "random")

_ALL_METHODS = "|".join(re.escape(m) for m in _METHOD_TO_BASE_REGIME)
_STEM_RE = re.compile(
    rf"^curation-(?P<method>{_ALL_METHODS})"
    r"-expWARC_natural-(?P<budget>[0-9eE+.\-]+)-d(?P<hidden>\d+)-L(?P<layers>\d+)-B(?P<batch>\d+)$"
)
_REGIME_STYLE = {"biased": (0, (5, 3)), "random": "solid"}  # dashed vs solid


def _sh(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def _cat_json(path: str) -> dict | None:
    out = _sh(["gcloud", "storage", "cat", path])
    try:
        return json.loads(out) if out.strip() else None
    except json.JSONDecodeError:
        return None


def _load() -> list[dict]:
    stem_region: dict[str, str] = {}
    for r in REGIONS:
        for line in _sh(["gcloud", "storage", "ls", f"gs://marin-{r}/{CORE_SUB}/"]).splitlines():
            fn = line.rsplit("/", 1)[-1]
            if not fn.endswith("_summary.json"):
                continue
            stem = fn[: -len("_summary.json")]
            if _STEM_RE.match(stem):  # excludes the -smoke run automatically
                stem_region.setdefault(stem, f"gs://marin-{r}/{CORE_SUB}")
    logger.info("Found %d warc100 core summaries (biased+random)", len(stem_region))

    def _one(stem: str, core_pref: str) -> dict | None:
        m = _STEM_RE.match(stem)
        base, regime = _METHOD_TO_BASE_REGIME[m.group("method")]
        cj = _cat_json(f"{core_pref}/{stem}_summary.json")
        core = (cj or {}).get("dclm", {}).get("Core_v2")
        tj = _cat_json(f"{TOK_PREFIX}/{stem}.json")
        tokens = (tj or {}).get("tokens", {}).get("tokens_trained")
        params = (tj or {}).get("model", {}).get("total_trainable_params")
        if core is None or tokens is None or params is None:
            logger.warning("skip %s (core=%s tokens=%s params=%s)", stem, core is not None, tokens, params)
            return None
        return dict(
            base=base,
            regime=regime,
            budget=float(m.group("budget")),
            hidden=int(m.group("hidden")),
            params=int(params),
            tokens=float(tokens),
            core=float(core),
        )

    recs: list[dict] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(_one, s, p) for s, p in stem_region.items()]
        for f in as_completed(futs):
            if (rec := f.result()) is not None:
                recs.append(rec)
    logger.info("Loaded %d joined records", len(recs))
    return recs


def _plabel(params: int) -> str:
    return f"{params / 1e9:.2f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"


def _figure(recs: list[dict], xkey: str, xlabel: str, out: Path, regimes: tuple[str, ...] = ("biased", "random")) -> None:
    recs = [r for r in recs if r["regime"] in regimes]
    dims = sorted({r["hidden"] for r in recs})
    params_for = {dim: max(r["params"] for r in recs if r["hidden"] == dim) for dim in dims}
    y_all = [r["core"] for r in recs]
    ylo, yhi = min(y_all), max(y_all)
    ypad = 0.06 * (yhi - ylo)
    ncol = min(len(dims), 3)
    nrow = math.ceil(len(dims) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow), squeeze=False)
    for idx, dim in enumerate(dims):
        ax = axes[idx // ncol][idx % ncol]
        for base in METHOD_ARMS:
            for regime in regimes:
                pts = sorted(
                    (r for r in recs if r["hidden"] == dim and r["base"] == base and r["regime"] == regime),
                    key=lambda r: r[xkey],
                )
                if not pts:
                    continue
                ax.plot(
                    [p[xkey] for p in pts],
                    [p["core"] for p in pts],
                    marker="o" if regime == "random" else "s",
                    ms=5,
                    lw=2,
                    color=COLORS[base],
                    linestyle=_REGIME_STYLE[regime],
                    alpha=0.95 if regime == "random" else 0.7,
                )
        ax.set_xscale("log")
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.axhline(0, color="#bbb", lw=0.8, ls=":")
        ax.set_title(f"d{dim} ({_plabel(params_for[dim])})", fontsize=11)
        ax.grid(True, which="major", alpha=0.25)
        if idx % ncol == 0:
            ax.set_ylabel("DCLM Core v2")
        if idx // ncol == nrow - 1:
            ax.set_xlabel(xlabel)
    for j in range(len(dims), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # legend: method colors + regime linestyles
    method_handles = [Line2D([0], [0], color=COLORS[b], lw=2.5, label=LABELS[b]) for b in METHOD_ARMS]
    _regime_label = {"random": "random (10k pool)", "biased": "biased (2013 head)"}
    _regime_marker = {"random": "o", "biased": "s"}
    regime_handles = [
        Line2D([0], [0], color="#444", lw=2, linestyle=_REGIME_STYLE[rg], marker=_regime_marker[rg], label=_regime_label[rg])
        for rg in regimes
    ]
    fig.legend(
        handles=method_handles + regime_handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.03)
    )
    _suffix = {("biased", "random"): "biased vs random", ("biased",): "biased (2013 head) only", ("random",): "random (10k pool) only"}
    fig.suptitle(
        f"DCLM Core v2 vs {xlabel} by model scale — N=100 WARC sweep ({_suffix.get(regimes, '/'.join(regimes))})",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("wrote %s", out)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    recs = _load()
    if not recs:
        raise SystemExit("no records")
    _figure(recs, "budget", "training FLOPs", OUT_DIR / "core_v2_warc100_by_size_x_flops.png")
    _figure(recs, "tokens", "tokens trained", OUT_DIR / "core_v2_warc100_by_size_x_tokens.png")
    _figure(recs, "budget", "training FLOPs", OUT_DIR / "core_v2_warc100_biased_x_flops.png", regimes=("biased",))
    _figure(recs, "tokens", "tokens trained", OUT_DIR / "core_v2_warc100_biased_x_tokens.png", regimes=("biased",))
    # console: HQ - DCLM gap per regime (the headline)
    for regime in ("biased", "random"):
        gaps = []
        for dim in sorted({r["hidden"] for r in recs}):
            for bud in sorted({r["budget"] for r in recs}):
                hq = [
                    r["core"]
                    for r in recs
                    if r["regime"] == regime
                    and r["base"] == "high_quality"
                    and r["hidden"] == dim
                    and r["budget"] == bud
                ]
                dc = [
                    r["core"]
                    for r in recs
                    if r["regime"] == regime and r["base"] == "dclm" and r["hidden"] == dim and r["budget"] == bud
                ]
                if hq and dc:
                    gaps.append(hq[0] - dc[0])
        if gaps:
            logger.info(
                "%s: HQ-DCLM mean gap %+.4f over %d matched cells (HQ better %d)",
                regime,
                sum(gaps) / len(gaps),
                len(gaps),
                sum(1 for g in gaps if g > 0),
            )


if __name__ == "__main__":
    main()
