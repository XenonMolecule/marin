# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!  FAKE / SIMULATED — DO NOT USE FOR ANY ANALYSIS OR REPORTING        !!!
!!!  Every record below is fabricated from a hand-tuned closed-form     !!!
!!!  loss model. NO real summary JSON is read. NO real GCS prefix is    !!!
!!!  touched. The output lives in a clearly-segregated directory and    !!!
!!!  every plot is overlaid with a giant red FAKE banner.               !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Purpose: visualize what the WARC-scaling sweep MIGHT look like once fully
populated, under a hypothetical outcome where:

  * At small scales (low tokens), {dclm, nemotron_full} WIN — they descend
    faster from a smaller A coefficient, so at the left edge of the plot
    they sit below the {resiliparse, llm_curated} cluster.
  * {dclm, nemotron_full} have the smallest D_obs and therefore overtrain
    first: their curves bend back up into a clear U-shape (parabola 1).
  * {resiliparse, llm_curated} start higher but keep descending. Their
    parabola is shifted to the right and dips even LOWER than parabola 1's
    minimum (parabola 2).
  * llm_curated is consistently a fixed amount (~0.05 LIMA loss) below
    resiliparse and reaches its bowl furthest to the right of all four.

Loss model:  L = E + A · t^(-α) + over · max(0, ln(t/D_obs))²
The first term controls "how fast the curve drops" (small A = drops fast,
wins early); E is the asymptote; the third term is the overtraining U.

Real script:  experiments/scaling_law_sweeps/plot_warc_scaling_sweep.py
Real output:  scratch/plots/warc_scaling/                  ← UNTOUCHED
Fake output:  scratch/plots/FAKE_SIMULATED_warc_scaling/   ← THIS SCRIPT

Usage:
    uv run python experiments/scaling_law_sweeps/FAKE_SIMULATED_warc_scaling_plot.py
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from experiments.scaling_law_sweeps.plot_curation_isoflop import load_summaries
from experiments.scaling_law_sweeps.plot_warc_scaling_sweep import (
    _FM_METHOD_MAP,
    WarcScalingRecord,
    _load_fm_lima_sidecar,
    _merge_sidecar_into_summary,
    fm_summary_to_record,
    plot_grid_compressed,
    plot_grid_overview,
)
from experiments.scaling_law_sweeps.warc_scaling_plan import (
    WARC_COUNTS,
    WARC_METHOD_BASE_NAMES,
    hidden_sizes_for,
)

# Real data anchor: the (h=1536, N=3000) cell is THE point we have measured.
# We graft real fixed-model summaries in for that one cell so the rest of
# the fabricated grid is visually grounded against it.
_REAL_ANCHOR_HIDDEN_DIM: int = 1536
_REAL_ANCHOR_N_WARCS: int = 3000
_LOCAL_FM_SUMMARIES: str = "scratch/fm_summaries/"
_LOCAL_FM_LIMA_SIDECAR: str = "scratch/fm_lima_sidecar/"

logger = logging.getLogger(__name__)

# ----- fabricated per-method parameters --------------------------------------
# Dimensionless D_obs scaling factor. D_obs(method, n) = factor[method] * n *
# _TOKENS_PER_WARC. Choices are NOT physical retention rates — they are tuning
# knobs that place each method's overtraining cliff (and therefore its bowl)
# on the tokens axis. dclm/nemotron get the smallest D_obs so they overtrain
# first and bend back up; llm_curated gets the largest so its bowl sits
# furthest right and barely overtrains within the plot range.
_D_OBS_FACTOR: dict[str, float] = {
    "dclm": 4.0,
    "nemotron_full": 5.0,
    "resiliparse": 30.0,
    "llm_curated": 30.0,  # tied to resiliparse; floor is the only difference
}

# Loss floor (asymptote of the power-law term).
# Two parabola ranges: dclm/nemotron at the higher floor, resi/llm at the
# lower floor with llm_curated 0.05 below resiliparse by design.
_FLOOR: dict[str, float] = {
    "dclm": 2.50,
    "nemotron_full": 2.50,
    "resiliparse": 2.45,
    "llm_curated": 2.40,
}

# Power-law A coefficient. Smaller A ⇒ less to drop ⇒ curve already near
# floor at small tokens ⇒ wins at the left of the plot. Larger A ⇒ starts
# high, drops slowly. We make {dclm, nemotron} small and {resi, llm} large
# so the crossover is built in.
# llm_curated is wired to track resiliparse: same A and same D_obs factor
# (see _D_OBS_FACTOR above) so the only difference between the two curves is
# their floor — giving a flat ~0.05 LIMA-loss gap at every token count.
_A_COEF: dict[str, float] = {
    "dclm": 46.0,
    "nemotron_full": 49.0,
    "resiliparse": 57.0,
    "llm_curated": 57.0,
}

_ALPHA: float = 0.20
_OVER_PENALTY: float = 0.04  # × ln(max(1, epochs))^2

# Tokens of D_obs per WARC at factor=1.0. Tuned for visual range, not real.
_TOKENS_PER_WARC: float = 1_500_000.0

# Hand-picked params at each hidden_dim that appears in the WARC-scaling and
# fixed-model plans. Used only to label subplot titles + compute tokens from
# FLOPs budget, so exact values don't matter — they just need to monotonically
# increase with hidden_dim.
_PARAMS_BY_HIDDEN: dict[int, int] = {
    256: 35_000_000,
    384: 75_000_000,
    512: 157_000_000,
    768: 405_000_000,
    1024: 666_000_000,
    1280: 850_000_000,
    1536: 998_000_000,  # matches the d=1536 subplot label on the real grid
    2048: 1_700_000_000,
    2432: 2_900_000_000,
    3328: 4_500_000_000,
    3584: 5_500_000_000,
}


def _budget_grid_for(n_warcs: int) -> list[float]:
    """FLOPs grid spanning under-, optimally-, and over-trained regimes."""
    if n_warcs <= 100:
        return [3e15, 1e16, 3e16, 1e17, 3e17, 1e18, 3e18]
    if n_warcs <= 500:
        return [1e16, 3e16, 1e17, 3e17, 1e18, 3e18, 1e19]
    if n_warcs <= 1000:
        return [3e16, 1e17, 3e17, 1e18, 3e18, 1e19, 3e19, 1e20]
    if n_warcs <= 2000:
        return [1e17, 3e17, 1e18, 3e18, 1e19, 3e19, 1e20, 3e20]
    return [3e17, 1e18, 3e18, 1e19, 3e19, 1e20, 3e20]


def _hidden_sizes_with_anchor(n_warcs: int) -> list[int]:
    if n_warcs == 3000:
        return [512, 1536, 2432]  # FM trio
    return sorted(set(hidden_sizes_for(n_warcs)) | {512})


def _d_obs(method: str, n_warcs: int) -> float:
    return _D_OBS_FACTOR[method] * n_warcs * _TOKENS_PER_WARC


def _fake_loss(method: str, tokens: float, d_obs: float) -> float:
    L = _FLOOR[method] + _A_COEF[method] * tokens ** (-_ALPHA)
    epochs = max(1.0, tokens / d_obs)
    L += _OVER_PENALTY * math.log(epochs) ** 2
    return L


def load_real_anchor_records(metric_key: str = "eval/lima/loss") -> list[WarcScalingRecord]:
    """Load REAL fixed-model summaries at h=1536 and rewrite them as N=3000
    WarcScalingRecords. This is the only piece of real data in the whole
    visualization — used to anchor the fabricated rest of the grid.
    """
    fm_summaries = load_summaries(_LOCAL_FM_SUMMARIES, list(_FM_METHOD_MAP.keys()), suffix="")
    sidecar = _load_fm_lima_sidecar(_LOCAL_FM_LIMA_SIDECAR)
    for s in fm_summaries:
        _merge_sidecar_into_summary(s, sidecar)

    records: list[WarcScalingRecord] = []
    for s in fm_summaries:
        r = fm_summary_to_record(s, metric_key)
        if r is None:
            continue
        if r.hidden_dim != _REAL_ANCHOR_HIDDEN_DIM:
            continue
        records.append(r)
    logger.info("Grafted %d REAL records at (h=%d, N=%d)", len(records), _REAL_ANCHOR_HIDDEN_DIM, _REAL_ANCHOR_N_WARCS)
    return records


def make_fake_records() -> list[WarcScalingRecord]:
    records: list[WarcScalingRecord] = []
    n_values = list(WARC_COUNTS) + [3000]
    for n in n_values:
        for h in _hidden_sizes_with_anchor(n):
            params = _PARAMS_BY_HIDDEN[h]
            for method in WARC_METHOD_BASE_NAMES:
                d_obs = _d_obs(method, n)
                for budget in _budget_grid_for(n):
                    tokens = budget / (6.0 * params)
                    if tokens < 1e7 or tokens > 5e12:
                        continue
                    loss = _fake_loss(method, tokens, d_obs)
                    epochs = tokens / d_obs
                    records.append(
                        WarcScalingRecord(
                            method=method,
                            sampled_warcs=n,
                            hidden_dim=h,
                            params=params,
                            flops=budget,
                            tokens=tokens,
                            loss=loss,
                            epochs_over_d_obs=epochs,
                        )
                    )
    return records


# ----- big red banner injected into every produced HTML ----------------------
_RED_BANNER_HTML = """
<div style="position:fixed;top:0;left:0;right:0;z-index:99999;background:#b00020;color:#fff;text-align:center;padding:18px 12px;font-family:monospace;font-size:30px;font-weight:900;letter-spacing:1px;border-bottom:6px solid #ff0033;box-shadow:0 6px 18px rgba(0,0,0,0.45);">
&#x26A0;&#xFE0F; FAKE / SIMULATED &mdash; FABRICATED DATA &mdash; NOT REAL RESULTS &#x26A0;&#xFE0F;<br>
<span style="font-size:14px;font-weight:normal;display:inline-block;margin-top:6px;">
generator: experiments/scaling_law_sweeps/FAKE_SIMULATED_warc_scaling_plot.py &mdash; do not use for analysis, reports, or decisions
</span>
</div>
<div style="height:104px;"></div>
"""

_TITLE_REPLACEMENTS = [
    ("WARC-scaling COMPRESSED grid", "[FAKE / SIMULATED] WARC-scaling COMPRESSED grid"),
    ("WARC-scaling grid", "[FAKE / SIMULATED] WARC-scaling grid"),
]


def _inject_fake_markers(html_path: Path) -> None:
    text = html_path.read_text()
    if "FABRICATED DATA" in text:  # idempotent: already stamped
        return
    new_text = re.sub(r"(<body[^>]*>)", r"\1" + _RED_BANNER_HTML, text, count=1)
    if new_text == text:
        new_text = _RED_BANNER_HTML + text
    for old, new in _TITLE_REPLACEMENTS:
        new_text = new_text.replace(old, new)
    html_path.write_text(new_text)


def _rename_with_fake_prefix(html_path: Path) -> Path:
    if html_path.name.startswith("FAKE_SIMULATED_"):
        return html_path
    new_path = html_path.with_name(f"FAKE_SIMULATED_{html_path.name}")
    html_path.rename(new_path)
    return new_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    output_root = Path("scratch/plots/FAKE_SIMULATED_warc_scaling")
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "README_THIS_IS_FAKE.txt").write_text(
        "EVERY FILE IN THIS DIRECTORY TREE IS FABRICATED.\n"
        "No real training run produced these numbers.\n"
        "Generator: experiments/scaling_law_sweeps/FAKE_SIMULATED_warc_scaling_plot.py\n"
        "Real plots live in: scratch/plots/warc_scaling/ (untouched by this script).\n"
        "Do not use these plots for any analysis, report, or decision-making.\n"
    )

    metric_key = "eval/lima/loss"
    records = make_fake_records()
    logger.info(
        "Generated %d fabricated records across %d (method, n, h) cells",
        len(records),
        len({(r.method, r.sampled_warcs, r.hidden_dim) for r in records}),
    )

    # Drop fabricated rows at the anchor cell and replace with REAL data.
    real_anchor = load_real_anchor_records(metric_key)
    records = [
        r for r in records if not (r.hidden_dim == _REAL_ANCHOR_HIDDEN_DIM and r.sampled_warcs == _REAL_ANCHOR_N_WARCS)
    ]
    records.extend(real_anchor)
    logger.info(
        "Final dataset: %d records (%d real anchor + %d fake elsewhere)",
        len(records),
        len(real_anchor),
        len(records) - len(real_anchor),
    )
    metric_dir = output_root / metric_key.replace("/", "_")
    metric_dir.mkdir(parents=True, exist_ok=True)

    for x_axis in ("tokens", "flops"):
        plot_grid_compressed(records, metric_key, metric_dir, x_axis=x_axis, mode="all_epochs")
        plot_grid_overview(records, metric_key, metric_dir, x_axis=x_axis, mode="all_epochs")

    n_processed = 0
    for html in sorted(metric_dir.rglob("*.html")):
        _inject_fake_markers(html)
        renamed = _rename_with_fake_prefix(html)
        n_processed += 1
        logger.info("Marked + renamed: %s", renamed.relative_to(output_root))
    logger.info("Done. %d FAKE plots in %s", n_processed, output_root.resolve())


if __name__ == "__main__":
    main()
