# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quality-tier winner regions over (N, C), one panel per WARC subsample.

Visualizes which quality tier (LQ / MQ / HQ) has the lowest predicted loss
across the (parameters N, compute C) plane, with smooth decision boundaries
between regions. One panel per fixed WARC subsample (e.g. 100 / 500 / 8M).

Uses the V3 Sedova fit:

    L(N, D, U) = E + C_m·N^(-β_m) + B_m·N^(δ_m) · D_eff_m^(-α_m)
    D_eff_m   = U·R_D_m·(1 − exp(−ε/R_D_m))         ε = D/U

Shared:    E
Per-method: R_D, C, β, B, δ, α
No damage; gold data filtered with drop_post_min.

Per panel, U_m = TPW_m · sampled_warcs. At each (N, C) we evaluate L_m for
each method (D = C/(6N), ε = D/U_m) and color the region by argmin_m L_m.
Boundary curves are extracted as zero contours of the pairwise L differences.

Usage:

    # Refresh local CSV (one-time after new gold runs land), then plot:
    uv run --with matplotlib --with numpy --with pandas python \\
        experiments/scaling_law_sweeps/plot_quality_winner_regions.py --pull

    # Fast re-plot from local data:
    uv run --with matplotlib --with numpy --with pandas python \\
        experiments/scaling_law_sweeps/plot_quality_winner_regions.py

    # Custom WARC values:
    uv run --with matplotlib --with numpy --with pandas python \\
        experiments/scaling_law_sweeps/plot_quality_winner_regions.py \\
        --warcs 100 1000 8000000

Outputs `quality_winner_regions__W{w1}_W{w2}_W{w3}.{png,pdf}` in
`scratch/plots/quality_winner_regions/`.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import logging
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# Fit infra lives alongside the production dashboard in scratch/plots/.
sys.path.insert(0, "scratch/plots")

logger = logging.getLogger(__name__)

DEFAULT_GOLD_CSV = Path("scratch/exports/warc_scaling_streamlined.csv")
DEFAULT_OUTPUT_DIR = Path("scratch/plots/quality_winner_regions")
DEFAULT_RESULTS_GS = "gs://marin-us-central1/metadata/data_curation_warc_scaling_results/"
DEFAULT_AUDIT_DIR = Path("scratch/audit_summaries")
FIT_CACHE_DIR = Path("scratch/plots/fit_cache")

METRIC = "eval_uncheatable_macro_loss"
METHODS_ORDER: tuple[str, ...] = ("low_quality", "med_quality", "high_quality")

DEFAULT_WARCS: tuple[int, ...] = (100, 500, 8_000_000)


@dataclass(frozen=True)
class MethodStyle:
    label: str
    line: str  # used for boundary curves and data markers
    shade: str  # used for filled winner regions


METHOD_STYLES: dict[str, MethodStyle] = {
    "high_quality": MethodStyle("HQ (ours)", "#2ca02c", "#c8e6c9"),
    "med_quality": MethodStyle("MQ (ours)", "#ff7f0e", "#fff59d"),
    "low_quality": MethodStyle("LQ (ours)", "#e91e63", "#ffcdd2"),
}


PAPER_RCPARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "legend.frameon": False,
    "figure.dpi": 120,
}


def fmt_params(x: float, _pos=None) -> str:
    """log10-tick formatter for parameters (x is log10 N)."""
    n = 10**x
    if n >= 1e12:
        return f"{n / 1e12:.0f}T"
    if n >= 1e9:
        return f"{n / 1e9:.0f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return f"{n:.0f}"


def fmt_warcs(w: int) -> str:
    if w >= 1_000_000:
        return f"{w / 1e6:.1f}M".replace(".0M", "M")
    if w >= 1_000:
        return f"{w / 1e3:.0f}K"
    return str(w)


def pull_warc_summaries(audit_dir: Path) -> None:
    if shutil.which("gcloud") is None:
        raise RuntimeError("`gcloud` not on PATH; cannot --pull.")
    audit_dir.mkdir(parents=True, exist_ok=True)
    src = DEFAULT_RESULTS_GS.rstrip("/") + "/*.json"
    logger.info("pulling %s -> %s", src, audit_dir)
    subprocess.run(["gcloud", "storage", "cp", src, str(audit_dir) + "/"], check=True)


def regenerate_streamlined_csv() -> None:
    """Re-run experiments.scaling_law_sweeps.export_csv to refresh the local
    streamlined CSV from the already-pulled warc summaries.
    """
    sys.path.insert(0, "experiments/scaling_law_sweeps")
    import export_csv as e

    e.LOCAL_FM_DIR.mkdir(parents=True, exist_ok=True)
    e.LOCAL_FM_LIMA_SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    saved_argv = sys.argv
    sys.argv = ["export_csv", "--out-dir", "scratch/exports"]
    try:
        e.main()
    finally:
        sys.argv = saved_argv


def load_gold(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("flops_target", "flops_actual", "tokens", "parameters", "epochs", "unique_tokens", METRIC):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[METRIC, "tokens", "parameters", "epochs", "unique_tokens"])
    df = df[(df.epochs > 0) & (df.tokens > 0) & (df.parameters > 0)]
    df = df[df["method"].isin(METHODS_ORDER)].copy().reset_index(drop=True)
    df["U_eff"] = df["tokens"] / df["epochs"]
    return df


def clip_to_min(df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, parameters, warcs): replace any L above the running-min
    (in token order) with the running-min. Keeps all rows; flattens the
    over-epoching upturn rather than dropping it."""
    new_L = df[METRIC].copy()
    for (_m, _p, _w), grp in df.groupby(["method", "parameters", "warcs"]):
        gs = grp.sort_values("tokens")
        rmin = gs[METRIC].cummin()
        idx = gs.index[gs[METRIC] > rmin + 1e-9]
        new_L.loc[idx] = rmin.loc[idx]
    return df.assign(**{METRIC: new_L})


def tokens_per_warc(df: pd.DataFrame) -> dict[str, float]:
    return {m: float((g["unique_tokens"].astype(float) / g["warcs"]).mean()) for m, g in df.groupby("method")}


def _predict_optionB(params_per_m, N, D, U):
    """Option B: D_eff = U·R_D·(1−exp(−ε/R_D)) · exp(−ε/R_decay).
    Decoupled saturation (R_D) and bend-up (R_decay) scales.
    8 params per method: E, C, β, B, δ, α, R_D, R_decay."""
    E, C, beta, B, delta, alpha, R_D, R_decay = params_per_m
    eps = D / U
    D_eff_sat = U * R_D * (1.0 - np.exp(-eps / R_D))
    decay = np.exp(-eps / R_decay)
    D_eff = np.maximum(D_eff_sat * decay, 1e-30)
    return E + C * N ** (-beta) + B * N**delta * D_eff ** (-alpha)


def _data_hash(df_fit: pd.DataFrame) -> str:
    """Stable sha256 of the fit-relevant columns, sorted."""
    cols = ["method", "parameters", "tokens", "epochs", "unique_tokens", METRIC]
    payload = df_fit[cols].sort_values(cols).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_hash() -> str:
    """Sha256 of the fit/predict source — invalidates the cache on any code edit
    to bounds, restart logic, or the predict function. Whitespace-sensitive,
    which is fine: false-positives just trigger a refit, never silent staleness.
    """
    src = inspect.getsource(_predict_optionB) + inspect.getsource(fit_w1)
    return hashlib.sha256(src.encode()).hexdigest()


def _fit_cache_key(df_fit: pd.DataFrame, n_restarts: int) -> str:
    h = hashlib.sha256()
    h.update(_data_hash(df_fit).encode())
    h.update(f"|n_restarts={n_restarts}".encode())
    h.update(f"|src={_source_hash()}".encode())
    return h.hexdigest()[:16]


def _manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def fit_w1_cached(df_fit: pd.DataFrame, n_restarts: int, refit: bool):
    """Cache-wrapped Option B fit.

    Cache key includes:
      • sha256 of the fit-relevant data columns
      • n_restarts
      • sha256 of the predict/fit source

    Any change in data, fit hyperparams, or fit/predict source code
    invalidates the cache automatically. Writes a `.json` manifest sidecar
    with human-readable provenance.
    """
    FIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _fit_cache_key(df_fit, n_restarts)
    cache_path = FIT_CACHE_DIR / f"optionB__{key}.pkl"
    manifest_path = _manifest_path(cache_path)

    if not refit and cache_path.exists():
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        if manifest_path.exists():
            mf = json.loads(manifest_path.read_text())
            logger.info(
                "Loaded cached fit from %s  " "(written %s, RMSE=%.4f, %d rows)",
                cache_path.name,
                mf.get("timestamp", "?"),
                mf.get("rmse", float("nan")),
                mf.get("n_rows", -1),
            )
        else:
            logger.info("Loaded cached fit from %s (no manifest)", cache_path)
        return cached

    if refit and cache_path.exists():
        logger.info("Refit requested; ignoring cache at %s", cache_path.name)

    res = fit_w1(df_fit, n_restarts=n_restarts)

    # Aggregate RMSE across all methods for the manifest.
    rss_total, n_total = 0.0, 0
    for m in METHODS_ORDER:
        sub = df_fit[df_fit["method"] == m]
        L = sub[METRIC].to_numpy(dtype=float)
        rss_total += float(np.sum((L - res["fits"][m]["pred"]) ** 2))
        n_total += len(L)
    rmse = (rss_total / n_total) ** 0.5

    with cache_path.open("wb") as f:
        pickle.dump(res, f)
    manifest = {
        "cache_key": key,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_restarts": n_restarts,
        "n_rows": len(df_fit),
        "data_hash": _data_hash(df_fit),
        "source_hash": _source_hash(),
        "rmse": rmse,
        "fit_summary": {m: {"params": [float(x) for x in res["fits"][m]["params"]]} for m in METHODS_ORDER},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Saved fit + manifest to %s", cache_path.name)
    return res


def clear_fit_cache() -> int:
    if not FIT_CACHE_DIR.exists():
        return 0
    count = 0
    for p in FIT_CACHE_DIR.iterdir():
        if p.is_file() and p.suffix in (".pkl", ".json"):
            p.unlink()
            count += 1
    return count


def fit_w1(df_fit: pd.DataFrame, n_restarts: int = 100):
    """Option B SPP: shared (E, C, β, R_D, R_decay); per-method (B, δ, α).
    NO clip_to_min — bend-up rows kept and modeled by exp(-ε/R_decay) factor.
    """
    from fit_muennighoff_v2 import _loss_fn, _MagnitudeStep
    from scipy.optimize import basinhopping

    pmd = {}
    for m in METHODS_ORDER:
        sub = df_fit[df_fit["method"] == m]
        pmd[m] = (
            sub["parameters"].to_numpy(dtype=float),
            sub["tokens"].to_numpy(dtype=float),
            sub["U_eff"].to_numpy(dtype=float),
            sub[METRIC].to_numpy(dtype=float),
        )
    methods = list(pmd)
    lo_shared = np.array([0.0, 0.0, 0.05, 0.1, 1.0])
    hi_shared = np.array([6.0, 1e10, 1.5, 200.0, 5000.0])
    x0_shared = np.array([2.0, 1e3, 0.40, 3.0, 100.0])
    lo_per = np.array([0.0, -0.5, 0.05])
    hi_per = np.array([1e10, 1.5, 1.5])
    x0_per = np.array([1e2, 0.18, 0.45])
    lo = np.concatenate([lo_shared, np.tile(lo_per, len(methods))])
    hi = np.concatenate([hi_shared, np.tile(hi_per, len(methods))])
    bounds = list(zip(lo, hi))

    def unpack(p):
        E, C, beta, R_D, R_decay = p[:5]
        per = {}
        for i, m in enumerate(methods):
            base = 5 + 3 * i
            B, delta, alpha = p[base : base + 3]
            per[m] = (E, C, beta, B, delta, alpha, R_D, R_decay)
        return per

    def obj(p):
        per = unpack(p)
        total = 0.0
        for m, params in per.items():
            N, D, U, L = pmd[m]
            pred = _predict_optionB(params, N, D, U)
            r = pred - L
            total += _loss_fn("huber_raw", r)
        return total

    rng = np.random.default_rng(0)
    take_step = _MagnitudeStep(lo, hi, stepsize=0.5, rng=rng)
    seed = np.clip(np.concatenate([x0_shared, np.tile(x0_per, len(methods))]), lo, hi)
    res = basinhopping(
        obj,
        seed,
        minimizer_kwargs=dict(method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000, "ftol": 1e-12}),
        niter=n_restarts,
        take_step=take_step,
        disp=False,
        seed=0,
    )
    p_final = np.clip(res.x, lo, hi)
    per = unpack(p_final)
    fits = {}
    for m in methods:
        N, D, U, L = pmd[m]
        fits[m] = {
            "params": per[m],
            "pred": _predict_optionB(per[m], N, D, U),
        }
    return {
        "E": p_final[0],
        "R_D": p_final[3],
        "tau_dmg": None,
        "C_shared": p_final[1],
        "beta_shared": p_final[2],
        "R_decay": p_final[4],
        "fits": fits,
    }


def evaluate_grid(fits, TPW, log_N, log_C, warcs):
    """Per-method L on a (log_N, log_C) grid; returns (losses, winner_idx)."""
    N = (10.0**log_N)[None, :]
    C = (10.0**log_C)[:, None]
    D = C / (6 * N)
    losses = []
    for m in METHODS_ORDER:
        U = TPW[m] * warcs
        params = fits[m]["params"]
        L = _predict_optionB(
            params,
            np.broadcast_to(N, D.shape),
            D,
            np.full_like(D, U),
        )
        losses.append(L)
    losses = np.stack(losses, axis=0)
    winner_idx = np.argmin(losses, axis=0)
    return losses, winner_idx


def render_panel(
    ax,
    fits,
    TPW: dict[str, float],
    warcs: int,
    log_N_grid: np.ndarray,
    log_C_grid: np.ndarray,
    data_for_warc: pd.DataFrame | None,
    min_tokens: float,
    trust_overlay: dict | None = None,
    boundary_min_logc: float | None = None,
    show_chinchilla: bool = False,
) -> None:
    """Render one (N, C) winner-region panel for a fixed warcs value.

    Cells with D = C/(6N) < min_tokens are masked out (too few training
    steps to trust the form's extrapolation).

    trust_overlay (optional): dict with keys
        "fracs"       (n_methods, n_C_b, n_N_b) — per-method bootstrap winner fraction
        "log_N_grid"  (n_N_b,)
        "log_C_grid"  (n_C_b,)
    When present, the base fill is replaced by a per-cell RGB blend of the
    method shade colors weighted by their bootstrap winner fractions. Cells
    where two methods compete appear as visible mixtures (peach for LQ/MQ,
    olive for MQ/HQ, etc.) rather than a solid color.
    """
    losses, winner_idx = evaluate_grid(fits, TPW, log_N_grid, log_C_grid, warcs)

    # Under-trained mask: D = C/(6N) >= min_tokens (no-op when min_tokens<=0)
    if min_tokens > 0:
        log_N_col = log_N_grid[None, :]
        log_C_row = log_C_grid[:, None]
        log_D = log_C_row - np.log10(6.0) - log_N_col
        trained_enough = log_D >= np.log10(min_tokens)
    else:
        trained_enough = np.ones_like(winner_idx, dtype=bool)

    if trust_overlay is not None and "fracs" in trust_overlay:
        # Each cell maps to an RGB color derived from per-method bootstrap
        # winner fractions. Two modes:
        #   - weighted:  full weighted RGB mix across all 3 methods (V1).
        #   - top2_mix:  pure top-1 color when trust ≥ threshold; otherwise
        #                50/50 of the top-1 and top-2 colors. Gives crisp
        #                trust regions and a single discrete "uncertain"
        #                color per pair of competing methods.
        fracs = trust_overlay["fracs"]
        lN_b = trust_overlay["log_N_grid"]
        lC_b = trust_overlay["log_C_grid"]
        mode = trust_overlay.get("blend_mode", "weighted")
        shade_rgb = np.stack(
            [np.array(to_rgb(METHOD_STYLES[m].shade)) for m in METHODS_ORDER],
            axis=0,
        )
        if mode == "top2_mix":
            threshold = trust_overlay.get("threshold", 0.95)
            order = np.argsort(-fracs, axis=0)  # descending; (n_methods, H, W)
            top1_color = shade_rgb[order[0]]  # (H, W, 3)
            top2_color = shade_rgb[order[1]]
            mix = 0.5 * top1_color + 0.5 * top2_color
            confident = (fracs.max(axis=0) >= threshold)[..., None]
            blend = np.where(confident, top1_color, mix)
        else:
            blend = np.einsum("mij,mc->ijc", fracs, shade_rgb)
        # Apply min-tokens mask on the bootstrap grid (resample by digitize).
        if min_tokens > 0:
            log_D_b = lC_b[:, None] - np.log10(6.0) - lN_b[None, :]
            mask_b = log_D_b >= np.log10(min_tokens)
            alpha = mask_b.astype(float)
        else:
            alpha = np.ones(blend.shape[:2])
        rgba = np.concatenate([blend, alpha[..., None]], axis=2)
        ax.imshow(
            rgba,
            origin="lower",
            extent=(lN_b[0], lN_b[-1], lC_b[0], lC_b[-1]),
            aspect="auto",
            interpolation="bilinear",
            zorder=1,
        )
    else:
        # Fallback: solid consensus fill, same as the non-overlay variant.
        winner_for_fill = winner_idx.astype(float)
        winner_for_fill[~trained_enough] = np.nan
        shade_colors = [METHOD_STYLES[m].shade for m in METHODS_ORDER]
        ax.contourf(
            log_N_grid,
            log_C_grid,
            winner_for_fill,
            levels=[-0.5, 0.5, 1.5, 2.5],
            colors=shade_colors,
            zorder=1,
        )

    # Smooth boundary curves: for each pair, the zero contour of L_a - L_b,
    # masked to where (a) the pair beats the third method, AND (b) we are in
    # the trained-enough region.
    third_method_for_pair = {
        (0, 1): 2,  # LQ vs MQ → third is HQ
        (0, 2): 1,  # LQ vs HQ → third is MQ
        (1, 2): 0,  # MQ vs HQ → third is LQ
    }
    if boundary_min_logc is not None:
        boundary_logc_ok = log_C_grid[:, None] >= boundary_min_logc
    else:
        boundary_logc_ok = np.ones_like(trained_enough, dtype=bool)
    for (i, j), k in third_method_for_pair.items():
        diff = losses[i] - losses[j]
        pair_beats_third = (losses[i] <= losses[k]) | (losses[j] <= losses[k])
        mask = trained_enough & pair_beats_third & boundary_logc_ok
        diff_masked = np.where(mask, diff, np.nan)
        ax.contour(
            log_N_grid,
            log_C_grid,
            diff_masked,
            levels=[0.0],
            colors="#222222",
            linewidths=1.8,
            linestyles="dotted",
            zorder=4,
        )

    # No trust contour for now — the blend already communicates uncertainty.

    # Diagonal line marking the min-tokens floor (only when floor is active).
    if min_tokens > 0:
        floor_logC = np.log10(6.0 * min_tokens) + log_N_grid
        in_view = (floor_logC >= log_C_grid[0]) & (floor_logC <= log_C_grid[-1])
        if in_view.any():
            ax.plot(log_N_grid[in_view], floor_logC[in_view], color="#555555", linewidth=1.4, alpha=0.7, zorder=3)

    # Chinchilla-optimal training: D = 20·N, C = 6·N·D = 120·N²
    # → log10 C = log10(120) + 2·log10 N.
    if show_chinchilla:
        chin_logC = np.log10(120.0) + 2.0 * log_N_grid
        in_view = (chin_logC >= log_C_grid[0]) & (chin_logC <= log_C_grid[-1])
        if in_view.any():
            ax.plot(
                log_N_grid[in_view],
                chin_logC[in_view],
                color="#ff7f0e",
                linewidth=1.8,
                alpha=0.95,
                zorder=3,
                label="Chinchilla-optimal",
            )

    # Gold empirical-winner dots: one dot per (parameters, flops_target) cell
    # where ≥2 methods coexist, colored by the method with the lowest loss
    # at that cell. Matches the dashboard's semantics — a color mismatch
    # between dot and background fill reveals where the fit disagrees with
    # the empirical winner.
    if data_for_warc is not None and len(data_for_warc):
        cell_keys = ["parameters", "flops_target"]
        counts = data_for_warc.groupby(cell_keys)["method"].nunique()
        contested = counts[counts >= 2].index
        df_contested = data_for_warc[data_for_warc.set_index(cell_keys).index.isin(contested)]
        if len(df_contested):
            winners = df_contested.loc[df_contested.groupby(cell_keys)[METRIC].idxmin()]
            for m in METHODS_ORDER:
                style = METHOD_STYLES[m]
                sub = winners[winners["method"] == m]
                if not len(sub):
                    continue
                ax.scatter(
                    np.log10(sub["parameters"].astype(float)),
                    np.log10(sub["flops_actual"].astype(float)),
                    s=70,
                    color=style.line,
                    edgecolors="white",
                    linewidths=1.2,
                    zorder=6,
                    alpha=0.95,
                )

    ax.set_xlim(log_N_grid[0], log_N_grid[-1])
    ax.set_ylim(log_C_grid[0], log_C_grid[-1])
    ax.set_title(f"WARCs = {fmt_warcs(warcs)}", pad=10, fontweight="semibold")
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.55, zorder=0)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_params))
    ax.set_xlabel("parameters (N)")


def render_figure(
    fits,
    TPW: dict[str, float],
    df_full: pd.DataFrame,
    warcs_values: list[int],
    log_N_ranges: list[tuple[float, float]],
    log_C_ranges: list[tuple[float, float]],
    n_grid: int,
    min_tokens: float,
    out_dir: Path,
    suffix: str,
    trust_overlays: dict[int, dict] | None = None,
    boundary_min_logc_per_panel: list[float | None] | None = None,
    chinchilla_panels: set[int] | None = None,
) -> tuple[Path, Path]:
    plt.rcParams.update(PAPER_RCPARAMS)

    n_panels = len(warcs_values)
    # If all panels share the same range, share axes; otherwise let each have
    # its own.
    share_x = all(r == log_N_ranges[0] for r in log_N_ranges)
    share_y = all(r == log_C_ranges[0] for r in log_C_ranges)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.5 * n_panels, 8.5),
        sharex=share_x,
        sharey=share_y,
    )
    if n_panels == 1:
        axes = [axes]

    avail = sorted(df_full["warcs"].unique())
    if boundary_min_logc_per_panel is None:
        boundary_min_logc_per_panel = [None] * n_panels
    chinchilla_panels = chinchilla_panels or set()
    for idx, (ax, warcs, lN_range, lC_range, b_min) in enumerate(
        zip(
            axes,
            warcs_values,
            log_N_ranges,
            log_C_ranges,
            boundary_min_logc_per_panel,
            strict=True,
        ),
        start=1,
    ):
        log_N_grid = np.linspace(*lN_range, n_grid)
        log_C_grid = np.linspace(*lC_range, n_grid)
        # Only overlay data dots if the requested warcs is exactly in the data
        # (otherwise we'd be confusing the reader by showing data at a
        # different warcs than the panel is computing for).
        data_for_warc = df_full[df_full["warcs"] == warcs] if warcs in avail else None
        trust_for_warc = (trust_overlays or {}).get(warcs)
        render_panel(
            ax,
            fits,
            TPW,
            warcs,
            log_N_grid,
            log_C_grid,
            data_for_warc,
            min_tokens,
            trust_overlay=trust_for_warc,
            boundary_min_logc=b_min,
            show_chinchilla=(idx in chinchilla_panels),
        )

    axes[0].set_ylabel("log₁₀ compute (FLOPs)")

    # Shared bottom legend with method squares + boundary line
    legend_handles = [
        Patch(
            facecolor=METHOD_STYLES[m].shade,
            edgecolor=METHOD_STYLES[m].line,
            linewidth=1.2,
            label=METHOD_STYLES[m].label,
        )
        for m in METHODS_ORDER
    ]
    from matplotlib.lines import Line2D

    legend_handles.append(
        Line2D([0], [0], color="#222222", linewidth=1.8, linestyle="dotted", label="winner boundary"),
    )
    if min_tokens > 0:
        legend_handles.append(
            Line2D([0], [0], color="#555555", linewidth=1.4, label="min-tokens floor"),
        )
    if chinchilla_panels:
        legend_handles.append(
            Line2D([0], [0], color="#ff7f0e", linewidth=1.8, label="Chinchilla-optimal"),
        )
    if trust_overlays:
        # Show example blend swatches so readers can decode the mixed colors.
        lq = np.array(to_rgb(METHOD_STYLES["low_quality"].shade))
        mq = np.array(to_rgb(METHOD_STYLES["med_quality"].shade))
        hq = np.array(to_rgb(METHOD_STYLES["high_quality"].shade))
        legend_handles.append(
            Patch(facecolor=tuple(0.5 * (lq + mq)), edgecolor="none", label="LQ/MQ uncertain"),
        )
        legend_handles.append(
            Patch(facecolor=tuple(0.5 * (mq + hq)), edgecolor="none", label="MQ/HQ uncertain"),
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.02),
        handlelength=2.4,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=(0.02, 0.10, 0.99, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "quality_winner_regions__OPTIONB_TRUST_BLEND__" + "_".join(f"W{w}" for w in warcs_values)
    if suffix:
        stem += f"__{suffix}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return png, pdf


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--warcs", nargs="+", type=int, default=list(DEFAULT_WARCS), help="WARC subsample sizes to render as panels."
    )
    parser.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-grid", type=int, default=600, help="Resolution of the (log N, log C) grid per panel.")
    parser.add_argument(
        "--log-n-range",
        nargs=2,
        type=float,
        default=[7.8, 12.0],
        help="Default log10 N range (min max) used for all "
        "panels. Default 7.8 .. 12. Overridden per-panel "
        "by --log-n-ranges if provided.",
    )
    parser.add_argument(
        "--log-c-range",
        nargs=2,
        type=float,
        default=[17.0, 30.0],
        help="Default log10 C range (min max) used for all "
        "panels. Default 17 .. 30. Overridden per-panel "
        "by --log-c-ranges if provided.",
    )
    parser.add_argument(
        "--log-n-ranges",
        nargs="+",
        default=None,
        help="Per-panel log10 N ranges as 'lo,hi' strings, "
        "one per --warcs panel. Example: "
        "--log-n-ranges 7.8,12 7.8,11 7.8,10",
    )
    parser.add_argument(
        "--log-c-ranges",
        nargs="+",
        default=None,
        help="Per-panel log10 C ranges as 'lo,hi' strings, "
        "one per --warcs panel. Example: "
        "--log-c-ranges 17,30 17,28 17,25",
    )
    parser.add_argument(
        "--bootstrap-npz",
        type=Path,
        default=None,
        help="Path to bootstrap results .npz (produced by "
        "bootstrap_trust_regions__optionB.py). If set, "
        "overlays uncertain-region hatch + trust contour.",
    )
    parser.add_argument(
        "--trust-threshold",
        type=float,
        default=0.95,
        help="Trust threshold for the bootstrap overlay. "
        "Cells with consensus winner fraction below this "
        "get the hatched 'uncertain' treatment. "
        "Default 0.95.",
    )
    parser.add_argument(
        "--blend-mode",
        choices=["weighted", "top2_mix"],
        default="weighted",
        help="How uncertain cells render. 'weighted' = RGB mix "
        "weighted by all per-method fractions (V1 default). "
        "'top2_mix' = pure consensus color if trust ≥ "
        "threshold, else 50/50 of the top two methods.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma (in bootstrap-grid cells) "
        "applied to the per-method bootstrap fractions before "
        "rendering. Smooths both the blend gradient and the "
        "95%% contour. Set 0 to disable. Default 2.0.",
    )
    parser.add_argument(
        "--boundary-min-logc",
        nargs="+",
        default=None,
        help="Per-panel min log10 C for drawing the winner "
        "boundary. One value per --warcs panel; use 'none' "
        "to disable for that panel. Example: "
        "--boundary-min-logc none none 22",
    )
    parser.add_argument(
        "--chinchilla-panels",
        nargs="*",
        type=int,
        default=None,
        help="1-based panel indices on which to draw the "
        "Chinchilla-optimal training line "
        "(C = 6·N·D with D=20·N → log10 C = log10(120) + "
        "2·log10 N). Example: --chinchilla-panels 3",
    )
    parser.add_argument(
        "--min-tokens",
        type=float,
        default=0.0,
        help="Minimum D = C/(6N) for predictions to be drawn. "
        "Default 0 (no floor). Set e.g. 1e6 or 1e9 to "
        "mask the under-trained corner with a dashed "
        "diagonal.",
    )
    parser.add_argument("--n-restarts", type=int, default=100, help="basinhopping restarts for the V3 fit.")
    parser.add_argument("--refit", action="store_true", help="Force a refit and overwrite the cached fit.")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete every .pkl/.json in scratch/plots/fit_cache/ "
        "before running. Useful when changing fit code outside "
        "_predict_optionB/fit_w1 (which auto-invalidates).",
    )
    parser.add_argument("--suffix", default="", help="Optional filename suffix.")
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Refresh warc-scaling summaries from gs:// and " "regenerate the streamlined CSV before plotting.",
    )
    parser.add_argument("--pull-only", action="store_true", help="Refresh data and exit without plotting.")
    args = parser.parse_args(argv)

    if args.clear_cache:
        n = clear_fit_cache()
        logger.info("Cleared %d files from %s", n, FIT_CACHE_DIR)

    if args.pull or args.pull_only:
        pull_warc_summaries(DEFAULT_AUDIT_DIR)
        regenerate_streamlined_csv()
        if args.pull_only:
            logger.info("--pull-only: exiting before plot.")
            return

    logger.info("Loading gold from %s", args.gold_csv)
    df_full = load_gold(args.gold_csv)
    logger.info("  %d rows across %s", len(df_full), METHODS_ORDER)

    # Option B: NO clip_to_min — bend-up rows kept and modeled by R_decay.
    df_fit = df_full
    logger.info("  Using all %d rows (no clip; R_decay handles bend-up)", len(df_fit))

    logger.info("Fitting W1 (shared E, R_D, C; per-method β, B, δ, α; no damage)…")
    res = fit_w1_cached(df_fit, n_restarts=args.n_restarts, refit=args.refit)
    fits = res["fits"]
    logger.info("  shared: E=%.4f  R_D=%.2f  C=%.3g", res["E"], res["R_D"], res["C_shared"])
    # Per-method R² + agg RMSE
    rss_total, n_total = 0.0, 0
    for m in METHODS_ORDER:
        sub = df_fit[df_fit["method"] == m]
        L = sub[METRIC].to_numpy(dtype=float)
        pred = fits[m]["pred"]
        ss = float(np.sum((L - pred) ** 2))
        tss = float(np.sum((L - L.mean()) ** 2))
        r2 = 1 - ss / tss
        rss_total += ss
        n_total += len(L)
        p = fits[m]["params"]
        logger.info(
            "  %-14s β=%.3f  B=%.3g  δ=%.3f  α=%.3f  R²=%.4f",
            m,
            p[2],
            p[3],
            p[4],
            p[5],
            r2,
        )
    logger.info("  aggregate RMSE = %.4f", (rss_total / n_total) ** 0.5)

    TPW = tokens_per_warc(df_full)
    logger.info("Tokens-per-WARC: %s", {m: f"{v:.3e}" for m, v in TPW.items()})

    # Resolve per-panel ranges: --log-n-ranges/--log-c-ranges override the
    # single-pair default; otherwise broadcast the default to every panel.
    n_panels = len(args.warcs)

    def _parse_per_panel(arg_list, default_pair, label):
        if not arg_list:
            return [tuple(default_pair)] * n_panels
        if len(arg_list) != n_panels:
            raise SystemExit(
                f"--{label} expects {n_panels} 'lo,hi' entries (one per " f"--warcs panel); got {len(arg_list)}."
            )
        out = []
        for s in arg_list:
            try:
                lo_str, hi_str = s.split(",")
                out.append((float(lo_str), float(hi_str)))
            except ValueError as e:
                raise SystemExit(f"--{label} entry {s!r} not in 'lo,hi' format") from e
        return out

    log_n_ranges = _parse_per_panel(args.log_n_ranges, args.log_n_range, "log-n-ranges")
    log_c_ranges = _parse_per_panel(args.log_c_ranges, args.log_c_range, "log-c-ranges")

    if args.boundary_min_logc is None:
        boundary_min_logc_per_panel = [None] * n_panels
    else:
        if len(args.boundary_min_logc) != n_panels:
            raise SystemExit(
                f"--boundary-min-logc expects {n_panels} entries (one per "
                f"--warcs panel); got {len(args.boundary_min_logc)}."
            )
        boundary_min_logc_per_panel = [None if s.lower() == "none" else float(s) for s in args.boundary_min_logc]

    trust_overlays = None
    if args.bootstrap_npz is not None:
        logger.info("Loading bootstrap overlay from %s", args.bootstrap_npz)
        npz = np.load(args.bootstrap_npz, allow_pickle=False)
        trust_overlays = {}
        n_methods = len(METHODS_ORDER)
        for w in args.warcs:
            key = f"winners_W{w}"
            if key not in npz.files:
                logger.warning("  no bootstrap winners for warcs=%d in npz", w)
                continue
            winners = npz[key]  # (B, n_C, n_N)
            fracs = np.stack(
                [(winners == i).mean(axis=0) for i in range(n_methods)],
                axis=0,
            )
            # Light Gaussian smoothing on the spatial axes of `fracs` removes
            # the discrete bootstrap noise (each cell's fraction takes values
            # k/100). Renormalize so each cell still sums to 1.
            if args.smooth_sigma > 0:
                from scipy.ndimage import gaussian_filter

                smoothed = np.stack(
                    [gaussian_filter(fracs[i], sigma=args.smooth_sigma, mode="nearest") for i in range(n_methods)],
                    axis=0,
                )
                # Renormalize across methods so each cell's fracs sum to 1.
                smoothed /= smoothed.sum(axis=0, keepdims=True).clip(min=1e-12)
                fracs = smoothed
            trust_fraction = np.max(fracs, axis=0)
            trust_overlays[w] = {
                "fracs": fracs,
                "trust_fraction": trust_fraction,
                "log_N_grid": npz[f"log_N_W{w}"],
                "log_C_grid": npz[f"log_C_W{w}"],
                "threshold": float(args.trust_threshold),
                "blend_mode": args.blend_mode,
            }
            uncertain_pct = (trust_fraction < args.trust_threshold).mean() * 100
            logger.info("  W=%d: uncertain (<%.0f%% trust) cells = %.1f%%", w, 100 * args.trust_threshold, uncertain_pct)

    png, pdf = render_figure(
        fits=fits,
        TPW=TPW,
        df_full=df_full,
        warcs_values=list(args.warcs),
        log_N_ranges=log_n_ranges,
        log_C_ranges=log_c_ranges,
        n_grid=args.n_grid,
        min_tokens=float(args.min_tokens),
        out_dir=args.output_dir,
        suffix=args.suffix,
        trust_overlays=trust_overlays,
        boundary_min_logc_per_panel=boundary_min_logc_per_panel,
        chinchilla_panels=set(args.chinchilla_panels or []),
    )
    logger.info("wrote %s", png)
    logger.info("wrote %s", pdf)


if __name__ == "__main__":
    main()
