# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Option B bootstrap fit library for Zephyr fan-out.

VENDORED on purpose: the production fit code lives under `scratch/` which is
gitignored and therefore does NOT ship in an Iris/Zephyr workspace. This module
copies the minimal pieces inline so it is importable on remote workers with only
numpy / pandas / scipy (core marin deps).

Option B scaling law (per method, 8 params: E, C, β, B, δ, α, R_D, R_decay):

    L(N, D, U) = E + C·N^(-β) + B·N^δ · D_eff^(-α)
    D_eff      = U·R_D·(1 − exp(−ε/R_D)) · exp(−ε/R_decay)      ε = D/U

Shared across methods: E, C, β, R_D, R_decay. Per-method: B, δ, α.
NO clip_to_min — bend-up rows are kept and modeled by the exp(−ε/R_decay) factor.

The gold CSV is read from GCS (also gitignored locally); see GOLD_GS.
`fit_one_seed(seed)` is the top-level map function for Zephyr.
"""

from __future__ import annotations

import base64
import functools
import io
import subprocess
import tempfile

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping

# ─── config ────────────────────────────────────────────────────────────────
GOLD_GS = "gs://marin-us-central1/scratch/bootstrap_optionB/gold_298rows.csv"
METRIC = "eval_uncheatable_macro_loss"
KEEP_METHODS = ("low_quality", "med_quality", "high_quality")

N_GRID = 200
WARCS_PANELS = (100, 500, 7_925_398)
LOG_N_RANGES = {
    100: (7.8, 10.0),
    500: (7.8, 10.699),
    7_925_398: (7.8, 13.0),
}
LOG_C_RANGES = {
    100: (17.0, 23.0),
    500: (17.0, 24.0),
    7_925_398: (17.0, 50.0),
}
N_RESTARTS_DEFAULT = 100


# ─── vendored loss + basinhopping step (from fit_muennighoff_v2.py) ──────────
def _loss_huber_raw(residuals: np.ndarray) -> float:
    """scipy-equivalent Huber loss with δ=1.0, summed."""
    delta = 1.0
    absr = np.abs(residuals)
    quad = np.minimum(absr, delta)
    lin = absr - quad
    return float(np.sum(0.5 * quad**2 + delta * lin))


class _MagnitudeStep:
    """Basin-hopping step scaled by parameter magnitude (params span 0–1e10)."""

    def __init__(self, lo, hi, stepsize=0.5, rng=None):
        self.lo = lo
        self.hi = hi
        self.stepsize = stepsize
        self.rng = rng or np.random.default_rng(0)

    def __call__(self, x):
        scale = np.maximum(np.abs(x), 0.5) * self.stepsize
        new_x = x + self.rng.uniform(-1, 1, size=x.shape) * scale
        return np.clip(new_x, self.lo, self.hi)


# ─── Option B predict + fit ──────────────────────────────────────────────────
def predict_optionB(params_per_m, N, D, U):
    """params_per_m = (E, C, β, B, δ, α, R_D, R_decay)."""
    E, C, beta, B, delta, alpha, R_D, R_decay = params_per_m
    eps = D / U
    D_eff_sat = U * R_D * (1.0 - np.exp(-eps / R_D))
    decay = np.exp(-eps / R_decay)
    D_eff = np.maximum(D_eff_sat * decay, 1e-30)
    return E + C * N ** (-beta) + B * N**delta * D_eff ** (-alpha)


def fit_optionB(pmd, n_restarts=N_RESTARTS_DEFAULT):
    """Shared (E, C, β, R_D, R_decay); per-method (B, δ, α). NO clip.

    pmd: {method -> (N, D, U, L)} numpy arrays.
    Returns {method -> (E, C, β, B, δ, α, R_D, R_decay)} tuple.
    """
    methods = list(pmd)
    lo_shared = np.array([0.0, 0.0, 0.05, 0.1, 1.0])  # E, C, β, R_D, R_decay
    hi_shared = np.array([6.0, 1e10, 1.5, 200.0, 5000.0])
    x0_shared = np.array([2.0, 1e3, 0.40, 3.0, 100.0])
    lo_per = np.array([0.0, -0.5, 0.05])  # B, δ, α
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
            total += _loss_huber_raw(predict_optionB(params, N, D, U) - L)
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
    return unpack(np.clip(res.x, lo, hi))


# ─── gold data loading (GCS, cached per worker process) ──────────────────────
def _read_gold_bytes() -> bytes:
    """Fetch the gold CSV bytes from GOLD_GS. Try fsspec, fall back to gcloud."""
    try:
        import fsspec

        with fsspec.open(GOLD_GS, "rb") as f:
            return f.read()
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(["gcloud", "storage", "cp", GOLD_GS, tmp_path], check=True)
        with open(tmp_path, "rb") as f:
            return f.read()


@functools.lru_cache(maxsize=1)
def load_gold() -> pd.DataFrame:
    """Load + clean the 3-tier gold frame once per worker process."""
    df = pd.read_csv(io.BytesIO(_read_gold_bytes()))
    for col in ("flops_target", "flops_actual", "tokens", "parameters", "epochs", "unique_tokens", METRIC):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[METRIC, "tokens", "parameters", "epochs", "unique_tokens"])
    df = df[(df.epochs > 0) & (df.tokens > 0) & (df.parameters > 0)]
    df = df[df["method"].isin(KEEP_METHODS)].copy().reset_index(drop=True)
    df["U_eff"] = df["tokens"] / df["epochs"]
    return df


def tokens_per_warc(df: pd.DataFrame) -> dict[str, float]:
    return {m: float((g["unique_tokens"].astype(float) / g["warcs"]).mean()) for m, g in df.groupby("method")}


def pmd_from(df: pd.DataFrame) -> dict:
    return {
        m: (
            df[df["method"] == m]["parameters"].to_numpy(float),
            df[df["method"] == m]["tokens"].to_numpy(float),
            df[df["method"] == m]["U_eff"].to_numpy(float),
            df[df["method"] == m][METRIC].to_numpy(float),
        )
        for m in KEEP_METHODS
    }


def winners_on_grid(per_params, tpw, warcs, log_N_grid, log_C_grid) -> np.ndarray:
    """(n_C, n_N) int8 winner index per (N, C) cell for one warcs panel."""
    N = (10.0**log_N_grid)[None, :]
    C = (10.0**log_C_grid)[:, None]
    D = C / (6 * N)
    losses = []
    for m in KEEP_METHODS:
        U = tpw[m] * warcs
        L = predict_optionB(per_params[m], np.broadcast_to(N, D.shape), D, np.full_like(D, U))
        losses.append(L)
    return np.argmin(np.stack(losses, axis=0), axis=0).astype(np.int8)


def _encode_grid(arr: np.ndarray) -> str:
    """int8 grid → base64 string (compact, JSON-safe)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.int8).tobytes()).decode("ascii")


def decode_grid(b64: str, n_grid: int = N_GRID) -> np.ndarray:
    """base64 string → (n_grid, n_grid) int8 grid."""
    raw = base64.b64decode(b64.encode("ascii"))
    return np.frombuffer(raw, dtype=np.int8).reshape(n_grid, n_grid)


# ─── top-level map function for Zephyr ───────────────────────────────────────
def fit_one_seed(seed: int, n_restarts: int = N_RESTARTS_DEFAULT) -> dict:
    """Bootstrap-resample the gold data with `seed`, fit Option B, return
    base64 winner grids per warcs panel. Top-level + cloudpickle-friendly."""
    try:
        raw = load_gold()
        rng = np.random.default_rng(seed)
        boot = raw.sample(n=len(raw), replace=True, random_state=int(rng.integers(2**31)))
        tpw = tokens_per_warc(boot)
        pmd = pmd_from(boot)
        if any(len(pmd[m][3]) < 3 for m in KEEP_METHODS):
            return {"seed": int(seed), "ok": False, "reason": "missing_method"}
        per_params = fit_optionB(pmd, n_restarts=n_restarts)
        out = {"seed": int(seed), "ok": True}
        for warcs in WARCS_PANELS:
            log_N = np.linspace(*LOG_N_RANGES[warcs], N_GRID)
            log_C = np.linspace(*LOG_C_RANGES[warcs], N_GRID)
            out[f"W{warcs}"] = _encode_grid(winners_on_grid(per_params, tpw, warcs, log_N, log_C))
        return out
    except Exception as e:  # never let one bad seed kill the shard
        return {"seed": int(seed), "ok": False, "reason": repr(e)[:200]}


def fit_central(n_restarts: int = N_RESTARTS_DEFAULT) -> dict:
    """Central Option B fit on the ORIGINAL (unresampled) gold. Returns the
    per-method param tuples + tpw so the figure can render the colored regions
    without refitting."""
    raw = load_gold()
    tpw = tokens_per_warc(raw)
    per_params = fit_optionB(pmd_from(raw), n_restarts=n_restarts)
    return {
        "params": {m: list(per_params[m]) for m in KEEP_METHODS},
        "tpw": tpw,
        "n_grid": N_GRID,
        "warcs_panels": list(WARCS_PANELS),
        "log_n_ranges": {str(w): list(LOG_N_RANGES[w]) for w in WARCS_PANELS},
        "log_c_ranges": {str(w): list(LOG_C_RANGES[w]) for w in WARCS_PANELS},
    }
