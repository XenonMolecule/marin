# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot MMLU (``mmlu_sl_verb``, 5-shot) vs compute for the 10k curation isoflop sweep.

Replicates ``plot_core_v2_warc100.py``'s style (one panel per model scale, one
colored line per curation method, metric on y; two figures with x = FLOPs and
x = tokens), but for MMLU and — by default — for the SOFT metric rather than accuracy.

WHY NOT ACCURACY (the default is ``choice_logprob``, not ``acc``):
MMLU accuracy is pinned at the 0.25 random baseline across this whole sweep
(1e17..2e21 FLOPs, d512..d3584). Measured over the 30-cell top-of-grid: mean acc
0.2599, std 0.0094, with 4/30 cells scoring BELOW chance — i.e. scatter, not signal.
MMLU emergence sits around 1e22-1e23 FLOPs; this sweep's ceiling is 2e21. An acc
figure is a flat noise band, so it is available via ``--metric acc`` (useful as a
contrast) but is not the default.

READ THE Y-AXIS CAREFULLY. Verified against ``lm_eval/api/task.py`` (the exact source,
~L1676-1694) — every metric is computed from the 4 choice log-probs ``lls`` of the FULL
choice strings (``doc_to_choice`` renders "A. <text>", so byte lengths are of "A. 0",
not of the bare letter):

  * NOT normalized over the choice set — these are the LM-quality metrics:
      ``logprob`` = lls[gold]                                   (raw)
      ``bpb``     = -lls[gold] / bytes(choice[gold]) * log2(e)  (per-byte)
  * NORMALIZED over the 4 choices — these are the "which choice" metrics:
      ``choice_logprob``      = log_softmax(lls)[gold]        — raw lls, so length-SENSITIVE
      ``choice_prob_norm``    = softmax(-bpb_values)[gold]    — per-byte lls, length-CORRECTED
      ``choice_logprob_norm`` = log(choice_prob_norm)

So ``choice_logprob`` is NOT an unnormalized/fluency quantity: the log_softmax subtracts
logsumexp, cancelling most of the model's general fluency. It rises -4.27 -> -2.33 over
this sweep as the model's distribution over the 4 choices stops being arbitrarily peaked.

DO NOT READ ``choice_logprob`` AGAINST ln(0.25) AS A CHANCE LINE. It aggregates as
mean-of-LOG, so by Jensen a model scores below ln(0.25) whenever its per-question
probabilities have any dispersion — even at exactly chance. Every model in this sweep
sits under that line and that is EXPECTED, not underperformance. Measured on the best
cell (nemotron d3584 @2e21): choice_logprob = -1.5539, but its mean probability on the
correct choice is 0.2598 (log of the mean = -1.3479) — i.e. ABOVE 0.25. The Jensen
dispersion cost is 0.206 nats. The metric's TRUE null is model-specific: shuffling gold
while keeping that model's own dispersion gives -1.5802 +/- 0.0095, so the observed
-1.5539 is +2.8 sigma ABOVE its no-knowledge null. A single horizontal line cannot
express that null, which is why the drawn line is labelled "uniform predictor", not
"chance".

CAUTION — ``choice_prob_norm`` / ``choice_logprob_norm`` are LOW-SENSITIVITY on MMLU and
their flatness is PARTLY A METRIC ARTIFACT, not purely a fact about the models. They
softmax over per-BYTE scores, and MMLU choices average ~45 bytes, so a 10-nat advantage
on the correct choice moves the metric only ~0.25 -> ~0.29. Measured per-question on the
best cell: choice_prob_norm spans p1..p99 = 0.208..0.296 (std 0.015, 73% within
0.25+/-0.01), whereas softmax over RAW lls spans 0.004..0.626 (std 0.113). Do not read a
flat choice_prob_norm line as proof of "no knowledge" — the metric can barely express it.

THERE IS A SMALL BUT REAL SIGNAL at the top of the grid. On nemotron d3584 @2e21 (all
14042 questions), a permutation test that shuffles gold while PRESERVING both marginals
(so answer-position bias is held fixed) gives: observed acc 0.2875 vs null 0.2564+/-0.0034
= +9.0 sigma, p ~ 0. Decomposed against the +3.75 pts above uniform chance: +0.64 pts is
position bias x gold skew, and **+3.11 pts is genuine question-level discrimination**. So
these models do know a little MMLU — far below the ~0.54 a data-matched 3B reaches, but
not zero and not noise. Per-cell binomial se is ~0.0037, so top-cell method differences of
~3 pts (e.g. nemotron 0.2875 vs fineweb_edu 0.2550) are ~6 sigma and NOT dismissable.

WHAT THE MODELS ACTUALLY DO (same cell): this is loglikelihood rank-classification — the
model NEVER generates. lm-eval scores the 4 candidate continuations and argmaxes, so a
malformed or refused answer is impossible by construction. The model picks **D 46.5% /
A 11.8%** against a balanced gold distribution (~25% each): a large answer-POSITION
(recency) bias. It is NOT length-biased — picks the shortest choice 24.9% of the time
(unbiased = 25%). Any per-method ranking on raw ``acc`` is therefore contaminated by each
model's position bias; compare the debiased ``obs - E[acc | picks ⟂ gold]`` instead.

DATA SOURCES (and why this differs from the Core_v2 plotters):
  * MMLU: the CONSOLIDATED CSV from ``consolidate_mmlu_results.py``, not the raw
    per-run results.json. The Core_v2 summaries are a few KB each, so those plotters
    cat them directly; MMLU's results.json are ~8MB each (lm-eval duplicates its
    logged ``outputs`` into all 58 rows), so catting 181 of them to a laptop would be
    ~1.4GB of cross-region egress. The consolidator already does that scan IN-CLOUD
    across all 5 regions and emits a ~43KB CSV; this reads that.
  * tokens + params: ``us-central1/metadata/data_curation_10k_natural_results/<stem>.json``
    -> ``tokens.tokens_trained`` and ``model.total_trainable_params``, joined by run
    stem — the same join the Core_v2 plotters use, so panel labels and the x=tokens
    axis are the real trained values rather than a C=6ND estimate.

Colors reuse ``plot_core_v2_by_size_10k.METHOD_COLORS`` so a curation method keeps the
SAME color here as in every Core_v2 figure.

Usage::
    .venv/bin/python -m experiments.scaling_law_sweeps.plot_mmlu_by_size
    .venv/bin/python -m experiments.scaling_law_sweeps.plot_mmlu_by_size --metric acc
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from experiments.scaling_law_sweeps.plot_core_v2_by_size_10k import METHOD_COLORS

logger = logging.getLogger(__name__)

# A cell = (method, hidden_dim, budget). A few cells hold more than one run, and a line
# plot cannot show two y-values at one x — hence this dedup. Follows the project rule in
# `dclm_core.launch_dclm_core_sweep.drop_rerun_variants`:
#   * TRUE re-run variants (-seed*, -pcache*, -retry*, -v2/-v3) are re-runs of a plain run
#     and are dropped WHEN a plain run exists at that cell; if the cell has only variants,
#     one is kept deterministically.
#   * Distinct BATCH sizes are NOT re-runs — they are different training configs. Only the
#     d512/2e19 cell has such a pair (B64+B128, same 25.8B tokens / 3.52 epochs, identical
#     but for batch). Both are legitimate and both stay in the eval set; for the LINE we
#     must still pick one, so we take the LARGER batch because the fixed-model planner's
#     canonical choice at that cell is B256 and B128 is the nearer of the two. Every drop
#     is logged. Pass --keep-dupes to plot every run (the line will zig-zag vertically).
_RERUN_RE = re.compile(r"-(seed\d+|pcache|retry|v2|v3)\b")
_BATCH_RE = re.compile(r"-B(\d+)")

DEFAULT_CSV = "gs://marin-us-east5/metadata/mmlu_sl_verb_5shot_partial.csv"
TOKENS_PREFIX = "gs://marin-us-central1/metadata/data_curation_10k_natural_results"
OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "mmlu"

# Chance level for 4-way multiple choice; drawn as a reference line on the metrics
# where "chance" is meaningful (acc / choice_prob_norm), and on those only.
CHANCE_ACC = 0.25
CHANCE_LOGPROB_NORM = math.log(0.25)


@dataclass(frozen=True)
class MetricSpec:
    """How to render one MMLU metric: axis label, reference line, and framing."""

    label: str
    chance: float | None
    """Value of the drawn reference line, or None for no line.

    NOT always a null! See `ref_label`. For the mean-of-prob metrics (acc, acc_norm,
    choice_prob_norm) this is uniform random guessing and models genuinely straddle it.
    For the mean-of-LOG metrics (choice_logprob, choice_logprob_norm) it is the value a
    PERFECTLY UNIFORM predictor would score, and by Jensen ANY model with per-question
    dispersion scores below it even at exact chance — so it must not be labelled "chance".
    """

    note: str
    ref_label: str | None = None
    """Legend text for the reference line. Required when `chance` is set."""
    noise_sd: float | None = None
    """Per-cell sampling SD of this metric, measured by permutation at n=14042.

    Drives a shaded +/-2 sigma band around `chance`, which is how this plotter avoids the
    truncated-axis lie WITHOUT hiding real effects. The earlier approach — forcing a wide
    minimum y-span — over-corrected: it made a genuine +3.1pt / +9 sigma signal invisible.
    A band lets the axis fit the data (so small real effects stay legible) while marking
    exactly which excursions are indistinguishable from noise: inside the band = noise,
    outside = real. Only meaningful with `chance`.
    """


METRICS: dict[str, MetricSpec] = {
    "choice_logprob": MetricSpec(
        label="MMLU choice_logprob (5-shot)",
        # log_softmax over the 4 choices, so uniform IS the reference level. Aggregated
        # as mean-of-log, so by Jensen it sits below ln(0.25) even at exact chance --
        # the line marks "uniform", not a null the data should hug.
        chance=CHANCE_LOGPROB_NORM,
        ref_label="uniform predictor (ln .25) — NOT a null; zero-knowledge sits BELOW",
        note="log-softmax over the 4 choices; mean-of-log — dashed line is uniform, NOT a chance null",
        noise_sd=0.00925,
    ),
    "choice_prob_norm": MetricSpec(
        label="MMLU choice_prob_norm (5-shot)",
        chance=CHANCE_ACC,
        ref_label="uniform random guessing (0.25)",
        note="softmax over per-byte lls — low sensitivity on MMLU's ~45-byte choices (p1..p99 = .21-.30)",
        # +/-0.05 around chance: wide enough that a real effect would be visible, so the
        # observed ~0.003 spread correctly reads as a flat line rather than a ranking.
        noise_sd=0.000121,
    ),
    "choice_logprob_norm": MetricSpec(
        label="MMLU choice_logprob_norm (5-shot)",
        chance=CHANCE_LOGPROB_NORM,
        ref_label="uniform predictor (ln .25) — NOT a null (mean-of-log)",
        note="log of choice_prob_norm — inherits its per-byte compression; dashed = uniform, not a null",
        noise_sd=0.000506,
    ),
    "acc": MetricSpec(
        label="MMLU accuracy (5-shot)",
        chance=CHANCE_ACC,
        ref_label="uniform random guessing (0.25)",
        note="best cell is +3.1 pts over its own position-bias null (+9 sigma); each model's true null ~0.256",
        noise_sd=0.00345,
    ),
    "acc_norm": MetricSpec(
        label="MMLU acc_norm (5-shot)",
        chance=CHANCE_ACC,
        ref_label="uniform random guessing (0.25)",
        note="near the 0.25 random baseline; each model's true null is its own",
        noise_sd=0.00345,
    ),
    "bpb": MetricSpec(
        label="MMLU bits-per-byte (5-shot, lower=better)",
        chance=None,
        note="LM quality on the correct choice string (not normalized over choices)",
    ),
    # --- knowledge-only metrics, from analyze_mmlu_signal.py's CSV (--csv .../..._signal.csv).
    # These subtract each model's OWN zero-knowledge null, so general language modeling,
    # calibration and answer-position bias cancel out and 0.0 is a TRUE null: the value a
    # model with no MMLU knowledge scores. This is the honest "did curation teach it
    # anything" view; the raw choice_logprob curve is ~92% language modeling.
    "clp_gap": MetricSpec(
        label="MMLU knowledge (nats vs null)",
        chance=0.0,
        ref_label="zero knowledge (a TRUE null)",
        note="choice_logprob minus this model's zero-knowledge null — LM/calibration/position bias cancel",
        noise_sd=0.00925,
    ),
    "cpn_gap": MetricSpec(
        label="MMLU knowledge (cpn vs null)",
        chance=0.0,
        ref_label="zero knowledge (a TRUE null)",
        # THE recommended knowledge view. choice_prob_norm is the best fluency control in
        # the suite: its zero-knowledge null drifts just -0.0001 across 1e17..2e21, vs
        # +2.2306 nats for choice_logprob. Being a mean-of-PROB over a normalized
        # distribution pins its null at ~0.25 for any model, and the per-byte step also
        # removes the length bias that makes small models look anti-correlated under
        # choice_logprob. Compressed, yes -- but its sampling SD is only 0.00012, so the
        # signal/noise survives: every budget is +4..+6 sigma.
        note="choice_prob_norm minus own null — best fluency control (null drifts only -0.0001 across the sweep)",
        noise_sd=0.000121,
    ),
    "acc_gap": MetricSpec(
        label="MMLU knowledge (acc vs null)",
        chance=0.0,
        ref_label="zero knowledge (a TRUE null)",
        note="accuracy minus each model's own position-bias null — the debiased knowledge signal",
        noise_sd=0.00345,
    ),
    "logprob": MetricSpec(
        label="MMLU logprob of correct choice (5-shot)",
        chance=None,  # raw lls[gold]: not a probability over the choice set
        note="raw log-prob of the correct choice string (not normalized over choices)",
    ),
}


@dataclass(frozen=True)
class Record:
    """One run: scale, method, compute, and the plotted metric."""

    run_stem: str
    method: str
    hidden_dim: int
    budget: float
    tokens: float
    params: int
    value: float


def _sh(args: list[str]) -> str:
    """Run a command, returning stdout ("" on failure).

    Uses the gcloud CLI rather than a Python GCS client on purpose: the laptop's
    gcsfs/aiohttp path hits CERTIFICATE_VERIFY_FAILED against storage.googleapis.com
    while the CLI works. ``plot_core_v2_warc100`` does the same.
    """
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        return ""
    return r.stdout


def _read_csv(path: str) -> list[dict]:
    text = Path(path).read_text() if not path.startswith("gs://") else _sh(["gcloud", "storage", "cat", path])
    if not text.strip():
        raise SystemExit(f"could not read metric CSV: {path}")
    return list(csv.DictReader(io.StringIO(text)))


def _load(csv_path: str, metric: str) -> list[Record]:
    """Read the consolidated MMLU CSV and join real tokens/params by run stem."""
    rows = _read_csv(csv_path)
    logger.info("read %d rows from %s", len(rows), csv_path)

    def _one(row: dict) -> Record | None:
        stem = row["run_stem"]
        raw = row.get(metric, "")
        if not raw:
            logger.warning("skip %s: no %s", stem, metric)
            return None
        tj = _sh(["gcloud", "storage", "cat", f"{TOKENS_PREFIX}/{stem}.json"])
        try:
            nat = json.loads(tj) if tj.strip() else None
        except json.JSONDecodeError:
            nat = None
        tokens = (nat or {}).get("tokens", {}).get("tokens_trained")
        params = (nat or {}).get("model", {}).get("total_trainable_params")
        if tokens is None or params is None:
            logger.warning("skip %s: tokens=%s params=%s", stem, tokens, params)
            return None
        return Record(
            run_stem=stem,
            method=row["method"],
            hidden_dim=int(row["hidden_dim"]),
            budget=float(row["budget"].replace("e+", "e")),
            tokens=float(tokens),
            params=int(params),
            value=float(raw),
        )

    recs: list[Record] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(_one, r) for r in rows]
        for f in as_completed(futs):
            if (rec := f.result()) is not None:
                recs.append(rec)
    logger.info("joined %d records", len(recs))
    return recs


def _dedup_cells(recs: list[Record]) -> list[Record]:
    """One run per (method, hidden_dim, budget) — see the module-level rule."""
    by_cell: dict[tuple[str, int, float], list[Record]] = {}
    for r in recs:
        by_cell.setdefault((r.method, r.hidden_dim, r.budget), []).append(r)
    kept: list[Record] = []
    for (method, dim, budget), group in by_cell.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        plain = [r for r in group if not _RERUN_RE.search(r.run_stem)] or group

        def _batch(r: Record) -> int:
            m = _BATCH_RE.search(r.run_stem)
            return int(m.group(1)) if m else -1

        winner = max(plain, key=lambda r: (_batch(r), r.run_stem))
        kept.append(winner)
        for r in group:
            if r is not winner:
                logger.info(
                    "dedup %s d%d @%g: keeping %s, dropping %s", method, dim, budget, winner.run_stem, r.run_stem
                )
    return kept


def _plabel(params: int) -> str:
    return f"{params / 1e9:.2f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"


def _figure(recs: list[Record], metric: str, xkey: str, xlabel: str, out: Path) -> None:
    spec = METRICS[metric]
    dims = sorted({r.hidden_dim for r in recs})
    params_for = {d: max(r.params for r in recs if r.hidden_dim == d) for d in dims}
    methods = sorted({r.method for r in recs})

    y_all = [r.value for r in recs]
    ylo, yhi = min(y_all), max(y_all)
    if spec.chance is not None:
        # Keep the chance line in frame even when every point sits on top of it.
        ylo, yhi = min(ylo, spec.chance), max(yhi, spec.chance)
    if spec.chance is not None and spec.noise_sd is not None:
        # Keep the noise band visible, but never inflate beyond it: the axis fits the
        # data, so a real effect stays legible instead of being squashed to a hairline.
        band = 2 * spec.noise_sd
        ylo, yhi = min(ylo, spec.chance - band), max(yhi, spec.chance + band)
    ypad = 0.06 * (yhi - ylo) if yhi > ylo else 0.01

    ncol = min(len(dims), 3)
    nrow = math.ceil(len(dims) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow), squeeze=False)
    for idx, dim in enumerate(dims):
        ax = axes[idx // ncol][idx % ncol]
        if spec.chance is not None:
            if spec.noise_sd is not None:
                ax.axhspan(
                    spec.chance - 2 * spec.noise_sd,
                    spec.chance + 2 * spec.noise_sd,
                    color="#bbb",
                    alpha=0.35,
                    lw=0,
                    zorder=0,
                )
            ax.axhline(spec.chance, color="#999", lw=1.0, ls="--", zorder=1)
        for method in methods:
            pts = sorted((r for r in recs if r.hidden_dim == dim and r.method == method), key=lambda r: getattr(r, xkey))
            if not pts:
                continue
            ax.plot(
                [getattr(p, xkey) for p in pts],
                [p.value for p in pts],
                marker="o",
                ms=4.5,
                lw=2,
                color=METHOD_COLORS.get(method, "#333333"),
                alpha=0.95,
                zorder=2,
            )
        ax.set_xscale("log")
        ax.set_ylim(ylo - ypad, yhi + ypad)
        ax.set_title(f"d{dim} ({_plabel(params_for[dim])})", fontsize=11)
        ax.grid(True, which="major", alpha=0.25)
        if idx % ncol == 0:
            ax.set_ylabel(spec.label)
        if idx // ncol == nrow - 1 or idx >= len(dims) - ncol:
            ax.set_xlabel(xlabel)
    for j in range(len(dims), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    # Coverage is UNEVEN: the broad 245-run sweep was stopped part-way, so some
    # methods (resiliparse especially) have only their top-of-grid cell per width and
    # render as isolated markers rather than a curve. Put each method's run count in
    # the legend so a sparse series reads as "not measured yet", never as "collapsed".
    n_by_method = {m: sum(1 for r in recs if r.method == m) for m in methods}
    handles = [
        Line2D([0], [0], color=METHOD_COLORS.get(m, "#333333"), lw=2.5, marker="o", label=f"{m} (n={n_by_method[m]})")
        for m in methods
    ]
    if spec.chance is not None:
        handles.append(Line2D([0], [0], color="#999", lw=1.0, ls="--", label=spec.ref_label or "reference"))
        if spec.noise_sd is not None:
            handles.append(Patch(facecolor="#bbb", alpha=0.35, label="+/-2 sigma sampling noise"))
    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9
    )
    sparse = [m for m in methods if n_by_method[m] <= len(dims)]
    coverage = f"; sparse (top-of-grid only): {', '.join(sparse)}" if sparse else ""
    fig.suptitle(f"MMLU 5-shot ({metric}) vs {xlabel} by model scale — 10k curation isoflop", fontsize=13, y=1.005)
    fig.text(
        0.5,
        0.972,
        f"{spec.note}\n{len(recs)} runs; uneven coverage{coverage}",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("wrote %s", out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric", choices=sorted(METRICS), default="choice_logprob", help="Metric on the y-axis.")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Consolidated MMLU CSV (gs:// or local).")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Directory for the PNGs.")
    ap.add_argument(
        "--keep-dupes",
        action="store_true",
        help="Plot every run instead of one per cell; the line zig-zags where a cell has two runs.",
    )
    args = ap.parse_args(argv)

    recs = _load(args.csv, args.metric)
    if not recs:
        raise SystemExit("no records")
    if not args.keep_dupes:
        before = len(recs)
        recs = _dedup_cells(recs)
        if before != len(recs):
            logger.info("deduped %d -> %d runs (one per method x width x budget)", before, len(recs))
    _figure(recs, args.metric, "budget", "training FLOPs", args.out_dir / f"mmlu_{args.metric}_by_size_x_flops.png")
    _figure(recs, args.metric, "tokens", "tokens trained", args.out_dir / f"mmlu_{args.metric}_by_size_x_tokens.png")

    # Console readout: per-scale best-method ranking, so the trend is legible
    # without opening the PNGs.
    for dim in sorted({r.hidden_dim for r in recs}):
        best: dict[str, float] = {}
        for r in (r for r in recs if r.hidden_dim == dim):
            best[r.method] = max(best.get(r.method, -math.inf), r.value)
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        logger.info("d%d best-%s: %s", dim, args.metric, ", ".join(f"{m}={v:.4f}" for m, v in ranked))


if __name__ == "__main__":
    main()
