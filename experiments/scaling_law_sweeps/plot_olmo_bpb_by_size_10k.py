# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot OLMo Base-Easy bpb vs training tokens, gridded by model scale.

Sibling of ``plot_core_v2_by_size_10k`` — identical layout and palette, but the
y-axis is OLMo Base-Easy bits-per-byte instead of DCLM Core_v2, so the two
figures can be read side by side for the same 10k-WARC curation isoflop sweep.
**LOWER bpb is better**, the opposite of Core_v2, so the per-scale ranking logs
and the "best" annotations are inverted relative to that module.

One figure is emitted per metric in ``CATEGORIES`` (macro plus the Code / Math /
QA groupings from ``olmo_bpb_tasks_set``), since a corpus can win on prose while
losing badly on code — the macro alone hides exactly the effect these sweeps are
run to measure.

Data sources:

  * bpb per run: ``<region>/metadata/olmo_bpb_results/<run_stem>/results.json``
    (``tasks/<task>/<variant>/bpb`` + ``averages/macro_bpb``), written by
    ``run_olmo_bpb_eval``. Results land in the bucket of whichever region the
    eval ran in, so ALL regions are scanned and merged — reading only the
    canonical us-central1 prefix silently drops most of a method's points.
  * tokens per run (x-axis): ``data_curation_10k_natural_results/<stem>.json``
    in us-central1, joined by run stem, exactly as the Core_v2 plotter does.

Usage::

    export SSL_CERT_FILE=$(.venv/bin/python -m certifi)
    .venv/bin/python -m experiments.scaling_law_sweeps.plot_olmo_bpb_by_size_10k
    .venv/bin/python -m experiments.scaling_law_sweeps.plot_olmo_bpb_by_size_10k --include lpv11
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from experiments.data_mixing.olmix_tasks import build_olmix_exact_tasks
from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import (
    CODE_BPB,
    MATH_BPB,
    MT_MBPP_BPB,
    QA_LANG_BPB,
    QA_RC_MC_BPB,
)
from experiments.scaling_law_sweeps.plot_core_v2_by_size_10k import (
    _STEM_RE,
    FASTPIPE_COLORS,
    LPV11_COMPARE_METHODS,
    OLMIXEXACT_COMPARE_METHODS,
    METHOD_COLORS,
    _FASTPIPE_RE,
    _is_fastpipe,
    _method_label,
    _method_sort_key,
    _params_label,
)

logger = logging.getLogger(__name__)

# bpb results are written region-locally; these are the buckets curation evals
# have ever run in. Listing a bucket that holds none is harmless.
BPB_REGIONS = ["us-central1", "us-east5", "us-central2", "us-east1", "us-west4"]
BPB_SUBPATH = "metadata/olmo_bpb_results"
DEFAULT_TOKENS_PREFIX = "gs://marin-us-central1/metadata/data_curation_10k_natural_results/"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "olmo_bpb_10k"

# metric -> task keys ("<task>/<variant>") averaged for it; None = read the
# precomputed macro straight out of ``averages``.
CATEGORIES: dict[str, list[str] | None] = {
    "macro_bpb": None,
    "Code_bpb": list(CODE_BPB) + list(MT_MBPP_BPB),
    "Math_bpb": list(MATH_BPB),
    "QA_bpb": list(QA_LANG_BPB) + list(QA_RC_MC_BPB),
    # The olmix paper's exact 51-task suite (Table 9) — the objective the
    # *_olmixexact_* mixtures were solved against, so it is the metric on which
    # those arms are compared. NOT held out: contains the 10 Core v2 tasks.
    "olmix51_bpb": list(build_olmix_exact_tasks(include_mmlu=True)),
    # The 51 minus the 4 MMLU category tasks. The natural-baseline sweeps were
    # evaluated before MMLU joined the suite, so this is the largest subset on
    # which mix-vs-natural is comparable without re-running evals.
    "olmix47_bpb": list(build_olmix_exact_tasks(include_mmlu=False)),
}


@dataclass(frozen=True)
class BpbRecord:
    """One trained run: its scale, method, tokens, and per-category bpb."""

    run_stem: str
    method: str
    hidden_dim: int
    budget: str
    tokens: float
    total_params: int
    bpb: dict[str, float]


def _open_gcs(path: str):
    from rigging.filesystem import filesystem

    return filesystem("gcs").open(path if path.startswith("gs://") else "gs://" + path)


def _category_bpb(results: dict) -> dict[str, float]:
    """Average bpb per category. A category with no scored task is omitted.

    Missing tasks are skipped rather than treated as zero: a partially-evaluated
    run must not manufacture an artificially low (better-looking) bpb.
    """
    tasks = results.get("tasks") or {}
    out: dict[str, float] = {}
    macro = (results.get("averages") or {}).get("macro_bpb")
    if macro is not None:
        out["macro_bpb"] = float(macro)
    for name, keys in CATEGORIES.items():
        if keys is None:
            continue
        vals = [tasks[k]["bpb"] for k in keys if k in tasks and tasks[k].get("bpb") is not None]
        if vals:
            out[name] = sum(vals) / len(vals)
    return out


def _load_records(include_methods: set[str] | None) -> list[BpbRecord]:
    """Scan every region's bpb results, join tokens by run stem, return records."""
    from rigging.filesystem import filesystem

    fs = filesystem("gcs")
    result_paths: list[str] = []
    for region in BPB_REGIONS:
        pattern = f"marin-{region}/{BPB_SUBPATH}/*/results.json"
        try:
            found = fs.glob(pattern)
        except FileNotFoundError:
            found = []
        logger.info("Region %s: %d bpb results", region, len(found))
        result_paths.extend(found)

    def _load_one(path: str) -> BpbRecord | None:
        stem = path.rstrip("/").rsplit("/", 2)[-2]
        m = _STEM_RE.match(stem)
        if m is None:
            return None  # random-ladder / other sweeps share this prefix
        method = m.group("method")
        if include_methods is not None and method not in include_methods:
            return None
        with _open_gcs(path) as fh:
            bpb = _category_bpb(json.load(fh))
        if not bpb:
            logger.warning("No usable bpb in %s", stem)
            return None
        try:
            with _open_gcs(DEFAULT_TOKENS_PREFIX + stem + ".json") as fh:
                natural = json.load(fh)
        except FileNotFoundError:
            logger.warning("No tokens summary joined for %s", stem)
            return None
        tokens = (natural.get("tokens") or {}).get("tokens_trained")
        total_params = (natural.get("model") or {}).get("total_trainable_params")
        if tokens is None or total_params is None:
            logger.warning("Incomplete tokens/params for %s", stem)
            return None
        return BpbRecord(
            run_stem=stem,
            method=method,
            hidden_dim=int(m.group("hidden")),
            budget=m.group("budget"),
            tokens=float(tokens),
            total_params=int(total_params),
            bpb=bpb,
        )

    records: dict[str, BpbRecord] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_load_one, p) for p in result_paths]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is not None:
                records.setdefault(rec.run_stem, rec)  # same run mirrored in 2 regions
    logger.info("Loaded %d joined records", len(records))
    return list(records.values())


def _build_figure(records: list[BpbRecord], metric: str, title_main: str):
    """Grid of subplots (one per hidden_dim) of `metric` bpb vs tokens, per method."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    usable = [r for r in records if metric in r.bpb]
    dims = sorted({r.hidden_dim for r in usable})
    params_for_dim = {r.hidden_dim: r.total_params for r in usable}

    n_panels = len(dims)
    n_cols = 3
    n_rows = math.ceil(n_panels / n_cols)

    def _row_col(idx: int) -> tuple[int, int]:
        return idx // n_cols + 1, idx % n_cols + 1

    bottom_panel: dict[int, int] = {}
    for idx in range(n_panels):
        _, col = _row_col(idx)
        bottom_panel[col] = idx

    y_vals = [r.bpb[metric] for r in usable]
    y_lo, y_hi = min(y_vals), max(y_vals)
    y_pad = 0.04 * (y_hi - y_lo) if y_hi > y_lo else 0.01
    y_range = [y_lo - y_pad, y_hi + y_pad]

    titles = [_params_label(d, params_for_dim[d]) for d in dims]
    titles += [""] * (n_rows * n_cols - n_panels)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    seen_methods: set[str] = set()
    for idx, dim in enumerate(dims):
        row, col = _row_col(idx)
        by_method: dict[str, list[BpbRecord]] = {}
        for r in usable:
            if r.hidden_dim == dim:
                by_method.setdefault(r.method, []).append(r)
        for method in sorted(by_method, key=_method_sort_key):
            pts = sorted(by_method[method], key=lambda r: r.tokens)
            is_fp = _is_fastpipe(method)
            if is_fp:
                color = FASTPIPE_COLORS.get(int(_FASTPIPE_RE.match(method).group("thr")), "#333333")
            else:
                color = METHOD_COLORS.get(method, "#333333")
            show_legend = method not in seen_methods
            seen_methods.add(method)
            fig.add_trace(
                go.Scatter(
                    x=[p.tokens for p in pts],
                    y=[p.bpb[metric] for p in pts],
                    mode="lines+markers",
                    marker=dict(size=7, color=color, symbol="diamond" if is_fp else "circle"),
                    line=dict(color=color, width=2, dash="dash" if is_fp else "solid"),
                    name=_method_label(method),
                    legendgroup="fastpipe_v3" if is_fp else method,
                    showlegend=show_legend,
                    hovertemplate=("%{fullData.name}<br>tokens=%{x:.3s}<br>" + metric + "=%{y:.4f}<extra></extra>"),
                ),
                row=row,
                col=col,
            )
        fig.update_yaxes(range=y_range, row=row, col=col, title_text=metric if col == 1 else None)
        fig.update_xaxes(
            type="log",
            row=row,
            col=col,
            title_text="training tokens" if bottom_panel.get(col) == idx else None,
        )

    fig.update_layout(
        title=dict(text=f"{title_main}<br><sub>lower is better</sub>", x=0.5),
        height=380 * n_rows + 140,
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
        margin=dict(t=110, b=140),
        template="plotly_white",
    )
    return fig


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include",
        choices=["all", "lpv11", "olmixexact"],
        default="all",
        help="'lpv11' restricts to the corpus baselines plus lpv11_fastpipe_v1; "
        "'olmixexact' to the four paper-exact mixture arms head to head.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    include = {
        "all": None,
        "lpv11": LPV11_COMPARE_METHODS,
        "olmixexact": OLMIXEXACT_COMPARE_METHODS,
    }[args.include]
    records = _load_records(include)
    if not records:
        raise SystemExit("No bpb records loaded.")

    suffix = "" if args.include == "all" else f"_{args.include}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in CATEGORIES:
        usable = [r for r in records if metric in r.bpb]
        if not usable:
            logger.warning("No runs carry %s; skipping", metric)
            continue
        fig = _build_figure(records, metric, f"OLMo Base-Easy {metric} vs tokens by model scale (10k curation)")
        out = args.output_dir / f"olmo_bpb_{metric}_grid_x_tokens{suffix}.html"
        fig.write_html(out, include_plotlyjs="cdn")
        logger.info("Wrote %s", out)
        for dim in sorted({r.hidden_dim for r in usable}):
            best: dict[str, float] = {}
            for r in usable:
                if r.hidden_dim == dim:
                    best[r.method] = min(best.get(r.method, math.inf), r.bpb[metric])
            ranking = ", ".join(f"{m}={v:.4f}" for m, v in sorted(best.items(), key=lambda kv: kv[1]))
            logger.info("d%d best-%s (lower better): %s", dim, metric, ranking)


if __name__ == "__main__":
    main()
