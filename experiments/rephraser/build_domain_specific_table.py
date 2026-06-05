# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the domain-specific continued-pretraining table from GCS result JSONs.

This is the single source of truth for Table `tab:domain_specific` (Code/Math/Medical
× Qwen3-0.6B/8B/14B). It mechanically derives every cell from:

  - Training summaries: ``gs://marin-us-central1/metadata/sft_base_results/*.json``
    (one per SFT run: branch, model size, config, hyperparameters, token count).
  - Eval summaries: ``gs://marin-us-central1/metadata/{domain}_sft_base_eval_results/*.json``
    (one per eval; ``output_path`` points at the lm-eval-harness / levanter results).

For each (domain, model size, branch) it selects the hyperparameter cell with the
highest **validation** score and reports that cell's **test** score. The headline
delta is test(best-val-cell) − test(baseline). Token counts come from the winning
run's training summary.

Two result layouts are handled transparently:
  - lm-eval-harness (vLLM, code/math): ``{output_path}/{task}/{model_dir}/results_*.json``
    (multiple timestamped reruns — the latest is used).
  - levanter (medical logprob): a single ``{output_path}/results.json``.

Flexibility: which benchmark is validation vs test is declared per domain in
``DOMAIN_SPECS`` below. To swap them (e.g. make MBPP the test set and HumanEval the
validator), swap ``val_tasks``/``test_tasks`` (and the matching ``*_metric``) on that
spec — nothing else changes. Both sets are evaluated in the same eval run, so the
results already exist on disk.

The script prints a diff against the published table so drift is visible rather than
silently absorbed.

Usage::

    .venv/bin/python experiments/rephraser/build_domain_specific_table.py
    .venv/bin/python experiments/rephraser/build_domain_specific_table.py --json-out scratch/table.json --latex-out scratch/table.tex
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import fsspec

logger = logging.getLogger(__name__)

METADATA_ROOT = "gs://marin-us-central1/metadata"
TRAINING_PREFIX = f"{METADATA_ROOT}/sft_base_results/"

# Model scales, in table column order. Keys match the `model_size` field written by
# the training launcher; the parser also recognises them inside baseline model names.
MODEL_SIZES: list[tuple[str, str]] = [
    ("0.6b", "Qwen3-0.6B-Base"),
    ("8b", "Qwen3-8B-Base"),
    ("14b", "Qwen3-14B-Base"),
]

# Branch -> column letter. Extraction is the method under test; resiliparse is the baseline extractor.
BRANCHES: list[tuple[str, str]] = [("extraction", "E"), ("resiliparse", "R")]

# Minerva MATH = mean over the 7 subtasks at 4-shot (the SFT runs only ran these, not the
# parallel hendrycks_*_0shot baseline eval, which scores ~2% and is NOT what the table uses).
_MATH_TEST_TASKS = (
    "minerva_math_algebra_4shot",
    "minerva_math_counting_and_prob_4shot",
    "minerva_math_geometry_4shot",
    "minerva_math_intermediate_algebra_4shot",
    "minerva_math_num_theory_4shot",
    "minerva_math_prealgebra_4shot",
    "minerva_math_precalc_4shot",
)


@dataclass(frozen=True)
class DomainSpec:
    """How to locate and score one domain's runs.

    ``val_tasks``/``test_tasks`` are lm-eval task names; when more than one is given the
    scores are averaged (Minerva MATH = mean over the 7 Hendrycks subtasks). ``*_metric``
    is matched against result keys by prefix (e.g. ``"pass@1"`` matches ``pass@1,create_test``).
    """

    key: str  # matches the training summary `domain` field
    display: str
    benchmark_display: str
    eval_prefix: str
    val_tasks: tuple[str, ...]
    test_tasks: tuple[str, ...]
    val_metric: str
    test_metric: str
    # Eval `model_name` = training `run_name` + this suffix (medical logprob appends one).
    eval_model_suffix: str = ""


DOMAIN_SPECS: list[DomainSpec] = [
    DomainSpec(
        key="code",
        display="Code",
        benchmark_display="HumanEval",
        eval_prefix=f"{METADATA_ROOT}/code_sft_base_eval_results/",
        val_tasks=("mbpp_3shot",),
        test_tasks=("humaneval_0shot",),
        # lm-eval names the metric differently per task: mbpp -> pass_at_1, humaneval -> pass@1.
        val_metric="pass_at_1",
        test_metric="pass@1",
    ),
    DomainSpec(
        key="math",
        display="Math",
        benchmark_display="Minerva MATH",
        eval_prefix=f"{METADATA_ROOT}/math_sft_base_eval_results/",
        val_tasks=("gsm8k_platinum_cot_8shot",),
        test_tasks=_MATH_TEST_TASKS,
        val_metric="exact_match",
        test_metric="exact_match",
    ),
    DomainSpec(
        key="medical",
        display="Medical",
        # The paper headline metric is mmlu_clinical_knowledge (single task), NOT the
        # 7-subtask average. Labelled "MMLU Clinical" to match the eval, not "MMLU Medical".
        benchmark_display="MMLU Clinical",
        eval_prefix=f"{METADATA_ROOT}/medical_logprob_sft_base_eval_results/",
        val_tasks=("mmlu_college_medicine_logprob_5shot",),
        test_tasks=("mmlu_clinical_knowledge_logprob_5shot",),
        val_metric="acc",
        test_metric="acc",
        eval_model_suffix="-medical-logprob",
    ),
]


# Published values (percentage points) for the diff. base/dE/dR keyed by model size.
@dataclass(frozen=True)
class PublishedRow:
    tokens_e: str
    tokens_r: str
    base: dict[str, float]
    delta_e: dict[str, float]
    delta_r: dict[str, float]


PUBLISHED: dict[str, PublishedRow] = {
    "code": PublishedRow(
        tokens_e="260M",
        tokens_r="793M",
        base={"0.6b": 30.5, "8b": 64.6, "14b": 54.3},
        delta_e={"0.6b": 3.0, "8b": 3.7, "14b": 23.1},
        delta_r={"0.6b": 0.6, "8b": -7.9, "14b": 15.2},
    ),
    "math": PublishedRow(
        tokens_e="253M",
        tokens_r="490M",
        base={"0.6b": 26.7, "8b": 49.4, "14b": 53.6},
        delta_e={"0.6b": 3.1, "8b": 3.2, "14b": 7.8},
        delta_r={"0.6b": -3.8, "8b": 0.8, "14b": 1.2},
    ),
    "medical": PublishedRow(
        tokens_e="560M",
        tokens_r="2.43B",
        base={"0.6b": 62.3, "8b": 80.4, "14b": 83.0},
        delta_e={"0.6b": -0.4, "8b": 0.8, "14b": 1.1},
        delta_r={"0.6b": -0.4, "8b": 0.4, "14b": 0.4},
    ),
}

DIFF_TOLERANCE = 0.1  # points; published values are rounded to one decimal


# --------------------------------------------------------------------------------------
# GCS readers
# --------------------------------------------------------------------------------------
def _list_jsons(prefix: str) -> list[str]:
    fs, urlpath = fsspec.core.url_to_fs(prefix.rstrip("/") + "/")
    if not fs.exists(urlpath):
        logger.warning("Prefix does not exist: %s", prefix)
        return []
    # gcsfs.ls returns scheme-less paths; restore gs:// so fsspec.open targets GCS, not local.
    return [fs.unstrip_protocol(p) for p in fs.ls(urlpath, detail=False) if p.endswith(".json")]


def _read_json(path: str) -> dict | None:
    try:
        with fsspec.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _extract_metric(entry: dict[str, Any], metric: str) -> tuple[float, str] | None:
    """Return (value, key) for ``metric`` in an lm-eval results entry, preferring ``,none``."""
    exact = f"{metric},none"
    if isinstance(entry.get(exact), (int, float)):
        return float(entry[exact]), exact
    for k, v in entry.items():
        if k.startswith(f"{metric},") and not k.startswith(f"{metric}_stderr") and isinstance(v, (int, float)):
            return float(v), k
    return None


def _task_score(output_path: str, task: str, metric: str) -> float | None:
    """Read one task's metric from either the levanter or harness layout."""
    fs, _ = fsspec.core.url_to_fs(output_path)
    base = output_path.rstrip("/")

    # Levanter: single results.json holding all tasks keyed by full task name.
    single = f"{base}/results.json"
    if fs.exists(single):
        d = _read_json(single)
        results = d.get("results", d) if d else None
        if isinstance(results, dict) and task in results:
            got = _extract_metric(results[task], metric)
            return got[0] if got else None
        # Fall through if this layout lacks the task.

    # Harness: {output_path}/{task}/{model_dir}/results_{timestamp}.json (latest wins).
    task_dir = f"{base}/{task}"
    if not fs.exists(task_dir):
        return None
    candidates: list[str] = []
    for inner in fs.ls(task_dir, detail=False):
        try:
            for f_path in fs.ls(inner, detail=False):
                name = f_path.rstrip("/").split("/")[-1]
                if name.startswith("results_") and name.endswith(".json"):
                    candidates.append(fs.unstrip_protocol(f_path))
        except Exception:
            continue
    if not candidates:
        return None
    d = _read_json(max(candidates))  # ISO timestamps sort chronologically
    results = d.get("results", {}) if d else {}
    for entry in results.values():
        got = _extract_metric(entry, metric)
        if got:
            return got[0]
    return None


def _mean_score(output_path: str, tasks: Sequence[str], metric: str) -> float | None:
    vals = [s for t in tasks if (s := _task_score(output_path, t, metric)) is not None]
    if len(vals) != len(tasks):
        # Partial coverage would silently bias an average; treat as missing.
        return None
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
@dataclass
class Cell:
    domain: str
    size: str
    branch: str
    base_test: float | None
    best_test: float | None
    best_val: float | None
    delta: float | None
    tokens: int | None
    winner_run: str | None
    config_name: str | None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    n_candidates: int = 0


def _parse_size(name: str) -> str | None:
    for size, _ in MODEL_SIZES:
        if f"qwen3-{size}-base" in name:
            return size
    return None


def _build_cells(spec: DomainSpec, train_by: dict[str, dict]) -> list[Cell]:
    eval_paths = _list_jsons(spec.eval_prefix)
    evals = [d for p in eval_paths if (d := _read_json(p))]
    logger.info("[%s] %d eval summaries", spec.key, len(evals))

    # Baseline test scores, by size. Baselines are evals whose model_name starts with
    # "baseline"; the size is parsed from the name (naming varies slightly per domain).
    base_by_size: dict[str, float | None] = {size: None for size, _ in MODEL_SIZES}
    for ev in evals:
        name = ev.get("model_name", "")
        if not name.startswith("baseline"):
            continue
        size = _parse_size(name)
        if size is None or size not in base_by_size:
            continue
        score = _mean_score(ev["output_path"], spec.test_tasks, spec.test_metric)
        base_by_size[size] = None if score is None else score * 100
    for size, _ in MODEL_SIZES:
        if base_by_size[size] is None:
            logger.warning("[%s] no baseline test score for %s", spec.key, size)

    # SFT candidates: join eval -> training summary to recover branch/size/tokens/HP.
    candidates: dict[tuple[str, str], list[dict]] = {}
    for ev in evals:
        model_name = ev.get("model_name", "")
        if model_name.startswith("baseline"):
            continue
        run_name = (
            model_name[: -len(spec.eval_model_suffix)]
            if spec.eval_model_suffix and model_name.endswith(spec.eval_model_suffix)
            else model_name
        )
        train = train_by.get(run_name)
        if train is None:
            logger.warning("[%s] eval %s has no training summary", spec.key, run_name)
            continue
        size, branch = train.get("model_size"), train.get("branch")
        if size is None or branch is None:
            continue
        val = _mean_score(ev["output_path"], spec.val_tasks, spec.val_metric)
        test = _mean_score(ev["output_path"], spec.test_tasks, spec.test_metric)
        if val is None or test is None:
            logger.warning("[%s] %s missing val/test score (val=%s test=%s)", spec.key, run_name, val, test)
            continue
        candidates.setdefault((size, branch), []).append(
            {
                "run_name": run_name,
                "val": val * 100,
                "test": test * 100,
                "tokens": (train.get("tokens") or {}).get("total_tokens"),
                "config_name": train.get("config_name"),
                "hyperparameters": train.get("hyperparameters", {}),
            }
        )

    for (size, branch), pool in sorted(candidates.items()):
        logger.info("[%s] %s/%s: %d candidate runs", spec.key, size, branch, len(pool))

    cells: list[Cell] = []
    for size, _ in MODEL_SIZES:
        base = base_by_size[size]
        for branch, _letter in BRANCHES:
            pool = candidates.get((size, branch), [])
            if not pool:
                cells.append(Cell(spec.key, size, branch, base, None, None, None, None, None, None))
                continue
            winner = max(pool, key=lambda c: c["val"])
            delta = None if base is None else winner["test"] - base
            cells.append(
                Cell(
                    domain=spec.key,
                    size=size,
                    branch=branch,
                    base_test=base,
                    best_test=winner["test"],
                    best_val=winner["val"],
                    delta=delta,
                    tokens=winner["tokens"],
                    winner_run=winner["run_name"],
                    config_name=winner["config_name"],
                    hyperparameters=winner["hyperparameters"],
                    n_candidates=len(pool),
                )
            )
    return cells


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def _fmt_tokens(n: int | None) -> str:
    if n is None:
        return "?"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    return f"{n // 1_000_000}M"


def _fmt_delta(d: float | None, bold: bool) -> str:
    if d is None:
        return "?"
    s = f"{d:+.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def _cells_by(cells: list[Cell]) -> dict[tuple[str, str, str], Cell]:
    return {(c.domain, c.size, c.branch): c for c in cells}


def _branch_tokens(by: dict[tuple[str, str, str], Cell], domain: str, branch: str) -> int | None:
    """Token count for a (domain, branch). It is dataset-level (constant across model sizes),
    so return the first populated cell rather than tying it to one scale."""
    for size, _ in MODEL_SIZES:
        cell = by.get((domain, size, branch))
        if cell and cell.tokens is not None:
            return cell.tokens
    return None


def render_latex(specs: list[DomainSpec], cells: list[Cell]) -> str:
    by = _cells_by(cells)
    header_models = " & ".join(f"\\multicolumn{{3}}{{c|}}{{\\textbf{{{disp}}}}}" for _, disp in MODEL_SIZES[:-1])
    header_models += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{MODEL_SIZES[-1][1]}}}}}"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\setlength\\tabcolsep{5pt}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll | rr | ccc | ccc | ccc}",
        "\\toprule",
        f"\\rowcolor[gray]{{0.9}} & & \\multicolumn{{2}}{{c|}}{{\\textbf{{Tokens}}}} & {header_models} \\\\",
        "\\rowcolor[gray]{0.9} \\textbf{Domain} & \\textbf{Benchmark} & \\multicolumn{1}{c}{\\textbf{E}} & "
        "\\multicolumn{1}{c|}{\\textbf{R}} & "
        + " & ".join(
            ["\\textbf{Base} & \\textbf{$\\Delta$\\,E}\\,$\\uparrow$ & \\textbf{$\\Delta$\\,R}\\,$\\uparrow$"] * 3
        )
        + " \\\\",
        "\\midrule",
    ]
    for spec in specs:
        tok_e = _fmt_tokens(_branch_tokens(by, spec.key, "extraction"))
        tok_r = _fmt_tokens(_branch_tokens(by, spec.key, "resiliparse"))
        row = [f"{spec.display:<8}", f"{spec.benchmark_display:<13}", tok_e, tok_r]
        for size, _ in MODEL_SIZES:
            e = by.get((spec.key, size, "extraction"))
            r = by.get((spec.key, size, "resiliparse"))
            base = e.base_test if e else None
            de = e.delta if e else None
            dr = r.delta if r else None
            bold_e = de is not None and dr is not None and de >= dr
            row += [
                "?" if base is None else f"{base:.1f}",
                _fmt_delta(de, bold_e),
                _fmt_delta(dr, dr is not None and not bold_e),
            ]
        lines.append(" & ".join(row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\caption{Continued-pretraining gains over Qwen3 base, by extraction method and scale. "
        "Extraction (E) matches or beats resiliparse (R) at every (scale, domain) cell despite training on "
        "2--4$\\times$ fewer tokens. $\\Delta$ values are absolute point gains over base.}",
        "\\label{tab:domain_specific}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def render_diff(cells: list[Cell]) -> tuple[str, bool]:
    by = _cells_by(cells)
    rows: list[str] = []
    all_ok = True
    header = f"{'domain':<8} {'size':<5} {'field':<8} {'computed':>9} {'published':>10} {'Δ':>7}  status"
    rows.append(header)
    rows.append("-" * len(header))
    for spec in DOMAIN_SPECS:
        pub = PUBLISHED.get(spec.key)
        if pub is None:
            continue
        for size, _ in MODEL_SIZES:
            e = by.get((spec.key, size, "extraction"))
            r = by.get((spec.key, size, "resiliparse"))
            checks = [
                ("base", e.base_test if e else None, pub.base.get(size)),
                ("dE", e.delta if e else None, pub.delta_e.get(size)),
                ("dR", r.delta if r else None, pub.delta_r.get(size)),
            ]
            for fname, comp, published in checks:
                if comp is None or published is None:
                    status, diff_s, comp_s = "MISSING", "", "—" if comp is None else f"{comp:.1f}"
                    all_ok = False
                else:
                    diff = comp - published
                    ok = abs(diff) <= DIFF_TOLERANCE
                    all_ok = all_ok and ok
                    status, diff_s, comp_s = ("ok" if ok else "MISMATCH"), f"{diff:+.1f}", f"{comp:.1f}"
                pub_s = "—" if published is None else f"{published:.1f}"
                rows.append(f"{spec.key:<8} {size:<5} {fname:<8} {comp_s:>9} {pub_s:>10} {diff_s:>7}  {status}")
    # Token diff
    for spec in DOMAIN_SPECS:
        pub = PUBLISHED.get(spec.key)
        for letter, branch, published in [("tok_E", "extraction", pub.tokens_e), ("tok_R", "resiliparse", pub.tokens_r)]:
            comp = _fmt_tokens(_branch_tokens(by, spec.key, branch))
            ok = comp == published
            all_ok = all_ok and ok
            rows.append(
                f"{spec.key:<8} {'':<5} {letter:<8} {comp:>9} {published:>10} {'':>7}  {'ok' if ok else 'MISMATCH'}"
            )
    return "\n".join(rows), all_ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--training-prefix", default=TRAINING_PREFIX)
    parser.add_argument("--json-out", default=None, help="Write the flat cell list here.")
    parser.add_argument("--latex-out", default=None, help="Write the LaTeX table here.")
    parser.add_argument("--no-diff", action="store_true", help="Skip the published-table diff.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    train_by: dict[str, dict] = {}
    for p in _list_jsons(args.training_prefix):
        d = _read_json(p)
        if d and d.get("run_name"):
            train_by[d["run_name"]] = d
    logger.info("Loaded %d training summaries", len(train_by))

    cells: list[Cell] = []
    for spec in DOMAIN_SPECS:
        cells.extend(_build_cells(spec, train_by))

    latex = render_latex(DOMAIN_SPECS, cells)
    print("\n" + latex + "\n")

    if not args.no_diff:
        diff, all_ok = render_diff(cells)
        print("\n=== DIFF vs published table ===")
        print(diff)
        print(f"\n{'ALL CELLS MATCH' if all_ok else 'DRIFT DETECTED — see MISMATCH/MISSING rows above'}\n")

    if args.json_out:
        flat = [vars(c) for c in cells]
        with fsspec.open(args.json_out, "w") as f:
            f.write(json.dumps(flat, indent=2, default=str))
        logger.info("Wrote %d cells to %s", len(flat), args.json_out)
    if args.latex_out:
        with fsspec.open(args.latex_out, "w") as f:
            f.write(latex + "\n")
        logger.info("Wrote LaTeX to %s", args.latex_out)


if __name__ == "__main__":
    main()
