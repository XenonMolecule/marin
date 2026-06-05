# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a LaTeX CORE table from per-run bootstrap result JSONs.

Reads:
  - a layout YAML defining rows (methods) x columns (model scales, budgets)
  - a directory or glob of `*_bootstrap.json` written by `bootstrap_core.py`

For each (row, column) cell:
  - derives run_name deterministically via the launcher's naming convention:
        curation-{method}-{experiment_tag}-{budget:.0e}-d{hidden_size}-L{L}-B{B}
    where (L, B) come from `_candidate_for_fixed_model(hidden_size, budget)`.
  - looks up the matching bootstrap JSON (basename `{run_name}_bootstrap.json`)
  - extracts core_mean, core_stdev

Writes:
  - CSV (one row per cell)
  - LaTeX table rendered from a Jinja2 template

Bold modes:
  - "ours":  the row marked `bold: true` in the layout is always bolded
  - "max":   the column-max is bolded
  - "none":  no bolding
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import jinja2
import yaml

from experiments.scaling_law_sweeps.dclm_core.bootstrap_core import _read_partial
from experiments.scaling_law_sweeps.fixed_model_plan import (
    EXPERIMENT_TAG,
    _candidate_for_fixed_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cell:
    row_label: str
    column_header: str
    method: str
    hidden_size: int
    budget: float
    run_name: str
    core_mean: float | None
    core_stdev: float | None
    n_tasks_present: int | None
    is_bold: bool


def _run_name_prefix(method: str, hidden_size: int, budget: float, experiment_tag: str) -> str:
    """Prefix shared across the (L, B) drift — the launcher's HP heuristic may
    have changed between training and now, so the L/B suffix isn't a stable
    identifier. (method, hidden, budget) IS stable.
    """
    return f"curation-{method}-{experiment_tag}-{budget:.0e}-d{hidden_size}-"


def _resolve_run_name(
    method: str,
    hidden_size: int,
    budget: float,
    experiment_tag: str,
    bootstrap_index: dict[str, Path],
) -> str:
    """Find the bootstrap JSON's run_name by (method, hidden, budget) prefix.

    Errors if 0 or >1 bootstrap JSONs match the prefix.
    """
    prefix = _run_name_prefix(method, hidden_size, budget, experiment_tag)
    matches = sorted(rn for rn in bootstrap_index if rn.startswith(prefix))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        # Fall back to deterministic derivation for the error message — useful
        # when the bootstrap JSON simply hasn't been computed yet.
        candidate = _candidate_for_fixed_model(hidden_size, budget)
        derived = (
            "(planner-rejected)"
            if candidate is None
            else (f"{prefix}L{candidate.model_config.num_layers}-B{candidate.batch_size}")
        )
        raise FileNotFoundError(f"No bootstrap JSON for prefix {prefix!r} " f"(expected something like {derived!r})")
    raise ValueError(f"Ambiguous: {len(matches)} bootstrap JSONs match prefix {prefix!r}: {matches}")


def _load_bootstrap_index(bootstrap_glob: str) -> dict[str, Path]:
    """Build {run_name -> path-to-bootstrap-json} from a glob."""
    paths = (
        sorted(Path().glob(bootstrap_glob))
        if not bootstrap_glob.startswith("/")
        else sorted(Path("/").glob(bootstrap_glob.lstrip("/")))
    )
    index: dict[str, Path] = {}
    for p in paths:
        stem = p.name
        if stem.endswith("_bootstrap.json"):
            run_name = stem[: -len("_bootstrap.json")]
            index[run_name] = p
    return index


def _build_cell(
    row: dict,
    column: dict,
    col_idx: int,
    experiment_tag: str,
    bootstrap_index: dict[str, Path],
) -> Cell:
    method = row["method"]
    hidden_size = int(column["hidden_size"])
    # Budgets are per-row (one per column) so each (method, scale) cell can
    # point at a different FLOPs point — e.g. the best-Uncheatable-loss
    # checkpoint per method at the same model size.
    budget = float(row["budgets"][col_idx])

    core_mean: float | None = None
    core_stdev: float | None = None
    n_tasks_present: int | None = None
    try:
        run_name = _resolve_run_name(method, hidden_size, budget, experiment_tag, bootstrap_index)
        bs_path = bootstrap_index[run_name]
        data = _read_partial(str(bs_path))
        if data is not None:
            core_mean = data.get("core_mean")
            core_stdev = data.get("core_stdev")
            n_tasks_present = data.get("n_tasks_present")
        else:
            logger.warning("Could not parse bootstrap JSON at %s", bs_path)
    except (FileNotFoundError, ValueError) as e:
        logger.warning("[%s d=%d budget=%.0e] %s", method, hidden_size, budget, e)
        run_name = _run_name_prefix(method, hidden_size, budget, experiment_tag) + "??"

    return Cell(
        row_label=row["label"],
        column_header=column["header"],
        method=method,
        hidden_size=hidden_size,
        budget=budget,
        run_name=run_name,
        core_mean=core_mean,
        core_stdev=core_stdev,
        n_tasks_present=n_tasks_present,
        is_bold=False,  # filled in by _apply_bold
    )


def _apply_bold(cells: list[list[Cell]], rows: list[dict], bold_mode: str) -> list[list[Cell]]:
    """Set is_bold on the appropriate cells per the chosen mode."""
    if bold_mode == "none":
        return cells

    if bold_mode == "ours":
        ours_rows = {i for i, r in enumerate(rows) if r.get("bold")}
        return [
            [Cell(**{**c.__dict__, "is_bold": i in ours_rows}) for c in row_cells] for i, row_cells in enumerate(cells)
        ]

    if bold_mode == "max":
        n_cols = len(cells[0]) if cells else 0
        winners: list[int | None] = []
        for col in range(n_cols):
            col_vals = [(i, cells[i][col].core_mean) for i in range(len(cells))]
            valid = [(i, v) for i, v in col_vals if v is not None]
            winners.append(max(valid, key=lambda iv: iv[1])[0] if valid else None)
        return [
            [Cell(**{**c.__dict__, "is_bold": winners[j] == i}) for j, c in enumerate(row_cells)]
            for i, row_cells in enumerate(cells)
        ]

    raise ValueError(f"Unknown bold_mode: {bold_mode!r}")


def _format_cell(cell: Cell, precision: int, scale: float) -> str:
    """LaTeX cell: '21.54 {\\tiny $\\pm$ 0.59}' or '\\textbf{21.54} ...' or '---'.

    `scale` multiplies both mean and stdev before formatting — set to 100 to
    report CORE as percentages.
    """
    if cell.core_mean is None or cell.core_stdev is None:
        return "---"
    fmt = f"{{:.{precision}f}}"
    mean_str = fmt.format(cell.core_mean * scale)
    stdev_str = fmt.format(cell.core_stdev * scale)
    if cell.is_bold:
        mean_str = f"\\textbf{{{mean_str}}}"
    return f"{mean_str} {{\\tiny $\\pm$ {stdev_str}}}"


def render(layout_path: Path) -> None:
    with layout_path.open() as f:
        layout = yaml.safe_load(f)

    experiment_tag = layout.get("experiment_tag", EXPERIMENT_TAG)
    bold_mode = layout.get("bold_mode", "ours")
    precision = int(layout.get("precision", 3))
    scale = float(layout.get("scale", 1.0))
    bootstrap_index = _load_bootstrap_index(layout["bootstrap_glob"])
    logger.info("Indexed %d bootstrap JSONs from %s", len(bootstrap_index), layout["bootstrap_glob"])

    rows = layout["rows"]
    columns = layout["columns"]

    grid: list[list[Cell]] = []
    for row in rows:
        if "budgets" not in row or len(row["budgets"]) != len(columns):
            raise ValueError(
                f"Row {row.get('label')!r} must have a 'budgets' list with "
                f"{len(columns)} entries (one per column), got {row.get('budgets')!r}"
            )
        row_cells = [_build_cell(row, col, i, experiment_tag, bootstrap_index) for i, col in enumerate(columns)]
        grid.append(row_cells)

    grid = _apply_bold(grid, rows, bold_mode)

    # CSV output.
    csv_path = Path(layout["output_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_label",
                "column_header",
                "method",
                "hidden_size",
                "budget",
                "run_name",
                "core_mean",
                "core_stdev",
                "n_tasks_present",
                "is_bold",
            ]
        )
        for row_cells in grid:
            for c in row_cells:
                writer.writerow(
                    [
                        c.row_label,
                        c.column_header,
                        c.method,
                        c.hidden_size,
                        f"{c.budget:.0e}",
                        c.run_name,
                        "" if c.core_mean is None else c.core_mean,
                        "" if c.core_stdev is None else c.core_stdev,
                        "" if c.n_tasks_present is None else c.n_tasks_present,
                        c.is_bold,
                    ]
                )
    logger.info("Wrote CSV to %s", csv_path)

    # LaTeX template.
    template_path = Path(layout["template"])
    # LaTeX-friendly delimiters: `{%` is a literal LaTeX line continuation
    # in \resizebox{...}{%, so we move Jinja's block/variable markers away
    # from `{` and `%`.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_path.parent),
        autoescape=False,
        keep_trailing_newline=True,
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((#",
        comment_end_string="#))",
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)

    # Build a per-group view so the template can emit \rowcolor section headers.
    groups: list[dict] = []
    for i, row in enumerate(rows):
        group_name = row.get("group", "")
        if not groups or groups[-1]["name"] != group_name:
            groups.append({"name": group_name, "rows": []})
        groups[-1]["rows"].append(
            {
                "label": row["label"],
                "cells": [_format_cell(c, precision, scale) for c in grid[i]],
            }
        )

    rendered = template.render(
        columns=columns,
        groups=groups,
        n_columns=len(columns),
        caption=layout.get("caption", ""),
        label=layout.get("label", "tab:dclm-core"),
    )
    tex_path = Path(layout["output_tex"])
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(rendered)
    logger.info("Wrote LaTeX to %s", tex_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True, help="Path to layout YAML.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    render(Path(args.layout))


if __name__ == "__main__":
    main()
