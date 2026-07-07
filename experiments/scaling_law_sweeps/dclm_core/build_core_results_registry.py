# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a tidy DCLM CORE results registry from per-model eval result JSONs.

Scans a CORE results prefix (the `*_summary.json` siblings written by
`run_dclm_core_eval.py`) and assembles two tidy tables for tracking + plotting:

  - **wide** (one row per model): run_name, method, hidden_dim, budget_flops,
    batch, region, Core_v2, Core, plus one column per CORE task's centered score.
  - **long** (one row per model-task pair): the same identity columns plus
    `task`, `raw`, `centered`.

The script is a stateless re-scan: run it as often as you like as evals land
(it's idempotent and preemption-proof — no state to lose). It reads only the
small summary siblings, never the ~1 GB finals.

Usage (CPU-only Iris job, co-located with the results bucket):

    iris --cluster marin job run --region us-central1 \\
        --cpu 2 --memory 8GB --disk 10GB --priority batch --no-wait \\
        -- python -m experiments.scaling_law_sweeps.dclm_core.build_core_results_registry \\
               --results-prefix gs://marin-us-central1/metadata/data_curation_10k_core_results/ \\
               --out-prefix    gs://marin-us-central1/metadata/data_curation_10k_core_registry/
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field

from rigging.filesystem import filesystem as marin_filesystem

logger = logging.getLogger(__name__)

# The 22 low-variance CORE v2 tasks, in canonical order. Sourced from the
# `dclm.low_variance_datasets` field of any complete result; pinned here so the
# wide table has a stable column order even when a model is missing a task.
CORE_TASKS: tuple[str, ...] = (
    "hellaswag_zeroshot",
    "jeopardy",
    "bigbench_qa_wikidata",
    "arc_easy",
    "arc_challenge",
    "copa",
    "commonsense_qa",
    "piqa",
    "openbook_qa",
    "lambada_openai",
    "hellaswag",
    "winograd",
    "winogrande",
    "bigbench_dyck_languages",
    "agi_eval_lsat_ar",
    "bigbench_cs_algorithms",
    "bigbench_operators",
    "bigbench_repeat_copy_logic",
    "squad",
    "coqa",
    "boolq",
    "bigbench_language_identification",
)

# run_name looks like: curation-<method>-expFM_natural-<budget>-d<dim>-L<layers>-B<batch>[-suffix]
_RUN_RE = re.compile(
    r"^curation-(?P<method>.+?)-(?P<tag>expFM_natural)-(?P<budget>[0-9.e+]+)-d(?P<dim>\d+)-L(?P<layers>\d+)-B(?P<batch>\d+)"
)
_REGION_RE = re.compile(r"gs://marin-([a-z0-9-]+)/")


@dataclass
class ModelRow:
    run_name: str
    method: str
    hidden_dim: int
    budget_flops: float
    batch: int
    region: str
    core_v2: float | None
    core: float | None
    centered: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)


def _gcs_ls(prefix: str) -> list[str]:
    try:
        entries = marin_filesystem("gcs").ls(prefix, detail=False)
    except FileNotFoundError:
        return []
    return [e if e.startswith("gs://") else f"gs://{e}" for e in entries]


def _gcs_cat(path: str) -> bytes:
    with marin_filesystem("gcs").open(path, "rb") as f:
        return f.read()


def parse_row(summary: dict) -> ModelRow | None:
    """Build a ModelRow from one `*_summary.json` doc, or None if unparseable."""
    run_name = summary.get("run_name")
    if not run_name:
        return None
    m = _RUN_RE.match(run_name)
    if m is None:
        logger.warning("run_name does not match expected pattern, skipping: %s", run_name)
        return None
    checkpoint = summary.get("checkpoint") or ""
    region_match = _REGION_RE.search(checkpoint)
    dclm = summary.get("dclm") or {}
    return ModelRow(
        run_name=run_name,
        method=m["method"],
        hidden_dim=int(m["dim"]),
        budget_flops=float(m["budget"]),
        batch=int(m["batch"]),
        region=region_match.group(1) if region_match else "",
        core_v2=dclm.get("Core_v2"),
        core=dclm.get("Core"),
        centered=dict(dclm.get("centered_results") or {}),
        raw=dict(dclm.get("raw_results") or {}),
    )


def collect_rows(results_prefix: str) -> list[ModelRow]:
    """Read every `*_summary.json` under results_prefix into ModelRows."""
    summaries = [p for p in _gcs_ls(results_prefix) if p.endswith("_summary.json")]
    logger.info("Found %d summary siblings under %s", len(summaries), results_prefix)
    rows: list[ModelRow] = []
    for sp in summaries:
        try:
            doc = json.loads(_gcs_cat(sp))
        except Exception as e:
            logger.warning("Skipping unreadable summary %s: %s", sp, e)
            continue
        row = parse_row(doc)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r.method, r.hidden_dim, r.budget_flops))
    logger.info("Parsed %d model rows", len(rows))
    return rows


def wide_csv(rows: list[ModelRow]) -> str:
    """One row per model: identity + Core + per-task centered scores."""
    buf = io.StringIO()
    header = ["run_name", "method", "hidden_dim", "budget_flops", "batch", "region", "core_v2", "core"]
    header += [f"centered__{t}" for t in CORE_TASKS]
    w = csv.writer(buf)
    w.writerow(header)
    for r in rows:
        row = [r.run_name, r.method, r.hidden_dim, r.budget_flops, r.batch, r.region, r.core_v2, r.core]
        row += [r.centered.get(t) for t in CORE_TASKS]
        w.writerow(row)
    return buf.getvalue()


def long_csv(rows: list[ModelRow]) -> str:
    """One row per model-task pair: raw and centered scores side by side."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run_name", "method", "hidden_dim", "budget_flops", "batch", "region", "task", "raw", "centered"])
    for r in rows:
        for t in CORE_TASKS:
            w.writerow(
                [
                    r.run_name,
                    r.method,
                    r.hidden_dim,
                    r.budget_flops,
                    r.batch,
                    r.region,
                    t,
                    r.raw.get(t),
                    r.centered.get(t),
                ]
            )
    return buf.getvalue()


def _gcs_write(path: str, text: str) -> None:
    with marin_filesystem("gcs").open(path, "w") as f:
        f.write(text)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-prefix", required=True, help="GCS prefix with CORE result `*_summary.json` siblings.")
    ap.add_argument("--out-prefix", required=True, help="GCS prefix to write the registry tables under.")
    ap.add_argument("--local-only", action="store_true", help="Print a coverage summary; do not write tables.")
    args = ap.parse_args()

    rows = collect_rows(args.results_prefix)

    by_method: dict[str, int] = {}
    scored = 0
    for r in rows:
        by_method[r.method] = by_method.get(r.method, 0) + 1
        if r.core_v2 is not None:
            scored += 1
    logger.info("Coverage: %d models, %d with a Core_v2 score", len(rows), scored)
    for m in sorted(by_method):
        logger.info("  %-20s %d", m, by_method[m])

    if args.local_only:
        return

    out = args.out_prefix.rstrip("/")
    _gcs_write(f"{out}/core_results_wide.csv", wide_csv(rows))
    _gcs_write(f"{out}/core_results_long.csv", long_csv(rows))
    logger.info("Wrote %s/core_results_wide.csv and core_results_long.csv", out)


if __name__ == "__main__":
    main()
