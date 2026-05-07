# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print a Markdown progress table for an in-flight fixed-model sweep.

Maps (method, budget) -> {summary exists, or peak checkpoint step across
regions} and renders as a table with %-of-target. The target step count comes
from the planner so the math is always consistent with what's actually
training.

Example:
    uv run python experiments/scaling_law_sweeps/sweep_progress.py \\
        --hidden-sizes 2432 \\
        --methods dclm nemotron_full_bos_fixed llm_curated_bos_fixed resiliparse \\
        --budgets 3e18 9e18 1.8e19 3e19 9e19 1.8e20 3e20 9e20 1.8e21

Subprocess shells `gcloud storage ls` per (method, budget, region) — runs in
parallel via concurrent.futures so the wall-clock stays bounded for the
~5-region x ~4-method x ~9-budget grid.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

from experiments.scaling_law_sweeps.curation_plan import METHODS
from experiments.scaling_law_sweeps.fixed_model_plan import enumerate_fixed_model_plans

logger = logging.getLogger(__name__)

REGIONS = ["us-central1", "us-east5", "eu-west4", "us-east1", "us-central2"]
SUMMARY_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"


def _ckpt_path(region: str, run_name: str) -> str:
    return f"gs://marin-{region}/checkpoints/isoflop-curation/{run_name}/checkpoints/"


def _peak_step(region: str, run_name: str) -> tuple[str, int] | None:
    """Return (region, peak_step) for the run in this region, or None."""
    try:
        out = subprocess.run(
            ["gcloud", "storage", "ls", _ckpt_path(region, run_name)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        # Some buckets occasionally hang on transient list pagination — skip rather than crash.
        return None
    if out.returncode != 0:
        return None
    steps = [int(m.group(1)) for m in re.finditer(r"step-(\d+)", out.stdout)]
    if not steps:
        return None
    return (region, max(steps))


def _existing_summaries(hidden_dim: int) -> set[str]:
    """Set of run_names that already have a summary JSON."""
    out = subprocess.run(
        ["gcloud", "storage", "ls", SUMMARY_PREFIX],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if out.returncode != 0:
        return set()
    runs = set()
    for line in out.stdout.split():
        name = line.rsplit("/", 1)[-1].removesuffix(".json")
        if f"d{hidden_dim}-" in name:
            runs.add(name)
    return runs


def _budget_str(budget: float) -> str:
    return f"{budget:.0e}".replace("e+0", "e+").replace("e+", "e+0") if budget < 1e19 else f"{budget:.0e}"


def _normalize_budget_for_run_name(budget: float) -> str:
    """run_name uses f'{budget:.0e}' which rounds 1.8e19 -> '2e+19', etc."""
    return f"{budget:.0e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[2432])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dclm", "nemotron_full_bos_fixed", "llm_curated_bos_fixed", "resiliparse"],
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20, 9e20, 1.8e21],
    )
    parser.add_argument(
        "--display-method-names",
        nargs="+",
        help="Optional: pretty names for table header. Same length as --methods.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(args.hidden_sizes) != 1:
        raise ValueError("--hidden-sizes must be a single value for a coherent table.")
    hidden = args.hidden_sizes[0]

    method_objs = [METHODS[m] for m in args.methods]
    plans = enumerate_fixed_model_plans(
        methods=method_objs,
        hidden_sizes=(hidden,),
        budgets=tuple(args.budgets),
    )
    plan_by_key: dict[tuple[str, float], object] = {(p.method_name, p.budget): p for p in plans}

    existing = _existing_summaries(hidden)
    logger.info("Found %d existing summaries for d%d", len(existing), hidden)

    # Build all (method, budget, region) probes
    probes = []
    for m in args.methods:
        for b in args.budgets:
            p = plan_by_key.get((m, b))
            if p is None:
                continue
            run_name = f"curation-{p.method_name}-expFM_natural-{_normalize_budget_for_run_name(b)}-d{p.hidden_dim}-L{p.num_layers}-B{p.batch_size}"
            for region in REGIONS:
                probes.append((m, b, run_name, region, p))

    # Run all probes in parallel
    cells: dict[tuple[str, float], dict] = {
        (m, b): {"plan": plan_by_key.get((m, b))} for m in args.methods for b in args.budgets
    }

    def probe(args_tuple):
        m, b, run_name, region, p = args_tuple
        result = _peak_step(region, run_name)
        return (m, b, run_name, region, result)

    with ThreadPoolExecutor(max_workers=20) as ex:
        for m, b, run_name, region, result in ex.map(probe, probes):
            cell = cells[(m, b)]
            cell["run_name"] = run_name
            cell["done"] = run_name in existing
            if result is not None:
                _, step = result
                if step > cell.get("peak_step", -1):
                    cell["peak_step"] = step
                    cell["peak_region"] = result[0]

    # Render Markdown table
    pretty = dict(zip(args.methods, args.display_method_names or args.methods))
    print()
    header = "| budget | tokens | target |" + "".join(f" {pretty[m]} |" for m in args.methods)
    sep = "|---|---|---|" + "".join("---|" for _ in args.methods)
    print(header)
    print(sep)
    for b in args.budgets:
        # use the planner's target_steps for this budget (same across methods at fixed hidden_dim)
        sample_p = next((cells[(m, b)]["plan"] for m in args.methods if cells[(m, b)].get("plan") is not None), None)
        if sample_p is None:
            continue
        target = sample_p.train_steps
        tokens_total = sample_p.batch_size * target * sample_p.seq_len
        if tokens_total >= 1e12:
            toks_str = f"{tokens_total / 1e12:.2f}T"
        elif tokens_total >= 1e9:
            toks_str = f"{tokens_total / 1e9:.2f}B"
        else:
            toks_str = f"{tokens_total / 1e6:.0f}M"
        row = f"| {b:.1e} | {toks_str} | {target:,} |"
        for m in args.methods:
            cell = cells[(m, b)]
            if cell.get("done"):
                row += " ✓ |"
            elif cell.get("peak_step") is not None:
                pct = cell["peak_step"] / target * 100
                row += f" {pct:.0f}% ({cell['peak_step']:,}) |"
            else:
                row += " — |"
        print(row)
    print()


if __name__ == "__main__":
    main()
