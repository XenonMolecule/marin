# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect DCLM Core v2 results over the OLMIX swarm and report per-task signal.

The swarm proxies are d512 / 157M / ~4.3e9 tokens, small enough that several Core v2
tasks sit at their random baseline. A log-linear law fit on such a task contributes
noise to the objective with the same 1/n weight as a real one, so the task set must be
chosen from the *measured* spread across the swarm rather than assumed.

For each of the 22 Core v2 tasks this compares two quantities:

* ``sd_swarm`` -- the standard deviation of centered accuracy across the swarm. This is
  the signal the mixture actually moves.
* ``se_noise``  -- the binomial standard error of a single run's score,
  ``sqrt(p(1-p)/n) / (1 - baseline)``, with ``n`` and ``baseline`` read from
  ``eval_meta_data.csv``. Centering divides by ``1 - baseline``, so it inflates noise on
  tasks with a high random baseline (boolq's 62% baseline multiplies its noise by 2.6).

``snr = sd_swarm / se_noise`` is the ratio to judge on. A task at snr ~1 is measuring its
own sampling error, not the data mixture.

Usage::

    python -m experiments.data_mixing.collect_swarm_core_v2 --out scratch/core_v2_signal
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import statistics
from dataclasses import dataclass, field

import fsspec

from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logger = logging.getLogger(__name__)

SWARM_CORE_V2_ROOT = "metadata/olmix_swarm_core_v2"
DEFAULT_REGIONS = ("us-east5", "us-central1", "europe-west4", "us-west4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DCLM_CORE_DIR = os.path.join(_HERE, "..", "scaling_law_sweeps", "dclm_core")
_AGGREGATION_JSON = os.path.join(_DCLM_CORE_DIR, "additional_aggregation.json")
_EVAL_META_CSV = os.path.join(_DCLM_CORE_DIR, "eval_meta_data.csv")

# snr thresholds for the reported verdict. Below ~1.5 a task's across-swarm spread is
# within a small factor of the noise a single measurement carries.
SNR_STRONG = 5.0
SNR_USABLE = 2.5
SNR_WEAK = 1.5

# Below this many runs, `sd_swarm` is itself so uncertain that the verdicts are noise.
# The sd of a sample sd is ~sd/sqrt(2(n-1)): +-50% at n=3, +-13% at n=30. Reading a
# "DEAD" off a handful of early runs and dropping a task on it would be a real mistake,
# so the report refuses to stand behind verdicts below this and says so loudly.
MIN_RUNS_FOR_VERDICT = 30


@dataclass
class TaskMeta:
    """Doc count and random baseline for one Core v2 task, from DCLM's own metadata."""

    n_docs: int
    baseline: float  # fraction in [0, 1)


@dataclass
class SwarmCoreRow:
    run_name: str
    corpus: str
    index: int
    region: str
    core_v2: float | None
    centered: dict[str, float] = field(default_factory=dict)


def core_v2_tasks(aggregation_json: str = _AGGREGATION_JSON) -> list[str]:
    with open(aggregation_json) as f:
        return list(json.load(f)["low_variance_datasets"])


def task_metadata(csv_path: str = _EVAL_META_CSV) -> dict[str, TaskMeta]:
    """Doc counts and random baselines, keyed by DCLM eval-task name."""
    meta: dict[str, TaskMeta] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["Eval Task"].strip()
            if not name:
                continue
            meta[name] = TaskMeta(
                n_docs=int(row["#datapoints"]),
                baseline=float(row["Random baseline"]) / 100.0,
            )
    return meta


_RUN_NAME_RE = re.compile(r"^olmix-(?P<corpus>.+)-s\d+-K\d+-i(?P<index>\d+)-w[0-9a-f]+$")


def parse_run_name(run_name: str) -> tuple[str, int] | None:
    """``olmix-dclm_10k-s42-K363-i0085-w7983b27c`` -> ``("dclm_10k", 85)``.

    The corpus may itself contain hyphens, so anchor on the fixed ``-s<seed>-K<K>-i<idx>-w<hash>``
    tail rather than splitting on ``-``.
    """
    m = _RUN_NAME_RE.match(run_name)
    if m is None:
        return None
    return m.group("corpus"), int(m.group("index"))


def collect(regions: tuple[str, ...]) -> tuple[list[SwarmCoreRow], dict[str, int]]:
    """Read every per-run Core v2 summary, deduping a run to its first-seen region."""
    rows: dict[str, SwarmCoreRow] = {}
    skipped: dict[str, int] = {"duplicate_run": 0, "unparsed_name": 0, "core_v2_na": 0}

    for region in regions:
        bucket = REGION_TO_BUCKET[region]
        fs, root = fsspec.core.url_to_fs(f"{bucket}/{SWARM_CORE_V2_ROOT}")
        if not fs.exists(root):
            logger.info("%-14s no results prefix yet", region)
            continue
        paths = [p for p in fs.ls(root, detail=False) if p.endswith("_summary.json")]
        logger.info("%-14s %d summaries", region, len(paths))
        for path in paths:
            with fs.open(path, "r") as fh:
                doc = json.load(fh)
            run_name = doc["run_name"]
            if run_name in rows:
                skipped["duplicate_run"] += 1
                continue
            parsed = parse_run_name(run_name)
            if parsed is None:
                skipped["unparsed_name"] += 1
                continue
            corpus, index = parsed
            dclm = doc["dclm"]
            # compute_core emits the STRING "N/A due to missing tasks: [...]" rather than a
            # float when any of the 22 is absent, so this must be type-checked, not float()'d.
            raw_core = dclm.get("Core_v2")
            core = float(raw_core) if isinstance(raw_core, (int, float)) else None
            if core is None:
                skipped["core_v2_na"] += 1
            rows[run_name] = SwarmCoreRow(
                run_name=run_name,
                corpus=corpus,
                index=index,
                region=region,
                core_v2=core,
                centered={k: float(v) for k, v in dclm.get("centered_results", {}).items()},
            )
    return list(rows.values()), skipped


def binomial_se_centered(p_centered: float, meta: TaskMeta) -> float:
    """SE of a single run's *centered* score, propagated from the raw accuracy."""
    denom = 1.0 - meta.baseline
    p_raw = min(max(p_centered * denom + meta.baseline, 0.0), 1.0)
    return math.sqrt(max(p_raw * (1.0 - p_raw), 1e-12) / meta.n_docs) / denom


def verdict(snr: float) -> str:
    if snr >= SNR_STRONG:
        return "strong"
    if snr >= SNR_USABLE:
        return "usable"
    if snr >= SNR_WEAK:
        return "weak"
    return "DEAD"


def signal_report(rows: list[SwarmCoreRow], tasks: list[str], meta: dict[str, TaskMeta]) -> list[dict]:
    report = []
    for task in tasks:
        vals = [r.centered[task] for r in rows if task in r.centered]
        if len(vals) < 2:
            report.append({"task": task, "n_runs": len(vals), "verdict": "no_data"})
            continue
        sd = statistics.pstdev(vals)
        mean = statistics.mean(vals)
        tm = meta.get(task)
        se = binomial_se_centered(mean, tm) if tm else float("nan")
        snr = sd / se if se and math.isfinite(se) and se > 0 else float("nan")
        report.append(
            {
                "task": task,
                "n_runs": len(vals),
                "n_docs": tm.n_docs if tm else None,
                "baseline": tm.baseline if tm else None,
                "mean": mean,
                "sd_swarm": sd,
                "min": min(vals),
                "max": max(vals),
                "se_noise": se,
                "snr": snr,
                "verdict": verdict(snr) if math.isfinite(snr) else "unknown",
            }
        )
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", action="append", dest="regions", default=None)
    ap.add_argument("--out", default=None, help="Write <out>.json (rows) and <out>_signal.csv.")
    args = ap.parse_args()

    regions = tuple(args.regions) if args.regions else DEFAULT_REGIONS
    tasks = core_v2_tasks()
    meta = task_metadata()

    rows, skipped = collect(regions)
    logger.info("Collected %d distinct runs; skipped=%s", len(rows), skipped)
    if not rows:
        logger.warning("No Core v2 results found yet under %s.", SWARM_CORE_V2_ROOT)
        return

    by_corpus: dict[str, int] = {}
    for r in rows:
        by_corpus[r.corpus] = by_corpus.get(r.corpus, 0) + 1
    logger.info("Per corpus: %s", by_corpus)

    cores = [r.core_v2 for r in rows if r.core_v2 is not None]
    if cores:
        logger.info(
            "Core_v2 (x100): mean=%.3f sd=%.3f min=%.3f max=%.3f over %d runs",
            100 * statistics.mean(cores),
            100 * statistics.pstdev(cores) if len(cores) > 1 else 0.0,
            100 * min(cores),
            100 * max(cores),
            len(cores),
        )

    report = signal_report(rows, tasks, meta)
    provisional = len(rows) < MIN_RUNS_FOR_VERDICT
    logger.info("")
    if provisional:
        logger.warning(
            "PROVISIONAL: only %d runs collected (< %d). sd_swarm is estimated to about "
            "+-%.0f%% at this n, so the verdicts below are NOT yet meaningful -- do not drop a "
            "task on them. Re-run once the fleet has filled in.",
            len(rows),
            MIN_RUNS_FOR_VERDICT,
            100 / (2 * max(len(rows) - 1, 1)) ** 0.5,
        )
    logger.info("%-34s %6s %8s %9s %9s %7s  %s", "task", "nruns", "mean", "sd_swarm", "se_noise", "snr", "verdict")
    for e in sorted(report, key=lambda x: -(x.get("snr") or -1)):
        if e["verdict"] in ("no_data",):
            logger.info("%-34s %6d  (no data)", e["task"], e["n_runs"])
            continue
        logger.info(
            "%-34s %6d %8.3f %9.3f %9.3f %7.1f  %s",
            e["task"],
            e["n_runs"],
            100 * e["mean"],
            100 * e["sd_swarm"],
            100 * e["se_noise"],
            e["snr"],
            e["verdict"],
        )
    keep = [e["task"] for e in report if e.get("verdict") in ("strong", "usable")]
    logger.info("")
    logger.info("Tasks clearing snr>=%.1f: %d / %d", SNR_USABLE, len(keep), len(tasks))
    logger.info("  %s", ", ".join(keep))
    if provisional:
        logger.warning("^ PROVISIONAL at n=%d; not a task-selection decision yet.", len(rows))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(f"{args.out}.json", "w") as f:
            json.dump(
                {
                    "runs": [
                        {
                            "run_name": r.run_name,
                            "corpus": r.corpus,
                            "index": r.index,
                            "region": r.region,
                            "core_v2": r.core_v2,
                            "centered": r.centered,
                        }
                        for r in rows
                    ],
                    "skipped": skipped,
                },
                f,
            )
        with open(f"{args.out}_signal.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
            w.writeheader()
            w.writerows(report)
        logger.info("Wrote %s.json and %s_signal.csv", args.out, args.out)


if __name__ == "__main__":
    main()
