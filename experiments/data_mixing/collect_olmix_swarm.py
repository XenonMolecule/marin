# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the swarm's ``(mixture, BPB)`` dataset into olmix's own CSV schema.

Emits `ratios.csv` and `metrics.csv` exactly as olmix's `fit` CLI expects them:

    ratios.csv    run,name,index,<domain>,...
    metrics.csv   run,name,index,<task>,...

Two reasons for matching their schema rather than inventing one:

* it makes our fit **independently checkable** -- the same inputs can be fed to
  `olmix fit` and the proposed mixtures compared, which is the strongest end-to-end
  validation available short of running their whole pipeline;
* olmix's loader enforces `np.allclose(row_sums, 1.0, atol=0.01)` and joins ratios to
  metrics on the run id, so writing their format forces us to satisfy their invariants.

A run only contributes if it has BOTH a training result and a complete set of BPB scores.
A run evaluated on a subset of tasks is excluded with a warning rather than being padded:
a missing task would otherwise become a hole in one task's regression, silently changing
what that task's law was fit on.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
from dataclasses import dataclass

import fsspec
import numpy as np

from experiments.data_mixing.build_olmix_bpb_manifest import SWARM_BPB_ROOT
from experiments.data_mixing.olmix_plan import SwarmManifest, effective_domains, read_manifest
from experiments.data_mixing.olmix_plan import run_name as build_run_name
from experiments.data_mixing.olmix_tasks import build_target_tasks
from experiments.data_mixing.run_olmix_swarm_standalone import DEFAULT_RESULTS_PREFIX, PROXY_TRAIN_STEPS
from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logger = logging.getLogger(__name__)

RATIOS_FILENAME = "ratios.csv"
METRICS_FILENAME = "metrics.csv"

# olmix's loader tolerance on mixture row sums.
ROW_SUM_ATOL = 0.01


@dataclass(frozen=True)
class SwarmRow:
    """One completed, fully-evaluated swarm member."""

    run_name: str
    index: int
    weights: dict[str, float]
    bpb: dict[str, float]


def _read_json(path: str) -> dict:
    with fsspec.open(path) as fh:
        return json.load(fh)


CORE_V2_ROOT = "metadata/olmix_swarm_core_v2"
# BLEnD cultural-knowledge bpb. Same document shape as the olmo bpb suite, so it reuses
# `task_bpb_scores` and only the prefix differs. Kept in its own tree on purpose: BLEnD is a
# diagnostic, and fitting a mixture on it is an explicit opt-in via --metric blend, never
# something a default `--tasks all` can pull in.
BLEND_ROOT = "metadata/olmix_swarm_blend"


def task_core_v2_error(summary_doc: dict) -> dict[str, float]:
    """Pull ``{task: 1 - centered_accuracy}`` out of ``run_dclm_core_eval``'s summary.

    The metric is INVERTED on purpose. ``olmix_solve`` minimises
    ``sum(exp(t @ x))``, which is convex -- ``cp.Maximize`` of it is not DCP and ECOS
    refuses it. So the target must be a quantity where *lower is better*, exactly as bpb
    is. Fitting ``1 - centered`` keeps the law ``exp(log_c) + exp(t.x)``, the
    droppable additive ``log_c``, and the Minimize direction all correct, and the argmin
    over mixtures is the argmax of centered accuracy.

    ``compute_core`` emits the STRING ``"N/A due to missing tasks: [...]"`` rather than a
    float when a task is absent, so values are type-checked rather than ``float()``-ed.
    """
    dclm = summary_doc.get("dclm")
    if not isinstance(dclm, dict):
        raise ValueError(f"no 'dclm' block (keys: {sorted(summary_doc)}); not run_dclm_core_eval output")
    centered = dclm.get("centered_results")
    if not isinstance(centered, dict):
        raise ValueError(f"no 'dclm.centered_results' map (dclm keys: {sorted(dclm)})")
    return {task: 1.0 - float(v) for task, v in centered.items() if isinstance(v, (int, float))}


def task_bpb_scores(bpb_doc: dict) -> dict[str, float]:
    """Pull ``{task: bpb}`` out of ``run_olmo_bpb_eval``'s results.json.

    The harness writes ``tasks: {<task>: {"bpb": float, "bpb_no_leading_space": float,
    "n_docs": int}}`` and puts the mean under ``averages.macro_bpb``. There is no flat
    top-level ``bpb`` map. Reading the document as though there were one yields zero
    matching task names, which does not raise -- it reports every run as having
    incomplete BPB and quietly produces an empty dataset.
    """
    tasks = bpb_doc.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(
            f"BPB results have no 'tasks' map (keys: {sorted(bpb_doc)}); " f"this is not run_olmo_bpb_eval output"
        )
    return {task: float(score["bpb"]) for task, score in tasks.items() if isinstance(score, dict) and "bpb" in score}


def run_name_for(manifest: SwarmManifest, index: int) -> str:
    """The run name this manifest's row `index` must produce."""
    return build_run_name(manifest.corpus, manifest.seed, manifest.k, index, manifest.row(index))


@dataclass(frozen=True)
class RegionSource:
    """One region's completed-run results and the BPB scores that go with them.

    The two travel together and cannot be pooled under a single prefix: an eval writes its
    results.json into the *checkpoint's own bucket*, so a run trained in us-central1 has its
    scores in us-central1 no matter where the collection runs.
    """

    results_prefix: str
    bpb_prefix: str
    # "bpb"      -> <bpb_prefix>/<run>/results.json, read task_bpb_scores
    # "blend"    -> same shape as "bpb"; only sources_from_regions' prefix differs
    # "core_v2"  -> <bpb_prefix>/<run>_summary.json, read task_core_v2_error
    metric: str = "bpb"

    def score_path(self, run_name: str) -> str:
        base = self.bpb_prefix.rstrip("/")
        return f"{base}/{run_name}_summary.json" if self.metric == "core_v2" else f"{base}/{run_name}/results.json"

    def read_scores(self, doc: dict) -> dict[str, float]:
        return task_core_v2_error(doc) if self.metric == "core_v2" else task_bpb_scores(doc)


def _list_result_paths(results_prefix: str) -> list[str]:
    """Fully-qualified result JSON paths under one prefix (fsspec strips the protocol)."""
    fs, root = fsspec.core.url_to_fs(results_prefix)
    if not fs.exists(root):
        logger.warning("results prefix does not exist yet: %s", results_prefix)
        return []
    scheme = results_prefix.split("://", 1)[0] + "://" if "://" in results_prefix else ""
    return sorted(
        f"{scheme}{p}" if scheme and "://" not in p else p for p in fs.ls(root, detail=False) if p.endswith(".json")
    )


def load_swarm_rows(
    sources: list[RegionSource],
    manifest: SwarmManifest,
    task_names: list[str],
    expected_train_steps: int | None = None,
) -> tuple[list[SwarmRow], dict[str, int]]:
    """Pair each completed run with its BPB scores, merging across regions.

    Returns the usable rows plus a tally of why runs were skipped, so a shrinking
    dataset is visible rather than silent.
    """
    skipped = {
        "no_bpb": 0,
        "incomplete_bpb": 0,
        "wrong_train_steps": 0,
        "index_mismatch": 0,
        "duplicate_run": 0,
        "manifest_mismatch": 0,
    }

    # A run's scores normally sit in the same bucket as its training result. That breaks
    # when a checkpoint is MIGRATED between regions to chase capacity: the eval then writes
    # its summary into the destination bucket while the training record stays behind, and a
    # same-bucket lookup silently drops the run as `no_bpb`. Index every region's scores up
    # front and look up by run name, so placement never changes which runs the fit sees.
    score_index: dict[str, str] = {}
    for source in sources:
        base = source.bpb_prefix.rstrip("/")
        fs_i, root_i = fsspec.core.url_to_fs(base)
        if not fs_i.exists(root_i):
            continue
        scheme = base.split("://", 1)[0] + "://" if "://" in base else ""
        for entry in fs_i.ls(root_i, detail=False):
            leaf = entry.rstrip("/").rsplit("/", 1)[-1]
            name = leaf[: -len("_summary.json")] if leaf.endswith("_summary.json") else leaf
            full = f"{scheme}{entry}" if scheme and "://" not in entry else entry
            path = full if source.metric == "core_v2" else f"{full.rstrip('/')}/results.json"
            score_index.setdefault(name, path)
    logger.info("indexed scores for %d runs across %d regions", len(score_index), len(sources))

    rows: list[SwarmRow] = []
    seen: dict[str, str] = {}
    for source in sources:
        for path in _list_result_paths(source.results_prefix):
            result = _read_json(path)
            run_name = result["run_name"]
            index = int(result["index"])

            # The same (corpus, index) can finish in two regions when coordinators own
            # overlapping ranges. Both trained the same mixture, so either row is valid --
            # but admitting both would double that mixture's weight in the regression.
            if run_name in seen:
                skipped["duplicate_run"] += 1
                logger.warning(
                    "%s completed in more than one region (%s and %s); keeping the first",
                    run_name,
                    seen[run_name],
                    source.results_prefix,
                )
                continue

            # A partial run (a smoke, or one relaunched with different steps) must never
            # enter the objective: its BPB reflects a different amount of TRAINING, not a
            # different mixture, which is precisely the confound the swarm exists to
            # isolate. One such run already appeared, from a 3,100-step smoke written
            # under a real index.
            if expected_train_steps is not None and int(result.get("train_steps", -1)) != expected_train_steps:
                skipped["wrong_train_steps"] += 1
                logger.warning(
                    "%s: train_steps=%s but the swarm is %d; excluded (partial runs cannot enter the objective)",
                    run_name,
                    result.get("train_steps"),
                    expected_train_steps,
                )
                continue
            if index < 0 or index >= manifest.k:
                skipped["index_mismatch"] += 1
                logger.warning("%s: index %d outside the manifest's [0, %d)", run_name, index, manifest.k)
                continue

            # Merging regions is only sound if both sampled the SAME swarm. The run name
            # ends in a hash of its mixture, so comparing it against the manifest row for
            # this index catches a region whose manifest was resampled or ordered
            # differently -- which would otherwise pair one region's BPB with another
            # region's weights and silently corrupt the design matrix.
            expected_name = run_name_for(manifest, index)
            if run_name != expected_name:
                skipped["manifest_mismatch"] += 1
                logger.warning(
                    "%s does not match this manifest's row %d (expected %s); " "the region's swarm differs -- excluded",
                    run_name,
                    index,
                    expected_name,
                )
                continue

            bpb_path = score_index.get(run_name)
            if bpb_path is None:
                skipped["no_bpb"] += 1
                continue
            scores = source.read_scores(_read_json(bpb_path))
            missing = [t for t in task_names if t not in scores]
            if missing:
                skipped["incomplete_bpb"] += 1
                logger.warning(
                    "%s: missing %d/%d tasks (e.g. %s); excluded", run_name, len(missing), len(task_names), missing[:3]
                )
                continue

            seen[run_name] = source.results_prefix
            rows.append(
                SwarmRow(
                    run_name=run_name,
                    index=index,
                    weights={d: float(w) for d, w in zip(manifest.domains, manifest.weights[index], strict=True)},
                    bpb={t: float(scores[t]) for t in task_names},
                )
            )
    return rows, skipped


def write_ratios_csv(rows: list[SwarmRow], domains: list[str]) -> str:
    """olmix `ratios.csv`: one column per domain, rows summing to 1."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run", "name", "index", *domains])
    for r in rows:
        vals = [r.weights.get(d, 0.0) for d in domains]
        total = sum(vals)
        if abs(total - 1.0) > ROW_SUM_ATOL:
            raise ValueError(f"{r.run_name}: mixture sums to {total!r}, outside olmix's {ROW_SUM_ATOL} tolerance")
        w.writerow([r.run_name, r.run_name, r.index, *vals])
    return buf.getvalue()


def write_metrics_csv(rows: list[SwarmRow], task_names: list[str]) -> str:
    """olmix `metrics.csv`: one column per task, joined to ratios on the run id."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run", "name", "index", *task_names])
    for r in rows:
        w.writerow([r.run_name, r.run_name, r.index, *[r.bpb[t] for t in task_names]])
    return buf.getvalue()


def live_domains(rows: list[SwarmRow], manifest: SwarmManifest) -> tuple[list[str], list[str]]:
    """Split the manifest's domains into (live, dead) **across the collected rows only**.

    A domain that no collected mixture samples has an all-zero design column, so the fit
    strips it -- which means the identifiability bar is ``m_live + 1``, not ``m + 1`` over
    all 120 cells. Computing it from the rows we actually have (rather than from the whole
    sampled swarm) is what makes the number match what the law will see.

    The bar MOVES, and not always in our favour: a newly collected run adds one row but may
    activate several previously-unseen domains, raising the requirement by more than it
    raises the supply. So "runs needed" can grow as runs land, and must be recomputed rather
    than cached.
    """
    # An empty collection is the normal early state (runs finished, evals still pending), not
    # an error -- and `np.array([])` is 1-D, so it would raise on the column check instead of
    # reporting the obvious answer: with no rows, nothing is live.
    if not rows:
        return [], list(manifest.domains)
    weights = np.array([[row.weights.get(d, 0.0) for d in manifest.domains] for row in rows])
    # Pin the token dict to the manifest's canonical domain order; effective_domains derives
    # its names from that mapping's iteration order, so a mismatch would mislabel columns.
    ordered_tokens = {d: manifest.tokens[d] for d in manifest.domains}
    return effective_domains(ordered_tokens, weights)


def collect(
    manifest: SwarmManifest,
    sources: list[RegionSource],
    task_names: list[str],
    output_prefix: str,
    expected_train_steps: int | None = None,
) -> dict:
    """Write ratios.csv + metrics.csv and return a summary of what was included."""
    rows, skipped = load_swarm_rows(sources, manifest, task_names, expected_train_steps)
    if not rows:
        raise RuntimeError(f"no usable swarm rows across {[s.results_prefix for s in sources]}; skipped={skipped}")
    ratios = write_ratios_csv(rows, list(manifest.domains))
    metrics = write_metrics_csv(rows, task_names)
    for name, payload in ((RATIOS_FILENAME, ratios), (METRICS_FILENAME, metrics)):
        path = f"{output_prefix.rstrip('/')}/{name}"
        with fsspec.open(path, "w") as fh:
            fh.write(payload)
        logger.info("wrote %s (%d rows)", path, len(rows))

    # The identifiability bar is m_live+1, NOT m+1: the fit strips domains no collected
    # mixture samples, so those columns are never estimated. Measuring against all m
    # overstates the requirement and would report "not ready" on a dataset the law can
    # actually solve.
    m = len(manifest.domains)
    live, dead = live_domains(rows, manifest)
    needed = len(live) + 1
    if len(rows) < needed:
        logger.warning(
            "only %d usable runs for %d LIVE domains (%d of %d never sampled); the "
            "log-linear fit needs >= m_live+1 = %d for a unique solution. Wait for more "
            "runs before trusting a solve.",
            len(rows),
            len(live),
            len(dead),
            m,
            needed,
        )
    return {
        "rows": len(rows),
        "domains": m,
        "live_domains": len(live),
        "dead_domains": len(dead),
        "runs_needed_for_unique_fit": needed,
        "tasks": len(task_names),
        "skipped": skipped,
        "sufficient_for_unique_fit": len(rows) >= needed,
    }


def sources_from_regions(regions: list[str], corpus: str, metric: str = "bpb") -> list[RegionSource]:
    """Pair each region's swarm results with that same bucket's per-run scores."""
    root = {"core_v2": CORE_V2_ROOT, "blend": BLEND_ROOT}.get(metric, SWARM_BPB_ROOT)
    sources = []
    for region in regions:
        bucket = REGION_TO_BUCKET[region]
        sources.append(
            RegionSource(
                results_prefix=f"{bucket}/{DEFAULT_RESULTS_PREFIX.format(corpus=corpus)}",
                bpb_prefix=f"{bucket}/{root}",
                metric=metric,
            )
        )
    return sources


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", required=True)
    p.add_argument(
        "--region",
        action="append",
        required=True,
        help="Repeatable. A swarm split across regions must be merged here -- each region "
        "holds only the runs it trained, and their BPB scores live in that same bucket.",
    )
    p.add_argument("--manifest", required=True, help="gs:// path to the swarm manifest JSON.")
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--expected-train-steps", type=int, default=PROXY_TRAIN_STEPS)
    p.add_argument("--include-mmlu", action="store_true", default=True)
    p.add_argument("--no-include-mmlu", dest="include_mmlu", action="store_false")
    args = p.parse_args()

    manifest = read_manifest(args.manifest)
    task_names = list(build_target_tasks(include_mmlu=args.include_mmlu))
    summary = collect(
        manifest=manifest,
        sources=sources_from_regions(args.region, args.corpus),
        task_names=task_names,
        output_prefix=args.output_prefix,
        expected_train_steps=args.expected_train_steps,
    )
    logger.info("collected: %s", summary)


if __name__ == "__main__":
    main()
