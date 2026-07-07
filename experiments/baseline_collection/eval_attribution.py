# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-example eval attribution — Phase A: find the eval examples hq LOSES on.

Idea (Michael, 2026-07-06): for eval examples where hq underperforms dclm/nemotron,
extract topic keywords, then n-gram-search each pretraining corpus for docs carrying
those keywords (esp. dclm/nemo-kept, hq-dropped) → a held-out attribution set + the
domains/website-types hq is missing. This is Phase A: identify the hq-worse examples.

The 10k CoreV2 evals were run with lm-eval-harness `--log_samples`, so each
`.../metadata/data_curation_10k_core_tasks_results/<run>/results.json` carries a
per-example `samples[task]` list (doc text, gold, per-choice logprobs). We extract a
per-example score per method and flag where hq is significantly worse.

STRICT IN-REGION: the 370MB result files are read only in their own region. Run this
in us-east5 (hq+dclm+nemo core_tasks results are co-located there at d1536/d2432).
Output is a small parquet (per-example scores + hq-worse flags) → downloaded for Phase B.

Launch (us-east5, in-region):
    iris --cluster marin job run --region us-east5 --enable-extra-resources \\
        --cpu 16 --memory 128GB --extra cpu \\
        -- python experiments/baseline_collection/eval_attribution.py --region us-east5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.baseline_collection.provenance_audit_10k import _assert_no_cross_region, _gcs_ls_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_attribution")

METHODS = {"high_quality_10k": "hq", "dclm_10k": "dclm", "nemotron_10k": "nemo"}
OUT_PREFIX = "gs://marin-us-east5/scratch/eval_attribution"
# Both per-example eval suites (OLMES adds sciq/social_iqa/winogrande + clean per-example bpb;
# CoreV2 core_tasks adds lambada/agieval/copa). OLMo-bpb has NO per-example → separate re-eval.
SUITES = ("data_curation_10k_core_tasks_results", "olmes_base_results")


def _suite_glob(region: str, suite: str) -> str:
    return f"gs://marin-{region}/metadata/{suite}"


def _cell_of(run_dir: str) -> tuple[str, int] | None:
    m = re.search(r"expFM_natural-([0-9e+]+)-d(\d+)-", run_dir)
    return (m.group(1), int(m.group(2))) if m else None


def _method_of(run_dir: str) -> str | None:
    for full, short in METHODS.items():
        if f"curation-{full}-" in run_dir:
            return short
    return None


def _per_example_score(sample: dict) -> float | None:
    """Extract a per-example difficulty score (lower=better for the model).

    lm-eval-harness sample dicts vary by task. Prefer the gold continuation's
    negative-loglik-per-byte; fall back to (1 - acc) so a wrong MC answer = 1.0.
    """
    # bits-per-byte style: some harness configs log 'bpb' directly per example.
    for k in ("bpb", "byte_perplexity", "bits_per_byte"):
        if isinstance(sample.get(k), (int, float)):
            return float(sample[k])
    resps = sample.get("filtered_resps") or sample.get("resps")
    gold = sample.get("target")

    def _ll(r):
        return float(r[0]) if isinstance(r, (list, tuple)) else float(r)

    # MC tasks: gold is an int index into per-choice logliks.
    if resps and isinstance(gold, int) and 0 <= gold < len(resps):
        try:
            return -_ll(resps[gold])
        except (TypeError, ValueError, IndexError):
            pass
    # Generative/cloze tasks (jeopardy, lambada, naturalqs): resps holds the gold
    # continuation's loglik (single request). Same target across methods → comparable.
    if resps and not isinstance(gold, int):
        try:
            return -_ll(resps[0])
        except (TypeError, ValueError, IndexError):
            pass
    # fallback: correctness metric.
    for k in ("acc", "acc_norm", "exact_match", "em"):
        if isinstance(sample.get(k), (int, float)):
            return 1.0 - float(sample[k])
    return None


def _doc_text(sample: dict) -> str:
    doc = sample.get("doc") or {}
    for k in ("query", "question", "goal", "ctx", "text", "premise", "activity_label"):
        if isinstance(doc.get(k), str):
            return doc[k][:600]
    # arguments[0][0] is often the full prompt string.
    args = sample.get("arguments")
    if args and isinstance(args[0], (list, tuple)) and isinstance(args[0][0], str):
        return args[0][0][:600]
    return json.dumps(doc)[:600]


def _load_examples(results_path: str) -> dict[str, list[dict]]:
    """task -> list of {idx, text, gold, score} from one run's results.json (in-region)."""
    _assert_no_cross_region(results_path)
    with fsspec.open(results_path, "rb") as f:
        data = json.load(f)
    samples = data.get("samples") or {}
    # Cover ALL tasks (breadth), not a hand-picked subset. Log what's present/skipped.
    logger.info("  samples has %d tasks: %s", len(samples), sorted(samples))
    out: dict[str, list[dict]] = {}
    skipped: list[str] = []
    for task, rows in samples.items():
        recs = []
        for i, s in enumerate(rows):
            sc = _per_example_score(s)
            if sc is None:
                continue
            recs.append({"idx": s.get("doc_id", i), "text": _doc_text(s), "gold": str(s.get("target")), "score": sc})
        if recs:
            out[task] = recs
        else:
            skipped.append(task)
    if skipped:
        logger.info("  tasks with no extractable per-example score (format unhandled): %s", skipped)
    return out


# Model scales to attribute at, then UNION the hq-worse sets (per Michael): an example
# counts if hq is worse at EITHER scale. 3584 ~= 8B, 2432 ~= 2.9B compute-optimal dim.
SCALES = (3584, 2432)


def _suite_hq_worse(region: str, suite: str, threshold: float) -> list[dict]:
    """Extract hq-worse per-example rows from one eval suite, at each scale in SCALES."""
    run_dirs = _gcs_ls_glob(f"{_suite_glob(region, suite)}/curation-*")
    by_cell: dict[tuple, dict[str, str]] = defaultdict(dict)
    for d in run_dirs:
        meth, cell = _method_of(d), _cell_of(d)
        if meth and cell:
            by_cell[cell][meth] = d.rstrip("/") + "/results.json"
    rows: list[dict] = []
    for dim in SCALES:
        # largest-flops cell at this dim with all 3 methods present
        cands = [(c, m) for c, m in by_cell.items() if c[1] == dim and len(m) >= 3]
        if not cands:
            logger.warning("[%s] no all-3-method cell at d%d; skipping scale", suite, dim)
            continue
        cell, cell_methods = sorted(cands, key=lambda cm: float(cm[0][0].replace("+", "")))[-1]
        logger.info("[%s] scale d%d → cell %s; methods %s", suite, dim, cell, list(cell_methods))
        rows.extend(_cell_hq_worse(cell, cell_methods, suite, threshold))
    return rows


def _cell_hq_worse(cell: tuple, cell_methods: dict[str, str], suite: str, threshold: float) -> list[dict]:
    per_method = {meth: _load_examples(path) for meth, path in cell_methods.items()}
    rows = []
    for task in sorted(per_method.get("hq", {})):
        hq_rows = {r["idx"]: r for r in per_method["hq"].get(task, [])}
        dclm_rows = {r["idx"]: r for r in per_method.get("dclm", {}).get(task, [])}
        nemo_rows = {r["idx"]: r for r in per_method.get("nemo", {}).get(task, [])}
        for idx, hq in hq_rows.items():
            d, n = dclm_rows.get(idx), nemo_rows.get(idx)
            best_other = min([x["score"] for x in (d, n) if x is not None], default=None)
            if best_other is None:
                continue
            delta = hq["score"] - best_other
            rows.append(
                {
                    "suite": suite.replace("data_curation_10k_", "").replace("_results", ""),
                    "task": task,
                    "cell": f"{cell[0]}_d{cell[1]}",
                    "idx": str(idx),
                    "text": hq["text"],
                    "gold": hq["gold"],
                    "hq": hq["score"],
                    "dclm": d["score"] if d else None,
                    "nemo": n["score"] if n else None,
                    "delta_hq_minus_best": delta,
                    "hq_worse": delta > threshold,
                }
            )
    return rows


def run(args: argparse.Namespace) -> int:
    rows_out: list[dict] = []
    for suite in SUITES:
        rows_out.extend(_suite_hq_worse(args.region, suite, args.threshold))
    n_worse = sum(r["hq_worse"] for r in rows_out)
    logger.info(
        "Extracted %d joined examples across %d suites; %d hq-worse (delta>%.3f)",
        len(rows_out),
        len(SUITES),
        n_worse,
        args.threshold,
    )
    out_path = f"{OUT_PREFIX}/hq_worse_examples_all_suites.parquet"
    _assert_no_cross_region(out_path)
    with fsspec.filesystem("gcs").open(out_path, "wb") as fh:
        pq.write_table(pa.Table.from_pylist(rows_out), fh, compression="zstd")
    logger.info("Wrote %s", out_path)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", default="us-east5")
    p.add_argument("--threshold", type=float, default=0.05, help="min (hq - best_other) per-example score gap to flag")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
