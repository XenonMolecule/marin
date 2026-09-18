# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Universal eval-integrity verifier: every data point, every namespace, hard verdicts.

For every trained run of the audited sweeps, verifies its eval in each applicable
namespace on three axes:

1. FINAL CHECKPOINT.
   * ``olmo_bpb_results`` documents carry an explicit ``checkpoint`` path: the step in
     that path must equal the run's ``plan.train_steps`` (or -1, Levanter's off-by-one).
   * ``mmlu_target_models`` / ``cultural_target_models`` documents carry no checkpoint
     path, so the structural test is used: the eval's OWN bucket (results land in the
     region whose checkpoint the eval read) must hold the final-step hf export, AND the
     eval must have STARTED after that export existed (lm-eval's ``date`` field when
     present; else the results.json mtime, which is weaker — flagged UNPROVEN when
     intermediate step exports also exist in the bucket). The eval runner always loads
     the max step present, so start-after-export pins the scored checkpoint to final.
2. COMPLETE: exact scored-task counts — BPB 60 tasks with bpb values, MMLU 58 /
   cultural 20 results entries carrying ``choice_logprob,none`` (matching what
   consolidate_model_evals rolls up). Never a partial average.
3. Verdict per (run, namespace): VERIFIED | MISSING | INCOMPLETE | STALE | UNPROVEN.

Writes ``gs://marin-us-central1/metadata/eval_integrity/<namespace>.csv`` and prints a
summary. Exit code 1 if anything is STALE/INCOMPLETE/UNPROVEN (MISSING fails only with
``--require-complete-coverage``). Run ON-CLUSTER: MMLU documents are ~228 MB each.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import io
import json
import logging
import re
import sys

import fsspec

from experiments.data_mixing.olmix_tasks import build_olmix_exact_tasks
from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import (
    CODE_BPB,
    MATH_BPB,
    MT_MBPP_BPB,
    QA_LANG_BPB,
    QA_RC_MC_BPB,
)

logger = logging.getLogger(__name__)

# Canonical BPB task sets (mirrors plot_olmo_bpb_by_size_10k.CATEGORIES without pulling
# in matplotlib). olmixexact runs must cover the olmix51 suite; natural-arm runs predate
# MMLU joining the suite and must cover olmix47 + the Code/Math/QA categories. Supersets
# pass: several waves scored extra tasks, which harms nothing.
NATURAL_BPB_TASKS = (
    frozenset(build_olmix_exact_tasks(include_mmlu=False))
    | frozenset(CODE_BPB)
    | frozenset(MT_MBPP_BPB)
    | frozenset(MATH_BPB)
    | frozenset(QA_LANG_BPB)
    | frozenset(QA_RC_MC_BPB)
)
# olmixexact runs additionally need the 4 MMLU category tasks: their macro trusts every
# category, so the requirement is the full canonical union (60 tasks).
OLMIXEXACT_BPB_TASKS = NATURAL_BPB_TASKS | frozenset(build_olmix_exact_tasks(include_mmlu=True))

BUCKETS = (
    "marin-us-east5",
    "marin-us-east1",
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-west4",
    "marin-eu-west4",
)
TRAIN_PREFIX = "marin-us-central1/metadata/data_curation_10k_natural_results"
CKPT_TEMPLATE = "{bucket}/checkpoints/isoflop-curation/{run}/hf"
OUT_PREFIX = "marin-us-central1/metadata/eval_integrity"
STEP_RE = re.compile(r"step-(\d+)")

NAMESPACES = {
    # namespace -> (results dir under each bucket, run filter substrings,
    #               expected scored-task count, results carry a checkpoint path)
    "olmo_bpb_results": ("metadata/olmo_bpb_results", ("olmixexact",), 60, True),
    "mmlu_target_models": ("metadata/mmlu_target_models/5shot", ("_mix_mmlu_", "fineweb_edu_10k-expFM"), 58, False),
    "cultural_target_models": (
        "metadata/cultural_target_models/5shot",
        ("_mix_cultural_clp", "fineweb_edu_10k-expFM"),
        20,
        False,
    ),
}


def _epoch(value) -> float:
    """Normalize gcsfs mtime/updated (datetime or ISO string) to epoch seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime.datetime):
        return value.timestamp()
    return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _mtime(fs, path: str) -> float:
    info = fs.info(path)
    return _epoch(info.get("mtime") or info.get("updated"))


def hf_step_exports(fs, bucket: str, run: str) -> dict[int, str]:
    """{step: config.json path} for every hf export of `run` in `bucket`."""
    root = CKPT_TEMPLATE.format(bucket=bucket, run=run)
    try:
        listing = fs.ls(root, detail=False)
    except FileNotFoundError:
        return {}
    out = {}
    for d in listing:
        m = STEP_RE.search(d.rsplit("/", 1)[-1])
        if m and fs.exists(f"{d}/config.json"):
            out[int(m.group(1))] = f"{d}/config.json"
    return out


def completeness_failure(ns: str, run: str, doc: dict, expected_tasks: int) -> str | None:
    """None when the eval is complete, else a human-readable deficit."""
    if ns == "olmo_bpb_results":
        scored = {k for k, v in (doc.get("tasks") or {}).items() if isinstance(v, dict) and v.get("bpb") is not None}
        required = OLMIXEXACT_BPB_TASKS if "olmixexact" in run else NATURAL_BPB_TASKS
        missing = required - scored
        return f"missing {len(missing)} required tasks: {sorted(missing)[:4]}..." if missing else None
    n = sum(1 for v in (doc.get("results") or {}).values() if isinstance(v, dict) and "choice_logprob,none" in v)
    return f"{n}/{expected_tasks} scored tasks" if n != expected_tasks else None


def verify_run(fs, ns: str, subdir: str, expected_tasks: int, has_provenance: bool, run: str) -> tuple:
    want = int(json.load(fs.open(f"{TRAIN_PREFIX}/{run}.json"))["plan"]["train_steps"])
    found = next((b for b in BUCKETS if fs.exists(f"{b}/{subdir}/{run}/results.json")), None)
    if found is None:
        return (run, "MISSING", "", want, "")
    path = f"{found}/{subdir}/{run}/results.json"
    doc = json.load(fs.open(path))

    deficit = completeness_failure(ns, run, doc, expected_tasks)
    if deficit:
        return (run, "INCOMPLETE", deficit, want, found)

    # A provenance.json sidecar (written by run_mmlu_eval / run_cultural_eval since
    # 2026-09-17, before scoring starts) names the exact checkpoint scored — hard proof.
    # Only trust it when the results POSTDATE it: an older sidecar cannot describe them,
    # and a newer one belongs to an in-flight re-eval.
    sidecar = f"{found}/{subdir}/{run}/provenance.json"
    if not has_provenance and fs.exists(sidecar) and _mtime(fs, path) >= _mtime(fs, sidecar):
        m = STEP_RE.search(json.load(fs.open(sidecar)).get("checkpoint", ""))
        if m:
            evaled = int(m.group(1))
            if evaled in (want, want - 1):
                return (run, "VERIFIED", f"provenance sidecar: step-{evaled}", want, found)
            return (run, "STALE", f"provenance sidecar: evaled step-{evaled}, final {want}", want, found)

    if has_provenance:
        m = STEP_RE.search(doc.get("checkpoint", ""))
        if not m:
            return (run, "UNPROVEN", "no checkpoint field in results", want, found)
        evaled = int(m.group(1))
        if evaled not in (want, want - 1):
            return (run, "STALE", f"evaled step-{evaled}, final {want}", want, found)
        return (run, "VERIFIED", f"checkpoint step-{evaled}", want, found)

    # Provenance-free namespaces: structural final-in-bucket + start-after-export test.
    exports = hf_step_exports(fs, found, run)
    final_steps = [s for s in exports if s in (want, want - 1)]
    if not final_steps:
        have = ",".join(f"step-{s}" for s in sorted(exports)) or "none"
        return (run, "STALE", f"final export absent from eval bucket (has {have})", want, found)
    export_time = _mtime(fs, exports[max(final_steps)])
    eval_start = doc.get("date")  # lm-eval start timestamp (epoch), when present
    if eval_start is not None:
        if float(eval_start) <= export_time:
            return (run, "STALE", "eval started before final export landed", want, found)
        return (run, "VERIFIED", "eval started after final export", want, found)
    finish_time = _mtime(fs, path)
    if finish_time <= export_time:
        return (run, "STALE", "results predate final export in eval bucket", want, found)
    if len(exports) > 1:
        # A long eval could have started on an intermediate export and finished after
        # the final one landed; without a start timestamp this cannot be excluded.
        return (run, "UNPROVEN", "intermediate exports present; only finish-time ordering available", want, found)
    return (run, "VERIFIED", "sole export is final; results postdate it", want, found)


def verify_namespace(fs, ns: str, workers: int) -> list[tuple]:
    subdir, filters, expected_tasks, has_provenance = NAMESPACES[ns]
    trained_all = {
        p.rsplit("/", 1)[-1].removesuffix(".json") for p in fs.ls(TRAIN_PREFIX, detail=False) if p.endswith(".json")
    }
    trained_all = {t for t in trained_all if "." not in t}  # drop quarantined (.badmodel etc.)
    scoped = {t for t in trained_all if any(f in t for f in filters)}
    # Every results.json actually present is audited too, so datapoints outside the
    # name filters (natural arms, ad-hoc evals) cannot escape verification.
    present: set[str] = set()
    for b in BUCKETS:
        try:
            present |= {
                p.rsplit("/", 2)[-2] for p in fs.find(f"{b}/{subdir}", maxdepth=2) if p.endswith("/results.json")
            }
        except FileNotFoundError:
            continue
    for orphan in sorted(present - trained_all):
        logger.warning("%s: results with NO training record (skipped): %s", ns, orphan)
    trained = sorted(scoped | (present & trained_all))
    logger.info("%s: %d runs in scope (%d with results present)", ns, len(trained), len(present & trained_all))
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verify_run, fs, ns, subdir, expected_tasks, has_provenance, r) for r in trained]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            rows.append(fut.result())
            if i % 40 == 0:
                logger.info("%s: %d/%d", ns, i, len(trained))
    return sorted(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--namespaces", nargs="+", default=list(NAMESPACES), choices=list(NAMESPACES))
    ap.add_argument("--workers", type=int, default=4, help="Keep low: MMLU documents are ~228 MB each.")
    ap.add_argument(
        "--require-complete-coverage",
        action="store_true",
        help="MISSING evals also fail the exit code (else they only warn).",
    )
    args = ap.parse_args()

    fs = fsspec.filesystem("gcs", use_listings_cache=False)
    failed = False
    for ns in args.namespaces:
        rows = verify_namespace(fs, ns, args.workers)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["run", "verdict", "detail", "final_step", "bucket"])
        w.writerows(rows)
        with fs.open(f"{OUT_PREFIX}/{ns}.csv", "w") as fh:
            fh.write(buf.getvalue())
        counts: dict[str, int] = {}
        for r in rows:
            counts[r[1]] = counts.get(r[1], 0) + 1
        logger.info("%s VERDICTS: %s", ns, counts)
        for r in rows:
            if r[1] in ("STALE", "INCOMPLETE", "UNPROVEN"):
                logger.warning("  %s %s: %s", r[1], r[0], r[2])
                failed = True
            elif r[1] == "MISSING" and args.require_complete_coverage:
                failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
