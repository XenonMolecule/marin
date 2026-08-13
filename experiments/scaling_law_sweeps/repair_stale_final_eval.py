# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find and repair per-run summaries whose `eval` block is from a NON-final eval cycle.

`run_curation_train_standalone._fetch_final_eval_from_wandb` used to validate freshness with
`run.summary["_step"]` while reading the `eval/*` keys. Those move on different cadences:
`_step` advances every training log, `eval/*` only when an eval cycle runs. A child that queried
its own W&B run after training but before the final eval synced saw `_step == train_steps-1`
(guard passes) alongside the PREVIOUS eval's values (wrong data). Measured on
`dclm_10k_mix 3e+17-d512`: the summary recorded the step-10000 eval (uncheatable 1.4265) when
the true step-13137 value was 1.3166 — a 0.11 BPB error that manufactured visible spikes in the
scaling curves.

Ground truth is `{output_path}/checkpoints/eval_metrics.jsonl`, which carries one row per eval
cycle WITH its own `step`. This tool compares each summary's eval block against the row at the
expected final step and rewrites the summary when they disagree.

    # report only
    python -m experiments.scaling_law_sweeps.repair_stale_final_eval --methods dclm_10k_mix ...
    # rewrite (originals copied to a sibling *_prerepair prefix, never beside the results)
    python -m experiments.scaling_law_sweeps.repair_stale_final_eval --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import fsspec

logger = logging.getLogger(__name__)

RESULTS = "gs://marin-us-central1/metadata/data_curation_10k_natural_results"
# Backups go in a SIBLING prefix, never beside the results. Writing `<run>.json.bak` into the
# results prefix made every downstream counter that globs `*.json` double-count the repaired
# runs -- the sweep monitor reported 76/76 complete when the truth was 55/76.
BACKUPS = "gs://marin-us-central1/metadata/data_curation_10k_natural_results_prerepair"
# Levanter logs the final eval at train_steps - 1; allow the same slack the writer used.
STEP_SLACK = 10
_STEM = re.compile(r"^curation-(?P<m>.+?)-expFM_natural-\S+?-d\d+-L\d+-B\d+$")
# The metric the scaling plots key on; also the one where the error was visible.
PROBE = "eval/uncheatable_eval/macro_bpb"


def _is_timing_key(key: str) -> bool:
    """Wall-clock keys (`eval/total_time`, `eval/loading_time`).

    These are measurements of the eval RUN, not of the model, so the W&B summary and
    `eval_metrics.jsonl` disagree on them even for the same cycle. Comparing them flagged 36
    runs as stale whose model metrics were all identical -- a false positive that would have
    churned every summary in the sweep.
    """
    return key.endswith("_time")


def _eval_rows(fs, output_path: str) -> list[dict]:
    path = f"{output_path.rstrip('/')}/checkpoints/eval_metrics.jsonl"
    try:
        return [json.loads(x) for x in fs.cat_file(path).decode().splitlines() if x.strip()]
    except Exception:
        return []


def audit_one(args) -> dict | None:
    fs, path = args
    try:
        summary = json.loads(fs.cat_file(path))
    except Exception:
        return None
    plan, run = summary.get("plan", {}), summary.get("run", {})
    expected = int(plan.get("train_steps", 0)) - 1
    output_path = run.get("output_path")
    if not output_path or expected <= 0:
        return None
    rows = _eval_rows(fs, output_path)
    if not rows:
        return {"path": path, "status": "no_eval_file"}
    final = max(rows, key=lambda r: int(r.get("step", -1)))
    if int(final.get("step", -1)) < expected - STEP_SLACK:
        # The run's own eval file never got a final row -- nothing to repair from.
        return {"path": path, "status": "file_lacks_final", "file_step": final.get("step"), "expected": expected}
    # Compare EVERY eval/* key, not just one probe. W&B summary keys are written per metric and
    # can land from different eval cycles, so a single-probe check passes whenever that one metric
    # happens to be current while others are stale. Observed the all-keys case on
    # high_quality_10k_mix 3e+19-d1536: all 62 keys came from step 35000 instead of 38013, which
    # inflated Paloma by +0.031 BPB and uncheatable macro_loss by +0.103 -- a visible scaling-curve
    # spike. Any mismatched key condemns the whole block; a summary mixing cycles is not usable.
    evals = summary.get("eval", {})
    pairs = [
        (k, float(v), float(final[k]))
        for k, v in evals.items()
        if k.startswith("eval/")
        and not _is_timing_key(k)
        and isinstance(v, (int, float))
        and isinstance(final.get(k), (int, float))
    ]
    if not pairs:
        return {"path": path, "status": "probe_missing"}
    bad = [(k, v, w) for k, v, w in pairs if abs(v - w) >= 1e-9]
    if not bad:
        return {"path": path, "status": "ok"}
    worst = max(bad, key=lambda x: abs(x[1] - x[2]))
    return {
        "path": path,
        "status": "STALE",
        "expected": expected,
        "n_stale_keys": len(bad),
        "n_keys": len(pairs),
        "worst_key": worst[0],
        "worst_summary": worst[1],
        "worst_final": worst[2],
        "worst_delta": worst[2] - worst[1],
        "final_step": int(final.get("step", -1)),
        "final_row": final,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", nargs="+", default=None, help="Restrict to these method names.")
    p.add_argument("--apply", action="store_true", help="Rewrite stale summaries (originals copied to BACKUPS).")
    args = p.parse_args()

    fs = fsspec.filesystem("gs")
    paths = [
        p_
        for p_ in fs.ls(RESULTS, detail=False)
        if p_.endswith(".json")
        and (m := _STEM.match(p_.rsplit("/", 1)[-1].removesuffix(".json")))
        and (args.methods is None or m["m"] in args.methods)
    ]
    logger.info("auditing %d summaries", len(paths))
    with ThreadPoolExecutor(32) as ex:
        results = [r for r in ex.map(audit_one, [(fs, x) for x in paths]) if r]

    by_status: dict[str, list] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r)
    for st, rs in sorted(by_status.items()):
        logger.info("%-16s %d", st, len(rs))

    stale = by_status.get("STALE", [])
    for r in sorted(stale, key=lambda x: -abs(x["worst_delta"]))[:20]:
        logger.info(
            "  STALE %s: %d/%d metric keys stale; final@step%d; worst %s %.4f -> %.4f (%+.4f)",
            r["path"].rsplit("/", 1)[-1][9:60],
            r["n_stale_keys"],
            r["n_keys"],
            r["final_step"],
            r["worst_key"],
            r["worst_summary"],
            r["worst_final"],
            r["worst_delta"],
        )
    if not args.apply:
        logger.info("dry run -- pass --apply to rewrite %d stale summaries", len(stale))
        return

    def _fix(r):
        summary = json.loads(fs.cat_file(r["path"]))
        bak = f"{BACKUPS}/{r['path'].rsplit('/', 1)[-1]}"
        with fs.open(bak, "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        # Replace the whole eval block with the final row's eval/* keys, so every metric in the
        # summary comes from one consistent eval cycle rather than a mix of cycles.
        summary["eval"] = {k: v for k, v in r["final_row"].items() if k.startswith("eval/")}
        summary.setdefault("repair", {})["final_eval_step"] = r["final_step"]
        summary["repair"]["reason"] = "summary held a non-final eval cycle (wandb _step vs eval/* cadence race)"
        with fs.open(r["path"], "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        return r["path"]

    with ThreadPoolExecutor(16) as ex:
        fixed = list(ex.map(_fix, stale))
    logger.info("repaired %d summaries (originals copied to %s)", len(fixed), BACKUPS)


if __name__ == "__main__":
    main()
