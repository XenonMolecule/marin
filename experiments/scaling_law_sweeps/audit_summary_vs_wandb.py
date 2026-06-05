# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit per-run summary JSONs against W&B and (optionally) correct them.

The fixed-model sweep's plots (`plot_fixed_model_sweep.py`) read each run's
final eval metrics from a per-run summary JSON written by
`run_curation_train_standalone.py`. Those summaries source `eval/*` from the
last row of `eval_metrics.jsonl`, which in some cases drifts from W&B's
canonical run summary (partial flush, post-jsonl eval cycles, preemption).
When the local summary's loss disagrees with W&B, the plot silently shows
the wrong number.

This script compares each summary's `eval/*` to the corresponding W&B run's
`run.summary` and either reports mismatches (default, dryrun) or rewrites the
summary file to match W&B (`--apply`).

Dryrun (default — read-only, writes a JSON report only):
    uv run python experiments/scaling_law_sweeps/audit_summary_vs_wandb.py

Apply (rewrites local summary files, no GCS push):
    uv run python experiments/scaling_law_sweeps/audit_summary_vs_wandb.py --apply

Apply + push corrected files back to the GCS results bucket:
    uv run python experiments/scaling_law_sweeps/audit_summary_vs_wandb.py --apply --upload-gcs

Spot-check a single run / pattern:
    uv run python experiments/scaling_law_sweeps/audit_summary_vs_wandb.py \\
        --filter resiliparse-expFM_natural-2e+21-d1536
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from pathlib import Path

import wandb

DEFAULT_SUMMARY_DIR = Path("scratch/fm_summaries")
DEFAULT_GCS_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"
DEFAULT_WANDB_PROJECT = "marin-community/marin"
# 1e-4 relative tolerance: tighter than visual-plot precision (4 decimal places
# in hover text), looser than float32 round-trip noise. Mismatches above this
# represent real divergence, not float-format jitter.
DEFAULT_TOLERANCE = 1e-4

logger = logging.getLogger(__name__)


def _is_numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _close(a, b, tol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if not (_is_numeric(a) and _is_numeric(b)):
        return a == b
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol * 1e-3)


def _wandb_eval_dict(run) -> dict:
    """Pull eval/* keys from a run's W&B summary, casting numbers to float."""
    out: dict = {}
    for k, v in run.summary.items():
        if not k.startswith("eval/"):
            continue
        if v is None:
            continue
        if _is_numeric(v):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def audit_one(summary_path: Path, api: wandb.Api, project: str, tol: float) -> dict:
    with summary_path.open() as f:
        summary = json.load(f)
    run_name = summary.get("plan", {}).get("run_name") or summary_path.stem
    record: dict = {
        "run_name": run_name,
        "summary_path": str(summary_path),
        "status": None,
        "mismatches": [],
        "missing_in_summary": [],
        "missing_in_wandb": [],
    }
    try:
        run = api.run(f"{project}/{run_name}")
    except Exception as e:
        record["status"] = f"wandb_lookup_failed: {e}"
        return record
    record["wandb_state"] = run.state

    summary_eval = summary.get("eval") or {}
    wandb_eval = _wandb_eval_dict(run)
    summary_keys = {k for k in summary_eval.keys() if k.startswith("eval/")}
    wandb_keys = set(wandb_eval.keys())

    record["missing_in_summary"] = sorted(wandb_keys - summary_keys)
    record["missing_in_wandb"] = sorted(summary_keys - wandb_keys)
    for k in sorted(summary_keys & wandb_keys):
        sv = summary_eval.get(k)
        wv = wandb_eval.get(k)
        if not _close(sv, wv, tol):
            record["mismatches"].append({"key": k, "summary": sv, "wandb": wv})

    record["wandb_eval"] = wandb_eval  # consumed by --apply; stripped from on-disk report
    has_drift = bool(record["mismatches"] or record["missing_in_summary"] or record["missing_in_wandb"])
    record["status"] = "mismatch" if has_drift else "ok"
    return record


def fix_one(record: dict, summary_path: Path) -> None:
    """Rewrite summary['eval'] using W&B's eval dict.

    Behavior:
      - Replace each eval/* key with the W&B value.
      - Preserve any non eval/* keys that previously lived under summary['eval']
        (defensive — current writer only puts eval/* there, but don't drop
        unknown keys silently).
      - Add eval/* keys present in W&B but missing from summary.
    """
    with summary_path.open() as f:
        summary = json.load(f)
    new_eval: dict = dict(record["wandb_eval"])
    for k, v in (summary.get("eval") or {}).items():
        if k not in new_eval:
            new_eval[k] = v
    summary["eval"] = new_eval
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)


def upload_to_gcs(summary_path: Path, gcs_prefix: str) -> None:
    """Upload via `gcloud storage cp` rather than fsspec/gcsfs.

    Reason: gcsfs depends on aiohttp's SSL stack, which fails on some local
    machines with `CERTIFICATE_VERIFY_FAILED` even when gcloud's own auth is
    fine. Shelling out to gcloud sidesteps the cert issue entirely and uses
    whatever credentials the user has configured.
    """
    target = f"{gcs_prefix.rstrip('/')}/{summary_path.name}"
    subprocess.run(
        ["gcloud", "storage", "cp", str(summary_path), target],
        check=True,
        capture_output=True,
    )


def _format_mismatch_line(m: dict) -> str:
    sv, wv = m["summary"], m["wandb"]
    if _is_numeric(sv) and _is_numeric(wv):
        delta = float(wv) - float(sv)
        rel = (delta / float(sv)) if sv else float("inf")
        return f"    {m['key']}: summary={sv:.6f} wandb={wv:.6f} delta={delta:+.6f} ({rel:+.2%})"
    return f"    {m['key']}: summary={sv} wandb={wv}"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--gcs-prefix", default=DEFAULT_GCS_PREFIX)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite local summary files for runs whose eval/* drifted from W&B. "
        "Without this flag, the script only audits and writes a report.",
    )
    parser.add_argument(
        "--upload-gcs",
        action="store_true",
        help="When --apply is set, also push corrected summaries back to --gcs-prefix.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("scratch/fm_audit_report.json"),
        help="Where to write the per-run audit report (always written).",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Substring filter on summary filename (e.g. 'expFM_natural-2e+21').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap number of summaries audited (0 = all). Useful for spot checks.",
    )
    args = parser.parse_args(argv)

    files = sorted(args.summary_dir.glob("*.json"))
    if args.filter:
        files = [p for p in files if args.filter in p.name]
    if args.limit:
        files = files[: args.limit]
    if not files:
        logger.error("No summaries found under %s (filter=%r)", args.summary_dir, args.filter)
        return

    mode = "APPLY" if args.apply else "DRYRUN"
    logger.info("[%s] Auditing %d summaries against %s", mode, len(files), args.wandb_project)

    api = wandb.Api()
    records: list[dict] = []
    for i, p in enumerate(files):
        try:
            rec = audit_one(p, api, args.wandb_project, args.tolerance)
        except Exception as e:
            rec = {"run_name": p.stem, "summary_path": str(p), "status": f"audit_failed: {e}"}
        records.append(rec)
        status = rec["status"]
        prefix = f"[{i + 1:4d}/{len(files)}]"
        if status == "ok":
            logger.info("%s OK       %s", prefix, rec["run_name"])
        elif status == "mismatch":
            n_mis = len(rec.get("mismatches", []))
            n_miss_s = len(rec.get("missing_in_summary", []))
            n_miss_w = len(rec.get("missing_in_wandb", []))
            logger.warning(
                "%s MISMATCH %s — %d differing, %d new-in-wandb, %d only-in-summary",
                prefix,
                rec["run_name"],
                n_mis,
                n_miss_s,
                n_miss_w,
            )
            for m in rec.get("mismatches", [])[:6]:
                logger.warning(_format_mismatch_line(m))
            if n_mis > 6:
                logger.warning("    ...and %d more mismatched keys", n_mis - 6)
        else:
            logger.warning("%s %s — %s", prefix, rec["run_name"], status)

    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_mm = sum(1 for r in records if r["status"] == "mismatch")
    n_other = len(records) - n_ok - n_mm
    logger.info("Totals: %d ok, %d mismatched, %d errored", n_ok, n_mm, n_other)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_records = [{k: v for k, v in r.items() if k != "wandb_eval"} for r in records]
    with args.report.open("w") as f:
        json.dump(
            {
                "mode": mode,
                "wandb_project": args.wandb_project,
                "summary_dir": str(args.summary_dir),
                "tolerance": args.tolerance,
                "n_ok": n_ok,
                "n_mismatched": n_mm,
                "n_errored": n_other,
                "records": report_records,
            },
            f,
            indent=2,
        )
    logger.info("Wrote audit report: %s", args.report)

    if not args.apply:
        if n_mm or n_other:
            logger.info("Dryrun: no files modified. Re-run with --apply to fix.")
        return

    n_fixed = 0
    n_uploaded = 0
    for r in records:
        if r.get("status") != "mismatch":
            continue
        p = Path(r["summary_path"])
        try:
            fix_one(r, p)
            n_fixed += 1
            logger.info("Fixed %s", p)
        except Exception as e:
            logger.error("Failed to fix %s: %s", p, e)
            continue
        if args.upload_gcs:
            try:
                upload_to_gcs(p, args.gcs_prefix)
                n_uploaded += 1
            except Exception as e:
                logger.error("Failed to upload %s to GCS: %s", p, e)
    logger.info(
        "Applied corrections to %d summaries (%d uploaded to GCS=%s)",
        n_fixed,
        n_uploaded,
        args.gcs_prefix if args.upload_gcs else "—",
    )


if __name__ == "__main__":
    main()
