"""Copy delphi-sweep fixed-model summaries for LC and Resiliparse → ExpC results dir.

Why this is sound: for data-rich methods (target_epochs < 1 at T=33T), training
naturally on D_obs for T_exp tokens is statistically equivalent to drawing T_exp
i.i.d. tokens from D_proj — the simulation that ExpC for LC/Res asks for. So
delphi-sweep fixed-model runs at matching (method, hidden, layers, batch, budget)
ARE valid ExpC data points; they just need the ExpC label.

We ONLY copy delphi-sweep canonical runs — those with
  - hidden_dim ∈ TARGET_HIDDEN_SIZES (512, 1536, 2432, 3328, 3584)
  - budget ∈ BUDGETS (the canonical 7-point grid)
  - batch_size = the heuristic's pick for (hidden, budget) (no -canonical-suffix
    variants, no batch-size sweeps)
This is exactly what `fixed_model_plan.enumerate_fixed_model_plans([LC, Res])`
produces. Other fixed-model artifacts (different batch sizes, off-grid budgets,
runs with -canonical/-dev suffixes) are skipped.

Uses `gcloud storage cp` via subprocess to avoid gcsfs SSL issues on this host.

Run with:
    uv run python experiments/scaling_law_sweeps/copy_fixed_model_to_expc.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from experiments.scaling_law_sweeps import curation_plan, fixed_model_plan
from experiments.scaling_law_sweeps.curation_plan import METHODS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIXED_MODEL_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results"
FIXED_MODEL_LIMA_SIDECAR_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results"
EXPC_PREFIX = "gs://marin-us-central1/metadata/data_curation_isoflop_results"


def _gcs_exists(path: str) -> bool:
    """True iff the GCS object at `path` exists."""
    r = subprocess.run(
        ["gcloud", "storage", "ls", path],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0 and path in r.stdout


def _gcs_read_json(path: str) -> dict:
    r = subprocess.run(
        ["gcloud", "storage", "cat", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)


def _gcs_write_json(path: str, obj: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(obj, f)
        tmp_path = f.name
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", tmp_path, path],
            check=True,
            capture_output=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _rewrite_summary(summary: dict, expc_run_name: str, expc_tag: str, t_target: float) -> dict:
    """Return a copy of `summary` with ExpC labels swapped in.

    Touches only metadata fields. Eval results and plan hyperparameters are
    preserved exactly — the actual training run is identical, this just
    relabels for plot consumption.
    """
    out = json.loads(json.dumps(summary))  # deep copy
    out["plan"]["experiment_tag"] = expc_tag
    out["plan"]["run_name"] = expc_run_name
    out["plan"]["run_name_core"] = expc_run_name
    out["plan"]["t_target"] = float(t_target)
    return out


def _merge_lima_sidecar(summary: dict, fm_run_name: str) -> tuple[dict, bool]:
    """Merge the post-hoc LIMA sidecar's eval/lima/* keys into summary['eval'].

    Older fixed-model runs were trained BEFORE LIMA was added to the in-training
    mixture, so their summaries lack eval/lima/*. The post-hoc evaluator
    (`launch_lima_eval.py` / `run_lima_eval_standalone.py`) writes a sidecar
    file at `data_curation_fixed_model_lima_results/{run_name}.json` containing
    `eval/lima/bpb` and `eval/lima/loss` (plus `eval/loss`). The plotter merges
    these in at plot time; we do the same here so the ExpC summary stands alone.

    Returns the (possibly merged) summary and a bool indicating whether a merge
    actually happened (False = LIMA was already present, or sidecar didn't
    exist, or both — diagnostic only).
    """
    eval_block = summary.get("eval") or {}
    if "eval/lima/bpb" in eval_block or "eval/lima/loss" in eval_block:
        # LIMA is already in-summary (newer FM run, trained with LIMA in mixture).
        return summary, False

    sidecar_path = f"{FIXED_MODEL_LIMA_SIDECAR_PREFIX}/{fm_run_name}.json"
    if not _gcs_exists(sidecar_path):
        # No sidecar — this run never got post-hoc LIMA eval. Summary is
        # legitimately LIMA-less; downstream plotter will skip it for that metric.
        return summary, False

    try:
        sidecar = _gcs_read_json(sidecar_path)
    except Exception as e:
        logger.warning("Failed reading LIMA sidecar %s: %s", sidecar_path, e)
        return summary, False

    out = json.loads(json.dumps(summary))  # deep copy
    out.setdefault("eval", {})
    merged = 0
    for k, v in sidecar.items():
        if k.startswith("eval/lima/") and v is not None:
            out["eval"][k] = v
            merged += 1
    if merged == 0:
        return summary, False
    logger.info("merged %d LIMA keys from sidecar for %s", merged, fm_run_name)
    return out, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied; don't write.")
    args = parser.parse_args()

    # Build the canonical delphi-sweep grid for the two data-rich methods.
    methods = [METHODS["llm_curated_bos_fixed"], METHODS["resiliparse"]]
    fm_plans = fixed_model_plan.enumerate_fixed_model_plans(methods)
    fm_run_names = {p.run_name_core for p in fm_plans}
    logger.info("Delphi-sweep canonical FM plans: %d (%d methods × %d hidden_sizes × %d budgets)",
                len(fm_plans),
                len(methods),
                len(fixed_model_plan.TARGET_HIDDEN_SIZES),
                len(curation_plan.BUDGETS))

    # Now enumerate the corresponding ExpC plans for the same two methods.
    expc_plans = curation_plan.enumerate_plans(methods, [("C", curation_plan.DEFAULT_T_TARGET_C)])
    logger.info("ExpC plans for LC + Res: %d", len(expc_plans))

    copied = 0
    skipped_dest_exists = 0
    skipped_no_fm = 0
    skipped_not_delphi = 0
    lima_sidecar_merged = 0
    failed = 0

    for plan in expc_plans:
        expc_run_name = plan.run_name_core
        fm_run_name = expc_run_name.replace(plan.experiment_tag, "expFM_natural")

        # Filter: only delphi-sweep canonical runs. Off-grid candidates (e.g.
        # hidden=640 from the IsoFLOP grid that aren't in TARGET_HIDDEN_SIZES)
        # never had a delphi-sweep run, so there's nothing to copy.
        if fm_run_name not in fm_run_names:
            skipped_not_delphi += 1
            continue

        fm_path = f"{FIXED_MODEL_PREFIX}/{fm_run_name}.json"
        expc_path = f"{EXPC_PREFIX}/{expc_run_name}.json"

        if not _gcs_exists(fm_path):
            skipped_no_fm += 1
            logger.debug("FM summary missing (not yet completed): %s", fm_run_name)
            continue

        if _gcs_exists(expc_path):
            skipped_dest_exists += 1
            logger.debug("Dest exists, skip: %s", expc_path)
            continue

        try:
            fm_summary = _gcs_read_json(fm_path)
            fm_summary, did_merge = _merge_lima_sidecar(fm_summary, fm_run_name)
            if did_merge:
                lima_sidecar_merged += 1
            expc_summary = _rewrite_summary(
                fm_summary,
                expc_run_name=expc_run_name,
                expc_tag=plan.experiment_tag,
                t_target=plan.t_target,
            )
            if args.dry_run:
                lima_present = "eval/lima/bpb" in (expc_summary.get("eval") or {})
                logger.info("[DRY-RUN would copy] %s -> %s (T_target %.1e -> %.1e, LIMA %s)",
                            fm_run_name, expc_run_name,
                            fm_summary["plan"].get("t_target", float("nan")),
                            plan.t_target,
                            "present" if lima_present else "missing")
            else:
                _gcs_write_json(expc_path, expc_summary)
                logger.info("[%d] copied %s -> %s%s",
                            copied + 1, fm_run_name, expc_run_name,
                            " (+LIMA from sidecar)" if did_merge else "")
            copied += 1
        except Exception as e:
            failed += 1
            logger.exception("FAILED to copy %s: %s", fm_run_name, e)

    logger.info("=" * 60)
    logger.info("Migration summary:")
    logger.info("  copied%s:                   %d", " (dry-run)" if args.dry_run else "", copied)
    logger.info("  of which LIMA from sidecar: %d", lima_sidecar_merged)
    logger.info("  skipped (dest exists):      %d", skipped_dest_exists)
    logger.info("  skipped (FM not completed): %d", skipped_no_fm)
    logger.info("  skipped (off delphi grid):  %d", skipped_not_delphi)
    logger.info("  failed:                     %d", failed)
    logger.info("  total ExpC plans considered: %d", len(expc_plans))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
