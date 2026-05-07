# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the WARC-scaling sweep results to CSV(s).

Reads per-run summary.json from BOTH the warc-scaling prefix (N≤2000, tag
`expWARC_natural`) and the fixed-model 3000-WARC prefix (tag `expFM_natural`).
Filters out non-natural-epoching variants (expB_T20T, expC_T33T, etc.) so the
exported scaling curves are reproducible under one epoching policy.

For older fixed-model runs that pre-date LIMA being in the validation set, the
LIMA sidecar JSONs (FM_LIMA_SIDECAR_PREFIX) are merged into the summary's eval
dict so eval/lima/loss is populated for N=3000 rows too. Mirrors the merge
logic in plot_warc_scaling_sweep._load_fm_lima_sidecar.

Outputs:
    scratch/exports/warc_scaling_streamlined.csv  (analysis-focused, 11 cols)
    scratch/exports/warc_scaling_complete.csv     (reproducibility, 25 cols)

Both rows are sorted by (method, warcs, parameters, tokens) so each method/warc
group reads top-to-bottom by scale.

Usage:
    uv run python experiments/scaling_law_sweeps/export_csv.py
    uv run python experiments/scaling_law_sweeps/export_csv.py --refresh
        # adds gcloud storage cp from all three GCS prefixes first
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

WARC_SCALING_PREFIX = "gs://marin-us-central1/metadata/data_curation_warc_scaling_results/"
# 3000-WARC fixed-model sweep (natural epoching, tag `expFM_natural`). Match
# what plot_warc_scaling_sweep.FM_RESULTS_PREFIX uses; the older
# data_curation_isoflop_results/ prefix carries chinchilla-style fixed-token
# variants (expB_T20T, expC_T33T) which we explicitly do NOT want in the
# scaling-law CSV — natural epoching only.
FIXED_MODEL_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"
# LIMA sidecar: many older fixed-model (N=3000) runs were evaluated before LIMA
# was added to the validation suite. Sidecar JSONs hold eval/lima/{loss,bpb}
# keyed by run_name and are merged into the main summary's eval dict at load
# time so cross-N analyses see LIMA at N=3000 too. Mirrors the merge logic in
# plot_warc_scaling_sweep._load_fm_lima_sidecar / _merge_sidecar_into_summary.
FM_LIMA_SIDECAR_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results/"

# Only keep natural-epoching tags (drops expB_T20T/expC_T33T fixed-token grids).
NATURAL_EPOCHING_TAGS = {"expWARC_natural", "expFM_natural"}

LOCAL_WARC_DIR = Path("scratch/audit_summaries")
LOCAL_FM_DIR = Path("scratch/fm_summaries")
LOCAL_FM_LIMA_SIDECAR_DIR = Path("scratch/fm_lima_sidecar")

OUT_DIR = Path("scratch/exports")
OUT_STREAMLINED = OUT_DIR / "warc_scaling_streamlined.csv"
OUT_COMPLETE = OUT_DIR / "warc_scaling_complete.csv"

# Canonical method bases included in the cross-method comparison.
# llm_curated_bos_fixed → llm_curated, nemotron_full_bos_fixed → nemotron_full.
# llm_curated_dedup is kept as its own base (NOT folded into llm_curated) so
# the dedup A/B can be read off the CSV directly.
CANONICAL_METHODS = {"dclm", "nemotron_full", "llm_curated", "resiliparse", "llm_curated_dedup"}

# Suffix variants we should NOT pick up as separate rows when scanning the
# warc-scaling prefix — they're rescue/retry copies of canonical cells. The
# pattern is intentionally specific (literal "v3", not "v\d+") because the FM
# (3000-WARC) sweep run names end in "-v1"/"-v4" as TPU type markers and we
# don't want to accidentally exclude those.
SUFFIX_RE = re.compile(r"-(retry\d+|v3|mem|memv\d+|c2|recovery)$")


def _normalize_method(method_name: str) -> tuple[str, int | None]:
    """Return `(method_base, warcs_or_None)` from a `method.name` string.

    `method.sampled_warcs` is the canonical source for WARC count; this just
    normalizes the human-readable method base name across schemas:
        - "dclm_500" → ("dclm", 500)
        - "dclm" → ("dclm", None)
        - "llm_curated_bos_fixed" → ("llm_curated", None)
        - "nemotron_full_bos_fixed_2000" → ("nemotron_full", 2000)
    """
    name = method_name
    # Strip trailing _<digits> (warc-scaling subsample suffix).
    m = re.match(r"^(.*?)_(\d+)$", name)
    if m:
        name = m.group(1)
        warcs = int(m.group(2))
    else:
        warcs = None
    # Strip "_bos_fixed" (FM sweep canonical naming).
    name = re.sub(r"_bos_fixed$", "", name)
    return name, warcs


def _refresh_local(prefix: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["gcloud", "storage", "cp", "-r", f"{prefix}*.json", str(local_dir) + "/"]
    subprocess.run(cmd, check=False, capture_output=True, timeout=120)


def _load_fm_lima_sidecar(local_dir: Path) -> dict[str, dict]:
    """Load LIMA sidecar JSONs keyed by `run_name`.

    Each sidecar carries `eval/lima/loss` and `eval/lima/bpb` for older
    fixed-model (N=3000) runs that were evaluated before LIMA was in the
    validation set. Mirrors `plot_warc_scaling_sweep._load_fm_lima_sidecar`.
    """
    sidecars: dict[str, dict] = {}
    if not local_dir.exists():
        return sidecars
    for fname in sorted(os.listdir(local_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(local_dir / fname) as f:
                sd = json.load(f)
        except Exception:
            logger.warning("Failed to parse sidecar %s; skipping", fname)
            continue
        run_name = sd.get("run_name")
        if not run_name:
            continue
        metrics = {k: sd[k] for k in ("eval/lima/loss", "eval/lima/bpb") if sd.get(k) is not None}
        if metrics:
            sidecars[run_name] = metrics
    logger.info("Loaded %d FM LIMA sidecars from %s", len(sidecars), local_dir)
    return sidecars


def _merge_sidecar_into_summary(summary: dict, sidecars: dict[str, dict]) -> None:
    """Fill missing eval/lima/* keys from sidecar; never overwrite existing."""
    plan = summary.get("plan", {}) or {}
    run_name = plan.get("run_name") or plan.get("run_name_core")
    if not run_name or run_name not in sidecars:
        return
    eval_d = summary.get("eval")
    if eval_d is None:
        eval_d = {}
        summary["eval"] = eval_d
    for k, v in sidecars[run_name].items():
        eval_d.setdefault(k, v)


def _iter_summaries(local_dir: Path):
    if not local_dir.exists():
        return
    for fname in sorted(os.listdir(local_dir)):
        if not (fname.startswith("curation-") and fname.endswith(".json")):
            continue
        base = fname[: -len(".json")]
        if SUFFIX_RE.search(base):
            continue
        path = local_dir / fname
        try:
            with open(path) as f:
                yield base, json.load(f)
        except Exception:
            logger.warning("Failed to parse %s; skipping", path)


def _row_from_summary(base: str, j: dict) -> tuple[dict, dict] | None:
    """Build (streamlined_row, complete_row) from a single summary.

    Returns None if the summary should be excluded (non-canonical method, etc).
    """
    plan = j.get("plan", {})
    method = j.get("method", {})
    model = j.get("model", {})
    tokens_blk = j.get("tokens", {})
    run = j.get("run", {})
    eval_blk = j.get("eval", {})

    method_full = method.get("name", "")
    method_base, parsed_warcs = _normalize_method(method_full)
    if method_base not in CANONICAL_METHODS:
        return None

    # Natural-epoching only: drop expB_T20T/expC_T33T fixed-token variants so
    # the scaling curves are reproducible under one epoching policy. Keeps
    # expWARC_natural (N≤2000) and expFM_natural (N=3000).
    experiment_tag = plan.get("experiment_tag", "")
    if experiment_tag not in NATURAL_EPOCHING_TAGS:
        return None

    # BOS-fix filter at N=3000: the FM (expFM_natural) sweep contains both
    # pre-BOS-fix and BOS-fixed runs for llm_curated/nemotron_full. Pre-fix
    # runs sit ~0.06–0.10 nats off in lima — pooling them with the BOS-fixed-
    # only N≤2000 sweep would silently bias scaling-law fits. Keep only the
    # `_bos_fixed` variants for these two methods. dclm/resiliparse are
    # untouched (no BOS rebuild was needed).
    if experiment_tag == "expFM_natural" and method_full in ("llm_curated", "nemotron_full"):
        return None

    # method.sampled_warcs is the canonical source for warcs count.
    warcs = method.get("sampled_warcs", parsed_warcs)
    if warcs is None:
        return None

    params = model.get("total_trainable_params") or 0
    tokens_trained = tokens_blk.get("tokens_trained") or 0
    unique_tokens = method.get("d_obs_tokens") or 0
    epochs = tokens_blk.get("effective_epochs", "")
    flops_target = plan.get("budget_flops", "")
    flops_actual = 6 * params * tokens_trained if (params and tokens_trained) else ""
    lima_loss = eval_blk.get("eval/lima/loss", "")

    streamlined = {
        "method": method_base,
        "warcs": warcs,
        "experiment_tag": plan.get("experiment_tag", ""),
        "hidden_dim": plan.get("hidden_dim", ""),
        "unique_tokens": unique_tokens,
        "tokens": tokens_trained,
        "epochs": epochs,
        "parameters": params,
        "flops_target": flops_target,
        "flops_actual": flops_actual,
        "eval_lima_loss": lima_loss,
    }
    complete = {
        "run_name": plan.get("run_name_core", base),
        "method": method_base,
        "warcs": warcs,
        "experiment_tag": plan.get("experiment_tag", ""),
        "hidden_dim": plan.get("hidden_dim", ""),
        "num_layers": plan.get("num_layers", ""),
        "num_heads": plan.get("num_heads", ""),
        "intermediate_dim": plan.get("intermediate_dim", ""),
        "parameters": params,
        "batch_size": plan.get("batch_size", ""),
        "seq_len": plan.get("seq_len", ""),
        "train_steps": plan.get("train_steps", ""),
        "tokenizer": method.get("tokenizer", ""),
        "learning_rate": plan.get("learning_rate", ""),
        "unique_tokens": unique_tokens,
        "tokens": tokens_trained,
        "epochs": epochs,
        "flops_target": flops_target,
        "flops_actual": flops_actual,
        "eval_lima_loss": lima_loss,
        "eval_loss": eval_blk.get("eval/loss", ""),
        "paloma_macro_loss": eval_blk.get("eval/paloma/macro_loss", ""),
        "eval_bpb": eval_blk.get("eval/bpb", ""),
        "region": run.get("region", ""),
        "completed_at": run.get("completed_at", ""),
    }
    return streamlined, complete


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="gcloud-cp summaries from GCS into the local cache before exporting.",
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for CSVs.")
    args = parser.parse_args(argv)

    if args.refresh:
        logger.info("Refreshing local cache from GCS...")
        _refresh_local(WARC_SCALING_PREFIX, LOCAL_WARC_DIR)
        _refresh_local(FIXED_MODEL_PREFIX, LOCAL_FM_DIR)
        _refresh_local(FM_LIMA_SIDECAR_PREFIX, LOCAL_FM_LIMA_SIDECAR_DIR)

    sidecars = _load_fm_lima_sidecar(LOCAL_FM_LIMA_SIDECAR_DIR)

    streamlined_rows: list[dict] = []
    complete_rows: list[dict] = []
    seen_run_names: set[str] = set()
    lima_filled_from_sidecar = 0

    for source_dir, merge_sidecar in ((LOCAL_WARC_DIR, False), (LOCAL_FM_DIR, True)):
        for base, j in _iter_summaries(source_dir):
            if merge_sidecar:
                eval_before = (j.get("eval") or {}).get("eval/lima/loss")
                _merge_sidecar_into_summary(j, sidecars)
                if eval_before is None and (j.get("eval") or {}).get("eval/lima/loss") is not None:
                    lima_filled_from_sidecar += 1
            row = _row_from_summary(base, j)
            if row is None:
                continue
            streamlined, complete = row
            if complete["run_name"] in seen_run_names:
                continue
            seen_run_names.add(complete["run_name"])
            streamlined_rows.append(streamlined)
            complete_rows.append(complete)
    logger.info("Filled eval/lima/loss from sidecar for %d FM rows", lima_filled_from_sidecar)

    # Sort: method, warcs, parameters, tokens.
    def sort_key(r):
        return (
            str(r["method"]),
            int(r["warcs"]) if r["warcs"] not in ("", None) else 0,
            int(r["parameters"]) if r["parameters"] not in ("", None) else 0,
            int(r["tokens"]) if r["tokens"] not in ("", None) else 0,
        )

    streamlined_rows.sort(key=sort_key)
    complete_rows.sort(key=sort_key)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_streamlined = out_dir / "warc_scaling_streamlined.csv"
    out_complete = out_dir / "warc_scaling_complete.csv"

    with open(out_streamlined, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(streamlined_rows[0].keys()))
        w.writeheader()
        w.writerows(streamlined_rows)
    with open(out_complete, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(complete_rows[0].keys()))
        w.writeheader()
        w.writerows(complete_rows)

    logger.info("Streamlined: %d rows → %s", len(streamlined_rows), out_streamlined)
    logger.info("Complete:    %d rows → %s", len(complete_rows), out_complete)
    # Per-method/warc summary so the user can sanity-check coverage. Also count
    # how many rows are missing eval/lima/loss — those won't appear on lima plots.
    from collections import Counter

    cov = Counter((r["method"], r["warcs"]) for r in streamlined_rows)
    cov_missing_lima = Counter((r["method"], r["warcs"]) for r in streamlined_rows if r["eval_lima_loss"] in ("", None))
    logger.info("Coverage (rows / missing eval_lima_loss):")
    for (m, n), c in sorted(cov.items()):
        miss = cov_missing_lima.get((m, n), 0)
        logger.info("  %-15s warcs=%5s  %d rows  (%d missing lima)", m, n, c, miss)


if __name__ == "__main__":
    main()
