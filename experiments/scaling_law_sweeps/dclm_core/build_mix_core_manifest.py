# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the DCLM Core_v2 checkpoint manifest for the OLMIX-mixture 10k arms.

The mix arms finish over a huge span of wall clock (the 9 smallest cells are 0.18% of the
sweep's compute, the 4 biggest are 64%), so Core evals go out in rolling waves rather than
one batch at the end. This emits the `run_name,region,output_path,hf_dir` CSV that
`launch_10k_manifest.py` consumes, restricted to runs that are ready and not already covered.

Three exclusions, each learned the hard way:

* **already evaluated** -- a scores summary exists at `metadata/data_curation_10k_core_results/`.
  `launch_10k_manifest --skip-existing` also checks this, but doing it here keeps the wave
  size honest so `--wave-size` throttling means what it says.
* **already in flight** -- `--skip-existing` only sees *finished* evals, so a run that is
  queued or mid-eval in an earlier wave looks un-evaluated and would be submitted twice.
  Pass the live child names via `--exclude`.
* **no HF export yet** -- training wrote its results JSON but `hf/step-N` is not there, so
  `_resolve_final_step` would return None and the row would be dropped downstream anyway.

Waves matter for a reason that is not throughput: the HF Hub caps at **1000 requests / 5 min
per token**, and the model load is inherently online (levanter probes gpt2), so a big wave
cold-starting together gets everything rate-limited at once. Keep waves small.

    python -m experiments.scaling_law_sweeps.dclm_core.build_mix_core_manifest \\
        --out experiments/core_eval_manifests/mix_core_wave1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import fsspec

from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import OLMO_BASE_EASY_BPB, QA_RC_MC_BPB

logger = logging.getLogger(__name__)

CANONICAL_OLMO_TASKS = tuple(OLMO_BASE_EASY_BPB) + tuple(QA_RC_MC_BPB)
# Eval scores are written in-region by whichever region the eval child landed in.
SCORE_BUCKETS = (
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-east1",
    "marin-us-east5",
    "marin-us-west4",
    "marin-eu-west4",
)

RESULTS_PREFIX = "marin-us-central1/metadata/data_curation_10k_natural_results"
SCORES_SUBPATH = "metadata/data_curation_10k_core_results/{run_name}_summary.json"
# Each suite writes a different "already done" marker; `--suite` picks which one decides
# completion, so one tool can drive all three passes.
OLMO_BPB_SCORES_SUBPATH = "metadata/olmo_bpb_results/{run_name}/results.json"
# The MMLU pass runs `--merge` into the canonical suite's results.json, so results.json
# ALREADY EXISTS before it starts and cannot signal completion -- keying on it would report
# every run as done and the wave would be empty. The child's `--done-marker` file is the
# only honest signal here.
MMLU_SCORES_SUBPATH = "metadata/olmo_bpb_results/{run_name}/_mmlu.done"
# BLEnD is a third `--merge` pass into the same results.json, so like MMLU it needs its own
# completion marker; results.json existence would report every run as already done.
BLEND_SCORES_SUBPATH = "metadata/olmo_bpb_results/{run_name}/_blend.done"
SUITE_SUBPATHS = {
    "core_v2": SCORES_SUBPATH,
    "olmo_bpb": OLMO_BPB_SCORES_SUBPATH,
    "mmlu": MMLU_SCORES_SUBPATH,
    "blend": BLEND_SCORES_SUBPATH,
}
# EVERY mixture arm, listed explicitly. These used to be two substrings that happened to match
# the newer arms too; once matching became exact that silently dropped 132 runs from coverage,
# so new arms must be added here by hand.
METHOD_TOKENS = (
    "dclm_10k_mix",
    "high_quality_10k_mix",
    "dclm_10k_mix_lambda0p01",
    "high_quality_10k_mix_lambda0p01",
    "dclm_10k_mix_corev2_lambda0p05",
    "dclm_10k_mix_corev2_lambda0p01",
    "high_quality_10k_mix_corev2_lambda0p05",
    "high_quality_10k_mix_corev2_lambda0p01",
    "dclm_10k_mix_blend_lambda0p01",
    "high_quality_10k_mix_blend_lambda0p01",
)
# Arms that are ONLY ever evaluated on BLEnD (user directive 2026-08-05: "for these new runs we
# just need to eval them on Blend not OLMO or DCLM Core"). They are skipped for every other
# suite. Without this they would be swept in automatically, because their names contain
# "dclm_10k_mix"/"high_quality_10k_mix".
BLEND_ONLY_METHODS = ("dclm_10k_mix_blend_lambda0p01", "high_quality_10k_mix_blend_lambda0p01")


def _olmo_canonical_done(fs, path: str) -> bool:
    """True only if `results.json` holds the FULL canonical suite, not just some tasks.

    The MMLU pass runs `--merge`, which CREATES `results.json` when none exists -- so a run that
    never had the canonical 56-task eval ends up with a 4-task file. Treating mere existence as
    "scored" then permanently hides it from the canonical wave: measured, 4 mix runs sat at 4
    tasks while the monitor counted them as complete.
    """
    try:
        tasks = json.loads(fs.cat_file(path)).get("tasks", {})
    except Exception:
        return False
    return all(t in tasks for t in CANONICAL_OLMO_TASKS)


def scored_index(fs, suite: str) -> set[str]:
    """Run names already scored for `suite`, indexed across every region in one pass.

    Probing `fs.exists` per run per bucket is 6 round-trips per run; at ~500 runs that is
    ~3000 serial calls and the builder times out. Listing each bucket's result prefix once is
    two calls per bucket regardless of run count.
    """
    names: set[str] = set()
    for bucket in SCORE_BUCKETS:
        if suite == "core_v2":
            root = f"{bucket}/metadata/data_curation_10k_core_results"
            if fs.exists(root):
                names |= {
                    p.rsplit("/", 1)[-1].removesuffix("_summary.json")
                    for p in fs.ls(root, detail=False)
                    if p.endswith("_summary.json")
                }
            continue
        root = f"{bucket}/metadata/olmo_bpb_results"
        if not fs.exists(root):
            continue
        if suite in ("mmlu", "blend"):
            marker = "_mmlu.done" if suite == "mmlu" else "_blend.done"
            names |= {p.rsplit("/", 2)[-2] for p in fs.glob(f"{root}/*/{marker}")}
        else:
            dirs = list(fs.ls(root, detail=False))
            with ThreadPoolExecutor(32) as ex:
                oks = list(ex.map(lambda p: _olmo_canonical_done(fs, f"{p}/results.json"), dirs))
            names |= {d.rsplit("/", 1)[-1] for d, ok in zip(dirs, oks, strict=True) if ok}
    return names


def ready_rows(
    method_tokens: tuple[str, ...], exclude: set[str], scores_subpath: str = SCORES_SUBPATH
) -> tuple[list[dict], dict[str, int]]:
    """Completed runs that have an HF export and no Core scores yet."""
    fs = fsspec.filesystem("gs")
    suite = next(k for k, v in SUITE_SUBPATHS.items() if v is scores_subpath)
    already = scored_index(fs, suite)
    rows: list[dict] = []
    tally = {"seen": 0, "no_hf": 0, "already_scored": 0, "in_flight": 0}
    for path in sorted(fs.ls(RESULTS_PREFIX, detail=False)):
        if not path.endswith(".json"):
            continue
        # EXACT method match, not substring. `"dclm_10k" in path` also matches dclm_10k_mix,
        # dclm_10k_decon, dclm_10k_mix_corev2_lambda0p05 ... which turned a 6-corpus BLEnD
        # baseline into 517 runs. Run names are always `curation-<method>-expFM_natural-...`.
        stem = path.rsplit("/", 1)[-1].removesuffix(".json")
        method = next((t for t in method_tokens if stem.startswith(f"curation-{t}-expFM_natural-")), None)
        if method is None:
            continue
        if suite != "blend" and method in BLEND_ONLY_METHODS:
            continue
        summary = json.loads(fs.cat_file(path))
        run_name = summary["plan"]["run_name"]
        region = summary["run"]["region"]
        output_path = summary["run"]["output_path"].rstrip("/")
        tally["seen"] += 1
        if run_name in exclude:
            tally["in_flight"] += 1
            continue
        if run_name in already:
            tally["already_scored"] += 1
            continue
        hf_dir = f"{output_path}/hf/"
        steps = [p for p in (fs.ls(hf_dir, detail=False) if fs.exists(hf_dir) else []) if "/step-" in p]
        # A `step-N/` dir is NOT proof of a loadable export. A child killed mid-export leaves
        # `model.safetensors` (complete, right size) with no `config.json`, and every eval wave
        # then dies on it with FileNotFoundError -- once per suite, forever, since a failed eval
        # writes no completion marker and the run stays "ready". Require the config.
        steps = [p for p in steps if fs.exists(f"{p.rstrip('/')}/config.json")]
        if not steps:
            tally["no_hf"] += 1
            continue
        rows.append({"run_name": run_name, "region": region, "output_path": output_path, "hf_dir": hf_dir})
    return rows, tally


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--exclude", action="append", default=[], help="File of run names already in flight.")
    p.add_argument("--max-rows", type=int, default=None, help="Cap the wave size.")
    p.add_argument(
        "--suite",
        choices=tuple(SUITE_SUBPATHS),
        default="core_v2",
        help="Which eval suite's completion marker decides 'already done'.",
    )
    p.add_argument("--methods", nargs="+", default=list(METHOD_TOKENS), help="Method names to include.")
    args = p.parse_args()

    exclude: set[str] = set()
    for f in args.exclude:
        with open(f) as fh:
            exclude |= {line.strip() for line in fh if line.strip()}

    rows, tally = ready_rows(tuple(args.methods), exclude, SUITE_SUBPATHS[args.suite])
    logger.info(
        "mix runs complete=%d | in_flight=%d already_scored=%d no_hf_export=%d -> READY=%d",
        tally["seen"],
        tally["in_flight"],
        tally["already_scored"],
        tally["no_hf"],
        len(rows),
    )
    if args.max_rows:
        rows = rows[: args.max_rows]
        logger.info("capped this wave to %d rows", len(rows))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_name", "region", "output_path", "hf_dir"])
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %d rows to %s", len(rows), args.out)
    for r in rows:
        logger.info("  %s (%s)", r["run_name"], r["region"])


if __name__ == "__main__":
    main()
