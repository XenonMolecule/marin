# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn completed curation-sweep cells into a checkpoint manifest for the bpb eval launcher.

`launch_10k_natural` writes `metadata/data_curation_10k_natural_results/curation-<method>-<cell>.json`
when a cell finishes, carrying the `run.output_path` its checkpoints live under and the region it
trained in. That JSON set is the authoritative list of evaluable runs -- an Iris "succeeded" does
not imply the HF export landed, so rows whose `hf/` directory is absent are dropped and reported
rather than handed to the launcher to fail on.

Emits the `run_name,region,output_path` CSV that
`scaling_law_sweeps.olmo_bpb.launch_olmo_bpb_manifest` reads. Safe to re-run as a sweep fills in:
pass `--skip-evaluated` to emit only cells that do not yet have a `results.json`, which is what a
top-up wave wants.

    python -m experiments.scaling_law_sweeps.olmo_bpb.build_curation_bpb_manifest \\
        --method lpv11_forkrs_10k_mix_olmixexact_lambda0p01 \\
        --out experiments/core_eval_manifests/olmix_bpb_lpv11_forkrs.txt --skip-evaluated
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import fsspec

logger = logging.getLogger(__name__)

RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_10k_natural_results/"
BPB_SUBPATH = "metadata/olmo_bpb_results"
BUCKETS = ("marin-us-east5", "marin-us-central1", "marin-us-east1", "marin-us-central2")


def _fs():
    return fsspec.filesystem("gcs")


def cell_rows(methods: list[str], results_prefix: str = RESULTS_PREFIX) -> list[dict]:
    """One `{run_name, region, output_path}` per completed cell of `methods`, newest first."""
    fs = _fs()
    root = results_prefix.split("://", 1)[1]
    names = [p for p in fs.ls(root, refresh=True) if p.endswith(".json") and ".badmodel" not in p]
    wanted = [(m, f"curation-{m}-") for m in methods]
    todo = [p for p in names for _, pre in wanted if p.rsplit("/", 1)[-1].startswith(pre)]

    def read(path: str) -> dict:
        payload = json.loads(fs.cat(path))
        return {
            "run_name": payload["plan"]["run_name"],
            "region": payload["run"]["region"],
            "output_path": payload["run"]["output_path"].rstrip("/"),
        }

    with ThreadPoolExecutor(32) as pool:
        return sorted(pool.map(read, todo), key=lambda r: r["run_name"])


def has_hf_export(row: dict) -> bool:
    return _fs().exists(row["output_path"].replace("gs://", "") + "/hf")


def already_evaluated(row: dict) -> bool:
    fs = _fs()
    return any(fs.exists(f"{b}/{BPB_SUBPATH}/{row['run_name']}/results.json") for b in BUCKETS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", action="append", required=True, help="Repeatable.")
    p.add_argument("--out", required=True, help="CSV path to write.")
    p.add_argument("--results-prefix", default=RESULTS_PREFIX)
    p.add_argument("--skip-evaluated", action="store_true", help="Emit only cells with no results.json yet.")
    args = p.parse_args()

    rows = cell_rows(args.method, args.results_prefix)
    logger.info("%d completed cells for %s", len(rows), args.method)
    with ThreadPoolExecutor(32) as pool:
        keep = [r for r, ok in zip(rows, pool.map(has_hf_export, rows), strict=True) if ok]
    if len(keep) != len(rows):
        logger.warning("dropped %d cells with no hf/ export", len(rows) - len(keep))
    if args.skip_evaluated:
        with ThreadPoolExecutor(32) as pool:
            done = list(pool.map(already_evaluated, keep))
        before = len(keep)
        keep = [r for r, d in zip(keep, done, strict=True) if not d]
        logger.info("skipping %d already-evaluated cells", before - len(keep))
    if not keep:
        raise SystemExit("nothing to evaluate")
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_name", "region", "output_path"])
        writer.writeheader()
        writer.writerows(keep)
    by_region: dict[str, int] = {}
    for r in keep:
        by_region[r["region"]] = by_region.get(r["region"], 0) + 1
    logger.info("wrote %s: %d rows %s", args.out, len(keep), by_region)


if __name__ == "__main__":
    main()
