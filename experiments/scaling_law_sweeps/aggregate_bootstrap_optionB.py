# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate Zephyr bootstrap outputs into a single trust-region npz.

Pulls `{run}/seeds-*.jsonl.gz` (base64 winner grids per seed) + `central_fit.json`
from GCS, stacks the bootstrap winners into (B, n_C, n_N) per warcs panel,
computes per-cell consensus + trust fraction, and saves one npz that the figure
script can load to render BOTH the colored winner regions (central fit) and the
trust overlay (bootstrap fractions) — no refit.

Usage:

    uv run --with numpy --with pandas python \\
      experiments/scaling_law_sweeps/aggregate_bootstrap_optionB.py \\
      --run gs://marin-us-central1/scratch/bootstrap_optionB/run_<ts> \\
      --out scratch/plots/bootstrap_results/optionB_bootstrap_zephyr.npz
"""

from __future__ import annotations

import argparse
import base64
import glob
import gzip
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

KEEP_METHODS = ("low_quality", "med_quality", "high_quality")
N_GRID = 200


def _decode_grid(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("ascii"))
    return np.frombuffer(raw, dtype=np.int8).reshape(N_GRID, N_GRID)


def _pull(run: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", f"{run.rstrip('/')}/seeds-*.jsonl.gz", str(local_dir) + "/"], check=True)
    subprocess.run(["gcloud", "storage", "cp", f"{run.rstrip('/')}/central_fit.json", str(local_dir) + "/"], check=True)
    return local_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="gs:// run prefix.")
    p.add_argument("--out", required=True, help="Local .npz output path.")
    p.add_argument("--local-cache", default="scratch/plots/bootstrap_results/iris_optionB")
    args = p.parse_args()

    local_dir = _pull(args.run, Path(args.local_cache))
    central = json.loads((local_dir / "central_fit.json").read_text())
    warcs_panels = central["warcs_panels"]

    # Collect per-warcs winner grids across all seeds.
    grids: dict[int, list[np.ndarray]] = {w: [] for w in warcs_panels}
    n_ok = n_fail = 0
    for path in sorted(glob.glob(str(local_dir / "seeds-*.jsonl.gz"))):
        with gzip.open(path, "rt") as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("ok"):
                    n_fail += 1
                    continue
                n_ok += 1
                for w in warcs_panels:
                    grids[w].append(_decode_grid(rec[f"W{w}"]))
    logger.info("Loaded %d successful bootstrap seeds (%d failed)", n_ok, n_fail)

    payload: dict = {
        "n_bootstraps_completed": n_ok,
        "n_grid": N_GRID,
        "warcs_values": np.array(warcs_panels, dtype=np.int64),
        "central_params_json": json.dumps(central["params"]),
        "central_tpw_json": json.dumps(central["tpw"]),
        "central_log_n_ranges_json": json.dumps(central["log_n_ranges"]),
        "central_log_c_ranges_json": json.dumps(central["log_c_ranges"]),
    }
    print("\nConsensus summary:")
    for w in warcs_panels:
        arr = np.stack(grids[w], axis=0)  # (B, n_C, n_N)
        fracs = np.stack([(arr == i).mean(axis=0) for i in range(len(KEEP_METHODS))], axis=0)
        consensus = np.argmax(fracs, axis=0).astype(np.int8)
        trust = np.max(fracs, axis=0).astype(np.float32)
        payload[f"winners_W{w}"] = arr
        payload[f"consensus_W{w}"] = consensus
        payload[f"trust_W{w}"] = trust
        payload[f"log_N_W{w}"] = np.linspace(*central["log_n_ranges"][str(w)], N_GRID)
        payload[f"log_C_W{w}"] = np.linspace(*central["log_c_ranges"][str(w)], N_GRID)
        for i, m in enumerate(KEEP_METHODS):
            sel = consensus == i
            cells = sel.mean() * 100
            mt = trust[sel].mean() if sel.any() else 0.0
            print(f"  W={w:>10,}  {m:14s}  {cells:5.1f}% of grid  mean trust {mt:.3f}")
        print(f"  W={w:>10,}  uncertain (<95% trust): {(trust < 0.95).mean()*100:.1f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
