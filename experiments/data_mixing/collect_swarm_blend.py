# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect BLEnD cultural-knowledge bpb over the OLMIX swarm.

BLEnD asks everyday-life questions grounded in 16 specific cultures, scored here as
5-shot rc-cloze gold bpb by ``run_olmo_bpb_eval``. These are **diagnostics** and are
deliberately excluded from the mixture objective, so they live under their own
``metadata/olmix_swarm_blend/`` tree rather than in the fit's results file.

The question this is built to answer is not "which mixture is best on BLEnD" but
whether BLEnD carries *culture-specific* signal at all at this scale. Two mixtures can
differ on every country simply by being better models. What would justify treating
cultural coverage as a distinct mixing target is variance that does **not** collapse
onto one axis.

So the report leads with the eigenvalue spectrum of the 16x16 across-swarm correlation
matrix. If PC1 explains nearly all the variance, the swarm moves every culture together
and BLEnD is measuring general model quality wearing 16 hats; the residual after
projecting PC1 out is the part a mixture could actually steer.

Usage::

    python -m experiments.data_mixing.collect_swarm_blend --out scratch/blend_signal
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import statistics

import fsspec
import numpy as np

from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logger = logging.getLogger(__name__)

SWARM_BLEND_ROOT = "metadata/olmix_swarm_blend"
DEFAULT_REGIONS = ("us-east5", "us-central1", "europe-west4", "us-west4")

COUNTRIES = (
    "algeria",
    "assam",
    "azerbaijan",
    "china",
    "ethiopia",
    "greece",
    "indonesia",
    "iran",
    "mexico",
    "north_korea",
    "northern_nigeria",
    "south_korea",
    "spain",
    "uk",
    "us",
    "west_java",
)
VARIANT = "rc_5shot"
BLEND_TASKS = tuple(f"blend_{c}/{VARIANT}" for c in COUNTRIES)

# Below this many runs the across-swarm sd is itself too uncertain to read a verdict off
# (the sd of a sample sd is ~sd/sqrt(2(n-1)): +-50% at n=3, +-13% at n=30).
MIN_RUNS_FOR_VERDICT = 30

_RUN_NAME_RE = re.compile(r"^olmix-(?P<corpus>.+)-s\d+-K\d+-i(?P<index>\d+)-w[0-9a-f]+$")


def parse_run_name(run_name: str) -> tuple[str, int] | None:
    """``olmix-dclm_10k-s42-K363-i0085-w7983b27c`` -> ``("dclm_10k", 85)``.

    The corpus may contain hyphens, so anchor on the fixed ``-s<seed>-K<K>-i<idx>-w<hash>``
    tail rather than splitting on ``-``.
    """
    m = _RUN_NAME_RE.match(run_name)
    return (m.group("corpus"), int(m.group("index"))) if m else None


def read_blend_results(regions: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """``{run_name: {task: bpb}}`` across every region.

    A run is indexed by name, not by region, so a checkpoint MIGRATED between regions to
    chase capacity is still found. Later regions do not overwrite an earlier hit.
    """
    out: dict[str, dict[str, float]] = {}
    for region in regions:
        bucket = REGION_TO_BUCKET[region]
        root = f"gs://{bucket}/{SWARM_BLEND_ROOT}"
        fs, _ = fsspec.core.url_to_fs(root)
        try:
            paths = fs.glob(f"{bucket}/{SWARM_BLEND_ROOT}/*/results.json")
        except FileNotFoundError:
            logger.warning("no BLEnD results tree in %s", region)
            continue
        logger.info("%-14s %4d results.json", region, len(paths))
        for path in paths:
            run_name = path.split("/")[-2]
            if run_name in out:
                continue
            with fs.open(path) as f:
                doc = json.load(f)
            tasks = doc.get("tasks") or {}
            scores = {t: float(tasks[t]["bpb"]) for t in BLEND_TASKS if t in tasks and "bpb" in tasks[t]}
            if scores:
                out[run_name] = scores
    return out


def spectrum(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Eigenvalue variance shares of a correlation matrix, plus the PC1 share."""
    eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
    shares = eigenvalues / eigenvalues.sum()
    return shares, float(shares[0])


def report(scores: dict[str, dict[str, float]], out_dir: str) -> None:
    rows = []
    for run_name, task_scores in sorted(scores.items()):
        parsed = parse_run_name(run_name)
        if parsed is None:
            logger.warning("unparseable run name, skipping: %s", run_name)
            continue
        corpus, index = parsed
        rows.append({"run_name": run_name, "corpus": corpus, "index": index, **task_scores})

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "blend_swarm_bpb.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "corpus", "index", *BLEND_TASKS])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d runs)", csv_path, len(rows))

    by_corpus: dict[str, list[dict]] = {}
    for row in rows:
        by_corpus.setdefault(row["corpus"], []).append(row)

    summary: dict[str, dict] = {}
    for corpus, corpus_rows in sorted(by_corpus.items()):
        complete = [r for r in corpus_rows if all(t in r for t in BLEND_TASKS)]
        print(f"\n=== {corpus}: {len(complete)}/{len(corpus_rows)} runs with all 16 countries ===")
        if len(complete) < MIN_RUNS_FOR_VERDICT:
            print(f"  only {len(complete)} runs -- too few for a stable sd; NOT reporting verdicts")
            continue

        print(f"  {'country':<20} {'mean bpb':>9} {'sd':>8} {'cv %':>7}")
        per_country_sd = {}
        for task in BLEND_TASKS:
            values = [r[task] for r in complete]
            mean, sd = statistics.fmean(values), statistics.stdev(values)
            per_country_sd[task] = sd
            print(f"  {task.split('/')[0][6:]:<20} {mean:>9.4f} {sd:>8.4f} {100 * sd / mean:>7.2f}")

        matrix = np.array([[r[t] for t in BLEND_TASKS] for r in complete])
        corr = np.corrcoef(matrix, rowvar=False)
        shares, pc1 = spectrum(corr)
        # Residual after removing the shared "general quality" axis: what a mixture could
        # steer per-culture rather than by lifting all boats.
        centered = (matrix - matrix.mean(0)) / matrix.std(0)
        pc1_vector = np.linalg.eigh(corr)[1][:, -1]
        residual = centered - np.outer(centered @ pc1_vector, pc1_vector)
        residual_share = float((residual.std(0) ** 2).mean())

        print(f"  PC1 explains {100 * pc1:.1f}% of across-swarm correlation")
        print(f"  top-4 eigenvalue shares: {', '.join(f'{100 * s:.1f}%' for s in shares[:4])}")
        print(f"  mean residual variance after removing PC1: {100 * residual_share:.1f}%")
        print(
            "  => "
            + (
                "BLEnD moves as ONE axis here; culture-specific mixing has little to grab"
                if pc1 > 0.9
                else "there IS culture-specific structure beyond general quality"
            )
        )
        summary[corpus] = {
            "n_runs": len(complete),
            "pc1_share": pc1,
            "eigenvalue_shares": [float(s) for s in shares],
            "residual_variance_share": residual_share,
            "per_country_sd": {t: per_country_sd[t] for t in BLEND_TASKS},
        }

    summary_path = os.path.join(out_dir, "blend_signal_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s", summary_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--out", default="scratch/blend_signal")
    args = ap.parse_args()

    regions = tuple(r.strip() for r in args.regions.split(",") if r.strip())
    scores = read_blend_results(regions)
    logger.info("collected %d runs with BLEnD scores", len(scores))
    report(scores, args.out)


if __name__ == "__main__":
    main()
