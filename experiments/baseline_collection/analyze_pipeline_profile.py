# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-group pipeline timing sidecars into a run-level report.

Reads the ``batch_NNNN.timing.json`` files written by run_extract_standalone.py's
``--pipeline`` path and prints stage-time share, throughput, drop rate, and
loop-guard activity — the artifact for the tokenizer / partials / group-size
decisions. Works on a local dir or a ``gs://`` prefix.

Usage::

    python -m experiments.baseline_collection.analyze_pipeline_profile \
        gs://marin-us-central1/documents/baseline_llm_extraction/llm_pipeline_v1
"""

from __future__ import annotations

import argparse
import json
import logging

import fsspec

logger = logging.getLogger(__name__)


def _find_timing_files(run_dir: str) -> list[str]:
    """Glob every batch timing sidecar under a run dir (any data-*/ subdir)."""
    is_gcs = run_dir.startswith("gs://")
    fs = fsspec.filesystem("gcs") if is_gcs else fsspec.filesystem("file")
    pattern = f"{run_dir.rstrip('/')}/data-*/batch_*.timing.json"
    glob_pat = pattern.replace("gs://", "") if is_gcs else pattern
    paths = fs.glob(glob_pat)
    return [f"gs://{p}" if is_gcs else p for p in paths]


def _load(path: str) -> dict | None:
    try:
        with fsspec.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def aggregate(profiles: list[dict]) -> dict:
    """Sum counts, stage seconds, and loop-guard activity across groups."""
    agg: dict = {
        "groups": len(profiles),
        "n_docs": 0,
        "kept": 0,
        "dropped": 0,
        "errored": 0,
        "n_chunks": 0,
        "stage_seconds": {},
        "loop_guard": {"amplified": 0, "rung1": 0, "rung2": 0, "truncated": 0},
    }
    for p in profiles:
        for k in ("n_docs", "kept", "dropped", "errored", "n_chunks"):
            agg[k] += p.get(k, 0)
        for stage, secs in (p.get("stage_seconds") or {}).items():
            agg["stage_seconds"][stage] = agg["stage_seconds"].get(stage, 0.0) + secs
        for k, v in (p.get("loop_guard") or {}).items():
            agg["loop_guard"][k] = agg["loop_guard"].get(k, 0) + v
    return agg


def format_report(agg: dict) -> str:
    total_secs = sum(agg["stage_seconds"].values()) or 1.0
    lines = [
        f"groups={agg['groups']}  docs={agg['n_docs']}  "
        f"kept={agg['kept']} dropped={agg['dropped']} errored={agg['errored']}  "
        f"chunks={agg['n_chunks']}",
    ]
    if agg["n_docs"]:
        lines.append(
            f"drop_rate={agg['dropped'] / agg['n_docs']:.1%}  "
            f"chunks_per_kept={agg['n_chunks'] / max(agg['kept'], 1):.2f}"
        )
    lines.append("")
    lines.append("stage                    seconds     % of total")
    lines.append("-" * 48)
    for stage, secs in sorted(agg["stage_seconds"].items(), key=lambda kv: -kv[1]):
        lines.append(f"{stage:<24} {secs:>8.1f}   {secs / total_secs:>8.1%}")
    lines.append("-" * 48)
    lines.append(f"{'TOTAL':<24} {total_secs:>8.1f}")
    lg = agg["loop_guard"]
    lines.append("")
    lines.append(
        f"loop-guard: amplified={lg['amplified']} rung1_fixed={lg['rung1']} "
        f"rung2_fixed={lg['rung2']} truncated={lg['truncated']}"
    )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run dir (local or gs://) holding data-*/batch_*.timing.json")
    args = parser.parse_args()

    paths = _find_timing_files(args.run_dir)
    if not paths:
        logger.warning("No timing sidecars found under %s", args.run_dir)
        return
    profiles = [p for p in (_load(path) for path in paths) if p is not None]
    logger.info("Loaded %d/%d timing sidecars", len(profiles), len(paths))
    print(format_report(aggregate(profiles)))


if __name__ == "__main__":
    main()
