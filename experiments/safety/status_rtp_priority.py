# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot status of the 14 RTP eval jobs.

Checks both iris job state and GCS results.json presence so we don't get fooled
by stale job-list pagination.
"""

import subprocess

from experiments.safety.launch_rtp_priority import RUNS, safe_label

OUTPUT_PREFIX = "gs://marin-us-central1/metadata/rtp_eval"


def _results_exists(model_name: str) -> bool:
    path = f"{OUTPUT_PREFIX}/{model_name}/results.json"
    p = subprocess.run(["gcloud", "storage", "ls", path], capture_output=True, text=True)
    return p.returncode == 0


def _iris_state(job_name: str) -> str:
    p = subprocess.run(
        [
            "uv",
            "run",
            "iris",
            "--cluster",
            "marin",
            "job",
            "list",
            "--prefix",
            f"/michaelryan/{job_name}",
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return "?"
    # The output is human-formatted; look for STATE keywords.
    # iris emits lowercase states ("pending", "running", "succeeded", "failed", etc.)
    for state in ("running", "succeeded", "completed", "failed", "killed", "pending", "queued", "preempted"):
        if state in p.stdout:
            return state
    return "unknown"


def main() -> None:
    print(f"{'priority':<8s}  {'label':<42s}  {'iris':<12s}  {'results.json':<14s}")
    print("-" * 84)
    done = 0
    for r in RUNS:
        job_name = f"rtp-eval-{safe_label(r.label)}"
        model_name = safe_label(r.label)
        has_results = _results_exists(model_name)
        state = _iris_state(job_name)
        marker = "DONE" if has_results else "—"
        print(f"{r.priority:<8s}  {r.label:<42s}  {state:<12s}  {marker:<14s}")
        if has_results:
            done += 1
    print("-" * 84)
    print(f"Done: {done}/{len(RUNS)}")


if __name__ == "__main__":
    main()
