# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Building the eval manifest from completed swarm runs.

The dangerous failure here is SILENT OMISSION rather than an error. The swarm is split
across regions and results are bucket-local, so a builder pointed at one region returns an
empty row set for a corpus running elsewhere and writes a perfectly valid, perfectly wrong
manifest. That really happened: `--region us-east5 --region us-central1` on a single-valued
flag resolved to us-central1 alone, and 28 completed dclm runs silently became 0 rows.
"""

from __future__ import annotations

import csv

import pytest

from experiments.data_mixing import build_olmix_bpb_manifest as builder

STEPS = builder.PROXY_TRAIN_STEPS


def _summary(run_name: str, region: str, steps: int = STEPS) -> dict:
    return {
        "run_name": run_name,
        "region": region,
        "output_path": f"gs://marin-{region}/checkpoints/olmix-swarm/{run_name}",
        "train_steps": steps,
    }


@pytest.fixture
def by_region(monkeypatch):
    """Stub the GCS listing with a {(bucket, corpus): [summary, ...]} table."""
    table: dict[tuple[str, str], list[dict]] = {}

    def fake(bucket: str, corpus: str) -> list[dict]:
        return table.get((bucket, corpus), [])

    monkeypatch.setattr(builder, "completed_runs", fake)
    return table


def _run(tmp_path, corpora, regions):
    out = tmp_path / "manifest.txt"
    argv = ["prog"]
    for c in corpora:
        argv += ["--corpus", c]
    for r in regions:
        argv += ["--region", r]
    argv += ["--out", str(out)]
    import sys

    sys_argv = sys.argv
    sys.argv = argv
    try:
        builder.main()
    finally:
        sys.argv = sys_argv
    with open(out) as fh:
        return list(csv.DictReader(fh))


def test_collects_runs_from_every_region(tmp_path, by_region):
    by_region[("gs://marin-us-east5", "dclm_10k")] = [_summary("run-a", "us-east5")]
    by_region[("gs://marin-us-central1", "high_quality_10k")] = [_summary("run-b", "us-central1")]

    rows = _run(tmp_path, ["dclm_10k", "high_quality_10k"], ["us-east5", "us-central1"])

    assert {r["run_name"] for r in rows} == {"run-a", "run-b"}
    assert {r["region"] for r in rows} == {"us-east5", "us-central1"}


def test_single_region_silently_omits_the_other(tmp_path, by_region, caplog):
    """The exact bug: hq runs only in us-central1, so a us-east5-only build yields a valid
    manifest with hq missing entirely. It must at least WARN rather than pass quietly."""
    by_region[("gs://marin-us-east5", "dclm_10k")] = [_summary("run-a", "us-east5")]
    by_region[("gs://marin-us-central1", "high_quality_10k")] = [_summary("run-b", "us-central1")]

    with caplog.at_level("WARNING"):
        rows = _run(tmp_path, ["dclm_10k", "high_quality_10k"], ["us-east5"])

    assert [r["run_name"] for r in rows] == ["run-a"]
    assert "high_quality_10k: no completed runs" in caplog.text


def test_a_run_is_listed_once_even_if_both_regions_report_it(tmp_path, by_region):
    """A duplicated row would launch two evals for one checkpoint, both writing the same
    results.json."""
    dup = _summary("run-a", "us-east5")
    by_region[("gs://marin-us-east5", "dclm_10k")] = [dup]
    by_region[("gs://marin-us-central1", "dclm_10k")] = [dup]

    rows = _run(tmp_path, ["dclm_10k"], ["us-east5", "us-central1"])
    assert [r["run_name"] for r in rows] == ["run-a"]


def test_partial_run_is_excluded(tmp_path, by_region):
    """A short run's BPB reflects less TRAINING, not a different mixture."""
    by_region[("gs://marin-us-east5", "dclm_10k")] = [
        _summary("run-full", "us-east5"),
        _summary("run-smoke", "us-east5", steps=3100),
    ]
    rows = _run(tmp_path, ["dclm_10k"], ["us-east5"])
    assert [r["run_name"] for r in rows] == ["run-full"]
