# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assembling the (mixture, BPB) dataset the fit consumes.

The dangerous failures here are all *silent shrinkage or contamination* of the dataset:
a run with a missing task padded rather than dropped, a partial run admitted alongside
full ones, a mixture row that does not sum to 1 sneaking past olmix's loader tolerance,
or -- once the swarm is split across regions -- the same mixture counted twice while
another region's weights are paired with the wrong scores.

The BPB documents here are written in the shape ``run_olmo_bpb_eval`` actually emits
(``tasks: {<task>: {"bpb": ...}}``, mean under ``averages``), because assuming a flat
top-level map is exactly the bug that made the collector drop every run.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from experiments.data_mixing.collect_olmix_swarm import (
    ROW_SUM_ATOL,
    RegionSource,
    SwarmRow,
    collect,
    live_domains,
    load_swarm_rows,
    run_name_for,
    task_bpb_scores,
    write_metrics_csv,
    write_ratios_csv,
)
from experiments.data_mixing.olmix_plan import SwarmManifest

DOMAINS = ("c00_q0", "c00_q1", "c01_q0")
TASKS = ["gsm8k/gold_bpb_5shot", "drop/bpb_5shot"]
FULL_STEPS = 32844
NO_SKIPS = {
    "no_bpb": 0,
    "incomplete_bpb": 0,
    "wrong_train_steps": 0,
    "index_mismatch": 0,
    "duplicate_run": 0,
    "manifest_mismatch": 0,
}


def _manifest(**kw) -> SwarmManifest:
    defaults = dict(
        corpus="dclm_10k",
        region="us-east5",
        seed=42,
        domains=DOMAINS,
        weights=((0.5, 0.5, 0.0), (0.25, 0.25, 0.5), (1.0, 0.0, 0.0)),
        tokens={d: 10_000_000 for d in DOMAINS},
        cache_dirs={d: f"gs://marin-us-east5/x/{d}" for d in DOMAINS},
        tokenizer="marin-community/marin-tokenizer",
    )
    defaults.update(kw)
    return SwarmManifest(**defaults)


def _bpb_doc(scores: dict[str, float]) -> dict:
    """A BPB results.json in the real harness shape."""
    return {
        "run_name": "x",
        "tasks": {t: {"bpb": v, "bpb_no_leading_space": v + 0.01, "n_docs": 100} for t, v in scores.items()},
        "averages": {"macro_bpb": sum(scores.values()) / len(scores)} if scores else {},
    }


def _write_run(results_dir, bpb_dir, run_name, index, steps=FULL_STEPS, bpb=None, drop_task=None):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{run_name}.json").write_text(
        json.dumps({"run_name": run_name, "index": index, "train_steps": steps})
    )
    if bpb is not None:
        d = bpb_dir / run_name
        d.mkdir(parents=True, exist_ok=True)
        scores = dict(bpb)
        if drop_task:
            scores.pop(drop_task, None)
        (d / "results.json").write_text(json.dumps(_bpb_doc(scores)))


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "results", tmp_path / "bpb", tmp_path / "out"


def _sources(results, bpb) -> list[RegionSource]:
    return [RegionSource(results_prefix=str(results), bpb_prefix=str(bpb))]


def _name(index: int, manifest: SwarmManifest | None = None) -> str:
    """Run name for a row. MUST be derived from the manifest the test then collects
    against -- the name embeds the mixture hash, so naming from a different manifest is
    rejected as manifest_mismatch (which is the guard working, not a test bug to suppress)."""
    return run_name_for(manifest if manifest is not None else _manifest(), index)


def test_reads_the_real_harness_bpb_shape():
    """`run_olmo_bpb_eval` nests per-task scores under `tasks` and the mean under
    `averages`. Treating the document as a flat {task: float} map matches zero tasks and
    silently reports every run as incomplete instead of raising."""
    doc = _bpb_doc({TASKS[0]: 1.5, TASKS[1]: 2.5})
    assert task_bpb_scores(doc) == {TASKS[0]: 1.5, TASKS[1]: 2.5}


def test_rejects_a_document_that_is_not_eval_output():
    with pytest.raises(ValueError, match="no 'tasks' map"):
        task_bpb_scores({"bpb": {TASKS[0]: 1.0}})


def test_pairs_runs_with_their_bpb(dirs):
    results, bpb, _ = dirs
    _write_run(results, bpb, _name(0), 0, bpb={t: 1.0 for t in TASKS})
    _write_run(results, bpb, _name(1), 1, bpb={t: 2.0 for t in TASKS})
    rows, skipped = load_swarm_rows(_sources(results, bpb), _manifest(), TASKS, FULL_STEPS)
    assert [r.index for r in rows] == [0, 1]
    assert rows[0].weights == {"c00_q0": 0.5, "c00_q1": 0.5, "c01_q0": 0.0}
    assert skipped == NO_SKIPS


def test_run_without_bpb_is_skipped_not_padded(dirs):
    results, bpb, _ = dirs
    _write_run(results, bpb, _name(0), 0, bpb=None)
    rows, skipped = load_swarm_rows(_sources(results, bpb), _manifest(), TASKS, FULL_STEPS)
    assert rows == []
    assert skipped["no_bpb"] == 1


def test_run_missing_one_task_is_excluded_entirely(dirs):
    """Padding a missing task would put a hole in exactly one task's regression while
    leaving the others intact -- a silent, per-task change to the fitting data."""
    results, bpb, _ = dirs
    _write_run(results, bpb, _name(0), 0, bpb={t: 1.0 for t in TASKS}, drop_task=TASKS[1])
    rows, skipped = load_swarm_rows(_sources(results, bpb), _manifest(), TASKS, FULL_STEPS)
    assert rows == []
    assert skipped["incomplete_bpb"] == 1


def test_partial_run_cannot_enter_the_objective(dirs):
    """A 3,100-step smoke really did land under a real index once. Its BPB reflects less
    TRAINING, not a different mixture, which is the confound the swarm isolates."""
    results, bpb, _ = dirs
    _write_run(results, bpb, _name(0), 0, steps=3100, bpb={t: 1.0 for t in TASKS})
    rows, skipped = load_swarm_rows(_sources(results, bpb), _manifest(), TASKS, FULL_STEPS)
    assert rows == []
    assert skipped["wrong_train_steps"] == 1


def test_index_outside_the_manifest_is_rejected(dirs):
    results, bpb, _ = dirs
    _write_run(results, bpb, "olmix-dclm_10k-s42-K3-i0099-wdeadbeef", 99, bpb={t: 1.0 for t in TASKS})
    rows, skipped = load_swarm_rows(_sources(results, bpb), _manifest(), TASKS, FULL_STEPS)
    assert rows == []
    assert skipped["index_mismatch"] == 1


def test_run_from_a_different_swarm_is_rejected(dirs):
    """The run name ends in a hash of its mixture. If a region resampled its manifest, its
    index 0 is a different mixture -- pairing its BPB with this manifest's row 0 would
    corrupt the design matrix with no error anywhere."""
    results, bpb, _ = dirs
    _write_run(results, bpb, "olmix-dclm_10k-s42-K3-i0000-wbadhash1", 0, bpb={t: 1.0 for t in TASKS})
    rows, skipped = load_swarm_rows(_sources(results, bpb), _manifest(), TASKS, FULL_STEPS)
    assert rows == []
    assert skipped["manifest_mismatch"] == 1


def test_merges_runs_across_regions(tmp_path):
    """A swarm split across regions must be pooled: each region holds only the runs it
    trained, and their scores live in that same bucket."""
    east_r, east_b = tmp_path / "e5" / "results", tmp_path / "e5" / "bpb"
    central_r, central_b = tmp_path / "c1" / "results", tmp_path / "c1" / "bpb"
    _write_run(east_r, east_b, _name(0), 0, bpb={t: 1.0 for t in TASKS})
    _write_run(central_r, central_b, _name(2), 2, bpb={t: 3.0 for t in TASKS})

    sources = [
        RegionSource(str(east_r), str(east_b)),
        RegionSource(str(central_r), str(central_b)),
    ]
    rows, skipped = load_swarm_rows(sources, _manifest(), TASKS, FULL_STEPS)
    assert sorted(r.index for r in rows) == [0, 2]
    assert skipped == NO_SKIPS


def test_a_run_completed_in_two_regions_is_counted_once(tmp_path):
    """Overlapping coordinator ranges can finish the same index twice. Both rows are
    valid, but admitting both would double that mixture's weight in the regression."""
    east_r, east_b = tmp_path / "e5" / "results", tmp_path / "e5" / "bpb"
    central_r, central_b = tmp_path / "c1" / "results", tmp_path / "c1" / "bpb"
    _write_run(east_r, east_b, _name(1), 1, bpb={t: 1.0 for t in TASKS})
    _write_run(central_r, central_b, _name(1), 1, bpb={t: 9.0 for t in TASKS})

    rows, skipped = load_swarm_rows(
        [RegionSource(str(east_r), str(east_b)), RegionSource(str(central_r), str(central_b))],
        _manifest(),
        TASKS,
        FULL_STEPS,
    )
    assert len(rows) == 1
    assert rows[0].bpb[TASKS[0]] == 1.0  # first source wins, deterministically
    assert skipped["duplicate_run"] == 1


def test_a_region_with_no_results_yet_is_not_an_error(tmp_path):
    """us-central1 is empty until its first run lands; that must not abort the merge."""
    east_r, east_b = tmp_path / "e5" / "results", tmp_path / "e5" / "bpb"
    _write_run(east_r, east_b, _name(0), 0, bpb={t: 1.0 for t in TASKS})
    rows, _ = load_swarm_rows(
        [RegionSource(str(east_r), str(east_b)), RegionSource(str(tmp_path / "missing"), str(tmp_path / "nope"))],
        _manifest(),
        TASKS,
        FULL_STEPS,
    )
    assert len(rows) == 1


def test_ratios_csv_matches_olmix_schema():
    rows = [SwarmRow("run-a", 0, {"c00_q0": 0.5, "c00_q1": 0.5, "c01_q0": 0.0}, {})]
    text = write_ratios_csv(rows, list(DOMAINS))
    parsed = list(csv.reader(io.StringIO(text)))
    assert parsed[0] == ["run", "name", "index", *DOMAINS]
    assert parsed[1][0] == "run-a" and parsed[1][2] == "0"
    assert sum(float(v) for v in parsed[1][3:]) == pytest.approx(1.0)


def test_metrics_csv_column_order_follows_task_names():
    """The fit indexes tasks positionally off this order, so it must be the caller's."""
    rows = [SwarmRow("run-a", 0, {}, {TASKS[0]: 1.5, TASKS[1]: 2.5})]
    parsed = list(csv.reader(io.StringIO(write_metrics_csv(rows, TASKS))))
    assert parsed[0][3:] == TASKS
    assert parsed[1][3:] == ["1.5", "2.5"]


def test_ratios_row_must_sum_to_one():
    rows = [SwarmRow("run-a", 0, {"c00_q0": 0.5, "c00_q1": 0.2, "c01_q0": 0.0}, {})]
    with pytest.raises(ValueError, match="outside olmix's"):
        write_ratios_csv(rows, list(DOMAINS))
    assert ROW_SUM_ATOL == 0.01


def test_collect_writes_both_files_and_flags_insufficient_data(dirs):
    """m+1 runs are needed for a unique log-linear solution; fewer must warn loudly
    rather than produce a confident-looking but underdetermined fit."""
    results, bpb, out = dirs
    _write_run(results, bpb, _name(0), 0, bpb={t: 1.0 for t in TASKS})
    summary = collect(_manifest(), _sources(results, bpb), TASKS, str(out), FULL_STEPS)
    assert summary["rows"] == 1
    assert summary["domains"] == 3
    assert summary["sufficient_for_unique_fit"] is False
    assert (out / "ratios.csv").exists()
    assert (out / "metrics.csv").exists()


def test_collect_raises_when_nothing_is_usable(dirs):
    results, bpb, out = dirs
    _write_run(results, bpb, _name(0), 0, bpb=None)
    with pytest.raises(RuntimeError, match="no usable swarm rows"):
        collect(_manifest(), _sources(results, bpb), TASKS, str(out), FULL_STEPS)


def test_readiness_uses_live_domains_not_all_domains(dirs):
    """The law strips domains no collected mixture samples, so it only ever estimates
    m_live+1 parameters. Measuring readiness against all m overstates the bar and reports
    "not ready" on a dataset that is actually solvable."""
    results, bpb, out = dirs
    # Rows 0 and 1 together sample only c00_q0 and c00_q1; c01_q0 stays dead.
    manifest = _manifest(weights=((0.5, 0.5, 0.0), (0.25, 0.75, 0.0), (1.0, 0.0, 0.0)))
    _write_run(results, bpb, _name(0, manifest), 0, bpb={t: 1.0 for t in TASKS})
    _write_run(results, bpb, _name(1, manifest), 1, bpb={t: 2.0 for t in TASKS})

    summary = collect(manifest, _sources(results, bpb), TASKS, str(out), FULL_STEPS)
    assert summary["domains"] == 3
    assert summary["live_domains"] == 2
    assert summary["dead_domains"] == 1
    # 2 live domains need 3 runs; we have 2, so still not sufficient -- but the bar is 3, not 4.
    assert summary["runs_needed_for_unique_fit"] == 3
    assert summary["sufficient_for_unique_fit"] is False


def test_a_dataset_can_be_sufficient_below_the_full_domain_count(dirs):
    """With most cells never sampled, m_live+1 can be cleared long before m+1 could be."""
    results, bpb, out = dirs
    # Every mixture touches only the first two domains.
    manifest = _manifest(weights=((0.5, 0.5, 0.0), (0.25, 0.75, 0.0), (0.9, 0.1, 0.0)))
    for i in range(3):
        _write_run(results, bpb, _name(i, manifest), i, bpb={t: float(i + 1) for t in TASKS})

    summary = collect(manifest, _sources(results, bpb), TASKS, str(out), FULL_STEPS)
    assert summary["live_domains"] == 2
    assert summary["runs_needed_for_unique_fit"] == 3
    assert summary["sufficient_for_unique_fit"] is True


def test_live_domain_bar_can_rise_faster_than_rows():
    """Adding one run that activates several dormant domains raises the requirement by more
    than it raises the supply, so readiness is not monotonic and must be recomputed."""
    manifest = _manifest(weights=((1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.4, 0.3, 0.3)))
    two = [SwarmRow(_name(0, manifest), 0, {"c00_q0": 1.0, "c00_q1": 0.0, "c01_q0": 0.0}, {})]
    live_two, _ = live_domains(two, manifest)
    assert len(live_two) + 1 == 2

    three = [*two, SwarmRow(_name(2, manifest), 2, {"c00_q0": 0.4, "c00_q1": 0.3, "c01_q0": 0.3}, {})]
    live_three, _ = live_domains(three, manifest)
    # supply went 1 -> 2 runs, but the bar went 2 -> 4: one run woke two dormant domains.
    assert len(live_three) + 1 == 4


def test_live_domains_handles_an_empty_collection():
    """Runs finished but evals still pending is the normal early state, not an error."""
    manifest = _manifest()
    live, dead = live_domains([], manifest)
    assert live == []
    assert dead == list(DOMAINS)
