# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A vendored mixture must load through the REAL sweep loader.

The solve calls its weight map ``mixture`` and the loader reads ``weights``, so a straight copy
of the GCS artifact raises ``KeyError`` only once someone plans a sweep. These tests close that
gap by round-tripping through ``GridMixCurationMethod.load_weights`` itself rather than asserting
on the file's shape.
"""

from __future__ import annotations

import json

import pytest

from experiments.data_mixing.vendor_olmix_mixture import CARRIED_FIELDS, vendor_mixture
from experiments.scaling_law_sweeps.data_curation_math import GridMixCurationMethod


def _solve_artifact(n_cells: int = 8, tiny: float = 0.0) -> dict:
    """A solve artifact shaped like run_olmix_fit's output, weights summing to 1."""
    bulk = 1.0 - tiny
    weights = {f"c{i:02d}_q2": bulk / n_cells for i in range(n_cells)}
    if tiny:
        weights["c99_q0"] = tiny
    return {
        "corpus": "test_10k",
        "kl_reg": 0.01,
        "live_domains": n_cells,
        "natural": {k: 1.0 / len(weights) for k in weights},
        "regression_fit": {"average_bpb": 0.98},
        "repetition_factor": 20.0,
        "requested_tokens": 3.0e10,
        "runs": 363,
        "mixture": weights,
        # Diagnostics that must NOT be vendored into git.
        "interaction_matrix": [[0.1] * n_cells] * 4,
        "log_c": [0.1] * 4,
        "domains": list(weights),
        "dead_domains": [],
    }


@pytest.fixture
def vendored(tmp_path, monkeypatch):
    def _run(artifact, grid_corpus="test_10k", tag="_olmixexact_lambda0p01"):
        src = tmp_path / "mix_R3e+10_k20.json"
        src.write_text(json.dumps(artifact))
        monkeypatch.setattr("experiments.data_mixing.vendor_olmix_mixture.MIXTURES_DIR", tmp_path)
        return vendor_mixture(str(src), grid_corpus, tag)

    return _run


def test_vendored_file_loads_through_the_real_sweep_loader(vendored):
    """The point of the whole script: the sweep must be able to read what we wrote."""
    out = vendored(_solve_artifact())
    # load_weights resolves `repo_root / mixture_rel_path`; joining with an absolute path yields
    # that path, so the real loader reads our temp file without writing into the repo.
    method = GridMixCurationMethod(
        name="test",
        grid_corpus="test_10k",
        mixture_rel_path=str(out),
        tokenized_rel_path="tokenized/deadbeef/",
        d_obs_tokens=1_000_000,
    )
    weights = method.load_weights()
    assert weights, "the real loader returned nothing for a valid vendored mixture"
    assert abs(sum(weights.values()) - 1.0) < 1e-9, "loader renormalises to 1.0"


def test_solve_key_is_renamed_not_duplicated(vendored):
    raw = json.loads(vendored(_solve_artifact()).read_text())
    assert "mixture" not in raw, "leaving 'mixture' alongside 'weights' invites editing the wrong one"


def test_diagnostics_are_stripped(vendored):
    """interaction_matrix is n_tasks x n_domains; vendoring it would bloat git."""
    raw = json.loads(vendored(_solve_artifact()).read_text())
    for heavy in ("interaction_matrix", "log_c", "domains", "dead_domains"):
        assert heavy not in raw


def test_provenance_is_recorded(vendored):
    raw = json.loads(vendored(_solve_artifact()).read_text())
    assert raw["source"].endswith("mix_R3e+10_k20.json")
    assert raw["swarm_runs"] == 363


def test_carried_fields_survive(vendored):
    raw = json.loads(vendored(_solve_artifact()).read_text())
    for field in CARRIED_FIELDS:
        assert field in raw, f"{field} is needed to interpret the solve later"


def test_filename_matches_the_path_curation_plan_builds(vendored):
    out = vendored(_solve_artifact(), grid_corpus="lpv11_fastpipe_v1_10k", tag="_olmixexact_lambda0p01")
    assert out.name == "lpv11_fastpipe_v1_10k_R3e10_k20_olmixexact_lambda0p01.json"


def test_a_mixture_that_does_not_sum_to_one_is_rejected(vendored):
    bad = _solve_artifact()
    bad["mixture"]["c00_q2"] += 0.5
    with pytest.raises(ValueError, match="sums to"):
        vendored(bad)


def test_an_artifact_without_a_mixture_is_rejected(vendored):
    notasolve = _solve_artifact()
    del notasolve["mixture"]
    with pytest.raises(KeyError, match="mixture"):
        vendored(notasolve)
