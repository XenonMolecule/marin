# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OLMIX-mixture arms must differ from their base arm in the WEIGHTS AND NOTHING ELSE.

The whole point of `dclm_10k_mix` / `high_quality_10k_mix` is a controlled comparison against
`dclm_10k` / `high_quality_10k` on the frozen 10k grid. That only holds if `d_obs`, `s`,
`sampled_warcs` and the tokenizer are identical, so the natural-epoch target lands on the same
number at every cell. These tests pin that, plus the two ways the mixture itself can be silently
wrong: a leftover single-cache component (which would train on the un-mixed corpus at weight 1.0
alongside the grid), and weights below Levanter's truncation floor (which contribute zero samples
while still counting toward normalisation).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from experiments.scaling_law_sweeps.curation_plan import METHODS
from experiments.scaling_law_sweeps.data_curation_math import GridMixCurationMethod

PAIRS = [("dclm_10k_mix", "dclm_10k"), ("high_quality_10k_mix", "high_quality_10k")]


@pytest.mark.parametrize(("mix_name", "base_name"), PAIRS)
def test_mix_arm_matches_its_base_arm_exactly(mix_name, base_name):
    """Anything unequal here makes the comparison measure two things at once."""
    mix, base = METHODS[mix_name], METHODS[base_name]
    assert mix.d_obs_tokens == base.d_obs_tokens
    assert mix.sampled_warcs == base.sampled_warcs
    assert mix.total_warcs == base.total_warcs
    assert mix.tokenizer == base.tokenizer
    assert mix.s == base.s  # => identical natural-epoch target at every grid cell


@pytest.mark.parametrize(("mix_name", "_base"), PAIRS)
def test_registered_as_a_grid_mix_method(mix_name, _base):
    m = METHODS[mix_name]
    assert isinstance(m, GridMixCurationMethod)
    assert m.grid_corpus and m.mixture_rel_path
    assert 0 < m.mixture_block_size < 2**16  # Levanter packs ids as (i << 16)


@pytest.mark.parametrize(("mix_name", "_base"), PAIRS)
def test_vendored_mixture_is_a_normalised_distribution(mix_name, _base):
    """The mixture ships in-repo so the executed weights are version-controlled."""
    m = METHODS[mix_name]
    path = pathlib.Path(__file__).resolve().parents[2] / m.mixture_rel_path
    payload = json.loads(path.read_text())
    weights = payload["weights"]
    assert payload["repetition_factor"] == 20
    assert payload["requested_tokens"] == pytest.approx(3.0e10)
    assert all(v >= 0 for v in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(("mix_name", "_base"), PAIRS)
def test_loaded_weights_clear_the_truncation_floor_and_renormalise(mix_name, _base):
    """Levanter turns w into int(w * block) samples, so anything under 1/block contributes
    nothing. We drop those explicitly and renormalise rather than letting them rot silently."""
    m = METHODS[mix_name]
    w = m.load_weights()
    floor = 1.0 / m.mixture_block_size
    assert w, "every cell was dropped"
    assert min(w.values()) >= floor
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(("mix_name", "_base"), PAIRS)
def test_dropped_mass_is_negligible(mix_name, _base):
    """Dropping sub-floor cells is only acceptable because they carry ~nothing. If a config
    change ever pushes real mass under the floor, fail loudly instead of quietly losing it."""
    m = METHODS[mix_name]
    path = pathlib.Path(__file__).resolve().parents[2] / m.mixture_rel_path
    raw = json.loads(path.read_text())["weights"]
    floor = 1.0 / m.mixture_block_size
    dropped = sum(v for v in raw.values() if v < floor)
    assert dropped < 1e-3, f"{mix_name} would silently drop {dropped:.4%} of the mixture"


@pytest.mark.parametrize(("mix_name", "_base"), PAIRS)
def test_block_size_is_the_maximum(mix_name, _base):
    """A smaller block raises the floor and deletes more cells, so the max is the only
    defensible choice given the solve leaves weights as small as 7e-8."""
    assert METHODS[mix_name].mixture_block_size == 65535


def test_mixture_names_are_grid_cell_ids():
    """Weights are keyed by `c<CC>_q<Q>`; a rename upstream must fail here, not at 3am on a TPU."""
    from experiments.data_mixing.olmix_domains import domain_name

    m = METHODS["dclm_10k_mix"]
    path = pathlib.Path(__file__).resolve().parents[2] / m.mixture_rel_path
    names = set(json.loads(path.read_text())["weights"])
    assert names, "no weights"
    # Generate the ids the same way the grid does, rather than re-deriving the format here.
    legal = {domain_name(c, q) for c in range(24) for q in range(5)}
    assert names <= legal, sorted(names - legal)[:5]


def test_base_arms_are_not_grid_mix_methods():
    """Guard against a copy-paste that turns a control arm into a treatment arm."""
    for _, base_name in PAIRS:
        assert not isinstance(METHODS[base_name], GridMixCurationMethod)


def test_mix_arms_are_on_the_frozen_10k_grid():
    """They must ride the identical 38-cell grid as every other 10k method."""
    from experiments.scaling_law_sweeps.launch_10k_natural import METHOD_NAMES, enumerate_10k_natural_plans

    for mix_name, base_name in PAIRS:
        assert mix_name in METHOD_NAMES
        mix_cells = {(p.hidden_dim, p.budget, p.batch_size) for p in enumerate_10k_natural_plans((mix_name,))}
        base_cells = {(p.hidden_dim, p.budget, p.batch_size) for p in enumerate_10k_natural_plans((base_name,))}
        assert mix_cells == base_cells
        assert len(mix_cells) == 38
