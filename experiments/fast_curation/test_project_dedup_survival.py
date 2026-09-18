# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from experiments.fast_curation.project_dedup_survival import (
    fit_power_law,
    load_point,
    local_exponents,
    report,
)

WARCS = [10_000, 100_000, 1_000_000]


def test_fit_recovers_the_exponent_of_an_exact_power_law_and_extrapolates_it():
    unique = [int(5e6 * warcs**0.8) for warcs in WARCS]

    fit = fit_power_law(WARCS, unique)

    assert fit.exponent == pytest.approx(0.8, abs=1e-6)
    assert fit.at(8_000_000) == pytest.approx(5e6 * 8_000_000**0.8, rel=1e-6)


def test_a_bending_curve_shows_falling_exponents_and_a_lower_tail_projection():
    # Growth slows from warcs^0.9 to warcs^0.6 in the second decade.
    unique = [1_000_000, int(1_000_000 * 10**0.9), int(1_000_000 * 10**0.9 * 10**0.6)]

    first, second = local_exponents(WARCS, unique)
    all_points = fit_power_law(WARCS, unique)
    last_two = fit_power_law(WARCS[-2:], unique[-2:])

    assert (first, second) == (pytest.approx(0.9, abs=1e-6), pytest.approx(0.6, abs=1e-6))
    assert last_two.at(8_000_000) < all_points.at(8_000_000)


def test_a_point_is_the_content_each_rule_leaves(tmp_path):
    path = tmp_path / "survival.json"
    path.write_text(
        json.dumps(
            {
                "docs": 100,
                "chars": 1000,
                "cc_canonical_removed_docs": 40,
                "cc_canonical_removed_chars": 550,
                "c075_removed_docs": 10,
                "c075_removed_chars": 120,
            }
        )
    )

    point = load_point(10_000, str(path))

    assert point.unique_docs == {"cc_canonical": 60, "c075": 90}
    assert point.unique_chars == {"cc_canonical": 450, "c075": 880}
    assert "c075" in report([point, load_point(20_000, str(path))], [8_000_000])


def test_one_point_cannot_be_fitted():
    with pytest.raises(ValueError, match="at least two points"):
        fit_power_law([10_000], [5])
