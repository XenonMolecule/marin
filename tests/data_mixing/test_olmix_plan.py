# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The coordinator/child contract: manifest invariants and run identity.

These guard the failure modes that would be silent at scale -- a mixture that trains on
nothing because a weight fell under the block-size floor, or two runs colliding on a
checkpoint path because the run name does not encode the mixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.data_mixing.olmix_plan import (
    MIXTURE_BLOCK_SIZE,
    SwarmManifest,
    effective_domains,
    manifest_path,
    read_manifest,
    run_name,
    weight_hash,
    write_manifest,
)

DOMAINS = ("c00_q0", "c00_q1", "c01_q0")


def _manifest(weights, **kw) -> SwarmManifest:
    defaults = dict(
        corpus="dclm_10k",
        region="us-east5",
        seed=42,
        domains=DOMAINS,
        weights=tuple(tuple(r) for r in weights),
        tokens={d: 10_000_000 for d in DOMAINS},
        cache_dirs={d: f"gs://marin-us-east5/datakit/store/dclm_10k_gridv1/{d}" for d in DOMAINS},
    )
    defaults.update(kw)
    return SwarmManifest(**defaults)


def test_row_drops_zero_weight_domains():
    """The child must not open a cache it will never read -- with 118 domains and ~10
    non-zero per mix that is ~108 caches spared per run."""
    m = _manifest([[0.5, 0.5, 0.0]])
    assert m.row(0) == {"c00_q0": 0.5, "c00_q1": 0.5}
    assert "c01_q0" not in m.row(0)


def test_row_weights_still_sum_to_one_after_dropping_zeros():
    m = _manifest([[0.25, 0.0, 0.75]])
    assert sum(m.row(0).values()) == pytest.approx(1.0)


def test_rows_must_sum_to_one():
    with pytest.raises(ValueError, match="sums to"):
        _manifest([[0.5, 0.4, 0.0]])


def test_row_width_must_match_domains():
    with pytest.raises(ValueError, match="expected 3"):
        _manifest([[0.5, 0.5]])


def test_tokens_and_cache_dirs_must_cover_the_domains():
    with pytest.raises(ValueError, match="tokens keys"):
        _manifest([[1.0, 0.0, 0.0]], tokens={"c00_q0": 1})
    with pytest.raises(ValueError, match="cache_dirs keys"):
        _manifest([[1.0, 0.0, 0.0]], cache_dirs={"c00_q0": "gs://x"})


def test_index_out_of_range_raises():
    m = _manifest([[1.0, 0.0, 0.0]])
    with pytest.raises(IndexError):
        m.row(1)


def test_assert_trainable_rejects_a_weight_below_the_block_floor():
    """Levanter only warns when a non-zero weight truncates to zero samples per block, so
    this must fail at launch time instead of being buried in K training logs."""
    below = 0.5 / MIXTURE_BLOCK_SIZE
    m = _manifest([[below, 1.0 - below, 0.0]])
    with pytest.raises(ValueError, match="below the block-size floor"):
        m.assert_trainable()


def test_assert_trainable_accepts_weights_at_the_clip():
    """A realistic mixture at clip 0.01 clears the 1/32768 floor with wide margin."""
    m = _manifest([[0.01, 0.99, 0.0], [0.5, 0.25, 0.25]])
    m.assert_trainable()


def test_assert_trainable_honours_block_size():
    w = 1.5 / 32768
    m = _manifest([[w, 1.0 - w, 0.0]])
    m.assert_trainable(block_size=32768)
    with pytest.raises(ValueError, match="below the block-size floor"):
        m.assert_trainable(block_size=2048)


def test_run_name_encodes_the_mixture():
    """Existing sweep run names encode (method, budget, arch, batch) but NOT the mixture,
    so without the hash two swarm runs would collide on checkpoint path, DONE marker,
    region-tracker key and results JSON."""
    a = run_name("dclm_10k", 42, 363, 7, {"c00_q0": 0.5, "c00_q1": 0.5})
    b = run_name("dclm_10k", 42, 363, 7, {"c00_q0": 0.6, "c00_q1": 0.4})
    assert a != b
    assert a.startswith("olmix-dclm_10k-s42-K363-i0007-w")
    assert len(a) <= 200, "must fit Iris's job-name cap"


def test_weight_hash_is_order_and_float_repr_stable():
    assert weight_hash({"a": 0.5, "b": 0.5}) == weight_hash({"b": 0.5, "a": 0.5})
    assert weight_hash({"a": 1 / 3}) == weight_hash({"a": 0.333333333})
    assert weight_hash({"a": 0.5, "b": 0.5}) != weight_hash({"a": 0.4, "b": 0.6})


def test_manifest_round_trips_through_gcs_style_io(tmp_path):
    m = _manifest([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]], sampler={"minimum_weight": 0.01})
    path = str(tmp_path / "swarm.json")
    write_manifest(m, path)
    back = read_manifest(path)
    assert back.domains == m.domains
    assert back.weights == m.weights
    assert back.sampler == {"minimum_weight": 0.01}
    assert back.k == 2


def test_manifest_path_layout():
    p = manifest_path("gs://marin-us-east5", "dclm_10k", 42, 363)
    assert p == "gs://marin-us-east5/metadata/olmix/dclm_10k/swarm_s42_K363.json"


def test_effective_domains_separates_never_sampled():
    """Never-sampled domains have an all-zero design column, so their coefficient is
    data-free and the solver acts on garbage -- measured, it drives them to ~0 rather
    than leaving them at natural. They must be identified so the fit can exclude them."""
    tokens = {d: 1 for d in DOMAINS}
    weights = np.array([[0.5, 0.5, 0.0], [0.25, 0.75, 0.0]])
    sampled, dead = effective_domains(tokens, weights)
    assert sampled == ["c00_q0", "c00_q1"]
    assert dead == ["c01_q0"]


def test_effective_domains_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="columns"):
        effective_domains({d: 1 for d in DOMAINS}, np.zeros((2, 2)))
