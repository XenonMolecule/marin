# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-region egress safety for the hq dilution-ablation variants.

The current curation design uses REGION-LOCAL ``gs://`` cache paths (not the old
``mirror://`` scheme) validated in the training child by
``_assert_all_components_local``. These tests pin the invariant the user cares
about most — NO cross-region reads of data — for the reweight variant that adds a
second (dense) training component, and prove the base ``CurationMethod`` behaviour
is unchanged (no regression).
"""

from __future__ import annotations

import pytest

from experiments.scaling_law_sweeps.data_curation_math import CurationMethod, ReweightedCurationMethod
from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

REGION = "us-central1"
BASE = "tokenized/high_quality_decon_10364warcs-6451c8/"
DENSE = "tokenized/hq_dense-abcdef/"


def _all_component_paths(cfg) -> list[str]:
    paths: list[str] = []
    for comp in cfg.components.values():
        paths.append(comp.cache_dir)
        if comp.source is not None and getattr(comp.source, "cache_dir", None):
            paths.append(comp.source.cache_dir)
    return paths


def test_base_method_single_training_component_unchanged():
    """Regression: a plain CurationMethod still emits exactly one non-zero training weight."""
    m = CurationMethod("hq", BASE, d_obs_tokens=21_296_896_949, sampled_warcs=10_364)
    cfg = m.as_lm_mixture_config(REGION)
    nonzero = {k: w for k, w in cfg.train_weights.items() if w > 0}
    assert nonzero == {"hq": 1.0}


def test_reweight_emits_two_local_training_components():
    m = ReweightedCurationMethod(
        "hq_reweight",
        BASE,
        d_obs_tokens=21_296_896_949,
        sampled_warcs=10_364,
        pin_region=REGION,
        dense_rel_path=DENSE,
        dense_weight=0.144,
    )
    cfg = m.as_lm_mixture_config(REGION)
    nonzero = {k: w for k, w in cfg.train_weights.items() if w > 0}
    assert set(nonzero) == {"hq_reweight", "hq_reweight__dense"}
    assert nonzero["hq_reweight"] == pytest.approx(0.856)
    assert nonzero["hq_reweight__dense"] == pytest.approx(0.144)
    assert sum(nonzero.values()) == pytest.approx(1.0)


def test_reweight_no_cross_region_paths():
    """THE egress guard: every component path (incl. the dense one) is region-local."""
    m = ReweightedCurationMethod(
        "hq_reweight",
        BASE,
        d_obs_tokens=21_296_896_949,
        sampled_warcs=10_364,
        pin_region=REGION,
        dense_rel_path=DENSE,
        dense_weight=0.144,
    )
    cfg = m.as_lm_mixture_config(REGION)
    local_prefix = REGION_TO_BUCKET[REGION] + "/"
    paths = _all_component_paths(cfg)
    assert paths, "expected component paths"
    for p in paths:
        assert p.startswith(local_prefix), f"cross-region path would egress: {p!r} (want {local_prefix})"
    # The dense cache specifically resolves into the local bucket.
    assert f"{local_prefix}{DENSE}" in paths


def test_reweight_validates_weight_and_path():
    with pytest.raises(ValueError, match="dense_weight"):
        ReweightedCurationMethod("bad", BASE, d_obs_tokens=1, dense_rel_path=DENSE, dense_weight=0.0)
    with pytest.raises(ValueError, match="dense_weight"):
        ReweightedCurationMethod("bad", BASE, d_obs_tokens=1, dense_rel_path=DENSE, dense_weight=1.0)
    with pytest.raises(ValueError, match="bare relative path"):
        ReweightedCurationMethod(
            "bad", BASE, d_obs_tokens=1, dense_rel_path="gs://marin-us-central1/tokenized/x/", dense_weight=0.1
        )
