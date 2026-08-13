# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The training child's pre-TPU guards.

``build_mixture`` is the last thing that runs before a swarm child allocates a TPU and
starts reading tokens, so every failure it does not catch becomes either wasted compute or
a silently wrong experiment. The three it must catch:

* a ``cache_dir`` outside the region-local bucket -- real egress on every step, and the
  live us-east5 store's artifact genuinely records us-central1 paths;
* a non-zero weight below ``1/mixture_block_size`` -- Levanter truncates ``int(w*block)``
  to zero samples and only ``warnings.warn``s, so that cell contributes nothing while the
  run looks healthy;
* a malformed mixture (empty, or not summing to 1).
"""

from __future__ import annotations

import pytest

from experiments.data_mixing.olmix_plan import MIXTURE_BLOCK_SIZE
from experiments.data_mixing.run_olmix_swarm_standalone import (
    PROXY_TRAIN_STEPS,
    assert_distinct_identity,
    build_mixture,
)

REGION = "us-east5"
BUCKET = "gs://marin-us-east5"
TOKENIZER = "marin-community/marin-tokenizer"


def _cache_dirs(names, bucket: str = BUCKET) -> dict[str, str]:
    return {n: f"{bucket}/datakit/store/dclm_10k_gridv1/{n}/sub=0" for n in names}


def test_builds_one_component_per_nonzero_weight():
    weights = {"c00_q2": 0.6, "c01_q3": 0.4}
    cfg = build_mixture(weights, _cache_dirs(["c00_q2", "c01_q3", "c02_q4"]), TOKENIZER, REGION, MIXTURE_BLOCK_SIZE)
    assert set(cfg.components) == {"c00_q2", "c01_q3"}
    assert cfg.train_weights == {"c00_q2": 0.6, "c01_q3": 0.4}
    # The unused cell must not be opened at all -- ~108 of 118 per run in the real sweep.
    assert "c02_q4" not in cfg.components


def test_plumbs_tokenizer_and_block_size():
    cfg = build_mixture({"c00_q2": 1.0}, _cache_dirs(["c00_q2"]), TOKENIZER, REGION, 32768)
    assert cfg.tokenizer == TOKENIZER
    assert cfg.mixture_block_size == 32768


def test_component_cache_dirs_are_the_parent_of_the_split_level():
    """Levanter appends ``/<split>`` itself, so cache_dir must NOT already end in /train."""
    cfg = build_mixture({"c00_q2": 1.0}, _cache_dirs(["c00_q2"]), TOKENIZER, REGION, MIXTURE_BLOCK_SIZE)
    cache_dir = cfg.components["c00_q2"].cache_dir
    assert not cache_dir.rstrip("/").endswith("/train")
    assert cfg.components["c00_q2"].source.cache_dir == cache_dir


def test_rejects_cross_region_cache_dir():
    """The live us-east5 store artifact records us-central1 paths, so this is not
    hypothetical -- it is the exact bug this guard exists for."""
    dirs = _cache_dirs(["c00_q2"], bucket="gs://marin-us-central1")
    with pytest.raises(ValueError, match="non-local cache_dir"):
        build_mixture({"c00_q2": 1.0}, dirs, TOKENIZER, REGION, MIXTURE_BLOCK_SIZE)


def test_rejects_weight_below_the_block_floor():
    below = 0.5 / MIXTURE_BLOCK_SIZE
    weights = {"c00_q2": below, "c01_q3": 1.0 - below}
    with pytest.raises(ValueError, match="below the block-size floor"):
        build_mixture(weights, _cache_dirs(["c00_q2", "c01_q3"]), TOKENIZER, REGION, MIXTURE_BLOCK_SIZE)


def test_accepts_weights_at_the_operating_clip():
    """clip 0.01 against a 1/32768 floor is a ~328x margin; a realistic mix must pass."""
    weights = {"c00_q2": 0.01, "c01_q3": 0.99}
    cfg = build_mixture(weights, _cache_dirs(["c00_q2", "c01_q3"]), TOKENIZER, REGION, MIXTURE_BLOCK_SIZE)
    assert len(cfg.components) == 2


def test_block_floor_check_follows_the_block_size_argument():
    w = 1.5 / 32768
    weights = {"c00_q2": w, "c01_q3": 1.0 - w}
    dirs = _cache_dirs(["c00_q2", "c01_q3"])
    build_mixture(weights, dirs, TOKENIZER, REGION, 32768)  # fine
    with pytest.raises(ValueError, match="below the block-size floor"):
        build_mixture(weights, dirs, TOKENIZER, REGION, 2048)


def test_rejects_empty_mixture():
    with pytest.raises(ValueError, match="no non-zero components"):
        build_mixture({}, _cache_dirs(["c00_q2"]), TOKENIZER, REGION, MIXTURE_BLOCK_SIZE)


def test_rejects_weights_not_summing_to_one():
    weights = {"c00_q2": 0.3, "c01_q3": 0.3}
    with pytest.raises(ValueError, match="sum to"):
        build_mixture(weights, _cache_dirs(["c00_q2", "c01_q3"]), TOKENIZER, REGION, MIXTURE_BLOCK_SIZE)


def test_no_validation_components_are_added():
    """Unlike the curation sweep, the swarm carries no weight-0 validation sets: the
    objective is measured by a separate BPB pass over the exported checkpoint, so every
    component here is a training component."""
    cfg = build_mixture({"c00_q2": 0.5, "c01_q3": 0.5}, _cache_dirs(["c00_q2", "c01_q3"]), TOKENIZER, REGION, 32768)
    assert all(w > 0 for w in cfg.train_weights.values())
    assert set(cfg.train_weights) == set(cfg.components)


def test_realistic_118_domain_mixture_at_the_operating_point():
    """End to end at the real shape: ~11 non-zero of 118 on a 0.01 grid."""
    names = [f"c{t:02d}_q{q}" for t in range(24) for q in range(5)][:118]
    nonzero = {n: 0.09 for n in names[:11]}
    nonzero[names[0]] = 1.0 - 0.09 * 10
    cfg = build_mixture(nonzero, _cache_dirs(names), TOKENIZER, REGION, MIXTURE_BLOCK_SIZE)
    assert len(cfg.components) == 11
    assert sum(cfg.train_weights.values()) == pytest.approx(1.0)
    floor = 1.0 / MIXTURE_BLOCK_SIZE
    assert min(cfg.train_weights.values()) >= floor


def test_short_run_must_declare_a_distinct_identity():
    """Guarded before any I/O, so a mis-parameterised smoke run dies at argv rather than after
    it has written a checkpoint under a real member's name."""
    with pytest.raises(ValueError, match="run-name-suffix"):
        assert_distinct_identity(3100, "")


def test_short_run_with_a_suffix_is_allowed():
    assert_distinct_identity(3100, "-smoke")
    assert_distinct_identity(PROXY_TRAIN_STEPS, "")
