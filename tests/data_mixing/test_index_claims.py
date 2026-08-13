# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic claiming of swarm indices.

These tests exist because check-then-act demonstrably does not work here. Reading state and
then submitting lets two coordinators reach the same decision in the gap: one measured hour had
53 of 57 completions be re-runs, and 21 run names trained twice across regions. The contract
under test is that exactly ONE caller can win a claim, even when several ask at once.
"""

from __future__ import annotations

import json

import fsspec
import pytest

from experiments.data_mixing.index_claims import (
    CLAIM_TTL_SECONDS,
    claim_path,
    claimed_run_names,
    reap_dead_claims,
    try_claim,
)

RUN = "olmix-dclm_10k-s42-K363-i0042-wdeadbeef"


@pytest.fixture
def root(tmp_path):
    """A local filesystem registry; exercises the non-atomic fallback path deterministically."""
    return str(tmp_path / "claims")


def test_first_caller_wins(root):
    assert try_claim("dclm_10k", RUN, owner="coord-a", root=root, now=1000.0) is True


def test_second_caller_loses(root):
    assert try_claim("dclm_10k", RUN, owner="coord-a", root=root, now=1000.0) is True
    assert try_claim("dclm_10k", RUN, owner="coord-b", root=root, now=1001.0) is False


def test_exactly_one_winner_under_contention(root):
    """The property that matters: many coordinators, one winner."""
    wins = [try_claim("dclm_10k", RUN, owner=f"coord-{i}", root=root, now=1000.0) for i in range(8)]
    assert sum(wins) == 1, wins


def test_distinct_indices_do_not_block_each_other(root):
    a = "olmix-dclm_10k-s42-K363-i0001-waaa"
    b = "olmix-dclm_10k-s42-K363-i0002-wbbb"
    assert try_claim("dclm_10k", a, owner="x", root=root, now=1000.0) is True
    assert try_claim("dclm_10k", b, owner="x", root=root, now=1000.0) is True


def test_same_index_in_different_corpora_is_independent(root):
    """Claims are namespaced by corpus; dclm i0042 must not block hq i0042."""
    assert try_claim("dclm_10k", RUN, owner="x", root=root, now=1000.0) is True
    assert try_claim("high_quality_10k", RUN, owner="x", root=root, now=1000.0) is True


def test_stale_claim_is_reclaimable(root):
    """A coordinator can die holding claims. Without expiry those indices become unrunnable
    forever, which is strictly worse than duplicating them."""
    assert try_claim("dclm_10k", RUN, owner="dead", root=root, now=1000.0) is True
    later = 1000.0 + CLAIM_TTL_SECONDS + 1
    assert try_claim("dclm_10k", RUN, owner="live", root=root, now=later) is True
    fs, p = fsspec.core.url_to_fs(claim_path("dclm_10k", RUN, root))
    assert json.loads(fs.cat_file(p))["owner"] == "live"


def test_claim_just_under_ttl_is_respected(root):
    """Long runs (1.5-4h plus preemption churn) must never be stolen out from under themselves."""
    assert try_claim("dclm_10k", RUN, owner="running", root=root, now=1000.0) is True
    assert try_claim("dclm_10k", RUN, owner="thief", root=root, now=1000.0 + CLAIM_TTL_SECONDS - 1) is False


def test_only_one_thief_wins_a_stale_claim(root):
    """Several coordinators can notice staleness together; the delete-then-atomic-create
    sequence must still yield a single winner."""
    assert try_claim("dclm_10k", RUN, owner="dead", root=root, now=1000.0) is True
    later = 1000.0 + CLAIM_TTL_SECONDS + 1
    wins = [try_claim("dclm_10k", RUN, owner=f"thief-{i}", root=root, now=later) for i in range(5)]
    assert sum(wins) == 1, wins


def test_claimed_run_names_reports_live_claims_only(root):
    try_claim("dclm_10k", RUN, owner="x", root=root, now=1000.0)
    assert claimed_run_names("dclm_10k", root=root, now=1000.0) == {RUN}
    assert claimed_run_names("dclm_10k", root=root, now=1000.0 + CLAIM_TTL_SECONDS + 1) == set()


def test_missing_registry_is_not_an_error(root):
    assert claimed_run_names("dclm_10k", root=root, now=1000.0) == set()


def test_reaper_frees_a_claim_whose_run_died(root):
    """The observed failure: a preempted worker killed 30 children at once. Their claims
    outlived them and would have blocked retry for the full 8h TTL with nothing running."""
    try_claim("dclm_10k", RUN, owner="dead-coord", root=root, now=1000.0)
    freed = reap_dead_claims("dclm_10k", live_run_names=set(), done_run_names=set(), root=root)
    assert freed == 1
    # Released, so the index is immediately re-claimable rather than waiting out the TTL.
    assert try_claim("dclm_10k", RUN, owner="retry", root=root, now=1001.0) is True


def test_reaper_never_frees_a_LIVE_claim(root):
    """The dangerous direction. Releasing a claim on a still-training run would let a second
    coordinator duplicate it -- the exact failure this module exists to prevent."""
    try_claim("dclm_10k", RUN, owner="running", root=root, now=1000.0)
    assert reap_dead_claims("dclm_10k", live_run_names={RUN}, done_run_names=set(), root=root) == 0
    assert try_claim("dclm_10k", RUN, owner="thief", root=root, now=1001.0) is False


def test_reaper_never_frees_a_COMPLETED_claim(root):
    """A finished run's claim is harmless; releasing it invites a pointless re-run."""
    try_claim("dclm_10k", RUN, owner="finished", root=root, now=1000.0)
    assert reap_dead_claims("dclm_10k", live_run_names=set(), done_run_names={RUN}, root=root) == 0


def test_reaper_is_idempotent(root):
    try_claim("dclm_10k", RUN, owner="dead", root=root, now=1000.0)
    assert reap_dead_claims("dclm_10k", set(), set(), root=root) == 1
    assert reap_dead_claims("dclm_10k", set(), set(), root=root) == 0


def test_reaper_on_empty_registry(root):
    assert reap_dead_claims("dclm_10k", set(), set(), root=root) == 0
