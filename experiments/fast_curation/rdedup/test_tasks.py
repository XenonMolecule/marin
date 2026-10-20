# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.fast_curation.rdedup.store import LocalStore, PreconditionFailed
from experiments.fast_curation.rdedup.tasks import SEAL, ClaimConfig, TaskBoard


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def board(tmp_path):
    clock = Clock()
    store = LocalStore(clock=clock)
    config = ClaimConfig(heartbeat_seconds=1, stale_seconds=100, max_attempts=3)
    return TaskBoard(store, str(tmp_path / "stages" / "s"), 4, config), clock


def test_exactly_one_worker_wins_a_fresh_claim(board):
    board, _ = board
    first = board.try_claim(2, "a")
    assert first is not None
    assert board.try_claim(2, "b") is None
    state = board.state()
    assert board.claimable(state) == [0, 1, 3]


def test_stale_claim_is_taken_over_once_and_the_old_owner_loses_it(board):
    board, clock = board
    old = board.try_claim(1, "a")
    clock.now += 101
    assert board.claimable(board.state()) == [0, 1, 2, 3]
    new = board.try_claim(1, "b")
    assert new is not None and new.attempt == 2
    # A second taker sees the fresh takeover and backs off.
    assert board.try_claim(1, "c") is None
    # The original owner's next heartbeat fails, which is what makes it stop writing.
    with pytest.raises(PreconditionFailed):
        board.heartbeat(old, "a")
    board.heartbeat(new, "b")


def test_heartbeat_keeps_a_long_task_from_going_stale(board):
    board, clock = board
    claim = board.try_claim(0, "a")
    for _ in range(5):
        clock.now += 60
        board.heartbeat(claim, "a")
    assert board.try_claim(0, "b") is None
    assert 0 not in board.claimable(board.state())


def test_task_that_keeps_killing_workers_is_poisoned(board):
    board, clock = board
    assert board.try_claim(3, "a") is not None
    for worker in ("b", "c"):
        clock.now += 101
        assert board.try_claim(3, worker) is not None
    clock.now += 101
    assert board.try_claim(3, "d") is None


def test_done_tasks_leave_the_pending_set_and_seal_opens_when_all_are_done(board):
    board, _ = board
    for t in range(4):
        claim = board.try_claim(t, "a")
        board.complete(claim, {"value": t})
    state = board.state()
    assert board.pending(state) == [] and board.claimable(state) == []
    assert [r["value"] for r in board.done_results()] == [0, 1, 2, 3]
    seal = board.try_claim(SEAL, "a")
    board.complete(seal, {"total": 6})
    assert board.is_sealed() and board.sealed_result()["total"] == 6


def test_a_second_completion_keeps_the_first_result(board):
    board, clock = board
    slow = board.try_claim(0, "a")
    clock.now += 101
    fast = board.try_claim(0, "b")
    board.complete(fast, {"by": "b"})
    board.complete(slow, {"by": "a"})
    assert json.loads(board.store.read(board.done_path(0)))["by"] == "b"
