# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converting a wedged coordinator into a restartable failure.

The bug this guards against is silence, not a crash: a coordinator blocked in its submit path
keeps reporting RUNNING while its children drain and are never replaced. So the tests that
matter are as much about *not* firing as about firing -- a watchdog that kills a coordinator
which is legitimately waiting on cluster capacity is worse than no watchdog at all.

The clock is injected throughout. Nothing here sleeps, so the hour-long timeout is exercised
in microseconds.
"""

from __future__ import annotations

import pytest

from experiments.data_mixing.dispatch_watchdog import (
    STALL_TIMEOUT,
    DispatchState,
    DispatchWatchdog,
    is_stalled,
)


class _FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _watchdog(state: DispatchState, clock: _FakeClock) -> tuple[DispatchWatchdog, list[str], dict]:
    """A watchdog whose abort is captured instead of killing the test process.

    Returns the mutable state holder too, so a test can change what the coordinator reports
    between checks the way a real dispatch loop would.
    """
    fired: list[str] = []
    holder = {"state": state}
    wd = DispatchWatchdog(state=lambda: holder["state"], now=clock, abort=fired.append)
    return wd, fired, holder


def test_fires_when_silent_with_work_outstanding_and_nothing_pending():
    """The live failure: work left to dispatch, no pending children to wait on, and the
    dispatch loop has stopped iterating."""
    clock = _FakeClock()
    wd, fired, _ = _watchdog(DispatchState(outstanding=True, pending=0), clock)

    clock.advance(STALL_TIMEOUT + 1)
    assert wd.check_once() is True
    assert len(fired) == 1
    assert "wedged" in fired[0]


def test_does_not_fire_while_children_are_pending():
    """pending > 0 means the coordinator is deliberately withholding work until the cluster
    catches up. That silence is correct behaviour -- killing it would be a false positive on
    every capacity-constrained sweep."""
    clock = _FakeClock()
    wd, fired, _ = _watchdog(DispatchState(outstanding=True, pending=12), clock)

    clock.advance(STALL_TIMEOUT * 10)
    assert wd.check_once() is False
    assert fired == []


def test_does_not_fire_once_the_swarm_is_fully_dispatched():
    """A coordinator with nothing left to submit holds its children open forever by design.
    That is completion, not a hang."""
    clock = _FakeClock()
    wd, fired, _ = _watchdog(DispatchState(outstanding=False, pending=0), clock)

    clock.advance(STALL_TIMEOUT * 10)
    assert wd.check_once() is False
    assert fired == []


def test_progress_resets_the_timer():
    clock = _FakeClock()
    wd, fired, _ = _watchdog(DispatchState(outstanding=True, pending=0), clock)

    clock.advance(STALL_TIMEOUT - 1)
    assert wd.check_once() is False

    wd.record_progress()
    clock.advance(STALL_TIMEOUT - 1)
    assert wd.check_once() is False
    assert fired == []

    clock.advance(2)
    assert wd.check_once() is True


def test_does_not_fire_exactly_below_the_threshold():
    clock = _FakeClock()
    wd, fired, _ = _watchdog(DispatchState(outstanding=True, pending=0), clock)
    clock.advance(STALL_TIMEOUT - 0.001)
    assert wd.check_once() is False
    assert fired == []


def test_state_is_re_read_each_check_not_captured_once():
    """The watchdog must see the loop's *current* state; a snapshot taken at construction
    would keep firing on stale counters after the coordinator recovered."""
    clock = _FakeClock()
    wd, _, holder = _watchdog(DispatchState(outstanding=True, pending=0), clock)

    holder["state"] = DispatchState(outstanding=True, pending=5)
    clock.advance(STALL_TIMEOUT * 2)
    assert wd.check_once() is False

    holder["state"] = DispatchState(outstanding=True, pending=0)
    assert wd.check_once() is True


@pytest.mark.parametrize(
    ("outstanding", "pending", "elapsed", "expected"),
    [
        (True, 0, STALL_TIMEOUT, True),
        (True, 0, STALL_TIMEOUT - 1, False),
        (True, 1, STALL_TIMEOUT * 5, False),
        (False, 0, STALL_TIMEOUT * 5, False),
        (False, 3, STALL_TIMEOUT * 5, False),
    ],
)
def test_is_stalled_truth_table(outstanding, pending, elapsed, expected):
    assert is_stalled(DispatchState(outstanding=outstanding, pending=pending), elapsed) is expected


def test_timeout_cannot_fire_on_a_healthy_slow_cycle():
    """Guards the constant itself, which is the easiest thing here to get wrong.

    The worst legitimate quiet period is not the 240s check interval. On a stall the loop
    sleeps a cooldown of up to 1800s, and it enters that cooldown *because* pending > 0 --
    but children can start while it sleeps, so the watchdog can observe pending == 0 on a
    coordinator that is simply mid-cooldown. The timeout must clear cooldown + one check.
    """
    max_backoff_cooldown = 1800.0
    check_interval = 240.0
    assert STALL_TIMEOUT > max_backoff_cooldown + check_interval


def test_watchdog_thread_is_a_daemon():
    """It must never hold the process open on its own."""
    clock = _FakeClock()
    wd, _, _ = _watchdog(DispatchState(outstanding=False, pending=0), clock)
    thread = wd.start()
    assert thread.daemon is True
