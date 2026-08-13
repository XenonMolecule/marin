# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn a silently wedged swarm coordinator into a loud, restartable failure.

A coordinator that stops dispatching does not crash: Iris keeps reporting it RUNNING while
its children drain to zero and never get replaced. This happened live -- ``olmix-coord5-both``
sat wedged in its submit-retry path for over two hours, its dclm children fell 22 -> 10 with
nothing queued behind them, and nothing noticed until someone sampled the process stack by
hand. The failure is invisible precisely because every external signal looks healthy.

**A watchdog cannot repair this.** The main thread is blocked in I/O, so no monitoring thread
can interrupt it -- ``KeyboardInterrupt``/exception injection cannot unwind a thread parked in
a syscall. The only useful move is to convert the hang into a visible, non-zero exit and let
the orchestrator restart the job.

**THIS ONLY SELF-HEALS IF THE JOB IS SUBMITTED WITH ``--max-retries N`` (N >= 1).** Without a
retry policy the watchdog converts a wedged job into a *dead* job: more visible, but no longer
making progress. Launch coordinators as::

    iris --cluster marin job run --job-name olmix-coordN --max-retries 20 ... \\
        -- python -m experiments.data_mixing.launch_olmix_swarm ...

Wiring (two lines in the dispatch loop)::

    watchdog = DispatchWatchdog(state=lambda: DispatchState(
        outstanding=cursor < len(todo), pending=pending))
    watchdog.start()
    ...
    watchdog.record_progress()   # after every successful submit and every loop iteration
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

# Must exceed the longest quiet period a HEALTHY coordinator can have, which is NOT just the
# 240s check interval: on a stall the loop sleeps a backoff cooldown of up to 1800s, and it
# enters that cooldown because `pending > 0`. Children can start while it sleeps, so the
# watchdog can wake to `pending == 0` on a coordinator that is merely mid-cooldown. The worst
# legitimate gap is therefore cooldown + check interval = 2040s. 3600s leaves ~1.8x headroom.
# Detection is correspondingly slower, which is the right trade: the live wedge ran for over
# two hours, so an hour to a loud restart is a large improvement, whereas killing a healthy
# coordinator costs real training work.
STALL_TIMEOUT = 60 * 60.0

# How often the watchdog thread re-evaluates. Cheap: it only reads counters.
POLL_INTERVAL = 30.0

# Distinct, non-zero, and conventionally "temporary failure" -- so a restart is the obviously
# correct response and the exit is not confused with a task-level bug.
WATCHDOG_EXIT_CODE = 75


@dataclass(frozen=True)
class DispatchState:
    """What the watchdog needs to decide whether silence is legitimate.

    Attributes:
        outstanding: is there still work to dispatch (``cursor < len(todo)``)? A coordinator
            that has dispatched everything is *finished*, not wedged, and holds its children
            open forever by design.
        pending: children submitted but not yet running. While this is above zero the
            coordinator is deliberately withholding work until the cluster catches up, so
            silence is correct behaviour rather than a hang.
    """

    outstanding: bool
    pending: int


def is_stalled(state: DispatchState, seconds_since_progress: float, stall_timeout: float = STALL_TIMEOUT) -> bool:
    """Should a coordinator that has been silent this long be considered wedged?

    Kept as a pure function so the decision is testable without threads, sleeping, or a real
    clock -- the repo forbids ``time.sleep`` in tests, and a watchdog whose logic can only be
    exercised by waiting is a watchdog nobody tests.
    """
    if not state.outstanding:
        return False
    if state.pending > 0:
        return False
    return seconds_since_progress >= stall_timeout


def _abort(message: str) -> None:
    """Report and hard-exit.

    Deliberately writes to stderr rather than through ``logging``: the wedge this guards
    against is a *blocked logging transport*, so routing the alarm through the same transport
    could hang the watchdog too. ``os._exit`` skips interpreter cleanup on purpose -- normal
    shutdown would join the blocked main thread and never return.
    """
    sys.stderr.write(f"WATCHDOG: {message}\n")
    sys.stderr.flush()
    os._exit(WATCHDOG_EXIT_CODE)


class DispatchWatchdog:
    """Watches a dispatch loop and kills the process if it goes silent with work left."""

    def __init__(
        self,
        state: Callable[[], DispatchState],
        *,
        stall_timeout: float = STALL_TIMEOUT,
        poll_interval: float = POLL_INTERVAL,
        now: Callable[[], float] = time.monotonic,
        abort: Callable[[str], None] = _abort,
    ) -> None:
        self._state = state
        self._stall_timeout = stall_timeout
        self._poll_interval = poll_interval
        self._now = now
        self._abort = abort
        self._lock = threading.Lock()
        self._last_progress = now()

    def record_progress(self) -> None:
        """Call on every successful submit and every dispatch-loop iteration."""
        with self._lock:
            self._last_progress = self._now()

    def seconds_since_progress(self) -> float:
        with self._lock:
            return self._now() - self._last_progress

    def check_once(self) -> bool:
        """Evaluate once; abort if wedged. Returns whether it fired (for tests)."""
        elapsed = self.seconds_since_progress()
        state = self._state()
        if not is_stalled(state, elapsed, self._stall_timeout):
            return False
        self._abort(
            f"no dispatch progress for {elapsed:.0f}s with work outstanding and 0 pending; "
            f"the coordinator is wedged. Exiting {WATCHDOG_EXIT_CODE} so Iris restarts it "
            f"(requires --max-retries)."
        )
        return True

    def start(self) -> threading.Thread:
        """Run checks on a daemon thread so it never keeps the process alive."""
        thread = threading.Thread(target=self._loop, name="dispatch-watchdog", daemon=True)
        thread.start()
        return thread

    def _loop(self) -> None:
        while True:
            time.sleep(self._poll_interval)
            self.check_once()
