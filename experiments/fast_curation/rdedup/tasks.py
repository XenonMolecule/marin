# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Claimable tasks: the only coordination the dedup pipeline has.

A stage is ``n`` numbered tasks plus one seal. Under the stage directory::

    claims/{task:06d}.json   who holds the task: created atomically, refreshed by heartbeat
    done/{task:06d}.json     the task's result, written once after its outputs are durable
    claims/seal.json         the same protocol for the seal
    sealed.json              the stage's aggregate result; its presence opens the next stage

Any number of stateless workers poll these. A worker claims a task by creating its claim
object (exactly one creator wins), heartbeats by replacing the claim against the generation it
last wrote (so a worker that lost its claim finds out on its next beat), and finishes by
writing ``done``. A claim whose heartbeat is older than ``stale_seconds`` belongs to a dead
worker and is taken over by replacing it against the stale generation, which again has exactly
one winner. Task outputs are deterministic, so a slow former owner that still writes after a
takeover writes the same bytes.

A task that keeps killing its workers (an OOM, a poison input) is not retried forever: each
takeover increments ``attempt``, and a claim at ``max_attempts`` is left alone and reported as
poisoned, which stalls the stage visibly instead of burning the fleet.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from experiments.fast_curation.rdedup.store import ObjectStore, PreconditionFailed, dumps, read_json

logger = logging.getLogger(__name__)

SEAL = "seal"


@dataclasses.dataclass(frozen=True)
class ClaimConfig:
    """Heartbeat cadence and the age at which a silent claim is presumed dead."""

    heartbeat_seconds: float = 60.0
    stale_seconds: float = 480.0
    max_attempts: int = 6


@dataclasses.dataclass
class Claim:
    """A task this worker holds. ``generation`` tracks the claim object's latest write."""

    path: str
    task: str
    token: str
    attempt: int
    generation: int
    started: float


@dataclasses.dataclass(frozen=True)
class BoardState:
    """One listing of a stage: which tasks are done, and the claim metadata of the rest."""

    done: frozenset[str]
    claims: dict[str, tuple[int, float]]  # task -> (generation, updated)
    now: float


def _task_name(task: int | str) -> str:
    return task if isinstance(task, str) else f"{task:06d}"


def worker_identity() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


class TaskBoard:
    """Claims, heartbeats and completion marks for one stage's tasks."""

    def __init__(self, store: ObjectStore, stage_dir: str, n_tasks: int, config: ClaimConfig) -> None:
        self.store = store
        self.stage_dir = stage_dir.rstrip("/")
        self.n_tasks = n_tasks
        self.config = config

    def claim_path(self, task: int | str) -> str:
        return f"{self.stage_dir}/claims/{_task_name(task)}.json"

    def done_path(self, task: int | str) -> str:
        if task == SEAL:
            return self.sealed_path
        return f"{self.stage_dir}/done/{_task_name(task)}.json"

    @property
    def sealed_path(self) -> str:
        return f"{self.stage_dir}/sealed.json"

    def state(self) -> BoardState:
        """List done marks and claims. Raises if either listing fails."""
        done = frozenset(
            os.path.basename(info.path).removesuffix(".json")
            for info in self.store.list(f"{self.stage_dir}/done/")
            if info.path.endswith(".json")
        )
        claims = {
            os.path.basename(info.path).removesuffix(".json"): (info.generation, info.updated)
            for info in self.store.list(f"{self.stage_dir}/claims/")
            if info.path.endswith(".json")
        }
        return BoardState(done=done, claims=claims, now=self.store.now())

    def pending(self, state: BoardState) -> list[int]:
        """Tasks without a done mark."""
        return [t for t in range(self.n_tasks) if _task_name(t) not in state.done]

    def claimable(self, state: BoardState) -> list[int]:
        """Tasks that are not done and are either unclaimed or held by a stale claim."""
        out = []
        for t in self.pending(state):
            claim = state.claims.get(_task_name(t))
            if claim is None or state.now - claim[1] > self.config.stale_seconds:
                out.append(t)
        return out

    def try_claim(self, task: int | str, worker: str) -> Claim | None:
        """Claim ``task``, taking over a stale claim if needed; ``None`` if another worker holds it."""
        path = self.claim_path(task)
        token = uuid.uuid4().hex
        now = self.store.now()
        body = {"worker": worker, "token": token, "attempt": 1, "claimed_at": now, "heartbeat": now}
        try:
            generation = self.store.create(path, dumps(body))
            return Claim(path, _task_name(task), token, 1, generation, now)
        except PreconditionFailed:
            pass
        info = self.store.stat(path)
        if info is None or self.store.now() - info.updated <= self.config.stale_seconds:
            return None
        if self.store.stat(self.done_path(task)) is not None:
            return None
        try:
            previous = json.loads(self.store.read(path, generation=info.generation))
        except (PreconditionFailed, FileNotFoundError):
            return None
        attempt = int(previous.get("attempt", 1)) + 1
        if attempt > self.config.max_attempts:
            logger.error(
                "task %s/%s is POISONED: %d attempts died without finishing; leaving it for a human",
                self.stage_dir,
                _task_name(task),
                attempt - 1,
            )
            return None
        body.update(attempt=attempt, previous_worker=previous.get("worker"))
        try:
            generation = self.store.replace(path, dumps(body), info.generation)
        except PreconditionFailed:
            return None
        logger.warning("took over stale claim %s (attempt %d, previous owner %s)", path, attempt, previous.get("worker"))
        return Claim(path, _task_name(task), token, attempt, generation, now)

    def heartbeat(self, claim: Claim, worker: str) -> None:
        """Refresh ``claim``. Raises :class:`PreconditionFailed` if another worker has taken it over."""
        now = self.store.now()
        body = {
            "worker": worker,
            "token": claim.token,
            "attempt": claim.attempt,
            "claimed_at": claim.started,
            "heartbeat": now,
        }
        claim.generation = self.store.replace(claim.path, dumps(body), claim.generation)

    def complete(self, claim: Claim, result: dict) -> None:
        """Record ``result`` as the task's outcome. A done mark from a concurrent owner wins."""
        payload = {**result, "_task": claim.task, "_attempt": claim.attempt, "_finished": self.store.now()}
        payload.setdefault("_started", claim.started)
        try:
            self.store.create(self.done_path(claim.task), dumps(payload))
        except PreconditionFailed:
            logger.warning("%s already done by another owner; keeping its mark", claim.path)

    def is_sealed(self) -> bool:
        return self.store.stat(self.sealed_path) is not None

    def sealed_result(self) -> dict:
        return read_json(self.store, self.sealed_path)

    def done_results(self, threads: int = 64) -> list[dict]:
        """Every task's done mark, in task order. Raises if any is missing."""
        with ThreadPoolExecutor(threads) as pool:
            return list(pool.map(lambda t: read_json(self.store, self.done_path(t)), range(self.n_tasks)))


class Heartbeat:
    """Background heartbeat for a held claim, with a hard deadline.

    If the claim is lost (another worker took it over) or the task runs past ``deadline_seconds``,
    the process exits immediately: a task that cannot finish or no longer owns its outputs must
    not keep writing. The worker supervisor restarts the slot, and the claim goes stale and is
    retried elsewhere.
    """

    def __init__(self, board: TaskBoard, claim: Claim, worker: str, deadline_seconds: float) -> None:
        self._board = board
        self._claim = claim
        self._worker = worker
        self._deadline = time.monotonic() + deadline_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"heartbeat-{claim.task}")

    def __enter__(self) -> Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def _run(self) -> None:
        interval = self._board.config.heartbeat_seconds
        failures = 0
        while not self._stop.wait(interval):
            if time.monotonic() > self._deadline:
                logger.error("task %s passed its deadline; exiting so it is retried", self._claim.path)
                os._exit(75)
            try:
                self._board.heartbeat(self._claim, self._worker)
                failures = 0
            except PreconditionFailed:
                logger.error("lost claim %s to another worker; abandoning the task", self._claim.path)
                os._exit(76)
            except Exception:
                failures += 1
                logger.exception("heartbeat for %s failed (%d in a row)", self._claim.path, failures)
                if failures * interval > self._board.config.stale_seconds / 2:
                    logger.error("cannot heartbeat %s; exiting before the claim goes stale", self._claim.path)
                    os._exit(77)
