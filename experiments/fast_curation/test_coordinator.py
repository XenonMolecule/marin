# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator control loop: delivery-gated scale-up, death backfill, finite-work exit."""

from __future__ import annotations

import argparse

from iris.rpc import job_pb2

from experiments.fast_curation import coordinator


def _args(**over):
    base = dict(
        mode="fused",
        spec="lpv11_fastpipe_v2_1_fused",
        tpu_type="v6e-8",
        region="us-east1",
        max_count=6,
        initial_batch=2,
        chunk_size=2,
        check_interval=0,
        patience=2,
        seed_start=0,
        max_shard=None,
        manifest="m",
        a_procs=2,
        extract_procs_per_worker=2,
        queue_depth=2,
        batch_size=8,
        claim_stale_hours=0.2,
        child_max_idle_passes=5,
        child_cpu=1.0,
        child_memory="1GB",
    )
    base.update(over)
    return argparse.Namespace(**base)


class _FakeClient:
    """Submissions always succeed; per-job states follow a mutable script."""

    def __init__(self):
        self.states: dict[str, int] = {}
        self.n = 0

    def submit(self, **kw):
        self.n += 1
        jid = f"/u/{kw['name']}"
        self.states[jid] = job_pb2.JOB_STATE_RUNNING

        class J:
            job_id = jid

        return J()

    def status(self, job_id):
        class S:
            state = self.states[str(job_id)]

        return S()


def test_scales_to_ceiling_then_backfills_deaths(monkeypatch):
    """All-running -> chunked scale-up to max_count; killed children free slots -> backfill."""
    client = _FakeClient()
    ticks = iter(range(20))
    done_after = 6

    monkeypatch.setattr(coordinator.time, "sleep", lambda s: None)
    monkeypatch.setattr(coordinator, "region_work_done", lambda *a: next(ticks) >= done_after)
    monkeypatch.setattr(coordinator, "get_spec", lambda s: object())

    # After the loop reaches the ceiling (6 live), kill two children on the 4th check.
    orig_count = coordinator._count_child_states
    calls = {"n": 0}

    def counting(client_, ids):
        calls["n"] += 1
        if calls["n"] == 4:
            for jid in list(client.states)[:2]:
                client.states[jid] = job_pb2.JOB_STATE_KILLED
        return orig_count(client_, ids)

    monkeypatch.setattr(coordinator, "_count_child_states", counting)
    coordinator.run(client, _args())
    # 2 initial + 2 + 2 to hit ceiling, then 2 backfilled after the kills.
    assert client.n == 8, f"expected ceiling fill + backfill, submitted {client.n}"


def test_pending_children_block_scale_up(monkeypatch):
    """A pool that never delivers must never receive more than the initial batch + probes."""
    client = _FakeClient()

    def submit_pending(**kw):
        j = _FakeClient.submit(client, **kw)
        client.states[j.job_id] = job_pb2.JOB_STATE_PENDING
        return j

    client.submit = submit_pending
    ticks = iter(range(8))
    monkeypatch.setattr(coordinator.time, "sleep", lambda s: None)
    monkeypatch.setattr(coordinator, "region_work_done", lambda *a: next(ticks) >= 6)
    monkeypatch.setattr(coordinator, "get_spec", lambda s: object())
    coordinator.run(client, _args())
    assert client.n == 2, f"pending pool must not scale, submitted {client.n}"
