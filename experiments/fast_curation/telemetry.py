# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-worker progress heartbeats — the dashboard's scalable telemetry source.

The dashboard must scale to the 7M-WARC production run, where listing per-WARC outputs or the
central completed-registry every refresh (O(WARCs)) is far too slow/costly. Instead each worker
writes a tiny rolling heartbeat keyed by its slot, and the dashboard aggregates **O(workers)**.

Heartbeat path (central, so one read covers all regions, like the registry/claims):

    gs://marin-us-central1/{subdir}/_heartbeats/{phase}/{region}-seed{N}.json

**Monotonic across preemption.** A preemptible worker restarts as a new process for the same slot
``{region}-seed{N}``. On construction the heartbeat reads its own prior file and seeds the
cumulative counters from it, so the slot's ``warcs_done`` only ever grows. Central claims guarantee
each WARC is processed by exactly one slot, so summing ``warcs_done`` across slots is an accurate,
O(workers) done-count. The registry remains the correctness backstop; a crash loses < 1 heartbeat.

Telemetry is strictly best-effort: every GCS touch is guarded so a heartbeat failure can never fail
a WARC.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque

import fsspec

from experiments.fast_curation.spec import PipelineSpec

logger = logging.getLogger(__name__)

HEARTBEAT_ROOT_BUCKET = "gs://marin-us-central1"
# Write at most this often per slot (seconds). A WARC takes minutes, so we write on every WARC; this
# only throttles idle/poll ticks.
MIN_WRITE_INTERVAL = 30.0
RECENT_WALLS = 20  # rolling window of per-WARC wall times for instantaneous-rate cross-check


def _bucket_to_region() -> dict[str, str]:
    """Canonical ``gs://bucket`` -> region (so europe-west4's ``marin-eu-west4`` maps correctly)."""
    from rigging.filesystem import data_config

    return {f"gs://{s.name}": r for r, s in data_config().region_buckets.items()}


def region_from_bucket(bucket: str) -> str:
    """Resolve the canonical region name for a regional output bucket."""
    return _bucket_to_region().get(bucket.rstrip("/"), bucket.rsplit("marin-", 1)[-1].strip("/"))


class Heartbeat:
    """A worker's rolling progress beacon (one GCS object per slot).

    Construct once per worker; call :meth:`record_warc` after each completed WARC, :meth:`tick` on
    idle polls, and :meth:`close` on graceful exit.
    """

    def __init__(
        self,
        spec: PipelineSpec,
        *,
        phase: str,
        region: str,
        seed: int,
        kind: str,  # "cpu" | "tpu"
        now=time.time,
    ):
        self._now = now
        self._path = f"{HEARTBEAT_ROOT_BUCKET}/{spec.subdir()}/_heartbeats/{phase}/{region}-seed{seed}.json"
        self._meta = {
            "spec_id": spec.spec_id,
            "version": spec.version(),
            "phase": phase,
            "region": region,
            "seed": seed,
            "kind": kind,
        }
        self._started = now()
        prior = self._read_prior()
        pc = prior.get("cumulative", {}) if isinstance(prior, dict) else {}
        self._cum: dict = {
            "warcs_done": int(pc.get("warcs_done", 0)),
            "docs_in": int(pc.get("docs_in", 0)),
            "docs_out": int(pc.get("docs_out", 0)),
            "wall_seconds": float(pc.get("wall_seconds", 0.0)),
            "compute_seconds": dict(pc.get("compute_seconds", {})),
        }
        self._recent_walls: deque[float] = deque(maxlen=RECENT_WALLS)
        self._last_warc: str | None = None
        self._last_warc_at: float | None = None
        self._last_write = 0.0
        self._write("running")  # announce presence immediately

    def _read_prior(self) -> dict:
        try:
            with fsspec.open(self._path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def record_warc(
        self,
        *,
        warc_hash: str,
        docs_in: int,
        docs_out: int,
        wall_seconds: float,
        compute_seconds: dict[str, float],
        status: str = "running",
    ) -> None:
        self._cum["warcs_done"] += 1
        self._cum["docs_in"] += int(docs_in)
        self._cum["docs_out"] += int(docs_out)
        self._cum["wall_seconds"] += float(wall_seconds)
        for k, v in compute_seconds.items():
            self._cum["compute_seconds"][k] = self._cum["compute_seconds"].get(k, 0.0) + float(v)
        self._recent_walls.append(float(wall_seconds))
        self._last_warc = warc_hash
        self._last_warc_at = self._now()
        self._write(status)  # every WARC (minutes apart) -> cheap and fresh

    def tick(self, status: str) -> None:
        """Update liveness on an idle/poll cycle without a completed WARC (throttled)."""
        if self._now() - self._last_write >= MIN_WRITE_INTERVAL:
            self._write(status)

    def close(self, status: str = "done") -> None:
        self._write(status)

    def _write(self, status: str) -> None:
        payload = {
            **self._meta,
            "started_at": self._started,
            "updated_at": self._now(),
            "status": status,
            "cumulative": self._cum,
            "recent": {
                "warc_wall_seconds": list(self._recent_walls),
                "last_warc": self._last_warc,
                "last_warc_at": self._last_warc_at,
            },
        }
        try:
            with fsspec.open(self._path, "w") as f:
                json.dump(payload, f)
            self._last_write = self._now()
        except Exception as e:  # telemetry must never fail a WARC
            logger.warning("heartbeat write failed (%s): %s", self._path, e)
