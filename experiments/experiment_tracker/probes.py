# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe primitives for the experiment tracker.

A probe measures one experiment's progress at refresh time and returns a `Reading`.
Probes run ONLY on explicit refresh (the dashboard has no polling loop), so each may
cost a handful of GCS listings or one iris CLI call. New experiment types plug in by
composing these, or by passing any zero-arg callable returning a `Reading` to
`Experiment.probe` (see registry.py).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field

import fsspec

logger = logging.getLogger(__name__)


def gcs_fs():
    """A GCS filesystem whose listings are never cached.

    The dashboard is a long-running process; fsspec's instance + listings caches would
    otherwise hide every directory created after startup (fit dirs vanished this way on
    2026-09-13 — fresh fits showed 0/4 until restart)."""
    return fsspec.filesystem("gcs", skip_instance_cache=True, use_listings_cache=False)


# Swarm artifact leaves carry the proxy index as ``-iNNNN-``. Counting raw files
# double-counts runs that completed (or were evaluated) in more than one region, so
# swarm probes count UNIQUE indices — the union-not-sum rule.
_SWARM_INDEX_RE = re.compile(r"-i(\d{4})-")


@dataclass
class Reading:
    """One measurement of an experiment: progress plus optional breakdown rows."""

    done: int | None = None
    total: int | None = None
    status: str = "running"  # running | blocked | done | failed | manual
    note: str = ""
    # (label, done, total) rows rendered under the headline bar, e.g. per-corpus counts.
    breakdown: list[tuple[str, int, int | None]] = field(default_factory=list)

    def finalize(self) -> Reading:
        if self.done is not None and self.total is not None and self.done >= self.total:
            self.status = "done"
        return self


def gcs_count(buckets: list[str], subpath: str, contains: str = "", total: int | None = None) -> Reading:
    """Count entries under `<bucket>/<subpath>` whose leaf name contains `contains`."""
    fs = gcs_fs()
    n = 0
    for bucket in buckets:
        try:
            n += sum(1 for p in fs.ls(f"{bucket}/{subpath}", detail=False) if contains in p.rsplit("/", 1)[-1])
        except FileNotFoundError:
            continue
    return Reading(done=n, total=total).finalize()


def gcs_count_by(
    buckets: list[str],
    subpath: str,
    groups: dict[str, str],
    per_group_total: int | None = None,
    unique_index: bool = False,
    require_file: str = "",
) -> Reading:
    """Count entries per group (label -> leaf-substring), with a summed headline.

    ``unique_index=True`` deduplicates by the ``-iNNNN-`` swarm index across buckets;
    use it for any swarm namespace, where cross-region duplicates inflate raw sums.
    ``require_file`` counts a run dir only if it contains that file (e.g.
    ``"results.json"``) — an eval that created its dir but died before writing results
    must not count as done (resiliparse i0183 burned a fit this way on 2026-09-13).
    """
    fs = gcs_fs()
    seen: dict[str, set[str] | int] = {label: set() if unique_index else 0 for label in groups}
    for bucket in buckets:
        try:
            if require_file:
                found = fs.find(f"{bucket}/{subpath}", maxdepth=2)
                leaves = [p.rsplit("/", 2)[-2] for p in found if p.rsplit("/", 1)[-1] == require_file]
            else:
                leaves = [p.rsplit("/", 1)[-1] for p in fs.ls(f"{bucket}/{subpath}", detail=False)]
        except FileNotFoundError:
            continue
        for label, needle in groups.items():
            for leaf in leaves:
                if needle not in leaf:
                    continue
                if unique_index:
                    m = _SWARM_INDEX_RE.search(leaf)
                    if m:
                        seen[label].add(m.group(1))
                else:
                    seen[label] += 1
    counts = {label: len(v) if unique_index else v for label, v in seen.items()}
    breakdown = [(label, n, per_group_total) for label, n in counts.items()]
    total = per_group_total * len(groups) if per_group_total else None
    return Reading(done=sum(counts.values()), total=total, breakdown=breakdown).finalize()


def gcs_exists_checklist(items: dict[str, str]) -> Reading:
    """Each item is (label -> gs:// path); done = how many exist."""
    fs = gcs_fs()
    breakdown = []
    for label, path in items.items():
        ok = fs.exists(path.removeprefix("gs://"))
        breakdown.append((label, int(ok), 1))
    done = sum(d for _, d, _ in breakdown)
    return Reading(done=done, total=len(items), breakdown=breakdown).finalize()


def iris_job_states(prefixes: list[str], cluster: str = "marin", limit: int = 300) -> Reading:
    """Aggregate child states under iris job prefixes; done = succeeded count."""
    counts: dict[str, int] = {}
    for prefix in prefixes:
        out = subprocess.run(
            ["uv", "run", "iris", "--cluster", cluster, "job", "list", "--prefix", prefix, "--limit", str(limit)],
            capture_output=True,
            text=True,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("/"):
                counts[parts[1]] = counts.get(parts[1], 0) + 1
    breakdown = [(state, n, None) for state, n in sorted(counts.items())]
    note = "" if counts else "no jobs matched (coordinator retired?)"
    return Reading(done=counts.get("succeeded", 0), total=sum(counts.values()) or None, breakdown=breakdown, note=note)


def manual(done: int | None = None, total: int | None = None, status: str = "manual", note: str = "") -> Reading:
    """A hand-maintained entry; edit its numbers in registry.py as the experiment moves."""
    return Reading(done=done, total=total, status=status, note=note)


def local_json_field(path: str, extract) -> Reading:
    """Read a local JSON file and let `extract(doc)` build the Reading (arbitrary hook)."""
    try:
        with open(path) as fh:
            return extract(json.load(fh))
    except FileNotFoundError:
        return Reading(status="blocked", note=f"missing {path}")


def timed(probe) -> tuple[Reading, float]:
    t0 = time.time()
    try:
        reading = probe()
    except Exception as e:  # a broken probe must not take down the whole refresh
        logger.exception("probe failed")
        reading = Reading(status="failed", note=f"probe error: {type(e).__name__}: {e}")
    return reading, time.time() - t0
