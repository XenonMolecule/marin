# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic cross-region claims on swarm indices, so two coordinators cannot train the same run.

Every duplicate-work incident in this project came from the same shape: a coordinator decides
what to dispatch by READING state (``already_done``, its own ``todo`` list) and then WRITING a
job some time later. Between the read and the write, another coordinator can make the same
decision. Widening the read (checking all regions instead of one) shrinks the window but cannot
close it, because the window is inherent to check-then-act.

Measured cost of not fixing this: in one hour, 53 of 57 completions were re-runs of finished
work, and 21 run names ended up trained twice across regions. The swarm was largely recomputing
itself while the cluster was fully occupied.

**The fix is an atomic claim, not a better check.** GCS object creation with
``ifGenerationMatch=0`` succeeds for exactly one caller when several race to create the same
object -- gcsfs exposes it as ``pipe_file(..., mode="create")``. So a coordinator must win a
claim before it may submit, and losing the race is normal, not an error.

Two design points that matter:

* **One global registry, not one per region.** Claims live in a single bucket regardless of
  where the coordinator or the data sits. A per-region registry would reintroduce exactly the
  blindness it exists to prevent -- that is the bug that let 20 hq indices train twice. Reading
  a few KB cross-region is irrelevant next to a 1.5-hour TPU run.
* **Claims expire.** A coordinator can die holding claims; without a TTL those indices would be
  unrunnable forever, which is worse than duplicating them. A stale claim is deleted and then
  re-created atomically, so only one stealer wins even if several notice staleness together.

A claim is an intent to run, NOT a record of completion. Completion is still the results JSON.
A run that fails releases nothing explicitly -- its claim simply ages out and becomes stealable.
"""

from __future__ import annotations

import json
import logging
import time

import fsspec

logger = logging.getLogger(__name__)

# Single global registry. Deliberately hardcoded to ONE bucket: the whole point is that every
# coordinator, in every region, consults the same namespace.
CLAIM_ROOT = "gs://marin-us-central1/metadata/olmix_claims"

# A claim older than this is presumed abandoned. Sized well above a run's wall clock (~1.5-4h)
# plus preemption/resume churn, so a healthy long run is never stolen out from under itself.
CLAIM_TTL_SECONDS = 8 * 3600


def claim_path(corpus: str, run_name: str, root: str = CLAIM_ROOT) -> str:
    return f"{root.rstrip('/')}/{corpus}/{run_name}.json"


def _atomic_create(fs, path: str, payload: bytes) -> bool:
    """Create `path` only if absent. True if we created it, False if someone else had it.

    Uses the GCS generation precondition via gcsfs's ``mode="create"``. Filesystems without
    that support (the in-memory one used by tests) fall back to exists-then-write, which is
    NOT atomic -- acceptable for a single-threaded test, never for production.
    """
    # Object stores need no parents; a local filesystem (tests) does.
    parent = path.rsplit("/", 1)[0]
    try:
        fs.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    try:
        fs.pipe_file(path, payload, mode="create")
        return True
    except FileExistsError:
        # Local filesystems signal a lost race this way; GCS uses a 412 below.
        return False
    except TypeError:
        # Filesystem does not support the create precondition at all. Not atomic -- fine for a
        # single-threaded test, never safe in production, so say so loudly.
        logger.warning("%s does not support atomic create; claim is best-effort only", type(fs).__name__)
        if fs.exists(path):
            return False
        fs.pipe_file(path, payload)
        return True
    except Exception as exc:
        if "412" in str(exc) or "conditionNotMet" in str(exc) or "PreconditionFailed" in type(exc).__name__:
            return False
        raise


def try_claim(
    corpus: str,
    run_name: str,
    owner: str,
    root: str = CLAIM_ROOT,
    ttl: float = CLAIM_TTL_SECONDS,
    now: float | None = None,
) -> bool:
    """Attempt to claim one index. True means this caller may submit it; False means skip.

    Losing is the normal outcome under contention and is not an error.
    """
    now = time.time() if now is None else now
    path = claim_path(corpus, run_name, root)
    fs, _ = fsspec.core.url_to_fs(path)
    payload = json.dumps({"owner": owner, "claimed_at": now, "run_name": run_name}).encode()

    if _atomic_create(fs, path, payload):
        return True

    try:
        existing = json.loads(fs.cat_file(path))
    except Exception:
        return False
    age = now - float(existing.get("claimed_at", 0))
    if age <= ttl:
        return False

    logger.warning(
        "claim on %s is %.1fh old (owner %s); treating as abandoned and re-claiming",
        run_name,
        age / 3600,
        existing.get("owner"),
    )
    try:
        fs.rm_file(path)
    except Exception:
        pass
    return _atomic_create(fs, path, payload)


def claimed_run_names(corpus: str, root: str = CLAIM_ROOT, ttl: float = CLAIM_TTL_SECONDS, now: float | None = None):
    """Run names with a LIVE claim, for reporting. Not a substitute for `try_claim`."""
    now = time.time() if now is None else now
    prefix = f"{root.rstrip('/')}/{corpus}"
    fs, path = fsspec.core.url_to_fs(prefix)
    if not fs.exists(path):
        return set()
    live = set()
    for p in fs.ls(path, detail=False):
        if not p.endswith(".json"):
            continue
        try:
            d = json.loads(fs.cat_file(p))
        except Exception:
            continue
        if now - float(d.get("claimed_at", 0)) <= ttl:
            live.add(p.rsplit("/", 1)[-1][: -len(".json")])
    return live


def reap_dead_claims(corpus: str, live_run_names: set[str], done_run_names: set[str], root: str = CLAIM_ROOT) -> int:
    """Release claims whose run is neither live nor finished. Returns how many were freed.

    The TTL alone is not enough. A claim is taken at SUBMIT time, but a child can die minutes
    later -- a preempted worker took 30 of them at once in one observed case -- and the claim
    then blocks any retry for the full TTL even though nothing is running. Age is the wrong
    signal for that; liveness is the right one.

    Deliberately conservative: a claim is only released when the caller can prove the run is
    absent from BOTH the live job list and the completed results. Releasing a claim whose run
    is actually still training would let a second coordinator duplicate it, which is the exact
    failure this module exists to prevent.
    """
    prefix = f"{root.rstrip('/')}/{corpus}"
    fs, path = fsspec.core.url_to_fs(prefix)
    if not fs.exists(path):
        return 0
    freed = 0
    for p in fs.ls(path, detail=False):
        if not p.endswith(".json"):
            continue
        name = p.rsplit("/", 1)[-1][: -len(".json")]
        if name in live_run_names or name in done_run_names:
            continue
        try:
            fs.rm_file(p)
            freed += 1
        except Exception as exc:
            logger.warning("could not release stale claim %s: %s", name, exc)
    if freed:
        logger.warning("released %d claims whose runs are neither live nor complete", freed)
    return freed
