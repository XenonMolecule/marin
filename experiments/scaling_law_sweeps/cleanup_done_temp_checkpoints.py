# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reclaim GCS by deleting rolling TEMP checkpoints for FINISHED curation runs.

Levanter writes 15-min rolling preemption-recovery checkpoints to a per-region
ttl=14d temp path:
    gs://marin-{region}/tmp/ttl=14d/checkpoints-temp/.../isoflop-curation/{run}/
Once a run is DONE, that rolling temp is dead weight. The bucket lifecycle reaps
it after 14 days, but a big cell (~12 GB at d2432, more at d3584) holds that
budget for two weeks for nothing.

This deletes ONLY temp dirs whose run has a `.data_curation_DONE` marker in the
PERMANENT path (where the force-saved FINAL checkpoint + hf export live). It:
  - NEVER touches a permanent `.../checkpoints/isoflop-curation/{run}/` path,
  - NEVER touches a run without a DONE marker (a still-running recovery ckpt),
  - ONLY removes paths containing both `tmp/ttl=` and `checkpoints-temp`.

Dry-run by default; pass --execute to actually delete. Metadata-only ops
(list/du/rm) so it's cheap to run cross-region.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from datetime import datetime, timezone

from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PERM_PREFIX = "checkpoints/isoflop-curation"
TEMP_MID = "tmp/ttl="  # guard: a temp path MUST contain this ...
TEMP_TAG = "checkpoints-temp"  # ... and this, else we refuse to delete.
# Second, independent gate: never delete a temp dir touched within this window —
# a still-live (or just-restarted) run writes its rolling checkpoint every 15 min,
# so anything younger than this could conceivably be active. Belt-and-suspenders
# on top of the DONE-marker gate, so we never interfere with a running model.
MIN_AGE_MINUTES = 60.0


def _ls(pattern: str) -> list[str]:
    """`gcloud storage ls <pattern>` -> gs:// lines (empty on no-match / error).

    gcsfs hits SSL cert-verification failures on this host; gcloud auth works,
    so all listing goes through the CLI.
    """
    out = subprocess.run(["gcloud", "storage", "ls", pattern], capture_output=True, text=True, timeout=300)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip().startswith("gs://")]


def done_run_names() -> set[str]:
    """Run names with a `.data_curation_DONE` marker in any region's permanent path."""
    done: set[str] = set()
    for bucket in REGION_TO_BUCKET.values():
        for marker in _ls(f"{bucket}/{PERM_PREFIX}/*/.data_curation_DONE"):
            done.add(marker.rstrip("/").split("/")[-2])  # .../{run}/.data_curation_DONE
    return done


def temp_run_dirs() -> list[str]:
    """All rolling-temp checkpoint run dirs (gs:// urls) across every region."""
    dirs: list[str] = []
    for bucket in REGION_TO_BUCKET.values():
        # ls the isoflop-curation/ dir (inner bucket name is the wildcard); children are run dirs.
        for d in _ls(f"{bucket}/tmp/ttl=14d/checkpoints-temp/*/checkpoints/isoflop-curation/"):
            if d.endswith("/") and "isoflop-curation/" in d:
                dirs.append(d)
    return dirs


def du_gb(url: str) -> float:
    try:
        out = subprocess.run(["gcloud", "storage", "du", "-s", url], capture_output=True, text=True, timeout=180)
        tok = out.stdout.split()
        return (int(tok[0]) / 1e9) if tok else 0.0
    except Exception:
        return 0.0


def newest_age_minutes(url: str) -> float:
    """Minutes since the newest object under `url` was written.

    Returns +inf if the dir is empty or unreadable -> the recency gate then
    treats it as old. We only ever DELETE when this is large, and a read error
    making it look old is harmless because the DONE-marker gate already passed.
    """
    out = subprocess.run(
        ["gcloud", "storage", "ls", "--long", "--recursive", url], capture_output=True, text=True, timeout=180
    )
    newest: datetime | None = None
    for line in out.stdout.splitlines():
        tok = line.split()
        if len(tok) < 3 or not tok[-1].startswith("gs://"):
            continue
        try:
            ts = datetime.fromisoformat(tok[1].replace("Z", "+00:00"))
        except ValueError:
            continue
        if newest is None or ts > newest:
            newest = ts
    if newest is None:
        return float("inf")
    return (datetime.now(timezone.utc) - newest).total_seconds() / 60.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--execute", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--size", action="store_true", help="compute per-dir sizes (slower)")
    args = ap.parse_args()

    done = done_run_names()
    temp = temp_run_dirs()
    logger.info("done runs (DONE marker): %d | temp run dirs found: %d", len(done), len(temp))

    to_delete: list[str] = []
    not_done = 0
    too_young: list[str] = []
    for d in temp:
        if TEMP_MID not in d or TEMP_TAG not in d:  # gate 0: never touch a non-temp path
            continue
        run = d.rstrip("/").split("/")[-1]
        if run not in done:  # gate 1: must have a DONE marker (finished)
            not_done += 1
            continue
        if newest_age_minutes(d) < MIN_AGE_MINUTES:  # gate 2: must be idle (not live)
            too_young.append(d)
            continue
        to_delete.append(d)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(
        f"\n=== {mode}: {len(to_delete)} finished+idle temp dirs to remove | "
        f"{not_done} skipped (no DONE marker / running) | {len(too_young)} skipped (active <{MIN_AGE_MINUTES:.0f}min) ==="
    )
    for d in too_young:
        print(f"  SKIP (recently active)  {d}")
    total = 0.0
    for d in to_delete:
        gb = du_gb(d) if (args.size or args.execute) else 0.0
        total += gb
        print(f"  {'rm  ' if args.execute else 'would rm'} {gb:6.2f}GB  {d}")
        if args.execute:
            # belt-and-suspenders guard immediately before the irreversible delete
            assert TEMP_MID in d and TEMP_TAG in d and "isoflop-curation" in d, f"GUARD FAILED, refusing: {d}"
            subprocess.run(["gcloud", "storage", "rm", "--recursive", d], check=False)
    print(f"\nTOTAL: {total:.1f} GB across {len(to_delete)} dirs {'reclaimable' if not args.execute else 'deleted'}")


if __name__ == "__main__":
    main()
