# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared parallel job-submit helper for the fast_curation launchers.

Submitting hundreds of Iris jobs serially (each `iris job run` re-establishes an SSH tunnel,
~8s) would take ~hour for a full fleet. Submit them concurrently with a bounded thread pool.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

MAX_CONCURRENT_SUBMITS = 8


def submit_workers(args: argparse.Namespace, build_command: Callable[[argparse.Namespace, int], list[str]]) -> None:
    """Build + submit ``args.num_workers`` jobs (seeds ``seed_start..+N``), bounded-concurrently."""
    cmds = [(args.seed_start + i, build_command(args, args.seed_start + i)) for i in range(args.num_workers)]
    if args.dry_run:
        for seed, cmd in cmds:
            logger.info("worker %d:\n  %s", seed, " ".join(cmd))
        return

    def _submit(item: tuple[int, list[str]]) -> bool:
        seed, cmd = item
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("worker %d submit FAILED: %s", seed, (result.stderr or result.stdout or "")[-400:])
            return False
        return True

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SUBMITS) as ex:
        ok = sum(ex.map(_submit, cmds))
    logger.info("submitted %d/%d workers", ok, len(cmds))
