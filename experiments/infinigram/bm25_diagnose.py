# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off in-cluster diagnostic: which BM25 targets actually resolve in GCS.

finelog logs are unreadable while the log service is down, but a *failed* job's
last stderr line is captured in its bug-report. So this prints per-target glob
counts and then raises with a compact summary, making the landed picture visible
via ``iris job bug-report``. Not part of the pipeline -- delete once the registry
landed flags are trusted again.

    iris job run --cluster marin --region us-central1 --extra cpu -- \
        python -m experiments.infinigram.bm25_diagnose
"""

import logging

from experiments.fsspec_paths import fsspec_glob
from experiments.infinigram.resolve import resolve_prefix
from experiments.infinigram.targets import all_targets

logger = logging.getLogger(__name__)


def _count(target) -> int:
    src = target.source
    try:
        if src.prefix is not None:
            return len(resolve_prefix(src.prefix))
        return sum(len(fsspec_glob(p)) for p in src.globs)
    except Exception as e:
        logger.info("%s glob error: %s", target.name, e)
        return -1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    counts = {t.name: _count(t) for t in all_targets()}
    for name, n in counts.items():
        logger.info("DIAG %s -> %s shards", name, n)
    landed = sorted(n for n, c in counts.items() if c and c > 0)
    # Compact final stderr line captured by bug-report.
    raise RuntimeError(f"DIAG_DONE landed={landed} counts={counts}")


if __name__ == "__main__":
    main()
