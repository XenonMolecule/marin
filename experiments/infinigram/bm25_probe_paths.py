# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-cluster probe: does each target's data exist by PLAIN glob (ignore confighash)?

The registry's ``.under()`` prefix sources only resolve when their wildcard dir is
a ``name-*`` confighash segment; a plain glob like ``300warcs*`` or a concrete
``data-*`` leaf fails ``resolve_prefix`` even if the files exist. This probe globs
each source pattern directly (fsspec treats ``*`` as a normal wildcard) and lists
the immediate children of each target's base dir, so we can tell "data absent"
apart from "data present but the registry pattern can't resolve it". Raises with a
compact summary for bug-report capture (finelog is down).
"""

import logging

from experiments.fsspec_paths import fsspec_glob
from experiments.infinigram.targets import all_targets

logger = logging.getLogger(__name__)


def _plain_glob_count(target) -> int:
    src = target.source
    patterns = src.globs if src.globs else (src.prefix,)
    total = 0
    for p in patterns:
        try:
            total += len(fsspec_glob(p))
        except Exception as e:
            logger.info("%s glob %s error: %s", target.name, p, e)
    return total


def _base_dir(pattern: str) -> str:
    """Everything up to the first path segment containing a wildcard."""
    parts = pattern.split("/")
    keep = []
    for seg in parts:
        if "*" in seg:
            break
        keep.append(seg)
    return "/".join(keep)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    plain: dict[str, int] = {}
    absent_children: dict[str, list] = {}
    for t in all_targets():
        src = t.source
        pattern = src.globs[0] if src.globs else src.prefix
        n = _plain_glob_count(t)
        plain[t.name] = n
        base = _base_dir(pattern)
        try:
            children = sorted({c.rstrip("/").split("/")[-1] for c in fsspec_glob(base.rstrip("/") + "/*")})[:12]
        except Exception as e:
            children = [f"<ls error: {str(e)[:40]}>"]
        logger.info("PROBE %s plain_glob=%d base=%s children=%s", t.name, n, base, children)
        if n == 0:
            # Only the trailing base segment + its children -- what actually exists
            # near where the data is expected (reveals a sibling-path landing).
            absent_children[t.name] = [base.split("/")[-1] + "/", *children]
    present = {n: c for n, c in plain.items() if c > 0}
    raise RuntimeError(f"PROBE_DONE present={present} absent_children={absent_children}")


if __name__ == "__main__":
    main()
