# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off: locate the resiliparse / fineweb_cc training document tiers in GCS.

Their expected paths came up empty, so list what actually exists under the
candidate parent dirs (two levels) to tell "wrong leaf" apart from "documents
cleaned up after tokenization". Raises with the listing for bug-report capture.
"""

import logging

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

PARENTS = [
    "gs://marin-us-central2/documents/baseline_resiliparse_deduped",
    "gs://marin-us-central2/documents/baseline_fineweb_cc_deduped",
    "gs://marin-us-central2/documents/baseline_fineweb_cc",
    "gs://marin-us-central2/documents/baseline_resiliparse_decon",
]


def _children(prefix: str, depth: int = 2) -> list[str]:
    """Immediate child names, then grandchildren for any dir that looks like Nwarcs."""
    out = []
    lvl1 = sorted({c.rstrip("/").split("/")[-1] for c in fsspec_glob(prefix.rstrip("/") + "/*")})
    for c in lvl1[:15]:
        out.append(c)
        if depth > 1 and "warc" in c:
            grand = sorted({g.rstrip("/").split("/")[-1] for g in fsspec_glob(f"{prefix.rstrip('/')}/{c}/*")})
            out.append(f"    {c}/ -> {grand[:12]}")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    summary: dict[str, list] = {}
    for p in PARENTS:
        try:
            kids = _children(p)
        except Exception as e:
            kids = [f"<err: {str(e)[:50]}>"]
        summary[p.split("/")[-1]] = kids
        logger.info("RESI_PROBE %s -> %s", p, kids)
    raise RuntimeError(f"RESI_DONE {summary}")


if __name__ == "__main__":
    main()
