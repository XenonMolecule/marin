# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-cluster status probe: read every uploaded BM25 manifest and summarize it.

finelog logs are unreadable while the log service is down, but a *failed* job's
last stderr line surfaces in ``iris job bug-report``. So this loads each target's
``manifest.json`` (if built), prints its metrics, and then raises with a compact
one-line summary -- giving a build-status snapshot (doc counts, sub-index counts,
build seconds, and the verified top-hit url proving metadata round-trips) without
any log access. Run it as an in-region Iris job.

    iris job run --cluster marin --region us-central1 --extra cpu -- \
        python -m experiments.infinigram.bm25_status
"""

import json
import logging

import fsspec

from experiments.fsspec_paths import fsspec_exists
from experiments.infinigram.bm25_build import bm25_index_dir
from experiments.infinigram.bm25_query import MANIFEST_NAME
from experiments.infinigram.bm25_sources import all_bm25_targets as all_targets

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    built: dict[str, str] = {}
    for t in all_targets():
        url = f"{bm25_index_dir(t).rstrip('/')}/{MANIFEST_NAME}"
        if not fsspec_exists(url):
            continue
        with fsspec.open(url, "r") as f:
            m = json.load(f)
        logger.info(
            "BUILT %s: docs=%s sub_indices=%s index_MiB=%.1f build_s=%s top_url=%s",
            t.name,
            m.get("doc_count"),
            m.get("num_sub_indices"),
            (m.get("index_bytes") or 0) / 1024**2,
            m.get("build_seconds"),
            (m.get("smoke") or {}).get("top_url"),
        )
        idx_mib = (m.get("index_bytes") or 0) / 1024**2
        in_mib = (m.get("input_bytes") or 0) / 1024**2
        built[t.name] = (
            f"{m.get('doc_count')}docs "
            f"{idx_mib:.0f}MiB_idx "
            f"{in_mib:.0f}MiB_in "
            f"{m.get('num_sub_indices')}sub "
            f"ratio={m.get('index_to_input_ratio')}"
        )
    raise RuntimeError(f"BM25_STATUS built={len(built)}/{len(all_targets())} {built}")


if __name__ == "__main__":
    main()
