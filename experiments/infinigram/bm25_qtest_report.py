# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the JSON results written by bm25_query_test and summarize them.

bm25-free (so its final raised line is not clobbered by bm25s's atexit output),
runnable in any region -- the result files are tiny. Raises with a per-index
top-url-per-query summary for bug-report capture while finelog is down.
"""

import json
import logging

import fsspec
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

_RESULT_GLOBS = [f"gs://marin-{r}/bm25_qtest_results/*.json" for r in ("us-central1", "us-central2", "us-east5")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    summary = {}
    for g in _RESULT_GLOBS:
        for path in fsspec_glob(g):
            with fsspec.open(path, "r") as f:
                d = json.load(f)
            name = f"{d['dataset']}-{d['collection']}"
            per_q = {q: {"h": r["hits"], "url": r["top_url"]} for q, r in d["queries"].items()}
            logger.info("QTEST %s docs=%s %s", name, d.get("num_docs"), json.dumps(per_q))
            summary[name] = {"docs": d.get("num_docs"), "q": per_q}
    raise RuntimeError(f"QTEST_REPORT {json.dumps(summary)}")


if __name__ == "__main__":
    main()
