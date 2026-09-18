# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Project ``kept/`` -> ``kept_text/`` IN-REGION, dropping the throwaway ModernBERT ``input_ids``.

The kept parquet carries per-doc ``input_ids`` (the tokens ModernBERT scored) plus ``n_tokens``.
Those are ~half the on-disk bytes and useless downstream: dedup/tokenize need only ``text``, and the
training corpus is re-tokenized with Llama 3. This rewrites each region's local ``kept/`` into a
compact ``kept_text/`` (text + provenance only) so the later consolidation to the training region
egresses ~half the bytes.

Region-local — reads AND writes the SAME regional bucket, so it incurs ZERO cross-region egress.
Run one job per region; it fans out over that region's kept files with a process pool. Idempotent
(skip-existing) so it's freely resumable.

    python -m experiments.fast_curation.project_text_only --bucket gs://marin-us-east5
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp

import fsspec

from experiments.fast_curation import batch_format
from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)

# Drop the ragged input_ids and its n_tokens; keep text (the payload) + provenance + every stage
# score the spec's kept schema carries (pooled specs add pooled_prob). `read_table` projects at read
# time, so input_ids is never even pulled off disk.
DROP_COLUMNS = ("input_ids", "n_tokens")


def keep_columns(spec) -> list[str]:
    return [name for name in batch_format.kept_schema_for(spec).names if name not in DROP_COLUMNS]


def _project_one(paths: tuple[str, str, tuple[str, ...]]) -> int:
    src, dst, columns = paths
    if fsspec.filesystem("gcs").exists(dst):
        return 0  # already projected (resumable)
    table = batch_format.read_table(src, columns=list(columns))
    batch_format.write_table(dst, table)
    return table.num_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v3")
    ap.add_argument("--bucket", required=True, help="Regional bucket to project IN-PLACE, e.g. gs://marin-us-east5.")
    ap.add_argument("--num-procs", type=int, default=16)
    args = ap.parse_args()

    spec = get_spec(args.spec)
    kept_prefix = spec.namespace(args.bucket) + "/kept"
    out_prefix = spec.namespace(args.bucket) + "/kept_text"
    listing = fsspec.filesystem("gcs").ls(kept_prefix.removeprefix("gs://"))
    files = [f"gs://{p}" for p in listing if p.endswith(".parquet")]
    columns = tuple(keep_columns(spec))
    jobs = [(f, f"{out_prefix}/{f.rsplit('/', 1)[1]}", columns) for f in files]
    logger.info("projecting %d kept files -> kept_text (drop input_ids) in %s", len(jobs), args.bucket)

    # spawn (not fork): the workers create fresh gcsfs handles; forking a live gcsfs event loop deadlocks.
    with mp.get_context("spawn").Pool(args.num_procs) as pool:
        counts = pool.map(_project_one, jobs)
    logger.info("done: %d files, %d docs -> %s", len(jobs), sum(counts), out_prefix)


if __name__ == "__main__":
    main()
