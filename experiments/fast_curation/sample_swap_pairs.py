# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print random (arm A, arm B) text pairs for the same documents, read from the stored corpora.

For each of ``--n`` random WARCs, one random record is drawn from B's shard; its A text is
fetched from A's post-decon tree via the ``a_shard``/``a_row`` recorded in the allow-list. Nothing
is re-extracted. Run in-region and read the job log.

    python -m experiments.fast_curation.sample_swap_pairs --n 10 --seed 0
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random

import fsspec
import pyarrow.parquet as pq

from experiments.baseline_collection.decode_warcs_clean import _load_manifest, _warc_path_hash
from experiments.fast_curation.build_swap_allowlist import A_TREE
from experiments.fast_curation.swap_extractor import OUTPUT_ROOT, shard_path


def _records(path: str):
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chars", type=int, default=700)
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--root", default=OUTPUT_ROOT)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    warcs = rng.sample(_load_manifest(args.manifest), args.n)
    for k, warc in enumerate(warcs, 1):
        h = _warc_path_hash(warc)
        b_rows = list(_records(shard_path(args.root, h)))
        b = rng.choice(b_rows)
        with fsspec.open(f"{args.root}/allowlist/data-{h}.parquet", "rb") as fh:
            allow = pq.read_table(io.BytesIO(fh.read()), columns=["doc_id", "a_shard", "a_row"]).to_pylist()
        loc = next(r for r in allow if r["doc_id"] == b["doc_id"])
        a = next(r for i, r in enumerate(_records(f"{A_TREE}/{loc['a_shard']}.jsonl.gz")) if i == loc["a_row"])
        assert a["doc_id"] == b["doc_id"]
        print("=" * 110)
        print(f"[{k}/{args.n}] WARC {h} ({warc.rsplit('/', 3)[1]})  doc_id {b['doc_id']}")
        print(f"URL: {b['url']}")
        print(f"A (fork, markdown): {len(a['text'])} chars   B (upstream resiliparse): {len(b['text'])} chars")
        print("--- A ---")
        print(a["text"][: args.chars])
        print("--- B ---")
        print(b["text"][: args.chars])
    print("=" * 110)
    print("DONE")


if __name__ == "__main__":
    main()
