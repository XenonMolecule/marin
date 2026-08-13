# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restrict the GLOBALLY-deduped+deconned fastpipe 100% band to an N-WARC subset — WITHOUT re-dedup.

The full-pool `baseline_{spec}_decon_deduped/{POOL}warcs/deduped/` corpus is already globally fuzzy-
deduped (over the whole 10,364-WARC pool) and deconned, but the dedup step drops WARC provenance
(records carry only `{text, modernbert_prob}`). This is a problem when we want the *same global
dedup* restricted to a specific WARC subset (e.g. the seed-0 random-300) — matching how
`{dclm,high_quality}_random_300` were built (subset a post-full-pool export), NOT a fresh
per-N dedup.

Recover the subset by text-membership: the per-WARC `kept_text/data-{warc_hash}.parquet` shards
(pre-dedup, but WITH `warc_hash`+`text`) give the exact set of texts belonging to the target WARCs.
Keep a full-pool deduped doc iff its `text` appears in that set. A doc from a target WARC that was
dropped as a global near-dup is (correctly) absent from the band and stays dropped; a survivor whose
text is in the target set is kept. Output is a drop-in `baseline_{spec}_decon_deduped/{n}warcs/deduped/`
for `threshold_split.py --n {n}` + `tokenize_deduped_extracted.py --spec {spec}_decon --n {n}`.

Run as an in-region iris CPU job (zero egress)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 16 --memory 32GB --disk 20GB --priority interactive --extra cpu \\
        --enable-extra-resources --region us-east5 --job-name fastpipe-subset-300 \\
        -e WANDB_API_KEY <k> -e HF_TOKEN <t> \\
        -- python -m experiments.fast_curation.subset_to_warcs --spec fastpipe_v3 --region us-east5 \\
           --warc-manifest experiments/distill/random_subsets/lpv1_300_done.txt --n 300 --pool-n 10364
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing as mp

import fsspec
import pyarrow.parquet as pq

from experiments.fast_curation.dedup import _kept_text_shard_paths

logger = logging.getLogger(__name__)

_KEYS: frozenset[int] = frozenset()


def _key(text: str) -> int:
    """Stable 64-bit key of a document's text (sha1 truncated — collision-safe at these scales)."""
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big")


def _keys_from_kept(path: str) -> set[int]:
    with fsspec.open(path, "rb") as f:
        table = pq.read_table(f, columns=["text"])
    return {_key(t) for t in table.column("text").to_pylist() if t}


def _init_worker(keys: frozenset[int]) -> None:
    global _KEYS
    _KEYS = keys


def _filter_shard(job: tuple[str, str]) -> int:
    in_path, out_path = job
    if fsspec.filesystem("gcs").exists(out_path):
        return -1  # skip-existing
    kept = 0
    with (
        fsspec.open(in_path, "rb") as fi,
        gzip.open(fi, "rt") as gi,
        fsspec.open(out_path, "wb") as fo,
        gzip.open(fo, "wt") as go,
    ):
        for line in gi:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text")
            if text and _key(text) in _KEYS:
                go.write(line)
                kept += 1
    return kept


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v3")
    ap.add_argument("--region", default="us-east5")
    ap.add_argument("--warc-manifest", required=True, help="WARC paths (one per line); hashed sha256(line)[:12].")
    ap.add_argument("--n", type=int, required=True, help="Output subset size label (e.g. 300).")
    ap.add_argument("--pool-n", type=int, default=10364, help="Full-pool warc count (the source band's {N}warcs).")
    ap.add_argument("--num-procs", type=int, default=16)
    args = ap.parse_args()

    base = f"gs://marin-{args.region}/documents"
    in_dir = f"{base}/baseline_{args.spec}_decon_deduped/{args.pool_n}warcs/deduped"
    out_dir = f"{base}/baseline_{args.spec}_decon_deduped/{args.n}warcs/deduped"

    # 1. Build the target-WARC text keyset from the provenance-bearing kept_text shards.
    with open(args.warc_manifest) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    warc_hashes = {hashlib.sha256(ln.encode()).hexdigest()[:12] for ln in lines}
    kept_paths = _kept_text_shard_paths(args.spec, args.region, warc_hashes)
    logger.info("Building text keyset from %d kept_text shards (%d WARCs)", len(kept_paths), len(warc_hashes))
    with mp.get_context("spawn").Pool(args.num_procs) as pool:
        parts = pool.map(_keys_from_kept, kept_paths)
    keys: frozenset[int] = frozenset().union(*parts)
    logger.info("Target keyset: %d distinct doc-texts across the %d WARCs", len(keys), len(warc_hashes))

    # 2. Filter every full-pool band shard to docs whose text is in the target set.
    fs = fsspec.filesystem("gcs")
    in_shards = sorted(f"gs://{p}" for p in fs.ls(in_dir.removeprefix("gs://")) if p.endswith(".jsonl.gz"))
    if not in_shards:
        raise RuntimeError(f"no full-pool band shards at {in_dir}")
    jobs = [(s, f"{out_dir}/{s.rsplit('/', 1)[1]}") for s in in_shards]
    logger.info("Filtering %d full-pool shards -> %s", len(in_shards), out_dir)
    with mp.get_context("spawn").Pool(args.num_procs, initializer=_init_worker, initargs=(keys,)) as pool:
        results = pool.map(_filter_shard, jobs)

    kept_total = sum(c for c in results if c >= 0)
    logger.info(
        "DONE. kept %d docs (of the global 100%% band) for the %d-WARC subset -> %s", kept_total, args.n, out_dir
    )
    with fsspec.open(f"{base}/baseline_{args.spec}_decon_deduped/{args.n}warcs/subset_to_warcs_summary.json", "w") as f:
        json.dump(
            {
                "spec": args.spec,
                "n_warcs": args.n,
                "pool_n": args.pool_n,
                "target_warcs": len(warc_hashes),
                "target_texts": len(keys),
                "kept_docs": kept_total,
                "source": in_dir,
                "output": out_dir,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
