# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Produce the resiliparse 300-WARC document subset for BM25 (resiliparse-300).

Resiliparse extraction docs carry only {text,url}, but the WARC metadata table is
shard-aligned 1:1 with the extraction (verified: resiliparse shard N ≡ metadata
shard N ≡ one WARC). So we read ONE record per metadata shard to map shard->WARC,
select the ~300 shards whose WARC is in the random-300 manifest, and copy just
those resiliparse shards into ``filtered_subsets/resiliparse_random_300warcs-
bm25v1/`` -- the path the BM25 resiliparse-small source already globs. No scan of
the ~400M-doc corpus. All reads/writes in-region (us-central2).
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import fsspec
from marin.utils import fsspec_glob
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

RESI = "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f/*.jsonl.gz"
META = "gs://marin-us-central2/metadata/dclm_400m_1x_10k_warc_metadata-79158f/*.jsonl.gz"
POOL_300 = "experiments/distill/random_subsets/random_warcs_300.txt"
OUT_DIR = "gs://marin-us-central2/filtered_subsets/resiliparse_random_300warcs-bm25v1"
_WORKERS = 32


def _basename(p: str) -> str:
    return p.rsplit("/", 1)[-1] if p else ""


def _load_300() -> set[str]:
    with open(POOL_300) as f:
        return {_basename(ln.strip()) for ln in f if ln.strip() and not ln.startswith("#")}


def _shard_warc(meta_url: str) -> str:
    """The (single) WARC basename a metadata shard covers -- from its first record."""
    for rec in load_file(meta_url):
        return _basename(str(rec.get("warc_file")))
    return ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    want = _load_300()
    resi = sorted(fsspec_glob(RESI))
    meta = sorted(fsspec_glob(META))
    if len(resi) != len(meta):
        raise RuntimeError(f"shard count mismatch: resi={len(resi)} meta={len(meta)}")
    logger.info("Mapping %d metadata shards -> WARC (want %d of them)...", len(meta), len(want))

    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        warcs = list(ex.map(_shard_warc, meta))
    selected = [resi[i] for i, w in enumerate(warcs) if w in want]
    matched = {w for w in warcs if w in want}
    logger.info("Selected %d resiliparse shards covering %d/%d wanted WARCs", len(selected), len(matched), len(want))
    if not selected:
        raise RuntimeError("selected 0 shards -- WARC basename format mismatch?")

    fs = fsspec.filesystem("gcs")

    def _copy(pair: tuple[int, str]) -> None:
        k, src = pair
        fs.copy(src, f"{OUT_DIR}/data-{k:05d}-of-{len(selected):05d}.jsonl.gz")

    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        list(ex.map(_copy, list(enumerate(selected))))
    logger.info("Copied %d shards -> %s", len(selected), OUT_DIR)
    # Missing WARCs (in manifest but not matched) -- expected 0 since 300 ⊂ 10k pool.
    missing = sorted(want - matched)
    raise RuntimeError(
        f"RESI300_DONE selected={len(selected)} matched_warcs={len(matched)}/{len(want)} "
        f"missing={missing[:5]} out={OUT_DIR}"
    )


if __name__ == "__main__":
    main()
