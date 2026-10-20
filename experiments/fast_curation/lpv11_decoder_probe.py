# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure how much the decoder choice changes the HTML the fork extractor sees.

lpv11's own pipeline decoded WARC payloads as ``content.decode("utf-8", errors="replace")``
(``download_warcs._download_one_warc``), while this swap uses the clean WHATWG decoder
(``decode_warcs_clean.decode_payload``). Both select records with identical filters, so no record
is lost either way -- but on pages that are not valid UTF-8 the old path emitted replacement
characters where the clean one decodes correctly.

For one WARC this reports, over that WARC's allow-listed records: how many decode identically,
how many differ, how many of the differing ones carry ``U+FFFD`` only under the naive decoder
(the expected class), and how many differ for some other reason. Run it before trusting a
NEW-vs-lpv11 comparison, and put the number in the runbook.

    python -m experiments.fast_curation.lpv11_decoder_probe --warc-index 0
"""

from __future__ import annotations

import argparse
import io
import json
import logging

import fsspec
import pyarrow.parquet as pq

from experiments.baseline_collection.decode_warcs_clean import (
    REPLACEMENT,
    _decode_one_warc_stream,
    _load_manifest,
    _normalize_record_id,
    _warc_path_hash,
)
from experiments.baseline_collection.download_warcs import _download_one_warc
from experiments.fast_curation.lpv11_fork_extractor import OUTPUT_ROOT

logger = logging.getLogger(__name__)


def naive_records(warc_path: str) -> dict[str, str]:
    """``{record_id: html}`` exactly as lpv11's own pipeline decoded it."""
    out: dict[str, str] = {}
    for rec in _download_one_warc(warc_path):
        out[_normalize_record_id(rec["id"])] = rec["html"]
    return out


def clean_records(warc_path: str) -> dict[str, str]:
    """``{record_id: html}`` as this swap decodes it."""
    out: dict[str, str] = {}
    for chunk in _decode_one_warc_stream(warc_path, chunk_records=2000):
        for rec in chunk:
            out[rec["doc_id"]] = rec["html"]
    return out


def compare(wanted: set[str], naive: dict[str, str], clean: dict[str, str]) -> dict:
    both = wanted & naive.keys() & clean.keys()
    identical = sum(1 for i in both if naive[i] == clean[i])
    differ = [i for i in both if naive[i] != clean[i]]
    fffd_only = sum(1 for i in differ if REPLACEMENT in naive[i] and REPLACEMENT not in clean[i])
    return {
        "allowlisted": len(wanted),
        "in_naive": len(wanted & naive.keys()),
        "in_clean": len(wanted & clean.keys()),
        "compared": len(both),
        "identical_html": identical,
        "differing_html": len(differ),
        "differ_fffd_only": fffd_only,
        "differ_other": len(differ) - fffd_only,
        "missing_from_clean": sorted(wanted - clean.keys())[:5],
        "missing_from_naive": sorted(wanted - naive.keys())[:5],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warc-index", type=int, default=0)
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--root", default=OUTPUT_ROOT)
    args = ap.parse_args()
    warc_path = _load_manifest(args.manifest)[args.warc_index]
    h = _warc_path_hash(warc_path)
    with fsspec.open(f"{args.root}/allowlist/data-{h}.parquet", "rb") as fh:
        wanted = set(
            pq.read_table(io.BytesIO(fh.read()), columns=["warc_record_id"]).column("warc_record_id").to_pylist()
        )
    logger.info("probing %s (%d allow-listed records)", warc_path, len(wanted))
    report = compare(wanted, naive_records(warc_path), clean_records(warc_path))
    print(json.dumps({"warc": warc_path, "warc_hash": h, **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
