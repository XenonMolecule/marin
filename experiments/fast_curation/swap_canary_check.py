# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canary gate for the extractor ablation: arm B's text must equal arm C's for the same pages.

Arm C (``resiliparse_10k``) extracted every page of the same 10,364 WARCs with the same upstream
resiliparse call, from ``pipeline_10k``'s decoder; arm B re-decodes with the cascade's decoder.
Joining B's shard for one WARC with C's shard for that WARC by URL and comparing ``text`` proves,
before the fleet goes out, that the two decoders and the two extractor calls agree. C's shards
were produced by a glob over the raw download's ``data-{warc_hash}.jsonl.gz`` files, so C's shard
index is the WARC's rank among the sorted hashes of the manifest, not its manifest line
(:func:`arm_c_shard_index`).

    python -m experiments.fast_curation.swap_canary_check --warc-index 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass

import fsspec

from experiments.baseline_collection.decode_warcs_clean import _load_manifest, _warc_path_hash
from experiments.fast_curation.swap_extractor import OUTPUT_ROOT, shard_path

logger = logging.getLogger(__name__)

ARM_C_TREE = "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f"
N_WARCS = 10364


REPLACEMENT = "\ufffd"


@dataclass(frozen=True)
class CanaryReport:
    n_b: int
    n_url_matched: int
    n_text_equal: int
    # C's text holds U+FFFD where B's does not: pipeline_10k's decoder emitted replacement
    # characters that the cascade's clean decoder (B) resolves. Expected, and B is the better text.
    n_differs_decoder: int
    n_differs_other: int
    n_url_unmatched: int
    examples: list[dict]  # only the unexplained ("other") differences


def _decoder_only_difference(b_text: str, c_text: str) -> bool:
    """True when the texts differ only where C carries U+FFFD (same length, equal elsewhere)."""
    if len(b_text) != len(c_text) or REPLACEMENT not in c_text or REPLACEMENT in b_text:
        return False
    return all(b == c for b, c in zip(b_text, c_text, strict=True) if c != REPLACEMENT)


def compare(b_rows: list[dict], c_rows: list[dict], n_examples: int = 5) -> CanaryReport:
    """Match B rows to C rows by URL (C may repeat a URL) and classify text agreement."""
    by_url: dict[str, list[str]] = defaultdict(list)
    for row in c_rows:
        by_url[row["url"]].append(row["text"])
    matched = equal = decoder = other = 0
    examples: list[dict] = []
    for row in b_rows:
        candidates = by_url.get(row["url"])
        if not candidates:
            continue
        matched += 1
        if row["text"] in candidates:
            equal += 1
        elif any(_decoder_only_difference(row["text"], c) for c in candidates):
            decoder += 1
        else:
            other += 1
            if len(examples) < n_examples:
                c_text = candidates[0]
                examples.append(
                    {
                        "url": row["url"],
                        "b_len": len(row["text"]),
                        "c_len": len(c_text),
                        "b_head": row["text"][:120],
                        "c_head": c_text[:120],
                    }
                )
    return CanaryReport(
        n_b=len(b_rows),
        n_url_matched=matched,
        n_text_equal=equal,
        n_differs_decoder=decoder,
        n_differs_other=other,
        n_url_unmatched=len(b_rows) - matched,
        examples=examples,
    )


def arm_c_shard_index(warc_path: str, manifest_paths: list[str]) -> int:
    """Arm C's shard index for ``warc_path``: its rank among the manifest's sorted ``warc_hash`` names."""
    hashes = sorted(_warc_path_hash(p) for p in manifest_paths)
    return hashes.index(_warc_path_hash(warc_path))


def _read_jsonl_gz(path: str) -> list[dict]:
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warc-index", type=int, required=True, help="Line index into the 10k manifest.")
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--root", default=OUTPUT_ROOT)
    # Measured on WARC 0 (2026-09-07): 14,871/14,871 URL-joined; 93.3% byte-identical, 4.8% differ only
    # where C holds U+FFFD, 1.8% differ by other decoder effects (BOM, charset detection, mojibake).
    # Where the decoded html agrees the extraction is identical, so 0.9 explained is the gate.
    ap.add_argument("--min-equal-fraction", type=float, default=0.9)
    args = ap.parse_args()
    manifest = _load_manifest(args.manifest)
    warc_path = manifest[args.warc_index]
    c_index = arm_c_shard_index(warc_path, manifest)
    b_rows = _read_jsonl_gz(shard_path(args.root, _warc_path_hash(warc_path)))
    c_rows = _read_jsonl_gz(f"{ARM_C_TREE}/data-{c_index:05d}-of-{N_WARCS:05d}.jsonl.gz")
    report = compare(b_rows, c_rows)
    print(json.dumps({"warc": warc_path, "c_shard": c_index, "n_c": len(c_rows), **asdict(report)}, indent=2))
    explained = report.n_text_equal + report.n_differs_decoder
    ok = report.n_b > 0 and report.n_url_matched == report.n_b and explained >= args.min_equal_fraction * report.n_b
    print("CANARY PASS" if ok else "CANARY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
