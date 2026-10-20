# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DCLM-baseline curation on single WARCs, to calibrate per-WARC token projections.

Runs the DCLM-baseline recipe on one Common Crawl WARC: upstream resiliparse main-content extraction
(``extractors.resiliparse_main``, the DCLM config), then the full DCLM-baseline filter port
(``marin.transform.dclm_filter``: URL ban lists, RefinedWeb and Gopher heuristics, fastText language id,
fastText OH-ELI5 quality at DCLM's 0.018112 threshold). Dedup is not applied; the per-WARC token count is
pre-dedup and is compared against the released DCLM-baseline restricted to the same WARCs.

Writes ``<output>/warc-<hash>/stats.json`` (counts and marin-tokenizer tokens at each stage) and
``kept.jsonl.gz`` (the surviving pages).

    iris job run --cpu 8 --memory 14GB --enable-extra-resources --extra cpu --extra dclm \\
        -- python -m experiments.baseline_collection.dclm_baseline_warc_tokens \\
        --warc s3://commoncrawl/crawl-data/.../x.warc.gz --output gs://bucket/users/me/dclm_baseline_warcs
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import tempfile
import time
from multiprocessing import Pool

import fsspec
from marin.transform.dclm_filter import DclmFilterConfig, _init_pipeline
from transformers import AutoTokenizer

from experiments.baseline_collection.decode_warcs_clean import _decode_one_warc_stream
from experiments.baseline_collection.extractors import resiliparse_main

logger = logging.getLogger(__name__)

LID_DIR = "gs://marin-us-central1/resources/dclm/lid_176-951189"
QUALITY_DIR = "gs://marin-us-central1/resources/dclm/fasttext_oh_eli5-d828e9"
BANLISTS_DIR = "gs://marin-us-central2/resources/dclm/banlists"
TOKENIZER = "marin-community/marin-tokenizer"

# Loaded once in the parent before the pool forks: the two fastText models are ~1.2 GB and the
# workers share them copy-on-write (per-worker copies OOM-killed the 14 GB CPU pool).
_pipeline = None
_tokenizer = None


def _load_models() -> None:
    global _pipeline, _tokenizer
    config = DclmFilterConfig(
        input_path="", output_path="", lid_model_path=LID_DIR, quality_model_path=QUALITY_DIR, banlists_path=BANLISTS_DIR
    )
    _pipeline = _init_pipeline(config, tempfile.mkdtemp(prefix="dclm_"))
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def process(record: dict) -> dict:
    """Extract, filter and count one page. Returns the stage reached and token counts."""
    text = resiliparse_main(record["html"], record["url"])
    if not text.strip():
        return {"stage": "empty_extraction"}
    extracted_tokens = _count_tokens(text)
    kept = _pipeline({"text": text, "url": record["url"]})
    if not kept:
        return {"stage": "filtered", "extracted_tokens": extracted_tokens}
    page = kept[0]
    return {
        "stage": "kept",
        "extracted_tokens": extracted_tokens,
        "kept_tokens": _count_tokens(page["text"]),
        "text": page["text"],
        "quality": page.get("fasttext_oh_eli5_vs_rw_v2_prob"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc", required=True)
    parser.add_argument("--output", required=True, help="gs:// directory")
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()

    warc_id = hashlib.sha256(args.warc.encode()).hexdigest()[:12]
    out_prefix = f"{args.output.replace('gs://', '').rstrip('/')}/warc-{warc_id}"
    fs = fsspec.filesystem("gcs")

    _load_models()
    # The WARC is decoded in chunks: a whole WARC's HTML (3-5 GB for 45k pages) plus the models
    # exceeds the 14 GB CPU pool.
    counts = {
        "html_records": 0,
        "empty_extraction": 0,
        "filtered": 0,
        "kept": 0,
        "extracted_tokens": 0,
        "kept_tokens": 0,
    }
    t0 = time.monotonic()
    with (
        Pool(args.procs) as pool,
        fs.open(f"{out_prefix}/kept.jsonl.gz", "wb") as f,
        gzip.open(f, "wt") as z,
    ):
        for records in _decode_one_warc_stream(args.warc):
            results = pool.map(process, records, chunksize=16)
            for rec, r in zip(records, results, strict=True):
                counts["html_records"] += 1
                counts[r["stage"]] += 1
                counts["extracted_tokens"] += r.get("extracted_tokens", 0)
                counts["kept_tokens"] += r.get("kept_tokens", 0)
                if r["stage"] == "kept":
                    row = {"doc_id": rec["doc_id"], "url": rec["url"], "text": r["text"], "quality": r["quality"]}
                    z.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("%d records so far: %s", counts["html_records"], counts)
    stats = {"warc": args.warc, "warc_hash": warc_id, **counts, "seconds": round(time.monotonic() - t0, 1)}
    with fs.open(f"{out_prefix}/stats.json", "w") as f:
        json.dump(stats, f)
    logger.info("%s", json.dumps(stats))


if __name__ == "__main__":
    main()
