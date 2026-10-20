# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the resiliparse Rust fork (the fastpipe cascade's extractor) over a small sample of
raw pages and write ``{warc_record_id, text}`` rows. Iris CPU job in us-east5, where the
prebuilt extractor artifact and the sample live::

    uv run iris --cluster marin job run --no-wait --region us-east5 --cpu 2 --memory 3GB \\
        --priority interactive --job-name rs-fork-dev100 \\
        -- python -m experiments.baseline_collection.extract_rs_fork_sample \\
        --sample gs://.../sample.jsonl.gz --ids gs://.../dev100_ids.txt --output gs://.../dev100_rs_fork.jsonl.gz

The call matches ``experiments/fast_curation/preprocess.py``:
``extract_plain_text(html, main_content=True, preserve_formatting="markdown")``.
"""

import argparse
import gzip
import json
import logging

import fsspec

from experiments.baseline_collection.score_resiliparse_rs import install_extractor

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", required=True, help="jsonl.gz with html + warc_record_id")
    ap.add_argument("--ids", required=True, help="text file, one warc_record_id per line")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    install_extractor()
    from resiliparse._extract_rs import extract_plain_text  # downloaded at runtime, not installed

    with fsspec.open(args.ids, "rt") as f:
        wanted = {line.strip() for line in f if line.strip()}
    rows = []
    with fsspec.open(args.sample, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["warc_record_id"] not in wanted:
                continue
            rows.append(
                {
                    "warc_record_id": r["warc_record_id"],
                    "text": extract_plain_text(r["html"], main_content=True, preserve_formatting="markdown"),
                }
            )
    missing = wanted - {r["warc_record_id"] for r in rows}
    if missing:
        raise RuntimeError(f"{len(missing)} ids not found in sample: {sorted(missing)[:3]}")
    with fsspec.open(args.output, "wt", compression="gzip") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_empty = sum(1 for r in rows if not r["text"].strip())
    logger.info("wrote %d rows to %s (%d empty)", len(rows), args.output, n_empty)


if __name__ == "__main__":
    main()
